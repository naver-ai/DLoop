# coding=utf-8
# Derived from SpecForge (https://github.com/sgl-project/SpecForge)
# MIT License
# Modified for DLoop

"""DFlash Training Wrapper."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

# NPU workaround: flex_attention is not available on Ascend NPU.
if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False


_VALID_LOSS_TYPES = {
    "dflash",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DPACE_LOSS_TYPES = _VALID_LOSS_TYPES - {"dflash"}


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block acceptance length."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def create_dflash_sdpa_mask(
    anchor_positions, block_keep_mask, S, block_size, device, n_carrier=0
):
    """KV layout: [context S | carriers N * n_carrier | draft N * block_size].

    Carriers (DLoop) are KV-only rows visible to their own block's queries only;
    ``n_carrier=0`` keeps the [context | draft] layout."""
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    C_LEN = N * n_carrier
    KV_LEN = S + C_LEN + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    if n_carrier > 0:
        is_carrier = (kv_indices >= S) & (kv_indices < S + C_LEN)
        mask_carrier = is_carrier & (q_block_ids == (kv_indices - S) // n_carrier)
    else:
        mask_carrier = torch.zeros_like(mask_context)

    is_draft = kv_indices >= S + C_LEN
    kv_block_ids = (kv_indices - S - C_LEN) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_carrier | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    n_carrier: int = 0,
):
    """Construct Flex Attention BlockMask for DFlash training.

    KV: [Context (S tokens) | Carriers (N * n_carrier) | Block_0 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
      5. Carriers (DLoop, n_carrier > 0) are visible to their own block only.
    """
    B, N = anchor_positions.shape
    # A single captured scalar (not S + C_LEN inside mask_mod) keeps inductor
    # lowering happy once the sequence length goes symbolic.
    DRAFT_START = S + N * n_carrier

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        if n_carrier > 0:
            is_carrier = (kv_idx >= S) & (kv_idx < DRAFT_START)
            mask_carrier = is_carrier & (q_block_id == (kv_idx - S) // n_carrier)
        else:
            mask_carrier = torch.zeros_like(mask_context)

        is_draft = kv_idx >= DRAFT_START
        kv_block_id = (kv_idx - DRAFT_START) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_carrier | mask_draft) & is_valid_block & in_bounds

    Q_LEN = N * block_size
    KV_LEN = DRAFT_START + N * block_size

    return create_block_mask(
        dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with DFlash and D-PACE losses."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
        loop_max_stage: int = 0,
    ):
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.loss_type = loss_type
        self.dpace_alpha = dpace_alpha
        # DLoop: every step also trains loop stage k ~ U{1..loop_max_stage}
        # (0 = the host objective alone).
        if loop_max_stage < 0:
            raise ValueError(f"loop_max_stage must be >= 0, got {loop_max_stage}")
        if loop_max_stage and loss_type != "dflash":
            raise ValueError("DLoop training uses --loss-type dflash")
        self.loop_max_stage = loop_max_stage

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        # DLoop: leave room for the main block plus one loop stage.
        max_anchor = max(seq_len - bs * (2 if self.loop_max_stage else 1), 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _create_position_ids(
        self, anchor_positions: torch.Tensor, offset: int = 0, rows: Optional[int] = None
    ) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks (rows at
        a+offset .. a+offset+rows-1)."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        rows = self.block_size if rows is None else rows
        offsets = torch.arange(offset, offset + rows, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(
        self, input_ids, anchor_positions, block_keep_mask, offset=0, rows=None
    ):
        """Block rows [t_{a+offset}, MASK x (rows-1)]."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size if rows is None else rows
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = (anchor_positions + offset).clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Compute detached D-PACE position weights.

        ``prob`` is the draft probability on the target token at each draft
        position. Invalid positions are treated as multiplicative no-ops inside
        prefix products and excluded from suffix sums; the caller still
        multiplies the returned weights by ``binary_mask`` before reduction.
        """
        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        if loss_type == "dpace-cumulative-confidence-only":
            return prefix

        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
            dims=[-1],
        )

        if loss_type == "dpace":
            return suffix
        if loss_type == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        sampled: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0,
        rows: Optional[int] = None,
        carrier: Optional[torch.Tensor] = None,
        position_offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One draft forward over all blocks: rows [t_{a+offset}, MASK x
        (rows-1)] at positions a+offset.., with ``carrier`` [bsz, N, n, d]
        (DLoop: hidden states of offsets 1..n) in K/V at their true positions.
        ``sampled`` reuses one anchor draw across forwards; ``position_offsets``
        shifts every RoPE position per sample (PoSE-style augmentation)."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        rows = self.block_size if rows is None else rows

        if sampled is None:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
        else:
            anchor_positions, block_keep_mask = sampled

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask, offset, rows
        )

        n_carrier = 0 if carrier is None else carrier.shape[2]
        position_id_parts = [
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        ]
        if n_carrier:
            carrier_offsets = torch.arange(1, n_carrier + 1, device=device)
            position_id_parts.append(
                (anchor_positions.unsqueeze(-1) + carrier_offsets).view(bsz, -1)
            )
        position_id_parts.append(
            self._create_position_ids(anchor_positions, offset, rows)
        )
        full_position_ids = torch.cat(position_id_parts, dim=1)
        if position_offsets is not None:
            full_position_ids = full_position_ids + position_offsets.view(-1, 1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=rows,
                device=device,
                n_carrier=n_carrier,
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=rows,
                device=device,
                n_carrier=n_carrier,
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
            extra_kv_hidden=(
                None if carrier is None else carrier.reshape(bsz, -1, carrier.shape[-1])
            ),
        )
        return anchor_positions, block_keep_mask, output_hidden

    def _sample_loop_depth(self) -> int:
        """DLoop stage depth k ~ U{1..loop_max_stage} (the supervised stage)."""
        return int(torch.randint(1, self.loop_max_stage + 1, ()))

    def _forward_loop_stages(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        sampled: Tuple[torch.Tensor, torch.Tensor],
        produced: torch.Tensor,
        k: int,
        lead: int,
        position_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DLoop stages 1..k on the same anchors, in the deployed layout.

        Every stage has the main block's layout: B rows [t_{a+o}, MASK x (B - 1)]
        and B - lead predictions, the same input and output length as stage 0.
        ``produced`` [bsz, N, n, d] holds the hidden states of the rows that
        drafted slots 1..n so far. Stage s anchors on the ground-truth token at
        slot o = s*(B - lead) (the last token of stage s-1) and carries slots
        1..o-1 into K/V, with gradients. ``lead`` is 1 for same-position labels
        (DFlash: row 0 predicts nothing) and 0 for next-token labels (DSpark /
        Domino). Only stage k is supervised; its B - lead prediction rows
        [bsz, N, B - lead, d] are returned."""
        bs = self.block_size
        bsz, n = sampled[0].shape
        for stage in range(1, k + 1):
            offset = stage * (bs - lead)
            _, _, out = self._forward_draft_blocks(
                input_ids,
                hidden_states,
                loss_mask,
                sampled=sampled,
                offset=offset,
                rows=bs,
                carrier=produced[:, :, : offset - 1],
                position_offsets=position_offsets,
            )
            out = out.view(bsz, n, bs, -1)[:, :, lead:]
            if stage < k:
                produced = torch.cat([produced, out], dim=2)
        return out

    def _block_loss(self, pred_hidden, input_ids, loss_mask, anchors, keep, label_offset):
        """Decayed CE of prediction rows [bsz, N, R, d] against t_{a+label_offset+j}.
        Returns (weighted CE sum, weight sum, #correct, #labels)."""
        bsz, n_blocks, n_rows, _ = pred_hidden.shape
        seq_len = input_ids.shape[1]
        device = input_ids.device

        logits = self.lm_head(pred_hidden.reshape(bsz, n_blocks * n_rows, -1))
        label_idx = anchors.unsqueeze(-1) + torch.arange(
            label_offset, label_offset + n_rows, device=device
        )
        safe_idx = label_idx.clamp(max=seq_len - 1)
        targets = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_idx
        )
        weight = keep.unsqueeze(-1).float() * (label_idx < seq_len).float()
        weight = weight * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_idx
        )

        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        loss_weight = weight
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            depth = torch.arange(n_rows, device=device).view(1, 1, -1)
            loss_weight = loss_weight * torch.exp(-depth.float() / self.loss_decay_gamma)
        loss_weight = loss_weight.view(-1)

        with torch.no_grad():
            mask = weight.view(-1) > 0.5
            correct = ((flat_logits.argmax(dim=-1) == flat_targets) & mask).sum().float()
        return (ce * loss_weight).sum(), loss_weight.sum(), correct, weight.sum()

    def _forward_dloop(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        position_offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """DFlash main block + DLoop stage k, decayed CE normalized jointly."""
        bsz, seq_len = input_ids.shape
        bs = self.block_size
        sampled = self._sample_anchor_positions(seq_len, loss_mask, input_ids.device)
        anchors, keep = sampled

        _, _, out = self._forward_draft_blocks(
            input_ids, hidden_states, loss_mask, sampled=sampled,
            position_offsets=position_offsets,
        )
        main = out.view(bsz, anchors.shape[1], bs, -1)[:, :, 1:]  # slots 1..B-1
        loss_main = self._block_loss(main, input_ids, loss_mask, anchors, keep, 1)

        k = self._sample_loop_depth()
        stage = self._forward_loop_stages(
            input_ids, hidden_states, loss_mask, sampled, main, k, lead=1,
            position_offsets=position_offsets,
        )
        loss_stage = self._block_loss(stage, input_ids, loss_mask, anchors, keep, k * (bs - 1) + 1)

        ce_sum, w_sum, correct, n_labels = (a + b for a, b in zip(loss_main, loss_stage))
        loss = ce_sum / (w_sum + 1e-6)
        accuracy = correct / (n_labels + 1e-6)
        metrics = {
            "accuracy_denom": n_labels,
            "main_accuracy": loss_main[2] / (loss_main[3] + 1e-6),
            "loop_accuracy": loss_stage[2] / (loss_stage[3] + 1e-6),
        }
        return loss, accuracy, metrics

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        position_offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel block-wise training forward pass; returns
        (loss, accuracy, metrics) — same shape as Domino's forward."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        if self.loop_max_stage:
            return self._forward_dloop(
                input_ids, hidden_states, loss_mask, position_offsets
            )
        if position_offsets is not None:
            raise ValueError("position_offsets is only wired into DLoop training")
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        logits = self.lm_head(output_hidden)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1)

        # --- Cross entropy ---
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        if self.loss_type == "dflash":
            # Preserve the existing DFlash weighted-mean behavior.
            loss_weights = weight_mask
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                k = torch.arange(self.block_size, device=device).view(1, 1, -1)
                decay_weights = torch.exp(
                    -(k - 1).clamp(min=0).float() / self.loss_decay_gamma
                )
                loss_weights = loss_weights * decay_weights

            flat_weights = loss_weights.view(-1)
            valid_token_count = flat_weights.sum() + 1e-6
            loss = (loss_per_token * flat_weights).sum() / valid_token_count
        elif self.loss_type in _DPACE_LOSS_TYPES:
            neg_log_q = loss_per_token.view_as(target_ids)
            with torch.no_grad():
                q = torch.exp(-neg_log_q)
                dpace_weights = self._dpace_weight(
                    q,
                    weight_mask,
                    weight_mask > 0,
                    self.loss_type,
                )
            loss_weights = weight_mask * dpace_weights
            loss = (neg_log_q * loss_weights).sum() / float(bsz)
        else:
            raise ValueError(f"unknown loss_type {self.loss_type!r}")

        # --- Accuracy ---
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy_denom = binary_eval_mask.sum()
            accuracy = correct.sum().float() / (accuracy_denom + 1e-6)

        return loss, accuracy, {"accuracy_denom": accuracy_denom}


class OnlineDominoModel(OnlineDFlashModel):
    """Domino online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        shift_label: bool = False,
        loop_max_stage: int = 0,
        loop_loss_weight: float = 0.25,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        self.shift_label = shift_label
        # DLoop: loss = (1 - w) * Domino loss + w * loop-stage loss.
        if loop_max_stage and not shift_label:
            raise ValueError("Domino DLoop training requires shift_label=True")
        if not 0.0 <= loop_loss_weight <= 1.0:
            raise ValueError("loop_loss_weight must be in [0, 1]")
        self.loop_max_stage = loop_max_stage
        self.loop_loss_weight = float(loop_loss_weight)

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = max(1, min(self.num_anchors, int(valid_counts.max().item()) - 1))

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _build_domino_head_inputs(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_ids: torch.Tensor,
        output_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n, bs = target_ids.shape
        hidden4d = output_hidden.reshape(bsz, n, bs, output_hidden.shape[-1])

        prev_ids = target_ids
        if self.shift_label:
            prev_offsets = torch.arange(
                0, self.block_size, device=input_ids.device
            ).view(1, 1, -1)
            prev_indices = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(
                max=input_ids.size(1) - 1
            )
            prev_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                prev_indices,
            )

        return hidden4d, prev_ids

    def _apply_domino_head(
        self,
        base_logits4d: torch.Tensor,
        hidden4d: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        head_token_ids = prev_ids if self.shift_label else target_ids
        head_token_embeddings = self.embed_tokens(head_token_ids)
        return self.draft_model.apply_logits_head(
            base_logits4d,
            hidden_states=hidden4d,
            prev_token_embeddings=head_token_embeddings,
        )

    def _compute_extra_metrics(
        self,
        pred_ids: torch.Tensor,
        flat_base_logits: torch.Tensor,
        flat_targets: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        actual_token_count: torch.Tensor,
        target_ids: torch.Tensor,
        eval_weight_mask: torch.Tensor,
        final_loss: torch.Tensor,
        base_loss: torch.Tensor,
        lambda_base: float,
    ) -> Dict[str, torch.Tensor]:
        bsz, n, bs = target_ids.shape

        base_pred_ids = torch.argmax(flat_base_logits, dim=-1)
        base_correct = (base_pred_ids == flat_targets) & (binary_eval_mask > 0.5)
        base_accuracy = base_correct.sum().float() / actual_token_count

        valid_mask_4d = (eval_weight_mask > 0).bool()
        pred_accept_len = compute_accept_len(
            pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
        )
        base_accept_len = compute_accept_len(
            base_pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
        )

        valid_block_mask = valid_mask_4d.any(dim=2)
        num_valid_blocks = valid_block_mask.sum().float() + 1e-6
        avg_accept_len = (
            (pred_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks
        base_avg_accept_len = (
            (base_accept_len + 1.0) * valid_block_mask.float()
        ).sum() / num_valid_blocks

        return {
            "final_loss": final_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_accuracy": base_accuracy.detach(),
            "accept_len": avg_accept_len.detach(),
            "base_accept_len": base_avg_accept_len.detach(),
            "lambda_base": torch.tensor(lambda_base, device=final_loss.device),
        }

    def _compute_weighted_losses(
        self,
        final_logits: torch.Tensor,
        base_logits: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        lambda_base: float,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flat_logits = final_logits.reshape(-1, final_logits.size(-1))
        flat_base_logits = base_logits.reshape(-1, base_logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)

        valid_token_count = flat_weights.sum() + 1e-6

        final_loss_per_token = F.cross_entropy(
            flat_logits, flat_targets, reduction="none"
        )
        final_loss = (final_loss_per_token * flat_weights).sum() / valid_token_count

        base_loss_per_token = F.cross_entropy(
            flat_base_logits, flat_targets, reduction="none"
        )
        base_loss = (base_loss_per_token * flat_weights).sum() / valid_token_count

        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss

        return loss, final_loss, base_loss, flat_logits, flat_base_logits, flat_targets

    def _domino_loop_loss(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        sampled: Tuple[torch.Tensor, torch.Tensor],
        main_hidden: torch.Tensor,
        lambda_base: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Domino loss of DLoop stage k (labels t_{a+kB+1..a+kB+B}).

        The backbone runs the deployed loop layout; the GRU head consumes the
        whole teacher-forced prefix t_a..t_{a+(k+1)B-1}, as its state is carried
        across stages at inference."""
        B = self.block_size
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = sampled
        n = anchor_positions.size(1)

        k = self._sample_loop_depth()
        h_last = self._forward_loop_stages(
            input_ids, hidden_states, loss_mask, sampled, main_hidden, k, lead=0
        )

        rows = (k + 1) * B
        prev_indices = (
            anchor_positions.unsqueeze(-1) + torch.arange(rows, device=device)
        ).clamp(max=seq_len - 1)
        prev_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1), 2, prev_indices
        )
        gru_out = self.draft_model._run_gru(
            self.embed_tokens(prev_ids).reshape(bsz * n, rows, -1)
        ).reshape(bsz, n, rows, -1)
        s_last = gru_out[:, :, k * B : k * B + B, :]

        base_last = self.lm_head(h_last.reshape(bsz, n * B, -1)).view(bsz, n, B, -1)
        bias = self.draft_model.embed_proj(
            torch.cat([h_last, s_last.to(h_last.dtype)], dim=-1)
        )
        final_last = base_last + bias

        label_indices = anchor_positions.unsqueeze(-1) + torch.arange(
            1, k * B + B + 1, device=device
        ).view(1, 1, -1)
        safe_indices = label_indices.clamp(max=seq_len - 1)
        safe_indices = torch.where(
            block_keep_mask.unsqueeze(-1), safe_indices, torch.zeros_like(safe_indices)
        )
        target_lm = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, n, -1), 2, safe_indices
        )
        # Only blocks whose whole ground-truth chain is valid are supervised.
        eval_full = (
            (label_indices < seq_len)
            & (target_lm > 0.5)
            & block_keep_mask.unsqueeze(-1)
        )
        eval_full = eval_full.to(torch.int32).cumprod(dim=-1).bool()
        loop_eval = eval_full[:, :, -B:]
        loop_target = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n, -1), 2, safe_indices[:, :, -B:]
        )

        weight = loop_eval.float()
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            j = torch.arange(B, device=device).view(1, 1, -1)
            weight = weight * torch.exp(-j.float() / self.loss_decay_gamma)

        loss, _, _, flat_logits, _, flat_targets = self._compute_weighted_losses(
            final_logits=final_last,
            base_logits=base_last,
            target_ids=loop_target,
            weight_mask=weight,
            lambda_base=lambda_base,
        )
        with torch.no_grad():
            correct = (torch.argmax(flat_logits, dim=-1) == flat_targets) & loop_eval.reshape(-1)
            accuracy = correct.sum().float() / (loop_eval.sum().float() + 1e-6)
        return loss, {"loop_accuracy": accuracy}

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float = 0.0,
    ):
        """Parallel Domino training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        label_start = 1 if self.shift_label else 0
        label_offsets = torch.arange(
            label_start, label_start + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_target_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )

        bsz, n, bs = target_ids.shape
        base_logits = self.lm_head(output_hidden)
        hidden4d, prev_ids = self._build_domino_head_inputs(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            target_ids=target_ids,
            output_hidden=output_hidden,
        )
        base_logits4d = base_logits.reshape(bsz, n, bs, -1)
        final_logits = self._apply_domino_head(
            base_logits4d=base_logits4d,
            hidden4d=hidden4d,
            prev_ids=prev_ids,
            target_ids=target_ids,
        ).reshape(bsz, n * bs, -1)

        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        if not self.shift_label:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_target_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        eval_weight_mask = weight_mask.clone()
        binary_eval_mask = weight_mask.view(-1)

        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            offset = 0 if self.shift_label else 1
            decay_weights = torch.exp(
                -(k - offset).clamp(min=0).float() / self.loss_decay_gamma
            )
            weight_mask = weight_mask * decay_weights

        loss, final_loss, base_loss, flat_logits, flat_base_logits, flat_targets = (
            self._compute_weighted_losses(
                final_logits=final_logits,
                base_logits=base_logits,
                target_ids=target_ids,
                weight_mask=weight_mask,
                lambda_base=lambda_base,
            )
        )

        loop_metrics: Dict[str, torch.Tensor] = {}
        if self.loop_max_stage:
            loop_loss, loop_metrics = self._domino_loop_loss(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                sampled=(anchor_positions, block_keep_mask),
                main_hidden=hidden4d,
                lambda_base=lambda_base,
            )
            w = self.loop_loss_weight
            loss = (1.0 - w) * loss + w * loop_loss

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            accuracy_denom = binary_eval_mask.sum()
            actual_token_count = accuracy_denom + 1e-6
            accuracy = correct.sum().float() / actual_token_count

            metrics = self._compute_extra_metrics(
                pred_ids=pred_ids,
                flat_base_logits=flat_base_logits,
                flat_targets=flat_targets,
                binary_eval_mask=binary_eval_mask,
                actual_token_count=actual_token_count,
                target_ids=target_ids,
                eval_weight_mask=eval_weight_mask,
                final_loss=final_loss,
                base_loss=base_loss,
                lambda_base=lambda_base,
            )
            metrics["accuracy_denom"] = accuracy_denom
            metrics.update(loop_metrics)

        return loss, accuracy, metrics


class OnlineDSparkModel(OnlineDFlashModel):
    """DSpark online training wrapper over DFlash block-parallel components."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        dspark_ce_loss_alpha: float = 0.1,
        dspark_l1_loss_alpha: float = 0.9,
        dspark_confidence_head_alpha: float = 1.0,
        loop_max_stage: int = 0,
        loop_loss_weight: float = 0.25,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            loss_type="dflash",
        )
        if dspark_ce_loss_alpha < 0:
            raise ValueError("dspark_ce_loss_alpha must be >= 0")
        if dspark_l1_loss_alpha < 0:
            raise ValueError("dspark_l1_loss_alpha must be >= 0")
        if dspark_confidence_head_alpha < 0:
            raise ValueError("dspark_confidence_head_alpha must be >= 0")

        self.loss_type = "dspark"
        self.dspark_ce_loss_alpha = float(dspark_ce_loss_alpha)
        self.dspark_l1_loss_alpha = float(dspark_l1_loss_alpha)
        self.dspark_confidence_head_alpha = float(dspark_confidence_head_alpha)
        # DLoop: loss = (1 - w) * DSpark loss + w * loop-stage loss.
        if not 0.0 <= loop_loss_weight <= 1.0:
            raise ValueError("loop_loss_weight must be in [0, 1]")
        self.loop_max_stage = loop_max_stage
        self.loop_loss_weight = float(loop_loss_weight)

    def _build_anchor_candidate_mask(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_candidates = max(seq_len - 1, 0)
        if num_candidates == 0:
            return loss_mask[:, :0].bool()
        anchor_valid = loss_mask[:, :num_candidates] > 0.5
        first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
        return anchor_valid & first_target_valid

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample fixed-width DSpark anchors; invalid slots are masked out."""
        valid = self._build_anchor_candidate_mask(seq_len, loss_mask)
        bsz = loss_mask.shape[0]
        num_candidates = valid.shape[1]
        max_n = int(self.num_anchors)
        if num_candidates == 0:
            anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
            keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
            return anchors, keep_mask

        valid_counts = valid.sum(dim=1)
        indices = (
            torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid,
            indices,
            torch.full_like(indices, seq_len + 1),
        )
        random_vals = torch.rand(bsz, num_candidates, device=device)
        random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        if num_candidates < max_n:
            pad = torch.full(
                (bsz, max_n - num_candidates),
                seq_len + 1,
                dtype=gathered.dtype,
                device=device,
            )
            gathered = torch.cat([gathered, pad], dim=1)
        anchors = gathered[:, :max_n].sort(dim=1).values
        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
            valid_counts.unsqueeze(1).clamp(max=max_n)
        )
        anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
        return anchors, keep_mask

    def _build_dspark_labels_and_mask(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = input_ids.shape[1]
        device = input_ids.device
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        target_valid = label_indices < seq_len
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
            2,
            safe_label_indices,
        )
        eval_mask = target_valid & (target_loss_mask > 0.5)
        eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
        eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
        return target_ids, eval_mask, safe_label_indices

    def _dspark_loss_weight_mask(
        self,
        eval_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_weight_mask = eval_mask.to(torch.float32)
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = torch.arange(self.block_size, device=eval_mask.device).view(
                1, 1, -1
            )
            decay_weights = torch.exp(-positions.float() / float(self.loss_decay_gamma))
            loss_weight_mask = loss_weight_mask * decay_weights
        return loss_weight_mask

    def _aligned_target_logits(
        self,
        target_last_hidden_states: Optional[torch.Tensor],
        safe_label_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if target_last_hidden_states is None:
            return None
        target_pred_indices = (safe_label_indices - 1).clamp(min=0)
        aligned_target_hidden = torch.gather(
            target_last_hidden_states.unsqueeze(1).expand(
                -1,
                safe_label_indices.size(1),
                -1,
                -1,
            ),
            2,
            target_pred_indices.unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                target_last_hidden_states.size(-1),
            ),
        )
        return self.lm_head(aligned_target_hidden)

    def _compute_dspark_loss(
        self,
        *,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
        confidence_pred: Optional[torch.Tensor],
        aligned_target_logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        vocab_size = draft_logits.size(-1)
        loss_weight_mask = self._dspark_loss_weight_mask(eval_mask)
        flat_logits = draft_logits.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)
        flat_weights = loss_weight_mask.reshape(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        ce_loss_den = flat_weights.sum()
        ce_loss = (loss_per_token * flat_weights).sum() / (ce_loss_den + 1e-6)

        l1_loss = ce_loss.new_zeros(())
        accept_rate_3d = None
        needs_target_distribution = (
            self.dspark_l1_loss_alpha > 0 or confidence_pred is not None
        )
        if aligned_target_logits is not None and needs_target_distribution:
            draft_probs = torch.softmax(draft_logits.float(), dim=-1)
            target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
            l1_dist = (draft_probs - target_probs).abs().sum(dim=-1)
            accept_rate_3d = 1.0 - 0.5 * l1_dist
            accept_rate_3d = accept_rate_3d.clamp_(0.0, 1.0)
            if self.dspark_l1_loss_alpha > 0:
                l1_loss = (l1_dist * loss_weight_mask).sum() / (ce_loss_den + 1e-6)
        elif self.dspark_l1_loss_alpha > 0 or self.dspark_confidence_head_alpha > 0:
            raise ValueError(
                "DSpark L1/confidence loss requires target_last_hidden_states. "
                "Use the disaggregated DSpark server-capture path so the "
                "consumer receives target_last_hidden_states."
            )

        confidence_loss = ce_loss.new_zeros(())
        confidence_abs_error = ce_loss.new_zeros(())
        if confidence_pred is not None:
            if accept_rate_3d is None:
                raise ValueError(
                    "DSpark confidence head requires aligned target logits."
                )
            confidence_errors = F.binary_cross_entropy_with_logits(
                confidence_pred.float(),
                accept_rate_3d.detach(),
                reduction="none",
            )
            confidence_loss = (confidence_errors * loss_weight_mask).sum() / (
                ce_loss_den + 1e-6
            )
            with torch.no_grad():
                confidence_abs_error = (
                    (confidence_pred.float().sigmoid() - accept_rate_3d).abs()
                    * loss_weight_mask
                ).sum() / (ce_loss_den + 1e-6)

        loss = (
            self.dspark_ce_loss_alpha * ce_loss
            + self.dspark_l1_loss_alpha * l1_loss
            + self.dspark_confidence_head_alpha * confidence_loss
        )
        metrics = {
            "ce_loss": ce_loss.detach(),
            "l1_loss": l1_loss.detach(),
            "confidence_loss": confidence_loss.detach(),
            "confidence_abs_error": confidence_abs_error.detach(),
        }
        return loss, metrics

    def _dspark_loop_loss(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        sampled: Tuple[torch.Tensor, torch.Tensor],
        hidden_4d: torch.Tensor,
        stage_k: int,
        target_last_hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """DSpark loss (CE + L1 + BCE on the Markov-corrected logits) of DLoop
        stage ``stage_k``: labels t_{a+kγ+1..a+kγ+γ}, previous tokens
        [t_{a+kγ}, labels[:-1]]."""
        gamma = self.block_size
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchor_positions, block_keep_mask = sampled
        num_blocks = anchor_positions.size(1)
        loop_logits = self.lm_head(
            hidden_4d.reshape(bsz, num_blocks * gamma, -1)
        ).view(bsz, num_blocks, gamma, -1)

        label_indices = anchor_positions.unsqueeze(-1) + torch.arange(
            1, stage_k * gamma + gamma + 1, device=device
        ).view(1, 1, -1)
        safe_indices = label_indices.clamp(max=seq_len - 1)
        safe_indices = torch.where(
            block_keep_mask.unsqueeze(-1), safe_indices, torch.zeros_like(safe_indices)
        )
        target_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, num_blocks, -1), 2, safe_indices
        )
        # Only blocks whose whole ground-truth chain is valid are supervised.
        eval_full = (
            (label_indices < seq_len)
            & (target_loss_mask > 0.5)
            & block_keep_mask.unsqueeze(-1)
        )
        eval_full = eval_full.to(torch.int32).cumprod(dim=-1).bool()
        loop_eval = eval_full[:, :, -gamma:]
        loop_target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_indices[:, :, -gamma:],
        )
        prev_indices = (
            anchor_positions.unsqueeze(-1)
            + torch.arange(stage_k * gamma, stage_k * gamma + gamma, device=device).view(1, 1, -1)
        ).clamp(max=seq_len - 1)
        prev_indices = torch.where(
            block_keep_mask.unsqueeze(-1), prev_indices, torch.zeros_like(prev_indices)
        )
        loop_prev_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1), 2, prev_indices
        )

        loop_logits = self.draft_model.apply_logits_head(
            loop_logits,
            prev_token_ids=loop_prev_ids,
            hidden_states=hidden_4d,
        )
        loop_conf = self.draft_model.predict_confidence(
            hidden_4d,
            prev_token_ids=loop_prev_ids,
        )
        loop_aligned = self._aligned_target_logits(
            target_last_hidden_states,
            safe_indices[:, :, -gamma:],
        )
        loop_loss, _ = self._compute_dspark_loss(
            draft_logits=loop_logits,
            target_ids=loop_target_ids,
            eval_mask=loop_eval,
            confidence_pred=loop_conf,
            aligned_target_logits=loop_aligned,
        )
        with torch.no_grad():
            flat = loop_logits.reshape(-1, loop_logits.size(-1))
            correct = (torch.argmax(flat, dim=-1) == loop_target_ids.reshape(-1)) & loop_eval.reshape(-1)
            accuracy = correct.sum().float() / (loop_eval.sum().float() + 1e-6)
        return loop_loss, {"loop_accuracy": accuracy}

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel DSpark training forward pass."""
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz = input_ids.shape[0]
        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )

        logits = self.lm_head(output_hidden)
        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)
        (
            target_ids,
            eval_mask,
            safe_label_indices,
        ) = self._build_dspark_labels_and_mask(
            input_ids=input_ids,
            loss_mask=loss_mask,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = logits.reshape(bsz, num_blocks, self.block_size, -1)
        draft_logits = self.draft_model.apply_logits_head(
            draft_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=output_hidden_4d,
        )
        confidence_pred = self.draft_model.predict_confidence(
            output_hidden_4d,
            prev_token_ids=prev_token_ids,
        )
        aligned_target_logits = self._aligned_target_logits(
            target_last_hidden_states,
            safe_label_indices,
        )
        loss, metrics = self._compute_dspark_loss(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )
        if self.loop_max_stage:
            sampled = (anchor_positions, block_keep_mask)
            k = self._sample_loop_depth()
            stage_hidden = self._forward_loop_stages(
                input_ids, hidden_states, loss_mask, sampled, output_hidden_4d, k, lead=0
            )
            loop_loss, loop_metrics = self._dspark_loop_loss(
                input_ids, loss_mask, sampled, stage_hidden, k, target_last_hidden_states
            )
            w = self.loop_loss_weight
            loss = (1.0 - w) * loss + w * loop_loss
            metrics.update(loop_metrics)
        flat_logits = draft_logits.reshape(-1, draft_logits.size(-1))
        flat_targets = target_ids.reshape(-1)
        binary_eval_mask = eval_mask.reshape(-1)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & binary_eval_mask
            accuracy_denom = binary_eval_mask.to(torch.float32).sum()
            accuracy = correct.sum().float() / (accuracy_denom + 1e-6)
        metrics["accuracy_denom"] = accuracy_denom
        return loss, accuracy, metrics
