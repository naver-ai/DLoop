# coding=utf-8
# DLoop
# Copyright (c) 2026-present NAVER Corp.

"""MTP hosts: Qwen3.5's built-in MTP head and the Gemma 4 assistant, plus the
target-side plumbing they need.

Both hosts ship an autoregressive single-step drafter conditioned on the
target's last hidden state and the embedding of the last committed token
(DeepSeek-V3-style MTP): one drafter forward per drafted token, the drafter's
own output hidden feeding its next step.

* Qwen3.5 (``--draft-model mtp``, or a trained ``Qwen35MTPDraftModel``
  checkpoint): the ``mtp.*`` tensors of the target checkpoint (fc / pre-fc norms
  / one full-attention decoder layer / norm; embeddings and lm_head shared with
  the target). The layer keeps its own KV cache over committed rows (row at
  position i = token x_{i+1} + target hidden H_i); draft rows are dropped after
  every round and the accepted tokens are re-fed with their target hidden.
* Gemma 4 (``--draft-model google/gemma-4-*-it-assistant`` or a trained
  assistant checkpoint): the official assistant — every layer cross-attends to
  the *target's* K/V of its last sliding / full layer (``shared_kv_states``); no
  drafter cache, constant query position = the anchor's position. The drafter
  always sees the anchor's producer hidden and exactly the committed prefix
  (rows < anchor position).

Target side: Qwen3.5's Gated DeltaNet layers carry a recurrent state that a KV
``crop`` cannot rewind, so ``LinearStateRollback`` snapshots the states before a
multi-row verify and recomputes them over the accepted prefix.

Requires transformers >= 5.16.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

MTP_MODEL_FAMILIES = ("qwen3_5", "gemma4")
QWEN35_MTP_ARCH = "Qwen35MTPDraftModel"  # trained Qwen3.5 head (specforge.modeling.draft.mtp)


def is_mtp_model(config) -> bool:
    """A target of an MTP host family (Qwen3.5 / Gemma 4, any variant)."""
    model_type = str(getattr(config, "model_type", ""))
    return model_type.startswith(MTP_MODEL_FAMILIES) and not model_type.endswith("_assistant")


def is_qwen35_mtp_ckpt(draft: str | None) -> bool:
    if not draft or draft == "mtp":
        return False
    try:
        archs = AutoConfig.from_pretrained(draft).architectures or []
    except Exception:  # noqa: BLE001
        return False
    return archs == [QWEN35_MTP_ARCH]


def is_mtp_draft(draft: str | None) -> bool:
    """``mtp`` (Qwen3.5 built-in head), a trained Qwen3.5 head checkpoint, or a
    Gemma 4 ``*_assistant`` checkpoint."""
    if not draft:
        return False
    if draft == "mtp" or is_qwen35_mtp_ckpt(draft):
        return True
    try:
        model_type = AutoConfig.from_pretrained(draft).model_type
    except Exception:  # noqa: BLE001
        return False
    return str(model_type).endswith("_assistant")


def stop_token_ids(tokenizer, target) -> list[int]:
    """tokenizer eos + generation_config / config eos lists (Gemma 4 ends turns
    with <end_of_turn>=106, not the tokenizer's <eos>=1)."""
    ids = {tokenizer.eos_token_id}
    for src in (getattr(target, "generation_config", None), target.config,
                target.config.get_text_config()):
        v = getattr(src, "eos_token_id", None)
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
    ids.discard(None)
    return sorted(ids)


def has_linear_attention(target) -> bool:
    return "linear_attention" in (
        getattr(target.config.get_text_config(), "layer_types", None) or [])


def make_target_cache(target, rollback: bool = False) -> DynamicCache:
    """Hybrid (linear-attention) targets need the config-aware cache (+ past
    recording so ``crop`` can rewind the conv states); everything else gets
    plain full-length layers (sliding windows are enforced by the mask)."""
    if has_linear_attention(target):
        cache = DynamicCache(config=target.config.get_text_config())
        if rollback:
            cache.activate_past_recording()
        return cache
    return DynamicCache()


def load_mtp_target(model_name: str, device, attn_impl: str = "sdpa") -> nn.Module:
    """Load straight onto ``device`` (no full host-memory copy per rank)."""
    return AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation=attn_impl, dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()


class LinearStateRollback:
    """Exact rollback of Qwen3.5 Gated DeltaNet states after a multi-row verify.

    ``snapshot`` clones every linear layer's recurrent state and arms capture of
    the per-row ``in_proj_a`` / ``in_proj_b`` outputs of the next forward;
    ``rollback(cache, keep, total)`` recomputes the state over the first ``keep``
    of the ``total`` rows that forward fed (conv over the recorded conv history,
    then the same chunk kernel from the snapshot) and crops the cache by
    ``total - keep``."""

    def __init__(self, target: nn.Module):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

        self._q35 = q35
        self.mods = [
            (i, layer.linear_attn) for i, layer in enumerate(target.model.layers)
            if getattr(layer, "linear_attn", None) is not None
        ]
        self.capturing = False
        self._ab: dict[int, dict[str, torch.Tensor]] = {}
        self._rec0: dict[int, torch.Tensor] = {}
        for i, m in self.mods:
            m.in_proj_a.register_forward_hook(self._hook(i, "a"))
            m.in_proj_b.register_forward_hook(self._hook(i, "b"))

    def _hook(self, i: int, key: str):
        def fn(_mod, _inp, out):
            if self.capturing:
                self._ab.setdefault(i, {})[key] = out
        return fn

    def snapshot(self, cache: DynamicCache) -> None:
        for i, _ in self.mods:
            self._rec0[i] = cache.layers[i].recurrent_states[0].clone()
        self._ab.clear()
        self.capturing = True

    def rollback(self, cache: DynamicCache, keep: int, total: int) -> None:
        self.capturing = False
        if keep < total:
            q35 = self._q35
            for i, m in self.mods:
                layer = cache.layers[i]
                hist = layer.conv_states[0]  # [B, C, prev + total] (recording mode)
                x = hist[..., : hist.shape[-1] - (total - keep)]
                y = q35.causal_conv1d_fn(
                    x, m.conv1d.weight.squeeze(1), m.conv1d.bias, activation=m.activation,
                )[:, :, -keep:].transpose(1, 2)
                query, key, value = torch.split(
                    y, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
                bsz = y.shape[0]
                query = query.reshape(bsz, keep, -1, m.head_k_dim)
                key = key.reshape(bsz, keep, -1, m.head_k_dim)
                value = value.reshape(bsz, keep, -1, m.head_v_dim)
                rep = m.num_v_heads // m.num_k_heads
                if rep > 1:
                    query = query.repeat_interleave(rep, dim=2)
                    key = key.repeat_interleave(rep, dim=2)
                a = self._ab[i]["a"][:, :keep]
                b = self._ab[i]["b"][:, :keep]
                g = -m.A_log.float().exp() * F.softplus(a.float() + m.dt_bias)
                _, state = q35.torch_chunk_gated_delta_rule(
                    query, key, value, g=g, beta=b.sigmoid(),
                    initial_state=self._rec0[i], output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                layer.recurrent_states[0].copy_(state)
        cache.crop(-(total - keep))
        self._ab.clear()
        self._rec0.clear()


class Qwen35MTPDrafter(nn.Module):
    """Qwen3.5 MTP head as a drafter over the target (shared embeddings /
    lm_head / rotary)."""

    has_prefill = True        # the first begin_round() builds the KV over the prompt
    needs_shared_kv = False

    def __init__(self, target: nn.Module):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
        )

        tcfg = copy.deepcopy(target.config.get_text_config())
        tcfg.layer_types = ["full_attention"]
        tcfg.num_hidden_layers = 1
        self.config = tcfg
        hidden = tcfg.hidden_size
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=tcfg.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden, eps=tcfg.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(tcfg, 0)])
        self.norm = Qwen3_5RMSNorm(hidden, eps=tcfg.rms_norm_eps)
        object.__setattr__(self, "target", target)  # shared, not a submodule
        self.rollback = LinearStateRollback(target) if has_linear_attention(target) else None
        self.cache: DynamicCache | None = None
        self._h: torch.Tensor | None = None
        self._draft_rows = 0

    def load_weights(self, model_name_or_path: str) -> None:
        """``mtp.*`` of a Qwen3.5 target checkpoint (public head) or a saved
        ``Qwen35MTPDraftModel`` directory (trained head, same module names)."""
        from specforge.modeling.draft.mtp import load_qwen35_mtp_state

        state = load_qwen35_mtp_state(model_name_or_path)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"mtp weights mismatch: missing={missing} unexpected={unexpected}")

    def reset(self) -> None:
        self.cache = DynamicCache()
        self._draft_rows = 0

    def observe_target(self, out, n_committed: int) -> None:
        pass

    def forward_rows(self, token_ids, hidden, positions):
        """Rows (token x_{p+1}, hidden H_p) at positions p -> (logits, mtp hidden)."""
        from transformers.masking_utils import create_causal_mask

        tgt = self.target
        emb = tgt.get_input_embeddings()(token_ids)
        x = self.fc(torch.cat(
            [self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hidden.to(emb.dtype))], dim=-1))
        pos_emb = tgt.model.rotary_emb(x, positions)
        mask = create_causal_mask(config=self.config, inputs_embeds=x, attention_mask=None,
                                  past_key_values=self.cache, position_ids=positions)
        h = self.layers[0](x, position_embeddings=pos_emb, attention_mask=mask,
                           position_ids=positions, past_key_values=self.cache, use_cache=True)
        h = self.norm(h)
        return tgt.lm_head(h), h

    def begin_round(self, tokens, hiddens, start: int):
        """Commit rows (new tokens ending with the anchor at ``start`` + their
        producer hidden) -> logits of the anchor row (= first draft)."""
        m = tokens.shape[1]
        pos = torch.arange(start - m, start, device=tokens.device)[None]
        logits, h = self.forward_rows(tokens, hiddens, pos)
        self._h = h[:, -1:]
        self._draft_rows = 0
        return logits[:, -1:]

    def step(self, token, start: int, j: int):
        """Draft row j (>= 1): token d_j + the previous MTP hidden at position start + j - 1."""
        pos = torch.tensor([[start + j - 1]], device=token.device)
        logits, h = self.forward_rows(token, self._h, pos)
        self._h = h
        self._draft_rows += 1
        return logits

    def end_round(self) -> None:
        if self._draft_rows:
            self.cache.crop(-self._draft_rows)
            self._draft_rows = 0


class Gemma4AssistantDrafter(nn.Module):
    """Gemma 4 assistant over the target: cross-attends to the target's shared
    K/V of the committed prefix, query position fixed at the anchor."""

    has_prefill = False
    needs_shared_kv = True

    def __init__(self, assistant: nn.Module, target: nn.Module):
        super().__init__()
        self.assistant = assistant
        object.__setattr__(self, "target", target)
        self.rollback = None
        self._kv: dict | None = None
        self._h: torch.Tensor | None = None

    def reset(self) -> None:
        self._kv = None

    def observe_target(self, out, n_committed: int) -> None:
        """Keep the target K/V rows of the committed prefix (positions < n_committed)."""
        self._kv = {k: (v[0][:, :, :n_committed], v[1][:, :, :n_committed])
                    for k, v in out.shared_kv_states.items()}

    def _step(self, token, hidden, position: int):
        emb = self.target.get_input_embeddings()(token)
        out = self.assistant(
            inputs_embeds=torch.cat([emb, hidden.to(emb.dtype)], dim=-1),
            position_ids=torch.tensor([[position]], device=emb.device),
            shared_kv_states=self._kv, use_cache=False,
        )
        self._h = out.last_hidden_state
        return out.logits

    def begin_round(self, tokens, hiddens, start: int):
        return self._step(tokens[:, -1:], hiddens[:, -1:], start)

    def step(self, token, start: int, j: int):
        return self._step(token, self._h, start)

    def end_round(self) -> None:
        pass


def load_drafter(draft_path: str, target: nn.Module, device, attn_impl: str = "sdpa"):
    """``mtp`` / a trained Qwen3.5 head checkpoint (Qwen3.5 target), or a Gemma 4
    assistant checkpoint (Gemma 4 target)."""
    if draft_path == "mtp" or is_qwen35_mtp_ckpt(draft_path):
        if target.config.get_text_config().model_type not in ("qwen3_5", "qwen3_5_text"):
            raise ValueError(f"--draft-model {draft_path} needs a Qwen3.5 target, "
                             f"got {target.config.model_type}")
        drafter = Qwen35MTPDrafter(target)
        drafter.load_weights(target.config._name_or_path if draft_path == "mtp" else draft_path)
        return drafter.to(device=device, dtype=torch.bfloat16).eval()
    assistant = AutoModelForCausalLM.from_pretrained(
        draft_path, attn_implementation=attn_impl, dtype=torch.bfloat16, device_map={"": device},
    ).eval()
    if not hasattr(assistant, "pre_projection"):
        raise ValueError(f"{draft_path} is not a Gemma 4 assistant checkpoint")
    return Gemma4AssistantDrafter(assistant, target).eval()
