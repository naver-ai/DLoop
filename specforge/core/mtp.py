# coding=utf-8
# DLoop
# Copyright (c) 2026-present NAVER Corp.

"""Online TTT trainer for the MTP hosts (Qwen3.5 MTP head / Gemma 4 assistant).

The 1-epoch DLoop adaptation of an autoregressive drafter is the EAGLE-3 recipe
(``OnlineEagle3Model``): unroll the drafter ``L`` steps over the whole sequence
with teacher-forced tokens and *self-generated* hidden states, a loss per step,
per-step weights ``0.8**i`` (``train_eagle3.run_backward_and_update``), full
backprop through the unroll, and ``L`` drawn uniformly from ``length_dist``
(the host's draft depth k times {1, 2, 3} stages).

What differs from EAGLE-3 is only what the hosts dictate:
  * target features = the target's last hidden state (post final norm, the
    tensor the evaluation feeds the drafter) — no aux layers;
  * embeddings / lm_head are the frozen target's (Qwen3.5) or the assistant's
    own (Gemma 4) — full vocabulary, no draft-vocab mapping;
  * loss = hard cross-entropy on the next data token, computed chunk-wise under
    activation checkpointing so the [S, V] logits are never held for the whole
    unroll;
  * row geometry / positions / masks follow each host's deployment
    (``specforge.modeling.draft.mtp``).

Row ``p`` at step ``t`` consumes token ``x_{p+1+t}`` (+ the step ``t-1`` hidden,
or the target hidden ``H_p`` at ``t=0``) and is supervised on ``x_{p+2+t}``
when that label is an assistant token.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

# Byte budget for one fp32 logits chunk of the CE (recomputed in backward, so
# its grad doubles it): the MTP hosts have 248k / 262k vocabularies.
CE_CHUNK_BYTES = 256 * 2**20


def mtp_family(config) -> str:
    model_type = str(getattr(config, "model_type", ""))
    if model_type.startswith("gemma4"):
        return "gemma4"
    if model_type.startswith("qwen3_5"):
        return "qwen3.5"
    raise ValueError(f"not an MTP-host target: model_type={model_type!r}")


def load_mtp_target(model_name_or_path: str, device) -> nn.Module:
    """Frozen HF target straight onto ``device`` (sdpa: the drafter's custom
    masks need it)."""
    target = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
    ).eval()
    target.requires_grad_(False)
    return target


def shift_left(x: torch.Tensor, n: int) -> torch.Tensor:
    """``out[:, p] = x[:, p + n]``, zero-filled tail."""
    if n == 0:
        return x
    return torch.cat([x[:, n:], x.new_zeros(x.shape[0], min(n, x.shape[1]))], dim=1)


def _ce_chunk(head_fn: Callable, z: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor):
    logits = head_fn(z).float()
    loss = (F.cross_entropy(logits, labels, reduction="none") * valid).sum()
    with torch.no_grad():
        correct = ((logits.argmax(-1) == labels) & valid).sum()
    return loss, correct


def ce_chunk_rows(vocab_size: int) -> int:
    """Rows per CE chunk so one fp32 [rows, vocab] logits tensor stays within
    ``CE_CHUNK_BYTES``."""
    return max(64, CE_CHUNK_BYTES // (4 * max(vocab_size, 1)))


def chunked_cross_entropy(
    head_fn: Callable,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    chunk_rows: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sum of CE over valid rows, #top-1 correct, #valid — head + CE recomputed
    in backward per chunk so only ``hidden`` is kept alive across the unroll."""
    z = hidden.reshape(-1, hidden.shape[-1])
    lab = labels.reshape(-1)
    val = valid.reshape(-1).bool()
    loss_sum = z.new_zeros((), dtype=torch.float32)
    correct = torch.zeros((), dtype=torch.long, device=z.device)
    for i in range(0, z.shape[0], chunk_rows):
        l, c = checkpoint(
            _ce_chunk, head_fn, z[i : i + chunk_rows], lab[i : i + chunk_rows],
            val[i : i + chunk_rows], use_reentrant=False,
        )
        loss_sum = loss_sum + l
        correct = correct + c
    return loss_sum, correct.float(), val.sum().float()


class OnlineMTPModel(nn.Module):
    """TTT unroll of an MTP-host drafter over a frozen target. Returns the
    ``OnlineEagle3Model`` 7-tuple (per-step lists zero-padded to ``length``) so
    the ``train_eagle3.py`` logging / backward stay unchanged."""

    def __init__(
        self,
        target: nn.Module,
        draft_model: nn.Module,
        length: int = 7,
        length_dist: Optional[List[int]] = None,
    ):
        super().__init__()
        self.draft_model = draft_model
        # frozen, not a submodule (FSDP wraps only the drafter)
        object.__setattr__(self, "target", target)
        target.requires_grad_(False)
        draft_model.bind_target(target)
        self.family = mtp_family(target.config)
        self.length_dist = length_dist
        if length_dist:
            length = max(length_dist)
        self.length = length
        self._length_rng = random.Random(1234)  # same draw on every rank
        self.ce_chunk_rows = ce_chunk_rows(target.config.get_text_config().vocab_size)

    def _sample_length(self) -> int:
        if not self.length_dist:
            return self.length
        return self._length_rng.choice(self.length_dist)

    @torch.no_grad()
    def capture_target(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Target last hidden state (post-norm) + what the drafter cross-attends to."""
        base = self.target.model
        if self.family == "gemma4":
            out = base(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                return_shared_kv_states=True,
            )
            kv = {k: (v[0], v[1]) for k, v in out.shared_kv_states.items()}
            return out.last_hidden_state, {"shared_kv": kv}
        out = base(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state, {}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor):
        hidden, target_ctx = self.capture_target(input_ids, attention_mask)
        seq_len = input_ids.shape[1]
        cur_length = self._sample_length()
        draft = self.draft_model
        ctx = draft.ttt_begin(seq_len, input_ids.device, target_ctx)

        plosses, acces, corrects, denoms, metric_losses, loss_denoms = [], [], [], [], [], []
        for t in range(cur_length):
            inputs_embeds = draft.embed(shift_left(input_ids, 1 + t))
            hidden, z = draft.ttt_step(ctx, t, inputs_embeds, hidden)
            labels = shift_left(input_ids, 2 + t)
            valid = shift_left(loss_mask, 2 + t)
            loss_sum, correct, denom = chunked_cross_entropy(
                draft.head, z, labels, valid, chunk_rows=self.ce_chunk_rows
            )
            loss = loss_sum / denom.clamp_min(1.0)
            plosses.append(loss)
            acces.append(correct / denom.clamp_min(1.0))
            corrects.append(correct)
            denoms.append(denom)
            metric_losses.append(loss.detach())
            loss_denoms.append(denom)

        # zero-pad to the fixed max length (rank-invariant collective shapes)
        for _ in range(self.length - cur_length):
            plosses.append(torch.zeros_like(plosses[0]))
            acces.append(torch.zeros_like(acces[0]))
            corrects.append(torch.zeros_like(corrects[0]))
            denoms.append(torch.zeros_like(denoms[0]))
            metric_losses.append(torch.zeros_like(metric_losses[0]))
            loss_denoms.append(torch.zeros_like(loss_denoms[0]))
        acceptance_rates = [torch.zeros_like(a) for a in acces]  # no target distribution
        return plosses, acceptance_rates, acces, corrects, denoms, metric_losses, loss_denoms


__all__ = [
    "OnlineMTPModel",
    "ce_chunk_rows",
    "chunked_cross_entropy",
    "load_mtp_target",
    "mtp_family",
    "shift_left",
]
