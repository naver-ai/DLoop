# coding=utf-8
# DLoop
# Copyright (c) 2026-present NAVER Corp.

"""MTP-host draft models for the 1-epoch DLoop adaptation.

Two autoregressive drafters that ship *with* their targets, trained with the
EAGLE-3-style TTT unroll (``specforge.core.mtp.OnlineMTPModel``):

* :class:`Qwen35MTPDraftModel` — Qwen3.5's built-in MTP head (the ``mtp.*``
  tensors of the target checkpoint): fc over [norm(embedding), norm(hidden)] ->
  one full-attention decoder layer with its own KV cache -> norm -> the target's
  lm_head. Same module names as ``eval.mtp_model.Qwen35MTPDrafter``, so a saved
  checkpoint drops into the evaluation unchanged.
* :class:`Gemma4AssistantDraftModel` — the official Gemma 4 assistant
  (``Gemma4UnifiedAssistantForCausalLM``) as-is: every layer cross-attends to the
  target's ``shared_kv_states`` (no drafter KV), query position fixed at the
  anchor. Saved checkpoints are standard HF assistant checkpoints.

Both expose the TTT protocol consumed by ``OnlineMTPModel``: ``bind_target``
(frozen target for embeddings / lm_head), ``ttt_begin`` (per-sequence state),
``ttt_step`` (one unroll step over all rows -> (hidden fed to the next step,
hidden fed to the head)) and ``head``.

The step-``t`` layout mirrors deployment (row ``p`` of step ``t`` = the ``t``-th
draft of the round anchored at ``p``):

  Qwen3.5  row p, step t: token x_{p+1+t} + previous hidden, RoPE position p+t.
           Its KV view is the deployment cache of the round anchored at p: the
           step-0 rows at indices <= p (the committed rows, built from the TARGET
           hidden) plus, from every later step block, only its own row p (that
           round's earlier draft rows).
  Gemma 4  row p, step t: token x_{p+1+t} + previous hidden, position p+1 for
           every step (the anchor's position), attending the target K/V rows <= p
           (full) / the last sliding_window+1 of them (sliding) — what the
           official single-position generator does per round.

Requires transformers >= 5.16 (Qwen3.5 / Gemma 4 unified model code).
"""

from __future__ import annotations

import copy
import glob
import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import transformers
from packaging import version
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint

if version.parse(transformers.__version__) < version.parse("5.16.0"):
    raise ImportError(
        "The MTP hosts (Qwen3.5 MTP head / Gemma 4 assistant) need transformers>=5.16, "
        f"found {transformers.__version__}"
    )

from transformers import AutoConfig, DynamicCache, PreTrainedModel  # noqa: E402
from transformers.models.gemma4_unified_assistant.configuration_gemma4_unified_assistant import (  # noqa: E402
    Gemma4UnifiedAssistantConfig,
)
from transformers.models.gemma4_unified_assistant.modeling_gemma4_unified_assistant import (  # noqa: E402
    Gemma4UnifiedAssistantForCausalLM,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)
from transformers.utils.hub import cached_file  # noqa: E402

from .registry import register_draft  # noqa: E402

QWEN35_MTP_PREFIX = "mtp."


def load_qwen35_mtp_state(model_name_or_path: str) -> Dict[str, torch.Tensor]:
    """The Qwen3.5 MTP head weights as an unprefixed state dict, from either a
    Qwen3.5 *target* checkpoint (its ``mtp.*`` tensors) or a saved
    :class:`Qwen35MTPDraftModel` directory (keys already unprefixed)."""
    local = model_name_or_path if os.path.isdir(model_name_or_path) else None
    index = (
        os.path.join(local, "model.safetensors.index.json")
        if local is not None
        else cached_file(
            model_name_or_path,
            "model.safetensors.index.json",
            _raise_exceptions_for_missing_entries=False,
        )
    )
    if index is not None and os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(
            {fn for k, fn in weight_map.items() if k.startswith(QWEN35_MTP_PREFIX)}
        ) or sorted(set(weight_map.values()))
    elif local is not None:
        files = [os.path.basename(p) for p in glob.glob(os.path.join(local, "*.safetensors"))]
    else:
        files = ["model.safetensors"]

    prefixed, plain = {}, {}
    for fn in files:
        path = os.path.join(local, fn) if local is not None else cached_file(model_name_or_path, fn)
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(QWEN35_MTP_PREFIX):
                    prefixed[k[len(QWEN35_MTP_PREFIX):]] = f.get_tensor(k)
                else:
                    plain[k] = f.get_tensor(k)
    if prefixed:
        return prefixed
    head_modules = {"fc", "pre_fc_norm_embedding", "pre_fc_norm_hidden", "layers", "norm"}
    if plain and all(k.split(".")[0] in head_modules for k in plain):
        return plain
    raise ValueError(f"{model_name_or_path}: no Qwen3.5 mtp.* tensors found")


def _maybe_checkpoint(fn, *args):
    """Activation checkpointing of one unroll step (recomputed in backward, so
    only its inputs stay alive across the unroll). ``fn`` must be pure."""
    if torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def _attn_mask(keep: torch.Tensor, attn_impl: Optional[str], dtype: torch.dtype) -> torch.Tensor:
    """Boolean keep-mask -> what the attention implementation expects
    (sdpa: boolean; others: additive float)."""
    if attn_impl == "sdpa":
        return keep
    return torch.zeros(keep.shape, dtype=dtype, device=keep.device).masked_fill(
        ~keep, torch.finfo(dtype).min
    )


@register_draft
class Qwen35MTPDraftModel(PreTrainedModel):
    """Qwen3.5 built-in MTP head as a trainable draft model."""

    config_class = Qwen3_5TextConfig
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _supports_sdpa = True
    _supports_attention_backend = True

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        hidden = config.hidden_size
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, 0)])
        self.norm = Qwen3_5RMSNorm(hidden, eps=config.rms_norm_eps)
        # rotary is built lazily on the compute device (not a submodule: nothing to save)
        object.__setattr__(self, "_rotary", None)
        object.__setattr__(self, "target", None)
        self.post_init()

    @classmethod
    def config_from_target(cls, target_config) -> Qwen3_5TextConfig:
        cfg = copy.deepcopy(target_config.get_text_config())
        cfg.layer_types = ["full_attention"]
        cfg.num_hidden_layers = 1
        cfg.architectures = [cls.__name__]
        cfg._attn_implementation = "sdpa"
        return cfg

    @classmethod
    def from_target(cls, target_name_or_path: str, dtype: torch.dtype = torch.bfloat16):
        """Initialise from the ``mtp.*`` tensors of a Qwen3.5 target checkpoint."""
        cfg = cls.config_from_target(AutoConfig.from_pretrained(target_name_or_path))
        model = cls(cfg)
        model.load_state_dict(load_qwen35_mtp_state(target_name_or_path), strict=True)
        return model.to(dtype)

    # --- TTT protocol ----------------------------------------------------
    def bind_target(self, target: nn.Module) -> None:
        object.__setattr__(self, "target", target)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.target.get_input_embeddings()(input_ids)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.target.lm_head(hidden)

    def _rotary_emb(self, x: torch.Tensor, position_ids: torch.Tensor):
        rot = self._rotary
        if rot is None or rot.inv_freq.device != x.device:
            rot = Qwen3_5TextRotaryEmbedding(self.config).to(x.device)
            object.__setattr__(self, "_rotary", rot)
        return rot(x, position_ids)

    def rows(self, inputs_embeds, hidden, position_ids, attention_mask, cache):
        """Rows (token embedding, hidden) at ``position_ids`` -> post-norm hidden."""
        x = self.fc(
            torch.cat(
                [
                    self.pre_fc_norm_embedding(inputs_embeds),
                    self.pre_fc_norm_hidden(hidden.to(inputs_embeds.dtype)),
                ],
                dim=-1,
            )
        )
        pos_emb = self._rotary_emb(x, position_ids)
        h = self.layers[0](
            x,
            position_embeddings=pos_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=cache is not None,
        )
        return self.norm(h)

    def ttt_begin(self, seq_len: int, device: torch.device, target_ctx: dict) -> dict:
        idx = torch.arange(seq_len, device=device)
        return {
            "pos": idx[None],
            # step-0 block = the committed rows -> causal; later blocks = this
            # row's own round only -> diagonal (see the module docstring)
            "commit": idx[None, :] <= idx[:, None],
            "own": idx[None, :] == idx[:, None],
            "k": None,  # K/V of the previous step blocks, [B, kvh, step*S, D]
            "v": None,
        }

    def _step_fn(self, inputs_embeds, hidden, position_ids, attention_mask, past_k, past_v):
        """One unroll step as a pure function (checkpoint-safe): the previous
        blocks' K/V come in as tensors, the new block's K/V go out."""
        cache = DynamicCache()
        if past_k is not None:
            cache.update(past_k, past_v, 0)
        h = self.rows(inputs_embeds, hidden, position_ids, attention_mask, cache)
        n = inputs_embeds.shape[1]
        layer = cache.layers[0]
        return h, layer.keys[:, :, -n:], layer.values[:, :, -n:]

    def ttt_step(self, ctx: dict, step: int, inputs_embeds: torch.Tensor, hidden: torch.Tensor):
        # [1, 1, S, (step+1)*S]: block 0 causal, blocks 1..step diagonal
        keep = torch.cat([ctx["commit"]] + [ctx["own"]] * step, dim=1)[None, None]
        mask = _attn_mask(keep, self.config.get_text_config()._attn_implementation, inputs_embeds.dtype)
        h, k_new, v_new = _maybe_checkpoint(
            self._step_fn, inputs_embeds, hidden, ctx["pos"] + step, mask, ctx["k"], ctx["v"]
        )
        ctx["k"] = k_new if ctx["k"] is None else torch.cat([ctx["k"], k_new], dim=2)
        ctx["v"] = v_new if ctx["v"] is None else torch.cat([ctx["v"], v_new], dim=2)
        return h, h


class Gemma4AssistantDraftModel(Gemma4UnifiedAssistantForCausalLM):
    """Official Gemma 4 assistant as a trainable draft model (weights, module
    names and saved format unchanged — only the TTT protocol is added)."""

    config_class = Gemma4UnifiedAssistantConfig

    def __init__(self, config: Gemma4UnifiedAssistantConfig):
        super().__init__(config)
        if config.use_ordered_embeddings:
            raise NotImplementedError("Gemma 4 assistant with use_ordered_embeddings is not supported")
        object.__setattr__(self, "target", None)

    def bind_target(self, target: nn.Module) -> None:
        object.__setattr__(self, "target", target)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.target.get_input_embeddings()(input_ids)

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def ttt_begin(self, seq_len: int, device: torch.device, target_ctx: dict) -> dict:
        text_cfg = self.config.get_text_config()
        idx = torch.arange(seq_len, device=device)
        q, c = idx[:, None], idx[None, :]
        full = c <= q  # anchor p+1 sees the target K/V rows < p+1
        swa = full & (c >= q - text_cfg.sliding_window)  # the last sliding_window+1 of them
        dtype = self.lm_head.weight.dtype
        impl = text_cfg._attn_implementation
        return {
            "masks": {
                "full_attention": _attn_mask(full[None, None], impl, dtype),
                "sliding_attention": _attn_mask(swa[None, None], impl, dtype),
            },
            "pos": (idx + 1)[None],
            "kv": target_ctx["shared_kv"],
        }

    def _step_fn(self, inputs_embeds, hidden, position_ids, mask_full, mask_swa, shared_kv):
        x = self.pre_projection(torch.cat([inputs_embeds, hidden.to(inputs_embeds.dtype)], dim=-1))
        out = self.model(
            inputs_embeds=x,
            attention_mask={"full_attention": mask_full, "sliding_attention": mask_swa},
            position_ids=position_ids,
            shared_kv_states=shared_kv,
            use_cache=False,
        )
        z = out.last_hidden_state
        return self.post_projection(z), z

    def ttt_step(self, ctx: dict, step: int, inputs_embeds: torch.Tensor, hidden: torch.Tensor):
        return _maybe_checkpoint(
            self._step_fn, inputs_embeds, hidden, ctx["pos"],
            ctx["masks"]["full_attention"], ctx["masks"]["sliding_attention"], ctx["kv"],
        )


# registered under its own name (what save_pretrained writes) and under the
# official architecture name (so `--ckpt-dir <official assistant>` resolves)
register_draft(Gemma4AssistantDraftModel)
register_draft(Gemma4AssistantDraftModel, name="Gemma4UnifiedAssistantForCausalLM")


__all__ = ["Qwen35MTPDraftModel", "Gemma4AssistantDraftModel", "load_qwen35_mtp_state"]
