# coding=utf-8
# Derived from EAGLE (https://github.com/SafeAILab/EAGLE)
# Apache License 2.0
# Modified for DLoop

# Copyright 2025 SafeAI Lab (SAIL)
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EAGLE-3 dynamic-tree decoding + DLoop (lossless).

The host is the official EAGLE-3 inference path (``cnets.Model.topK_genrate`` and
``ea_model.eagenerate``) on a SpecForge ``LlamaForCausalLMEagle3`` checkpoint:
tuple KV cache over committed rows, ``tree_depth`` beam levels of ``topk_k`` rows
(one draft forward each), one global cumulative-log-prob rerank to
``tree_budget`` nodes, one target verify with a tree mask, longest accepted path.

DLoop: after the host levels, while fewer than ``stage_cap`` stages were drafted
and the gate is open -- the best surviving beam's summed log-prob over the
levels just drafted >= g -- run the same beam loop for ``stage_len`` more levels
and grow the rerank budget by ``stage_tree_budget``. A positive ``loop_gate`` never
opens (host as-is). Target attention must accept a 4D mask (sdpa).
"""

import math
import time
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache

from specforge.modeling.draft.dflash import extract_context_feature
from specforge.modeling.draft.llama3_eagle import (
    LlamaForCausalLMEagle3,
    _make_causal_mask,
    repeat_kv,
    rotate_half,
)

from .tree_decode import DEFAULT_STAGE_CAP, _compact_cache, build_tree_attention


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _apply_rope(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
    sin = sin.squeeze(1).squeeze(0)[position_ids].unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Eagle3EvalModel(nn.Module):
    """Inference wrapper: SpecForge EAGLE-3 weights + the official KV / tree path."""

    def __init__(self, core: LlamaForCausalLMEagle3):
        super().__init__()
        self.core = core
        self.config = core.config
        self.target_layer_ids = None
        self.stable_kv = None

    @classmethod
    def from_pretrained(cls, path: str, dtype=torch.bfloat16):
        return cls(LlamaForCausalLMEagle3.from_pretrained(path, dtype=dtype))

    def bind_target(self, target: nn.Module) -> None:
        """The checkpoint has no token embedding (it is the target's); the
        context features are the target layers [1, L/2-1, L-4] (SpecForge)."""
        weight = target.model.embed_tokens.weight
        self.core.embed_tokens.weight.data.copy_(weight.data.to(self.core.embed_tokens.weight.dtype))
        n = target.config.num_hidden_layers
        self.target_layer_ids = [1, n // 2 - 1, n - 4]

    def forward(self, hidden_states, input_ids, past_key_values=None, position_ids=None, tree_mask=None):
        """One draft forward (official ``cnets.Model.forward``): new rows on top
        of the tuple KV ``past_key_values``; ``tree_mask`` masks the beam block."""
        bsz, seq_len, _ = hidden_states.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        device = hidden_states.device

        inputs_embeds = self.core.embed_tokens(input_ids).to(hidden_states.dtype)
        if position_ids is None:
            position_ids = torch.arange(past_len, seq_len + past_len, dtype=torch.long, device=device)[None]
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        mask = _make_causal_mask((bsz, seq_len), torch.float32, device=device, past_key_values_length=past_len)
        if tree_mask is not None:
            t0, t1 = tree_mask.shape[2], tree_mask.shape[3]
            sub = mask[:, :, -t0:, -t1:]
            mask[:, :, -t0:, -t1:] = torch.where(tree_mask.to(torch.bool), sub, torch.finfo(torch.float32).min)

        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.core.fc(hidden_states)

        layer = self.core.midlayer
        residual = hidden_states
        x = torch.cat([layer.input_layernorm(inputs_embeds), layer.hidden_norm(hidden_states)], dim=-1)
        attn = layer.self_attn
        q = attn.q_proj(x).view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(bsz, seq_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        cos, sin = attn.rotary_emb(v, seq_len=seq_len + past_len)
        q, k = _apply_rope(q, k, cos, sin, position_ids)
        if past_key_values is not None:
            k = torch.cat([past_key_values[0][0], k], dim=2)
            v = torch.cat([past_key_values[0][1], v], dim=2)
        new_kv = ((k, v),)
        kr = repeat_kv(k, attn.num_key_value_groups)
        vr = repeat_kv(v, attn.num_key_value_groups)
        w = torch.matmul(q, kr.transpose(2, 3)) / math.sqrt(attn.head_dim) + mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(w, vr).transpose(1, 2).reshape(bsz, seq_len, attn.num_heads * attn.head_dim)
        h = residual + attn.o_proj(out)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h, new_kv

    def _head(self, hidden):
        return F.log_softmax(self.core.lm_head(self.core.norm(hidden)), dim=-1)

    @torch.no_grad()
    def topk_generate(self, hidden_states, input_ids, depth, top_k, total_tokens,
                      loop_gate, stage_cap, stage_len, stage_tree_budget):
        """Draft one round's tree. ``hidden_states`` = target features of the
        newly committed rows, ``input_ids`` = all committed ids (incl. the root)."""
        device = hidden_states.device
        d2t = self.core.d2t
        budget = total_tokens - 1  # root excluded
        rows = input_ids[:, 1:]    # row i pairs (hidden_i, token_{i+1})

        kv_len = self.stable_kv[0][0].shape[2] if self.stable_kv is not None else 0
        out_hidden, past_key_values = self(hidden_states, rows[:, kv_len:], past_key_values=self.stable_kv)
        self.stable_kv = past_key_values
        len_posi = rows.shape[1]

        last_hidden = out_hidden[:, -1]
        last_p = self._head(last_hidden)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list = [scores[None]]
        parents_list = [torch.zeros(1, dtype=torch.long, device=device)]
        ss_token = [topk_index + d2t[topk_index]]
        beam_ids = topk_index + d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask_init = torch.eye(top_k, device=device)[None, None]
        tree_mask = tree_mask_init
        topk_cs_index = torch.arange(top_k, device=device)
        level_pos = torch.zeros(top_k, device=device, dtype=torch.long)

        def level(i: int):
            nonlocal input_hidden, beam_ids, tree_mask, topk_cs_index, scores, past_key_values, len_posi
            out_hidden, past_key_values = self(
                input_hidden, beam_ids, past_key_values=past_key_values,
                position_ids=len_posi + level_pos, tree_mask=tree_mask,
            )
            len_posi += 1
            bias = 1 + top_k**2 * max(0, i - 1) + (top_k if i > 0 else 0)
            parents_list.append(topk_cs_index + bias)
            last_p = self._head(out_hidden[0])
            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values
            cu_scores = topk_p + scores[:, None]
            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, scores = topk_cs.indices, topk_cs.values
            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            beam_ids = topk_index.view(-1)[topk_cs_index][None]
            beam_ids = beam_ids + d2t[beam_ids]
            ss_token.append(topk_index + d2t[topk_index])
            scores_list.append(cu_scores)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], tree_mask_init), dim=3)

        for i in range(depth):
            level(i)

        # DLoop: the gate reads the best beam's log-prob gained since the last decision.
        fired, i = 0, depth
        cap = stage_cap if loop_gate <= 0 else 0
        base, cum = 0.0, float(scores.max())
        while fired < cap and cum - base >= loop_gate:
            base = cum
            for _ in range(stage_len):
                level(i)
                i += 1
            fired += 1
            budget += stage_tree_budget
            cum = float(scores.max())

        # global cum-logp rerank (ancestor-closed by monotonicity)
        scores_flat = torch.cat(scores_list, dim=0).view(-1)
        parents_all = torch.cat(parents_list, dim=0)
        ss_token_flat = torch.cat(ss_token, dim=0).view(-1)
        b = min(budget, scores_flat.shape[0])
        top_scores_index = torch.sort(torch.topk(scores_flat, b, dim=-1).indices).values
        tokens = ss_token_flat[top_scores_index]
        draft_parents = parents_all[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = (mask_index + 1).tolist()
        closure = torch.eye(b + 1, dtype=torch.bool)
        closure[:, 0] = True
        for j in range(b):
            closure[j + 1] |= closure[mask_index[j]]
        node_depth = (closure.sum(dim=1) - 1).tolist()
        return SimpleNamespace(
            tokens=tokens.tolist(),
            parents=[m - 1 for m in mask_index],  # -1 = root
            depth=node_depth[1:],
            fired_stages=fired,
        )


def load_drafter(draft_path: str, target: nn.Module, device) -> Eagle3EvalModel:
    draft = Eagle3EvalModel.from_pretrained(draft_path).to(device).eval()
    draft.bind_target(target)
    return draft


@torch.no_grad()
def eagle3_generate(
    model: Eagle3EvalModel,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    tree_depth: int = 7,
    tree_budget: int = 60,
    topk_k: int = 10,
    loop_gate: float = 1.0,
    stage_cap: int = DEFAULT_STAGE_CAP,
    stage_len: int = 8,
    stage_tree_budget: int = 48,
    return_stats: bool = False,
):
    """Defaults = the official EAGLE-3 tree (depth 7, 60 nodes, top-k 10)."""
    device = target.device
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens
    pad = tree_depth + 1 + stage_len * stage_cap + tree_budget + 2

    output_ids = torch.zeros((1, max_length + pad), dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device)[None]
    tgt_cache = DynamicCache()
    model.stable_kv = None

    prefill_start = _cuda_time() if return_stats else None
    out = target(
        input_ids, position_ids=position_ids[:, :num_input], past_key_values=tgt_cache,
        use_cache=True, logits_to_keep=1, output_hidden_states=True,
    )
    new_feat = extract_context_feature(out.hidden_states, model.target_layer_ids)
    output_ids[:, :num_input] = input_ids
    output_ids[0, num_input] = out.logits[0, -1].argmax(dim=-1)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    stages_fired = [0] * stage_cap
    start = num_input  # the round's root = last committed token

    while start < max_length:
        tree = model.topk_generate(
            new_feat, output_ids[:, : start + 1], tree_depth, topk_k, tree_budget,
            loop_gate, stage_cap, stage_len, stage_tree_budget,
        )
        for s in range(tree.fired_stages):
            stages_fired[s] += 1

        # one target forward over the root and the tree rows
        row_ids = torch.cat([
            output_ids[:, start : start + 1],
            torch.tensor([tree.tokens], dtype=torch.long, device=device),
        ], dim=1)
        par = [-1] + [p + 1 if p >= 0 else 0 for p in tree.parents]
        out = target(
            row_ids,
            position_ids=torch.tensor([[start] + [start + d for d in tree.depth]], device=device),
            attention_mask=build_tree_attention(par, start, target.dtype, device),
            past_key_values=tgt_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = out.logits[0].argmax(dim=-1).tolist()

        # longest root path matching the target's argmax, plus the bonus
        children: dict[int, list[int]] = {}
        for j in range(1, len(par)):
            children.setdefault(par[j], []).append(j)
        ids = row_ids[0].tolist()
        path = [0]
        while True:
            nxt = next((c for c in children.get(path[-1], []) if ids[c] == posterior[path[-1]]), None)
            if nxt is None:
                break
            path.append(nxt)
        acc = len(path) - 1
        output_ids[0, start + 1 : start + acc + 1] = row_ids[0, path[1:]]
        output_ids[0, start + acc + 1] = posterior[path[-1]]
        acceptance_lengths.append(acc + 1)

        keep = torch.tensor(path, device=device)
        _compact_cache(tgt_cache, start, keep)
        new_feat = extract_context_feature(out.hidden_states, model.target_layer_ids)[:, keep, :]
        start += acc + 1

        if stop_token_ids is not None and any(
            s in output_ids[0, num_input : start + 1] for s in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop = torch.tensor(stop_token_ids, device=device)
        hits = torch.isin(output_ids[0, num_input:], stop).nonzero(as_tuple=True)[0]
        if hits.numel() > 0:
            output_ids = output_ids[:, : num_input + hits[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=(_cuda_time() - decode_start) / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        stages_fired=stages_fired,
    )
