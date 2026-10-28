# coding=utf-8
# DLoop
# Copyright (c) 2026-present NAVER Corp.

"""Tree verification + DLoop for DFlash drafters (lossless).

One round:
  1. draft the main block at the anchor -> log-probs of its B-1 mask rows
  2. best-first draft tree over them (budget N, DDTree-style)
  3. DLoop: while fewer than `stage_cap` stages were drafted, the gate is open
     (sum of top-1 log-probs of the segment just drafted >= g) and that
     segment's top-1 chain is a path of the tree, draft one more stage in the
     layout DLoop trains on, [last drafted token, MASK x B] with the earlier
     drafted tokens' hidden states as carriers, and hang its best-first
     subtree (budget e) under the end of the chain
  4. ONE target forward over the anchor and all tree rows (tree attention mask)
  5. accept the longest root path matching the target's argmax, plus a bonus
  6. compact the target KV cache and context features to the accepted path

A positive loop_gate never opens (tree only); linear decoding, with or without
DLoop, is linear_decode.py. The target must use an attention implementation
that accepts a 4D mask (sdpa).
"""

import heapq
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import DynamicCache

from specforge.modeling.draft.dflash import extract_context_feature

# Default stage cap; bounds padding and the verify width, firing decays long before it.
DEFAULT_STAGE_CAP = 16


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def build_bestfirst(vals, idx, budget: int):
    """Best-first tree over per-depth top-k marginals [D, K] (log-probs, CPU).

    Max-heap on cumulative log-prob with lazy sibling expansion; prefix-closed
    since log-probs <= 0. Returns (toks, par, dep) with node 0 = root and
    ``budget`` nodes below it."""
    D, K = vals.shape
    v = vals.tolist()
    ix = idx.tolist()
    toks, par, dep = [None], [-1], [0]
    # entry: (-score, depth, parent_node, rank_at_depth)
    heap = [(-v[0][0], 1, 0, 0)]
    while heap and len(toks) - 1 < budget:
        ns, d, p, r = heapq.heappop(heap)
        s = -ns
        toks.append(ix[d - 1][r])
        par.append(p)
        dep.append(d)
        node = len(toks) - 1
        if d < D:
            heapq.heappush(heap, (-(s + v[d][0]), d + 1, node, 0))
        if r + 1 < K:
            parent_score = s - v[d - 1][r]
            heapq.heappush(heap, (-(parent_score + v[d - 1][r + 1]), d, p, r + 1))
    return toks, par, dep


def _tree_keep(par: list[int]) -> torch.Tensor:
    """[T, T] bool: row i keeps its root path (ancestors and itself)."""
    T = len(par)
    keep = np.zeros((T, T), dtype=bool)
    keep[0, 0] = True
    for i in range(1, T):
        keep[i] = keep[par[i]]
        keep[i, i] = True
    return torch.from_numpy(keep)


def build_tree_attention(par: list[int], prefix_len: int, dtype, device):
    """Additive mask [1, 1, T, prefix_len + T]: every row sees the committed
    prefix and its own root path."""
    T = len(par)
    mask = torch.zeros(T, prefix_len + T, dtype=dtype, device=device)
    mask[:, prefix_len:].masked_fill_(~_tree_keep(par).to(device), torch.finfo(dtype).min)
    return mask[None, None]


def _children_index(par: list[int]) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {}
    for i in range(1, len(par)):
        ch.setdefault(par[i], []).append(i)
    return ch


def _top1_leaf(toks, par, top1: list[int], root: int = 0) -> Optional[int]:
    """Node at the end of the top-1 chain walked from ``root``, or None if the
    chain is not fully contained in the tree."""
    ch = _children_index(par)
    cur = root
    for want in top1:
        nxt = next((c for c in ch.get(cur, []) if toks[c] == want), None)
        if nxt is None:
            return None
        cur = nxt
    return cur


def _compact_cache(cache: DynamicCache, prefix_len: int, keep_rows: torch.Tensor):
    """Keep the prefix plus ``keep_rows`` (row indices into the appended block)."""
    idx = torch.cat([torch.arange(prefix_len, device=keep_rows.device), prefix_len + keep_rows])
    for layer in cache.layers:
        layer.keys = layer.keys.index_select(2, idx)
        layer.values = layer.values.index_select(2, idx)


@torch.inference_mode()
def dflash_tree_generate(
    model: nn.Module,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    tree_budget: int = 256,
    stage_tree_budget: int = 64,
    topk_k: Optional[int] = None,
    loop_gate: float = -3.0,
    stage_cap: int = DEFAULT_STAGE_CAP,
    return_stats: bool = False,
):
    """Each loop stage has the main block's layout ([anchor + MASK x (B-1)], B rows ->
    B-1 tokens) with the previous stage's last token as its anchor; a positive
    ``loop_gate`` never opens."""
    block = model.block_size
    mask_token_id = model.mask_token_id
    device = target.device
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens
    # a round commits at most the main block + every stage + the bonus
    pad = block + block * max(2, stage_cap) + 2

    output_ids = torch.full((1, max_length + pad), mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device)[None]
    tgt_cache = DynamicCache()
    draft_cache = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    out = target(
        input_ids,
        position_ids=position_ids[:, :num_input],
        past_key_values=tgt_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    next_tok = out.logits[0, -1].argmax(dim=-1)
    target_hidden = extract_context_feature(out.hidden_states, model.target_layer_ids)
    output_ids[:, :num_input] = input_ids
    output_ids[:, num_input] = next_tok
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    draft_prefill = True
    acceptance_lengths = []
    stages_fired = [0] * stage_cap  # rounds that drafted stage s+1
    start = num_input

    def draft_rows(rows_ids, n_new_ctx, n_pred, carrier=None, carrier_pos=None, rows_pos=None):
        """One draft forward over ``rows_ids``; returns the log-probs and
        hidden states of its last ``n_pred`` rows. Context K/V comes from the
        draft cache; only the main block brings the newly accepted context."""
        noise = target.model.embed_tokens(rows_ids)
        th = target_hidden if n_new_ctx else target_hidden[:, :0, :]
        if rows_pos is not None:
            pos = rows_pos
        else:
            pos = position_ids[:, draft_cache.get_seq_length() : start + rows_ids.shape[1]]
        kw = {}
        if carrier is not None:
            pos = torch.cat([carrier_pos, pos], dim=1)
            kw["extra_kv_hidden"] = carrier
        bh = model(
            target_hidden=th,
            noise_embedding=noise,
            position_ids=pos,
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
            **kw,
        )
        draft_cache.crop(start)
        lp = torch.log_softmax(target.lm_head(bh[0, -n_pred:, :]).float(), dim=-1)
        return lp, bh[:, -n_pred:, :]

    def draft_stage(acc_toks: list[int], acc_hid):
        """Additional drafting stage: rows [acc_toks[-1], MASK x (B-1)] at positions
        start+L.. (the main block's B-row layout), carriers = hidden states of the
        drafted tokens at start+1..start+L-1."""
        L = len(acc_toks)
        rows = torch.cat([
            torch.tensor([[acc_toks[-1]]], dtype=torch.long, device=device),
            torch.full((1, block - 1), mask_token_id, dtype=torch.long, device=device),
        ], dim=1)
        return draft_rows(
            rows, n_new_ctx=0, n_pred=block - 1,
            carrier=acc_hid[:, : L - 1, :],
            carrier_pos=position_ids[:, start + 1 : start + L],
            rows_pos=position_ids[:, start + L : start + L + block],
        )

    while start < max_length:
        # ---- 1-2. main block + best-first tree ----
        block_ids = output_ids[:, start : start + block].clone()
        block_ids[0, 1:] = mask_token_id
        lp, main_hidden = draft_rows(block_ids, n_new_ctx=1, n_pred=block - 1)
        if draft_prefill and return_stats:
            draft_prefill = False
            decode_start = _cuda_time()
        K = min(topk_k or tree_budget, lp.shape[-1])
        vals, idx = torch.topk(lp, K, dim=-1)
        vals_c, idx_c = vals.cpu(), idx.cpu()
        toks, par, dep = build_bestfirst(vals_c, idx_c, tree_budget)

        # ---- 3. DLoop stages under the end of the top-1 chain ----
        prev_vals_c, prev_top1 = vals_c, idx_c[:, 0].tolist()
        acc_toks, acc_hid = list(prev_top1), main_hidden  # drafted top-1 tokens / their hiddens
        root, stage = 0, 0
        while stage < stage_cap and float(prev_vals_c[:, 0].sum()) >= loop_gate:
            attach = _top1_leaf(toks, par, prev_top1, root=root)
            if attach is None:
                break
            stages_fired[stage] += 1
            lp_s, s_hidden = draft_stage(acc_toks, acc_hid)
            acc_hid = torch.cat([acc_hid, s_hidden], dim=1)
            K2 = min(topk_k or stage_tree_budget, lp_s.shape[-1])
            vals_s, idx_s = torch.topk(lp_s, K2, dim=-1)
            vals_s_c, idx_s_c = vals_s.cpu(), idx_s.cpu()
            t_s, p_s, d_s = build_bestfirst(vals_s_c, idx_s_c, stage_tree_budget)
            base = len(toks) - 1  # stage node j (> 0) lands at row base + j
            for j in range(1, len(t_s)):
                toks.append(t_s[j])
                par.append(attach if p_s[j] == 0 else base + p_s[j])
                dep.append((block - 1) * (stage + 1) + d_s[j])
            root = attach
            prev_vals_c, prev_top1 = vals_s_c, idx_s_c[:, 0].tolist()
            acc_toks.extend(prev_top1)
            stage += 1

        # ---- 4. verify anchor + tree rows in one target forward ----
        row_toks = torch.tensor(
            [int(output_ids[0, start])] + [int(t) for t in toks[1:]], device=device
        )[None]
        row_pos = torch.tensor([start + d for d in dep], device=device)[None]
        T = row_toks.shape[1]
        out = target(
            row_toks,
            position_ids=row_pos,
            attention_mask=build_tree_attention(par, start, target.dtype, device),
            past_key_values=tgt_cache,
            use_cache=True,
            cache_position=torch.arange(start, start + T, device=device),
            output_hidden_states=True,
        )
        posterior = out.logits[0].argmax(dim=-1)

        # ---- 5. accept the longest matching root path + bonus ----
        ch = _children_index(par)
        post_c = posterior.tolist()
        path = [0]
        while True:
            want = post_c[path[-1]]
            nxt = next((c for c in ch.get(path[-1], []) if toks[c] == want), None)
            if nxt is None:
                break
            path.append(nxt)
        k = len(path) - 1
        path_t = torch.tensor(path, device=device)
        output_ids[0, start + 1 : start + k + 1] = row_toks[0, path_t[1:]]
        output_ids[0, start + k + 1] = posterior[path[-1]]
        acceptance_lengths.append(k + 1)

        # ---- 6. compact the target cache and context features ----
        _compact_cache(tgt_cache, start, path_t)
        target_hidden = extract_context_feature(out.hidden_states, model.target_layer_ids)[:, path_t, :]
        start += k + 1

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
        time_per_output_token=(_cuda_time() - decode_start) / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
        stages_fired=stages_fired,
    )
