# coding=utf-8
# DLoop
# Copyright (c) 2026-present NAVER Corp.

"""DARTree (arXiv 2608.13524) for DSpark / Domino drafters, + DLoop.

One block forward gives the per-position base logits. Depth by depth, the
top-``topk`` base candidates of every active branch are scored with the host's
correction head (DSpark: Markov bias of the parent token; Domino: the branch's
GRU state; depth 1 uses the base logits), the ``tree_width`` best extensions by
cumulative corrected log-prob stay active, and the global top ``tree_budget``
nodes by cum + beta * depth (ancestor-closed for beta <= 0) form the tree that
one tree-masked target forward verifies.

DLoop (``loop_gate <= 0``): when the tree keeps a full-depth node, stages attach
to the best one. The gate reads the cumulative corrected log-prob of that path,
then the summed top-1 log-prob of the stage just drafted; a stage drafts a
block in the DLoop layout of head_decode.py and hangs as a chain under the leaf
(the paper). With ``stage_tree_budget`` > 0 the stage block instead gets its own
DARTree of that many nodes under the leaf; the next stage attaches to its best
full-depth node, and the gate reads that node's cumulative corrected log-prob.
"""

from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache

from specforge.modeling.draft.dflash import extract_context_feature

from .head_decode import (
    DEFAULT_STAGE_CAP,
    _cuda_time,
    _segment_logp,
    domino_sample_rows,
    draft_rows,
    host_kind,
)
from .tree_decode import _compact_cache, build_tree_attention


@torch.no_grad()
def build_dartree(base_logits, hidden, anchor_tok, model, target, budget, width, beta, topk=64, gru_init=None):
    """DARTree over one block (base_logits [B, V], hidden [1, B, D]). Returns
    (toks, par, depth, cum) with parents first, ``par`` indexing the selection
    (-1 = anchor) and ``cum`` the cumulative corrected log-prob. ``gru_init``
    (Domino, loop stages): the GRU state after the path so far, so that depth 1
    is corrected and the branches continue from it, as in the chain stage."""
    kind = host_kind(model)
    base = base_logits.float()
    k = min(topk, base.shape[-1])
    device = base.device
    embed = target.model.embed_tokens
    n_tok, n_par, n_cum, n_dep = [], [], [], []
    active = [-1]
    prev = torch.tensor([anchor_tok], device=device)
    cum = torch.zeros(1, device=device)
    for d in range(base.shape[0]):
        cand = torch.topk(base[d], k).indices
        if kind == "dspark":
            logq = torch.log_softmax(
                base[d][None] + model.markov_head.compute_step_bias(prev, None).float(), dim=-1
            )[:, cand]
        elif d == 0 and gru_init is not None:
            zs = torch.cat([hidden[:, 0:1, :], gru_init.transpose(0, 1)], dim=-1)
            logq = torch.log_softmax(base[0][None] + model.embed_proj(zs)[:, 0, :].float(), dim=-1)[:, cand]
        elif d == 0:
            logq = torch.log_softmax(base[0], dim=-1)[cand][None]
        else:
            z = hidden[:, d : d + 1, :].expand(len(active), -1, -1)
            bias = model.embed_proj(torch.cat([z, states.transpose(0, 1)], dim=-1))[:, 0, :].float()
            logq = torch.log_softmax(base[d][None] + bias, dim=-1)[:, cand]
        top = torch.topk((cum[:, None] + logq).flatten(), min(width, logq.numel()))
        rows = torch.div(top.indices, k, rounding_mode="floor")
        prev = cand[top.indices % k]
        n_par.extend(active[r] for r in rows.tolist())
        active = list(range(len(n_tok), len(n_tok) + prev.shape[0]))
        n_tok.extend(prev.tolist())
        n_cum.extend(top.values.tolist())
        n_dep.extend([d + 1] * prev.shape[0])
        cum = top.values
        if kind == "domino":
            if d == 0 and gru_init is not None:
                _, states = model.prefix_gru(embed(prev[:, None]), gru_init.expand(-1, prev.shape[0], -1).contiguous())
            elif d == 0:
                warm = torch.stack([torch.full_like(prev, anchor_tok), prev], dim=1)
                _, states = model.prefix_gru(embed(warm))  # [1, W, H]
            else:
                _, states = model.prefix_gru(embed(prev[:, None]), states[:, rows, :])

    # Global top-B by (score desc, depth asc), placed parents first.
    order = sorted(range(len(n_tok)), key=lambda i: (-(n_cum[i] + beta * n_dep[i]), n_dep[i]))
    sel = sorted(order[:budget], key=lambda i: (n_dep[i], i))
    pos = {gi: j for j, gi in enumerate(sel)}
    assert all(n_par[gi] == -1 or n_par[gi] in pos for gi in sel), "ancestor closure"
    toks = [n_tok[gi] for gi in sel]
    par = [-1 if n_par[gi] == -1 else pos[n_par[gi]] for gi in sel]
    return toks, par, [n_dep[gi] for gi in sel], [n_cum[gi] for gi in sel]


@torch.inference_mode()
def dartree_generate(
    model: nn.Module,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    tree_budget: int = 64,
    tree_width: int = 12,
    tree_beta: float = -0.2,
    loop_gate: float = 1.0,
    stage_cap: int = DEFAULT_STAGE_CAP,
    stage_tree_budget: int = 0,
    return_stats: bool = False,
):
    kind = host_kind(model)
    assert model.shift_label, "DSpark/Domino rows predict the next token (shift_label)"
    block = model.block_size
    mask_token_id = model.mask_token_id
    device = target.device
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens
    n_stages = stage_cap if loop_gate <= 0 else 0
    stage_rows = max(block, stage_tree_budget)
    pad = block + tree_budget + n_stages * stage_rows + 2

    output_ids = torch.full((1, max_length + pad), mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device)[None]
    tgt_cache, draft_cache = DynamicCache(), DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    out = target(
        input_ids,
        position_ids=position_ids[:, :num_input],
        past_key_values=tgt_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    target_hidden = extract_context_feature(out.hidden_states, model.target_layer_ids)
    output_ids[:, :num_input] = input_ids
    output_ids[0, num_input] = out.logits[0, -1].argmax(dim=-1)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    draft_prefill = True
    acceptance_lengths, stages_fired = [], [0] * stage_cap
    start = num_input
    masks = torch.full((1, block - 1), mask_token_id, dtype=torch.long, device=device)

    while start < max_length:
        anchor = output_ids[:, start : start + 1]
        base_logits, hidden = draft_rows(
            model, target, draft_cache, target_hidden, torch.cat([anchor, masks], dim=1),
            position_ids[:, draft_cache.get_seq_length() : start + block],
        )
        if draft_prefill and return_stats:
            draft_prefill = False
            decode_start = _cuda_time()
        toks, par, dep, cums = build_dartree(
            base_logits[0], hidden, int(anchor[0, 0]), model, target, tree_budget, tree_width, tree_beta
        )

        # DLoop stages under the best full-depth leaf (none kept: no loop).
        full = [i for i in range(len(toks)) if dep[i] == block] if n_stages else []
        if full:
            leaf = max(full, key=lambda i: cums[i])
            path, cur = [], leaf
            while cur != -1:
                path.append(cur)
                cur = par[cur]
            prev_ids = torch.cat([anchor, torch.tensor([[toks[i] for i in path[::-1]]], device=device)], dim=1)
            gate = cums[leaf]
            carriers = hidden  # main-block rows produced the path's slots
            stage = 0
            while stage < n_stages and start + 1 + len(toks) + stage_rows < max_length + pad and gate >= loop_gate:
                stages_fired[stage] += 1
                L = prev_ids.shape[1] - 1
                base_ext, hid_ext = draft_rows(
                    model, target, draft_cache, target_hidden[:, :0, :],
                    torch.cat([prev_ids[:, -1:], masks], dim=1), position_ids[:, start + L : start + L + block],
                    carrier=carriers[:, : L - 1, :], carrier_pos=position_ids[:, start + 1 : start + L],
                )
                carriers = torch.cat([carriers, hid_ext], dim=1)
                if stage_tree_budget:  # a DARTree under the leaf instead of a chain
                    gru0 = model.prefix_gru(target.model.embed_tokens(prev_ids))[1] if kind == "domino" else None
                    t_e, p_e, d_e, c_e = build_dartree(
                        base_ext[0], hid_ext, int(prev_ids[0, -1]), model, target,
                        stage_tree_budget, tree_width, tree_beta, gru_init=gru0)
                    base_i = len(toks)
                    for j in range(len(t_e)):
                        toks.append(t_e[j])
                        par.append(leaf if p_e[j] == -1 else base_i + p_e[j])
                        dep.append(dep[leaf] + d_e[j])
                    full_e = [j for j in range(len(t_e)) if d_e[j] == block]
                    stage += 1
                    if not full_e:  # no full-depth path to attach the next stage to
                        break
                    jbest = max(full_e, key=lambda j: c_e[j])
                    path_e, cur = [], jbest
                    while cur != -1:
                        path_e.append(cur)
                        cur = p_e[cur]
                    leaf, gate = base_i + jbest, c_e[jbest]
                    prev_ids = torch.cat([prev_ids, torch.tensor([[t_e[j] for j in path_e[::-1]]], device=device)], dim=1)
                    continue
                if kind == "dspark":
                    etoks, corrected = model.markov_head.sample_block_tokens(
                        base_ext, first_prev_token_ids=prev_ids[:, -1], hidden_states=hid_ext, temperature=0.0,
                    )
                else:
                    # The GRU state is a function of the consumed sequence:
                    # rebuild it from every committed token of the path.
                    _, gru_state = model.prefix_gru(target.model.embed_tokens(prev_ids))
                    etoks, corrected, _ = domino_sample_rows(
                        model, target, base_ext, hid_ext, prev_ids, gru_state, 0
                    )
                for j in range(block):
                    toks.append(int(etoks[0, j]))
                    par.append(leaf if j == 0 else len(toks) - 2)
                    dep.append(dep[leaf] + j + 1)
                leaf = len(toks) - 1
                gate = _segment_logp(corrected, etoks)
                prev_ids = torch.cat([prev_ids, etoks], dim=1)
                stage += 1

        # One tree-masked verify; accept the longest root-to-leaf path.
        row_ids = torch.cat([anchor, torch.tensor([toks], dtype=torch.long, device=device)], dim=1)
        par_rows = [-1] + [0 if p == -1 else p + 1 for p in par]
        out = target(
            row_ids,
            position_ids=torch.tensor([[start] + [start + d for d in dep]], device=device),
            attention_mask=build_tree_attention(par_rows, start, target.dtype, device),
            past_key_values=tgt_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = out.logits[0].argmax(dim=-1)
        children: dict[int, list[int]] = {}
        for j, p in enumerate(par_rows):
            if p >= 0:
                children.setdefault(p, []).append(j)
        post_l = posterior.tolist()
        accepted, cur = [], 0
        while True:
            nxt = next((c for c in children.get(cur, ()) if toks[c - 1] == post_l[cur]), None)
            if nxt is None:
                break
            accepted.append(nxt)
            cur = nxt
        acc = len(accepted)
        if acc:
            output_ids[0, start + 1 : start + acc + 1] = row_ids[0, accepted]
        output_ids[0, start + acc + 1] = posterior[cur]
        acceptance_lengths.append(acc + 1)

        keep = torch.tensor([0] + accepted, device=device)
        _compact_cache(tgt_cache, start, keep)
        target_hidden = extract_context_feature(out.hidden_states, model.target_layer_ids)[:, keep, :]
        start += acc + 1

        if stop_token_ids is not None and any(
            s in output_ids[0, num_input : start + 1] for s in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
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
