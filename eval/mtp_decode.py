# coding=utf-8
# DLoop
# Copyright (c) 2026-present NAVER Corp.
#
# mtp_target_ar_generate is derived from DFlash (https://github.com/z-lab/dflash)
# MIT License

"""MTP host decoding (+ DLoop) — Qwen3.5 MTP head / Gemma 4 assistant.

The hosts' deployment is a chain: k drafter steps (one forward each, the
drafter's own hidden feeding the next step) -> ONE target verify over
[anchor + drafted] -> longest matching prefix + bonus. DLoop adds gated
extension stages that ride the same verify:

  1. block: k draft steps from the anchor; gate material = sum of the top-1
     log-probs of the block;
  2. while the gate holds (material >= loop_gate, stage < stage_cap): stage_len
     more draft steps (material = the sum over that segment) — the AR drafter is
     simply run further, no layout change;
  3. verify once (causal target forward over anchor + all drafted rows);
  4. accept prefix + bonus, rewind the target state to the accepted rows (KV
     crop; Qwen3.5 recurrent states via ``LinearStateRollback``), drop the
     drafter's draft rows; the accepted rows' target hidden feed the next round.

A stop token in a segment closes the gate. loop_gate > 0 = the host as released.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

from .mtp_model import make_target_cache


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def mtp_generate(
    drafter: nn.Module,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    k: int = 3,
    loop_gate: float = 1.0,
    stage_len: Optional[int] = None,
    stage_cap: int = 16,
    return_stats: bool = False,
):
    """k: drafted tokens per round (the host block). stage_len: draft steps per
    loop stage (default k). stage_cap: max loop stages per round."""
    elen = k if stage_len is None else int(stage_len)
    stop = set(stop_token_ids or [])
    device = target.device
    stop_t = torch.tensor(sorted(stop), device=device) if stop else None
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens
    pad = k + elen * max(stage_cap, 1) + 2
    rollback = getattr(drafter, "rollback", None)
    tkw = {"return_shared_kv_states": True} if drafter.needs_shared_kv else {}

    output_ids = torch.zeros((1, max_length + pad), dtype=torch.long, device=device)
    output_ids[:, :num_input] = input_ids
    position_ids = torch.arange(output_ids.shape[1], device=device)[None]
    tgt_cache = make_target_cache(target, rollback=rollback is not None)

    prefill_start = _cuda_time() if return_stats else None
    out = target(
        input_ids, position_ids=position_ids[:, :num_input], past_key_values=tgt_cache,
        use_cache=True, logits_to_keep=1, output_hidden_states=True, **tkw,
    )
    if rollback is not None:
        tgt_cache.crop(0)  # recorded conv history -> kernel-size window
    output_ids[0, num_input] = out.logits[0, -1].argmax(dim=-1)
    start = num_input
    # drafter commit rows: tokens x_1..x_start with their producer hidden H_0..H_{start-1}
    commit_tokens = output_ids[:, 1 : start + 1]
    commit_hidden = out.hidden_states[-1]
    drafter.reset()
    drafter.observe_target(out, start)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    first_round = True
    acceptance_lengths: list[int] = []
    stages_fired = [0] * stage_cap

    while start < max_length:
        # ---- 1. block: k draft steps ----
        toks: list[torch.Tensor] = []
        logits = drafter.begin_round(commit_tokens, commit_hidden, start)
        if first_round and return_stats and drafter.has_prefill:
            decode_start = _cuda_time()  # drafter prefill excluded (dflash_generate convention)
        first_round = False

        def _draft(n: int, logits: torch.Tensor) -> tuple[torch.Tensor, bool]:
            """n draft steps from the pending row's ``logits``; returns (sum of the
            segment's top-1 log-probs, the segment contains a stop token)."""
            mat = torch.zeros((), device=device)
            first = len(toks)
            for _ in range(n):
                val, tok = torch.log_softmax(logits[0, -1].float(), dim=-1).max(dim=-1)
                toks.append(tok)
                mat = mat + val
                if len(toks) < n_total:
                    logits = drafter.step(tok.view(1, 1), start, len(toks))
            hit = stop_t is not None and bool(torch.isin(torch.stack(toks[first:]), stop_t).any())
            return mat, hit

        n_total = k
        mat, hit_stop = _draft(k, logits)

        # ---- 2. gated loop stages (DLoop) ----
        stage = 0
        while stage < stage_cap and not hit_stop and float(mat) >= loop_gate:
            stages_fired[stage] += 1
            n_total += elen
            logits = drafter.step(toks[-1].view(1, 1), start, len(toks))
            mat, hit_stop = _draft(elen, logits)
            stage += 1

        # ---- 3. verify: anchor + drafted rows in ONE causal target forward ----
        drafted = torch.stack(toks)
        row_toks = torch.cat([output_ids[:, start : start + 1], drafted[None]], dim=1)
        T = row_toks.shape[1]
        if rollback is not None:
            rollback.snapshot(tgt_cache)
        out = target(
            row_toks, position_ids=position_ids[:, start : start + T],
            past_key_values=tgt_cache, use_cache=True, output_hidden_states=True, **tkw,
        )
        posterior = out.logits[0].argmax(dim=-1)
        acc = int((drafted == posterior[: drafted.numel()]).cumprod(dim=0).sum())

        # ---- 4. accept + bonus, rewind target/drafter state, carry hidden ----
        output_ids[0, start + 1 : start + acc + 1] = drafted[:acc]
        output_ids[0, start + acc + 1] = posterior[acc]
        acceptance_lengths.append(acc + 1)
        keep = acc + 1
        if rollback is not None:
            rollback.rollback(tgt_cache, keep, T)
        elif keep < T:
            tgt_cache.crop(-(T - keep))
        drafter.end_round()
        new_start = start + keep
        drafter.observe_target(out, new_start)
        commit_tokens = output_ids[:, start + 1 : new_start + 1]
        commit_hidden = out.hidden_states[-1][:, :keep]
        start = new_start

        if stop_t is not None and bool(torch.isin(output_ids[0, num_input : start + 1], stop_t).any()):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_t is not None:
        hits = torch.isin(output_ids[0, num_input:], stop_t).nonzero(as_tuple=True)[0]
        if hits.numel() > 0:
            output_ids = output_ids[:, : num_input + hits[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        stages_fired=stages_fired,
    )


@torch.inference_mode()
def mtp_target_ar_generate(
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    return_stats: bool = False,
):
    """Autoregressive baseline on an MTP-host target (one target forward per
    token). ``dflash_generate`` with block size 1, except that hybrid
    (linear-attention) targets get their config-aware cache and are never
    cropped (an AR step has no rows to drop)."""
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    # The buffer filler must not be a stop token (the stop check scans the buffer).
    filler = next(i for i in range(8) if i not in set(stop_token_ids or []))
    output_ids = torch.full((1, max_length + 1), filler, dtype=torch.long, device=target.device)
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    cache = make_target_cache(target)

    prefill_start = _cuda_time() if return_stats else None
    output = target(input_ids, position_ids=position_ids[:, :num_input_tokens],
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = output.logits.argmax(dim=-1)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    start = num_input_tokens
    while start < max_length:
        block_output_ids = output_ids[:, start : start + 1].clone()
        output = target(block_output_ids, position_ids=position_ids[:, start : start + 1],
                        past_key_values=cache, use_cache=True)
        posterior = output.logits.argmax(dim=-1)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        acceptance_lengths.append(acceptance_length + 1)
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_t = torch.tensor(stop_token_ids, device=output_ids.device)
        hits = torch.isin(output_ids[0][num_input_tokens:], stop_t).nonzero(as_tuple=True)[0]
        if hits.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + hits[0] + 1]

    if not return_stats:
        return output_ids
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=(_cuda_time() - decode_start) / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
    )
