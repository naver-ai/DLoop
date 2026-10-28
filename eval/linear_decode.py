# coding=utf-8
# Derived from DFlash (https://github.com/z-lab/dflash)
# MIT License
# Modified for DLoop

"""Linear block decoding for DFlash drafters, + DLoop.

`dflash_generate` from the public DFlash code (https://github.com/z-lab/dflash,
`dflash/model.py`). A positive ``loop_gate`` (the default) never
opens: the drafter as released. With ``block_size=1`` it never calls the
drafter and is the autoregressive baseline.

DLoop (``loop_gate <= 0``): while fewer than ``stage_cap`` stages were drafted
and the summed top-1 log-prob of the segment just drafted is >= ``loop_gate``,
draft one more stage in the layout DLoop trains on, [last drafted token,
MASK x (B-1)] with the earlier drafted tokens' hidden states as carriers; one
target forward then verifies the anchor and every drafted token.
"""

import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache

from specforge.modeling.draft.dflash import extract_context_feature

# Default stage cap; bounds the padding and the verify width.
DEFAULT_STAGE_CAP = 16


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _top1_logp(logits) -> float:
    """Summed top-1 log-prob of a drafted segment (the DLoop gate score)."""
    return float(torch.log_softmax(logits.float(), dim=-1).max(dim=-1).values.sum())


@torch.inference_mode()
def dflash_generate(
    model: Optional[nn.Module],
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    loop_gate: float = 1.0,
    stage_cap: int = DEFAULT_STAGE_CAP,
    return_stats: bool = False,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id
    n_stages = stage_cap if loop_gate <= 0 and block_size > 1 else 0

    # a round commits at most the main block + every stage's B-1 tokens + the bonus
    output_ids = torch.full(
        (1, max_length + block_size + (block_size - 1) * n_stages), mask_token_id,
        dtype=torch.long, device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = output.logits.argmax(dim=-1)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    stages_fired = [0] * n_stages  # rounds that drafted stage s+1
    start = num_input_tokens
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_hidden = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, 1 - block_size :, :]
            draft_logits = target.lm_head(draft_hidden)
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = draft_logits.argmax(dim=-1)
            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

            # DLoop: stage s drafts [last drafted token, MASK x (B-1)] after the L
            # tokens drafted so far, with their hidden states (all but the
            # anchor's) as carriers; no new target context
            stage = 0
            while stage < n_stages and _top1_logp(draft_logits) >= loop_gate:
                stages_fired[stage] += 1
                L = block_output_ids.shape[1] - 1
                rows = torch.cat([
                    block_output_ids[:, -1:],
                    torch.full((1, block_size - 1), mask_token_id, dtype=torch.long, device=target.device),
                ], dim=1)
                stage_hidden = model(
                    target_hidden=target_hidden[:, :0, :],
                    noise_embedding=target.model.embed_tokens(rows),
                    position_ids=torch.cat([position_ids[:, start + 1 : start + L],
                                            position_ids[:, start + L : start + L + block_size]], dim=1),
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                    extra_kv_hidden=draft_hidden[:, : L - 1, :],
                )[:, 1 - block_size :, :]
                past_key_values_draft.crop(start)
                draft_logits = target.lm_head(stage_hidden)
                draft_hidden = torch.cat([draft_hidden, stage_hidden], dim=1)
                block_output_ids = torch.cat([block_output_ids, draft_logits.argmax(dim=-1)], dim=1)
                stage += 1
            block_position_ids = position_ids[:, start : start + block_output_ids.shape[1]]

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )

        posterior = output.logits.argmax(dim=-1)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
        stages_fired=stages_fired,
    )
