# coding=utf-8
# DLoop
# Copyright (c) 2026-present NAVER Corp.

"""Linear decoding for drafters with a sequential correction head.

DSpark (Markov bias) and Domino (GRU bias) draft a block in one backbone
forward and sample its rows with their head. With ``loop_gate > 0`` this is
the hosts' published loop (as-is). With a negative gate, DLoop adds drafting
stages while the summed top-1 log-prob of the block just drafted (head-
corrected logits) is >= ``loop_gate``. A stage drafts ``[c_L, MASK x (B-1)]``
at the true positions of the next block; the earlier drafted tokens c_1..c_{L-1}
enter the K/V stream as the hidden states of the rows that produced them
(carriers, fc bypassed). The head continues across the boundary (Domino: the
GRU state is carried). All drafted tokens are verified by one causal target
forward.
"""

import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, DynamicCache

from specforge.modeling.draft.dflash import extract_context_feature

DEFAULT_STAGE_CAP = 16

# The released DSpark config keeps the head fields at the top level; the
# SpecForge classes read them from ``dflash_config``.
_DSPARK_TOP_LEVEL_KEYS = (
    "target_layer_ids",
    "mask_token_id",
    "num_anchors",
    "markov_rank",
    "markov_head_type",
    "enable_confidence_head",
    "confidence_head_with_markov",
)


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def load_drafter(draft_path: str, target: nn.Module, device) -> nn.Module:
    """DSpark or Domino drafter (released or DLoop-trained checkpoint)."""
    from specforge.modeling.draft.domino import DominoDraftModel
    from specforge.modeling.draft.dspark import DSparkDraftModel

    config = AutoConfig.from_pretrained(draft_path)
    archs = getattr(config, "architectures", None) or []
    dflash_config = dict(getattr(config, "dflash_config", None) or {})
    if "Qwen3DSparkModel" in archs or "DSparkDraftModel" in archs or (
        dflash_config.get("projector_type") == "dspark"
    ):
        cls = DSparkDraftModel
        dflash_config.setdefault("projector_type", "dspark")
        dflash_config.setdefault("shift_label", True)  # DSpark labels are next-token
        for key in _DSPARK_TOP_LEVEL_KEYS:
            if hasattr(config, key):
                dflash_config.setdefault(key, getattr(config, key))
        config.dflash_config = dflash_config
        rope = getattr(config, "rope_parameters", None)
        if isinstance(rope, dict) and "rope_theta" in rope:
            config.rope_theta = rope["rope_theta"]  # transformers 4.x reads rope_theta
    elif "DominoDraftModel" in archs or dflash_config.get("projector_type") == "domino":
        cls = DominoDraftModel
    else:
        raise ValueError(f"{draft_path}: not a DSpark or Domino checkpoint")
    model, info = cls.from_pretrained(
        draft_path, config=config, attn_implementation="sdpa",
        dtype=torch.bfloat16, output_loading_info=True,
    )
    # The released DSpark checkpoint also stores frozen copies of the target's
    # embedding and LM head; the drafter uses the target's own at run time.
    unexpected = set(info["unexpected_keys"]) - {"embed_tokens.weight", "lm_head.weight"}
    assert not info["missing_keys"] and not unexpected, (info["missing_keys"], unexpected)
    return model.to(device).eval()


def host_kind(model: nn.Module) -> str:
    if getattr(model, "markov_head", None) is not None:
        return "dspark"
    if getattr(model, "prefix_gru", None) is not None:
        return "domino"
    raise ValueError("expected a DSpark (Markov head) or Domino (GRU head) drafter")


def domino_sample_rows(model, target, base_logits, hidden, prev_ids, gru_hidden, pure_prefix):
    """Domino's greedy row loop (its ``spec_generate``) over the rows of one
    block: returns (tokens [1, n], corrected logits [1, n, V], GRU state).

    With ``gru_hidden=None`` the first ``pure_prefix`` rows use the base logits
    and the state warms up on [prev_ids[-1], c_1..c_pure] as in the official
    loop; a carried state must already cover every committed token."""
    n = base_logits.shape[1]
    toks, logits = [], []
    for i in range(n):
        if i < pure_prefix and gru_hidden is None:
            step_logits = base_logits[:, i, :]
        else:
            if gru_hidden is None:
                warm_ids = torch.cat([prev_ids[:, -1:]] + [t.unsqueeze(1) for t in toks], dim=1)
                _, gru_hidden = model.prefix_gru(target.model.embed_tokens(warm_ids))
            zs = torch.cat([hidden[:, i : i + 1, :], gru_hidden.transpose(0, 1)], dim=-1)
            step_logits = (base_logits[:, i : i + 1, :] + model.embed_proj(zs))[:, 0, :]
        tok = step_logits.argmax(dim=-1)
        toks.append(tok)
        logits.append(step_logits.unsqueeze(1))
        if gru_hidden is not None and i + 1 < n:
            _, gru_hidden = model.prefix_gru(target.model.embed_tokens(tok.unsqueeze(1)), gru_hidden)
    return torch.stack(toks, dim=1), torch.cat(logits, dim=1), gru_hidden


def draft_rows(model, target, cache, target_hidden, rows, positions, carrier=None, carrier_pos=None):
    """One drafter forward over ``rows`` at ``positions`` (new context K/V in
    ``target_hidden``, carriers between context and rows). Returns (base
    logits, hidden) of every row; the rows' K/V are dropped from ``cache``."""
    kw = {}
    if carrier is not None:
        positions = torch.cat([carrier_pos, positions], dim=1)
        kw["extra_kv_hidden"] = carrier
    prefix = cache.get_seq_length()
    hidden = model(
        target_hidden=target_hidden,
        noise_embedding=target.model.embed_tokens(rows),
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
        is_causal=False,
        **kw,
    )[:, -rows.shape[1] :, :]
    cache.crop(prefix + target_hidden.shape[1])
    return target.lm_head(hidden), hidden


def _segment_logp(corrected, toks) -> float:
    lp = torch.log_softmax(corrected.float(), dim=-1)
    return float(lp.gather(-1, toks.unsqueeze(-1)).sum())


@torch.inference_mode()
def head_generate(
    model: nn.Module,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    loop_gate: float = 1.0,
    stage_cap: int = DEFAULT_STAGE_CAP,
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
    pad = block + n_stages * block + 2

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

    def sample(base_logits, hidden, prev_ids, gru_state, first):
        if kind == "dspark":
            toks, corrected = model.markov_head.sample_block_tokens(
                base_logits, first_prev_token_ids=prev_ids[:, -1], hidden_states=hidden, temperature=0.0,
            )
            return toks, corrected, None
        if first:
            return domino_sample_rows(
                model, target, base_logits, hidden, prev_ids, None, model.pure_draft_prefix_len
            )
        # The block's last token was never fed to the GRU; feed it, then carry.
        _, gru_state = model.prefix_gru(target.model.embed_tokens(prev_ids[:, -1:]), gru_state)
        return domino_sample_rows(model, target, base_logits, hidden, prev_ids, gru_state, 0)

    while start < max_length:
        # Main block: [anchor, MASK x (B-1)]; row j predicts position start+j+1.
        anchor = output_ids[:, start : start + 1]
        base_logits, hidden = draft_rows(
            model, target, draft_cache, target_hidden, torch.cat([anchor, masks], dim=1),
            position_ids[:, draft_cache.get_seq_length() : start + block],
        )
        if draft_prefill and return_stats:
            draft_prefill = False
            decode_start = _cuda_time()
        toks, corrected, gru_state = sample(base_logits, hidden, anchor, None, True)
        prev_ids = torch.cat([anchor, toks], dim=1)

        # DLoop stages, gated on the block just drafted.
        gate = _segment_logp(corrected, toks) if n_stages else None
        carriers = hidden  # row j's hidden produced the token at slot start+j+1
        stage = 0
        while stage < n_stages and start + prev_ids.shape[1] + block < max_length + pad and gate >= loop_gate:
            stages_fired[stage] += 1
            L = prev_ids.shape[1] - 1
            base_logits, hidden = draft_rows(
                model, target, draft_cache, target_hidden[:, :0, :],
                torch.cat([prev_ids[:, -1:], masks], dim=1), position_ids[:, start + L : start + L + block],
                carrier=carriers[:, : L - 1, :], carrier_pos=position_ids[:, start + 1 : start + L],
            )
            carriers = torch.cat([carriers, hidden], dim=1)
            toks, corrected, gru_state = sample(base_logits, hidden, prev_ids, gru_state, False)
            gate = _segment_logp(corrected, toks)
            prev_ids = torch.cat([prev_ids, toks], dim=1)
            stage += 1

        # One causal verify over [anchor + drafted].
        out = target(
            prev_ids,
            position_ids=position_ids[:, start : start + prev_ids.shape[1]],
            past_key_values=tgt_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = out.logits[0].argmax(dim=-1)
        acc = int((prev_ids[0, 1:] == posterior[:-1]).cumprod(dim=0).sum())
        output_ids[0, start + 1 : start + acc + 1] = prev_ids[0, 1 : acc + 1]
        output_ids[0, start + acc + 1] = posterior[acc]
        acceptance_lengths.append(acc + 1)
        tgt_cache.crop(start + acc + 1)
        target_hidden = extract_context_feature(out.hidden_states, model.target_layer_ids)[:, : acc + 1, :]
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
