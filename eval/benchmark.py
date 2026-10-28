# coding=utf-8
# Derived from DFlash (https://github.com/z-lab/dflash)
# MIT License
# Modified for DLoop

"""Speculative-decoding benchmark with DLoop (HF Transformers, sdpa).

Hosts, detected from --draft-model: DFlash, DSpark, Domino, EAGLE-3 and the MTP
hosts (Qwen3.5 built-in MTP head = `--draft-model mtp`, Gemma 4 assistant).

1) Measure the autoregressive baseline once per (target, dataset, GPU type):
    torchrun --nproc_per_node=8 -m eval.benchmark --baseline \
        --model Qwen/Qwen3-4B --dataset gsm8k
2) Measure a drafter; the speedup is relative to the cached baseline:
    torchrun --nproc_per_node=8 -m eval.benchmark \
        --model Qwen/Qwen3-4B --draft-model <checkpoint> --dataset gsm8k \
        [--loop-gate g] [host knobs]

--loop-gate > 0 (default) runs the host as released; a negative gate enables the
DLoop loop. Tree frames: --tree-budget N (DFlash DDTree-style tree, DSpark /
Domino DARTree); EAGLE-3 always decodes with its own dynamic tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache"
BASELINE_CACHE_DIR = ROOT / "baseline_cache"
RESULTS_DIR = ROOT / "results"


def _load_livecodebench():
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    allowed = [f"test{i}.jsonl" for i in ["", "2", "3", "4", "5", "6"]]
    local_dir = snapshot_download(
        repo_id="livecodebench/code_generation_lite",
        repo_type="dataset",
        allow_patterns=allowed,
    )
    paths = [f"{local_dir}/{fn}" for fn in allowed]
    return load_dataset("json", data_files={"test": paths})["test"]


def _format_lcb(doc):
    system_prompt = (
        "You are an expert Python programmer. You will be given a question (problem specification) "
        "and will generate a correct Python program that matches the specification and passes all tests. "
        "You will NOT return anything except for the program"
    )
    question_block = f"### Question:\n{doc['question_content']}"
    if doc.get("starter_code"):
        format_message = "### Format: Use the following code structure:"
        code_block = f"```python\n{doc['starter_code']}\n```"
    else:
        format_message = "### Format: Write your code in the following format:"
        code_block = "```python\n# YOUR CODE HERE\n```"
    answer_footer = "### Answer: (use the provided format with backticks)"
    return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"


DATASETS = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
        "max_samples": 128,
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
        "max_samples": 128,
    },
    "aime25": {
        "load_args": ("MathArena/aime_2025",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x),
        "max_samples": 30,
    },
    "alpaca": {
        "load_args": ("tatsu-lab/alpaca",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: (f"{x['instruction']}\n\nInput:\n{x['input']}" if x["input"] else x["instruction"]),
        "max_samples": 128,
    },
    "livecodebench": {
        "load_fn": _load_livecodebench,
        "format": _format_lcb,
        "max_samples": 128,
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x),
        "max_samples": 164,
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda x: x["prompt"],
        "max_samples": 128,
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda x: x["prompt"],
        "multi_turn": True,
        "max_samples": 80,
    },
}


def _prepare_dataset(name: str) -> Path:
    from datasets import load_dataset

    cfg = DATASETS[name]
    CACHE_DIR.mkdir(exist_ok=True)
    out_path = CACHE_DIR / f"{name}.jsonl"
    tmp_path = out_path.with_name(f"{out_path.name}.{os.getpid()}.tmp")
    print(f"[download] {name} ...")
    if "load_fn" in cfg:
        dataset = cfg["load_fn"]()
    else:
        dataset = load_dataset(*cfg["load_args"], **cfg["load_kwargs"])
    with open(tmp_path, "w") as f:
        for row in dataset:
            turns = cfg["format"](row) if cfg.get("multi_turn") else [cfg["format"](row)]
            f.write(json.dumps({"turns": turns}) + "\n")
    os.replace(tmp_path, out_path)
    return out_path


def load_and_process_dataset(name: str) -> list[dict]:
    path = CACHE_DIR / f"{name}.jsonl"
    if not path.exists():
        _prepare_dataset(name)
    with open(path) as f:
        return [json.loads(line) for line in f]


def _limit_dataset(dataset: list[dict], max_samples: int | None) -> list[dict]:
    # Same subset as `Dataset.shuffle(seed=0).select(range(N))` (paper protocol).
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    perm = np.random.default_rng(0).permutation(len(dataset))
    return [dataset[i] for i in perm[:max_samples]]


def _encode_turn(tokenizer, messages: list[dict], device):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer.encode(text, return_tensors="pt").to(device)


# ---- distributed helpers (torchrun data parallel over samples) ----

def _dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def _dist_gather(torch_dist, obj: Any):
    if not torch_dist.is_initialized():
        return [obj]
    objs = [None] * _dist_size() if _dist_rank() == 0 else None
    torch_dist.gather_object(obj, objs, dst=0)
    return objs


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


# ---- drafters (host detected from the checkpoint) ----

HOSTS = ("dflash", "dspark", "domino", "eagle3", "mtp")


def detect_host(draft_path: str) -> str:
    """From the raw config.json, so that it does not depend on the installed
    transformers knowing the model type (Gemma 4 assistants need >= 5.16)."""
    if draft_path == "mtp":  # the Qwen3.5 built-in MTP head
        return "mtp"
    if os.path.isdir(draft_path):
        config_path = os.path.join(draft_path, "config.json")
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(draft_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    archs = config.get("architectures") or []
    projector = (config.get("dflash_config") or {}).get("projector_type")
    if archs == ["Qwen35MTPDraftModel"] or str(config.get("model_type", "")).endswith("_assistant"):
        return "mtp"
    if "LlamaForCausalLMEagle3" in archs:
        return "eagle3"
    if "Qwen3DSparkModel" in archs or "DSparkDraftModel" in archs or projector == "dspark":
        return "dspark"
    if "DominoDraftModel" in archs or projector == "domino":
        return "domino"
    return "dflash"


def _fill_host_defaults(args, host: str, target) -> None:
    """Host-dependent knob defaults (None on the command line = the host's own)."""
    if host == "eagle3":  # the official EAGLE-3 tree: depth 7, 60 nodes, top-k 10
        args.tree_depth = 7 if args.tree_depth is None else args.tree_depth
        args.tree_budget = 60 if args.tree_budget is None else args.tree_budget
        args.topk_k = 10 if args.topk_k is None else args.topk_k
        args.stage_len = 8 if args.stage_len is None else args.stage_len
        args.stage_tree_budget = 48 if args.stage_tree_budget is None else args.stage_tree_budget
    if host == "mtp":  # official k: Qwen3.5 3, Gemma 4 6
        family = str(target.config.model_type)
        args.mtp_k = args.mtp_k or (6 if family.startswith("gemma4") else 3)
        args.stage_len = args.mtp_k if args.stage_len is None else args.stage_len
    if host == "dflash":
        args.stage_tree_budget = 64 if args.stage_tree_budget is None else args.stage_tree_budget
    args.tree_budget = args.tree_budget or 0


def _load_models(args, host: str, device):
    import torch
    from transformers import AutoModelForCausalLM

    # Paper setting: eager / sdpa target everywhere (tree verification needs a
    # 4D mask, which flash-attention does not take).
    if host == "mtp" or (host is None and _is_mtp_target(args.model)):
        from .mtp_model import load_mtp_target

        target = load_mtp_target(args.model, device)
    else:
        target = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation="sdpa", dtype=torch.bfloat16,
        ).to(device).eval()
    if host is None:
        return target, None
    if host == "dflash":
        from specforge.modeling.draft.dflash import DFlashDraftModel

        draft = DFlashDraftModel.from_pretrained(
            args.draft_model, attn_implementation="sdpa", dtype=torch.bfloat16,
        ).to(device).eval()
    elif host in ("dspark", "domino"):
        from .head_decode import load_drafter

        draft = load_drafter(args.draft_model, target, device)
    elif host == "eagle3":
        from .eagle3_decode import load_drafter

        draft = load_drafter(args.draft_model, target, device)
    else:
        from .mtp_model import load_drafter

        draft = load_drafter(args.draft_model, target, device)
    return target, draft


def _is_mtp_target(model_path: str) -> bool:
    from transformers import AutoConfig

    from .mtp_model import is_mtp_model

    return is_mtp_model(AutoConfig.from_pretrained(model_path))


def _stop_ids(tokenizer, target, mtp: bool) -> list[int]:
    if mtp:
        from .mtp_model import stop_token_ids

        return stop_token_ids(tokenizer, target)
    return [tokenizer.eos_token_id]


def _make_generate(args, host: str, draft, target, stop_ids):
    if host == "dflash":
        from .linear_decode import dflash_generate
        from .tree_decode import dflash_tree_generate

        def generate(input_ids, max_new_tokens, gate):
            if not args.tree_budget:
                return dflash_generate(draft, target, input_ids, max_new_tokens, stop_ids,
                                       loop_gate=gate, stage_cap=args.stage_cap, return_stats=True)
            return dflash_tree_generate(
                draft, target, input_ids, max_new_tokens, stop_ids,
                tree_budget=args.tree_budget, stage_tree_budget=args.stage_tree_budget, topk_k=args.topk_k,
                loop_gate=gate, stage_cap=args.stage_cap, return_stats=True,
            )
    elif host in ("dspark", "domino"):
        from .dartree import dartree_generate
        from .head_decode import head_generate

        def generate(input_ids, max_new_tokens, gate):
            if not args.tree_budget:
                return head_generate(draft, target, input_ids, max_new_tokens, stop_ids,
                                     loop_gate=gate, stage_cap=args.stage_cap, return_stats=True)
            return dartree_generate(
                draft, target, input_ids, max_new_tokens, stop_ids,
                tree_budget=args.tree_budget, tree_width=args.tree_width, tree_beta=args.tree_beta,
                loop_gate=gate, stage_cap=args.stage_cap, stage_tree_budget=args.stage_tree_budget or 0,
                return_stats=True,
            )
    elif host == "eagle3":
        from .eagle3_decode import eagle3_generate

        def generate(input_ids, max_new_tokens, gate):
            return eagle3_generate(
                draft, target, input_ids, max_new_tokens, stop_ids,
                tree_depth=args.tree_depth, tree_budget=args.tree_budget, topk_k=args.topk_k,
                loop_gate=gate, stage_cap=args.stage_cap, stage_len=args.stage_len,
                stage_tree_budget=args.stage_tree_budget, return_stats=True,
            )
    else:
        from .mtp_decode import mtp_generate

        def generate(input_ids, max_new_tokens, gate):
            return mtp_generate(
                draft, target, input_ids, max_new_tokens, stop_ids, k=args.mtp_k,
                loop_gate=gate, stage_len=args.stage_len, stage_cap=args.stage_cap, return_stats=True,
            )
    return generate


# ---- result files ----

def _baseline_path(args, gpu_name: str) -> Path:
    key = (f"{_sanitize(args.model)}_{args.dataset}_maxnew{args.max_new_tokens}"
           f"_samples{args.max_samples}_{_sanitize(gpu_name)}")
    return BASELINE_CACHE_DIR / f"{key}.json"


def _results_path(args, host: str, gpu_name: str) -> Path:
    key = (
        f"{args.dataset}_maxnew{args.max_new_tokens}"
        + f"_samples{args.max_samples}"
        + f"_{_sanitize(gpu_name)}"
    )
    loop = args.loop_gate <= 0
    if host == "dflash":
        if args.tree_budget:
            key += f"_tree{args.tree_budget}" + (f"k{args.topk_k}" if args.topk_k else "")
            key += f"_loop{args.stage_tree_budget}g{args.loop_gate}_cap{args.stage_cap}" if loop else ""
        else:
            key += f"_loopg{args.loop_gate}_cap{args.stage_cap}" if loop else ""
    elif host in ("dspark", "domino"):
        if args.tree_budget:
            key += f"_dartree{args.tree_budget}w{args.tree_width}b{args.tree_beta}"
        key += f"_loopg{args.loop_gate}_cap{args.stage_cap}" if loop else ""
        key += f"_st{args.stage_tree_budget}" if loop and args.stage_tree_budget else ""
    elif host == "eagle3":
        key += f"_tree{args.tree_budget}d{args.tree_depth}k{args.topk_k}"
        key += f"_loop{args.stage_len}x{args.stage_tree_budget}g{args.loop_gate}_cap{args.stage_cap}" if loop else ""
    else:
        key += f"_k{args.mtp_k}"
        key += f"_loop{args.stage_len}g{args.loop_gate}_cap{args.stage_cap}" if loop else ""
    draft = Path(args.draft_model)
    return RESULTS_DIR / _sanitize(draft.parent.name or "hub") / f"{_sanitize(draft.name)}__{key}.json"


def _setup(args):
    import torch
    from torch import distributed as torch_dist

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if "RANK" in os.environ:
        torch_dist.init_process_group(backend="nccl", init_method="env://")
    else:
        warnings.warn("RANK not set. Running on a single process.")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return torch_dist, device, torch.cuda.get_device_name(device)


def run_baseline(args) -> None:
    """Autoregressive decoding of the target alone."""
    from transformers import AutoTokenizer

    from .linear_decode import dflash_generate
    from .mtp_decode import mtp_target_ar_generate

    torch_dist, device, gpu_name = _setup(args)
    path = _baseline_path(args, gpu_name)
    mtp = _is_mtp_target(args.model)
    target, _ = _load_models(args, None, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_ids = _stop_ids(tokenizer, target, mtp)
    # The buffer filler must not be a stop token (the stop check scans the buffer).
    filler = next(i for i in range(8) if i not in stop_ids)
    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples)

    tpots = []
    for idx in tqdm(range(_dist_rank(), len(dataset), _dist_size()), disable=_dist_rank() != 0):
        messages = []
        for user_content in dataset[idx]["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_ids = _encode_turn(tokenizer, messages, device)
            if mtp:  # hybrid (linear-attention) caches are never cropped
                r = mtp_target_ar_generate(target, input_ids, args.max_new_tokens, stop_ids, return_stats=True)
            else:
                r = dflash_generate(None, target, input_ids, args.max_new_tokens, stop_ids,
                                    block_size=1, mask_token_id=filler, return_stats=True)
            tpots.append(r.time_per_output_token)
            text = tokenizer.decode(r.output_ids[0, r.num_input_tokens:], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})

    gathered = _dist_gather(torch_dist, tpots)
    if _dist_rank() != 0:
        return
    tpots = list(chain(*gathered))
    BASELINE_CACHE_DIR.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "gpu_name": gpu_name,
                   "max_new_tokens": args.max_new_tokens, "max_samples": args.max_samples,
                   "time_per_output_token": tpots}, f, indent=2)
    print(f"Baseline throughput: {1 / np.mean(tpots):.2f} tok/s ({len(tpots)} turns). Saved {path}")


def run_draft(args) -> None:
    from transformers import AutoTokenizer

    host = detect_host(args.draft_model)
    torch_dist, device, gpu_name = _setup(args)
    baseline_path = _baseline_path(args, gpu_name)
    if not baseline_path.exists():
        raise FileNotFoundError(f"{baseline_path} not found; measure it first with --baseline")
    with open(baseline_path) as f:
        baseline_tpot = float(np.mean(json.load(f)["time_per_output_token"]))

    target, draft = _load_models(args, host, device)
    _fill_host_defaults(args, host, target)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stop_ids = _stop_ids(tokenizer, target, host == "mtp")
    generate = _make_generate(args, host, draft, target, stop_ids)
    dataset = _limit_dataset(load_and_process_dataset(args.dataset), args.max_samples)
    indices = list(range(_dist_rank(), len(dataset), _dist_size()))

    def run_instance(instance, max_new_tokens, gate):
        responses, messages = [], []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            r = generate(_encode_turn(tokenizer, messages, device), max_new_tokens, gate)
            text = tokenizer.decode(r.output_ids[0, r.num_input_tokens:], skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})
            responses.append(r)
        return responses

    # Warm-up (kernels, allocator) with the gate fully open, so that the loop
    # paths are exercised before the measured cells.
    if indices:
        run_instance(dataset[indices[0]], 512, gate=-1e9)

    responses = []
    for idx in tqdm(indices, disable=_dist_rank() != 0):
        responses.extend(run_instance(dataset[idx], args.max_new_tokens, args.loop_gate))
    stats = [{
        "time_per_output_token": r.time_per_output_token,
        "num_output_tokens": r.num_output_tokens,
        "acceptance_lengths": list(r.acceptance_lengths),
        "stages_fired": list(getattr(r, "stages_fired", [])),
        "output_ids": r.output_ids[0, r.num_input_tokens:].tolist(),
    } for r in responses]
    gathered = _dist_gather(torch_dist, stats)
    if _dist_rank() != 0:
        return
    stats = list(chain(*gathered))

    n_stages = args.stage_cap if args.loop_gate <= 0 else 0
    accepted = list(chain.from_iterable(s["acceptance_lengths"] for s in stats))
    tpot = float(np.mean([s["time_per_output_token"] for s in stats]))
    rounds = len(accepted)
    summary = {
        "host": host,
        "baseline_tpot": baseline_tpot,
        "tpot": tpot,
        "speedup": baseline_tpot / tpot,
        "mean_acceptance_length": float(np.mean([np.mean(s["acceptance_lengths"]) for s in stats])),
        "acceptance_histogram": [accepted.count(b) / rounds for b in range(max(accepted) + 1)],
        "stage_fire_rate": [sum(s["stages_fired"][i] for s in stats) / rounds for i in range(n_stages)],
    }
    path = _results_path(args, host, gpu_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({**vars(args), "gpu_name": gpu_name, "block_size": getattr(draft, "block_size", None),
                   "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary,
                   "responses": stats}, f)
    print(f"[{host}] Speedup: {summary['speedup']:.3f}x  "
          f"(baseline {1 / baseline_tpot:.2f} tok/s, drafter {1 / tpot:.2f} tok/s)")
    print(f"Average acceptance length: {summary['mean_acceptance_length']:.3f}")
    if n_stages:
        rates = summary["stage_fire_rate"]
        shown = rates[: max([i + 1 for i, x in enumerate(rates) if x > 0] or [1])]
        print("Loop stage fire rate per round: " + ", ".join(f"s{i + 1}={x:.3f}" for i, x in enumerate(shown)))
    print(f"Saved {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="target model")
    p.add_argument("--draft-model",
                   help="drafter checkpoint (DFlash / DSpark / Domino / EAGLE-3, detected from its "
                        "config), `mtp` for the Qwen3.5 built-in MTP head, or a Qwen3.5 MTP / Gemma 4 "
                        "assistant checkpoint; not needed with --baseline")
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--max-samples", type=int, default=None,
                   help="samples evaluated (default: the paper's, 128 per benchmark; "
                        "AIME25 30, HumanEval 164, MT-Bench 80)")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--baseline", action="store_true", help="measure the autoregressive baseline")
    g = p.add_argument_group("DLoop")
    g.add_argument("--loop-gate", type=float, default=1.0,
                   help="threshold on the summed top-1 log-prob of the segment just drafted "
                        "(> 0 = loop off, the host as released)")
    g.add_argument("--stage-cap", type=int, default=16, help="max loop stages per round")
    g.add_argument("--stage-len", type=int, default=None,
                   help="tokens drafted per loop stage: EAGLE-3 tree levels (default 8), "
                        "MTP draft steps (default --mtp-k); block drafters use their block")
    g.add_argument("--stage-tree-budget", type=int, default=None,
                   help="tree nodes added per loop stage: DFlash tree (default 64), "
                        "EAGLE-3 (default 48), DARTree (default 0 = one chain)")
    t = p.add_argument_group("tree verification")
    t.add_argument("--tree-budget", type=int, default=None,
                   help="tree nodes per round: DFlash DDTree-style tree / DSpark, Domino DARTree "
                        "(default 0 = linear); EAGLE-3 dynamic tree (default 60)")
    t.add_argument("--topk-k", type=int, default=None,
                   help="candidates per depth: DFlash (default: budget), EAGLE-3 (default 10)")
    t.add_argument("--tree-depth", type=int, default=None, help="EAGLE-3 tree depth (default 7)")
    t.add_argument("--tree-width", type=int, default=12, help="DARTree width W")
    t.add_argument("--tree-beta", type=float, default=-0.2, help="DARTree length bonus beta")
    p.add_argument("--mtp-k", type=int, default=None,
                   help="MTP draft steps per round (default: Qwen3.5 3, Gemma 4 6)")
    args = p.parse_args()
    if args.max_samples is None:
        args.max_samples = DATASETS[args.dataset]["max_samples"]
    if args.baseline:
        run_baseline(args)
    else:
        if args.draft_model is None:
            p.error("--draft-model is required")
        run_draft(args)


if __name__ == "__main__":
    main()
