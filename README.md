## DLoop: Looped Speculative Decoding
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2610.07659-b31b1b.svg)](https://arxiv.org/abs/2610.07659)

Welcome to the official PyTorch implementation of DLoop!

**Authors**:

**[Geonmo Gu](https://geonm.github.io/)<sup>1,3</sup>, [Byeongho Heo](https://sites.google.com/view/byeongho-heo/home)<sup>1</sup>, HeeJae Jun<sup>2</sup>, Yoohoon Kang<sup>2</sup>, [Sangmin Lee](https://sites.google.com/view/pixel-lab-ai)<sup>3</sup>, [Sangdoo Yun](https://sangdooyun.github.io/)<sup>1</sup>, [Dongyoon Han](https://sites.google.com/site/dyhan0920/)<sup>1</sup>**

<sup>1</sup> NAVER AI Lab <sup>2</sup> NAVER AI Search Platform <sup>3</sup> Korea University


## 🚀 News
- **October 28, 2026** - The code and the [loop-aware trained draft models](#-overview) are released!
- **October 6, 2026** - The DLoop paper is available on arXiv!

## 💡 Overview

Speculative decoding verifies after every drafting stage. With capable draft models,
the target model often accepts every draft token of a stage, so the verification could
have waited while drafting continued. Adaptive draft length methods address this only for
autoregressive draft models: a parallel draft model would need target-model hidden
states for draft tokens that have not been verified yet. **DLoop** performs several
drafting stages before one verification:

- **Confidence-gated drafting loop**: after each drafting stage, a gate sums the draft
  model's log-probabilities of the tokens it has just produced. While the sum stays above
  a threshold `g`, another drafting stage follows; otherwise the target model verifies all
  accumulated draft tokens in one forward pass. The acceptance rule is unchanged, so
  decoding remains lossless.
- **Loop-aware training**: one epoch, starting from the released draft weights, in which
  additional drafting stages condition on the draft model's own hidden states for the
  unverified draft tokens. It adds no parameters and does not change the architecture.

Across EAGLE-3, DFlash, Domino, DSpark and multi-token prediction modules, DLoop improves
the wall-clock speedup by 5 to 41 percent. This repository trains and evaluates all five
host draft model families:

<table>
  <tr><th>Host</th><th>Baseline draft model</th><th>Loop-aware trained draft model</th></tr>
  <tr><td rowspan="2">DFlash</td><td><a href="https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16">z-lab/Qwen3-4B-DFlash-b16</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-DFlash-Qwen3-4B">naver-ai/DLoop-DFlash-Qwen3-4B</a></td></tr>
  <tr><td><a href="https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16">z-lab/Qwen3-8B-DFlash-b16</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-DFlash-Qwen3-8B">naver-ai/DLoop-DFlash-Qwen3-8B</a></td></tr>
  <tr><td rowspan="2">DSpark</td><td><a href="https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7">deepseek-ai/dspark_qwen3_4b_block7</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-DSpark-Qwen3-4B">naver-ai/DLoop-DSpark-Qwen3-4B</a></td></tr>
  <tr><td><a href="https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7">deepseek-ai/dspark_qwen3_8b_block7</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-DSpark-Qwen3-8B">naver-ai/DLoop-DSpark-Qwen3-8B</a></td></tr>
  <tr><td rowspan="2">Domino</td><td><a href="https://huggingface.co/Huang2020/Qwen3-4B-Domino-b16">Huang2020/Qwen3-4B-Domino-b16</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-Domino-Qwen3-4B">naver-ai/DLoop-Domino-Qwen3-4B</a></td></tr>
  <tr><td><a href="https://huggingface.co/Huang2020/Qwen3-8B-Domino-b16">Huang2020/Qwen3-8B-Domino-b16</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-Domino-Qwen3-8B">naver-ai/DLoop-Domino-Qwen3-8B</a></td></tr>
  <tr><td rowspan="2">EAGLE-3</td><td><a href="https://huggingface.co/naver-ai/EAGLE3-Qwen3-4B">naver-ai/EAGLE3-Qwen3-4B</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-EAGLE3-Qwen3-4B">naver-ai/DLoop-EAGLE3-Qwen3-4B</a></td></tr>
  <tr><td><a href="https://huggingface.co/naver-ai/EAGLE3-Qwen3-8B">naver-ai/EAGLE3-Qwen3-8B</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-EAGLE3-Qwen3-8B">naver-ai/DLoop-EAGLE3-Qwen3-8B</a></td></tr>
  <tr><td rowspan="2">MTP</td><td><a href="https://huggingface.co/Qwen/Qwen3.5-9B">Qwen/Qwen3.5-9B</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-MTP-Qwen3.5-9B">naver-ai/DLoop-MTP-Qwen3.5-9B</a></td></tr>
  <tr><td><a href="https://huggingface.co/google/gemma-4-12B-it-assistant">google/gemma-4-12B-it-assistant</a></td><td><a href="https://huggingface.co/naver-ai/DLoop-MTP-gemma-4-12B-it">naver-ai/DLoop-MTP-gemma-4-12B-it</a></td></tr>
</table>

The Qwen3.5-9B baseline is the MTP head built into the target checkpoint; the EAGLE-3
baselines were trained with SpecForge for 3 epochs.

## 🛠️ Installation

```bash
git clone https://github.com/naver-ai/DLoop.git
cd DLoop

uv venv -p 3.12 --seed
source .venv/bin/activate
uv pip install -e .
```

The code is SpecForge with the target model inside the trainer (`--target-model-backend hf`),
so no inference server is needed for training. It is tested with torch 2.8 and 2.9 and
transformers 5.8.1; the MTP hosts (Qwen3.5, Gemma 4) need transformers 5.16 or later
(`uv pip install -e ".[mtp]"`, which also installs flash-linear-attention: without its
kernels transformers runs Qwen3.5's linear-attention layers on a slow PyTorch path, and the
paper's Qwen3.5 timings use flash-linear-attention 0.5.2).
SGLang is only needed for `--target-model-backend sglang` and for regenerating the data.

## 🗂️ Dataset Preparation

The training set pairs PerfectBlend user prompts with responses
generated by the target model. The SpecForge data scripts produce it:

```bash
# 1. PerfectBlend prompts -> ./cache/dataset/perfectblend_train.jsonl
python scripts/prepare_data.py --dataset perfectblend

# 2. Regenerate the responses with the target model
python -m sglang.launch_server --model-path Qwen/Qwen3-4B --dtype bfloat16 \
    --mem-fraction-static 0.8 --port 30000
python scripts/regenerate_train_data.py \
    --model Qwen/Qwen3-4B \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --max-tokens 4096 \
    --concurrency 128 \
    --server-address localhost:30000 \
    --input-file-path ./cache/dataset/perfectblend_train.jsonl \
    --output-file-path ./cache/dataset/perfectblend_qwen3-4b_regen.jsonl
```

Pass several `--server-address` values to regenerate in data parallel. Regenerate once per
target model: Qwen3-8B for the 8B drafters, and Qwen3.5-9B / Gemma-4-12B for the MTP hosts
(the paper used their model-card sampling settings: temperature 0.7, top-p 0.8, top-k 20
for Qwen3.5 and temperature 1.0, top-p 0.95, top-k 64 for Gemma 4). Any JSONL with one
conversation per line works:

```json
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## 🔥 Training

Loop-aware training starts from the released draft model and runs for one epoch on the
regenerated data. DSpark on Qwen3-4B, for example, runs on every node as

```bash
torchrun --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT --nproc_per_node 8 scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3-4B \
    --target-model-backend hf \
    --draft-init-path deepseek-ai/dspark_qwen3_4b_block7 \
    --train-data-path ./cache/dataset/perfectblend_qwen3-4b_regen.jsonl \
    --on-the-fly-loader \
    --chat-template qwen \
    --max-length 4096 \
    --output-dir ./outputs/dloop-dspark-qwen3-4b \
    --num-epochs 1 --batch-size 1 \
    --learning-rate 6e-4 --warmup-ratio 0.04 --max-grad-norm 1.0 --seed 42 \
    --attention-backend flex_attention \
    --num-anchors 512 --loss-decay-gamma 4.0 \
    --dloop-max-stage 3 --dloop-loss-weight 0.25
```

The paper's settings for the other hosts are in [`run_dloop_dflash.sh`](examples/run_dloop_dflash.sh),
[`run_dloop_domino.sh`](examples/run_dloop_domino.sh), [`run_dloop_eagle3.sh`](examples/run_dloop_eagle3.sh)
and [`run_dloop_mtp.sh`](examples/run_dloop_mtp.sh). Each runs as
`DATA=<jsonl> bash examples/run_dloop_<host>.sh 4B` (or `8B`; `qwen3.5` or `gemma4` for MTP) on
one node, or on several with `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` set.
EAGLE-3 also needs `CKPT`, the checkpoint to start from:

```bash
CKPT=$(python -c "from huggingface_hub import snapshot_download as d; print(d('naver-ai/EAGLE3-Qwen3-4B'))") \
    DATA=<jsonl> bash examples/run_dloop_eagle3.sh 4B
```

Every training step keeps each host's own objective and adds one loop stage in the layout
the deployed loop runs. Block drafters (DFlash, DSpark, Domino) sample up to `--num-anchors`
anchors `a` per sequence (`B` = block size: 16 for DFlash and Domino, 7 for DSpark):

| drafting stage | input rows | extra K/V | labels |
| --- | --- | --- | --- |
| first (the host's block) | `[x_a, MASK×(B-1)]` | target hidden states before `a` | the host's own block labels |
| loop stage `k ~ U{1, …, K}` | `[x_{a+o}, MASK×(B-1)]`, `o = k(B − l)` | + draft hidden states of `a+1 … a+o-1` | the next `B − l` tokens after `a+o` |

`l = 1` for DFlash (same-position labels) and `0` for DSpark and Domino (next-token labels).
Every stage has the first stage's input and output length (`B` rows, `B − l` draft tokens);
its anchor `x_{a+o}` is the last token of the previous stage.
Stages `1 … k-1` run in the same layout without a loss and supply the draft hidden states;
input tokens are the ground truth, and gradients flow through the draft hidden states into
the stages that produced them. Autoregressive drafters (EAGLE-3, MTP) already draft from
their own hidden states, so the loop stage is a longer test-time-training unroll: the unroll
length is drawn uniformly from `--ttt-length-dist`, the host's draft depth times 1, 2 or 3.

| Flag | Script | Meaning |
| --- | --- | --- |
| `--draft-init-path` | `train_dflash.py`, `train_domino.py` | Released draft model to start from (its config and weights; DSpark checkpoints switch to the DSpark objective). |
| `--ckpt-dir` | `train_eagle3.py` | EAGLE-3 checkpoint to start from (its vocabulary mapping is kept); the Gemma 4 assistant for `--mtp gemma4`. |
| `--dloop-max-stage 3` | `train_dflash.py`, `train_domino.py` | Loop stage `k ~ U{1, 2, 3}` each step (0 = the host objective alone). |
| `--dloop-loss-weight 0.25` | DSpark, Domino | `loss = 0.75 × host loss + 0.25 × loop-stage loss`. DFlash normalizes the two CE terms jointly. |
| `--position-offset-max 28672` | DFlash | Shifts each sample's RoPE positions by a random offset in `[0, 28672 − length]`, for the target and the draft. |
| `--ttt-length-dist 7,14,21` | `train_eagle3.py` | Unroll lengths, one drawn uniformly per step (EAGLE-3 `7,14,21`; MTP `3,6,9` for Qwen3.5, `6,12,18` for Gemma 4). |
| `--on-the-fly-loader` | all | Tokenizes in the DataLoader workers instead of preprocessing the whole dataset upfront. |

Checkpoints `outputs/<run>/epoch_{e}_step_{s}` are standard checkpoints of their host
(SpecForge format for DFlash, DSpark, Domino and EAGLE-3; the Qwen3.5 MTP head and the Gemma 4
assistant in their Hugging Face formats) and load in `eval/` without extra flags.

## 💯 Evaluation

The evaluation uses greedy decoding with Hugging Face Transformers (SDPA attention), one
sample at a time, on eight benchmarks: GSM8K, MATH-500, AIME25, HumanEval, MBPP,
LiveCodeBench, MT-Bench and Alpaca. Measure the autoregressive baseline once per target
model, benchmark and GPU type, then each setting; the host is detected from the draft model:

```bash
torchrun --nproc_per_node=8 -m eval.benchmark --baseline \
    --model Qwen/Qwen3-4B --dataset gsm8k

torchrun --nproc_per_node=8 -m eval.benchmark \
    --model Qwen/Qwen3-4B --draft-model naver-ai/DLoop-DFlash-Qwen3-4B \
    --dataset gsm8k \
    --loop-gate -0.5
```

`--draft-model` takes a Hugging Face repo id or a local checkpoint, such as your own training
output (`./outputs/dloop-dflash-qwen3-4b/epoch_1_step_22202`).

The settings of the paper. Pass `--draft-model` the baseline draft model of the
[overview table](#-overview) in the "released" rows and the loop-aware trained one
(`naver-ai/DLoop-*`) in the "DLoop" rows, with the flags below (no flags = the host as released):

<table>
  <tr><th>Host</th><th>Setting</th><th>Qwen3-4B</th><th>Qwen3-8B</th></tr>
  <tr><td rowspan="4">DFlash</td><td>released</td><td></td><td></td></tr>
  <tr><td>+ DLoop</td><td><code>--loop-gate -0.5</code></td><td><code>--loop-gate -0.5</code></td></tr>
  <tr><td>released + DDTree</td><td><code>--tree-budget 256</code></td><td><code>--tree-budget 256</code></td></tr>
  <tr><td>DLoop + DDTree</td><td><code>--tree-budget 256 --stage-tree-budget 64 --loop-gate -0.5</code></td><td><code>--tree-budget 256 --stage-tree-budget 64 --loop-gate -0.5</code></td></tr>
  <tr><td rowspan="4">DSpark</td><td>released</td><td></td><td></td></tr>
  <tr><td>+ DLoop</td><td><code>--loop-gate -0.5</code></td><td><code>--loop-gate -0.5</code></td></tr>
  <tr><td>released + DARTree</td><td><code>--tree-budget 64 --tree-width 12 --tree-beta -0.2</code></td><td><code>--tree-budget 64 --tree-width 12 --tree-beta -0.2</code></td></tr>
  <tr><td>DLoop + DARTree</td><td><code>--tree-budget 64 --tree-width 12 --tree-beta -0.2 --loop-gate -0.5</code></td><td><code>--tree-budget 64 --tree-width 12 --tree-beta -0.2 --loop-gate -0.5</code></td></tr>
  <tr><td rowspan="4">Domino</td><td>released</td><td></td><td></td></tr>
  <tr><td>+ DLoop</td><td><code>--loop-gate -0.5</code></td><td><code>--loop-gate -0.5</code></td></tr>
  <tr><td>released + DARTree</td><td><code>--tree-budget 64 --tree-width 12 --tree-beta -0.2</code></td><td><code>--tree-budget 64 --tree-width 12 --tree-beta -0.2</code></td></tr>
  <tr><td>DLoop + DARTree</td><td><code>--tree-budget 64 --tree-width 12 --tree-beta -0.2 --loop-gate -0.5</code></td><td><code>--tree-budget 64 --tree-width 12 --tree-beta -0.2 --loop-gate -0.5</code></td></tr>
  <tr><td rowspan="2">EAGLE-3</td><td>released</td><td></td><td></td></tr>
  <tr><td>+ DLoop</td><td><code>--loop-gate -0.5 --stage-tree-budget 48</code></td><td><code>--loop-gate -0.75 --stage-tree-budget 96</code></td></tr>
</table>

<table>
  <tr><th>Host</th><th>Setting</th><th>Qwen3.5-9B (released: <code>--draft-model mtp</code>, the built-in head)</th><th>Gemma-4-12B</th></tr>
  <tr><td rowspan="2">MTP</td><td>released</td><td><code>--mtp-k 3</code></td><td><code>--mtp-k 6</code></td></tr>
  <tr><td>+ DLoop</td><td><code>--mtp-k 3 --loop-gate -0.5</code></td><td><code>--mtp-k 6 --loop-gate -0.5</code></td></tr>
</table>

| Flag | Meaning |
| --- | --- |
| `--loop-gate` | Gate threshold `g` on the summed top-1 log-probability of the segment just drafted (after the host's own logit corrections). The default, 1.0, never opens: the host as released. |
| `--tree-budget` | Tree nodes verified per round: DDTree-style best-first tree (DFlash), DARTree (DSpark, Domino), the EAGLE-3 dynamic tree (default 60). 0 = linear. |
| `--stage-tree-budget` | Tree nodes added per additional stage: DFlash tree (default 64), EAGLE-3 (default 48), DARTree (default 0 = one chain per stage, as in the paper; N > 0 = a DARTree of N nodes). |
| `--topk-k` | Candidates per tree depth (DFlash), EAGLE-3 top-k (default 10). |
| `--tree-width`, `--tree-beta` | DARTree width and length bonus. |
| `--mtp-k` | MTP draft steps per round (default: 3 for Qwen3.5, 6 for Gemma 4). |

Each benchmark is evaluated on the first samples after shuffling it with seed 0, as many as in
the paper: 128 per benchmark (all 30 for AIME25, 164 for HumanEval, 80 for MT-Bench);
`--max-samples` overrides the count. The paper uses `--max-new-tokens 2048` on H100 80GB GPUs.
Speedups are comparable only on the same GPU type as the baseline. Results, including every
round's acceptance length and the number of additional drafting stages, are written to `results/`.

## Citation
```
@article{gu2026dloop,
    title={DLoop: Looped Speculative Decoding},
    author={Gu, Geonmo and Heo, Byeongho and Jun, HeeJae and Kang, Yoohoon and Lee, Sangmin and Yun, Sangdoo and Han, Dongyoon},
    year={2026},
    journal={arXiv preprint arXiv:2610.07659},
}
```

## License
```
DLoop
Copyright (c) 2026-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

This repository is built on [SpecForge](https://github.com/sgl-project/SpecForge), with code from
[DFlash](https://github.com/z-lab/dflash) in `eval/` and from
[EAGLE](https://github.com/SafeAILab/EAGLE) in `eval/eagle3_decode.py`. Each
source file states its origin in its header.
