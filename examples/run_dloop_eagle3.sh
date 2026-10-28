#!/bin/bash
# DLoop
# Copyright (c) 2026-present NAVER Corp.
#
# One-epoch DLoop training of an EAGLE-3 drafter (paper recipe): the TTT unroll
# length is drawn uniformly from 7, 14, 21 (the host depth x {1, 2, 3} stages).
#   DATA=<jsonl> CKPT=<EAGLE-3 checkpoint (SpecForge format)> \
#       bash examples/run_dloop_eagle3.sh 4B|8B [extra train_eagle3.py args]
# Multi-node: also set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT (and NUM_GPUS per node).
SIZE=${1:?usage: run_dloop_eagle3.sh 4B|8B}; shift
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
NUM_GPUS=${NUM_GPUS:-8}
if [ -n "$NNODES" ]; then
    LAUNCH="--nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    LAUNCH="--standalone"
fi
: "${DATA:?set DATA to the regenerated training jsonl (see README, Dataset Preparation)}"
: "${CKPT:?set CKPT to the EAGLE-3 checkpoint to start from}"
LR=5e-4; [ "$SIZE" = "8B" ] && LR=1e-4  # 8B diverges at 5e-4

torchrun $LAUNCH --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-$SIZE \
    --target-model-backend hf \
    --ckpt-dir "$CKPT" \
    --train-data-path "$DATA" \
    --on-the-fly-loader \
    --chat-template qwen \
    --max-length 4096 \
    --output-dir ${OUTPUT_DIR:-$ROOT_DIR/outputs/dloop-eagle3-qwen3-${SIZE,,}} \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate $LR \
    --warmup-ratio 0.015 \
    --max-grad-norm 0.5 \
    --seed 0 \
    --attention-backend flex_attention \
    --ttt-length-dist 7,14,21 \
    --log-interval 50 \
    --save-interval 5000 \
    --report-to tensorboard \
    "$@"
