#!/bin/bash
# DLoop
# Copyright (c) 2026-present NAVER Corp.
#
# One-epoch DLoop training of the released Domino drafter (paper recipe; the
# base-loss weight stays at 0.1 for the whole epoch).
#   DATA=<jsonl> bash examples/run_dloop_domino.sh 4B|8B [extra train_domino.py args]
# Multi-node: also set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT (and NUM_GPUS per node).
SIZE=${1:?usage: run_dloop_domino.sh 4B|8B}; shift
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

torchrun $LAUNCH --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_domino.py \
    --target-model-path Qwen/Qwen3-$SIZE \
    --target-model-backend hf \
    --draft-init-path Huang2020/Qwen3-$SIZE-Domino-b16 \
    --train-data-path "$DATA" \
    --on-the-fly-loader \
    --chat-template qwen \
    --max-length 4096 \
    --output-dir ${OUTPUT_DIR:-$ROOT_DIR/outputs/dloop-domino-qwen3-${SIZE,,}} \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --seed 42 \
    --attention-backend flex_attention \
    --num-anchors 256 \
    --loss-decay-gamma 16.0 \
    --lambda-base-start 0.1 \
    --lambda-base-decay-ratio 1000000 \
    --dloop-max-stage 3 \
    --dloop-loss-weight 0.25 \
    --log-interval 50 \
    --save-interval 5000 \
    --report-to tensorboard \
    "$@"
