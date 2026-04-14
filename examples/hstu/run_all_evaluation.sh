#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-click script to evaluate all trained checkpoints.
# Each GPU runs one evaluation at a time (GPU 0: retrieval, GPU 1: ranking).
# Both GPUs run tasks in parallel, but sequentially within each GPU.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${PYTHONPATH}:$(realpath ../)"

# Configuration
# Modify these checkpoint directories based on your training output
CKPT_DIR_PREFIX="${1:-.}"  # Optional: pass checkpoint parent dir as first argument

# Log directory
LOG_DIR="eval_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Starting all evaluation jobs..."
echo "Log directory: $LOG_DIR"
echo "GPU 0: retrieval tasks (sequential)"
echo "GPU 1: ranking tasks (sequential)"
echo "========================================="

# GPU 0: retrieval evaluations (sequential, one at a time)
(
    # Job 1: kuairand pure retrieval
    CKPT_DIR="$CKPT_DIR_PREFIX/ckpt_kr_pure_retrieval/iter1000"
    if [ -d "$CKPT_DIR" ]; then
        echo "[GPU 0] Starting kuairand pure retrieval evaluation..."
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6010 \
            ./training/eval_checkpoint.py \
            --gin-config-file ./training/configs/kuairand_pure_retrieval.gin \
            --ckpt-load-dir "$CKPT_DIR" \
            > "$LOG_DIR/kuairand_pure_retrieval_eval.log" 2>&1 && \
            echo "[DONE] kuairand pure retrieval eval" || echo "[FAILED] kuairand pure retrieval eval"
    else
        echo "[SKIP] kuairand pure retrieval: checkpoint dir $CKPT_DIR not found"
    fi

    # Job 2: kuairand 1k retrieval
    CKPT_DIR="$CKPT_DIR_PREFIX/ckpt_kr_1k_retrieval/iter1000"
    if [ -d "$CKPT_DIR" ]; then
        echo "[GPU 0] Starting kuairand 1k retrieval evaluation..."
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6010 \
            ./training/eval_checkpoint.py \
            --gin-config-file ./training/configs/kuairand_1k_retrieval.gin \
            --ckpt-load-dir "$CKPT_DIR" \
            > "$LOG_DIR/kuairand_1k_retrieval_eval.log" 2>&1 && \
            echo "[DONE] kuairand 1k retrieval eval" || echo "[FAILED] kuairand 1k retrieval eval"
    else
        echo "[SKIP] kuairand 1k retrieval: checkpoint dir $CKPT_DIR not found"
    fi

    # Job 3: movielen retrieval
    CKPT_DIR="$CKPT_DIR_PREFIX/ckpt_ml_20m_retrieval/iter1000"
    if [ -d "$CKPT_DIR" ]; then
        echo "[GPU 0] Starting movielen retrieval evaluation..."
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6010 \
            ./training/eval_checkpoint.py \
            --gin-config-file ./training/configs/movielen_retrieval.gin \
            --ckpt-load-dir "$CKPT_DIR" \
            > "$LOG_DIR/movielen_retrieval_eval.log" 2>&1 && \
            echo "[DONE] movielen retrieval eval" || echo "[FAILED] movielen retrieval eval"
    else
        echo "[SKIP] movielen retrieval: checkpoint dir $CKPT_DIR not found"
    fi
) &
GPU0_PID=$!

# GPU 1: ranking evaluations (sequential, one at a time)
(
    # Job 4: kuairand pure ranking
    CKPT_DIR="$CKPT_DIR_PREFIX/ckpt_kr_pure_ranking/iter1000"
    if [ -d "$CKPT_DIR" ]; then
        echo "[GPU 1] Starting kuairand pure ranking evaluation..."
        CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6011 \
            ./training/eval_checkpoint.py \
            --gin-config-file ./training/configs/kuairand_pure_ranking.gin \
            --ckpt-load-dir "$CKPT_DIR" \
            > "$LOG_DIR/kuairand_pure_ranking_eval.log" 2>&1 && \
            echo "[DONE] kuairand pure ranking eval" || echo "[FAILED] kuairand pure ranking eval"
    else
        echo "[SKIP] kuairand pure ranking: checkpoint dir $CKPT_DIR not found"
    fi

    # Job 5: kuairand 1k ranking
    CKPT_DIR="$CKPT_DIR_PREFIX/ckpt_kr_1k_ranking/iter1000"
    if [ -d "$CKPT_DIR" ]; then
        echo "[GPU 1] Starting kuairand 1k ranking evaluation..."
        CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6011 \
            ./training/eval_checkpoint.py \
            --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
            --ckpt-load-dir "$CKPT_DIR" \
            > "$LOG_DIR/kuairand_1k_ranking_eval.log" 2>&1 && \
            echo "[DONE] kuairand 1k ranking eval" || echo "[FAILED] kuairand 1k ranking eval"
    else
        echo "[SKIP] kuairand 1k ranking: checkpoint dir $CKPT_DIR not found"
    fi

    # Job 6: movielen ranking
    CKPT_DIR="$CKPT_DIR_PREFIX/ckpt_ml_20m_ranking/iter1000"
    if [ -d "$CKPT_DIR" ]; then
        echo "[GPU 1] Starting movielen ranking evaluation..."
        CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6011 \
            ./training/eval_checkpoint.py \
            --gin-config-file ./training/configs/movielen_ranking.gin \
            --ckpt-load-dir "$CKPT_DIR" \
            > "$LOG_DIR/movielen_ranking_eval.log" 2>&1 && \
            echo "[DONE] movielen ranking eval" || echo "[FAILED] movielen ranking eval"
    else
        echo "[SKIP] movielen ranking: checkpoint dir $CKPT_DIR not found"
    fi
) &
GPU1_PID=$!

echo "========================================="
echo "All evaluation jobs started!"
echo "GPU 0 PID: $GPU0_PID (retrieval evals)"
echo "GPU 1 PID: $GPU1_PID (ranking evals)"
echo "========================================="
echo "Logs are in: $LOG_DIR/"
echo "========================================="

# Wait for both GPU processes to complete
wait $GPU0_PID && echo "[DONE] GPU 0 all retrieval evals" || echo "[FAILED] GPU 0 had errors"
wait $GPU1_PID && echo "[DONE] GPU 1 all ranking evals" || echo "[FAILED] GPU 1 had errors"

echo "========================================="
echo "All evaluation jobs completed!"
echo "========================================="