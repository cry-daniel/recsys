#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One-click script to train all kuairand pure/1k and movielen retrieval/ranking tasks.
# Each GPU runs one training at a time (GPU 0: retrieval, GPU 1: ranking).
# Both GPUs run tasks in parallel, but sequentially within each GPU.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${PYTHONPATH}:$(realpath ../)"

# Log directory
LOG_DIR="training_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Starting all training jobs..."
echo "Log directory: $LOG_DIR"
echo "GPU 0: retrieval tasks (sequential)"
# echo "GPU 1: ranking tasks (sequential)"
echo "========================================="

# GPU 0: retrieval tasks (sequential, one at a time)
(
    # Job 1: kuairand pure retrieval
    echo "[GPU 0] Starting kuairand pure retrieval..."
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
        ./training/pretrain_gr_retrieval.py \
        --gin-config-file ./training/configs/kuairand_pure_retrieval.gin \
        > "$LOG_DIR/kuairand_pure_retrieval.log" 2>&1 && \
        echo "[DONE] kuairand pure retrieval" || echo "[FAILED] kuairand pure retrieval"

    # Job 2: kuairand 1k retrieval
    echo "[GPU 0] Starting kuairand 1k retrieval..."
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
        ./training/pretrain_gr_retrieval.py \
        --gin-config-file ./training/configs/kuairand_1k_retrieval.gin \
        > "$LOG_DIR/kuairand_1k_retrieval.log" 2>&1 && \
        echo "[DONE] kuairand 1k retrieval" || echo "[FAILED] kuairand 1k retrieval"

    # Job 3: movielen retrieval
    echo "[GPU 0] Starting movielen retrieval..."
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 \
        ./training/pretrain_gr_retrieval.py \
        --gin-config-file ./training/configs/movielen_retrieval.gin \
        > "$LOG_DIR/movielen_retrieval.log" 2>&1 && \
        echo "[DONE] movielen retrieval" || echo "[FAILED] movielen retrieval"
) &
GPU0_PID=$!

# # GPU 1: ranking tasks (sequential, one at a time)
# (
#     # Job 4: kuairand pure ranking
#     echo "[GPU 1] Starting kuairand pure ranking..."
#     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6001 \
#         ./training/pretrain_gr_ranking.py \
#         --gin-config-file ./training/configs/kuairand_pure_ranking.gin \
#         > "$LOG_DIR/kuairand_pure_ranking.log" 2>&1 && \
#         echo "[DONE] kuairand pure ranking" || echo "[FAILED] kuairand pure ranking"

#     # Job 5: kuairand 1k ranking
#     echo "[GPU 1] Starting kuairand 1k ranking..."
#     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6001 \
#         ./training/pretrain_gr_ranking.py \
#         --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
#         > "$LOG_DIR/kuairand_1k_ranking.log" 2>&1 && \
#         echo "[DONE] kuairand 1k ranking" || echo "[FAILED] kuairand 1k ranking"

#     # Job 6: movielen ranking
#     echo "[GPU 1] Starting movielen ranking..."
#     CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_addr localhost --master_port 6001 \
#         ./training/pretrain_gr_ranking.py \
#         --gin-config-file ./training/configs/movielen_ranking.gin \
#         > "$LOG_DIR/movielen_ranking.log" 2>&1 && \
#         echo "[DONE] movielen ranking" || echo "[FAILED] movielen ranking"
# ) &
# GPU1_PID=$!

echo "========================================="
echo "All training jobs started!"
echo "GPU 0 PID: $GPU0_PID (retrieval tasks)"
# echo "GPU 1 PID: $GPU1_PID (ranking tasks)"
echo "========================================="
echo "Logs are in: $LOG_DIR/"
echo "========================================="

# Wait for both GPU processes to complete
wait $GPU0_PID && echo "[DONE] GPU 0 all retrieval tasks" || echo "[FAILED] GPU 0 had errors"
# wait $GPU1_PID && echo "[DONE] GPU 1 all ranking tasks" || echo "[FAILED] GPU 1 had errors"

echo "========================================="
echo "All training jobs completed!"
echo "========================================="
