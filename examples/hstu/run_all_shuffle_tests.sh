#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run all shuffle accuracy tests for all datasets and tasks
# Usage: ./run_all_shuffle_tests.sh [OUTPUT_DIR] [--normalize]

OUTPUT_DIR="${1:-./training/shuffle_test_results}"
NORMALIZE_ITEM_ACTION="${2:-}"

# Add _norm suffix to output files if normalize is enabled
if [ "${NORMALIZE_ITEM_ACTION}" = "--normalize" ]; then
    NORMALIZE_FLAG="--normalize_item_action"
else
    NORMALIZE_FLAG=""
fi
LOG_DIR="${OUTPUT_DIR}/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create output and log directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# "kuairand1k_retrieval|./training/configs/kuairand_1k_retrieval.gin|ckpt_kr_1k_retrieval/iter1000|${OUTPUT_DIR}/shuffle_kuairand1k_retrieval.csv"

# Define all test configurations
# Format: "dataset_name|gin_config|checkpoint_dir|output_file"
# Add _norm suffix to output files if normalize is enabled
if [ "${NORMALIZE_ITEM_ACTION}" = "--normalize" ]; then
    CONFIGS=(
        "kuairandpure_ranking|./training/configs/kuairand_pure_ranking.gin|ckpt_kr_pure_ranking/iter1000|${OUTPUT_DIR}/shuffle_kuairandpure_ranking_norm.csv"
        "kuairandpure_retrieval|./training/configs/kuairand_pure_retrieval.gin|ckpt_kr_pure_retrieval/iter1000|${OUTPUT_DIR}/shuffle_kuairandpure_retrieval_norm.csv"
        "ml20m_ranking|./training/configs/movielen_ranking.gin|ckpt_ml_20m_ranking/iter1000|${OUTPUT_DIR}/shuffle_ml20m_ranking_norm.csv"
        "ml20m_retrieval|./training/configs/movielen_retrieval.gin|ckpt_ml_20m_retrieval/iter1000|${OUTPUT_DIR}/shuffle_ml20m_retrieval_norm.csv"
    )
else
    CONFIGS=(
        "kuairandpure_ranking|./training/configs/kuairand_pure_ranking.gin|ckpt_kr_pure_ranking/iter1000|${OUTPUT_DIR}/shuffle_kuairandpure_ranking.csv"
        "kuairandpure_retrieval|./training/configs/kuairand_pure_retrieval.gin|ckpt_kr_pure_retrieval/iter1000|${OUTPUT_DIR}/shuffle_kuairandpure_retrieval.csv"
        "ml20m_ranking|./training/configs/movielen_ranking.gin|ckpt_ml_20m_ranking/iter1000|${OUTPUT_DIR}/shuffle_ml20m_ranking.csv"
        "ml20m_retrieval|./training/configs/movielen_retrieval.gin|ckpt_ml_20m_retrieval/iter1000|${OUTPUT_DIR}/shuffle_ml20m_retrieval.csv"
    )
fi

echo "============================================"
echo "All Shuffle Accuracy Tests"
echo "============================================"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Number of configurations: ${#CONFIGS[@]}"
echo "============================================"

# Track overall success
OVERALL_SUCCESS=true

# Run tests for each configuration
for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME GIN_CONFIG CHECKPOINT_DIR OUTPUT_FILE <<< "${CONFIG}"
    
    echo ""
    echo "============================================"
    echo "Testing: ${NAME}"
    echo "============================================"
    echo "GIN Config: ${GIN_CONFIG}"
    echo "Checkpoint: ${CHECKPOINT_DIR}"
    echo "Output: ${OUTPUT_FILE}"
    echo "============================================"
    
    # Check if config file exists
    if [ ! -f "${SCRIPT_DIR}/${GIN_CONFIG}" ]; then
        echo "WARNING: Config file ${GIN_CONFIG} not found, skipping..."
        continue
    fi
    
    # Check if checkpoint exists
    if [ ! -d "${SCRIPT_DIR}/${CHECKPOINT_DIR}" ]; then
        echo "WARNING: Checkpoint directory ${CHECKPOINT_DIR} not found, skipping..."
        continue
    fi
    
    # Run shuffle test
    cd "${SCRIPT_DIR}"
    
    for SHUFFLE_PCT in 0 10 20 30 40 50 60 70 80 90 100; do
    # for SHUFFLE_PCT in 0 40; do
        echo ""
        echo "--- ${NAME}: shuffle_pct=${SHUFFLE_PCT}% ---"
        
        # Create log file for this run
        LOG_FILE="${LOG_DIR}/${NAME}_shuffle_pct_${SHUFFLE_PCT}.log"
        
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
            torchrun --nproc_per_node 1 --master_addr localhost \
            --master_port 6000 ./training/eval_checkpoint_shuffle.py \
            --gin-config-file "${GIN_CONFIG}" \
            --ckpt-load-dir "${CHECKPOINT_DIR}" \
            --shuffle_pct "${SHUFFLE_PCT}" \
            --output_file "${OUTPUT_FILE}" \
            ${NORMALIZE_FLAG} 2>&1 | tee "${LOG_FILE}"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Test failed for ${NAME} at shuffle_pct=${SHUFFLE_PCT}%"
            OVERALL_SUCCESS=false
        fi
    done
    
    echo ""
    echo "============================================"
    echo "Completed: ${NAME}"
    echo "============================================"
done

echo ""
echo "============================================"
if [ "${OVERALL_SUCCESS}" = true ]; then
    echo "All tests completed successfully!"
else
    echo "Some tests failed. Please check the output above."
fi
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"

# Display summary of all results
echo ""
echo "Results Summary:"
echo "================"
for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r NAME GIN_CONFIG CHECKPOINT_DIR OUTPUT_FILE <<< "${CONFIG}"
    if [ -f "${OUTPUT_FILE}" ]; then
        echo ""
        echo "--- ${NAME} ---"
        cat "${OUTPUT_FILE}"
    fi
done