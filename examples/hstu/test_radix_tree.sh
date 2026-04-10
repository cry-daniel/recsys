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

# Test radix tree reordering strategies using training code's evaluation pipeline
# Usage: ./test_radix_tree.sh <dataset> [CHECKPOINT_DIR] [OUTPUT_FILE]
#   dataset: kuairand1k, kuairandpure, or ml20m

# Parse arguments
DATASET="${1:-kuairand1k}"

# Set dataset-specific configurations
case "${DATASET}" in
    kuairand1k)
        GIN_CONFIG_FILE="./training/configs/kuairand_1k_ranking.gin"
        CHECKPOINT_DIR="${2:-ckpt_kr_1k_ranking/iter1000}"
        OUTPUT_FILE="${3:-training_radix_tree_kuairand1k_results.csv}"
        ;;
    kuairandpure)
        GIN_CONFIG_FILE="./training/configs/kuairand_pure_ranking.gin"
        CHECKPOINT_DIR="${2:-ckpt_kr_pure_ranking/iter1000}"
        OUTPUT_FILE="${3:-training_radix_tree_kuairandpure_results.csv}"
        ;;
    ml20m)
        GIN_CONFIG_FILE="./training/configs/movielen_ranking.gin"
        CHECKPOINT_DIR="${2:-ckpt_ml_20m_ranking/iter1000}"
        OUTPUT_FILE="${3:-training_radix_tree_ml20m_results.csv}"
        ;;
    *)
        echo "ERROR: Unknown dataset '${DATASET}'"
        echo "Supported datasets: kuairand1k, kuairandpure, ml20m"
        exit 1
        ;;
esac

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output and log directories
OUTPUT_DIR="$(dirname "${SCRIPT_DIR}/${OUTPUT_FILE}")"
LOG_DIR="${OUTPUT_DIR}/radix_tree_logs"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "Training Radix Tree Reordering Test"
echo "============================================"
echo "Dataset: ${DATASET}"
echo "GIN Config: ${GIN_CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output File: ${OUTPUT_FILE}"
echo "Log Directory: ${LOG_DIR}"
echo "============================================"

# Remove existing output file if it exists
if [ -f "${SCRIPT_DIR}/${OUTPUT_FILE}" ]; then
    echo "Removing existing output file: ${OUTPUT_FILE}"
    rm "${SCRIPT_DIR}/${OUTPUT_FILE}"
fi

# Test different radix tree strategies
STRATEGIES=("prefix_score" "frequency" "hybrid" "unordered_tree")
UNORDERED_FLAGS=("" "" "" "--unordered")

for i in "${!STRATEGIES[@]}"; do
    STRATEGY="${STRATEGIES[$i]}"
    UNORDERED="${UNORDERED_FLAGS[$i]}"
    
    echo ""
    echo "============================================"
    echo "Testing strategy: ${STRATEGY} (unordered=${UNORDERED})"
    echo "============================================"
    
    cd "${SCRIPT_DIR}/"
    
    # Create log file for this run
    LOG_FILE="${LOG_DIR}/${DATASET}_${STRATEGY}${UNORDERED:+_unordered}.log"
    
    # Build command
    CMD="CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6003 ./training/eval_radix_tree.py \
        --gin-config-file ${GIN_CONFIG_FILE} \
        --ckpt-load-dir ${CHECKPOINT_DIR} \
        --radix_tree_strategy ${STRATEGY} \
        --output_file ${SCRIPT_DIR}/${OUTPUT_FILE}"
    
    # Add unordered flag if needed
    if [ -n "${UNORDERED}" ]; then
        CMD="${CMD} ${UNORDERED}"
    fi
    
    # Run the radix tree test
    eval "${CMD}" 2>&1 | tee "${LOG_FILE}"
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Test failed for strategy=${STRATEGY}"
        exit 1
    fi
done

echo ""
echo "============================================"
echo "All tests completed!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "Logs saved to: ${LOG_DIR}"
echo "============================================"

# Display results summary
if [ -f "${SCRIPT_DIR}/${OUTPUT_FILE}" ]; then
    echo ""
    echo "Results Summary:"
    echo "----------------"
    cat "${SCRIPT_DIR}/${OUTPUT_FILE}"
fi