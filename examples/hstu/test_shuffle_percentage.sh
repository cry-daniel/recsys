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

# Test shuffle accuracy from 10% to 100% using training code's evaluation pipeline
# Usage: ./test_shuffle_percentage.sh <dataset> [CHECKPOINT_DIR] [OUTPUT_FILE] [--normalize]
#   dataset: kuairand1k, kuairandpure, or ml20m

# Parse arguments
DATASET="${1:-kuairand1k}"
NORMALIZE_ITEM_ACTION="${4:-}"

# Set dataset-specific configurations
case "${DATASET}" in
    kuairand1k)
        GIN_CONFIG_FILE="./training/configs/kuairand_1k_ranking.gin"
        CHECKPOINT_DIR="${2:-ckpt_kr_1k_ranking/iter1000}"
        OUTPUT_FILE="${3:-training_shuffle_accuracy_kuairand1k_results.csv}"
        ;;
    kuairandpure)
        GIN_CONFIG_FILE="./training/configs/kuairand_pure_ranking.gin"
        CHECKPOINT_DIR="${2:-ckpt_kr_pure_ranking/iter1000}"
        OUTPUT_FILE="${3:-training_shuffle_accuracy_kuairandpure_results.csv}"
        ;;
    ml20m)
        GIN_CONFIG_FILE="./training/configs/movielen_ranking.gin"
        CHECKPOINT_DIR="${2:-ckpt_ml_20m_ranking/iter1000}"
        OUTPUT_FILE="${3:-training_shuffle_accuracy_ml20m_results.csv}"
        ;;
    *)
        echo "ERROR: Unknown dataset '${DATASET}'"
        echo "Supported datasets: kuairand1k, kuairandpure, ml20m"
        exit 1
        ;;
esac

# Add _norm suffix if normalize_item_action is enabled
if [ "${NORMALIZE_ITEM_ACTION}" = "--normalize" ]; then
    OUTPUT_FILE="${OUTPUT_FILE%.csv}_norm.csv"
fi

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "Training Shuffle Accuracy Test (0% to 100%)"
echo "============================================"
echo "Dataset: ${DATASET}"
echo "GIN Config: ${GIN_CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Output File: ${OUTPUT_FILE}"
echo "Normalize Item Action: ${NORMALIZE_ITEM_ACTION}"
echo "============================================"

# Remove existing output file if it exists
if [ -f "${OUTPUT_FILE}" ]; then
    echo "Removing existing output file: ${OUTPUT_FILE}"
    rm "${OUTPUT_FILE}"
fi

# Test shuffle percentages from 0% to 100%
# for SHUFFLE_PCT in 0 10 20 30 40 50 60 70 80 90 100; do
for SHUFFLE_PCT in 0 40; do
    echo ""
    echo "============================================"
    echo "Testing shuffle_pct=${SHUFFLE_PCT}%"
    echo "============================================"
    
    cd "${SCRIPT_DIR}/"
    
    # Build normalize flag
    NORMALIZE_FLAG=""
    if [ "${NORMALIZE_ITEM_ACTION}" = "--normalize" ]; then
        NORMALIZE_FLAG="--normalize_item_action"
    fi
    
    # Run the shuffle test
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6000 ./training/eval_checkpoint_shuffle.py \
        --gin-config-file "${GIN_CONFIG_FILE}" \
        --ckpt-load-dir "${CHECKPOINT_DIR}" \
        --shuffle_pct "${SHUFFLE_PCT}" \
        --output_file "${SCRIPT_DIR}/${OUTPUT_FILE}" \
        ${NORMALIZE_FLAG}
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Test failed for shuffle_pct=${SHUFFLE_PCT}%"
        exit 1
    fi
done

echo ""
echo "============================================"
echo "All tests completed!"
echo "Results saved to: ${OUTPUT_FILE}"
echo "============================================"

# Display results summary
if [ -f "${SCRIPT_DIR}/${OUTPUT_FILE}" ]; then
    echo ""
    echo "Results Summary:"
    echo "----------------"
    cat "${SCRIPT_DIR}/${OUTPUT_FILE}"
fi