#!/bin/bash

# batch_sizes=(1 2 4 8)
# candidates_list=(128 256 512 1024)
batch_sizes=(8)
candidates_list=(1024)

for bs in "${batch_sizes[@]}"; do
    for nc in "${candidates_list[@]}"; do
        echo "----------------------------------------------------------"
        echo "Running: Batch Size = $bs, Num Candidates = $nc"
        echo "----------------------------------------------------------"
        
        mkdir -p p-logs
        mkdir -p profiles
        rm -rf p-logs/*
        rm -rf profiles/*

        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
            nsys profile --trace=cuda,nvtx -o profiles/benchmark_bs"${bs}"_nc"${nc}" \
            python3 ./inference/benchmark/inference_benchmark.py --batch_size "$bs" --num_candidates "$nc" > p-logs/benchmark_bs"${bs}"_nc"${nc}".log 2>&1
        
        echo -e "\n"
    done
done

echo "All benchmarks completed!"