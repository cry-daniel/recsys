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
        
        mkdir -p pcu-logs
        mkdir -p profiles-cu
        rm -rf pcu-logs/*
        rm -rf profiles-cu/*
        
        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
            ncu --kernel-name initialize_with_index_addressor_kernel \
            --launch-skip 59 --launch-count 1 \
            --set roofline \
            -o profiles-cu/benchmark_bs"${bs}"_nc"${nc}"_embedding -f \
            python3 ./inference/benchmark/inference_benchmark.py \
            --batch_size "$bs" --num_candidates "$nc" \
            > pcu-logs/benchmark_bs"${bs}"_nc"${nc}"_embedding.log 2>&1

        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
            ncu --kernel-name ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn \
            --launch-skip 59 --launch-count 1 \
            --set roofline \
            -o profiles-cu/benchmark_bs"${bs}"_nc"${nc}"_tc -f \
            python3 ./inference/benchmark/inference_benchmark.py \
            --batch_size "$bs" --num_candidates "$nc" \
            > pcu-logs/benchmark_bs"${bs}"_nc"${nc}"_tc.log 2>&1

        CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
            ncu --kernel-name hstu_fwd_kernel \
            --launch-skip 472 --launch-count 1 \
            --set roofline \
            -o profiles-cu/benchmark_bs"${bs}"_nc"${nc}"_hstu_fwd -f \
            python3 ./inference/benchmark/inference_benchmark.py \
            --batch_size "$bs" --num_candidates "$nc" \
            > pcu-logs/benchmark_bs"${bs}"_nc"${nc}"_hstu_fwd.log 2>&1
        
        echo -e "\n"
    done
done

echo "All benchmarks completed!"