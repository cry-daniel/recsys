# Attention visualization
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
    torchrun --nproc_per_node 1 --master_addr localhost \
    --master_port 6000 ./training/eval_checkpoint_analysis.py \
    --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
    --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 \
    --analysis attention_viz \
    --output-dir ./analysis_output

# KV Cache difference analysis
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
    torchrun --nproc_per_node 1 --master_addr localhost \
    --master_port 6000 ./training/eval_checkpoint_analysis.py \
    --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
    --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 \
    --analysis kv_cache_diff \
    --output-dir ./analysis_output

# KV Cache replacement evaluation
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
    torchrun --nproc_per_node 1 --master_addr localhost \
    --master_port 6000 ./training/eval_checkpoint_analysis.py \
    --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
    --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 \
    --analysis kv_cache_replace \
    --output-dir ./analysis_output
