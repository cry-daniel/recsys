# docker run --gpus all -it --rm \
#     --name gr \
#     -v /home/ruiyang.chen/Code/GR/recsys:/workspace/recsys -w /workspace/recsys \
#    recsys-examples:latest

# PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 2 --master_addr localhost --master_port 6000  ./training/pretrain_gr_ranking.py --gin-config-file ./training/configs/kuairand_1
# k_ranking_tp2.gin

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node=1 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ckpts/iter550/ --mode eval

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node=1 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --checkpoint_dir ckpts/iter550/ --mode simulate

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=${PYTHONPATH}:$(realpath ../) python3 ./inference/benchmark/inference_benchmark.py