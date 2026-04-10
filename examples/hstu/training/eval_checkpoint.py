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
"""
Evaluate a training checkpoint using the training code's evaluation pipeline.

This script:
1. Loads a training checkpoint
2. Runs evaluation using the training code's evaluate() function
3. Reports metrics (AUC, NDCG, HR, etc.)

Supports both ranking and retrieval tasks based on eval_metrics:
- If eval_metrics contains "HR" or "NDCG", it's a retrieval task
- Otherwise, it's a ranking task

Usage:
    cd examples/hstu
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6000 ./training/eval_checkpoint.py \
        --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
        --ckpt-load-dir ckpt_kr_1k_ranking/iter550
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import argparse
import os
from typing import List, Union

import commons.utils.initialize as init
import gin
import torch
from commons.utils.logger import print_rank_0
from configs import RankingConfig, RetrievalConfig
from distributed.sharding import make_optimizer_and_shard
from megatron.core import parallel_state
from model import get_ranking_model, get_retrieval_model
from modules.metrics import RetrievalTaskMetricWithSampling, get_multi_event_metric_module
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from commons.checkpoint import get_unwrapped_module
from trainer.training import evaluate, maybe_load_ckpts
from trainer.utils import (
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
    get_data_loader,
    get_dataset_and_embedding_args,
    get_embedding_vector_storage_multiplier,
)
from utils import (
    DatasetArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    RankingArgs,
    RetrievalArgs,
    TensorModelParallelArgs,
    TrainerArgs,
)


def is_retrieval_task() -> bool:
    """Check if the task is retrieval based on whether RankingArgs can be instantiated."""
    try:
        ranking_args = RankingArgs()
        # If prediction_head_arch is set, it's a ranking task
        return ranking_args.prediction_head_arch is None
    except AssertionError:
        # RankingArgs fails to instantiate because prediction_head_arch is not configured
        # This means it's a retrieval task
        return True


def create_ranking_config(
    dataset_args,
    network_args,
    embedding_args,
) -> RankingConfig:
    ranking_args = RankingArgs()
    return RankingConfig(
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )


def create_retrieval_config(
    dataset_args,
    network_args,
    embedding_args,
) -> RetrievalConfig:
    retrieval_args = RetrievalArgs()
    return RetrievalConfig(
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        temperature=retrieval_args.temperature,
        l2_norm_eps=retrieval_args.l2_norm_eps,
        num_negatives=retrieval_args.num_negatives,
        eval_metrics=retrieval_args.eval_metrics,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Training Checkpoint")
    parser.add_argument("--gin-config-file", type=str, required=True)
    parser.add_argument("--ckpt-load-dir", type=str, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-eval-iters", type=int, default=None)
    parser.add_argument("--max-retrieval-items", type=int, default=500,
                        help="Maximum number of items to use for retrieval evaluation. Default: 500")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results CSV file")
    args = parser.parse_args()
    
    gin.parse_config_file(args.gin_config_file)

    trainer_args = TrainerArgs()
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()

    # Override from command line
    trainer_args.ckpt_load_dir = args.ckpt_load_dir
    if args.eval_batch_size is not None:
        trainer_args.eval_batch_size = args.eval_batch_size
    if args.max_eval_iters is not None:
        trainer_args.max_eval_iters = args.max_eval_iters

    # Determine task type based on whether prediction_head_arch is set
    is_retrieval = is_retrieval_task()
    
    if is_retrieval:
        retrieval_args = RetrievalArgs()
        eval_metrics = retrieval_args.eval_metrics
    else:
        ranking_args = RankingArgs()
        eval_metrics = ranking_args.eval_metrics

    print_rank_0(f"Task type: {'Retrieval' if is_retrieval else 'Ranking'}")
    print_rank_0(f"Eval metrics: {eval_metrics}")

    # Initialize distributed
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)

    print_rank_0(f"Checkpoint dir: {trainer_args.ckpt_load_dir}")
    print_rank_0(f"Eval batch size: {trainer_args.eval_batch_size}")
    print_rank_0(f"Max eval iters: {trainer_args.max_eval_iters}")

    # Create model and config based on task type
    hstu_config = create_hstu_config(network_args, tp_args)
    
    if is_retrieval:
        task_config = create_retrieval_config(dataset_args, network_args, embedding_args)
        model = get_retrieval_model(hstu_config=hstu_config, task_config=task_config)
    else:
        task_config = create_ranking_config(dataset_args, network_args, embedding_args)
        model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)

    # Create optimizer
    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )
    optimizer_param = create_optimizer_params(optimizer_args)
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )

    # Create metric module based on task type
    if is_retrieval:
        # Use max_retrieval_items as the candidate pool size for retrieval evaluation
        stateful_metric_module = RetrievalTaskMetricWithSampling(
            metric_types=task_config.eval_metrics, MAX_K=args.max_retrieval_items
        )
        print_rank_0(f"Max retrieval items (MAX_K): {args.max_retrieval_items}")
    else:
        stateful_metric_module = get_multi_event_metric_module(
            num_classes=task_config.prediction_head_arch[-1],
            num_tasks=task_config.num_tasks,
            metric_types=task_config.eval_metrics,
            comm_pg=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )

    # Get eval dataloader based on task type
    if is_retrieval:
        _, eval_dataloader = get_data_loader(
            "retrieval", dataset_args, trainer_args, 0
        )
    else:
        _, eval_dataloader = get_data_loader(
            "ranking", dataset_args, trainer_args, task_config.num_tasks
        )

    # Load checkpoint
    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)

    # Create pipeline
    if trainer_args.pipeline_type in ["prefetch", "native"]:
        pipeline_factory = (
            JaggedMegatronPrefetchTrainPipelineSparseDist
            if trainer_args.pipeline_type == "prefetch"
            else JaggedMegatronTrainPipelineSparseDist
        )
        pipeline = pipeline_factory(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    else:
        pipeline = JaggedMegatronTrainNonePipeline(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )

    # Run evaluation
    print_rank_0("Starting evaluation...")
    pipeline._model.eval()
    evaluate(
        pipeline,
        stateful_metric_module,
        trainer_args=trainer_args,
        eval_loader=eval_dataloader,
    )

    # Get metrics after evaluation
    if is_retrieval:
        retrieval_gr = get_unwrapped_module(pipeline._model)
        export_table_name = retrieval_gr.get_item_feature_table_name()
        eval_metric_dict, _, _ = stateful_metric_module.compute(
            *retrieval_gr._embedding_collection.export_local_embedding(
                export_table_name
            ),
        )
    else:
        eval_metric_dict = stateful_metric_module.compute()

    # Save results to CSV
    if args.output_file:
        results = {}
        results.update({k: v.item() if torch.is_tensor(v) else v for k, v in eval_metric_dict.items()})
        
        if os.path.exists(args.output_file):
            results_df = pd.read_csv(args.output_file)
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        else:
            results_df = pd.DataFrame([results])
        results_df.to_csv(args.output_file, index=False)
        print_rank_0(f"Results saved to {args.output_file}")

    init.destroy_global_state()


if __name__ == "__main__":
    main()