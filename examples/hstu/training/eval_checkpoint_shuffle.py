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
Evaluate a training checkpoint with shuffled user sequences.

This script:
1. Loads a training checkpoint
2. Shuffles the first shuffle_pct% of each user's item-action sequence
3. Runs evaluation using the training code's evaluate() function
4. Reports metrics (AUC, NDCG, HR, etc.)

Supports both ranking and retrieval tasks based on eval_metrics:
- If eval_metrics contains "HR" or "NDCG", it's a retrieval task
- Otherwise, it's a ranking task

Usage:
    cd examples/hstu
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6000 ./training/eval_checkpoint_shuffle.py \
        --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
        --ckpt-load-dir ckpt_kr_1k_ranking/iter550 \
        --shuffle_pct 80
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import argparse
import os
from typing import List, Union

import commons.utils.initialize as init
import gin
import numpy as np
import pandas as pd
import torch
from commons.utils.logger import print_rank_0
from configs import RankingConfig, RetrievalConfig
from dataset.sequence_dataset import SequenceDataset, load_seq
from distributed.sharding import make_optimizer_and_shard
from megatron.core import parallel_state
from model import get_ranking_model, get_retrieval_model
from modules.metrics import RetrievalTaskMetricWithSampling, get_multi_event_metric_module
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from preprocessor import get_common_preprocessors
from commons.checkpoint import get_unwrapped_module
from trainer.training import evaluate, maybe_load_ckpts
from trainer.utils import (
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
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


def normalize_item_actions(
    user_sequences,
    user_action_sequences,
):
    """
    Normalize actions so that each item has the same action across all users.
    
    For each unique item, the action from its first occurrence (across all users,
    in order of user_id) is used. All subsequent occurrences of that item will
    have their action replaced with the first occurrence's action.
    
    Args:
        user_sequences: Dict mapping user_id to item sequence
        user_action_sequences: Dict mapping user_id to action sequence
    
    Returns:
        Dict mapping user_id to normalized action sequence
    """
    # Build item -> first action mapping
    item_first_action = {}
    
    # Process users in sorted order to ensure deterministic first occurrence
    for uid in sorted(user_sequences.keys()):
        item_seq = user_sequences[uid]
        action_seq = user_action_sequences[uid]
        
        for item, action in zip(item_seq, action_seq):
            if item not in item_first_action:
                item_first_action[item] = action
    
    print_rank_0(f"Found {len(item_first_action)} unique items with actions")
    
    # Apply normalization to all user sequences
    normalized_action_sequences = {}
    for uid in user_action_sequences:
        item_seq = user_sequences[uid]
        action_seq = user_action_sequences[uid]
        
        normalized_actions = [item_first_action.get(item, action) 
                            for item, action in zip(item_seq, action_seq)]
        normalized_action_sequences[uid] = normalized_actions
    
    return normalized_action_sequences


def generate_shuffle_order(
    seq_len: int,
    shuffle_pct: float,
    rng: np.random.Generator,
) -> list:
    """
    Generate a shuffle order for a sequence.
    
    Only the first shuffle_pct% of items are shuffled.
    The remaining items keep their original order.

    Args:
        seq_len: Total sequence length
        shuffle_pct: Percentage of items to shuffle (0.0 to 100.0)
        rng: NumPy random generator

    Returns:
        List of indices representing the new order
    """
    # Handle 0% shuffle - return original order
    if shuffle_pct <= 0:
        return list(range(seq_len))
    
    num_to_shuffle = int(seq_len * shuffle_pct / 100.0)
    num_to_shuffle = max(1, min(num_to_shuffle, seq_len))

    if num_to_shuffle <= 1:
        return list(range(seq_len))

    shuffle_indices = rng.permutation(num_to_shuffle).tolist()
    remaining_indices = list(range(num_to_shuffle, seq_len))

    return shuffle_indices + remaining_indices


def is_retrieval_task() -> bool:
    """Check if the task is retrieval based on whether RankingArgs can be instantiated."""
    try:
        ranking_args = RankingArgs()
        return ranking_args.prediction_head_arch is None
    except AssertionError:
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


def get_eval_dataloader_with_shuffle(
    dataset_args,
    task_config,
    trainer_args,
    shuffle_pct: float,
    random_seed: int = 42,
    is_retrieval: bool = False,
    normalize_item_action: bool = False,
):
    """Create eval dataloader with shuffled user sequences."""
    common_preprocessors = get_common_preprocessors("")[dataset_args.dataset_name]
    dp = common_preprocessors
    
    # Read user sequences
    print_rank_0("Reading user sequences...")
    seq_logs_frame = pd.read_csv(dp._output_file, delimiter=",")
    
    user_sequences = {}
    user_action_sequences = {}
    for _, row in seq_logs_frame.iterrows():
        uid = row["user_id"]
        item_seq = load_seq(row[dp._item_feature_name])
        action_seq = load_seq(row[dp._action_feature_name])
        if uid not in user_sequences:
            user_sequences[uid] = item_seq
            user_action_sequences[uid] = action_seq

    print_rank_0(f"Found {len(user_sequences)} unique users")
    avg_seq_len = sum(len(seq) for seq in user_sequences.values()) / len(user_sequences)
    print_rank_0(f"Average sequence length: {avg_seq_len:.1f}")

    # Normalize item actions if requested
    item_action_norm = None
    if normalize_item_action:
        print_rank_0("Normalizing item actions across users...")
        normalized_action_sequences = normalize_item_actions(
            user_sequences, user_action_sequences
        )
        # Build flat dict from all users for SequenceDataset
        item_action_norm = {}
        for uid in sorted(user_sequences.keys()):
            item_seq = user_sequences[uid]
            action_seq = normalized_action_sequences[uid]
            for item, action in zip(item_seq, action_seq):
                if item not in item_action_norm:
                    item_action_norm[item] = action
        print_rank_0("Item action normalization complete.")

    # Generate shuffle orders for all users
    print_rank_0(f"Generating shuffle orders (shuffle_pct={shuffle_pct}%)...")
    user_sequence_orders = {}
    rng = np.random.default_rng(random_seed)
    
    for uid, item_seq in user_sequences.items():
        seq_len = len(item_seq)
        order = generate_shuffle_order(
            seq_len=seq_len,
            shuffle_pct=shuffle_pct,
            rng=rng,
        )
        user_sequence_orders[uid] = order
    
    print_rank_0("Shuffle order generation complete.")

    # Create dataset with shuffle orders
    eval_dataset = SequenceDataset(
        seq_logs_file=dp._output_file,
        batch_size=trainer_args.eval_batch_size,
        max_seqlen=dataset_args.max_sequence_length,
        item_feature_name=dp._item_feature_name,
        contextual_feature_names=dp._contextual_feature_names,
        action_feature_name=dp._action_feature_name,
        max_num_candidates=dataset_args.max_num_candidates if not is_retrieval else 0,
        num_tasks=0 if is_retrieval else 1,
        rank=0,
        world_size=1,
        shuffle=False,
        random_seed=0,
        is_train_dataset=False,
        user_sequence_orders=user_sequence_orders,
        userid_name="user_id",
        item_action_normalization=item_action_norm,
    )

    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=True,
    )
    
    return eval_dataloader


def main():
    parser = argparse.ArgumentParser(description="Evaluate Training Checkpoint with Shuffle")
    parser.add_argument("--gin-config-file", type=str, required=True)
    parser.add_argument("--ckpt-load-dir", type=str, required=True)
    parser.add_argument("--shuffle_pct", type=float, default=80.0,
                        help="Percentage of sequence to shuffle (0-100)")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results CSV file")
    parser.add_argument("--max-retrieval-items", type=int, default=500,
                        help="Maximum number of items to use for retrieval evaluation. Default: 500")
    parser.add_argument("--normalize_item_action", action="store_true",
                        help="Normalize item actions so same item has same action across all users")
    args = parser.parse_args()
    
    gin.parse_config_file(args.gin_config_file)

    trainer_args = TrainerArgs()
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()

    # Override from command line
    trainer_args.ckpt_load_dir = args.ckpt_load_dir

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
    print_rank_0(f"Shuffle pct: {args.shuffle_pct}%")
    print_rank_0(f"Random seed: {args.random_seed}")
    print_rank_0(f"Normalize item action: {args.normalize_item_action}")

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

    # Get eval dataloader with shuffle
    eval_dataloader = get_eval_dataloader_with_shuffle(
        dataset_args, task_config, trainer_args,
        shuffle_pct=args.shuffle_pct,
        random_seed=args.random_seed,
        is_retrieval=is_retrieval,
        normalize_item_action=args.normalize_item_action,
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

    # Run evaluation inline (to capture metrics)
    print_rank_0("Starting evaluation...")
    pipeline._model.eval()
    
    from itertools import islice
    from commons.utils.stringify import stringify_dict
    
    eval_iter = 0
    max_eval_iters = trainer_args.max_eval_iters or len(eval_dataloader)
    max_eval_iters = min(max_eval_iters, len(eval_dataloader))
    iterated_eval_loader = islice(eval_dataloader, len(eval_dataloader))
    
    with torch.no_grad():
        for i in range(max_eval_iters):
            eval_iter += 1
            reporting_loss, (_, logits, labels, _) = pipeline.progress(
                iterated_eval_loader
            )
            stateful_metric_module(logits, labels)
    
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
    
    dp_size = parallel_state.get_data_parallel_world_size()
    print_rank_0(
        f"[eval] [eval {eval_iter * dp_size * trainer_args.eval_batch_size} users]:\n    "
        + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
    )

    # Save results to CSV
    if args.output_file:
        results = {"shuffle_pct": args.shuffle_pct, "normalize_item_action": args.normalize_item_action}
        results.update({k: v.item() if torch.is_tensor(v) else v for k, v in eval_metric_dict.items()})
        
        # Append to CSV
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