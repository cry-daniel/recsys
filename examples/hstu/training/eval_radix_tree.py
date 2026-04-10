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
Evaluate a training checkpoint with radix tree reordering.

This script:
1. Loads a training checkpoint
2. Builds a radix tree from user sequences
3. Generates reorder based on radix tree traversal
4. Computes fine-grained radix tree metrics
5. Runs evaluation and reports model metrics

Usage:
    cd examples/hstu
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6000 ./training/eval_radix_tree.py \
        --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
        --ckpt-load-dir ckpt_kr_1k_ranking/iter550 \
        --radix_tree_strategy prefix_score
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import argparse
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import commons.utils.initialize as init
import gin
import numpy as np
import pandas as pd
import torch
from commons.utils.logger import print_rank_0
from commons.utils.stringify import stringify_dict
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


# ============================================================================
# Radix Tree Implementation (Optimized)
# ============================================================================

class RadixTreeNode:
    """A node in the radix tree."""
    __slots__ = ['item', 'children', 'count']
    
    def __init__(self, item=None):
        self.item = item
        self.children = {}
        self.count = 0  # Number of sequences passing through this node
    
    def __repr__(self):
        return f"RadixTreeNode(item={self.item}, count={self.count})"


class RadixTree:
    """Radix tree for storing user sequences."""
    def __init__(self):
        self.root = RadixTreeNode()
        self.item_frequency = Counter()
        self.item_max_depth = {}  # Cache: item -> max depth
    
    def insert(self, sequence: List):
        """Insert a sequence into the radix tree."""
        node = self.root
        node.count += 1
        for depth, item in enumerate(sequence, 1):
            if item not in node.children:
                node.children[item] = RadixTreeNode(item)
            node = node.children[item]
            node.count += 1
            self.item_frequency[item] += 1
            # Update max depth for item
            if item not in self.item_max_depth or depth > self.item_max_depth[item]:
                self.item_max_depth[item] = depth
    
    def get_item_score(self, item: int) -> int:
        """Get the score for an item (frequency-based)."""
        return self.item_frequency.get(item, 0)
    
    def get_item_depth(self, item: int) -> int:
        """Get the cached max depth for an item."""
        return self.item_max_depth.get(item, 0)


def build_radix_tree(user_sequences: Dict) -> RadixTree:
    """Build a radix tree from user sequences."""
    tree = RadixTree()
    for uid, seq in user_sequences.items():
        tree.insert(seq)
    return tree


def build_radix_tree_unordered(user_sequences: Dict) -> RadixTree:
    """Build an unordered radix tree (items sorted by frequency before insertion)."""
    # First pass: compute global item frequencies
    item_freq = Counter()
    for uid, seq in user_sequences.items():
        item_freq.update(seq)
    
    # Build tree with sequences sorted by frequency
    tree = RadixTree()
    tree.item_frequency = item_freq
    for uid, seq in user_sequences.items():
        # Precompute item positions for O(1) lookup
        item_positions = defaultdict(list)
        for pos, item in enumerate(seq):
            item_positions[item].append(pos)
        
        # Sort items by frequency (descending), then by first occurrence for ties
        sorted_seq = sorted(
            set(seq),
            key=lambda x: (-item_freq.get(x, 0), item_positions[x][0])
        )
        # Expand back to original length maintaining frequency order
        full_sorted_seq = []
        for item in sorted_seq:
            full_sorted_seq.extend([item] * len(item_positions[item]))
        
        tree.insert(full_sorted_seq)
    return tree


def generate_order_from_tree(
    tree: RadixTree,
    sequence: List,
    strategy: str = "prefix_score",
    shuffle_pct: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Generate a reorder for a sequence based on radix tree traversal.
    
    Uses cached item depths for O(1) lookup instead of O(tree_size) search.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    seq_len = len(sequence)
    num_to_reorder = max(1, int(seq_len * shuffle_pct / 100.0))
    num_to_reorder = min(num_to_reorder, seq_len)
    
    if num_to_reorder <= 1:
        return list(range(seq_len))
    
    # Get the prefix of the sequence to reorder
    prefix = sequence[:num_to_reorder]
    
    # Score each item in the prefix based on strategy (O(1) lookup with cache)
    item_scores = []
    for i, item in enumerate(prefix):
        if strategy == "prefix_score":
            score = tree.get_item_depth(item)
        elif strategy == "frequency":
            score = tree.get_item_score(item)
        elif strategy == "hybrid":
            depth_score = tree.get_item_depth(item)
            freq_score = tree.get_item_score(item)
            score = depth_score + freq_score
        else:
            score = 0
        item_scores.append((i, score))
    
    # Sort by score (descending), with random tie-breaking
    item_scores.sort(key=lambda x: (-x[1], rng.random()))
    
    # Generate the reorder
    reorder = [idx for idx, _ in item_scores]
    
    # Add remaining items in original order
    remaining = list(range(num_to_reorder, seq_len))
    
    return reorder + remaining


# ============================================================================
# Radix Tree Metrics (Optimized with sampling)
# ============================================================================

def compute_prefix_sharing_rate(
    reordered_seqs: List[List],
    prefix_lengths: List[int] = [5, 10, 20, 50],
) -> Dict[int, float]:
    """
    Compute the prefix sharing rate at different prefix lengths.
    """
    results = {}
    
    for plen in prefix_lengths:
        # Filter sequences long enough
        valid_prefixes = [tuple(seq[:plen]) for seq in reordered_seqs if len(seq) >= plen]
        
        if not valid_prefixes:
            results[plen] = 0.0
            continue
        
        # Count unique prefixes using numpy for speed
        unique_prefixes = len(set(valid_prefixes))
        total_prefixes = len(valid_prefixes)
        
        sharing_rate = 1.0 - (unique_prefixes / total_prefixes)
        results[plen] = sharing_rate
    
    return results


def compute_lcp_distribution(
    reordered_seqs: List[List],
) -> Dict[str, float]:
    """
    Compute the Longest Common Prefix (LCP) distribution between adjacent users.
    """
    if len(reordered_seqs) < 2:
        return {"mean_lcp": 0.0, "max_lcp": 0, "median_lcp": 0.0}
    
    # Sort sequences for better adjacency comparison
    sorted_seqs = sorted(reordered_seqs, key=lambda x: tuple(x[:10]))
    
    # Compute LCP between adjacent sequences using numpy
    lcp_values = np.zeros(len(sorted_seqs) - 1, dtype=np.int32)
    
    for i in range(len(sorted_seqs) - 1):
        seq1, seq2 = sorted_seqs[i], sorted_seqs[i + 1]
        min_len = min(len(seq1), len(seq2))
        # Use numpy for faster comparison
        arr1 = np.array(seq1[:min_len])
        arr2 = np.array(seq2[:min_len])
        lcp_values[i] = np.argmax(arr1 != arr2) if np.any(arr1 != arr2) else min_len
    
    return {
        "mean_lcp": float(np.mean(lcp_values)),
        "max_lcp": int(np.max(lcp_values)),
        "median_lcp": float(np.median(lcp_values)),
    }


def compute_topk_overlap(
    reordered_seqs: List[List],
    topk_values: List[int] = [5, 10, 20, 50],
    max_pairs: int = 1000,
) -> Dict[int, float]:
    """
    Compute the top-K item overlap (Jaccard similarity) across users.
    """
    results = {}
    n = len(reordered_seqs)
    
    if n < 2:
        for k in topk_values:
            results[k] = 0.0
        return results
    
    rng = np.random.default_rng(42)
    actual_pairs = min(max_pairs, n * (n - 1) // 2)
    
    for k in topk_values:
        jaccard_scores = []
        pairs_sampled = 0
        
        while pairs_sampled < actual_pairs:
            i, j = rng.choice(n, 2, replace=False)
            items1 = set(reordered_seqs[i][:k])
            items2 = set(reordered_seqs[j][:k])
            
            if not items1 or not items2:
                pairs_sampled += 1
                continue
            
            intersection = len(items1 & items2)
            union = len(items1 | items2)
            
            if union > 0:
                jaccard_scores.append(intersection / union)
            
            pairs_sampled += 1
        
        results[k] = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    
    return results


def compute_radix_tree_metrics(
    user_sequences: Dict,
    user_sequence_orders: Dict,
    sample_size: Optional[int] = None,
) -> Dict:
    """
    Compute all radix tree related metrics.
    
    Args:
        user_sequences: Original user sequences
        user_sequence_orders: Reorder for each user
        sample_size: If set, sample this many users for metrics computation
    
    Returns:
        Dict with all metrics
    """
    # Get reordered sequences
    reordered_seqs = []
    uids = list(user_sequences.keys())
    
    if sample_size and len(uids) > sample_size:
        rng = np.random.default_rng(42)
        sampled_uids = rng.choice(uids, sample_size, replace=False).tolist()
    else:
        sampled_uids = uids
    
    for uid in sampled_uids:
        seq = user_sequences[uid]
        order = user_sequence_orders.get(uid, list(range(len(seq))))
        reordered = [seq[i] for i in order if i < len(seq)]
        reordered_seqs.append(reordered)
    
    metrics = {}
    
    # Prefix sharing rates
    prefix_lengths = [5, 10, 20, 50]
    prefix_sharing = compute_prefix_sharing_rate(reordered_seqs, prefix_lengths)
    for plen, rate in prefix_sharing.items():
        metrics[f"prefix_sharing_rate@{plen}"] = rate
    
    # LCP distribution
    lcp_stats = compute_lcp_distribution(reordered_seqs)
    metrics["mean_lcp"] = lcp_stats["mean_lcp"]
    metrics["max_lcp"] = lcp_stats["max_lcp"]
    metrics["median_lcp"] = lcp_stats["median_lcp"]
    
    # Top-K overlap
    topk_values = [5, 10, 20, 50]
    topk_overlap = compute_topk_overlap(reordered_seqs, topk_values)
    for k, overlap in topk_overlap.items():
        metrics[f"topk_overlap@{k}"] = overlap
    
    return metrics


# ============================================================================
# Task Type Detection
# ============================================================================

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


def get_eval_dataloader_with_radix_order(
    dataset_args,
    task_config,
    trainer_args,
    user_sequence_orders: Dict,
):
    """Create eval dataloader with radix tree reorder."""
    common_preprocessors = get_common_preprocessors("")[dataset_args.dataset_name]
    dp = common_preprocessors
    
    is_retrieval = is_retrieval_task()
    
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
    parser = argparse.ArgumentParser(description="Evaluate Training Checkpoint with Radix Tree")
    parser.add_argument("--gin-config-file", type=str, required=True)
    parser.add_argument("--ckpt-load-dir", type=str, required=True)
    parser.add_argument("--shuffle_pct", type=float, default=100.0,
                        help="Percentage of sequence to reorder (0-100)")
    parser.add_argument("--radix_tree_strategy", type=str, default="prefix_score",
                        choices=["prefix_score", "frequency", "hybrid", "unordered_tree"])
    parser.add_argument("--unordered", action="store_true",
                        help="Use unordered radix tree mode")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results CSV file")
    parser.add_argument("--max-retrieval-items", type=int, default=500,
                        help="Maximum number of items to use for retrieval evaluation. Default: 500")
    parser.add_argument("--metrics-sample-size", type=int, default=None,
                        help="Number of users to sample for radix tree metrics. None = all users")
    args = parser.parse_args()
    
    gin.parse_config_file(args.gin_config_file)

    trainer_args = TrainerArgs()
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()

    # Override from command line
    trainer_args.ckpt_load_dir = args.ckpt_load_dir

    # Determine task type
    is_retrieval = is_retrieval_task()
    
    if is_retrieval:
        retrieval_args = RetrievalArgs()
        eval_metrics = retrieval_args.eval_metrics
    else:
        ranking_args = RankingArgs()
        eval_metrics = ranking_args.eval_metrics

    print_rank_0(f"Task type: {'Retrieval' if is_retrieval else 'Ranking'}")
    print_rank_0(f"Eval metrics: {eval_metrics}")
    print_rank_0(f"Radix tree strategy: {args.radix_tree_strategy}")
    print_rank_0(f"Unordered: {args.unordered}")
    print_rank_0(f"Shuffle pct: {args.shuffle_pct}%")
    print_rank_0(f"Metrics sample size: {args.metrics_sample_size or 'all'}")

    # Initialize distributed
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)

    print_rank_0(f"Checkpoint dir: {trainer_args.ckpt_load_dir}")

    # Read user sequences
    print_rank_0("Reading user sequences...")
    t0 = time.time()
    common_preprocessors = get_common_preprocessors("")[dataset_args.dataset_name]
    dp = common_preprocessors
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
    print_rank_0(f"Reading sequences took: {time.time() - t0:.2f}s")

    # Build radix tree
    print_rank_0("Building radix tree...")
    t0 = time.time()
    if args.unordered:
        tree = build_radix_tree_unordered(user_sequences)
    else:
        tree = build_radix_tree(user_sequences)
    
    print_rank_0(f"Radix tree built in {time.time() - t0:.2f}s. Unique items: {len(tree.item_frequency)}")
    print_rank_0(f"Top 10 most frequent items: {tree.item_frequency.most_common(10)}")

    # Generate reorder from radix tree
    print_rank_0(f"Generating reorder from radix tree (strategy={args.radix_tree_strategy})...")
    t0 = time.time()
    user_sequence_orders = {}
    rng = np.random.default_rng(args.random_seed)
    
    for uid, seq in user_sequences.items():
        order = generate_order_from_tree(
            tree=tree,
            sequence=seq,
            strategy=args.radix_tree_strategy,
            shuffle_pct=args.shuffle_pct,
            rng=rng,
        )
        user_sequence_orders[uid] = order
    
    print_rank_0(f"Reorder generation took: {time.time() - t0:.2f}s")

    # Compute radix tree metrics
    print_rank_0("\n" + "=" * 60)
    print_rank_0("RADIX TREE METRICS")
    print_rank_0("=" * 60)
    
    t0 = time.time()
    radix_metrics = compute_radix_tree_metrics(
        user_sequences, user_sequence_orders,
        sample_size=args.metrics_sample_size
    )
    print_rank_0(f"Metrics computation took: {time.time() - t0:.2f}s")
    print_rank_0(stringify_dict(radix_metrics, prefix="Radix Tree Metrics", sep="\n    "))

    # Create model
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

    # Create metric module
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

    # Get eval dataloader
    eval_dataloader = get_eval_dataloader_with_radix_order(
        dataset_args, task_config, trainer_args,
        user_sequence_orders=user_sequence_orders,
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
    print_rank_0("\n" + "=" * 60)
    print_rank_0("Starting evaluation...")
    print_rank_0("=" * 60)
    pipeline._model.eval()
    evaluate(
        pipeline,
        stateful_metric_module,
        trainer_args=trainer_args,
        eval_loader=eval_dataloader,
    )

    # Save results
    if args.output_file:
        results = {
            "strategy": args.radix_tree_strategy,
            "unordered": args.unordered,
            "shuffle_pct": args.shuffle_pct,
        }
        results.update(radix_metrics)
        
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