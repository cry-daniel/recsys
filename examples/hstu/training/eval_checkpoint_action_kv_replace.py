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
Evaluate a training checkpoint with advanced analysis capabilities.

This script provides three analysis modes:
1. attention_viz: Visualize attention maps for action tokens
2. kv_cache_diff: Analyze KV Cache differences for specific actions across positions
3. kv_cache_replace: Reuse repeated action KV within local top-K frequency windows

Usage:
    cd examples/hstu
    # Attention visualization
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6000 ./training/eval_checkpoint_analysis.py \
        --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
        --ckpt-load-dir ckpt_kr_1k_ranking/iter550 \
        --analysis attention_viz \
        --output-dir ./analysis_output

    # KV Cache difference analysis
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6000 ./training/eval_checkpoint_analysis.py \
        --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
        --ckpt-load-dir ckpt_kr_1k_ranking/iter550 \
        --analysis kv_cache_diff \
        --output-dir ./analysis_output

    # Window/top-K KV Cache replacement evaluation
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6000 ./training/eval_checkpoint_action_kv_replace.py \
        --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
        --ckpt-load-dir ckpt_kr_1k_ranking/iter550 \
        --analysis kv_cache_replace \
        --kv-reuse-window-size 512 \
        --kv-reuse-top-k 3 \
        --output-dir ./analysis_output
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import argparse
import json
import os
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import commons.utils.initialize as init
import gin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from commons.utils.logger import print_rank_0
from configs import KernelBackend, RankingConfig, RetrievalConfig
from configs.hstu_config import HSTULayerType
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
from modules.hstu_attention import create_hstu_attention
from modules.jagged_data import JaggedData
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
        return ranking_args.prediction_head_arch is None
    except AssertionError:
        return True


def create_ranking_config(dataset_args, network_args, embedding_args) -> RankingConfig:
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


def create_retrieval_config(dataset_args, network_args, embedding_args) -> RetrievalConfig:
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


# ============================================================================
# Helper: Compute attention weights from Q, K
# ============================================================================

def compute_attention_weights(
    q: torch.Tensor,  # [T, num_heads, head_dim]
    k: torch.Tensor,  # [T, num_heads, head_dim]
    seqlen_offsets: torch.Tensor,  # [batch_size + 1]
    num_heads: int,
    attention_dim: int,
    is_causal: bool,
    num_contextuals: Optional[torch.Tensor] = None,
    num_targets: Optional[torch.Tensor] = None,
    target_group_size: int = 1,
    scaling_seqlen: int = -1,
) -> torch.Tensor:
    """
    Compute attention weights manually using PyTorch.
    Returns attention weights of shape [T, num_heads, T] (jagged format).
    """
    alpha = 1.0 / (attention_dim ** 0.5)
    if scaling_seqlen == -1:
        max_seqlen = (seqlen_offsets[1:] - seqlen_offsets[:-1]).max().item()
        scaling_seqlen = max_seqlen

    batch_size = len(seqlen_offsets) - 1
    total_len = q.shape[0]
    max_seqlen = (seqlen_offsets[1:] - seqlen_offsets[:-1]).max().item()

    # Pad Q, K to dense format
    L, H, D = q.shape
    V = k.shape[2]

    padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
        values=q.reshape(L, H * D),
        offsets=[seqlen_offsets],
        max_lengths=[max_seqlen],
        padding_value=0.0,
    ).view(-1, max_seqlen, H, D).transpose(1, 2)  # [B, H, N, D]

    padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
        values=k.reshape(L, H * D),
        offsets=[seqlen_offsets],
        max_lengths=[max_seqlen],
        padding_value=0.0,
    ).view(-1, max_seqlen, H, D).transpose(1, 2)  # [B, H, N, D]

    # Compute QK^T
    qk_attn = torch.einsum("bhxa,bhya->bhxy", padded_q, padded_k) * alpha
    qk_attn = F.silu(qk_attn) / scaling_seqlen

    # Build valid attention mask
    seq_lengths = seqlen_offsets[1:] - seqlen_offsets[:-1]
    valid_mask = _get_valid_attn_mask(
        device=q.device,
        causal=is_causal,
        N=max_seqlen,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        num_contextuals=num_contextuals if num_contextuals is not None else 0,
        target_group_size=target_group_size,
    )

    qk_attn = qk_attn * valid_mask.unsqueeze(1)

    # Convert back to jagged format
    # qk_attn: [B, H, N, N] -> we want [T, H, N] for each query position
    # For each sequence, extract the valid attention weights
    # Note: Different sequences may have different lengths, so we pad to max_seqlen
    jagged_attn_list = []
    for b in range(batch_size):
        seq_len = seq_lengths[b].item()
        if seq_len == 0:
            continue
        # [H, seq_len, seq_len] -> [seq_len, H, N] (pad to max_seqlen)
        seq_attn = qk_attn[b, :, :seq_len, :seq_len].transpose(0, 1).contiguous()  # [seq_len, H, seq_len]
        # Pad to max_seqlen in the last dimension
        if seq_len < max_seqlen:
            padding = torch.zeros(seq_len, num_heads, max_seqlen - seq_len, device=q.device, dtype=q.dtype)
            seq_attn = torch.cat([seq_attn, padding], dim=2)
        jagged_attn_list.append(seq_attn)

    if jagged_attn_list:
        return torch.cat(jagged_attn_list, dim=0)  # [T, H, N]
    else:
        return torch.zeros(0, num_heads, max_seqlen, device=q.device)


def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    num_contextuals = 0,
    target_group_size: int = 1,
):
    """Build valid attention mask."""
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)

    if isinstance(num_contextuals, int):
        if num_contextuals > 0:
            ids = ids - num_contextuals + 1
            ids = torch.clamp(ids, min=0)
            max_ids = max_ids - num_contextuals + 1
    else:
        ids = ids - num_contextuals.view(-1, 1) + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - num_contextuals.view(-1, 1, 1) + 1

    row_ids = ids.unsqueeze(-1).expand(-1, N, N)
    col_ids = row_ids.transpose(1, 2)
    row_col_dist = row_ids - col_ids

    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)

    if num_targets is not None:
        target_group_row_ids = torch.clamp(
            row_ids - max_ids + num_targets.view(-1, 1, 1), min=-1
        ) // target_group_size
        target_group_col_ids = target_group_row_ids.transpose(1, 2)
        target_dist = target_group_row_ids - target_group_col_ids
        target_group_mask = torch.logical_or(
            target_dist == 0, (target_group_row_ids < 0) + (target_group_col_ids < 0)
        )
        valid_attn_mask = torch.logical_and(valid_attn_mask, target_group_mask)
        max_ids = max_ids - num_targets.view(-1, 1, 1)

    if (isinstance(num_contextuals, int) and num_contextuals > 0) or isinstance(
        num_contextuals, torch.Tensor
    ):
        valid_attn_mask = torch.logical_or(
            valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
        )

    return valid_attn_mask


# ============================================================================
# Feature 1: Attention Map Visualization
# ============================================================================

class AttentionMapCollector:
    """Collects attention maps for visualization during evaluation."""

    def __init__(self, num_layers: int, num_heads: int, attention_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.collected_data: List[Dict] = []
        self._hooks = []
        self._max_batches = 3  # Maximum number of batches to collect
        self._token_ids: List[torch.Tensor] = []  # Store actual token IDs from input

    def register_hooks(self, model):
        """Register forward pre-hooks on HSTU attention modules to capture Q, K."""
        from modules.native_hstu_layer import HSTULayer
        from modules.fused_hstu_layer import FusedHSTULayer
        unwrapped = get_unwrapped_module(model)
        hstu_block = unwrapped._hstu_block
        layer_idx = 0
        for layer in hstu_block._attention_layers:
            if isinstance(layer, FusedHSTULayer):
                # For FusedHSTULayer, we compute Q, K, V manually from weights
                def make_hook(li):
                    def hook_fn(mod, args, kwargs):
                        return self._capture_qkv_fused(mod, args, li)
                    return hook_fn
                hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                self._hooks.append(hook)
            elif isinstance(layer, HSTULayer):
                # For NativeHSTULayer, use the native method
                def make_hook(li):
                    def hook_fn(mod, args, kwargs):
                        return self._capture_qkv(mod, args, li)
                    return hook_fn
                hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                self._hooks.append(hook)
            layer_idx += 1

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _capture_qkv(self, module, args, layer_idx):
        """Pre-hook to capture Q, K, V for NativeHSTULayer."""
        inputs = args
        if len(self.collected_data) >= self._max_batches * self.num_layers:
            return None  # Must return None to not modify inputs

        jd = inputs[0]
        if jd.values is None:
            return None

        hidden = jd.values

        # Compute Q, K, V using the layer's linear projection
        with torch.no_grad():
            eps = module._eps
            input_layernorm_weight = module._input_layernorm_weight
            input_layernorm_bias = module._input_layernorm_bias
            embedding_dim = module._embedding_dim

            if input_layernorm_weight is not None:
                normed_x = F.layer_norm(
                    hidden,
                    normalized_shape=[embedding_dim],
                    weight=input_layernorm_weight,
                    bias=input_layernorm_bias,
                    eps=eps,
                )
            else:
                normed_x = hidden

            mixed_uvqk, _ = module._linear_uvqk(normed_x)
            silu_uvqk = F.silu(mixed_uvqk)

            split_arg_list = module._split_arg_list
            num_heads_compute = module._num_heads_per_partition

            silu_uvqk = silu_uvqk.view(-1, num_heads_compute, sum(split_arg_list))
            user, value, query, key = torch.split(silu_uvqk, split_arg_list, dim=-1)

            # Store Q, K, V with metadata
            self.collected_data.append({
                "layer_idx": layer_idx,
                "query": query.detach().clone(),
                "key": key.detach().clone(),
                "value": value.detach().clone(),
                "seqlen_offsets": jd.seqlen_offsets.cpu().clone(),
                "num_candidates": jd.num_candidates.cpu().clone() if jd.num_candidates is not None else None,
                "contextual_seqlen": jd.contextual_seqlen.cpu().clone() if jd.contextual_seqlen is not None else None,
                "has_interleaved_action": jd.has_interleaved_action,
                "scaling_seqlen": jd.scaling_seqlen,
                "is_causal": module._attn_func.is_causal,
            })

        return None  # Must return None to not modify inputs

    def set_token_ids(self, item_ids: torch.Tensor, action_ids: torch.Tensor):
        """Store the actual item and action IDs from the batch input.
        
        Args:
            item_ids: Item feature values (flattened tensor, one per item token)
            action_ids: Action feature values (flattened tensor, one per action token)
        """
        if len(self._token_ids) < self._max_batches:
            # Store item and action IDs separately - they are already flattened
            # In the JaggedData, items are at even positions (0, 2, 4...) and actions at odd (1, 3, 5...)
            self._token_ids.append({
                "item_ids": item_ids.cpu().clone(),
                "action_ids": action_ids.cpu().clone(),
            })

    def _get_token_id_at(self, user_idx: int, pos: int) -> int:
        """Get the token ID at a specific position in the interleaved sequence.
        
        Args:
            user_idx: Index of the user in the batch
            pos: Position in the sequence (0-based within user's sequence)
            
        Returns:
            The token ID at that position, or -1 if not available
        """
        if not self._token_ids or not self.collected_data:
            return -1
        
        sample = self.collected_data[0]
        seqlen_offsets = sample["seqlen_offsets"]
        
        # Calculate which item/action index this position corresponds to
        # In interleaved sequence: pos 0 = item 0, pos 1 = action 0, pos 2 = item 1, pos 3 = action 1, ...
        item_or_action_idx = pos // 2
        is_action = pos % 2 == 1
        
        for batch_data in self._token_ids:
            item_ids = batch_data.get("item_ids")
            action_ids = batch_data.get("action_ids")
            
            if is_action:
                # This is an action position
                if action_ids is not None and item_or_action_idx < action_ids.shape[0]:
                    return int(action_ids[item_or_action_idx].item())
            else:
                # This is an item position
                if item_ids is not None and item_or_action_idx < item_ids.shape[0]:
                    return int(item_ids[item_or_action_idx].item())
        
        return -1

    def _capture_qkv_fused(self, module, args, layer_idx):
        """Pre-hook to capture Q, K, V for FusedHSTULayer."""
        inputs = args
        if len(self.collected_data) >= self._max_batches * self.num_layers:
            return None  # Must return None to not modify inputs

        jd = inputs[0]
        if jd.values is None:
            return None

        hidden = jd.values

        # Compute Q, K, V manually using FusedHSTULayer's weights
        with torch.no_grad():
            eps = module._eps
            input_layernorm_weight = module._input_layernorm_weight
            input_layernorm_bias = module._input_layernorm_bias
            embedding_dim = module._embedding_dim
            num_heads = module._num_heads
            linear_dim_per_head = module._linear_dim_per_head
            attention_dim_per_head = module._attention_dim_per_head

            # Layer norm
            if input_layernorm_weight is not None:
                normed_x = F.layer_norm(
                    hidden,
                    normalized_shape=[embedding_dim],
                    weight=input_layernorm_weight,
                    bias=input_layernorm_bias,
                    eps=eps,
                )
            else:
                normed_x = hidden

            # Linear UVQK: [T, embedding_dim] @ [embedding_dim, (linear*2 + attn*2) * num_heads]
            linear_uvqk_weight = module._linear_uvqk_weight
            linear_uvqk_bias = module._linear_uvqk_bias
            mixed_uvqk = F.linear(normed_x, linear_uvqk_weight.t(), linear_uvqk_bias)
            silu_uvqk = F.silu(mixed_uvqk)

            # Split into U, V, Q, K
            # Layout: [U, V, Q, K] where each has shape [T, num_heads, head_dim]
            total_dim_per_head = linear_dim_per_head * 2 + attention_dim_per_head * 2
            silu_uvqk = silu_uvqk.view(-1, num_heads, total_dim_per_head)

            split_sizes = [linear_dim_per_head, linear_dim_per_head, attention_dim_per_head, attention_dim_per_head]
            user, value, query, key = torch.split(silu_uvqk, split_sizes, dim=-1)

            # Store Q, K, V with metadata
            self.collected_data.append({
                "layer_idx": layer_idx,
                "query": query.detach().clone(),
                "key": key.detach().clone(),
                "value": value.detach().clone(),
                "seqlen_offsets": jd.seqlen_offsets.cpu().clone(),
                "num_candidates": jd.num_candidates.cpu().clone() if jd.num_candidates is not None else None,
                "contextual_seqlen": jd.contextual_seqlen.cpu().clone() if jd.contextual_seqlen is not None else None,
                "has_interleaved_action": jd.has_interleaved_action,
                "scaling_seqlen": jd.scaling_seqlen,
                "is_causal": module._is_causal,
            })

        return None  # Must return None to not modify inputs

    def visualize(self, output_dir: str):
        """Generate attention map visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        print_rank_0(f"Generating attention visualizations in {output_dir}")

        if not self.collected_data:
            print_rank_0("No attention data collected.")
            return

        # Select layers to visualize: first, middle, last
        viz_layers = sorted(set([
            0,
            self.num_layers // 2,
            self.num_layers - 1,
        ]))
        viz_layers = [l for l in viz_layers if l < self.num_layers]

        # For each selected layer, visualize attention for action tokens
        for layer_idx in viz_layers:
            layer_data = [d for d in self.collected_data if d["layer_idx"] == layer_idx]
            if not layer_data:
                continue

            # Use the first batch
            sample = layer_data[0]
            query = sample["query"]  # [T, num_heads, head_dim]
            key = sample["key"]  # [T, num_heads, head_dim]
            seqlen_offsets = sample["seqlen_offsets"]
            has_interleaved = sample["has_interleaved_action"]
            num_candidates = sample["num_candidates"]
            contextual_seqlen = sample["contextual_seqlen"]
            scaling_seqlen = sample["scaling_seqlen"]
            is_causal = sample["is_causal"]

            # Compute attention weights - ensure seqlen_offsets is on the same device as Q, K
            device = query.device
            seqlen_offsets_device = seqlen_offsets.to(device)
            num_candidates_device = num_candidates.to(device) if num_candidates is not None else None
            contextual_seqlen_device = contextual_seqlen.to(device) if contextual_seqlen is not None else None

            attn_weights = compute_attention_weights(
                q=query,
                k=key,
                seqlen_offsets=seqlen_offsets_device,
                num_heads=self.num_heads,
                attention_dim=self.attention_dim,
                is_causal=is_causal,
                num_contextuals=contextual_seqlen_device,
                num_targets=num_candidates_device,
                scaling_seqlen=scaling_seqlen,
            )

            # Visualize for each user sequence
            batch_size = len(seqlen_offsets) - 1
            num_viz_users = min(2, batch_size)

            for user_idx in range(num_viz_users):
                seq_start = seqlen_offsets[user_idx].item()
                seq_end = seqlen_offsets[user_idx + 1].item()
                seq_len = seq_end - seq_start

                if seq_len == 0:
                    continue

                # Find action positions (odd indices in interleaved sequence)
                if has_interleaved:
                    action_positions = list(range(1, seq_len, 2))
                else:
                    action_positions = []

                if not action_positions:
                    continue

                # Select up to 3 action positions to visualize
                # Choose actions from early (~10%), middle, and late positions in the sequence
                num_actions_to_viz = min(3, len(action_positions))
                if len(action_positions) >= 3:
                    # Pick early (~10%), middle, and late actions
                    early_idx = max(0, len(action_positions) // 10)  # Around 10% position
                    middle_idx = len(action_positions) // 2
                    late_idx = -1
                    selected_action_pos = [
                        action_positions[early_idx],
                        action_positions[middle_idx],
                        action_positions[late_idx]
                    ]
                elif len(action_positions) == 2:
                    selected_action_pos = action_positions
                else:
                    selected_action_pos = action_positions[:num_actions_to_viz]

                # Create figure with subplots
                fig, axes = plt.subplots(
                    1, num_actions_to_viz,
                    figsize=(6 * num_actions_to_viz, 5)
                )
                if num_actions_to_viz == 1:
                    axes = [axes]

                for ax_idx, pos in enumerate(selected_action_pos):
                    ax = axes[ax_idx]

                    # Get attention weights for this action position
                    # attn_weights is [T, num_heads, N]
                    abs_pos = seq_start + pos
                    if abs_pos >= attn_weights.shape[0]:
                        continue

                    # Due to causal attention, this action can only see tokens 0 to pos (inclusive)
                    # Extract attention weights up to and including the action position
                    causal_limit = pos + 1  # Can see tokens 0, 1, ..., pos
                    attn_row = attn_weights[abs_pos].mean(dim=0)  # [N]
                    # Cast to float32 before numpy conversion (BFloat16 not supported)
                    attn_row_causal = attn_row[:causal_limit].cpu().float().numpy()

                    # Normalize within causal window
                    attn_sum = attn_row_causal.sum()
                    if attn_sum > 0:
                        attn_row_causal = attn_row_causal / attn_sum

                    # Find top-K attended tokens within causal window
                    top_k = min(10, causal_limit)
                    top_indices = np.argsort(attn_row_causal)[-top_k:][::-1]
                    top_scores = attn_row_causal[top_indices]

                    # Print detailed analysis with causal awareness
                    action_num = pos // 2
                    print_rank_0(f"\n=== Layer {layer_idx}, User {user_idx}, Action #{action_num} (token_id={self._get_token_id_at(user_idx, pos)}) at pos {pos} ===")
                    print_rank_0(f"Causal window: tokens 0 to {pos} (can see {causal_limit} tokens)")
                    print_rank_0(f"Top-{top_k} attended tokens (within causal window):")
                    for idx, score in zip(top_indices, top_scores):
                        token_type = "Item" if idx % 2 == 0 else "Action"
                        token_num = idx // 2
                        token_id_value = self._get_token_id_at(user_idx, idx)
                        print_rank_0(f"  Position {idx} ({token_type}#{token_num}, token_id={token_id_value}): attention = {score:.6f}")

                    # Check for sink token pattern (high attention to first few tokens)
                    first_5_avg = np.mean(attn_row_causal[:min(5, causal_limit)])
                    prev_item_attn = attn_row_causal[pos - 1] if pos > 0 else 0  # Previous item
                    self_attn = attn_row_causal[pos]  # Self attention

                    print_rank_0(f"  Average attention to first 5 tokens (sink): {first_5_avg:.6f}")
                    print_rank_0(f"  Attention to previous item (pos {pos-1}): {prev_item_attn:.6f}")
                    print_rank_0(f"  Self attention (pos {pos}): {self_attn:.6f}")

                    # Create a bar chart showing attention to key token groups
                    # Group 1: First 5 tokens (sink)
                    # Group 2: Previous item
                    # Group 3: Self
                    # Group 4: All other tokens in causal window (average)

                    other_indices = [i for i in range(causal_limit) if i < 5 or i == pos - 1 or i == pos]
                    other_avg = np.mean([attn_row_causal[i] for i in other_indices]) if other_indices else 0

                    groups = ['First 5\n(sink)', f'Prev Item\n(pos {pos-1})', f'Self\n(pos {pos})', 'Others\n(average)']
                    values = [
                        np.mean(attn_row_causal[:min(5, causal_limit)]),
                        prev_item_attn,
                        self_attn,
                        other_avg
                    ]

                    # Use distinct colors for each bar
                    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

                    bars = ax.bar(groups, values, color=colors, edgecolor='black', linewidth=0.5)

                    # Add value labels on top of ALL bars
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        # Always show label, but adjust position for very small values
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.4f}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')
                        else:
                            # For zero-height bars, place label slightly above axis
                            ax.text(bar.get_x() + bar.get_width()/2., 0.0001,
                                   f'{val:.4f}',
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')

                    ax.set_ylabel('Attention Score', fontsize=10)
                    ax.set_title(
                        f"Layer {layer_idx}, User {user_idx}\n"
                        f"Action #{pos//2} at pos {pos} (causal: sees 0-{pos})",
                        fontsize=11
                    )
                    ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 0.1)

                plt.tight_layout()
                filepath = os.path.join(
                    output_dir, f"attn_map_layer{layer_idx}_user{user_idx}.png"
                )
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                print_rank_0(f"  Saved: {filepath}")

        print_rank_0("Attention visualization complete.")


# ============================================================================
# Feature 2: KV Cache Difference Analysis
# ============================================================================

class KVCaptureHook:
    """Pre-hook to capture K and V tensors before attention computation."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.captured_kv: List[Dict] = []
        self._hooks = []
        self._max_batches = 2
        self._batch_count = 0
        self._layer_counter = {}
        self._max_users_per_batch = 2
        self._max_actions_per_user = 10  # Capture more actions
        self._action_ids: List[Dict] = []  # Store action token IDs

    def set_action_ids(self, action_ids: torch.Tensor):
        """Store action token IDs for grouping same action types."""
        if len(self._action_ids) < self._max_batches:
            self._action_ids.append(action_ids.cpu().clone())

    def register_hooks(self, model):
        """Register forward pre-hooks on HSTU attention modules."""
        from modules.native_hstu_layer import HSTULayer
        from modules.fused_hstu_layer import FusedHSTULayer
        unwrapped = get_unwrapped_module(model)
        hstu_block = unwrapped._hstu_block
        layer_idx = 0
        for layer in hstu_block._attention_layers:
            if isinstance(layer, FusedHSTULayer):
                # For FusedHSTULayer, compute Q, K, V manually from weights
                def make_hook(li):
                    def hook_fn(mod, args, kwargs):
                        return self._capture_kv_fused(mod, args, li)
                    return hook_fn
                hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                self._hooks.append(hook)
            elif isinstance(layer, HSTULayer):
                # For NativeHSTULayer, use the native method
                self._layer_counter[layer_idx] = 0
                def make_hook(li):
                    def hook_fn(mod, args, kwargs):
                        return self._capture_kv_pre(mod, args, li)
                    return hook_fn
                hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                self._hooks.append(hook)
            layer_idx += 1

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _capture_kv_pre(self, module, args, layer_idx):
        """Pre-hook to capture K and V for NativeHSTULayer."""
        inputs = args
        if self._batch_count >= self._max_batches:
            return None

        jd = inputs[0]
        if jd.values is None:
            return None

        hidden = jd.values

        with torch.no_grad():
            eps = module._eps
            input_layernorm_weight = module._input_layernorm_weight
            input_layernorm_bias = module._input_layernorm_bias
            embedding_dim = module._embedding_dim

            if input_layernorm_weight is not None:
                normed_x = F.layer_norm(
                    hidden,
                    normalized_shape=[embedding_dim],
                    weight=input_layernorm_weight,
                    bias=input_layernorm_bias,
                    eps=eps,
                )
            else:
                normed_x = hidden

            mixed_uvqk, _ = module._linear_uvqk(normed_x)
            silu_uvqk = F.silu(mixed_uvqk)

            split_arg_list = module._split_arg_list
            num_heads_compute = module._num_heads_per_partition

            silu_uvqk = silu_uvqk.view(-1, num_heads_compute, sum(split_arg_list))
            user, value, query, key = torch.split(silu_uvqk, split_arg_list, dim=-1)

            # Only capture limited users and actions for speed
            seqlen_offsets = jd.seqlen_offsets
            batch_size = len(seqlen_offsets) - 1
            limited_users = min(self._max_users_per_batch, batch_size)
            
            if limited_users < batch_size:
                # Create a mask for limited users
                last_offset = seqlen_offsets[limited_users].item()
                key = key[:last_offset].clone()
                value = value[:last_offset].clone()
                seqlen_offsets = seqlen_offsets[:limited_users + 1].clone()

            self.captured_kv.append({
                "layer_idx": layer_idx,
                "key": key.detach().clone(),
                "value": value.detach().clone(),
                "seqlen_offsets": seqlen_offsets.cpu().clone(),
                "num_candidates": jd.num_candidates.cpu().clone() if jd.num_candidates is not None else None,
                "contextual_seqlen": jd.contextual_seqlen.cpu().clone() if jd.contextual_seqlen is not None else None,
                "has_interleaved_action": jd.has_interleaved_action,
            })
            self._batch_count += 1

        return None

    def _capture_kv_fused(self, module, args, layer_idx):
        """Pre-hook to capture K and V for FusedHSTULayer."""
        inputs = args
        if self._batch_count >= self._max_batches:
            return None

        jd = inputs[0]
        if jd.values is None:
            return None

        hidden = jd.values

        with torch.no_grad():
            eps = module._eps
            input_layernorm_weight = module._input_layernorm_weight
            input_layernorm_bias = module._input_layernorm_bias
            embedding_dim = module._embedding_dim
            num_heads = module._num_heads
            linear_dim_per_head = module._linear_dim_per_head
            attention_dim_per_head = module._attention_dim_per_head

            if input_layernorm_weight is not None:
                normed_x = F.layer_norm(
                    hidden,
                    normalized_shape=[embedding_dim],
                    weight=input_layernorm_weight,
                    bias=input_layernorm_bias,
                    eps=eps,
                )
            else:
                normed_x = hidden

            linear_uvqk_weight = module._linear_uvqk_weight
            linear_uvqk_bias = module._linear_uvqk_bias
            mixed_uvqk = F.linear(normed_x, linear_uvqk_weight.t(), linear_uvqk_bias)
            silu_uvqk = F.silu(mixed_uvqk)

            total_dim_per_head = linear_dim_per_head * 2 + attention_dim_per_head * 2
            silu_uvqk = silu_uvqk.view(-1, num_heads, total_dim_per_head)

            split_sizes = [linear_dim_per_head, linear_dim_per_head, attention_dim_per_head, attention_dim_per_head]
            user, value, query, key = torch.split(silu_uvqk, split_sizes, dim=-1)

            # Only capture limited users for speed
            seqlen_offsets = jd.seqlen_offsets
            batch_size = len(seqlen_offsets) - 1
            limited_users = min(self._max_users_per_batch, batch_size)
            
            if limited_users < batch_size:
                last_offset = seqlen_offsets[limited_users].item()
                key = key[:last_offset].clone()
                value = value[:last_offset].clone()
                seqlen_offsets = seqlen_offsets[:limited_users + 1].clone()

            self.captured_kv.append({
                "layer_idx": layer_idx,
                "key": key.detach().clone(),
                "value": value.detach().clone(),
                "seqlen_offsets": seqlen_offsets.cpu().clone(),
                "num_candidates": jd.num_candidates.cpu().clone() if jd.num_candidates is not None else None,
                "contextual_seqlen": jd.contextual_seqlen.cpu().clone() if jd.contextual_seqlen is not None else None,
                "has_interleaved_action": jd.has_interleaved_action,
            })
            self._batch_count += 1

        return None

    def analyze(self, output_dir: str):
        """Analyze captured KV vectors - compare same action type across different positions."""
        os.makedirs(output_dir, exist_ok=True)
        print_rank_0(f"Analyzing {len(self.captured_kv)} captured KV samples")

        if not self.captured_kv:
            print_rank_0("No KV data captured.")
            return

        # Group by layer
        results_by_layer = {}
        for record in self.captured_kv:
            layer_idx = record["layer_idx"]
            if layer_idx not in results_by_layer:
                results_by_layer[layer_idx] = []
            results_by_layer[layer_idx].append(record)

        all_stats = []
        # Track action_type -> list of (position_category, K, V) across all users/batches
        # action_type is the actual token ID (e.g., 1 for "like")
        action_type_kvs = {}  # action_type -> {"key": [], "value": [], "positions": [], "pos_categories": []}

        for layer_idx in sorted(results_by_layer.keys()):
            layer_records = results_by_layer[layer_idx]
            if not layer_records:
                continue

            layer_stats = []

            for batch_idx, record in enumerate(layer_records):
                key = record["key"]
                value = record["value"]
                seqlen_offsets = record["seqlen_offsets"]
                has_interleaved = record["has_interleaved_action"]

                batch_size = len(seqlen_offsets) - 1
                limited_users = min(2, batch_size)

                # Get action IDs for this batch
                action_ids_batch = self._action_ids[batch_idx] if batch_idx < len(self._action_ids) else None

                for user_idx in range(limited_users):
                    seq_start = seqlen_offsets[user_idx].item()
                    seq_end = seqlen_offsets[user_idx + 1].item()
                    seq_len = seq_end - seq_start

                    if seq_len == 0:
                        continue

                    if has_interleaved:
                        action_positions = list(range(1, seq_len, 2))
                        action_positions = action_positions[:self._max_actions_per_user]
                    else:
                        action_positions = []

                    if len(action_positions) < 2:
                        continue

                    # Collect K/V for each action with its actual token ID and position
                    for pos in action_positions:
                        abs_pos = seq_start + pos
                        if abs_pos >= key.shape[0]:
                            continue

                        # Get actual action token ID
                        action_token_id = -1
                        if action_ids_batch is not None:
                            action_idx = pos // 2  # Action index in the action_ids tensor
                            if action_idx < action_ids_batch.shape[0]:
                                action_token_id = int(action_ids_batch[action_idx].item())

                        # Position category: early (first 1/3), mid (middle 1/3), late (last 1/3)
                        pos_category = "early" if pos < seq_len // 3 else ("mid" if pos < 2 * seq_len // 3 else "late")

                        action_key = f"action_type_{action_token_id}"
                        if action_key not in action_type_kvs:
                            action_type_kvs[action_key] = {
                                "key": [], "value": [], "positions": [], "pos_categories": []
                            }

                        action_type_kvs[action_key]["key"].append(key[abs_pos].detach().cpu())
                        action_type_kvs[action_key]["value"].append(value[abs_pos].detach().cpu())
                        action_type_kvs[action_key]["positions"].append(pos)
                        action_type_kvs[action_key]["pos_categories"].append(pos_category)

                    # Compare pairs within this user
                    max_pairs = 10
                    pair_count = 0
                    for i in range(len(action_positions)):
                        if pair_count >= max_pairs:
                            break
                        for j in range(i + 1, len(action_positions)):
                            if pair_count >= max_pairs:
                                break
                            pos_i = action_positions[i]
                            pos_j = action_positions[j]

                            abs_pos_i = seq_start + pos_i
                            abs_pos_j = seq_start + pos_j

                            if abs_pos_i >= key.shape[0] or abs_pos_j >= key.shape[0]:
                                continue

                            k_i = key[abs_pos_i]
                            k_j = key[abs_pos_j]
                            v_i = value[abs_pos_i]
                            v_j = value[abs_pos_j]

                            k_cos_sim = F.cosine_similarity(
                                k_i.reshape(1, -1), k_j.reshape(1, -1)
                            ).item()
                            v_cos_sim = F.cosine_similarity(
                                v_i.reshape(1, -1), v_j.reshape(1, -1)
                            ).item()
                            k_l2_dist = torch.norm(k_i - k_j).item()
                            v_l2_dist = torch.norm(v_i - v_j).item()

                            layer_stats.append({
                                "layer": layer_idx,
                                "user_idx": user_idx,
                                "pos_i": pos_i,
                                "pos_j": pos_j,
                                "pos_distance": abs(pos_j - pos_i),
                                "k_cosine_similarity": k_cos_sim,
                                "v_cosine_similarity": v_cos_sim,
                                "k_l2_distance": k_l2_dist,
                                "v_l2_distance": v_l2_dist,
                            })
                            pair_count += 1

            if layer_stats:
                results_by_layer[layer_idx] = layer_stats
                all_stats.extend(layer_stats)

        # Analysis: for each action type, compare KV across different positions
        print_rank_0("\n=== Action KV Consistency Analysis by Action Type ===")
        print_rank_0("This analyzes whether the SAME action type has consistent KV vectors at DIFFERENT positions")
        print_rank_0("")
        
        # Criteria for "significant difference":
        print_rank_0("Criteria for KV difference:")
        print_rank_0("  Cosine Sim > 0.95: Very similar (nearly identical) - position has LITTLE effect")
        print_rank_0("  Cosine Sim 0.80-0.95: Moderately similar - position has SOME effect")
        print_rank_0("  Cosine Sim 0.50-0.80: Somewhat different - position has SIGNIFICANT effect")
        print_rank_0("  Cosine Sim < 0.50: Very different - position has STRONG effect")
        print_rank_0("")

        # For each action type, compute pairwise similarity across different position categories
        for action_key, kv_data in sorted(action_type_kvs.items()):
            keys = kv_data["key"]
            values = kv_data["value"]
            positions = kv_data["positions"]
            pos_categories = kv_data["pos_categories"]
            
            if len(keys) < 2:
                continue
            
            # Group by position category
            by_pos_cat = {}
            for i, cat in enumerate(pos_categories):
                if cat not in by_pos_cat:
                    by_pos_cat[cat] = {"key": [], "value": []}
                by_pos_cat[cat]["key"].append(keys[i])
                by_pos_cat[cat]["value"].append(values[i])
            
            # Compute cross-position similarities (early vs mid, early vs late, mid vs late)
            cross_pos_sims = []
            categories = list(by_pos_cat.keys())
            for ci in range(len(categories)):
                for cj in range(ci + 1, len(categories)):
                    cat_i, cat_j = categories[ci], categories[cj]
                    for ki in by_pos_cat[cat_i]["key"]:
                        for kj in by_pos_cat[cat_j]["key"]:
                            sim = F.cosine_similarity(ki.reshape(1, -1), kj.reshape(1, -1)).item()
                            cross_pos_sims.append(sim)
            
            # Also compute within-position similarity for comparison
            within_pos_sims = []
            for cat in by_pos_cat:
                cat_keys = by_pos_cat[cat]["key"]
                for i in range(len(cat_keys)):
                    for j in range(i + 1, len(cat_keys)):
                        sim = F.cosine_similarity(cat_keys[i].reshape(1, -1), cat_keys[j].reshape(1, -1)).item()
                        within_pos_sims.append(sim)
            
            if cross_pos_sims:
                avg_cross_sim = np.mean(cross_pos_sims)
                avg_within_sim = np.mean(within_pos_sims) if within_pos_sims else -1
                
                def assess_similarity(avg_sim):
                    if avg_sim > 0.95:
                        return "VERY SIMILAR (position has LITTLE effect)"
                    elif avg_sim > 0.80:
                        return "MODERATELY SIMILAR (position has SOME effect)"
                    elif avg_sim > 0.50:
                        return "SOMEWHAT DIFFERENT (position has SIGNIFICANT effect)"
                    else:
                        return "VERY DIFFERENT (position has STRONG effect)"
                
                print_rank_0(f"{action_key} (action token ID = {action_key.split('_')[-1]})")
                print_rank_0(f"  Total samples: {len(keys)}")
                print_rank_0(f"  Position distribution: {dict(zip(*np.unique(pos_categories, return_counts=True)))}")
                print_rank_0(f"  Avg WITHIN-position K Cosine Sim: {avg_within_sim:.4f}")
                print_rank_0(f"  Avg CROSS-position K Cosine Sim: {avg_cross_sim:.4f} -> {assess_similarity(avg_cross_sim)}")
                print_rank_0(f"  Position sensitivity: {avg_within_sim - avg_cross_sim:.4f} (higher = more position-dependent)")
                print_rank_0("")

        # Generate summary statistics
        if all_stats:
            df = pd.DataFrame(all_stats)
            summary = df.groupby("layer").agg({
                "k_cosine_similarity": ["mean", "std", "min", "max"],
                "v_cosine_similarity": ["mean", "std", "min", "max"],
                "k_l2_distance": ["mean", "std", "min", "max"],
                "v_l2_distance": ["mean", "std", "min", "max"],
                "pos_distance": ["mean", "std"],
            }).round(6)

            print_rank_0("\n=== KV Cache Difference Summary by Layer ===")
            print_rank_0(summary.to_string())

            df.to_csv(os.path.join(output_dir, "kv_diff_detailed.csv"), index=False)
            summary.to_csv(os.path.join(output_dir, "kv_diff_summary.csv"))
            self._plot_kv_diff_trends(df, output_dir)

            print_rank_0(f"\nKV diff analysis saved to {output_dir}")
        else:
            print_rank_0("No action pairs found for comparison.")

    def _plot_kv_diff_trends(self, df: pd.DataFrame, output_dir: str):
        """Plot KV difference trends by position distance."""
        dist_groups = df.groupby("pos_distance").agg({
            "k_cosine_similarity": ["mean", "std"],
            "v_cosine_similarity": ["mean", "std"],
            "k_l2_distance": ["mean", "std"],
            "v_l2_distance": ["mean", "std"],
        }).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # K cosine similarity vs position distance
        axes[0, 0].errorbar(
            dist_groups["pos_distance"],
            dist_groups[("k_cosine_similarity", "mean")],
            yerr=dist_groups[("k_cosine_similarity", "std")],
            marker='o', capsize=3
        )
        axes[0, 0].set_title("K Cosine Similarity vs Position Distance")
        axes[0, 0].set_xlabel("Position Distance")
        axes[0, 0].set_ylabel("Cosine Similarity")
        axes[0, 0].grid(True, alpha=0.3)

        # V cosine similarity vs position distance
        axes[0, 1].errorbar(
            dist_groups["pos_distance"],
            dist_groups[("v_cosine_similarity", "mean")],
            yerr=dist_groups[("v_cosine_similarity", "std")],
            marker='o', color='orange', capsize=3
        )
        axes[0, 1].set_title("V Cosine Similarity vs Position Distance")
        axes[0, 1].set_xlabel("Position Distance")
        axes[0, 1].set_ylabel("Cosine Similarity")
        axes[0, 1].grid(True, alpha=0.3)

        # K L2 distance vs position distance
        axes[1, 0].errorbar(
            dist_groups["pos_distance"],
            dist_groups[("k_l2_distance", "mean")],
            yerr=dist_groups[("k_l2_distance", "std")],
            marker='o', color='green', capsize=3
        )
        axes[1, 0].set_title("K L2 Distance vs Position Distance")
        axes[1, 0].set_xlabel("Position Distance")
        axes[1, 0].set_ylabel("L2 Distance")
        axes[1, 0].grid(True, alpha=0.3)

        # V L2 distance vs position distance
        axes[1, 1].errorbar(
            dist_groups["pos_distance"],
            dist_groups[("v_l2_distance", "mean")],
            yerr=dist_groups[("v_l2_distance", "std")],
            marker='o', color='red', capsize=3
        )
        axes[1, 1].set_title("V L2 Distance vs Position Distance")
        axes[1, 1].set_xlabel("Position Distance")
        axes[1, 1].set_ylabel("L2 Distance")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kv_diff_trends.png"), dpi=150)
        plt.close()


# ============================================================================
# Feature 3: KV Cache Replacement
# ============================================================================

@dataclass(frozen=True)
class TokenKVReuseSpec:
    """Resolved reuse policy for one layer/user sequence."""

    window_size: int
    top_k: int


@dataclass(frozen=True)
class UserLengthBucket:
    """Optional policy override for a user sequence length range."""

    min_seq_len: int
    max_seq_len: Optional[int]
    window_size: Optional[int]
    top_k: Optional[int]

    def matches(self, seq_len: int) -> bool:
        if seq_len < self.min_seq_len:
            return False
        if self.max_seq_len is not None and seq_len > self.max_seq_len:
            return False
        return True


class TokenKVReusePolicy:
    """Resolves layer-wise and user-length-wise reuse parameters."""

    def __init__(
        self,
        default_window_size: int,
        default_top_k: int,
        layer_overrides: Optional[Dict[int, Dict[str, int]]] = None,
        user_length_buckets: Optional[List[UserLengthBucket]] = None,
        raw_policy_json: Optional[str] = None,
    ):
        self.default = TokenKVReuseSpec(
            window_size=self._validate_window_size(default_window_size),
            top_k=self._validate_top_k(default_top_k),
        )
        self.layer_overrides = layer_overrides or {}
        self.user_length_buckets = user_length_buckets or []
        self.raw_policy_json = raw_policy_json

    @classmethod
    def from_args(
        cls,
        default_window_size: int,
        default_top_k: int,
        policy_json: Optional[str],
    ) -> "TokenKVReusePolicy":
        if policy_json is None:
            return cls(default_window_size, default_top_k)

        raw_policy_json = policy_json
        if not policy_json.lstrip().startswith("{") and os.path.exists(policy_json):
            with open(policy_json, "r", encoding="utf-8") as f:
                policy_json = f.read()

        data = json.loads(policy_json)
        if not isinstance(data, dict):
            raise ValueError("--kv-reuse-policy-json must decode to a JSON object")

        default_cfg = data.get("default", {})
        if default_cfg is None:
            default_cfg = {}
        if not isinstance(default_cfg, dict):
            raise ValueError("policy default must be an object")

        resolved_default_window_size = default_cfg.get("window_size", default_window_size)
        resolved_default_top_k = default_cfg.get("top_k", default_top_k)

        layer_overrides: Dict[int, Dict[str, int]] = {}
        layers_cfg = data.get("layers", {})
        if layers_cfg is None:
            layers_cfg = {}
        if not isinstance(layers_cfg, dict):
            raise ValueError("policy layers must be an object keyed by layer index")
        for raw_layer_idx, raw_cfg in layers_cfg.items():
            if not isinstance(raw_cfg, dict):
                raise ValueError(f"policy layer {raw_layer_idx} must be an object")
            layer_idx = int(raw_layer_idx)
            layer_cfg: Dict[str, int] = {}
            if "window_size" in raw_cfg:
                layer_cfg["window_size"] = cls._validate_window_size(raw_cfg["window_size"])
            if "top_k" in raw_cfg:
                layer_cfg["top_k"] = cls._validate_top_k(raw_cfg["top_k"])
            layer_overrides[layer_idx] = layer_cfg

        buckets: List[UserLengthBucket] = []
        buckets_cfg = data.get("user_length_buckets", [])
        if buckets_cfg is None:
            buckets_cfg = []
        if not isinstance(buckets_cfg, list):
            raise ValueError("policy user_length_buckets must be a list")
        for idx, raw_bucket in enumerate(buckets_cfg):
            if not isinstance(raw_bucket, dict):
                raise ValueError(f"policy user_length_buckets[{idx}] must be an object")
            min_seq_len = int(raw_bucket.get("min_seq_len", 0))
            max_seq_len = raw_bucket.get("max_seq_len", None)
            max_seq_len = int(max_seq_len) if max_seq_len is not None else None
            if min_seq_len < 0:
                raise ValueError("bucket min_seq_len must be >= 0")
            if max_seq_len is not None and max_seq_len < min_seq_len:
                raise ValueError("bucket max_seq_len must be >= min_seq_len")
            buckets.append(
                UserLengthBucket(
                    min_seq_len=min_seq_len,
                    max_seq_len=max_seq_len,
                    window_size=cls._validate_window_size(raw_bucket["window_size"])
                    if "window_size" in raw_bucket
                    else None,
                    top_k=cls._validate_top_k(raw_bucket["top_k"])
                    if "top_k" in raw_bucket
                    else None,
                )
            )

        return cls(
            default_window_size=resolved_default_window_size,
            default_top_k=resolved_default_top_k,
            layer_overrides=layer_overrides,
            user_length_buckets=buckets,
            raw_policy_json=raw_policy_json,
        )

    @staticmethod
    def _validate_window_size(value: Any) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError("window_size must be > 0")
        return value

    @staticmethod
    def _validate_top_k(value: Any) -> int:
        value = int(value)
        if value < 0:
            raise ValueError("top_k must be >= 0")
        return value

    def resolve(self, layer_idx: int, seq_len: int) -> TokenKVReuseSpec:
        window_size = self.default.window_size
        top_k = self.default.top_k

        layer_cfg = self.layer_overrides.get(layer_idx, {})
        window_size = layer_cfg.get("window_size", window_size)
        top_k = layer_cfg.get("top_k", top_k)

        for bucket in self.user_length_buckets:
            if not bucket.matches(seq_len):
                continue
            if bucket.window_size is not None:
                window_size = bucket.window_size
            if bucket.top_k is not None:
                top_k = bucket.top_k
            break

        return TokenKVReuseSpec(window_size=window_size, top_k=top_k)


@dataclass(frozen=True)
class TokenRef:
    local_pos: int
    abs_pos: int
    token_id: int
    token_type: str


@dataclass(frozen=True)
class WindowReusePlan:
    replacements: Tuple[Tuple[int, int], ...]
    selected_token_types: int
    candidate_tokens: int
    has_tokens: bool


class TokenWindowSelector:
    """Selects top-K frequent token IDs in fixed local sequence windows."""

    @staticmethod
    def top_tokens(tokens: List[TokenRef], top_k: int) -> List[int]:
        if top_k <= 0 or not tokens:
            return []

        counts = Counter(token.token_id for token in tokens)
        first_pos: Dict[int, int] = {}
        for token in tokens:
            first_pos.setdefault(token.token_id, token.local_pos)

        ranked = sorted(
            counts.keys(),
            key=lambda token_id: (-counts[token_id], first_pos[token_id], token_id),
        )
        return ranked[:top_k]


class TokenKVReusePlanner:
    """Builds hidden-copy plans for one user/layer from token IDs and policy."""

    def __init__(self, policy: TokenKVReusePolicy):
        self._policy = policy
        self._selector = TokenWindowSelector()

    def build_plan(
        self,
        *,
        layer_idx: int,
        seq_start: int,
        seq_len: int,
        contextual_len: int,
        token_ids: Optional[torch.Tensor],
        token_offset: int,
        token_type: str,
        hidden_size_0: int,
    ) -> Tuple[List[WindowReusePlan], int, TokenKVReuseSpec]:
        spec = self._policy.resolve(layer_idx, seq_len)
        token_count = self.count_tokens(seq_len, contextual_len, token_type)
        if spec.top_k <= 0 or token_ids is None or token_count == 0:
            return [], token_count, spec

        tokens = self._collect_tokens(
            seq_start=seq_start,
            seq_len=seq_len,
            contextual_len=contextual_len,
            token_ids=token_ids,
            token_offset=token_offset,
            token_type=token_type,
            hidden_size_0=hidden_size_0,
        )
        if not tokens:
            return [], token_count, spec

        plans: List[WindowReusePlan] = []
        for window_start in range(0, seq_len, spec.window_size):
            window_end = min(seq_len, window_start + spec.window_size)
            window_tokens = [
                token
                for token in tokens
                if window_start <= token.local_pos < window_end
            ]
            if not window_tokens:
                continue

            selected_token_ids = set(self._selector.top_tokens(window_tokens, spec.top_k))
            first_abs_by_token: Dict[int, int] = {}
            replacements: List[Tuple[int, int]] = []
            candidate_tokens = 0
            for token in window_tokens:
                if token.token_id not in selected_token_ids:
                    continue
                if token.token_id not in first_abs_by_token:
                    first_abs_by_token[token.token_id] = token.abs_pos
                    continue
                candidate_tokens += 1
                replacements.append((token.abs_pos, first_abs_by_token[token.token_id]))

            plans.append(
                WindowReusePlan(
                    replacements=tuple(replacements),
                    selected_token_types=len(selected_token_ids),
                    candidate_tokens=candidate_tokens,
                    has_tokens=True,
                )
            )

        return plans, token_count, spec

    def collect_tokens(
        self,
        *,
        seq_start: int,
        seq_len: int,
        contextual_len: int,
        token_ids: Optional[torch.Tensor],
        token_offset: int,
        token_type: str,
        hidden_size_0: int,
    ) -> Tuple[List[TokenRef], int]:
        token_count = self.count_tokens(seq_len, contextual_len, token_type)
        if token_ids is None or token_count == 0:
            return [], token_count
        return (
            self._collect_tokens(
                seq_start=seq_start,
                seq_len=seq_len,
                contextual_len=contextual_len,
                token_ids=token_ids,
                token_offset=token_offset,
                token_type=token_type,
                hidden_size_0=hidden_size_0,
            ),
            token_count,
        )

    @staticmethod
    def count_tokens(seq_len: int, contextual_len: int, token_type: str) -> int:
        action_item_span = max(0, seq_len - contextual_len)
        if token_type == "item":
            return (action_item_span + 1) // 2
        if token_type == "action":
            return action_item_span // 2
        raise ValueError(f"Unsupported token_type={token_type}")

    @staticmethod
    def _collect_tokens(
        *,
        seq_start: int,
        seq_len: int,
        contextual_len: int,
        token_ids: torch.Tensor,
        token_offset: int,
        token_type: str,
        hidden_size_0: int,
    ) -> List[TokenRef]:
        tokens: List[TokenRef] = []
        action_item_span = max(0, seq_len - contextual_len)
        first_rel_pos = 0 if token_type == "item" else 1
        for rel_pos in range(first_rel_pos, action_item_span, 2):
            abs_pos = seq_start + contextual_len + rel_pos
            if abs_pos >= hidden_size_0:
                break
            token_idx = token_offset + (rel_pos // 2)
            if token_idx < 0 or token_idx >= token_ids.shape[0]:
                continue
            tokens.append(
                TokenRef(
                    local_pos=contextual_len + rel_pos,
                    abs_pos=abs_pos,
                    token_id=int(token_ids[token_idx].item()),
                    token_type=token_type,
                )
            )
        return tokens


# Backward-compatible names for older imports and scripts.
ActionKVReuseSpec = TokenKVReuseSpec
ActionKVReusePolicy = TokenKVReusePolicy
ActionTokenRef = TokenRef
ActionWindowSelector = TokenWindowSelector
ActionKVReusePlanner = TokenKVReusePlanner


class KVCacheReplacer:
    """
    Reuses action KV at a per-user, per-layer, per-window granularity.

    The hook copies the input hidden state from the first selected occurrence of
    an action within a local window to later occurrences of the same action. The
    layer then computes matching K/V from the copied hidden state.
    """

    def __init__(
        self,
        num_layers: int,
        reuse_policy: TokenKVReusePolicy,
        reuse_token_type: str = "action",
        reuse_strategy: str = "window_topk_same_id",
        reuse_max_distance: Optional[int] = None,
        replacement_impl: str = "hidden_proxy",
    ):
        if reuse_token_type not in {"action", "item", "both"}:
            raise ValueError("--kv-reuse-token-type must be one of: action, item, both")
        if reuse_strategy not in {
            "window_topk_same_id",
            "window_topk_same_id_max_distance",
            "global_same_id",
            "global_topk_same_id",
            "wrong_id_same_window",
            "global_first_any_action_legacy",
            "same_id_max_distance",
            "same_id_max_distance_no_chain",
        }:
            raise ValueError(
                "--kv-reuse-strategy must be one of: window_topk_same_id, "
                "window_topk_same_id_max_distance, "
                "global_same_id, global_topk_same_id, wrong_id_same_window, "
                "global_first_any_action_legacy, same_id_max_distance, "
                "same_id_max_distance_no_chain"
            )
        if replacement_impl not in {"hidden_proxy", "kv_only"}:
            raise ValueError("--kv-replace-implementation must be one of: hidden_proxy, kv_only")
        self.num_layers = num_layers
        self.reuse_policy = reuse_policy
        self.reuse_token_type = reuse_token_type
        self.reuse_strategy = reuse_strategy
        self.reuse_max_distance = reuse_max_distance
        self.replacement_impl = replacement_impl
        self._planner = TokenKVReusePlanner(reuse_policy)
        self._selector = TokenWindowSelector()
        self._hooks = []
        self._forward_overrides: List[Tuple[torch.nn.Module, Any]] = []
        self._current_action_ids: Optional[torch.Tensor] = None
        self._current_item_ids: Optional[torch.Tensor] = None
        self._legacy_first_hidden_by_layer: Dict[int, torch.Tensor] = {}
        self._legacy_first_projected_kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._enabled = False
        self._replacement_count = 0
        self._total_sequence_token_count = 0
        self._total_reuse_token_count = 0
        self._total_layer_sequence_token_count = 0
        self._total_layer_reuse_token_count = 0
        self._layer_stats: Dict[int, Dict[str, float]] = {}
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._replacement_count = 0
        self._total_sequence_token_count = 0
        self._total_reuse_token_count = 0
        self._total_layer_sequence_token_count = 0
        self._total_layer_reuse_token_count = 0
        self._legacy_first_hidden_by_layer = {}
        self._legacy_first_projected_kv_by_layer = {}
        self._layer_stats = {
            i: {
                "total_windows": 0,
                "eligible_windows": 0,
                "selected_token_types": 0,
                "candidate_tokens": 0,
                "considered": 0,
                "replaced": 0,
                "sum_window_size": 0,
                "sum_top_k": 0,
                "policy_resolutions": 0,
            }
            for i in range(self.num_layers)
        }

    def register_hooks(self, model):
        """Register layer interception for KV replacement."""
        from modules.native_hstu_layer import HSTULayer
        from modules.fused_hstu_layer import FusedHSTULayer
        from torch.nn.parallel import DistributedDataParallel
        from torchrec.distributed.model_parallel import DistributedModelParallel
        
        # Unwrap the model to get to the actual HSTU layers
        actual_model = model
        print_rank_0(f"[KV Replace] Initial model type: {type(actual_model)}")
        
        # Unwrap DistributedModelParallel first
        if isinstance(actual_model, DistributedModelParallel):
            actual_model = actual_model.module
            print_rank_0(f"[KV Replace] After DMP unwrap: {type(actual_model)}")
        
        # Then unwrap DistributedDataParallel
        if isinstance(actual_model, DistributedDataParallel):
            actual_model = actual_model.module
            print_rank_0(f"[KV Replace] After DDP unwrap: {type(actual_model)}")
        
        # Check if we need to unwrap again (sometimes there are multiple layers)
        if isinstance(actual_model, DistributedDataParallel):
            actual_model = actual_model.module
            print_rank_0(f"[KV Replace] After second DDP unwrap: {type(actual_model)}")
        
        print_rank_0(f"[KV Replace] Final unwrapped model type: {type(actual_model)}")
        print_rank_0(f"[KV Replace] Has _hstu_block: {hasattr(actual_model, '_hstu_block')}")
        
        # If still no _hstu_block, try to find it recursively
        if not hasattr(actual_model, '_hstu_block'):
            print_rank_0(f"[KV Replace] Searching for _hstu_block recursively...")
            for name, module in actual_model.named_modules():
                if hasattr(module, '_hstu_block'):
                    print_rank_0(f"[KV Replace] Found _hstu_block in module: {name}")
                    actual_model = module
                    break
        
        if not hasattr(actual_model, '_hstu_block'):
            print_rank_0(f"[KV Replace] ERROR: Could not find _hstu_block in model!")
            print_rank_0(f"[KV Replace] Model structure: {actual_model}")
            return
        
        hstu_block = actual_model._hstu_block
        print_rank_0(f"[KV Replace] Number of attention layers: {len(hstu_block._attention_layers)}")
        
        layer_idx = 0
        for layer in hstu_block._attention_layers:
            layer_type = type(layer).__name__
            print_rank_0(
                f"[KV Replace] Registering {self.replacement_impl} interception "
                f"on layer {layer_idx}: {layer_type}"
            )
            
            if self.replacement_impl == "kv_only":
                self._register_kv_only_forward(layer, layer_idx)
            elif isinstance(layer, FusedHSTULayer):
                def make_hook(li):
                    def hook_fn(mod, args, kwargs):
                        return self._replace_kv_fused(mod, args, li)
                    return hook_fn
                hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                self._hooks.append(hook)
            elif isinstance(layer, HSTULayer):
                def make_hook(li):
                    def hook_fn(mod, args, kwargs):
                        return self._replace_kv(mod, args, li)
                    return hook_fn
                hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                self._hooks.append(hook)
            layer_idx += 1
        
        print_rank_0(f"[KV Replace] Total hooks registered: {len(self._hooks)}")
        print_rank_0(f"[KV Replace] Total forward overrides registered: {len(self._forward_overrides)}")

    def remove_hooks(self):
        """Remove all registered hooks/forward overrides."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for layer, original_forward in reversed(self._forward_overrides):
            layer.forward = original_forward
        self._forward_overrides.clear()

    def _register_kv_only_forward(self, layer: torch.nn.Module, layer_idx: int) -> None:
        """Override an HSTU layer forward so only projected K/V rows are reused."""
        from modules.native_hstu_layer import HSTULayer
        from modules.fused_hstu_layer import FusedHSTULayer

        original_forward = layer.forward
        self._forward_overrides.append((layer, original_forward))

        if isinstance(layer, FusedHSTULayer):
            attn_func = create_hstu_attention(
                kernel_backend=layer._attn_backend,
                num_heads=layer._num_heads,
                attention_dim=layer._attention_dim_per_head,
                linear_dim=layer._linear_dim_per_head,
                is_causal=layer._is_causal,
            )

            def forward_kv_only(layer_self, jd):
                if jd.values is None or not self._policy_has_any_positive_top_k():
                    return original_forward(jd)
                return self._forward_fused_kv_only(layer_self, jd, layer_idx, attn_func)

            layer.forward = MethodType(forward_kv_only, layer)
        elif isinstance(layer, HSTULayer):
            def forward_kv_only(layer_self, jd):
                if jd.values is None or not self._policy_has_any_positive_top_k():
                    return original_forward(jd)
                return self._forward_native_kv_only(layer_self, jd, layer_idx)

            layer.forward = MethodType(forward_kv_only, layer)
        else:
            print_rank_0(
                f"[KV Replace] WARNING: unsupported layer type for kv_only override: {type(layer)}"
            )

    def _policy_has_any_positive_top_k(self) -> bool:
        policy = self._planner._policy
        if policy.default.top_k > 0:
            return True
        for cfg in policy.layer_overrides.values():
            if int(cfg.get("top_k", 0)) > 0:
                return True
        for bucket in policy.user_length_buckets:
            if bucket.top_k is not None and int(bucket.top_k) > 0:
                return True
        return False

    def _forward_fused_kv_only(self, layer, jd, layer_idx: int, attn_func) -> JaggedData:
        x = jd.values
        normed_x = self._fused_input_layer_norm(layer, x)
        silu_uvqk = self._fused_addmm_silu(
            x=normed_x,
            w=layer._linear_uvqk_weight,
            y=layer._linear_uvqk_bias,
            silu=True,
        )
        user, value, query, key = torch.split(
            silu_uvqk,
            [
                layer._linear_dim_per_head * layer._num_heads,
                layer._linear_dim_per_head * layer._num_heads,
                layer._attention_dim_per_head * layer._num_heads,
                layer._attention_dim_per_head * layer._num_heads,
            ],
            dim=-1,
        )
        value = value.view(-1, layer._num_heads, layer._linear_dim_per_head)
        query = query.view(-1, layer._num_heads, layer._attention_dim_per_head)
        key = key.view(-1, layer._num_heads, layer._attention_dim_per_head)
        key, value = self._apply_projected_kv_reuse(key, value, jd, layer_idx)
        jagged_attn_output = self._fused_cutlass_attention(
            layer=layer,
            query=query,
            key=key,
            value=value,
            jd=jd,
            fallback_attn_func=attn_func,
        )
        parallel_input = self._fused_output_norm_mul_dropout(layer, jagged_attn_output, user)
        residual = x if layer._residual else torch.zeros_like(x)
        output = self._fused_addmm_silu(
            x=parallel_input,
            w=layer._linear_proj_weight,
            y=residual,
            silu=False,
        )
        return JaggedData(
            values=output,
            seqlen=jd.seqlen,
            seqlen_offsets=jd.seqlen_offsets,
            max_seqlen=jd.max_seqlen,
            max_num_candidates=jd.max_num_candidates,
            num_candidates=jd.num_candidates,
            num_candidates_offsets=jd.num_candidates_offsets,
            contextual_max_seqlen=jd.contextual_max_seqlen,
            contextual_seqlen=jd.contextual_seqlen,
            contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
            has_interleaved_action=jd.has_interleaved_action,
            scaling_seqlen=jd.scaling_seqlen,
        )

    @staticmethod
    def _fused_sm_major() -> int:
        return torch.cuda.get_device_properties(0).major

    def _fused_addmm_silu(
        self,
        *,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        silu: bool,
    ) -> torch.Tensor:
        sm_major = self._fused_sm_major()
        if sm_major == 8:
            from ops.triton_ops.triton_addmm import triton_addmm_silu_fwd

            linear_out, silu_out = triton_addmm_silu_fwd(x=x, w=w, y=y, silu=silu)
            return silu_out if silu else linear_out
        if sm_major == 9:
            from ops.pt_ops.torch_addmm import torch_addmm_silu_fwd

            linear_out, silu_out = torch_addmm_silu_fwd(x=x, w=w, y=y, silu=silu)
            return silu_out if silu else linear_out
        return F.silu(torch.matmul(x, w) + y) if silu else torch.matmul(x, w) + y

    @staticmethod
    def _fused_input_layer_norm(layer, x: torch.Tensor) -> torch.Tensor:
        from ops.triton_ops.triton_layer_norm import triton_weighted_layer_norm_fwd

        normed_x, _, _, _, _ = triton_weighted_layer_norm_fwd(
            x=x,
            weight=layer._input_layernorm_weight,
            bias=layer._input_layernorm_bias,
            eps=layer._eps,
        )
        return normed_x

    def _fused_cutlass_attention(
        self,
        *,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        jd,
        fallback_attn_func,
    ) -> torch.Tensor:
        if layer._attn_backend.name != "CUTLASS":
            return fallback_attn_func(
                query,
                key,
                value,
                jd.seqlen_offsets,
                num_contextuals=jd.contextual_seqlen,
                num_candidates=jd.num_candidates,
                max_seqlen=jd.max_seqlen,
                scaling_seqlen=jd.scaling_seqlen,
                target_group_size=layer._target_group_size,
            )

        sm_major = self._fused_sm_major()
        extension_args = ()
        if sm_major == 8:
            import hstu_attn_2_cuda as flash_attn_cuda_ampere

            cutlass_hstu_varlen_fwd = flash_attn_cuda_ampere.varlen_fwd
            extension_args = (None, None, None, None, None)
        elif sm_major == 9:
            import hstu_hopper_cuda as flash_attn_cuda_hopper

            cutlass_hstu_varlen_fwd = flash_attn_cuda_hopper.varlen_fwd
            extension_args = (-1, None, None, None, None, None, None, None, None)
        else:
            return fallback_attn_func(
                query,
                key,
                value,
                jd.seqlen_offsets,
                num_contextuals=jd.contextual_seqlen,
                num_candidates=jd.num_candidates,
                max_seqlen=jd.max_seqlen,
                scaling_seqlen=jd.scaling_seqlen,
                target_group_size=layer._target_group_size,
            )

        num_contextuals = (
            jd.contextual_seqlen.to(torch.int32)
            if jd.contextual_seqlen is not None
            else None
        )
        num_candidates = (
            jd.num_candidates.to(torch.int32)
            if isinstance(jd.num_candidates, torch.Tensor)
            else None
        )
        attn_output, _ = cutlass_hstu_varlen_fwd(
            query,
            key,
            value,
            jd.seqlen_offsets.to(torch.int32),
            jd.seqlen_offsets.to(torch.int32),
            jd.max_seqlen,
            jd.max_seqlen,
            jd.scaling_seqlen,
            num_contextuals,
            num_candidates,
            layer._target_group_size,
            -1,
            0,
            layer._alpha,
            None,
            None,
            *extension_args,
        )
        return attn_output[:, :, : layer._linear_dim_per_head].reshape(
            -1,
            layer._num_heads * layer._linear_dim_per_head,
        )

    @staticmethod
    def _fused_output_norm_mul_dropout(layer, attn_output: torch.Tensor, user: torch.Tensor) -> torch.Tensor:
        from ops.triton_ops.triton_norm_mul_dropout import triton_layer_norm_mul_dropout_fwd

        out, _, _, _, _, _ = triton_layer_norm_mul_dropout_fwd(
            x=attn_output,
            u=user,
            weight=layer._output_layernorm_weight,
            bias=layer._output_layernorm_bias,
            eps=layer._eps,
            dropout_ratio=layer._dropout_ratio,
            training=layer.training,
            concat_ux=False,
            seed=layer._seed,
        )
        return out

    def _forward_native_kv_only(self, layer, jd, layer_idx: int) -> JaggedData:
        x = jd.values
        normed_x = F.layer_norm(
            x,
            normalized_shape=[layer._embedding_dim],
            weight=layer._input_layernorm_weight,
            bias=layer._input_layernorm_bias,
            eps=layer._eps,
        )
        tu, tv, tq, tk = layer.get_user_value_query_key_tensors(normed_x)
        tk, tv = self._apply_projected_kv_reuse(tk, tv, jd, layer_idx)
        jagged_attn_output = layer._attn_func(
            tq,
            tk,
            tv,
            jd.seqlen_offsets,
            num_contextuals=jd.contextual_seqlen,
            num_candidates=jd.num_candidates,
            max_seqlen=jd.max_seqlen,
            scaling_seqlen=jd.scaling_seqlen,
            target_group_size=layer._target_group_size,
        )
        padding_length = getattr(jd, "padding_length", 0)
        if padding_length > 0:
            last_valid_index = jd.seqlen_offsets[-1]
            jagged_attn_output = jagged_attn_output.clone()
            jagged_attn_output[last_valid_index : last_valid_index + padding_length, ...] = 0.0

        if layer._debug_shortcut_output_ln_mul_dropout:
            parallel_input = jagged_attn_output
        else:
            parallel_input = layer._output_ln_dropout_mul(jagged_attn_output, tu)

        if layer._debug_shortcut_proj_linear:
            from ops.collective_ops import gather_along_last_dim, split_along_first_dim

            output = gather_along_last_dim(
                parallel_input, parallel_state.get_tensor_model_parallel_group()
            )
            if layer._sequence_parallel:
                output = split_along_first_dim(
                    output, parallel_state.get_tensor_model_parallel_group()
                )
        else:
            output, _ = layer._linear_proj(parallel_input)

        if layer._residual:
            output = output + x
        return JaggedData(
            values=output,
            seqlen=jd.seqlen,
            seqlen_offsets=jd.seqlen_offsets,
            padding_length=padding_length,
            max_seqlen=jd.max_seqlen,
            max_num_candidates=jd.max_num_candidates,
            num_candidates=jd.num_candidates,
            num_candidates_offsets=jd.num_candidates_offsets,
            contextual_max_seqlen=jd.contextual_max_seqlen,
            contextual_seqlen=jd.contextual_seqlen,
            contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
            has_interleaved_action=jd.has_interleaved_action,
            scaling_seqlen=jd.scaling_seqlen,
        )

    def enable(self):
        """Enable KV replacement."""
        self._enabled = True
        self._reset_stats()

    def disable(self):
        """Disable KV replacement."""
        self._enabled = False

    def set_current_token_ids(
        self,
        *,
        item_ids: Optional[torch.Tensor],
        action_ids: Optional[torch.Tensor],
    ) -> None:
        """Set flattened item/action IDs for the current batch."""
        self._current_item_ids = item_ids.detach() if item_ids is not None else None
        self._current_action_ids = action_ids.detach() if action_ids is not None else None

    def set_current_action_ids(self, action_ids: Optional[torch.Tensor]) -> None:
        """Backward-compatible setter for action-only reuse."""
        self.set_current_token_ids(item_ids=None, action_ids=action_ids)

    def _replace_kv(self, module, args, layer_idx):
        """Pre-hook to replace KV of repeated actions with first-occurrence KV (NativeHSTULayer)."""
        return self._replace_hidden_inputs(args, layer_idx)

    def _replace_kv_fused(self, module, args, layer_idx):
        """Pre-hook to replace KV of repeated actions with first-occurrence KV (FusedHSTULayer).
        
        Key insight: We must modify the INPUT hidden states, not the locally computed K/V.
        The layer recomputes Q,K,V from inputs after our hook returns, so modifying local
        K/V has no effect. Instead, we store the first action's hidden state and copy it
        to subsequent action positions.
        """
        return self._replace_hidden_inputs(args, layer_idx)

    def _replace_hidden_inputs(self, args, layer_idx: int):
        if not self._enabled:
            return None

        jd = args[0]
        if jd.values is None or not jd.has_interleaved_action:
            return None

        hidden = jd.values
        seqlen_offsets = jd.seqlen_offsets
        if self.reuse_strategy == "global_first_any_action_legacy":
            return self._replace_legacy_first_any_action(hidden, jd, layer_idx)

        token_sources = self._active_token_sources()
        batch_size = len(seqlen_offsets) - 1
        token_offsets = {token_type: 0 for token_type, _ in token_sources}
        original_hidden = (
            hidden.detach().clone()
            if self.reuse_strategy == "same_id_max_distance_no_chain"
            else hidden
        )

        with torch.no_grad():
            for user_idx in range(batch_size):
                seq_start = int(seqlen_offsets[user_idx].item())
                seq_end = int(seqlen_offsets[user_idx + 1].item())
                seq_len = seq_end - seq_start
                contextual_len = self._get_contextual_len(jd, user_idx)

                counted_for_user = False
                for token_type, token_ids in token_sources:
                    token_count_for_user = self._planner.count_tokens(
                        seq_len, contextual_len, token_type
                    )
                    plans, token_count, spec = self._build_plans(
                        layer_idx=layer_idx,
                        seq_start=seq_start,
                        seq_len=seq_len,
                        contextual_len=contextual_len,
                        token_ids=token_ids,
                        token_offset=token_offsets[token_type],
                        token_type=token_type,
                        hidden_size_0=hidden.shape[0],
                    )
                    self._record_user_policy(
                        layer_idx,
                        seq_len,
                        token_count,
                        spec,
                        count_sequence_tokens=not counted_for_user,
                    )
                    counted_for_user = True
                    for plan in plans:
                        self._record_window_plan(layer_idx, plan)
                        for dst_abs, src_abs in plan.replacements:
                            hidden[dst_abs].copy_(original_hidden[src_abs].detach())
                            self._replacement_count += 1
                            self._layer_stats[layer_idx]["replaced"] += 1

                    token_offsets[token_type] += token_count_for_user
        return None

    def _apply_projected_kv_reuse(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        jd,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._enabled:
            return key, value
        if not jd.has_interleaved_action:
            return key, value
        if self.reuse_strategy == "global_first_any_action_legacy":
            return self._apply_legacy_first_any_action_projected(key, value, jd, layer_idx)

        token_sources = self._active_token_sources()
        seqlen_offsets = jd.seqlen_offsets
        batch_size = len(seqlen_offsets) - 1
        token_offsets = {token_type: 0 for token_type, _ in token_sources}
        replaced_any = False
        use_original_source = self.reuse_strategy == "same_id_max_distance_no_chain"
        original_key = (
            key.detach().clone()
            if use_original_source
            else key
        )
        original_value = (
            value.detach().clone()
            if use_original_source
            else value
        )

        with torch.no_grad():
            for user_idx in range(batch_size):
                seq_start = int(seqlen_offsets[user_idx].item())
                seq_end = int(seqlen_offsets[user_idx + 1].item())
                seq_len = seq_end - seq_start
                contextual_len = self._get_contextual_len(jd, user_idx)

                counted_for_user = False
                for token_type, token_ids in token_sources:
                    token_count_for_user = self._planner.count_tokens(
                        seq_len, contextual_len, token_type
                    )
                    plans, token_count, spec = self._build_plans(
                        layer_idx=layer_idx,
                        seq_start=seq_start,
                        seq_len=seq_len,
                        contextual_len=contextual_len,
                        token_ids=token_ids,
                        token_offset=token_offsets[token_type],
                        token_type=token_type,
                        hidden_size_0=key.shape[0],
                    )
                    self._record_user_policy(
                        layer_idx,
                        seq_len,
                        token_count,
                        spec,
                        count_sequence_tokens=not counted_for_user,
                    )
                    counted_for_user = True
                    for plan in plans:
                        self._record_window_plan(layer_idx, plan)
                        for dst_abs, src_abs in plan.replacements:
                            if not replaced_any:
                                key = key.clone()
                                value = value.clone()
                                replaced_any = True
                            source_key = original_key if use_original_source else key
                            source_value = original_value if use_original_source else value
                            key[dst_abs].copy_(source_key[src_abs].detach())
                            value[dst_abs].copy_(source_value[src_abs].detach())
                            self._replacement_count += 1
                            self._layer_stats[layer_idx]["replaced"] += 1

                    token_offsets[token_type] += token_count_for_user
        return key, value

    def _apply_legacy_first_any_action_projected(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        jd,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.reuse_token_type != "action":
            return key, value

        seqlen_offsets = jd.seqlen_offsets
        batch_size = len(seqlen_offsets) - 1
        replaced_any = False
        with torch.no_grad():
            for user_idx in range(batch_size):
                seq_start = int(seqlen_offsets[user_idx].item())
                seq_end = int(seqlen_offsets[user_idx + 1].item())
                seq_len = seq_end - seq_start
                contextual_len = self._get_contextual_len(jd, user_idx)
                action_count = self._planner.count_tokens(seq_len, contextual_len, "action")
                self._record_user_policy(
                    layer_idx,
                    seq_len,
                    action_count,
                    self._planner._policy.resolve(layer_idx, seq_len),
                    count_sequence_tokens=True,
                )
                if action_count <= 1:
                    continue

                first_action_abs = seq_start + contextual_len + 1
                if first_action_abs >= key.shape[0]:
                    continue
                if layer_idx not in self._legacy_first_projected_kv_by_layer:
                    self._legacy_first_projected_kv_by_layer[layer_idx] = (
                        key[first_action_abs].detach().clone(),
                        value[first_action_abs].detach().clone(),
                    )
                first_key, first_value = self._legacy_first_projected_kv_by_layer[layer_idx]

                replacements = []
                action_item_span = max(0, seq_len - contextual_len)
                for rel_pos in range(3, action_item_span, 2):
                    action_abs = seq_start + contextual_len + rel_pos
                    if action_abs >= key.shape[0]:
                        break
                    if not replaced_any:
                        key = key.clone()
                        value = value.clone()
                        replaced_any = True
                    key[action_abs].copy_(first_key)
                    value[action_abs].copy_(first_value)
                    replacements.append((action_abs, first_action_abs))
                    self._replacement_count += 1
                    self._layer_stats[layer_idx]["replaced"] += 1
                if replacements:
                    self._record_window_plan(
                        layer_idx,
                        WindowReusePlan(
                            replacements=tuple(replacements),
                            selected_token_types=1,
                            candidate_tokens=len(replacements),
                            has_tokens=True,
                        ),
                    )
        return key, value

    def _replace_legacy_first_any_action(self, hidden: torch.Tensor, jd, layer_idx: int):
        """Reproduce the old threshold-sweep bad baseline.

        The original script kept one first action hidden tensor per layer and
        copied it into every later action position in each user sequence,
        regardless of action id. This is intentionally not semantic reuse; it is
        a negative control for over-aggressive KV sharing.
        """
        if self.reuse_token_type != "action":
            return None

        seqlen_offsets = jd.seqlen_offsets
        batch_size = len(seqlen_offsets) - 1
        with torch.no_grad():
            for user_idx in range(batch_size):
                seq_start = int(seqlen_offsets[user_idx].item())
                seq_end = int(seqlen_offsets[user_idx + 1].item())
                seq_len = seq_end - seq_start
                contextual_len = self._get_contextual_len(jd, user_idx)
                action_count = self._planner.count_tokens(seq_len, contextual_len, "action")
                self._record_user_policy(
                    layer_idx,
                    seq_len,
                    action_count,
                    self._planner._policy.resolve(layer_idx, seq_len),
                    count_sequence_tokens=True,
                )
                if action_count <= 1:
                    continue

                first_action_abs = seq_start + contextual_len + 1
                if first_action_abs >= hidden.shape[0]:
                    continue
                if layer_idx not in self._legacy_first_hidden_by_layer:
                    self._legacy_first_hidden_by_layer[layer_idx] = (
                        hidden[first_action_abs].detach().clone()
                    )
                first_hidden = self._legacy_first_hidden_by_layer[layer_idx]

                replacements = []
                action_item_span = max(0, seq_len - contextual_len)
                for rel_pos in range(3, action_item_span, 2):
                    action_abs = seq_start + contextual_len + rel_pos
                    if action_abs >= hidden.shape[0]:
                        break
                    hidden[action_abs].copy_(first_hidden)
                    replacements.append((action_abs, first_action_abs))
                    self._replacement_count += 1
                    self._layer_stats[layer_idx]["replaced"] += 1
                if replacements:
                    self._record_window_plan(
                        layer_idx,
                        WindowReusePlan(
                            replacements=tuple(replacements),
                            selected_token_types=1,
                            candidate_tokens=len(replacements),
                            has_tokens=True,
                        ),
                    )
        return None

    def _build_plans(
        self,
        *,
        layer_idx: int,
        seq_start: int,
        seq_len: int,
        contextual_len: int,
        token_ids: Optional[torch.Tensor],
        token_offset: int,
        token_type: str,
        hidden_size_0: int,
    ) -> Tuple[List[WindowReusePlan], int, TokenKVReuseSpec]:
        if self.reuse_strategy == "window_topk_same_id":
            return self._planner.build_plan(
                layer_idx=layer_idx,
                seq_start=seq_start,
                seq_len=seq_len,
                contextual_len=contextual_len,
                token_ids=token_ids,
                token_offset=token_offset,
                token_type=token_type,
                hidden_size_0=hidden_size_0,
            )

        spec = self._planner._policy.resolve(layer_idx, seq_len)
        tokens, token_count = self._planner.collect_tokens(
            seq_start=seq_start,
            seq_len=seq_len,
            contextual_len=contextual_len,
            token_ids=token_ids,
            token_offset=token_offset,
            token_type=token_type,
            hidden_size_0=hidden_size_0,
        )
        if spec.top_k <= 0 or not tokens:
            return [], token_count, spec
        if self.reuse_strategy == "window_topk_same_id_max_distance":
            return self._build_window_topk_same_id_max_distance_plan(
                tokens, token_count, spec, seq_len
            ), token_count, spec
        if self.reuse_strategy == "global_same_id":
            return self._build_global_same_id_plan(tokens, token_count, spec), token_count, spec
        if self.reuse_strategy == "global_topk_same_id":
            return self._build_global_topk_same_id_plan(tokens, token_count, spec), token_count, spec
        if self.reuse_strategy in {"same_id_max_distance", "same_id_max_distance_no_chain"}:
            return self._build_same_id_max_distance_plan(tokens, token_count, spec), token_count, spec
        return self._build_wrong_id_same_window_plan(tokens, token_count, spec, seq_len), token_count, spec

    def _build_window_topk_same_id_max_distance_plan(
        self,
        tokens: List[TokenRef],
        token_count: int,
        spec: TokenKVReuseSpec,
        seq_len: int,
    ) -> List[WindowReusePlan]:
        plans: List[WindowReusePlan] = []
        max_distance = self.reuse_max_distance
        for window_start in range(0, seq_len, spec.window_size):
            window_end = min(seq_len, window_start + spec.window_size)
            window_tokens = [
                token for token in tokens if window_start <= token.local_pos < window_end
            ]
            if not window_tokens:
                continue
            selected_ids = set(self._selector.top_tokens(window_tokens, spec.top_k))
            first_abs_by_token: Dict[int, int] = {}
            replacements: List[Tuple[int, int]] = []
            candidate_tokens = 0
            for token in window_tokens:
                if token.token_id not in selected_ids:
                    continue
                src_abs = first_abs_by_token.get(token.token_id)
                if src_abs is None:
                    first_abs_by_token[token.token_id] = token.abs_pos
                    continue
                distance = token.abs_pos - src_abs
                if max_distance is None or max_distance < 0 or distance <= max_distance:
                    candidate_tokens += 1
                    replacements.append((token.abs_pos, src_abs))
            plans.append(
                WindowReusePlan(
                    replacements=tuple(replacements),
                    selected_token_types=len(selected_ids),
                    candidate_tokens=candidate_tokens,
                    has_tokens=True,
                )
            )
        return plans

    def _build_global_same_id_plan(
        self,
        tokens: List[TokenRef],
        token_count: int,
        spec: TokenKVReuseSpec,
    ) -> List[WindowReusePlan]:
        first_abs_by_token: Dict[int, int] = {}
        replacements: List[Tuple[int, int]] = []
        for token in tokens:
            if token.token_id not in first_abs_by_token:
                first_abs_by_token[token.token_id] = token.abs_pos
                continue
            replacements.append((token.abs_pos, first_abs_by_token[token.token_id]))
        if not replacements:
            return []
        return [
            WindowReusePlan(
                replacements=tuple(replacements),
                selected_token_types=len(first_abs_by_token),
                candidate_tokens=len(replacements),
                has_tokens=True,
            )
        ]

    def _build_global_topk_same_id_plan(
        self,
        tokens: List[TokenRef],
        token_count: int,
        spec: TokenKVReuseSpec,
    ) -> List[WindowReusePlan]:
        selected_ids = set(self._selector.top_tokens(tokens, spec.top_k))
        first_abs_by_token: Dict[int, int] = {}
        replacements: List[Tuple[int, int]] = []
        for token in tokens:
            if token.token_id not in selected_ids:
                continue
            if token.token_id not in first_abs_by_token:
                first_abs_by_token[token.token_id] = token.abs_pos
                continue
            replacements.append((token.abs_pos, first_abs_by_token[token.token_id]))
        if not replacements:
            return []
        return [
            WindowReusePlan(
                replacements=tuple(replacements),
                selected_token_types=len(selected_ids),
                candidate_tokens=len(replacements),
                has_tokens=True,
            )
        ]

    def _build_same_id_max_distance_plan(
        self,
        tokens: List[TokenRef],
        token_count: int,
        spec: TokenKVReuseSpec,
    ) -> List[WindowReusePlan]:
        """Reuse same-id KV only when a previous same-id source is close enough."""
        selected_ids = set(self._selector.top_tokens(tokens, spec.top_k))
        last_abs_by_token: Dict[int, int] = {}
        selected_seen: set = set()
        replacements: List[Tuple[int, int]] = []
        for token in tokens:
            if token.token_id not in selected_ids:
                continue
            selected_seen.add(token.token_id)
            src_abs = last_abs_by_token.get(token.token_id)
            if src_abs is not None:
                distance = token.abs_pos - src_abs
                if self.reuse_max_distance is None or self.reuse_max_distance < 0 or distance <= self.reuse_max_distance:
                    replacements.append((token.abs_pos, src_abs))
                last_abs_by_token[token.token_id] = token.abs_pos
            else:
                last_abs_by_token[token.token_id] = token.abs_pos
        if not replacements:
            return []
        return [
            WindowReusePlan(
                replacements=tuple(replacements),
                selected_token_types=len(selected_seen),
                candidate_tokens=len(replacements),
                has_tokens=True,
            )
        ]

    def _build_wrong_id_same_window_plan(
        self,
        tokens: List[TokenRef],
        token_count: int,
        spec: TokenKVReuseSpec,
        seq_len: int,
    ) -> List[WindowReusePlan]:
        plans: List[WindowReusePlan] = []
        for window_start in range(0, seq_len, spec.window_size):
            window_end = min(seq_len, window_start + spec.window_size)
            window_tokens = [
                token for token in tokens if window_start <= token.local_pos < window_end
            ]
            if not window_tokens:
                continue
            selected_ids = set(self._selector.top_tokens(window_tokens, spec.top_k))
            first_abs_by_token: Dict[int, int] = {}
            first_different_abs_by_token: Dict[int, int] = {}
            for token in window_tokens:
                first_abs_by_token.setdefault(token.token_id, token.abs_pos)
            for token in window_tokens:
                for other_id, other_abs in first_abs_by_token.items():
                    if other_id != token.token_id:
                        first_different_abs_by_token[token.token_id] = other_abs
                        break

            replacements: List[Tuple[int, int]] = []
            seen_selected_counts: Dict[int, int] = {}
            for token in window_tokens:
                if token.token_id not in selected_ids:
                    continue
                seen_selected_counts[token.token_id] = seen_selected_counts.get(token.token_id, 0) + 1
                if seen_selected_counts[token.token_id] == 1:
                    continue
                src_abs = first_different_abs_by_token.get(token.token_id)
                if src_abs is None:
                    continue
                replacements.append((token.abs_pos, src_abs))
            if replacements:
                plans.append(
                    WindowReusePlan(
                        replacements=tuple(replacements),
                        selected_token_types=len(selected_ids),
                        candidate_tokens=len(replacements),
                        has_tokens=True,
                    )
                )
        return plans

    def _active_token_sources(self) -> List[Tuple[str, Optional[torch.Tensor]]]:
        if self.reuse_token_type == "action":
            return [("action", self._current_action_ids)]
        if self.reuse_token_type == "item":
            return [("item", self._current_item_ids)]
        return [("item", self._current_item_ids), ("action", self._current_action_ids)]

    @staticmethod
    def _get_contextual_len(jd, user_idx: int) -> int:
        if jd.contextual_seqlen is None:
            return 0
        return int(jd.contextual_seqlen[user_idx].item())

    def _record_user_policy(
        self,
        layer_idx: int,
        seq_len: int,
        reuse_token_count: int,
        spec: TokenKVReuseSpec,
        count_sequence_tokens: bool,
    ) -> None:
        stat = self._layer_stats[layer_idx]
        stat["policy_resolutions"] += 1
        stat["sum_window_size"] += spec.window_size
        stat["sum_top_k"] += spec.top_k
        stat["total_windows"] += int(np.ceil(seq_len / spec.window_size)) if seq_len > 0 else 0

        if count_sequence_tokens:
            self._total_layer_sequence_token_count += seq_len
        self._total_layer_reuse_token_count += reuse_token_count
        if layer_idx == 0 and count_sequence_tokens:
            self._total_sequence_token_count += seq_len
        if layer_idx == 0:
            self._total_reuse_token_count += reuse_token_count

    def _record_window_plan(self, layer_idx: int, plan: WindowReusePlan) -> None:
        stat = self._layer_stats[layer_idx]
        if plan.has_tokens:
            stat["eligible_windows"] += 1
        stat["selected_token_types"] += plan.selected_token_types
        stat["candidate_tokens"] += plan.candidate_tokens
        stat["considered"] += plan.candidate_tokens

    def get_replacement_stats(self) -> Dict:
        """Get statistics about KV replacements."""
        layer_stats = []
        for layer_idx, s in sorted(self._layer_stats.items()):
            policy_resolutions = int(s["policy_resolutions"])
            considered = int(s["considered"])
            replaced = int(s["replaced"])
            layer_stats.append({
                "layer_idx": layer_idx,
                "total_windows": int(s["total_windows"]),
                "eligible_windows": int(s["eligible_windows"]),
                "selected_token_types": int(s["selected_token_types"]),
                "selected_action_types": int(s["selected_token_types"]),
                "candidate_tokens": int(s["candidate_tokens"]),
                "considered": considered,
                "replaced": replaced,
                "replace_rate": (replaced / considered) if considered > 0 else 0.0,
                "window_size_avg": (
                    s["sum_window_size"] / policy_resolutions
                    if policy_resolutions > 0
                    else float("nan")
                ),
                "top_k_avg": (
                    s["sum_top_k"] / policy_resolutions
                    if policy_resolutions > 0
                    else float("nan")
                ),
            })

        total_considered_count = sum(x["considered"] for x in layer_stats)
        candidate_reuse_token_count = sum(x["candidate_tokens"] for x in layer_stats)
        total_windows = sum(x["total_windows"] for x in layer_stats)
        total_eligible_windows = sum(x["eligible_windows"] for x in layer_stats)
        total_selected_token_types = sum(x["selected_token_types"] for x in layer_stats)
        global_replace_rate = (
            self._replacement_count / total_considered_count
            if total_considered_count > 0 else 0.0
        )
        selected_ratio_all_tokens = (
            candidate_reuse_token_count / self._total_layer_sequence_token_count
            if self._total_layer_sequence_token_count > 0 else 0.0
        )
        selected_ratio_all_reuse_tokens = (
            candidate_reuse_token_count / self._total_layer_reuse_token_count
            if self._total_layer_reuse_token_count > 0 else 0.0
        )
        selected_ratio_candidate_reuse_tokens = (
            self._replacement_count / candidate_reuse_token_count
            if candidate_reuse_token_count > 0 else 0.0
        )

        return {
            "enabled": self._enabled,
            "reuse_token_type": self.reuse_token_type,
            "reuse_strategy": self.reuse_strategy,
            "reuse_max_distance": self.reuse_max_distance,
            "replacement_impl": self.replacement_impl,
            "default_window_size": self.reuse_policy.default.window_size,
            "default_top_k": self.reuse_policy.default.top_k,
            "policy_json": self.reuse_policy.raw_policy_json,
            "layers_with_first_kv": sum(
                1 for layer_stat in layer_stats if layer_stat["candidate_tokens"] > 0
            ),
            "replacement_count": self._replacement_count,
            "selected_reuse_token_count": candidate_reuse_token_count,
            "candidate_reuse_token_count": candidate_reuse_token_count,
            "selected_action_token_count": candidate_reuse_token_count,
            "candidate_action_token_count": candidate_reuse_token_count,
            "total_sequence_token_count": self._total_sequence_token_count,
            "total_layer_sequence_token_count": self._total_layer_sequence_token_count,
            "total_reuse_token_count": self._total_reuse_token_count,
            "total_layer_reuse_token_count": self._total_layer_reuse_token_count,
            "total_action_count": self._total_reuse_token_count,
            "total_layer_action_count": self._total_layer_reuse_token_count,
            "total_replace_candidate_reuse_token_count": candidate_reuse_token_count,
            "total_replace_candidate_action_count": candidate_reuse_token_count,
            "total_considered_count": total_considered_count,
            "total_windows": total_windows,
            "total_eligible_windows": total_eligible_windows,
            "total_selected_token_types": total_selected_token_types,
            "total_selected_action_types": total_selected_token_types,
            "global_replace_rate": global_replace_rate,
            "selected_ratio_all_tokens": selected_ratio_all_tokens,
            "selected_ratio_all_reuse_tokens": selected_ratio_all_reuse_tokens,
            "selected_ratio_candidate_reuse_tokens": selected_ratio_candidate_reuse_tokens,
            "selected_ratio_all_actions": selected_ratio_all_reuse_tokens,
            "selected_ratio_candidate_actions": selected_ratio_candidate_reuse_tokens,
            "layer_stats": layer_stats,
        }


# ============================================================================
# Main Evaluation with Analysis
# ============================================================================

def run_attention_viz_analysis(
    model_train, model, eval_dataloader, stateful_metric_module, trainer_args, output_dir: str
):
    """Run attention visualization analysis."""
    print_rank_0("=== Running Attention Visualization Analysis ===")

    unwrapped = get_unwrapped_module(model)
    hstu_config = unwrapped._hstu_config
    num_layers = hstu_config.num_layers
    num_heads = hstu_config.num_attention_heads
    attention_dim = hstu_config.kv_channels

    collector = AttentionMapCollector(num_layers, num_heads, attention_dim)

    # Register hooks on the unwrapped model
    collector.register_hooks(model)

    # Create pipeline using the already-wrapped model_train
    device = torch.device("cuda", torch.cuda.current_device())
    pipeline = JaggedMegatronTrainNonePipeline(
        model_train,
        torch.optim.Adam(model.parameters(), lr=1e-5),
        device=device,
    )
    pipeline._model.eval()

    # Use pipeline progress to handle data transfer properly
    from itertools import islice
    max_batches = min(3, len(eval_dataloader))
    iterated_eval_loader = islice(eval_dataloader, len(eval_dataloader))
    
    # Run pipeline progress to collect attention data
    with torch.no_grad():
        batch_count = 0
        for batch in iterated_eval_loader:
            if batch_count >= max_batches:
                break
            # Extract token IDs from the batch features
            batch = batch.to(device)
            features = batch.features
            
            # Extract the values for item and action features
            # The KeyedJaggedTensor has keys like contextual features + item_feature + action_feature
            item_ids = None
            action_ids = None
            
            # Try to get feature names from the model config
            if hasattr(unwrapped, '_item_feature_name'):
                item_feature_name = unwrapped._item_feature_name
            else:
                # Default: look for the last two keys (item and action)
                all_keys = list(features.keys())
                item_feature_name = all_keys[-2] if len(all_keys) >= 2 else None
                action_feature_name = all_keys[-1] if len(all_keys) >= 1 else None
            
            if hasattr(unwrapped, '_action_feature_name'):
                action_feature_name = unwrapped._action_feature_name
            
            # Extract values
            if item_feature_name and item_feature_name in features.keys():
                item_ids = features[item_feature_name].values()
            if action_feature_name and action_feature_name in features.keys():
                action_ids = features[action_feature_name].values()
            
            if item_ids is not None and action_ids is not None:
                collector.set_token_ids(item_ids, action_ids)
            
            # Run the model
            pipeline._model(batch)
            batch_count += 1

    # Remove hooks
    collector.remove_hooks()

    # Generate visualizations
    collector.visualize(output_dir)

    # Run standard evaluation
    print_rank_0("\n=== Running Standard Evaluation ===")
    evaluate(pipeline, stateful_metric_module, trainer_args=trainer_args, eval_loader=eval_dataloader)


def run_kv_diff_analysis(
    model_train, model, eval_dataloader, stateful_metric_module, trainer_args, output_dir: str
):
    """Run KV Cache difference analysis."""
    print_rank_0("=== Running KV Cache Difference Analysis ===")

    unwrapped = get_unwrapped_module(model)
    hstu_config = unwrapped._hstu_config
    num_layers = hstu_config.num_layers

    analyzer = KVCaptureHook(num_layers)

    # Register hooks on the unwrapped model
    analyzer.register_hooks(model)

    # Create pipeline using the already-wrapped model_train
    device = torch.device("cuda", torch.cuda.current_device())
    pipeline = JaggedMegatronTrainNonePipeline(
        model_train,
        torch.optim.Adam(model.parameters(), lr=1e-5),
        device=device,
    )
    pipeline._model.eval()

    # Use pipeline progress to handle data transfer properly
    from itertools import islice
    max_batches = min(5, len(eval_dataloader))
    iterated_eval_loader = islice(eval_dataloader, len(eval_dataloader))
    
    # Run pipeline progress to collect KV data
    with torch.no_grad():
        batch_count = 0
        for batch in iterated_eval_loader:
            if batch_count >= max_batches:
                break
            try:
                batch = batch.to(device)
                # Extract action IDs for grouping
                features = batch.features
                action_ids = None
                if hasattr(unwrapped, '_action_feature_name'):
                    action_feature_name = unwrapped._action_feature_name
                else:
                    all_keys = list(features.keys())
                    action_feature_name = all_keys[-1] if all_keys else None
                
                if action_feature_name and action_feature_name in features.keys():
                    action_ids = features[action_feature_name].values()
                    analyzer.set_action_ids(action_ids)
                
                pipeline._model(batch)
                batch_count += 1
            except StopIteration:
                break

    # Remove hooks
    analyzer.remove_hooks()

    # Analyze
    analyzer.analyze(output_dir)

    # Run standard evaluation
    print_rank_0("\n=== Running Standard Evaluation ===")
    evaluate(pipeline, stateful_metric_module, trainer_args=trainer_args, eval_loader=eval_dataloader)


def _mark_reuse_auc_pareto(summary: pd.DataFrame) -> List[bool]:
    marks = []
    for _, row in summary.iterrows():
        dominated = False
        for _, other in summary.iterrows():
            if other["reuse_mode"] == row["reuse_mode"]:
                continue
            same_or_better = (
                other["selected_ratio_all_tokens"] >= row["selected_ratio_all_tokens"]
                and other["mean_kv_replaced"] >= row["mean_kv_replaced"]
            )
            strictly_better = (
                other["selected_ratio_all_tokens"] > row["selected_ratio_all_tokens"]
                or other["mean_kv_replaced"] > row["mean_kv_replaced"]
            )
            if same_or_better and strictly_better:
                dominated = True
                break
        marks.append(not dominated)
    return marks


def _write_kv_replace_eval_markdown(
    *,
    output_dir: str,
    comparison_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    auc_baseline_threshold: float = 0.6,
    report_command: Optional[str] = None,
) -> None:
    auc_rows = comparison_df[
        comparison_df["metric"].astype(str).str.upper().str.contains("AUC")
    ].copy()
    if auc_rows.empty:
        return

    eligible_metrics = (
        auc_rows[auc_rows["baseline"] > auc_baseline_threshold]["metric"]
        .drop_duplicates()
        .tolist()
    )
    filtered = auc_rows[auc_rows["metric"].isin(eligible_metrics)] if eligible_metrics else auc_rows
    group_cols = ["reuse_mode", "reuse_strategy", "reuse_token_type", "reuse_max_distance"]
    filtered_summary = (
        filtered.groupby(group_cols, dropna=False)
        .agg(
            filtered_task_count=("metric", "count"),
            mean_baseline=("baseline", "mean"),
            mean_kv_replaced=("kv_replaced", "mean"),
            mean_diff=("diff", "mean"),
            min_diff=("diff", "min"),
            max_auc_drop=("diff", lambda x: float(max(0.0, -x.min()))),
        )
        .reset_index()
    )
    filtered_summary["reuse_max_distance"] = pd.to_numeric(
        filtered_summary["reuse_max_distance"], errors="coerce"
    )
    stats_df = stats_df.copy()
    stats_df["reuse_max_distance"] = pd.to_numeric(
        stats_df["reuse_max_distance"], errors="coerce"
    )
    filtered_summary = filtered_summary.merge(
        stats_df[
            [
                "reuse_mode",
                "reuse_strategy",
                "reuse_token_type",
                "reuse_max_distance",
                "replacement_count",
                "selected_ratio_all_tokens",
                "selected_ratio_all_reuse_tokens",
                "replacement_impl",
            ]
        ],
        on=group_cols,
        how="left",
    )
    filtered_summary["pareto_optimal"] = _mark_reuse_auc_pareto(filtered_summary)
    filtered_summary.to_csv(
        os.path.join(output_dir, "reuse_auc_impact_summary_auc_gt_0p6.csv"),
        index=False,
    )

    table = filtered_summary.sort_values(
        ["pareto_optimal", "mean_kv_replaced", "selected_ratio_all_tokens"],
        ascending=[False, False, False],
    )
    command = report_command or " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    with open(os.path.join(output_dir, "report_commands.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "stage": "kv_replace_eval",
                    "command": command,
                }
            ],
            f,
            indent=2,
        )

    lines = [
        "# Action KV Reuse Evaluation Insights",
        "",
        f"- Filtered AUC tasks: {', '.join(eligible_metrics) if eligible_metrics else 'all AUC tasks'}",
        f"- AUC threshold: baseline > {auc_baseline_threshold:.1f}",
        f"- Implementation: {stats_df['replacement_impl'].dropna().iloc[0] if 'replacement_impl' in stats_df and not stats_df.empty else 'unknown'}",
        "",
        "## Reuse Ratio vs AUC",
        "",
        "Command:",
        "",
        "```bash",
        command,
        "```",
        "",
        "| Mode | Distance | Reuse Ratio | Mean AUC | Mean Diff | Max Drop | Replacements | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in table.iterrows():
        distance = row["reuse_max_distance"]
        if pd.isna(distance):
            distance_str = "-"
        elif int(distance) < 0:
            distance_str = "global"
        else:
            distance_str = str(int(distance))
        lines.append(
            "| "
            f"{row['reuse_mode']} | "
            f"{distance_str} | "
            f"{row['selected_ratio_all_tokens'] * 100:.2f}% | "
            f"{row['mean_kv_replaced']:.6f} | "
            f"{row['mean_diff']:+.6f} | "
            f"{row['max_auc_drop']:.6f} | "
            f"{int(row['replacement_count']) if not pd.isna(row['replacement_count']) else 0:,} | "
            f"{'yes' if bool(row['pareto_optimal']) else 'no'} |"
        )

    bad = table.sort_values("max_auc_drop", ascending=False).head(2)
    semantic_good = table[
        table["reuse_strategy"].isin(
            [
                "window_topk_same_id",
                "window_topk_same_id_max_distance",
                "same_id_max_distance",
                "same_id_max_distance_no_chain",
                "global_topk_same_id",
                "global_same_id",
            ]
        )
    ]
    good = semantic_good.sort_values(
        ["mean_kv_replaced", "selected_ratio_all_tokens"], ascending=[False, False]
    ).head(2)
    lines.extend(["", "## Takeaways", ""])
    if not bad.empty:
        worst = bad.iloc[0]
        lines.append(
            f"- Aggressive or semantically wrong reuse is the negative control: "
            f"`{worst['reuse_mode']}` reaches {worst['selected_ratio_all_tokens'] * 100:.2f}% reuse "
            f"but has max AUC drop {worst['max_auc_drop']:.6f}."
        )
    if not good.empty:
        best = good.iloc[0]
        lines.append(
            f"- Constrained same-action reuse is the useful regime: "
            f"`{best['reuse_mode']}` reaches {best['selected_ratio_all_tokens'] * 100:.2f}% reuse "
            f"with mean AUC diff {best['mean_diff']:+.6f}."
        )
    lines.append(
        "- The evaluation is now KV-only: source rows are copied after UVQK projection, so Q/U remain position-specific."
    )

    with open(os.path.join(output_dir, "KV_REUSE_EVAL_INSIGHTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_kv_replace_analysis(
    model_train,
    model,
    eval_dataloader,
    stateful_metric_module,
    trainer_args,
    output_dir: str,
    reuse_policy: TokenKVReusePolicy,
    deprecated_similarity_threshold: float,
    reuse_token_types: List[str],
    reuse_strategies: List[str],
    reuse_max_distances: Optional[List[int]] = None,
    replacement_impl: str = "hidden_proxy",
    report_command: Optional[str] = None,
    baseline_metrics_override: Optional[Dict[str, float]] = None,
):
    """Run KV Cache replacement analysis."""
    print_rank_0("=== Running KV Cache Replacement Analysis ===")
    print_rank_0(
        "Action KV reuse policy: "
        f"default_window_size={reuse_policy.default.window_size}, "
        f"default_top_k={reuse_policy.default.top_k}"
    )
    print_rank_0(
        "--kv-replace-sim-threshold is deprecated and ignored by window/top-K reuse "
        f"(received {deprecated_similarity_threshold:.4f})."
    )
    print_rank_0(f"KV reuse token modes: {reuse_token_types}")
    print_rank_0(f"KV reuse strategies: {reuse_strategies}")
    print_rank_0(f"KV reuse max distances: {reuse_max_distances or ['default']}")
    print_rank_0(f"KV replacement implementation: {replacement_impl}")

    unwrapped = get_unwrapped_module(model)
    hstu_config = unwrapped._hstu_config
    num_layers = hstu_config.num_layers

    # Create pipeline using the already-wrapped model_train
    device = torch.device("cuda", torch.cuda.current_device())
    pipeline = JaggedMegatronTrainNonePipeline(
        model_train,
        torch.optim.Adam(model.parameters(), lr=1e-5),
        device=device,
    )
    pipeline._model.eval()

    def _reset_metric_module() -> None:
        reset_found = False
        if hasattr(stateful_metric_module, 'reset'):
            stateful_metric_module.reset()
            reset_found = True
        elif hasattr(stateful_metric_module, '_reset'):
            stateful_metric_module._reset()
            reset_found = True
        else:
            for attr_name in ['metrics', '_metrics', 'metric', '_metric', 'eval_module', '_eval_module']:
                if hasattr(stateful_metric_module, attr_name):
                    inner = getattr(stateful_metric_module, attr_name)
                    if isinstance(inner, dict):
                        for v in inner.values():
                            if hasattr(v, 'reset'):
                                v.reset()
                                reset_found = True
                            elif hasattr(v, '_reset'):
                                v._reset()
                                reset_found = True
                    elif hasattr(inner, 'reset'):
                        inner.reset()
                        reset_found = True
                    elif hasattr(inner, '_reset'):
                        inner._reset()
                        reset_found = True

        if not reset_found:
            for attr_name in ['preds', 'target', '_update_called']:
                if hasattr(stateful_metric_module, attr_name):
                    val = getattr(stateful_metric_module, attr_name)
                    if isinstance(val, list):
                        val.clear()
                    elif isinstance(val, torch.Tensor):
                        setattr(stateful_metric_module, attr_name, [])

    # Use pipeline progress for evaluation.
    from itertools import islice
    requested_eval_iters = (
        trainer_args.max_eval_iters
        if trainer_args.max_eval_iters is not None
        else len(eval_dataloader)
    )
    max_batches = min(requested_eval_iters, len(eval_dataloader))

    if baseline_metrics_override is not None:
        baseline_metrics = {
            key: torch.tensor(float(value), device=device)
            for key, value in baseline_metrics_override.items()
        }
        print_rank_0("\n--- Baseline Evaluation (No KV Replacement) ---")
        print_rank_0(f"Baseline metrics reused from cache: {baseline_metrics}")
    else:
        # First, run baseline evaluation (no replacement) - collect logits/labels.
        print_rank_0("\n--- Baseline Evaluation (No KV Replacement) ---")
        _reset_metric_module()
        iterated_eval_loader = islice(eval_dataloader, len(eval_dataloader))

        all_logits = []
        all_labels = []
        batch_count = 0

        with torch.no_grad():
            for i in range(max_batches):
                try:
                    reporting_loss, (local_loss, logits, labels, seqlen_info) = pipeline.progress(iterated_eval_loader)
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(labels.detach().cpu())
                    batch_count += 1
                except StopIteration:
                    break

        # Compute baseline metrics - keep tensors on CUDA device for distributed sync
        if all_logits and all_labels:
            all_logits_baseline = torch.cat(all_logits, dim=0).to(device)
            all_labels_baseline = torch.cat(all_labels, dim=0).to(device)

            # Use the metric module to compute baseline
            stateful_metric_module(all_logits_baseline, all_labels_baseline)
            if isinstance(stateful_metric_module, RetrievalTaskMetricWithSampling):
                retrieval_gr = get_unwrapped_module(pipeline._model)
                export_table_name = retrieval_gr.get_item_feature_table_name()
                baseline_metrics, _, _ = stateful_metric_module.compute(
                    *retrieval_gr._embedding_collection.export_local_embedding(export_table_name)
                )
            else:
                baseline_metrics = stateful_metric_module.compute()
            print_rank_0(f"Baseline metrics: {baseline_metrics}")
        else:
            print_rank_0("No baseline data collected.")
            return

    # Now run with KV replacement
    class _TokenAwareEvalIter:
        """Iterator wrapper that feeds current batch item/action IDs to replacer."""

        def __init__(
            self,
            base_iter,
            replacer_obj,
            item_feature_name: Optional[str],
            action_feature_name: Optional[str],
        ):
            self._base_iter = base_iter
            self._replacer = replacer_obj
            self._item_feature_name = item_feature_name
            self._action_feature_name = action_feature_name
            self._resolved_item_feature_name: Optional[str] = item_feature_name
            self._resolved_action_feature_name: Optional[str] = action_feature_name
            self._missing_item_ids_batches = 0
            self._missing_action_ids_batches = 0
            self._seen_batches = 0

        def __iter__(self):
            return self

        def __next__(self):
            batch = next(self._base_iter)
            self._seen_batches += 1
            item_ids = None
            action_ids = None
            features = batch.features
            feature_keys = list(features.keys())

            def _resolve(
                configured_name: Optional[str],
                cached_name: Optional[str],
                fallback_name: Optional[str],
            ) -> Optional[str]:
                candidate_names: List[str] = []
                if configured_name is not None:
                    candidate_names.append(configured_name)
                if cached_name is not None:
                    candidate_names.append(cached_name)
                if fallback_name is not None:
                    candidate_names.append(fallback_name)
                for name in candidate_names:
                    if name is not None and name in features.keys():
                        return name
                return None

            item_fallback = feature_keys[-2] if len(feature_keys) >= 2 else None
            action_fallback = feature_keys[-1] if feature_keys else None

            resolved_item_name = _resolve(
                self._item_feature_name,
                self._resolved_item_feature_name,
                item_fallback,
            )
            if resolved_item_name is not None:
                self._resolved_item_feature_name = resolved_item_name
                item_ids = features[resolved_item_name].values()
            else:
                self._missing_item_ids_batches += 1
                if self._missing_item_ids_batches <= 3:
                    print_rank_0(
                        "[KV Replace] WARNING: item feature not found in batch keys "
                        f"{feature_keys}; item reuse will be skipped for this batch."
                    )

            resolved_action_name = _resolve(
                self._action_feature_name,
                self._resolved_action_feature_name,
                action_fallback,
            )
            if resolved_action_name is not None:
                self._resolved_action_feature_name = resolved_action_name
                action_ids = features[resolved_action_name].values()
            else:
                self._missing_action_ids_batches += 1
                if self._missing_action_ids_batches <= 3:
                    print_rank_0(
                        "[KV Replace] WARNING: action feature not found in batch keys "
                        f"{feature_keys}; action reuse will be skipped for this batch."
                    )
            self._replacer.set_current_token_ids(item_ids=item_ids, action_ids=action_ids)
            return batch

    if hasattr(unwrapped, "_item_feature_name"):
        item_feature_name = unwrapped._item_feature_name
    else:
        item_feature_name = None
    if hasattr(unwrapped, "_action_feature_name"):
        action_feature_name = unwrapped._action_feature_name
    else:
        action_feature_name = None

    all_comparison_rows = []
    all_overall_stats_rows = []
    for reuse_strategy in reuse_strategies:
        strategy_distances = (
            reuse_max_distances
            if reuse_strategy
            in {
                "same_id_max_distance",
                "same_id_max_distance_no_chain",
                "window_topk_same_id_max_distance",
            }
            and reuse_max_distances
            else [None]
        )
        for reuse_token_type in reuse_token_types:
            if reuse_strategy in {"wrong_id_same_window", "global_first_any_action_legacy"} and reuse_token_type != "action":
                continue
            for reuse_max_distance in strategy_distances:
                distance_suffix = (
                    ""
                    if reuse_max_distance is None
                    else f":dist<={reuse_max_distance if reuse_max_distance >= 0 else 'global'}"
                )
                file_distance_suffix = (
                    ""
                    if reuse_max_distance is None
                    else f"_dist_{reuse_max_distance if reuse_max_distance >= 0 else 'global'}"
                )
                reuse_mode = f"{reuse_strategy}:{reuse_token_type}{distance_suffix}"
                file_mode = f"{reuse_strategy}_{reuse_token_type}{file_distance_suffix}"
                print_rank_0(f"\n--- Evaluation with Selective KV Replacement ({reuse_mode}) ---")
                replacer = KVCacheReplacer(
                    num_layers,
                    reuse_policy=reuse_policy,
                    reuse_token_type=reuse_token_type,
                    reuse_strategy=reuse_strategy,
                    reuse_max_distance=reuse_max_distance,
                    replacement_impl=replacement_impl,
                )
                replacer.register_hooks(model_train)
                replacer.enable()
                print_rank_0(f"KV replacement enabled: {replacer._enabled}")
                print_rank_0(f"Number of hooks registered: {len(replacer._hooks)}")

                _reset_metric_module()

                base_iterated_eval_loader = islice(eval_dataloader, len(eval_dataloader))
                iterated_eval_loader = _TokenAwareEvalIter(
                    base_iterated_eval_loader,
                    replacer,
                    item_feature_name,
                    action_feature_name,
                )

                all_logits = []
                all_labels = []
                batch_count = 0

                with torch.no_grad():
                    for i in range(max_batches):
                        try:
                            reporting_loss, (local_loss, logits, labels, seqlen_info) = pipeline.progress(iterated_eval_loader)
                            all_logits.append(logits.detach())
                            all_labels.append(labels.detach())
                            batch_count += 1
                        except StopIteration:
                            break

                replacer.remove_hooks()
                replacer.disable()

                if not all_logits or not all_labels:
                    print_rank_0(f"No replaced data collected for mode={reuse_mode}.")
                    continue

                all_logits_replaced = torch.cat(all_logits, dim=0).to(device)
                all_labels_replaced = torch.cat(all_labels, dim=0).to(device)

                stateful_metric_module(all_logits_replaced, all_labels_replaced)
                if isinstance(stateful_metric_module, RetrievalTaskMetricWithSampling):
                    retrieval_gr = get_unwrapped_module(pipeline._model)
                    export_table_name = retrieval_gr.get_item_feature_table_name()
                    replaced_metrics, _, _ = stateful_metric_module.compute(
                        *retrieval_gr._embedding_collection.export_local_embedding(export_table_name)
                    )
                else:
                    replaced_metrics = stateful_metric_module.compute()

                print_rank_0(f"Replaced KV metrics ({reuse_mode}): {replaced_metrics}")

                print_rank_0("\n=== Metric Comparison ===")
                print_rank_0(f"{'Metric':<30} {'Baseline':<15} {'KV-Replaced':<15} {'Diff':<15}")
                print_rank_0("-" * 75)

                comparison_data = []
                for key in baseline_metrics:
                    if key in replaced_metrics:
                        baseline_val = baseline_metrics[key]
                        replaced_val = replaced_metrics[key]
                        if torch.is_tensor(baseline_val):
                            baseline_val = baseline_val.item()
                        if torch.is_tensor(replaced_val):
                            replaced_val = replaced_val.item()
                        diff = replaced_val - baseline_val
                        print_rank_0(f"{key:<30} {baseline_val:<15.6f} {replaced_val:<15.6f} {diff:<15.6f}")
                        row = {
                            "reuse_mode": reuse_mode,
                            "reuse_strategy": reuse_strategy,
                            "reuse_token_type": reuse_token_type,
                            "reuse_max_distance": reuse_max_distance,
                            "metric": key,
                            "baseline": baseline_val,
                            "kv_replaced": replaced_val,
                            "diff": diff,
                        }
                        comparison_data.append(row)
                        all_comparison_rows.append(row)

                if comparison_data:
                    df = pd.DataFrame(comparison_data)
                    mode_comparison_path = os.path.join(
                        output_dir, f"kv_replace_comparison_{file_mode}.csv"
                    )
                    df.to_csv(mode_comparison_path, index=False)
                    if len(reuse_token_types) == 1 and len(reuse_strategies) == 1:
                        df.to_csv(os.path.join(output_dir, "kv_replace_comparison.csv"), index=False)
                    print_rank_0(f"\nComparison saved to {mode_comparison_path}")

                stats = replacer.get_replacement_stats()
                print_rank_0("\n=== Replacement Stats ===")
                print_rank_0(
                    f"reuse_token_type={stats['reuse_token_type']}, "
                    f"default_window_size={stats['default_window_size']}, "
                    f"default_top_k={stats['default_top_k']}, "
                    f"candidate_reuse_token_count={stats['candidate_reuse_token_count']}, "
                    f"total_sequence_token_count={stats['total_sequence_token_count']}, "
                    f"selected_ratio_all_tokens={stats['selected_ratio_all_tokens']:.4f}, "
                    f"replacement_count={stats['replacement_count']}, "
                    f"total_considered_count={stats['total_considered_count']}, "
                    f"total_reuse_token_count={stats['total_reuse_token_count']}, "
                    f"total_windows={stats['total_windows']}, "
                    f"total_eligible_windows={stats['total_eligible_windows']}, "
                    f"total_selected_token_types={stats['total_selected_token_types']}, "
                    f"layers_with_first_kv={stats['layers_with_first_kv']}, "
                    f"global_replace_rate={stats['global_replace_rate']:.4f}"
                )
                print_rank_0(
                    f"{'Layer':<8} {'Windows':<10} {'Eligible':<10} {'Selected':<10} "
                    f"{'Candidates':<12} {'Replaced':<10} {'Rate':<10} {'WinAvg':<10} {'TopKAvg':<10}"
                )
                for layer_stat in stats["layer_stats"]:
                    print_rank_0(
                        f"{layer_stat['layer_idx']:<8d} "
                        f"{layer_stat['total_windows']:<10d} "
                        f"{layer_stat['eligible_windows']:<10d} "
                        f"{layer_stat['selected_token_types']:<10d} "
                        f"{layer_stat['candidate_tokens']:<12d} "
                        f"{layer_stat['replaced']:<10d} "
                        f"{layer_stat['replace_rate']:<10.4f} "
                        f"{layer_stat['window_size_avg']:<10.2f} "
                        f"{layer_stat['top_k_avg']:<10.2f}"
                    )

                layer_stats_df = pd.DataFrame(stats["layer_stats"])
                layer_stats_df.insert(0, "reuse_mode", reuse_mode)
                layer_stats_df.insert(1, "reuse_strategy", reuse_strategy)
                layer_stats_df.insert(0, "reuse_token_type", reuse_token_type)
                layer_stats_df.insert(3, "reuse_max_distance", reuse_max_distance)
                layer_stats_path = os.path.join(
                    output_dir, f"kv_replace_layer_stats_{file_mode}.csv"
                )
                layer_stats_df.to_csv(layer_stats_path, index=False)
                if len(reuse_token_types) == 1 and len(reuse_strategies) == 1:
                    layer_stats_df.to_csv(os.path.join(output_dir, "kv_replace_layer_stats.csv"), index=False)

                overall_stats = {
                    "reuse_mode": reuse_mode,
                    "reuse_strategy": stats["reuse_strategy"],
                    "reuse_token_type": stats["reuse_token_type"],
                    "reuse_max_distance": stats["reuse_max_distance"],
                    "replacement_impl": stats["replacement_impl"],
                    "default_window_size": stats["default_window_size"],
                    "default_top_k": stats["default_top_k"],
                    "policy_json": stats["policy_json"],
                    "selected_reuse_token_count": stats["selected_reuse_token_count"],
                    "candidate_reuse_token_count": stats["candidate_reuse_token_count"],
                    "selected_action_token_count": stats["selected_action_token_count"],
                    "candidate_action_token_count": stats["candidate_action_token_count"],
                    "total_sequence_token_count": stats["total_sequence_token_count"],
                    "total_layer_sequence_token_count": stats["total_layer_sequence_token_count"],
                    "selected_ratio_all_tokens": stats["selected_ratio_all_tokens"],
                    "selected_ratio_all_reuse_tokens": stats["selected_ratio_all_reuse_tokens"],
                    "selected_ratio_candidate_reuse_tokens": stats["selected_ratio_candidate_reuse_tokens"],
                    "selected_ratio_all_actions": stats["selected_ratio_all_actions"],
                    "selected_ratio_candidate_actions": stats["selected_ratio_candidate_actions"],
                    "replacement_count": stats["replacement_count"],
                    "total_reuse_token_count": stats["total_reuse_token_count"],
                    "total_layer_reuse_token_count": stats["total_layer_reuse_token_count"],
                    "total_action_count": stats["total_action_count"],
                    "total_layer_action_count": stats["total_layer_action_count"],
                    "total_replace_candidate_reuse_token_count": stats["total_replace_candidate_reuse_token_count"],
                    "total_replace_candidate_action_count": stats["total_replace_candidate_action_count"],
                    "total_considered_count": stats["total_considered_count"],
                    "total_windows": stats["total_windows"],
                    "total_eligible_windows": stats["total_eligible_windows"],
                    "total_selected_token_types": stats["total_selected_token_types"],
                    "total_selected_action_types": stats["total_selected_action_types"],
                    "layers_with_first_kv": stats["layers_with_first_kv"],
                    "global_replace_rate": stats["global_replace_rate"],
                }
                all_overall_stats_rows.append(overall_stats)
                mode_overall_stats_path = os.path.join(
                    output_dir, f"kv_replace_overall_stats_{file_mode}.csv"
                )
                pd.DataFrame([overall_stats]).to_csv(mode_overall_stats_path, index=False)
                if len(reuse_token_types) == 1 and len(reuse_strategies) == 1:
                    pd.DataFrame([overall_stats]).to_csv(
                        os.path.join(output_dir, "kv_replace_overall_stats.csv"), index=False
                    )
                print_rank_0(f"Saved layer stats to {layer_stats_path}")
                print_rank_0(f"Saved overall stats to {mode_overall_stats_path}")

    if all_comparison_rows:
        comparison_df = pd.DataFrame(all_comparison_rows)
        comparison_df.to_csv(os.path.join(output_dir, "reuse_auc_impact_by_task.csv"), index=False)
        auc_rows = comparison_df[
            comparison_df["metric"].astype(str).str.upper().str.contains("AUC")
        ]
        summary_source = auc_rows if not auc_rows.empty else comparison_df
        summary_df = (
            summary_source.groupby(
                ["reuse_mode", "reuse_strategy", "reuse_token_type", "reuse_max_distance"],
                dropna=False,
            )
            .agg(
                metric_count=("metric", "count"),
                mean_baseline=("baseline", "mean"),
                mean_kv_replaced=("kv_replaced", "mean"),
                mean_diff=("diff", "mean"),
                min_diff=("diff", "min"),
                max_abs_drop=("diff", lambda x: float(max(0.0, -x.min()))),
            )
            .reset_index()
        )
        if all_overall_stats_rows:
            stats_df = pd.DataFrame(all_overall_stats_rows)
            summary_df["reuse_max_distance"] = pd.to_numeric(
                summary_df["reuse_max_distance"], errors="coerce"
            )
            stats_df["reuse_max_distance"] = pd.to_numeric(
                stats_df["reuse_max_distance"], errors="coerce"
            )
            summary_df = summary_df.merge(
                stats_df[
                    [
                        "reuse_token_type",
                        "reuse_mode",
                        "reuse_strategy",
                        "reuse_max_distance",
                        "replacement_count",
                        "total_considered_count",
                        "candidate_reuse_token_count",
                        "total_reuse_token_count",
                        "global_replace_rate",
                        "selected_ratio_all_tokens",
                        "selected_ratio_all_reuse_tokens",
                    ]
                ],
                on=["reuse_mode", "reuse_strategy", "reuse_token_type", "reuse_max_distance"],
                how="left",
            )
            stats_df.to_csv(os.path.join(output_dir, "kv_replace_overall_stats_all_modes.csv"), index=False)
            _write_kv_replace_eval_markdown(
                output_dir=output_dir,
                comparison_df=comparison_df,
                stats_df=stats_df,
                auc_baseline_threshold=0.6,
                report_command=report_command,
            )
        summary_df.to_csv(os.path.join(output_dir, "reuse_auc_impact_summary.csv"), index=False)
        print_rank_0(
            f"Saved AUC impact summary to {os.path.join(output_dir, 'reuse_auc_impact_summary.csv')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Training Checkpoint with Analysis")
    parser.add_argument("--gin-config-file", type=str, required=True)
    parser.add_argument("--ckpt-load-dir", type=str, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-eval-iters", type=int, default=None)
    parser.add_argument("--max-retrieval-items", type=int, default=500,
                        help="Maximum number of items to use for retrieval evaluation. Default: 500")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save results CSV file")
    parser.add_argument("--analysis", type=str, required=True,
                        choices=["attention_viz", "kv_cache_diff", "kv_cache_replace"],
                        help="Type of analysis to perform")
    parser.add_argument("--output-dir", type=str, default="./analysis_output",
                        help="Directory to save analysis results")
    parser.add_argument(
        "--report-command",
        type=str,
        default=None,
        help=(
            "Optional exact shell command to embed in generated markdown reports. "
            "If omitted, the script records a reconstructed python command."
        ),
    )
    parser.add_argument(
        "--kv-replace-sim-threshold",
        type=float,
        default=0.90,
        help="Deprecated for kv_cache_replace: ignored by window/top-K action KV reuse.",
    )
    parser.add_argument(
        "--kv-replace-implementation",
        type=str,
        default="hidden_proxy",
        choices=["hidden_proxy", "kv_only"],
        help=(
            "For kv_cache_replace: hidden_proxy copies layer inputs before UVQK; "
            "kv_only copies only projected K/V rows and keeps Q/U unchanged."
        ),
    )
    parser.add_argument(
        "--kv-reuse-window-size",
        type=int,
        default=512,
        help="For kv_cache_replace: default local interleaved-token window size.",
    )
    parser.add_argument(
        "--kv-reuse-top-k",
        type=int,
        default=3,
        help="For kv_cache_replace: default number of high-frequency actions reused per user/window/layer.",
    )
    parser.add_argument(
        "--kv-reuse-policy-json",
        type=str,
        default=None,
        help=(
            "For kv_cache_replace: inline JSON object or path to JSON file with "
            "default/layers/user_length_buckets overrides."
        ),
    )
    parser.add_argument(
        "--kv-reuse-token-type",
        type=str,
        default="action",
        choices=["action", "item", "both"],
        help="For kv_cache_replace: which interleaved token type should use window/top-K KV reuse.",
    )
    parser.add_argument(
        "--kv-reuse-token-types",
        type=str,
        nargs="+",
        default=None,
        choices=["action", "item", "both"],
        help=(
            "For kv_cache_replace motivation sweeps: run multiple reuse modes in one process "
            "and write reuse_auc_impact_*.csv. Overrides --kv-reuse-token-type when set."
        ),
    )
    parser.add_argument(
        "--kv-reuse-strategy",
        type=str,
        default="window_topk_same_id",
        choices=[
            "window_topk_same_id",
            "window_topk_same_id_max_distance",
            "global_same_id",
            "global_topk_same_id",
            "wrong_id_same_window",
            "global_first_any_action_legacy",
            "same_id_max_distance",
            "same_id_max_distance_no_chain",
        ],
        help="For kv_cache_replace: how source positions are selected for KV reuse.",
    )
    parser.add_argument(
        "--kv-reuse-strategies",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "window_topk_same_id",
            "window_topk_same_id_max_distance",
            "global_same_id",
            "global_topk_same_id",
            "wrong_id_same_window",
            "global_first_any_action_legacy",
            "same_id_max_distance",
            "same_id_max_distance_no_chain",
        ],
        help="For kv_cache_replace motivation sweeps: run multiple reuse strategies.",
    )
    parser.add_argument(
        "--kv-reuse-max-distances",
        type=int,
        nargs="+",
        default=None,
        help=(
            "For same_id_max_distance sweeps: maximum interleaved-token distance "
            "from destination to previous same-id source. Use -1 for global."
        ),
    )
    args = parser.parse_args()
    reuse_policy = ActionKVReusePolicy.from_args(
        default_window_size=args.kv_reuse_window_size,
        default_top_k=args.kv_reuse_top_k,
        policy_json=args.kv_reuse_policy_json,
    )

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
    print_rank_0(f"Analysis mode: {args.analysis}")
    print_rank_0(f"Output directory: {args.output_dir}")
    if args.analysis == "kv_cache_replace":
        reuse_token_types = args.kv_reuse_token_types or [args.kv_reuse_token_type]
        reuse_strategies = args.kv_reuse_strategies or [args.kv_reuse_strategy]
        print_rank_0(
            "KV reuse defaults: "
            f"window_size={reuse_policy.default.window_size}, "
            f"top_k={reuse_policy.default.top_k}, "
            f"token_modes={reuse_token_types}, "
            f"strategies={reuse_strategies}, "
            f"max_distances={args.kv_reuse_max_distances}, "
            f"implementation={args.kv_replace_implementation}"
        )
        print_rank_0(
            "KV replace similarity threshold is deprecated and ignored: "
            f"{args.kv_replace_sim_threshold}"
        )

    # Initialize distributed
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)

    print_rank_0(f"Checkpoint dir: {trainer_args.ckpt_load_dir}")
    print_rank_0(f"Eval batch size: {trainer_args.eval_batch_size}")
    print_rank_0(f"Max eval iters: {trainer_args.max_eval_iters}")

    # Create model and config
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis - pass both model_train (DDP-wrapped) and model (unwrapped)
    if args.analysis == "attention_viz":
        run_attention_viz_analysis(
            model_train, model, eval_dataloader, stateful_metric_module, trainer_args, args.output_dir
        )
    elif args.analysis == "kv_cache_diff":
        run_kv_diff_analysis(
            model_train, model, eval_dataloader, stateful_metric_module, trainer_args, args.output_dir
        )
    elif args.analysis == "kv_cache_replace":
        run_kv_replace_analysis(
            model_train,
            model,
            eval_dataloader,
            stateful_metric_module,
            trainer_args,
            args.output_dir,
            reuse_policy,
            args.kv_replace_sim_threshold,
            args.kv_reuse_token_types or [args.kv_reuse_token_type],
            args.kv_reuse_strategies or [args.kv_reuse_strategy],
            args.kv_reuse_max_distances,
            args.kv_replace_implementation,
            args.report_command,
        )

    init.destroy_global_state()


if __name__ == "__main__":
    main()
