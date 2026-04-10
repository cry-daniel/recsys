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
3. kv_cache_replace: Replace KV Cache of repeated actions with first-occurrence KV Cache

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

    # KV Cache replacement evaluation
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        torchrun --nproc_per_node 1 --master_addr localhost \
        --master_port 6000 ./training/eval_checkpoint_analysis.py \
        --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
        --ckpt-load-dir ckpt_kr_1k_ranking/iter550 \
        --analysis kv_cache_replace \
        --output-dir ./analysis_output
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import argparse
import os
from typing import Dict, List, Optional, Tuple

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
                                k_i.view(1, -1), k_j.view(1, -1)
                            ).item()
                            v_cos_sim = F.cosine_similarity(
                                v_i.view(1, -1), v_j.view(1, -1)
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
                            sim = F.cosine_similarity(ki.view(1, -1), kj.view(1, -1)).item()
                            cross_pos_sims.append(sim)
            
            # Also compute within-position similarity for comparison
            within_pos_sims = []
            for cat in by_pos_cat:
                cat_keys = by_pos_cat[cat]["key"]
                for i in range(len(cat_keys)):
                    for j in range(i + 1, len(cat_keys)):
                        sim = F.cosine_similarity(cat_keys[i].view(1, -1), cat_keys[j].view(1, -1)).item()
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

class KVCacheReplacer:
    """
    Replaces KV Cache of repeated actions with first-occurrence KV Cache.

    This works by:
    1. During the first forward pass, storing the K/V for each action position
    2. For subsequent occurrences of the same action type, replacing their K/V
       with the first occurrence's K/V
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._hooks = []
        self._first_occurrence_kv: Dict[int, Dict[str, torch.Tensor]] = {}
        self._enabled = False
        self._replacement_count = 0
        self._total_action_count = 0
        self._layer_counter = {}

    def register_hooks(self, model):
        """Register forward pre-hooks for KV replacement."""
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
            print_rank_0(f"[KV Replace] Registering hook on layer {layer_idx}: {layer_type}")
            
            if isinstance(layer, FusedHSTULayer):
                def make_hook(li):
                    def hook_fn(mod, args, kwargs):
                        return self._replace_kv_fused(mod, args, li)
                    return hook_fn
                hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                self._hooks.append(hook)
            elif isinstance(layer, HSTULayer):
                self._layer_counter[layer_idx] = 0
                def make_hook(li):
                    def hook_fn(mod, args, kwargs):
                        return self._replace_kv(mod, args, li)
                    return hook_fn
                hook = layer.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
                self._hooks.append(hook)
            layer_idx += 1
        
        print_rank_0(f"[KV Replace] Total hooks registered: {len(self._hooks)}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def enable(self):
        """Enable KV replacement."""
        self._enabled = True
        self._first_occurrence_kv.clear()
        self._replacement_count = 0
        self._total_action_count = 0
        self._layer_counter = {}

    def disable(self):
        """Disable KV replacement."""
        self._enabled = False

    def _replace_kv(self, module, args, layer_idx):
        """Pre-hook to replace KV of repeated actions with first-occurrence KV (NativeHSTULayer)."""
        print_rank_0(f"[KV Replace DEBUG] _replace_kv called for layer {layer_idx}, enabled={self._enabled}")
        inputs = args
        if not self._enabled:
            print_rank_0(f"[KV Replace DEBUG] Skipping - not enabled")
            return None

        jd = inputs[0]
        if jd.values is None:
            return None

        hidden = jd.values
        has_interleaved = jd.has_interleaved_action
        seqlen_offsets = jd.seqlen_offsets

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

            if has_interleaved:
                batch_size = len(seqlen_offsets) - 1
                for user_idx in range(batch_size):
                    seq_start = seqlen_offsets[user_idx].item()
                    seq_end = seqlen_offsets[user_idx + 1].item()
                    seq_len = seq_end - seq_start

                    if seq_len < 3:
                        continue

                    first_action_pos = 1
                    first_action_abs = seq_start + first_action_pos

                    if first_action_abs >= key.shape[0]:
                        continue

                    if layer_idx not in self._first_occurrence_kv:
                        self._first_occurrence_kv[layer_idx] = {
                            "key": key[first_action_abs].detach().clone(),
                            "value": value[first_action_abs].detach().clone(),
                        }

                    first_key = self._first_occurrence_kv[layer_idx]["key"]
                    first_value = self._first_occurrence_kv[layer_idx]["value"]

                    for action_pos in range(3, seq_len, 2):
                        action_abs = seq_start + action_pos
                        if action_abs >= key.shape[0]:
                            break

                        key[action_abs].copy_(first_key)
                        value[action_abs].copy_(first_value)
                        self._replacement_count += 1

                    self._total_action_count += (seq_len - 1) // 2

        return None

    def _replace_kv_fused(self, module, args, layer_idx):
        """Pre-hook to replace KV of repeated actions with first-occurrence KV (FusedHSTULayer).
        
        Key insight: We must modify the INPUT hidden states, not the locally computed K/V.
        The layer recomputes Q,K,V from inputs after our hook returns, so modifying local
        K/V has no effect. Instead, we store the first action's hidden state and copy it
        to subsequent action positions.
        """
        inputs = args
        if not self._enabled:
            return None

        jd = inputs[0]
        if jd.values is None:
            return None

        hidden = jd.values
        has_interleaved = jd.has_interleaved_action
        seqlen_offsets = jd.seqlen_offsets

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

            if has_interleaved:
                batch_size = len(seqlen_offsets) - 1
                for user_idx in range(batch_size):
                    seq_start = seqlen_offsets[user_idx].item()
                    seq_end = seqlen_offsets[user_idx + 1].item()
                    seq_len = seq_end - seq_start

                    if seq_len < 3:
                        continue

                    first_action_pos = 1
                    first_action_abs = seq_start + first_action_pos

                    if first_action_abs >= hidden.shape[0]:
                        continue

                    # Store first occurrence's hidden state (not just K/V)
                    if layer_idx not in self._first_occurrence_kv:
                        self._first_occurrence_kv[layer_idx] = {
                            "hidden": normed_x[first_action_abs].detach().clone(),
                        }

                    # Replace subsequent action hidden states with first occurrence's hidden
                    # This ensures the layer's recomputation produces the same K/V
                    first_hidden = self._first_occurrence_kv[layer_idx]["hidden"]

                    for action_pos in range(3, seq_len, 2):
                        action_abs = seq_start + action_pos
                        if action_abs >= hidden.shape[0]:
                            break

                        # Modify the input hidden state - this is what actually affects the layer's computation
                        hidden[action_abs].copy_(first_hidden)
                        self._replacement_count += 1

                    self._total_action_count += (seq_len - 1) // 2

        return None

    def get_replacement_stats(self) -> Dict:
        """Get statistics about KV replacements."""
        return {
            "enabled": self._enabled,
            "layers_with_first_kv": len(self._first_occurrence_kv),
            "replacement_count": self._replacement_count,
            "total_action_count": self._total_action_count,
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


def run_kv_replace_analysis(
    model_train, model, eval_dataloader, stateful_metric_module, trainer_args, output_dir: str
):
    """Run KV Cache replacement analysis."""
    print_rank_0("=== Running KV Cache Replacement Analysis ===")

    unwrapped = get_unwrapped_module(model)
    hstu_config = unwrapped._hstu_config
    num_layers = hstu_config.num_layers

    replacer = KVCacheReplacer(num_layers)

    # Create pipeline using the already-wrapped model_train
    device = torch.device("cuda", torch.cuda.current_device())
    pipeline = JaggedMegatronTrainNonePipeline(
        model_train,
        torch.optim.Adam(model.parameters(), lr=1e-5),
        device=device,
    )
    pipeline._model.eval()

    # First, run baseline evaluation (no replacement) - collect logits/labels
    print_rank_0("\n--- Baseline Evaluation (No KV Replacement) ---")

    # Use pipeline progress for baseline evaluation
    from itertools import islice
    max_batches = min(10, len(eval_dataloader))
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
        baseline_metrics = stateful_metric_module.compute()
        print_rank_0(f"Baseline metrics: {baseline_metrics}")
    else:
        print_rank_0("No baseline data collected.")
        return

    # Now run with KV replacement
    print_rank_0("\n--- Evaluation with KV Replacement (First Occurrence) ---")

    # Register replacement hooks on model_train since that's what's used for forward
    # The hooks need to be on the actual module that runs forward()
    replacer.register_hooks(model_train)
    replacer.enable()
    print_rank_0(f"KV replacement enabled: {replacer._enabled}")
    print_rank_0(f"Number of hooks registered: {len(replacer._hooks)}")
    print_rank_0(f"First occurrence KV layers: {list(replacer._first_occurrence_kv.keys())}")

    # Reset metric module for replaced evaluation
    # Different metric modules may have different reset methods
    # Try to find and call reset on internal metric modules
    reset_found = False
    if hasattr(stateful_metric_module, 'reset'):
        stateful_metric_module.reset()
        reset_found = True
    elif hasattr(stateful_metric_module, '_reset'):
        stateful_metric_module._reset()
        reset_found = True
    else:
        # For MultiClassificationTaskMetric, try to reset internal metrics
        # Check for common attribute patterns
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
        # Last resort: manually reset common torchmetrics attributes
        for attr_name in ['preds', 'target', '_update_called']:
            if hasattr(stateful_metric_module, attr_name):
                val = getattr(stateful_metric_module, attr_name)
                if isinstance(val, list):
                    val.clear()
                elif isinstance(val, torch.Tensor):
                    setattr(stateful_metric_module, attr_name, [])

    # Use pipeline progress for KV replacement evaluation
    iterated_eval_loader = islice(eval_dataloader, len(eval_dataloader))
    
    all_logits = []
    all_labels = []
    batch_count = 0

    with torch.no_grad():
        for i in range(max_batches):
            try:
                reporting_loss, (local_loss, logits, labels, seqlen_info) = pipeline.progress(iterated_eval_loader)
                # Keep on CUDA device for distributed sync
                all_logits.append(logits.detach())
                all_labels.append(labels.detach())
                batch_count += 1
            except StopIteration:
                break

    # Remove hooks
    replacer.remove_hooks()
    replacer.disable()

    # Compute metrics with replaced KV - keep tensors on CUDA device
    if all_logits and all_labels:
        all_logits_replaced = torch.cat(all_logits, dim=0).to(device)
        all_labels_replaced = torch.cat(all_labels, dim=0).to(device)

        # Use the metric module to compute replaced
        stateful_metric_module(all_logits_replaced, all_labels_replaced)
        replaced_metrics = stateful_metric_module.compute()

        print_rank_0(f"Replaced KV metrics: {replaced_metrics}")

        # Compare baseline vs replaced
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
                comparison_data.append({
                    "metric": key,
                    "baseline": baseline_val,
                    "kv_replaced": replaced_val,
                    "diff": diff,
                })

        # Save comparison
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df.to_csv(os.path.join(output_dir, "kv_replace_comparison.csv"), index=False)
            print_rank_0(f"\nComparison saved to {os.path.join(output_dir, 'kv_replace_comparison.csv')}")

    # Print replacement stats
    stats = replacer.get_replacement_stats()
    print_rank_0(f"\nReplacement stats: {stats}")


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
            model_train, model, eval_dataloader, stateful_metric_module, trainer_args, args.output_dir
        )

    init.destroy_global_state()


if __name__ == "__main__":
    main()