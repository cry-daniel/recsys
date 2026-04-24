# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional real-KV motivation analysis for action reuse.

This module is imported only when `analyze_action_reuse_motivation.py` is run
with `--run-kv-analysis`, keeping the default data-side path lightweight.
"""

import json
import os
from itertools import combinations, islice
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import eval_checkpoint_action_kv_replace as base


class KVMotivationCaptureHook:
    """Capture per-layer K/V tensors and item/action IDs for similarity analysis."""

    def __init__(self, max_batches: int, max_users_per_batch: int, max_actions_per_user: int):
        self.max_batches = max_batches
        self.max_users_per_batch = max_users_per_batch
        self.max_actions_per_user = max_actions_per_user
        self.records: List[Dict] = []
        self._hooks = []
        self._current_item_ids: Optional[torch.Tensor] = None
        self._current_action_ids: Optional[torch.Tensor] = None
        self._current_batch_idx = -1

    def set_current_batch(
        self,
        item_ids: Optional[torch.Tensor],
        action_ids: Optional[torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._current_batch_idx = batch_idx
        self._current_item_ids = item_ids.detach().cpu().clone() if item_ids is not None else None
        self._current_action_ids = action_ids.detach().cpu().clone() if action_ids is not None else None

    def register_hooks(self, model) -> None:
        from modules.fused_hstu_layer import FusedHSTULayer
        from modules.native_hstu_layer import HSTULayer

        unwrapped = base.get_unwrapped_module(model)
        hstu_block = unwrapped._hstu_block
        for layer_idx, layer in enumerate(hstu_block._attention_layers):
            if isinstance(layer, FusedHSTULayer):
                hook = layer.register_forward_pre_hook(
                    self._make_hook(layer_idx, fused=True), with_kwargs=True
                )
                self._hooks.append(hook)
            elif isinstance(layer, HSTULayer):
                hook = layer.register_forward_pre_hook(
                    self._make_hook(layer_idx, fused=False), with_kwargs=True
                )
                self._hooks.append(hook)

    def _make_hook(self, layer_idx: int, fused: bool):
        def hook_fn(mod, args, kwargs):
            return self._capture(mod, args, layer_idx, fused=fused)

        return hook_fn

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _capture(self, module, args, layer_idx: int, fused: bool):
        if self._current_batch_idx < 0 or self._current_batch_idx >= self.max_batches:
            return None

        jd = args[0]
        if jd.values is None or not jd.has_interleaved_action:
            return None

        hidden = jd.values
        with torch.no_grad():
            normed_x = self._input_norm(module, hidden)
            if fused:
                key, value = self._compute_fused_kv(module, normed_x)
            else:
                key, value = self._compute_native_kv(module, normed_x)

            seqlen_offsets = jd.seqlen_offsets
            limited_users = min(self.max_users_per_batch, len(seqlen_offsets) - 1)
            last_offset = int(seqlen_offsets[limited_users].item())
            self.records.append(
                {
                    "batch_idx": self._current_batch_idx,
                    "layer_idx": layer_idx,
                    "key": key[:last_offset].detach().cpu().clone(),
                    "value": value[:last_offset].detach().cpu().clone(),
                    "seqlen_offsets": seqlen_offsets[: limited_users + 1].detach().cpu().clone(),
                    "contextual_seqlen": jd.contextual_seqlen[:limited_users].detach().cpu().clone()
                    if jd.contextual_seqlen is not None
                    else None,
                    "item_ids": self._current_item_ids,
                    "action_ids": self._current_action_ids,
                }
            )
        return None

    @staticmethod
    def _input_norm(module, hidden: torch.Tensor) -> torch.Tensor:
        if module._input_layernorm_weight is None:
            return hidden
        return F.layer_norm(
            hidden,
            normalized_shape=[module._embedding_dim],
            weight=module._input_layernorm_weight,
            bias=module._input_layernorm_bias,
            eps=module._eps,
        )

    @staticmethod
    def _compute_native_kv(module, normed_x: torch.Tensor):
        mixed_uvqk, _ = module._linear_uvqk(normed_x)
        silu_uvqk = F.silu(mixed_uvqk)
        split_sizes = module._split_arg_list
        silu_uvqk = silu_uvqk.view(-1, module._num_heads_per_partition, sum(split_sizes))
        _, value, _, key = torch.split(silu_uvqk, split_sizes, dim=-1)
        return key, value

    @staticmethod
    def _compute_fused_kv(module, normed_x: torch.Tensor):
        mixed_uvqk = F.linear(normed_x, module._linear_uvqk_weight.t(), module._linear_uvqk_bias)
        silu_uvqk = F.silu(mixed_uvqk)
        total_dim_per_head = module._linear_dim_per_head * 2 + module._attention_dim_per_head * 2
        silu_uvqk = silu_uvqk.view(-1, module._num_heads, total_dim_per_head)
        split_sizes = [
            module._linear_dim_per_head,
            module._linear_dim_per_head,
            module._attention_dim_per_head,
            module._attention_dim_per_head,
        ]
        _, value, _, key = torch.split(silu_uvqk, split_sizes, dim=-1)
        return key, value


def _extract_feature_ids(batch, feature_name: Optional[str], fallback_index: int):
    features = batch.features
    feature_keys = list(features.keys())
    candidates = []
    if feature_name is not None:
        candidates.append(feature_name)
    if fallback_index < 0 and len(feature_keys) >= abs(fallback_index):
        candidates.append(feature_keys[fallback_index])
    elif fallback_index >= 0 and len(feature_keys) > fallback_index:
        candidates.append(feature_keys[fallback_index])
    for name in candidates:
        if name is not None and name in features.keys():
            return features[name].values()
    return None


DISTANCE_BUCKETS = (128, 256, 512, 1024)


def _distance_bucket(distance: int, window_size: int) -> str:
    del window_size
    prev = 0
    for boundary in DISTANCE_BUCKETS:
        if distance <= boundary:
            return f"{prev + 1}-{boundary}" if prev else f"<= {boundary}"
        prev = boundary
    return f"> {DISTANCE_BUCKETS[-1]}"


def _pair_metrics(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    left = left.reshape(1, -1).float()
    right = right.reshape(1, -1).float()
    left_centered = left - left.mean(dim=1, keepdim=True)
    right_centered = right - right.mean(dim=1, keepdim=True)
    avg_norm = 0.5 * (left.norm(dim=1) + right.norm(dim=1)).clamp_min(1e-12)
    return {
        "cosine": F.cosine_similarity(left, right).item(),
        "centered_cosine": F.cosine_similarity(left_centered, right_centered).item(),
        "relative_l2": ((left - right).norm(dim=1) / avg_norm).item(),
    }


def _collect_token_refs(
    *,
    token_type: str,
    token_ids: Optional[torch.Tensor],
    token_offset: int,
    seq_start: int,
    contextual_len: int,
    action_item_span: int,
    key_len: int,
    max_tokens_per_user: int,
) -> List[Dict]:
    if token_ids is None:
        return []

    first_rel_pos = 0 if token_type == "item" else 1
    rel_positions = list(range(first_rel_pos, action_item_span, 2))
    if len(rel_positions) > max_tokens_per_user:
        sampled_indices = np.linspace(
            0,
            len(rel_positions) - 1,
            num=max_tokens_per_user,
            dtype=np.int64,
        )
        rel_positions = [rel_positions[int(idx)] for idx in sampled_indices]

    refs = []
    for rel_pos in rel_positions:
        token_idx = token_offset + rel_pos // 2
        if token_idx >= token_ids.shape[0]:
            continue
        local_pos = contextual_len + rel_pos
        abs_pos = seq_start + local_pos
        if abs_pos >= key_len:
            continue
        refs.append(
            {
                "token_type": token_type,
                "token_id": int(token_ids[token_idx].item()),
                "local_pos": local_pos,
                "abs_pos": abs_pos,
            }
        )
    return refs


def _iter_capped_pairs(refs: List[Dict], max_pairs_per_token_id: int):
    pairs = list(combinations(refs, 2))
    if max_pairs_per_token_id <= 0 or len(pairs) <= max_pairs_per_token_id:
        return pairs
    idx = np.linspace(
        0,
        len(pairs) - 1,
        num=max_pairs_per_token_id,
        dtype=np.int64,
    )
    return [pairs[int(i)] for i in idx]


def _records_to_pair_frame(
    records: List[Dict],
    window_size: int,
    max_tokens_per_user: int,
    max_pairs_per_token_id: int = 0,
) -> pd.DataFrame:
    rows = []
    for record in records:
        key = record["key"]
        value = record["value"]
        seqlen_offsets = record["seqlen_offsets"]
        contextual = record["contextual_seqlen"]
        item_ids = record["item_ids"]
        action_ids = record["action_ids"]
        item_offset = 0
        action_offset = 0
        for user_idx in range(len(seqlen_offsets) - 1):
            seq_start = int(seqlen_offsets[user_idx].item())
            seq_end = int(seqlen_offsets[user_idx + 1].item())
            contextual_len = int(contextual[user_idx].item()) if contextual is not None else 0
            action_item_span = max(0, seq_end - seq_start - contextual_len)

            token_refs = []
            token_refs.extend(
                _collect_token_refs(
                    token_type="item",
                    token_ids=item_ids,
                    token_offset=item_offset,
                    seq_start=seq_start,
                    contextual_len=contextual_len,
                    action_item_span=action_item_span,
                    key_len=key.shape[0],
                    max_tokens_per_user=max_tokens_per_user,
                )
            )
            token_refs.extend(
                _collect_token_refs(
                    token_type="action",
                    token_ids=action_ids,
                    token_offset=action_offset,
                    seq_start=seq_start,
                    contextual_len=contextual_len,
                    action_item_span=action_item_span,
                    key_len=key.shape[0],
                    max_tokens_per_user=max_tokens_per_user,
                )
            )

            by_token: Dict[tuple, List[Dict]] = {}
            for ref in token_refs:
                by_token.setdefault((ref["token_type"], ref["token_id"]), []).append(ref)
            for (token_type, token_id), refs in by_token.items():
                for left, right in _iter_capped_pairs(refs, max_pairs_per_token_id):
                    same_window = (left["local_pos"] // window_size) == (
                        right["local_pos"] // window_size
                    )
                    pos_distance = right["local_pos"] - left["local_pos"]
                    k_metrics = _pair_metrics(key[left["abs_pos"]], key[right["abs_pos"]])
                    v_metrics = _pair_metrics(value[left["abs_pos"]], value[right["abs_pos"]])
                    rows.append(
                        {
                            "batch_idx": record["batch_idx"],
                            "layer_idx": record["layer_idx"],
                            "user_idx": user_idx,
                            "token_type": token_type,
                            "token_id": token_id,
                            "pos_i": left["local_pos"],
                            "pos_j": right["local_pos"],
                            "pos_distance": pos_distance,
                            "distance_bucket": _distance_bucket(pos_distance, window_size),
                            "same_window": same_window,
                            "k_cosine": k_metrics["cosine"],
                            "k_centered_cosine": k_metrics["centered_cosine"],
                            "k_relative_l2": k_metrics["relative_l2"],
                            "v_cosine": v_metrics["cosine"],
                            "v_centered_cosine": v_metrics["centered_cosine"],
                            "v_relative_l2": v_metrics["relative_l2"],
                        }
                    )
            item_offset += (action_item_span + 1) // 2
            action_offset += action_item_span // 2
    return pd.DataFrame(rows)


def _sample_different_action_pairs(
    refs: List[Dict],
    max_pairs: int,
    rng: np.random.Generator,
) -> List[tuple]:
    if len(refs) < 2 or max_pairs <= 0:
        return []
    pairs = []
    attempts = 0
    max_attempts = max_pairs * 20
    while len(pairs) < max_pairs and attempts < max_attempts:
        attempts += 1
        left_idx, right_idx = rng.choice(len(refs), size=2, replace=False)
        left = refs[int(left_idx)]
        right = refs[int(right_idx)]
        if left["token_id"] == right["token_id"]:
            continue
        if left["local_pos"] > right["local_pos"]:
            left, right = right, left
        pairs.append((left, right))
    return pairs


def _records_to_action_same_diff_pair_frame(
    records: List[Dict],
    window_size: int,
    max_actions_per_user: int,
    max_same_pairs_per_action_id: int,
    max_different_pairs_per_user: int,
    random_seed: int = 2025,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_seed)
    for record in records:
        key = record["key"]
        value = record["value"]
        seqlen_offsets = record["seqlen_offsets"]
        contextual = record["contextual_seqlen"]
        action_ids = record["action_ids"]
        action_offset = 0
        for user_idx in range(len(seqlen_offsets) - 1):
            seq_start = int(seqlen_offsets[user_idx].item())
            seq_end = int(seqlen_offsets[user_idx + 1].item())
            contextual_len = int(contextual[user_idx].item()) if contextual is not None else 0
            action_item_span = max(0, seq_end - seq_start - contextual_len)
            action_refs = _collect_token_refs(
                token_type="action",
                token_ids=action_ids,
                token_offset=action_offset,
                seq_start=seq_start,
                contextual_len=contextual_len,
                action_item_span=action_item_span,
                key_len=key.shape[0],
                max_tokens_per_user=max_actions_per_user,
            )

            pair_specs = []
            by_action: Dict[int, List[Dict]] = {}
            for ref in action_refs:
                by_action.setdefault(ref["token_id"], []).append(ref)
            for refs in by_action.values():
                for left, right in _iter_capped_pairs(refs, max_same_pairs_per_action_id):
                    pair_specs.append((True, left, right))
            for left, right in _sample_different_action_pairs(
                action_refs,
                max_different_pairs_per_user,
                rng,
            ):
                pair_specs.append((False, left, right))

            for same_action, left, right in pair_specs:
                same_window = (left["local_pos"] // window_size) == (
                    right["local_pos"] // window_size
                )
                pos_distance = right["local_pos"] - left["local_pos"]
                k_metrics = _pair_metrics(key[left["abs_pos"]], key[right["abs_pos"]])
                v_metrics = _pair_metrics(value[left["abs_pos"]], value[right["abs_pos"]])
                rows.append(
                    {
                        "batch_idx": record["batch_idx"],
                        "layer_idx": record["layer_idx"],
                        "user_idx": user_idx,
                        "same_action": same_action,
                        "action_id_i": left["token_id"],
                        "action_id_j": right["token_id"],
                        "pos_i": left["local_pos"],
                        "pos_j": right["local_pos"],
                        "pos_distance": pos_distance,
                        "distance_bucket": _distance_bucket(pos_distance, window_size),
                        "same_window": same_window,
                        "k_cosine": k_metrics["cosine"],
                        "k_centered_cosine": k_metrics["centered_cosine"],
                        "k_relative_l2": k_metrics["relative_l2"],
                        "v_cosine": v_metrics["cosine"],
                        "v_centered_cosine": v_metrics["centered_cosine"],
                        "v_relative_l2": v_metrics["relative_l2"],
                    }
                )
            action_offset += action_item_span // 2
    return pd.DataFrame(rows)


def _build_action_same_diff_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    return (
        pair_df.groupby(["same_action", "layer_idx", "distance_bucket"], sort=False)
        .agg(
            k_cosine_mean=("k_cosine", "mean"),
            k_cosine_median=("k_cosine", "median"),
            k_cosine_p10=("k_cosine", lambda x: x.quantile(0.10)),
            k_cosine_p90=("k_cosine", lambda x: x.quantile(0.90)),
            k_centered_cosine_mean=("k_centered_cosine", "mean"),
            k_relative_l2_mean=("k_relative_l2", "mean"),
            v_cosine_mean=("v_cosine", "mean"),
            v_cosine_median=("v_cosine", "median"),
            v_cosine_p10=("v_cosine", lambda x: x.quantile(0.10)),
            v_cosine_p90=("v_cosine", lambda x: x.quantile(0.90)),
            v_centered_cosine_mean=("v_centered_cosine", "mean"),
            v_relative_l2_mean=("v_relative_l2", "mean"),
            pos_distance_mean=("pos_distance", "mean"),
            pair_count=("k_cosine", "count"),
        )
        .reset_index()
    )


def _plot_action_same_diff_outputs(pair_df: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    if pair_df.empty:
        return

    layer_summary = (
        pair_df.groupby(["same_action", "layer_idx"])["k_cosine"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for same_action, group in layer_summary.groupby("same_action"):
        ax.plot(
            group["layer_idx"],
            group["k_cosine"],
            marker="o",
            label=f"same_action={same_action}",
        )
    ax.set_title("Action K cosine: same action vs different action")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean K cosine")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_kv_same_vs_diff_by_layer.png"), dpi=160)
    plt.close(fig)


def _build_similarity_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    return (
        pair_df.groupby(["token_type", "layer_idx", "same_window", "distance_bucket"])
        .agg(
            k_cosine_mean=("k_cosine", "mean"),
            k_cosine_median=("k_cosine", "median"),
            k_cosine_p10=("k_cosine", lambda x: x.quantile(0.10)),
            k_cosine_p90=("k_cosine", lambda x: x.quantile(0.90)),
            k_cosine_std=("k_cosine", "std"),
            k_centered_cosine_mean=("k_centered_cosine", "mean"),
            k_relative_l2_mean=("k_relative_l2", "mean"),
            v_cosine_mean=("v_cosine", "mean"),
            v_cosine_median=("v_cosine", "median"),
            v_cosine_p10=("v_cosine", lambda x: x.quantile(0.10)),
            v_cosine_p90=("v_cosine", lambda x: x.quantile(0.90)),
            v_cosine_std=("v_cosine", "std"),
            v_centered_cosine_mean=("v_centered_cosine", "mean"),
            v_relative_l2_mean=("v_relative_l2", "mean"),
            pos_distance_mean=("pos_distance", "mean"),
            pair_count=("k_cosine", "count"),
        )
        .reset_index()
    )


def _build_matched_item_action_summary(pair_df: pd.DataFrame, random_seed: int = 2025) -> pd.DataFrame:
    if pair_df.empty or "item" not in set(pair_df["token_type"]) or "action" not in set(pair_df["token_type"]):
        return pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    matched_parts = []
    group_cols = ["layer_idx", "same_window", "distance_bucket"]
    for _, group in pair_df.groupby(group_cols):
        item_group = group[group["token_type"] == "item"]
        action_group = group[group["token_type"] == "action"]
        n = min(len(item_group), len(action_group))
        if n == 0:
            continue
        matched_parts.append(item_group.iloc[rng.choice(len(item_group), size=n, replace=False)])
        matched_parts.append(action_group.iloc[rng.choice(len(action_group), size=n, replace=False)])
    if not matched_parts:
        return pd.DataFrame()

    matched_df = pd.concat(matched_parts, ignore_index=True)
    summary = (
        matched_df.groupby(["layer_idx", "same_window", "distance_bucket", "token_type"])
        .agg(
            k_cosine_mean=("k_cosine", "mean"),
            k_cosine_median=("k_cosine", "median"),
            k_centered_cosine_mean=("k_centered_cosine", "mean"),
            k_relative_l2_mean=("k_relative_l2", "mean"),
            v_cosine_mean=("v_cosine", "mean"),
            v_cosine_median=("v_cosine", "median"),
            v_centered_cosine_mean=("v_centered_cosine", "mean"),
            v_relative_l2_mean=("v_relative_l2", "mean"),
            pair_count=("k_cosine", "count"),
        )
        .reset_index()
    )
    pivot = summary.pivot_table(
        index=["layer_idx", "same_window", "distance_bucket"],
        columns="token_type",
        values=["k_cosine_mean", "k_cosine_median", "v_cosine_mean", "v_cosine_median", "pair_count"],
    )
    pivot.columns = [f"{metric}_{token_type}" for metric, token_type in pivot.columns]
    pivot = pivot.reset_index()
    if "k_cosine_mean_action" in pivot and "k_cosine_mean_item" in pivot:
        pivot["k_cosine_mean_action_minus_item"] = (
            pivot["k_cosine_mean_action"] - pivot["k_cosine_mean_item"]
        )
    if "v_cosine_mean_action" in pivot and "v_cosine_mean_item" in pivot:
        pivot["v_cosine_mean_action_minus_item"] = (
            pivot["v_cosine_mean_action"] - pivot["v_cosine_mean_item"]
        )
    return pivot


def _plot_kv_outputs(pair_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    if pair_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    pair_df.boxplot(column="k_cosine", by=["token_type", "same_window"], ax=ax, rot=30)
    ax.set_title("K cosine: item/action same-token pairs")
    ax.set_xlabel("token_type, same_window")
    ax.set_ylabel("K cosine")
    fig.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "item_action_kv_similarity_boxplot.png"), dpi=160)
    plt.close(fig)

    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        layer_summary = (
            pair_df.groupby(["token_type", "layer_idx"])["k_cosine"]
            .mean()
            .reset_index()
        )
        for token_type, group in layer_summary.groupby("token_type"):
            ax.plot(
                group["layer_idx"],
                group["k_cosine"],
                marker="o",
                label=token_type,
            )
        ax.set_title("Layer-wise K cosine: action vs item")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean K cosine")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "item_vs_action_kv_similarity_by_layer.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        distance_summary = (
            pair_df.groupby(["token_type", "distance_bucket"])["k_cosine"]
            .mean()
            .reset_index()
        )
        for token_type, group in distance_summary.groupby("token_type"):
            ax.plot(
                group["distance_bucket"],
                group["k_cosine"],
                marker="o",
                label=token_type,
            )
        ax.set_title("K cosine by position distance bucket")
        ax.set_xlabel("Distance bucket")
        ax.set_ylabel("Mean K cosine")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "item_vs_action_kv_similarity_by_distance.png"), dpi=160)
        plt.close(fig)


def run_kv_motivation_analysis(args) -> None:
    if hasattr(base.gin, "clear_config"):
        base.gin.clear_config()
    base.gin.parse_config_file(args.gin_config_file)

    trainer_args = base.TrainerArgs()
    dataset_args, embedding_args = base.get_dataset_and_embedding_args()
    network_args = base.NetworkArgs()
    optimizer_args = base.OptimizerArgs()
    tp_args = base.TensorModelParallelArgs()
    trainer_args.ckpt_load_dir = args.ckpt_load_dir
    if args.auc_max_eval_iters is not None:
        trainer_args.max_eval_iters = args.auc_max_eval_iters

    base.init.initialize_distributed()
    base.init.initialize_model_parallel(tensor_model_parallel_size=tp_args.tensor_model_parallel_size)
    base.init.set_random_seed(trainer_args.seed)

    hstu_config = base.create_hstu_config(network_args, tp_args)
    is_retrieval = base.is_retrieval_task()
    if is_retrieval:
        task_config = base.create_retrieval_config(dataset_args, network_args, embedding_args)
        model = base.get_retrieval_model(hstu_config=hstu_config, task_config=task_config)
    else:
        task_config = base.create_ranking_config(dataset_args, network_args, embedding_args)
        model = base.get_ranking_model(hstu_config=hstu_config, task_config=task_config)

    dynamic_options_dict = base.create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=base.get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )
    optimizer_param = base.create_optimizer_params(optimizer_args)
    model_train, dense_optimizer = base.make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )

    if is_retrieval:
        _, eval_dataloader = base.get_data_loader("retrieval", dataset_args, trainer_args, 0)
    else:
        _, eval_dataloader = base.get_data_loader(
            "ranking", dataset_args, trainer_args, task_config.num_tasks
        )

    base.maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)
    device = torch.device("cuda", torch.cuda.current_device())
    pipeline = base.JaggedMegatronTrainNonePipeline(
        model_train,
        torch.optim.Adam(model.parameters(), lr=1e-5),
        device=device,
    )
    pipeline._model.eval()

    unwrapped = base.get_unwrapped_module(model)
    item_feature_name = getattr(unwrapped, "_item_feature_name", None)
    action_feature_name = getattr(unwrapped, "_action_feature_name", None)
    capture = KVMotivationCaptureHook(
        max_batches=args.kv_max_batches,
        max_users_per_batch=args.kv_max_users_per_batch,
        max_actions_per_user=args.kv_max_actions_per_user,
    )
    capture.register_hooks(model_train)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(islice(eval_dataloader, args.kv_max_batches)):
                item_ids = _extract_feature_ids(batch, item_feature_name, -2)
                action_ids = _extract_feature_ids(batch, action_feature_name, -1)
                capture.set_current_batch(item_ids, action_ids, batch_idx)
                batch = batch.to(device)
                pipeline._model(batch)
    finally:
        capture.remove_hooks()
        base.init.destroy_global_state()

    pair_df = _records_to_pair_frame(
        capture.records,
        window_size=args.kv_window_size,
        max_tokens_per_user=args.kv_max_actions_per_user,
        max_pairs_per_token_id=args.kv_max_pairs_per_token_id,
    )
    action_same_diff_df = _records_to_action_same_diff_pair_frame(
        capture.records,
        window_size=args.kv_window_size,
        max_actions_per_user=args.kv_max_actions_per_user,
        max_same_pairs_per_action_id=args.kv_max_pairs_per_token_id,
        max_different_pairs_per_user=args.kv_max_different_action_pairs_per_user,
    )
    action_same_diff_path = os.path.join(args.output_dir, "action_kv_same_vs_diff_pairs.csv")
    action_same_diff_df.to_csv(action_same_diff_path, index=False)
    action_same_diff_summary_df = _build_action_same_diff_summary(action_same_diff_df)
    action_same_diff_summary_df.to_csv(
        os.path.join(args.output_dir, "action_kv_same_vs_diff_summary.csv"),
        index=False,
    )
    _plot_action_same_diff_outputs(action_same_diff_df, args.output_dir)

    pair_path = os.path.join(args.output_dir, "item_action_kv_similarity_pairs.csv")
    pair_df.to_csv(pair_path, index=False)
    pair_df.to_csv(os.path.join(args.output_dir, "kv_pair_similarity.csv"), index=False)
    if pair_df.empty:
        pd.DataFrame().to_csv(os.path.join(args.output_dir, "kv_similarity_summary.csv"), index=False)
        pd.DataFrame().to_csv(
            os.path.join(args.output_dir, "item_action_kv_similarity_summary.csv"),
            index=False,
        )
        pd.DataFrame().to_csv(
            os.path.join(args.output_dir, "item_action_kv_similarity_matched_summary.csv"),
            index=False,
        )
        return

    summary_df = _build_similarity_summary(pair_df)
    matched_summary_df = _build_matched_item_action_summary(pair_df)
    summary_df.to_csv(os.path.join(args.output_dir, "kv_similarity_summary.csv"), index=False)
    summary_df.to_csv(
        os.path.join(args.output_dir, "item_action_kv_similarity_summary.csv"),
        index=False,
    )
    matched_summary_df.to_csv(
        os.path.join(args.output_dir, "item_action_kv_similarity_matched_summary.csv"),
        index=False,
    )
    _plot_kv_outputs(pair_df, summary_df, args.output_dir)


def run_auc_impact_analysis(args) -> None:
    """Run checkpoint AUC impact comparison for action/item/both reuse modes."""
    if hasattr(base.gin, "clear_config"):
        base.gin.clear_config()
    base.gin.parse_config_file(args.gin_config_file)

    trainer_args = base.TrainerArgs()
    dataset_args, embedding_args = base.get_dataset_and_embedding_args()
    network_args = base.NetworkArgs()
    optimizer_args = base.OptimizerArgs()
    tp_args = base.TensorModelParallelArgs()
    trainer_args.ckpt_load_dir = args.ckpt_load_dir
    if args.auc_max_eval_iters is not None:
        trainer_args.max_eval_iters = args.auc_max_eval_iters

    base.init.initialize_distributed()
    base.init.initialize_model_parallel(tensor_model_parallel_size=tp_args.tensor_model_parallel_size)
    base.init.set_random_seed(trainer_args.seed)

    try:
        hstu_config = base.create_hstu_config(network_args, tp_args)
        is_retrieval = base.is_retrieval_task()
        if is_retrieval:
            task_config = base.create_retrieval_config(dataset_args, network_args, embedding_args)
            model = base.get_retrieval_model(hstu_config=hstu_config, task_config=task_config)
        else:
            task_config = base.create_ranking_config(dataset_args, network_args, embedding_args)
            model = base.get_ranking_model(hstu_config=hstu_config, task_config=task_config)

        dynamic_options_dict = base.create_dynamic_optitons_dict(
            embedding_args,
            network_args.hidden_size,
            training=True,
            embedding_dim_multiplier=base.get_embedding_vector_storage_multiplier(
                optimizer_args.optimizer_str
            ),
        )
        optimizer_param = base.create_optimizer_params(optimizer_args)
        model_train, dense_optimizer = base.make_optimizer_and_shard(
            model,
            config=hstu_config,
            sparse_optimizer_param=optimizer_param,
            dense_optimizer_param=optimizer_param,
            dynamicemb_options_dict=dynamic_options_dict,
            pipeline_type=trainer_args.pipeline_type,
        )

        if is_retrieval:
            stateful_metric_module = base.RetrievalTaskMetricWithSampling(
                metric_types=task_config.eval_metrics,
                MAX_K=args.max_retrieval_items,
            )
            _, eval_dataloader = base.get_data_loader("retrieval", dataset_args, trainer_args, 0)
        else:
            stateful_metric_module = base.get_multi_event_metric_module(
                num_classes=task_config.prediction_head_arch[-1],
                num_tasks=task_config.num_tasks,
                metric_types=task_config.eval_metrics,
                comm_pg=base.parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
            _, eval_dataloader = base.get_data_loader(
                "ranking", dataset_args, trainer_args, task_config.num_tasks
            )

        base.maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)
        reuse_policy = base.TokenKVReusePolicy.from_args(
            default_window_size=args.auc_reuse_window_size,
            default_top_k=args.auc_reuse_top_k,
            policy_json=args.auc_reuse_policy_json,
        )
        base.run_kv_replace_analysis(
            model_train=model_train,
            model=model,
            eval_dataloader=eval_dataloader,
            stateful_metric_module=stateful_metric_module,
            trainer_args=trainer_args,
            output_dir=args.output_dir,
            reuse_policy=reuse_policy,
            deprecated_similarity_threshold=0.0,
            reuse_token_types=args.auc_reuse_token_types,
            reuse_strategies=args.auc_reuse_strategies,
            reuse_max_distances=args.auc_reuse_max_distances,
        )
        _write_filtered_auc_summary_and_markdown(args.output_dir, args.auc_filter_baseline_threshold)
    finally:
        base.init.destroy_global_state()


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _write_filtered_auc_summary_and_markdown(output_dir: str, threshold: float) -> None:
    by_task_path = os.path.join(output_dir, "reuse_auc_impact_by_task.csv")
    stats_path = os.path.join(output_dir, "kv_replace_overall_stats_all_modes.csv")
    if not os.path.exists(by_task_path) or not os.path.exists(stats_path):
        return

    by_task = pd.read_csv(by_task_path)
    stats = pd.read_csv(stats_path)
    base_auc = by_task[["metric", "baseline"]].drop_duplicates()
    eligible_metrics = base_auc[base_auc["baseline"] > threshold]["metric"].tolist()
    if not eligible_metrics:
        filtered = by_task
    else:
        filtered = by_task[by_task["metric"].isin(eligible_metrics)]

    group_cols = ["reuse_mode", "reuse_strategy", "reuse_token_type"]
    if "reuse_max_distance" in by_task.columns and "reuse_max_distance" in stats.columns:
        group_cols.append("reuse_max_distance")
    summary = (
        filtered.groupby(group_cols, dropna=False)
        .agg(
            filtered_task_count=("metric", "count"),
            mean_baseline_auc=("baseline", "mean"),
            mean_reuse_auc=("kv_replaced", "mean"),
            mean_auc_diff=("diff", "mean"),
            min_auc_diff=("diff", "min"),
            max_auc_drop=("diff", lambda x: float(max(0.0, -x.min()))),
        )
        .reset_index()
    )
    summary = summary.merge(
        stats[
            [
                "reuse_mode",
                "reuse_strategy",
                "reuse_token_type",
                *([] if "reuse_max_distance" not in stats.columns else ["reuse_max_distance"]),
                "replacement_count",
                "selected_ratio_all_tokens",
                "selected_ratio_all_reuse_tokens",
            ]
        ],
        on=group_cols,
        how="left",
    ).rename(
        columns={
            "selected_ratio_all_tokens": "reuse_ratio_all_tokens",
            "selected_ratio_all_reuse_tokens": "reuse_ratio_target_tokens",
        }
    )
    summary["pareto_optimal"] = _mark_pareto(summary)
    filtered_path = os.path.join(output_dir, "reuse_auc_impact_summary_auc_gt_0p6.csv")
    summary.to_csv(filtered_path, index=False)
    _write_motivation_insights_markdown(output_dir, summary, eligible_metrics, threshold)


def _mark_pareto(summary: pd.DataFrame) -> List[bool]:
    marks = []
    for _, row in summary.iterrows():
        dominated = False
        for _, other in summary.iterrows():
            if other["reuse_mode"] == row["reuse_mode"]:
                continue
            same_or_better = (
                other["reuse_ratio_all_tokens"] >= row["reuse_ratio_all_tokens"]
                and other["mean_reuse_auc"] >= row["mean_reuse_auc"]
            )
            strictly_better = (
                other["reuse_ratio_all_tokens"] > row["reuse_ratio_all_tokens"]
                or other["mean_reuse_auc"] > row["mean_reuse_auc"]
            )
            if same_or_better and strictly_better:
                dominated = True
                break
        marks.append(not dominated)
    return marks


def _load_action_same_diff_rollup(output_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(output_dir, "action_kv_same_vs_diff_pairs.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return (
        df.groupby("same_action")
        .agg(
            k_cosine_mean=("k_cosine", "mean"),
            k_cosine_median=("k_cosine", "median"),
            v_cosine_mean=("v_cosine", "mean"),
            v_cosine_median=("v_cosine", "median"),
            pair_count=("k_cosine", "count"),
        )
        .reset_index()
    )


def _load_action_distance_rollup(output_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(output_dir, "action_kv_same_vs_diff_pairs.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "k_centered_cosine" not in df.columns:
        df["k_centered_cosine"] = np.nan
    if "v_centered_cosine" not in df.columns:
        df["v_centered_cosine"] = np.nan
    return (
        df[df["layer_idx"] > 0]
        .groupby(["same_action", "distance_bucket"], sort=False)
        .agg(
            k_cosine_mean=("k_cosine", "mean"),
            k_centered_cosine_mean=("k_centered_cosine", "mean"),
            v_cosine_mean=("v_cosine", "mean"),
            v_centered_cosine_mean=("v_centered_cosine", "mean"),
            pos_distance_mean=("pos_distance", "mean"),
            pair_count=("k_cosine", "count"),
        )
        .reset_index()
    )


def _load_item_action_distance_rollup(output_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(output_dir, "item_action_kv_similarity_pairs.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "k_centered_cosine" not in df.columns:
        df["k_centered_cosine"] = np.nan
    return (
        df[df["layer_idx"] > 0]
        .groupby(["token_type", "distance_bucket"], sort=False)
        .agg(
            k_cosine_mean=("k_cosine", "mean"),
            k_centered_cosine_mean=("k_centered_cosine", "mean"),
            v_cosine_mean=("v_cosine", "mean"),
            pair_count=("k_cosine", "count"),
        )
        .reset_index()
    )


def _load_data_summary(output_dir: str) -> Dict:
    path = os.path.join(output_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_report_commands(output_dir: str) -> Dict[str, str]:
    path = os.path.join(output_dir, "report_commands.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    commands: Dict[str, str] = {}
    for row in rows:
        stage = row.get("stage")
        command = row.get("command")
        if stage and command:
            commands[stage] = command
    return commands


def _append_command_block(lines: List[str], commands: Dict[str, str], stage: str) -> None:
    command = commands.get(stage)
    if not command:
        return
    lines.extend(
        [
            "Command:",
            "",
            "```bash",
            command,
            "```",
            "",
        ]
    )


def _write_motivation_insights_markdown(
    output_dir: str,
    auc_summary: pd.DataFrame,
    eligible_metrics: List[str],
    threshold: float,
) -> None:
    data_summary = _load_data_summary(output_dir)
    report_commands = _load_report_commands(output_dir)
    action_rollup = _load_action_same_diff_rollup(output_dir)
    action_distance = _load_action_distance_rollup(output_dir)
    item_action_distance = _load_item_action_distance_rollup(output_dir)

    lines = [
        "# Action KV Reuse Motivation Insights",
        "",
        "## Takeaway",
        "",
        "GR KV reuse should be constrained to token types and positions where the hidden semantics stay close. "
        "The evidence below says action tokens are the right target: their id space is small, repeated actions have high real-KV similarity, and AUC degrades when reuse ignores semantic id or distance.",
        "",
    ]

    token_space = data_summary.get("token_space", []) if data_summary else []
    if token_space:
        lines.extend(["## Token Space", ""])
        _append_command_block(lines, report_commands, "data_analysis")
        lines.extend(
            [
                "| Token Type | Global Unique IDs | Mean User Unique IDs | Mean User Top-1 Share | Mean User Top-3 Share |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in token_space:
            lines.append(
                f"| {row['token_type']} | {int(row['global_unique_ids']):,} | "
                f"{row['mean_user_unique_ids']:.2f} | "
                f"{_format_pct(row.get('mean_user_top1_share', float('nan')))} | "
                f"{_format_pct(row.get('mean_user_top3_share', float('nan')))} |"
            )
        lines.extend(
            [
                "",
                "Insight: action has a much smaller semantic space than item, so repeated action ids are common enough to create a real reuse opportunity.",
                "",
            ]
        )

    if action_rollup is not None and set(action_rollup["same_action"]) == {False, True}:
        same = action_rollup[action_rollup["same_action"] == True].iloc[0]
        diff = action_rollup[action_rollup["same_action"] == False].iloc[0]
        lines.extend(["## Real KV Similarity", ""])
        _append_command_block(lines, report_commands, "kv_analysis")
        lines.extend(
            [
                "| Action Pair | K Cosine Mean | V Cosine Mean | Pair Count |",
                "|---|---:|---:|---:|",
                f"| Same action | {same['k_cosine_mean']:.4f} | {same['v_cosine_mean']:.4f} | {int(same['pair_count']):,} |",
                f"| Different action | {diff['k_cosine_mean']:.4f} | {diff['v_cosine_mean']:.4f} | {int(diff['pair_count']):,} |",
                f"| Same - Different | {same['k_cosine_mean'] - diff['k_cosine_mean']:+.4f} | {same['v_cosine_mean'] - diff['v_cosine_mean']:+.4f} |  |",
                "",
                "Insight: KV vectors are much more consistent when the action id is the same. This is the direct evidence that action identity carries reusable KV structure.",
                "",
            ]
        )

    if action_distance is not None and not action_distance.empty:
        lines.extend(["## KV Similarity vs Distance", ""])
        _append_command_block(lines, report_commands, "kv_analysis")
        lines.extend(
            [
                "Layer 0 is excluded in this distance view so the trend reflects contextual HSTU layers rather than raw embedding identity.",
                "",
                "| Pair Type | Distance Bucket | Mean Distance | K Cosine | K Centered Cosine | V Cosine | Pair Count |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        display = action_distance.copy()
        display["pair_type"] = np.where(display["same_action"], "same action", "different action")
        for _, row in display.iterrows():
            lines.append(
                f"| {row['pair_type']} | {row['distance_bucket']} | {row['pos_distance_mean']:.1f} | "
                f"{row['k_cosine_mean']:.4f} | {row['k_centered_cosine_mean']:.4f} | "
                f"{row['v_cosine_mean']:.4f} | {int(row['pair_count']):,} |"
            )
        if item_action_distance is not None and not item_action_distance.empty:
            lines.extend(
                [
                    "",
                    "| Same-ID Token Type | Distance Bucket | K Cosine | K Centered Cosine | V Cosine | Pair Count |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for _, row in item_action_distance.iterrows():
                lines.append(
                    f"| {row['token_type']} | {row['distance_bucket']} | "
                    f"{row['k_cosine_mean']:.4f} | {row['k_centered_cosine_mean']:.4f} | "
                    f"{row['v_cosine_mean']:.4f} | {int(row['pair_count']):,} |"
                )
        lines.extend(
            [
                "",
                "Insight: same-action KV remains far closer than different-action KV, but distance still matters; this motivates a local window instead of one unbounded action cache.",
                "",
            ]
        )

    if data_summary:
        action_conc = data_summary.get("action_concentration", {})
        default_policy = data_summary.get("default_512_top3_policy", {})
        lines.extend(["## Reuse Opportunity", ""])
        _append_command_block(lines, report_commands, "data_analysis")
        lines.extend(
            [
                "| Signal | Value |",
                "|---|---:|",
                f"| Mean user top-1 action share | {_format_pct(action_conc.get('mean_top1_share', float('nan')))} |",
                f"| Mean user top-3 action share | {_format_pct(action_conc.get('mean_top3_share', float('nan')))} |",
                f"| Window=512, topK=3 action coverage | {_format_pct(default_policy.get('coverage', float('nan')))} |",
                f"| Window=512, topK=3 candidate reuse rate | {_format_pct(default_policy.get('candidate_rate', float('nan')))} |",
                "",
                "Insight: action tokens are concentrated enough that a small local topK policy can cover most reuse candidates.",
                "",
            ]
        )

    if not auc_summary.empty:
        display = auc_summary.sort_values(["reuse_ratio_all_tokens", "mean_reuse_auc"])
        lines.extend(["## AUC vs Reuse", ""])
        _append_command_block(lines, report_commands, "auc_impact_analysis")
        lines.extend(
            [
                f"AUC is averaged only over tasks with baseline AUC > {threshold:.1f}: "
                + (", ".join(f"`{x}`" for x in eligible_metrics) if eligible_metrics else "all available AUC tasks")
                + ".",
                "",
                "| Mode | Max Distance | Reuse Ratio | Mean AUC | AUC Diff | Max AUC Drop | Replacements | Pareto |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in display.iterrows():
            max_distance = row.get("reuse_max_distance", np.nan)
            if pd.isna(max_distance):
                distance_text = "-"
            elif int(max_distance) < 0:
                distance_text = "global"
            else:
                distance_text = str(int(max_distance))
            lines.append(
                f"| {row['reuse_mode']} | {distance_text} | {_format_pct(row['reuse_ratio_all_tokens'])} | "
                f"{row['mean_reuse_auc']:.6f} | {row['mean_auc_diff']:+.6f} | "
                f"{row['max_auc_drop']:.6f} | {int(row['replacement_count']):,} | "
                f"{'yes' if row['pareto_optimal'] else 'no'} |"
            )
        lines.extend(
            [
                "",
                "Insight: the legacy first-action and wrong-action rows are negative controls: high replacement without semantic or distance constraints can damage ranking quality. The distance sweep shows how much locality is needed before reuse becomes low-risk.",
                "",
                "## Conclusion",
                "",
                "Action KV reuse is motivated by three facts: action ids have a much smaller reuse space than item ids, same-action KV stays substantially closer than the controls, and AUC risk grows when reuse ignores action identity or position distance. The method should be presented as constrained action reuse, not generic KV sharing.",
            ]
        )

    with open(os.path.join(output_dir, "MOTIVATION_INSIGHTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
