# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional real-KV motivation analysis for action reuse.

This module is imported only when `analyze_action_reuse_motivation.py` is run
with `--run-kv-analysis`, keeping the default data-side path lightweight.
"""

import json
import os
import re
import shutil
from itertools import combinations, islice
from typing import Dict, List, Optional, Tuple

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
        split_sizes = [
            module._linear_dim_per_head * module._num_heads,
            module._linear_dim_per_head * module._num_heads,
            module._attention_dim_per_head * module._num_heads,
            module._attention_dim_per_head * module._num_heads,
        ]
        _, value, _, key = torch.split(silu_uvqk, split_sizes, dim=-1)
        value = value.view(-1, module._num_heads, module._linear_dim_per_head)
        key = key.view(-1, module._num_heads, module._attention_dim_per_head)
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


DEFAULT_DISTANCE_BUCKETS = (64, 128, 256, 512, 1024)


def _normalize_distance_buckets(raw_buckets) -> List[int]:
    buckets = sorted({int(boundary) for boundary in raw_buckets if int(boundary) > 0})
    return buckets or list(DEFAULT_DISTANCE_BUCKETS)


def _distance_bucket(distance: int, distance_buckets) -> str:
    buckets = _normalize_distance_buckets(distance_buckets)
    prev = 0
    for boundary in buckets:
        if distance <= boundary:
            return f"{prev + 1}-{boundary}" if prev else f"<= {boundary}"
        prev = boundary
    return f"> {buckets[-1]}"


def _distance_bucket_sort_key(bucket: str) -> int:
    text = str(bucket).strip()
    if text.startswith("<="):
        return int(text.split("<=", 1)[1].strip())
    if text.startswith(">"):
        return int(text.split(">", 1)[1].strip()) + 1
    if "-" in text:
        return int(text.split("-", 1)[0].strip())
    return 10**12


def _cap_pairs_by_distance_bucket(
    pairs: List[tuple],
    *,
    distance_buckets,
    max_pairs_per_distance_bucket: int,
    bucket_key_offset: int = 0,
) -> List[tuple]:
    if max_pairs_per_distance_bucket <= 0 or not pairs:
        return pairs

    grouped: Dict[str, List[tuple]] = {}
    for pair in pairs:
        left = pair[bucket_key_offset]
        right = pair[bucket_key_offset + 1]
        distance = right["local_pos"] - left["local_pos"]
        grouped.setdefault(_distance_bucket(distance, distance_buckets), []).append(pair)

    capped = []
    for bucket_pairs in grouped.values():
        if len(bucket_pairs) <= max_pairs_per_distance_bucket:
            capped.extend(bucket_pairs)
            continue
        idx = np.linspace(
            0,
            len(bucket_pairs) - 1,
            num=max_pairs_per_distance_bucket,
            dtype=np.int64,
        )
        capped.extend(bucket_pairs[int(i)] for i in idx)
    return capped


def _cap_action_pair_specs_by_distance_bucket(
    pair_specs: List[tuple],
    *,
    distance_buckets,
    max_pairs_per_distance_bucket: int,
) -> List[tuple]:
    if max_pairs_per_distance_bucket <= 0 or not pair_specs:
        return pair_specs

    grouped: Dict[tuple, List[tuple]] = {}
    for pair_spec in pair_specs:
        same_action, left, right = pair_spec
        distance = right["local_pos"] - left["local_pos"]
        key = (same_action, _distance_bucket(distance, distance_buckets))
        grouped.setdefault(key, []).append(pair_spec)

    capped = []
    for bucket_specs in grouped.values():
        if len(bucket_specs) <= max_pairs_per_distance_bucket:
            capped.extend(bucket_specs)
            continue
        idx = np.linspace(
            0,
            len(bucket_specs) - 1,
            num=max_pairs_per_distance_bucket,
            dtype=np.int64,
        )
        capped.extend(bucket_specs[int(i)] for i in idx)
    return capped


def _as_head_matrix(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    if tensor.ndim == 1:
        return tensor.reshape(1, -1)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def _cksim(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    left = _as_head_matrix(left)
    right = _as_head_matrix(right)
    dot = (left * right).sum(dim=-1)
    denom = left.norm(dim=-1) * right.norm(dim=-1) + eps
    return (dot / denom).mean()


def _centered_cksim(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    left = _as_head_matrix(left)
    right = _as_head_matrix(right)
    left = left - left.mean(dim=-1, keepdim=True)
    right = right - right.mean(dim=-1, keepdim=True)
    dot = (left * right).sum(dim=-1)
    denom = left.norm(dim=-1) * right.norm(dim=-1) + eps
    return (dot / denom).mean()


def _pair_metrics(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    left_heads = _as_head_matrix(left)
    right_heads = _as_head_matrix(right)
    left_flat = left_heads.reshape(1, -1)
    right_flat = right_heads.reshape(1, -1)
    avg_norm = 0.5 * (left_flat.norm(dim=1) + right_flat.norm(dim=1)).clamp_min(1e-12)
    return {
        "cosine": _cksim(left_heads, right_heads).item(),
        "centered_cosine": _centered_cksim(left_heads, right_heads).item(),
        "relative_l2": ((left_flat - right_flat).norm(dim=1) / avg_norm).item(),
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
    distance_buckets,
    max_items_per_user: int,
    max_actions_per_user: int,
    max_pairs_per_token_id: int = 0,
    max_pairs_per_distance_bucket: int = 0,
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
                    max_tokens_per_user=max_items_per_user,
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
                    max_tokens_per_user=max_actions_per_user,
                )
            )

            by_token: Dict[tuple, List[Dict]] = {}
            for ref in token_refs:
                by_token.setdefault((ref["token_type"], ref["token_id"]), []).append(ref)
            for (token_type, token_id), refs in by_token.items():
                pairs = _iter_capped_pairs(refs, max_pairs_per_token_id)
                pairs = _cap_pairs_by_distance_bucket(
                    pairs,
                    distance_buckets=distance_buckets,
                    max_pairs_per_distance_bucket=max_pairs_per_distance_bucket,
                )
                for left, right in pairs:
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
                            "distance_bucket": _distance_bucket(pos_distance, distance_buckets),
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


def _sample_different_token_pairs(
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
        left_identity = (left["token_type"], left["token_id"])
        right_identity = (right["token_type"], right["token_id"])
        if left_identity == right_identity:
            continue
        if left["local_pos"] > right["local_pos"]:
            left, right = right, left
        pairs.append((left, right))
    return pairs


def _records_to_action_same_diff_pair_frame(
    records: List[Dict],
    window_size: int,
    distance_buckets,
    max_actions_per_user: int,
    max_same_pairs_per_action_id: int,
    max_different_pairs_per_user: int,
    max_pairs_per_distance_bucket: int,
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
            pair_specs = _cap_action_pair_specs_by_distance_bucket(
                pair_specs,
                distance_buckets=distance_buckets,
                max_pairs_per_distance_bucket=max_pairs_per_distance_bucket,
            )

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
                        "distance_bucket": _distance_bucket(pos_distance, distance_buckets),
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


def _records_to_random_different_token_pair_frame(
    records: List[Dict],
    window_size: int,
    distance_buckets,
    max_items_per_user: int,
    max_actions_per_user: int,
    max_pairs_per_user: int,
    max_pairs_per_distance_bucket: int,
    random_seed: int = 2025,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(random_seed + 17)
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
                    max_tokens_per_user=max_items_per_user,
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
                    max_tokens_per_user=max_actions_per_user,
                )
            )
            pairs = _sample_different_token_pairs(token_refs, max_pairs_per_user, rng)
            pairs = _cap_pairs_by_distance_bucket(
                pairs,
                distance_buckets=distance_buckets,
                max_pairs_per_distance_bucket=max_pairs_per_distance_bucket,
            )
            for left, right in pairs:
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
                        "token_type_i": left["token_type"],
                        "token_type_j": right["token_type"],
                        "token_id_i": left["token_id"],
                        "token_id_j": right["token_id"],
                        "pos_i": left["local_pos"],
                        "pos_j": right["local_pos"],
                        "pos_distance": pos_distance,
                        "distance_bucket": _distance_bucket(pos_distance, distance_buckets),
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


def _build_action_same_diff_distance_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()

    parts = []
    for layer_group, group in (
        ("all_layers", pair_df),
        ("contextual_layers", pair_df[pair_df["layer_idx"] > 0]),
    ):
        if group.empty:
            continue
        summary = (
            group.groupby(["same_action", "distance_bucket"], sort=False)
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
        summary.insert(0, "layer_group", layer_group)
        parts.append(summary)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["_distance_sort"] = out["distance_bucket"].map(_distance_bucket_sort_key)
    out = out.sort_values(
        ["layer_group", "same_action", "_distance_sort"],
        ascending=[True, False, True],
    ).drop(
        columns=["_distance_sort"]
    )
    return out


def _build_identity_baselines_by_layer_summary(
    action_same_diff_df: pd.DataFrame,
    item_action_pair_df: pd.DataFrame,
    random_token_df: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    if not action_same_diff_df.empty:
        action_summary = (
            action_same_diff_df.groupby(["same_action", "layer_idx"])["k_cosine"]
            .agg(k_cosine_mean="mean", pair_count="count")
            .reset_index()
        )
        action_summary["baseline"] = action_summary["same_action"].map(
            {True: "same_action=True", False: "same_action=False"}
        )
        parts.append(action_summary[["baseline", "layer_idx", "k_cosine_mean", "pair_count"]])

    if not item_action_pair_df.empty:
        item_pairs = item_action_pair_df[item_action_pair_df["token_type"] == "item"]
        if not item_pairs.empty:
            item_summary = (
                item_pairs.groupby("layer_idx")["k_cosine"]
                .agg(k_cosine_mean="mean", pair_count="count")
                .reset_index()
            )
            item_summary["baseline"] = "same_item=True"
            parts.append(item_summary[["baseline", "layer_idx", "k_cosine_mean", "pair_count"]])

    if not random_token_df.empty:
        random_summary = (
            random_token_df.groupby("layer_idx")["k_cosine"]
            .agg(k_cosine_mean="mean", pair_count="count")
            .reset_index()
        )
        random_summary["baseline"] = "random_different_token"
        parts.append(random_summary[["baseline", "layer_idx", "k_cosine_mean", "pair_count"]])

    order = {
        "same_action=True": 0,
        "same_action=False": 1,
        "same_item=True": 2,
        "random_different_token": 3,
    }
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    layers = sorted(out["layer_idx"].dropna().unique())
    missing_parts = []
    existing = set(out["baseline"].unique())
    for baseline in order:
        if baseline in existing:
            continue
        missing_parts.append(
            pd.DataFrame(
                {
                    "baseline": baseline,
                    "layer_idx": layers,
                    "k_cosine_mean": np.nan,
                    "pair_count": 0,
                }
            )
        )
    if missing_parts:
        out = pd.concat([out, *missing_parts], ignore_index=True)
    out["_baseline_sort"] = out["baseline"].map(order).fillna(len(order))
    return out.sort_values(["_baseline_sort", "layer_idx"]).drop(columns=["_baseline_sort"])


def _plot_identity_baselines_by_layer(summary_df: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for baseline, group in summary_df.groupby("baseline", sort=False):
        ax.plot(
            group["layer_idx"],
            group["k_cosine_mean"],
            marker="o",
            label=baseline,
        )
    ax.set_title("K CKSim identity baselines by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean K CKSim")
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
    ax.set_title("Same-ID K CKSim distribution")
    ax.set_xlabel("Token type and whether both positions are in the same window")
    ax.set_ylabel("K CKSim")
    # ax.text(
    #     0.01,
    #     -0.22,
    #     "Each box summarizes same-id token pairs: center line=median, box=IQR, whiskers=range without outliers.",
    #     transform=ax.transAxes,
    #     fontsize=8,
    #     va="top",
    # )
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
        ax.set_title("Layer-wise K CKSim: action vs item")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean K CKSim")
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
        distance_summary["_distance_sort"] = distance_summary["distance_bucket"].map(
            _distance_bucket_sort_key
        )
        distance_summary = distance_summary[distance_summary["token_type"] == "action"]
        distance_summary = distance_summary.sort_values(["_distance_sort"])
        for token_type, group in distance_summary.groupby("token_type"):
            ax.plot(
                group["distance_bucket"],
                group["k_cosine"],
                marker="o",
                label=token_type,
            )
        ax.set_title("Action K CKSim by position distance bucket")
        ax.set_xlabel("Distance bucket")
        ax.set_ylabel("Mean K CKSim")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "item_vs_action_kv_similarity_by_distance.png"), dpi=160)
    plt.close(fig)


def _write_pair_frame_for_debug(
    df: pd.DataFrame,
    *,
    output_dir: str,
    raw_name: str,
    sample_name: str,
    write_raw: bool,
    max_sample_rows: int,
) -> None:
    if write_raw:
        df.to_csv(os.path.join(output_dir, raw_name), index=False)
        return
    if max_sample_rows <= 0 or df.empty:
        return
    df.head(max_sample_rows).to_csv(os.path.join(output_dir, sample_name), index=False)


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
        distance_buckets=args.kv_distance_buckets,
        max_items_per_user=args.kv_max_items_per_user,
        max_actions_per_user=args.kv_max_actions_per_user,
        max_pairs_per_token_id=args.kv_max_pairs_per_token_id,
        max_pairs_per_distance_bucket=args.kv_max_pairs_per_distance_bucket,
    )
    action_same_diff_df = _records_to_action_same_diff_pair_frame(
        capture.records,
        window_size=args.kv_window_size,
        distance_buckets=args.kv_distance_buckets,
        max_actions_per_user=args.kv_max_actions_per_user,
        max_same_pairs_per_action_id=args.kv_max_pairs_per_token_id,
        max_different_pairs_per_user=args.kv_max_different_action_pairs_per_user,
        max_pairs_per_distance_bucket=args.kv_max_pairs_per_distance_bucket,
    )
    random_token_df = _records_to_random_different_token_pair_frame(
        capture.records,
        window_size=args.kv_window_size,
        distance_buckets=args.kv_distance_buckets,
        max_items_per_user=args.kv_max_items_per_user,
        max_actions_per_user=args.kv_max_actions_per_user,
        max_pairs_per_user=args.kv_max_different_action_pairs_per_user,
        max_pairs_per_distance_bucket=args.kv_max_pairs_per_distance_bucket,
    )
    _write_pair_frame_for_debug(
        action_same_diff_df,
        output_dir=args.output_dir,
        raw_name="action_kv_same_vs_diff_pairs.csv",
        sample_name="action_kv_same_vs_diff_pairs_sample.csv",
        write_raw=args.write_raw_kv_pairs,
        max_sample_rows=args.max_raw_kv_pair_rows,
    )
    action_same_diff_summary_df = _build_action_same_diff_summary(action_same_diff_df)
    action_same_diff_summary_df.to_csv(
        os.path.join(args.output_dir, "action_kv_same_vs_diff_summary.csv"),
        index=False,
    )
    action_same_diff_distance_summary_df = _build_action_same_diff_distance_summary(
        action_same_diff_df
    )
    action_same_diff_distance_summary_df.to_csv(
        os.path.join(args.output_dir, "action_kv_same_vs_diff_distance_summary.csv"),
        index=False,
    )
    identity_baseline_summary_df = _build_identity_baselines_by_layer_summary(
        action_same_diff_df,
        pair_df,
        random_token_df,
    )
    identity_baseline_summary_df.to_csv(
        os.path.join(args.output_dir, "kv_identity_baselines_by_layer_summary.csv"),
        index=False,
    )
    _plot_identity_baselines_by_layer(identity_baseline_summary_df, args.output_dir)

    _write_pair_frame_for_debug(
        pair_df,
        output_dir=args.output_dir,
        raw_name="item_action_kv_similarity_pairs.csv",
        sample_name="item_action_kv_similarity_pairs_sample.csv",
        write_raw=args.write_raw_kv_pairs,
        max_sample_rows=args.max_raw_kv_pair_rows,
    )
    if args.write_raw_kv_pairs:
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
            replacement_impl=args.auc_kv_replace_implementation,
            report_command=args.report_command,
        )
        _write_filtered_auc_summary_and_markdown(args.output_dir, args.auc_filter_baseline_threshold)
    finally:
        base.init.destroy_global_state()


def _policy_json_for_grid(
    *,
    mode: str,
    window_size: int,
    top_k: int,
    layer_idx: Optional[int] = None,
    bucket: Optional[Dict] = None,
) -> str:
    if mode == "global":
        policy = {"default": {"window_size": window_size, "top_k": top_k}}
    elif mode == "layer":
        if layer_idx is None:
            raise ValueError("layer policy grid requires layer_idx")
        policy = {
            "default": {"window_size": window_size, "top_k": 0},
            "layers": {str(layer_idx): {"window_size": window_size, "top_k": top_k}},
        }
    elif mode == "user_bucket":
        if bucket is None:
            raise ValueError("user bucket policy grid requires bucket")
        policy = {
            "default": {"window_size": window_size, "top_k": 0},
            "user_length_buckets": [
                {
                    "min_seq_len": bucket["min_seq_len"],
                    **(
                        {}
                        if bucket["max_seq_len"] is None
                        else {"max_seq_len": bucket["max_seq_len"]}
                    ),
                    "window_size": window_size,
                    "top_k": top_k,
                }
            ],
        }
    else:
        raise ValueError(f"unknown policy grid mode: {mode}")
    return json.dumps(policy, sort_keys=True)


def _user_length_buckets_from_args(args) -> List[Dict]:
    starts = sorted({int(x) for x in args.long_user_buckets if int(x) >= 0})
    if not starts or starts[0] != 0:
        starts = [0, *starts]
    buckets = []
    for idx, start in enumerate(starts):
        next_start = starts[idx + 1] if idx + 1 < len(starts) else None
        buckets.append(
            {
                "bucket": f"{start}+" if next_start is None else f"{start}-{next_start - 1}",
                "min_seq_len": start,
                "max_seq_len": None if next_start is None else next_start - 1,
            }
        )
    return buckets


def _annotate_policy_summary(
    summary: pd.DataFrame,
    *,
    mode: str,
    window_size: int,
    top_k: int,
    run_dir: str,
    layer_idx: Optional[int] = None,
    bucket: Optional[Dict] = None,
) -> pd.DataFrame:
    out = summary.copy()
    out.insert(0, "policy_scope", mode)
    out.insert(1, "window_size", window_size)
    out.insert(2, "top_k", top_k)
    out.insert(3, "layer_idx", layer_idx if layer_idx is not None else np.nan)
    out.insert(4, "user_bucket", bucket["bucket"] if bucket is not None else "")
    out.insert(5, "run_dir", run_dir)
    return out


def _annotate_policy_by_task(
    by_task: pd.DataFrame,
    *,
    mode: str,
    window_size: int,
    top_k: int,
    run_dir: str,
    layer_idx: Optional[int] = None,
    bucket: Optional[Dict] = None,
) -> pd.DataFrame:
    out = by_task.copy()
    out.insert(0, "policy_scope", mode)
    out.insert(1, "window_size", window_size)
    out.insert(2, "top_k", top_k)
    out.insert(3, "layer_idx", layer_idx if layer_idx is not None else np.nan)
    out.insert(4, "user_bucket", bucket["bucket"] if bucket is not None else "")
    out.insert(5, "run_dir", run_dir)
    return out


def _mark_policy_pareto(summary: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    marks = pd.Series(False, index=summary.index)
    for _, idxs in summary.groupby(group_cols, dropna=False).groups.items():
        group = summary.loc[list(idxs)]
        group_marks = []
        for idx, row in group.iterrows():
            dominated = False
            for other_idx, other in group.iterrows():
                if other_idx == idx:
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
            group_marks.append((idx, not dominated))
        for idx, mark in group_marks:
            marks.loc[idx] = mark
    return marks


def _write_policy_grid_outputs(
    output_dir: str,
    mode: str,
    rows: List[pd.DataFrame],
    by_task_rows: Optional[List[pd.DataFrame]] = None,
) -> None:
    if not rows:
        return
    summary = pd.concat(rows, ignore_index=True)
    summary["policy_pareto"] = _mark_policy_pareto(
        summary,
        {
            "global": ["policy_scope"],
            "layer": ["policy_scope", "layer_idx"],
            "user_bucket": ["policy_scope", "user_bucket"],
        }[mode],
    )
    prefix = {
        "global": "policy_grid",
        "layer": "layer_policy",
        "user_bucket": "user_bucket_policy",
    }[mode]
    summary_path = os.path.join(output_dir, f"{prefix}_auc_summary.csv")
    pareto_path = os.path.join(output_dir, f"{prefix}_pareto.csv")
    summary.to_csv(summary_path, index=False)
    summary[summary["policy_pareto"]].to_csv(pareto_path, index=False)
    if by_task_rows:
        by_task = pd.concat(by_task_rows, ignore_index=True)
        by_task.to_csv(os.path.join(output_dir, f"{prefix}_auc_by_task.csv"), index=False)
    if prefix != "layer_policy":
        _plot_policy_grid_auc(summary, output_dir, prefix)


def _read_and_annotate_policy_outputs(
    run_dir: str,
    *,
    mode: str,
    window_size: int,
    top_k: int,
    layer_idx: Optional[int] = None,
    bucket: Optional[Dict] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    summary_path = os.path.join(run_dir, "reuse_auc_impact_summary_auc_gt_0p6.csv")
    by_task_path = os.path.join(run_dir, "reuse_auc_impact_by_task.csv")
    annotated_summary = None
    annotated_by_task = None
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        if not summary.empty:
            annotated_summary = _annotate_policy_summary(
                summary,
                mode=mode,
                window_size=window_size,
                top_k=top_k,
                run_dir=run_dir,
                layer_idx=layer_idx,
                bucket=bucket,
            )
    if os.path.exists(by_task_path):
        by_task = pd.read_csv(by_task_path)
        if not by_task.empty:
            annotated_by_task = _annotate_policy_by_task(
                by_task,
                mode=mode,
                window_size=window_size,
                top_k=top_k,
                run_dir=run_dir,
                layer_idx=layer_idx,
                bucket=bucket,
            )
    return annotated_summary, annotated_by_task


def _load_baseline_metrics_from_by_task(path: str) -> Optional[Dict[str, float]]:
    if not os.path.exists(path):
        return None
    by_task = pd.read_csv(path)
    if by_task.empty or "metric" not in by_task.columns or "baseline" not in by_task.columns:
        return None
    baseline = by_task[["metric", "baseline"]].drop_duplicates("metric")
    metrics = {
        str(row["metric"]): float(row["baseline"])
        for _, row in baseline.iterrows()
        if not pd.isna(row["baseline"])
    }
    return metrics or None


def _plot_policy_grid_auc(summary: pd.DataFrame, output_dir: str, prefix: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for top_k, group in summary.groupby("top_k"):
        ax.scatter(
            group["mean_reuse_auc"],
            group["reuse_ratio_all_tokens"],
            s=45,
            label=f"K={int(top_k)}",
            alpha=0.8,
        )
    pareto = summary[summary["policy_pareto"]]
    if not pareto.empty:
        ax.scatter(
            pareto["mean_reuse_auc"],
            pareto["reuse_ratio_all_tokens"],
            s=95,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Pareto",
        )
    ax.set_xlabel("Mean filtered AUC")
    ax.set_ylabel("Reuse ratio over all sequence tokens")
    ax.set_title(prefix.replace("_", " ").title())
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_auc_reuse_scatter.png"), dpi=160)
    plt.close(fig)


def _policy_point_from_run_name(run_name: str) -> Optional[Tuple[int, int]]:
    match = re.fullmatch(r"w(\d+)_k(\d+)", run_name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _layer_policy_point_from_run_name(run_name: str) -> Optional[Tuple[int, int, int]]:
    match = re.fullmatch(r"layer(\d+)_w(\d+)_k(\d+)", run_name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _copy_runs_source_to_output(source_dir: str, output_dir: str, dirname: str) -> str:
    if not source_dir or not os.path.isdir(source_dir):
        raise ValueError(f"{dirname} source directory does not exist: {source_dir}")
    dest_dir = os.path.join(output_dir, dirname)
    source_abs = os.path.abspath(source_dir)
    dest_abs = os.path.abspath(dest_dir)
    if source_abs == dest_abs:
        return dest_dir
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    return dest_dir


def aggregate_policy_grid_auc_from_runs(args, mode: str) -> None:
    rows: List[pd.DataFrame] = []
    by_task_rows: List[pd.DataFrame] = []
    if mode == "global":
        run_root = _copy_runs_source_to_output(
            args.policy_grid_runs_source_dir,
            args.output_dir,
            "policy_grid_runs",
        )
        parse_run_name = _policy_point_from_run_name
    elif mode == "layer":
        run_root = _copy_runs_source_to_output(
            args.layer_policy_runs_source_dir,
            args.output_dir,
            "layer_policy_runs",
        )
        parse_run_name = _layer_policy_point_from_run_name
    else:
        raise ValueError("existing-run aggregation only supports global and layer policy grids.")

    for run_name in sorted(os.listdir(run_root)):
        point = parse_run_name(run_name)
        if point is None:
            continue
        if mode == "global":
            window_size, top_k = point
            layer_idx = None
        else:
            layer_idx, window_size, top_k = point
        run_dir = os.path.join(run_root, run_name)
        if not os.path.isdir(run_dir):
            continue
        annotated_summary, annotated_by_task = _read_and_annotate_policy_outputs(
            run_dir,
            mode=mode,
            window_size=window_size,
            top_k=top_k,
            layer_idx=layer_idx,
        )
        if annotated_summary is None:
            continue
        rows.append(annotated_summary)
        if annotated_by_task is not None:
            by_task_rows.append(annotated_by_task)

    if not rows:
        raise ValueError(f"No reusable {mode} policy outputs found in {run_root}")
    _write_policy_grid_outputs(args.output_dir, mode, rows, by_task_rows)
    _write_motivation_insights_markdown(
        args.output_dir,
        auc_summary=pd.DataFrame(),
        eligible_metrics=[],
        threshold=args.auc_filter_baseline_threshold,
    )


def run_policy_grid_auc_analysis(args, mode: str) -> None:
    """Run global/layer/user-bucket window-topK AUC grids and aggregate Pareto rows."""
    if mode == "global" and getattr(args, "policy_grid_runs_source_dir", None):
        aggregate_policy_grid_auc_from_runs(args, mode)
        return
    if mode == "layer" and getattr(args, "layer_policy_runs_source_dir", None):
        aggregate_policy_grid_auc_from_runs(args, mode)
        return

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

    rows: List[pd.DataFrame] = []
    by_task_rows: List[pd.DataFrame] = []
    prefix = {
        "global": "policy_grid",
        "layer": "layer_policy",
        "user_bucket": "user_bucket_policy",
    }[mode]
    run_root = os.path.join(args.output_dir, f"{prefix}_runs")
    os.makedirs(run_root, exist_ok=True)
    grid_window_sizes = list(args.layer_window_sizes) if mode == "layer" else list(args.window_sizes)
    grid_top_ks = list(args.layer_top_ks) if mode == "layer" else list(args.top_ks)

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
        layer_indices = range(hstu_config.num_layers) if mode == "layer" else [None]
        buckets = _user_length_buckets_from_args(args) if mode == "user_bucket" else [None]
        baseline_metrics_override = _load_baseline_metrics_from_by_task(
            os.path.join(args.output_dir, "kv_only_no_reuse_auc_by_task.csv")
        )

        if mode == "global" and args.auc_kv_replace_implementation == "kv_only":
            control_dir = os.path.join(run_root, "kv_only_no_reuse")
            os.makedirs(control_dir, exist_ok=True)
            control_summary_path = os.path.join(
                control_dir,
                "reuse_auc_impact_summary_auc_gt_0p6.csv",
            )
            if not os.path.exists(control_summary_path):
                control_policy = base.TokenKVReusePolicy.from_args(
                    default_window_size=args.window_sizes[0],
                    default_top_k=0,
                    policy_json=json.dumps(
                        {"default": {"window_size": args.window_sizes[0], "top_k": 0}}
                    ),
                )
                base.run_kv_replace_analysis(
                    model_train=model_train,
                    model=model,
                    eval_dataloader=eval_dataloader,
                    stateful_metric_module=stateful_metric_module,
                    trainer_args=trainer_args,
                    output_dir=control_dir,
                    reuse_policy=control_policy,
                    deprecated_similarity_threshold=0.0,
                    reuse_token_types=["action"],
                    reuse_strategies=["window_topk_same_id"],
                    reuse_max_distances=None,
                    replacement_impl=args.auc_kv_replace_implementation,
                    report_command=args.report_command,
                )
                _write_filtered_auc_summary_and_markdown(
                    control_dir,
                    args.auc_filter_baseline_threshold,
                )
            control_by_task_path = os.path.join(control_dir, "reuse_auc_impact_by_task.csv")
            if os.path.exists(control_by_task_path):
                pd.read_csv(control_by_task_path).to_csv(
                    os.path.join(args.output_dir, "kv_only_no_reuse_auc_by_task.csv"),
                    index=False,
                )
                baseline_metrics_override = _load_baseline_metrics_from_by_task(
                    os.path.join(args.output_dir, "kv_only_no_reuse_auc_by_task.csv")
                )
            if os.path.exists(control_summary_path):
                pd.read_csv(control_summary_path).to_csv(
                    os.path.join(args.output_dir, "kv_only_no_reuse_auc_summary.csv"),
                    index=False,
                )

        for layer_idx in layer_indices:
            for bucket in buckets:
                for window_size in grid_window_sizes:
                    for top_k in grid_top_ks:
                        policy_json = _policy_json_for_grid(
                            mode=mode,
                            window_size=window_size,
                            top_k=top_k,
                            layer_idx=layer_idx,
                            bucket=bucket,
                        )
                        name_parts = [f"w{window_size}", f"k{top_k}"]
                        if layer_idx is not None:
                            name_parts.insert(0, f"layer{layer_idx}")
                        if bucket is not None:
                            name_parts.insert(0, f"user_{bucket['bucket'].replace('+', 'plus')}")
                        run_name = "_".join(name_parts)
                        run_dir = os.path.join(run_root, run_name)
                        os.makedirs(run_dir, exist_ok=True)
                        annotated_summary, annotated_by_task = _read_and_annotate_policy_outputs(
                            run_dir,
                            mode=mode,
                            window_size=window_size,
                            top_k=top_k,
                            layer_idx=layer_idx,
                            bucket=bucket,
                        )
                        if annotated_summary is not None:
                            rows.append(annotated_summary)
                            if annotated_by_task is not None:
                                by_task_rows.append(annotated_by_task)
                            continue

                        reuse_policy = base.TokenKVReusePolicy.from_args(
                            default_window_size=args.auc_reuse_window_size,
                            default_top_k=args.auc_reuse_top_k,
                            policy_json=policy_json,
                        )
                        base.run_kv_replace_analysis(
                            model_train=model_train,
                            model=model,
                            eval_dataloader=eval_dataloader,
                            stateful_metric_module=stateful_metric_module,
                            trainer_args=trainer_args,
                            output_dir=run_dir,
                            reuse_policy=reuse_policy,
                            deprecated_similarity_threshold=0.0,
                            reuse_token_types=["action"],
                            reuse_strategies=["window_topk_same_id"],
                            reuse_max_distances=None,
                            replacement_impl=args.auc_kv_replace_implementation,
                            report_command=args.report_command,
                            baseline_metrics_override=baseline_metrics_override,
                        )
                        _write_filtered_auc_summary_and_markdown(
                            run_dir,
                            args.auc_filter_baseline_threshold,
                        )
                        annotated_summary, annotated_by_task = _read_and_annotate_policy_outputs(
                            run_dir,
                            mode=mode,
                            window_size=window_size,
                            top_k=top_k,
                            layer_idx=layer_idx,
                            bucket=bucket,
                        )
                        if annotated_summary is None:
                            continue
                        rows.append(annotated_summary)
                        if annotated_by_task is not None:
                            by_task_rows.append(annotated_by_task)
        _write_policy_grid_outputs(args.output_dir, mode, rows, by_task_rows)
        _write_motivation_insights_markdown(
            args.output_dir,
            auc_summary=pd.DataFrame(),
            eligible_metrics=[],
            threshold=args.auc_filter_baseline_threshold,
        )
    finally:
        base.init.destroy_global_state()


def _format_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/a"
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
    summary_path = os.path.join(output_dir, "action_kv_same_vs_diff_distance_summary.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        if not summary.empty:
            if "layer_group" in summary.columns:
                contextual = summary[summary["layer_group"] == "contextual_layers"]
                if not contextual.empty:
                    summary = contextual
            summary = summary.copy()
            summary["_distance_sort"] = summary["distance_bucket"].map(_distance_bucket_sort_key)
            return summary.sort_values(
                ["same_action", "_distance_sort"],
                ascending=[False, True],
            ).drop(columns=["_distance_sort"])

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
    out = (
        df[df["layer_idx"] > 0]
        .groupby(["same_action", "distance_bucket"], sort=False)
        .agg(
            k_cosine_mean=("k_cosine", "mean"),
            k_cosine_median=("k_cosine", "median"),
            k_cosine_p10=("k_cosine", lambda x: x.quantile(0.10)),
            k_cosine_p90=("k_cosine", lambda x: x.quantile(0.90)),
            k_centered_cosine_mean=("k_centered_cosine", "mean"),
            v_cosine_mean=("v_cosine", "mean"),
            v_cosine_median=("v_cosine", "median"),
            v_cosine_p10=("v_cosine", lambda x: x.quantile(0.10)),
            v_cosine_p90=("v_cosine", lambda x: x.quantile(0.90)),
            v_centered_cosine_mean=("v_centered_cosine", "mean"),
            pos_distance_mean=("pos_distance", "mean"),
            pair_count=("k_cosine", "count"),
        )
        .reset_index()
    )
    out["_distance_sort"] = out["distance_bucket"].map(_distance_bucket_sort_key)
    return out.sort_values(["same_action", "_distance_sort"], ascending=[False, True]).drop(
        columns=["_distance_sort"]
    )


def _load_item_action_distance_rollup(output_dir: str) -> Optional[pd.DataFrame]:
    summary_path = os.path.join(output_dir, "item_action_kv_similarity_summary.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        if summary.empty:
            return None
        if "layer_idx" in summary.columns:
            summary = summary[summary["layer_idx"] > 0]
        if summary.empty:
            return None

        def _weighted_rollup(group: pd.DataFrame) -> pd.Series:
            weights = group["pair_count"].astype(float).clip(lower=0)
            if weights.sum() <= 0:
                weights = None
            return pd.Series(
                {
                    "k_cosine_mean": np.average(group["k_cosine_mean"], weights=weights),
                    "k_centered_cosine_mean": np.average(
                        group["k_centered_cosine_mean"],
                        weights=weights,
                    ),
                    "v_cosine_mean": np.average(group["v_cosine_mean"], weights=weights),
                    "pair_count": group["pair_count"].sum(),
                }
            )

        out = (
            summary.groupby(["token_type", "distance_bucket"], sort=False)
            .apply(_weighted_rollup)
            .reset_index()
        )
        out["_distance_sort"] = out["distance_bucket"].map(_distance_bucket_sort_key)
        return out.sort_values(["token_type", "_distance_sort"]).drop(columns=["_distance_sort"])

    path = os.path.join(output_dir, "item_action_kv_similarity_pairs.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "k_centered_cosine" not in df.columns:
        df["k_centered_cosine"] = np.nan
    out = (
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
    out["_distance_sort"] = out["distance_bucket"].map(_distance_bucket_sort_key)
    return out.sort_values(["token_type", "_distance_sort"]).drop(columns=["_distance_sort"])


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


def _append_glossary(lines: List[str], entries: Dict[str, str]) -> None:
    lines.extend(["Variables:", ""])
    for name, description in entries.items():
        lines.append(f"- `{name}`: {description}")
    lines.append("")


def _read_csv_if_exists(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df if not df.empty else pd.DataFrame()


def _append_not_run(lines: List[str], output_name: str) -> None:
    lines.extend(
        [
            f"`{output_name}` was not generated in this run.",
            "",
        ]
    )


def _recommend_policy_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, group in df.groupby(group_cols, dropna=False):
        safe = group[group["max_auc_drop"] <= 0.001]
        source = safe if not safe.empty else group[group["policy_pareto"]]
        if source.empty:
            source = group
        rows.append(
            source.sort_values(
                ["reuse_ratio_all_tokens", "mean_reuse_auc"],
                ascending=[False, False],
            ).iloc[0]
        )
    return pd.DataFrame(rows)


def _append_policy_grid_section(
    *,
    lines: List[str],
    commands: Dict[str, str],
    output_dir: str,
    title: str,
    stage: str,
    summary_file: str,
    pareto_file: str,
    group_cols: List[str],
    question_text: str,
) -> None:
    lines.extend([f"## {title}", "", question_text, ""])
    _append_command_block(lines, commands, stage)
    _append_glossary(
        lines,
        {
            "window_size": "Interleaved-token window size used by action KV reuse.",
            "top_k": "Number of high-frequency action ids reused per user/window/layer.",
            "reuse_ratio_all_tokens": "Replaced action KV rows divided by all sequence tokens; higher means more compute/cache saved.",
            "mean_reuse_auc": "Mean AUC after KV reuse, filtered to tasks whose baseline AUC is above the configured threshold.",
            "mean_auc_diff": "Mean AUC change relative to no-reuse baseline.",
            "max_auc_drop": "Worst AUC drop among filtered tasks; lower is safer.",
            "policy_pareto": "True if no other point has both higher/equal reuse ratio and higher/equal AUC.",
        },
    )
    summary = _read_csv_if_exists(os.path.join(output_dir, summary_file))
    pareto = _read_csv_if_exists(os.path.join(output_dir, pareto_file))
    if summary.empty:
        _append_not_run(lines, summary_file)
        return
    prefix = summary_file.replace("_auc_summary.csv", "")
    by_task_file = f"{prefix}_auc_by_task.csv"
    by_task = _read_csv_if_exists(os.path.join(output_dir, by_task_file))

    if not by_task.empty:
        baseline_tasks = (
            by_task[["metric", "baseline"]]
            .drop_duplicates()
            .sort_values("metric")
        )
        lines.extend(["All-task baseline AUC used by this grid:", ""])
        lines.append("| Metric | Baseline AUC | Included In Pareto |")
        lines.append("|---|---:|---:|")
        for _, row in baseline_tasks.iterrows():
            included = row["baseline"] > 0.6
            lines.append(
                f"| {row['metric']} | {row['baseline']:.6f} | "
                f"{'yes' if included else 'no'} |"
            )
        lines.append("")

    if stage == "policy_grid_auc_analysis":
        control = _read_csv_if_exists(os.path.join(output_dir, "kv_only_no_reuse_auc_by_task.csv"))
        if not control.empty:
            lines.extend(["KV-only no-reuse sanity check:", ""])
            lines.append("| Metric | Original Baseline | KV-only No-Reuse | Diff |")
            lines.append("|---|---:|---:|---:|")
            for _, row in control.sort_values("metric").iterrows():
                lines.append(
                    f"| {row['metric']} | {row['baseline']:.6f} | "
                    f"{row['kv_replaced']:.6f} | {row['diff']:+.6f} |"
                )
            lines.append("")

    recommended = _recommend_policy_rows(summary, group_cols)
    if not recommended.empty:
        display_cols = [
            col
            for col in [
                *group_cols,
                "window_size",
                "top_k",
                "reuse_ratio_all_tokens",
                "mean_reuse_auc",
                "mean_auc_diff",
                "max_auc_drop",
            ]
            if col in recommended.columns
        ]
        lines.extend(["Recommended operating points:", ""])
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")
        for _, row in recommended[display_cols].iterrows():
            values = []
            for col in display_cols:
                value = row[col]
                if col == "reuse_ratio_all_tokens":
                    values.append(_format_pct(value))
                elif col in {"mean_reuse_auc", "mean_auc_diff", "max_auc_drop"}:
                    values.append(f"{value:.6f}")
                elif pd.isna(value):
                    values.append("-")
                else:
                    values.append(str(int(value)) if isinstance(value, (float, np.floating)) and value.is_integer() else str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    if not pareto.empty:
        pareto = pareto.sort_values(["reuse_ratio_all_tokens", "mean_reuse_auc"], ascending=[False, False])
        pareto = pareto.head(12)
        display_cols = [
            col
            for col in [
                *group_cols,
                "window_size",
                "top_k",
                "reuse_ratio_all_tokens",
                "mean_reuse_auc",
                "mean_auc_diff",
                "max_auc_drop",
            ]
            if col in pareto.columns
        ]
        lines.extend(["Pareto frontier preview:", ""])
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("|" + "|".join(["---"] * len(display_cols)) + "|")
        for _, row in pareto[display_cols].iterrows():
            values = []
            for col in display_cols:
                value = row[col]
                if col == "reuse_ratio_all_tokens":
                    values.append(_format_pct(value))
                elif col in {"mean_reuse_auc", "mean_auc_diff", "max_auc_drop"}:
                    values.append(f"{value:.6f}")
                elif pd.isna(value):
                    values.append("-")
                else:
                    values.append(str(int(value)) if isinstance(value, (float, np.floating)) and value.is_integer() else str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    lines.extend(
        [
            f"Full table: `{summary_file}`",
            f"All-task AUC table: `{by_task_file}`",
            f"Pareto table: `{pareto_file}`",
        ]
    )
    if prefix != "layer_policy":
        lines.append(f"Scatter plot: `{prefix}_auc_reuse_scatter.png`")
    lines.append("")


def _write_motivation_insights_markdown(
    output_dir: str,
    auc_summary: pd.DataFrame,
    eligible_metrics: List[str],
    threshold: float,
) -> None:
    data_summary = _load_data_summary(output_dir)
    report_commands = _load_report_commands(output_dir)
    action_distance = _load_action_distance_rollup(output_dir)
    item_action_distance = _load_item_action_distance_rollup(output_dir)

    lines = [
        "# Action KV Reuse Motivation Report",
        "",
        "This report is organized around the five questions needed to motivate and tune action KV reuse.",
        "",
        "Definitions used throughout: AUC summaries only include tasks whose baseline AUC is above the configured threshold; Pareto optimality is computed in the reuse-ratio vs AUC plane.",
        "",
    ]

    token_space = data_summary.get("token_space", []) if data_summary else []
    lines.extend(["## Q1. Why Is Action Better Than Item For Reuse?", ""])
    _append_command_block(lines, report_commands, "data_analysis")
    _append_glossary(
        lines,
        {
            "global_unique_ids": "Number of distinct ids in the whole analyzed dataset.",
            "mean_user_unique_ids": "Average number of distinct ids per user sequence.",
            "mean_user_top1_share": "Average fraction of a user's tokens covered by their most frequent id.",
            "mean_user_top3_share": "Average fraction covered by their three most frequent ids.",
        },
    )
    if token_space:
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
    else:
        _append_not_run(lines, "token_space_summary.csv")

    lines.extend(["## Q2. How Similar Is Action KV At Different Distances?", ""])
    if action_distance is not None and not action_distance.empty:
        _append_command_block(lines, report_commands, "kv_analysis")
        _append_glossary(
            lines,
            {
                "same action": "Both KV rows come from the same action id in the same user sequence.",
                "different action": "The two KV rows come from different action ids in the same user sequence.",
                "distance_bucket": "Interleaved-token distance between the two positions.",
                "K/V CKSim": "Mean head-wise cosine similarity of projected K or V vectors. Each head is compared along its feature dimension, then averaged across heads.",
                "P10/P90": "10th/90th percentile; these expose tail behavior hidden by the mean.",
                "pair_count": "Number of sampled pairs in this bucket.",
            },
        )
        lines.extend(
            [
                "Layer 0 is excluded here so the trend reflects contextual HSTU layers rather than raw embedding identity.",
                "",
                "| Pair Type | Distance Bucket | Mean Distance | K CKSim | K P10 | K P90 | K Centered CKSim | V CKSim | V P10 | V P90 | Pair Count |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        display = action_distance.copy()
        display["pair_type"] = np.where(display["same_action"], "same action", "different action")
        for _, row in display.iterrows():
            lines.append(
                f"| {row['pair_type']} | {row['distance_bucket']} | {row['pos_distance_mean']:.1f} | "
                f"{row['k_cosine_mean']:.4f} | {row.get('k_cosine_p10', np.nan):.4f} | "
                f"{row.get('k_cosine_p90', np.nan):.4f} | {row['k_centered_cosine_mean']:.4f} | "
                f"{row['v_cosine_mean']:.4f} | {row.get('v_cosine_p10', np.nan):.4f} | "
                f"{row.get('v_cosine_p90', np.nan):.4f} | {int(row['pair_count']):,} |"
            )

        pivot = display.pivot_table(
            index="distance_bucket",
            columns="same_action",
            values=["k_cosine_mean", "v_cosine_mean", "pair_count"],
            aggfunc="first",
        )
        if True in pivot.get("k_cosine_mean", {}) and False in pivot.get("k_cosine_mean", {}):
            pivot = pivot.loc[sorted(pivot.index, key=_distance_bucket_sort_key)]
            lines.extend(
                [
                    "",
                    "| Distance Bucket | K Same-Diff Gap | V Same-Diff Gap | Same Pairs | Different Pairs |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for bucket in pivot.index:
                k_same = pivot[("k_cosine_mean", True)].get(bucket, np.nan)
                k_diff = pivot[("k_cosine_mean", False)].get(bucket, np.nan)
                v_same = pivot[("v_cosine_mean", True)].get(bucket, np.nan)
                v_diff = pivot[("v_cosine_mean", False)].get(bucket, np.nan)
                same_count = pivot[("pair_count", True)].get(bucket, np.nan)
                diff_count = pivot[("pair_count", False)].get(bucket, np.nan)
                lines.append(
                    f"| {bucket} | {k_same - k_diff:+.4f} | {v_same - v_diff:+.4f} | "
                    f"{int(same_count) if not pd.isna(same_count) else 0:,} | "
                    f"{int(diff_count) if not pd.isna(diff_count) else 0:,} |"
                )
        lines.extend(
            [
                "",
                "Insight: action identity is reusable mainly as a local signal: same-action KV is much closer than different-action KV in short-distance buckets, while the distance trend explains why an unbounded global action cache is not the right motivation.",
                "",
            ]
        )
        if item_action_distance is not None and not item_action_distance.empty:
            lines.extend(
                [
                    "",
                    "Same-id action/item comparison, using the same distance buckets:",
                    "",
                    "| Same-ID Token Type | Distance Bucket | K CKSim | K Centered CKSim | V CKSim | Pair Count |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for _, row in item_action_distance.iterrows():
                lines.append(
                    f"| {row['token_type']} | {row['distance_bucket']} | "
                    f"{row['k_cosine_mean']:.4f} | {row['k_centered_cosine_mean']:.4f} | "
                    f"{row['v_cosine_mean']:.4f} | {int(row['pair_count']):,} |"
                )
            lines.append("")
        lines.extend(
            [
                "Distribution plots:",
                "- `action_kv_same_vs_diff_by_layer.png` (same action, different action, same item, and random different-token baselines)",
                "- `kv_identity_baselines_by_layer_summary.csv`",
                "- `item_action_kv_similarity_boxplot.png`: each group is same-id pairs split by token type and `same_window`. The center line is the median K CKSim, the box is the interquartile range, whiskers show the non-outlier range, and dots are outlier pairs. Higher boxes mean the same id keeps more similar K vectors across positions.",
                "- `item_vs_action_kv_similarity_by_distance.png`: action-only same-id K CKSim by distance bucket, sorted from short to long distance.",
                "",
            ]
        )
    else:
        _append_not_run(lines, "action_kv_same_vs_diff_distance_summary.csv")

    _append_policy_grid_section(
        lines=lines,
        commands=report_commands,
        output_dir=output_dir,
        title="Q3. Which Global Top-K/Window Points Are Pareto Optimal?",
        stage="policy_grid_auc_analysis",
        summary_file="policy_grid_auc_summary.csv",
        pareto_file="policy_grid_pareto.csv",
        group_cols=["policy_scope"],
        question_text="This grid answers how top-K and window size trade reuse ratio against final inference AUC.",
    )

    _append_policy_grid_section(
        lines=lines,
        commands=report_commands,
        output_dir=output_dir,
        title="Q4. Which Layers Are Most Sensitive?",
        stage="layer_policy_auc_analysis",
        summary_file="layer_policy_auc_summary.csv",
        pareto_file="layer_policy_pareto.csv",
        group_cols=["layer_idx"],
        question_text=(
            "Each run enables reuse in one HSTU layer only. By default this uses "
            "top_k=1 and window_size=64, so the table isolates layer sensitivity "
            "without running a full top-K/window grid for every layer."
        ),
    )

    _append_policy_grid_section(
        lines=lines,
        commands=report_commands,
        output_dir=output_dir,
        title="Q5. How Should Top-K/Window Be Chosen For Different Users?",
        stage="user_bucket_policy_auc_analysis",
        summary_file="user_bucket_policy_auc_summary.csv",
        pareto_file="user_bucket_policy_pareto.csv",
        group_cols=["user_bucket"],
        question_text="Each run enables reuse for one user sequence-length bucket only, which estimates the best policy for short vs long users.",
    )

    if data_summary:
        action_conc = data_summary.get("action_concentration", {})
        default_policy = data_summary.get("default_512_top3_policy", {})
        lines.extend(["## Dataset-Only Reuse Opportunity Reference", ""])
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
        lines.extend(["## Legacy/Strategy AUC Controls", ""])
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
                "These controls are useful for explaining what not to do, but Q3-Q5 are the policy-selection sections.",
            ]
        )

    report_text = "\n".join(lines) + "\n"
    with open(os.path.join(output_dir, "MOTIVATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report_text)
    insights_path = os.path.join(output_dir, "MOTIVATION_INSIGHTS.md")
    if os.path.exists(insights_path):
        os.remove(insights_path)
