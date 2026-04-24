# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump real K/V vectors for repeated actions at multiple positions.

This is a small diagnostic utility for inspecting whether the same action keeps
similar K/V vectors across positions and layers. It runs a capped checkpoint
forward pass, samples repeated actions, and writes a compact torch.save payload.

Example:
    cd examples/hstu
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) \
    torchrun --nproc_per_node 1 --master_addr localhost --master_port 6370 \
        ./training/dump_action_kv_vector_samples.py \
        --gin-config-file ./training/configs/kuairand_1k_ranking.gin \
        --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 \
        --output-dir ./analysis_output/action_kv_vector_samples
"""

import argparse
import json
import os
from collections import defaultdict
from itertools import islice
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

import eval_checkpoint_action_kv_replace as base
from analyze_action_reuse_motivation_kv_impl import (
    KVMotivationCaptureHook,
    _collect_token_refs,
    _extract_feature_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save real K/V vectors for a few repeated actions across positions/layers."
    )
    parser.add_argument("--gin-config-file", type=str, required=True)
    parser.add_argument("--ckpt-load-dir", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_output/action_kv_vector_samples",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="action_kv_vector_samples.pt",
        help="File name under --output-dir, or an absolute .pt path.",
    )
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--max-users-per-batch", type=int, default=4)
    parser.add_argument("--max-actions-per-user", type=int, default=512)
    parser.add_argument("--max-sampled-actions", type=int, default=8)
    parser.add_argument(
        "--max-samples-per-action",
        type=int,
        default=2,
        help="Cap samples per action id to keep the dump diverse. Use 0 for no cap.",
    )
    parser.add_argument("--max-positions-per-action", type=int, default=8)
    parser.add_argument("--min-positions-per-action", type=int, default=3)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Layers to save. Omit to save every captured HSTU layer.",
    )
    parser.add_argument(
        "--prefer-action-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional action ids to prioritize when they appear repeatedly.",
    )
    parser.add_argument("--max-retrieval-items", type=int, default=500)
    return parser.parse_args()


def _linspace_select(items: List[Dict], max_items: int) -> List[Dict]:
    if len(items) <= max_items:
        return items
    if max_items <= 1:
        return [items[0]]
    indices = torch.linspace(0, len(items) - 1, steps=max_items).long().tolist()
    return [items[int(idx)] for idx in indices]


def _action_refs_by_user(record: Dict, max_actions_per_user: int) -> Dict[int, List[Dict]]:
    seqlen_offsets = record["seqlen_offsets"]
    contextual = record["contextual_seqlen"]
    action_ids = record["action_ids"]
    action_offset = 0
    refs_by_user: Dict[int, List[Dict]] = {}
    for user_idx in range(len(seqlen_offsets) - 1):
        seq_start = int(seqlen_offsets[user_idx].item())
        seq_end = int(seqlen_offsets[user_idx + 1].item())
        contextual_len = int(contextual[user_idx].item()) if contextual is not None else 0
        action_item_span = max(0, seq_end - seq_start - contextual_len)
        refs = _collect_token_refs(
            token_type="action",
            token_ids=action_ids,
            token_offset=action_offset,
            seq_start=seq_start,
            contextual_len=contextual_len,
            action_item_span=action_item_span,
            key_len=record["key"].shape[0],
            max_tokens_per_user=max_actions_per_user,
        )
        refs_by_user[user_idx] = refs
        action_offset += action_item_span // 2
    return refs_by_user


def _build_sample_specs(
    records: List[Dict],
    *,
    max_actions_per_user: int,
    max_sampled_actions: int,
    max_positions_per_action: int,
    min_positions_per_action: int,
    max_samples_per_action: int,
    prefer_action_ids: Optional[List[int]],
) -> List[Dict]:
    layer0_records = [record for record in records if int(record["layer_idx"]) == 0]
    preferred = set(prefer_action_ids or [])
    candidates = []
    for record in layer0_records:
        refs_by_user = _action_refs_by_user(record, max_actions_per_user)
        for user_idx, refs in refs_by_user.items():
            by_action: Dict[int, List[Dict]] = defaultdict(list)
            for ref in refs:
                by_action[int(ref["token_id"])].append(ref)
            for action_id, action_refs in by_action.items():
                if len(action_refs) < min_positions_per_action:
                    continue
                action_refs = sorted(action_refs, key=lambda ref: ref["local_pos"])
                selected_refs = _linspace_select(action_refs, max_positions_per_action)
                candidates.append(
                    {
                        "batch_idx": int(record["batch_idx"]),
                        "user_idx": int(user_idx),
                        "action_id": int(action_id),
                        "num_occurrences_in_sampled_user": len(action_refs),
                        "selected_refs": selected_refs,
                    }
                )

    candidates.sort(
        key=lambda row: (
            0 if row["action_id"] in preferred else 1,
            -row["num_occurrences_in_sampled_user"],
            row["batch_idx"],
            row["user_idx"],
            row["action_id"],
        )
    )
    selected = []
    selected_count_by_action: Dict[int, int] = defaultdict(int)
    for candidate in candidates:
        action_id = int(candidate["action_id"])
        if max_samples_per_action > 0 and selected_count_by_action[action_id] >= max_samples_per_action:
            continue
        selected.append(candidate)
        selected_count_by_action[action_id] += 1
        if len(selected) >= max_sampled_actions:
            break
    return selected


def _cosine_to_first(vectors: torch.Tensor) -> Tuple[float, float]:
    if vectors.shape[0] <= 1:
        return float("nan"), float("nan")
    flattened = vectors.reshape(vectors.shape[0], -1).float()
    first = flattened[:1].expand_as(flattened[1:])
    cos = F.cosine_similarity(first, flattened[1:], dim=1)
    return float(cos.mean().item()), float(cos.min().item())


def _materialize_samples(
    records: List[Dict],
    sample_specs: List[Dict],
    layers: Optional[Iterable[int]],
) -> Tuple[List[Dict], List[Dict]]:
    requested_layers = set(layers) if layers is not None else None
    records_by_batch_layer = {
        (int(record["batch_idx"]), int(record["layer_idx"])): record for record in records
    }
    available_layers = sorted({int(record["layer_idx"]) for record in records})
    save_layers = [layer for layer in available_layers if requested_layers is None or layer in requested_layers]
    samples = []
    summary_rows = []

    for sample_id, spec in enumerate(sample_specs):
        positions = [
            {
                "local_pos": int(ref["local_pos"]),
                "abs_pos": int(ref["abs_pos"]),
                "token_id": int(ref["token_id"]),
            }
            for ref in spec["selected_refs"]
        ]
        sample = {
            "sample_id": sample_id,
            "batch_idx": spec["batch_idx"],
            "user_idx": spec["user_idx"],
            "action_id": spec["action_id"],
            "num_occurrences_in_sampled_user": spec["num_occurrences_in_sampled_user"],
            "positions": positions,
            "layers": {},
        }

        abs_positions = [pos["abs_pos"] for pos in positions]
        for layer_idx in save_layers:
            record = records_by_batch_layer.get((spec["batch_idx"], layer_idx))
            if record is None:
                continue
            if any(pos >= record["key"].shape[0] for pos in abs_positions):
                continue
            key_vectors = torch.stack([record["key"][pos].clone() for pos in abs_positions], dim=0)
            value_vectors = torch.stack([record["value"][pos].clone() for pos in abs_positions], dim=0)
            key_mean_cos, key_min_cos = _cosine_to_first(key_vectors)
            value_mean_cos, value_min_cos = _cosine_to_first(value_vectors)
            sample["layers"][int(layer_idx)] = {
                "key": key_vectors,
                "value": value_vectors,
                "key_cosine_to_first_mean": key_mean_cos,
                "key_cosine_to_first_min": key_min_cos,
                "value_cosine_to_first_mean": value_mean_cos,
                "value_cosine_to_first_min": value_min_cos,
            }
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "batch_idx": spec["batch_idx"],
                    "user_idx": spec["user_idx"],
                    "action_id": spec["action_id"],
                    "layer_idx": int(layer_idx),
                    "num_occurrences_in_sampled_user": spec["num_occurrences_in_sampled_user"],
                    "saved_position_count": len(abs_positions),
                    "local_positions": json.dumps([pos["local_pos"] for pos in positions]),
                    "key_shape": list(key_vectors.shape),
                    "value_shape": list(value_vectors.shape),
                    "key_cosine_to_first_mean": key_mean_cos,
                    "key_cosine_to_first_min": key_min_cos,
                    "value_cosine_to_first_mean": value_mean_cos,
                    "value_cosine_to_first_min": value_min_cos,
                }
            )
        samples.append(sample)

    return samples, summary_rows


def _build_model_and_capture(args: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    if hasattr(base.gin, "clear_config"):
        base.gin.clear_config()
    base.gin.parse_config_file(args.gin_config_file)

    trainer_args = base.TrainerArgs()
    dataset_args, embedding_args = base.get_dataset_and_embedding_args()
    network_args = base.NetworkArgs()
    optimizer_args = base.OptimizerArgs()
    tp_args = base.TensorModelParallelArgs()
    trainer_args.ckpt_load_dir = args.ckpt_load_dir
    trainer_args.max_eval_iters = args.max_batches

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
            max_batches=args.max_batches,
            max_users_per_batch=args.max_users_per_batch,
            max_actions_per_user=args.max_actions_per_user,
        )
        capture.register_hooks(model_train)
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(islice(eval_dataloader, args.max_batches)):
                    item_ids = _extract_feature_ids(batch, item_feature_name, -2)
                    action_ids = _extract_feature_ids(batch, action_feature_name, -1)
                    capture.set_current_batch(item_ids, action_ids, batch_idx)
                    batch = batch.to(device)
                    pipeline._model(batch)
        finally:
            capture.remove_hooks()

        metadata = {
            "gin_config_file": args.gin_config_file,
            "ckpt_load_dir": args.ckpt_load_dir,
            "max_batches": args.max_batches,
            "max_users_per_batch": args.max_users_per_batch,
            "max_actions_per_user": args.max_actions_per_user,
            "captured_record_count": len(capture.records),
            "captured_layers": sorted({int(record["layer_idx"]) for record in capture.records}),
            "task_type": "retrieval" if is_retrieval else "ranking",
        }
        return capture.records, metadata
    finally:
        base.init.destroy_global_state()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output_file
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.output_dir, output_path)

    records, metadata = _build_model_and_capture(args)
    sample_specs = _build_sample_specs(
        records,
        max_actions_per_user=args.max_actions_per_user,
        max_sampled_actions=args.max_sampled_actions,
        max_positions_per_action=args.max_positions_per_action,
        min_positions_per_action=args.min_positions_per_action,
        max_samples_per_action=args.max_samples_per_action,
        prefer_action_ids=args.prefer_action_ids,
    )
    samples, summary_rows = _materialize_samples(records, sample_specs, args.layers)

    payload = {
        "metadata": {
            **metadata,
            "output_path": output_path,
            "max_sampled_actions": args.max_sampled_actions,
            "max_samples_per_action": args.max_samples_per_action,
            "max_positions_per_action": args.max_positions_per_action,
            "min_positions_per_action": args.min_positions_per_action,
            "requested_layers": args.layers,
            "prefer_action_ids": args.prefer_action_ids,
            "sample_count": len(samples),
        },
        "samples": samples,
    }
    torch.save(payload, output_path)

    summary_path = os.path.join(args.output_dir, "action_kv_vector_samples_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved {len(samples)} repeated-action samples to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
