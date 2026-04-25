# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Motivation experiments for window/top-K action KV reuse.

The default path is dataset-only and runs from processed sequence CSVs. It
quantifies whether action reuse has enough opportunity, why local windows are a
better granularity than one global action cache, and how window/top-K choices
trade off reuse coverage and risk.

An optional checkpoint path captures real HSTU K/V tensors and compares same
action pairs inside vs outside local windows.
"""

'''
默认数据侧运行示例：
cd examples/hstu
PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
python ./training/analyze_action_reuse_motivation.py \
--dataset-name kuairand-1k \
--window-sizes 128 256 512 1024 \
--top-ks 1 2 3 5 \
--output-dir ./analysis_output/action_reuse_motivation

可选真实 KV 分析：

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
python ./training/analyze_action_reuse_motivation.py \
--dataset-name kuairand-1k \
--output-dir ./analysis_output/action_reuse_motivation_kv \
--run-kv-analysis \
--gin-config-file ./training/configs/kuairand_1k_ranking.gin \
--ckpt-load-dir ckpt_kr_1k_ranking/iter1000
'''

import argparse
import datetime
import json
import os
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from preprocessor import dataset_names, get_common_preprocessors

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


@dataclass(frozen=True)
class UserSequence:
    user_id: object
    item_seq: List[int]
    action_seq: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze motivation evidence for window/top-K action KV reuse."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=list(dataset_names),
        help="Dataset name used by examples/hstu preprocessor.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="",
        help="Optional dataset root passed to get_common_preprocessors().",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training/action_reuse_motivation",
        help="Directory for CSV, JSON, and PNG outputs.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="If > 0, only analyze the first N unique users.",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Interleaved token window sizes to sweep.",
    )
    parser.add_argument(
        "--top-ks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Top-K action counts to sweep per user/window.",
    )
    parser.add_argument(
        "--long-user-buckets",
        type=int,
        nargs="+",
        default=[0, 256, 1024],
        help="Sequence-length bucket starts in interleaved token units.",
    )
    parser.add_argument(
        "--max-risk-pairs-per-user",
        type=int,
        default=2000,
        help="Maximum same-action pairs sampled per user for global-vs-window risk.",
    )
    parser.add_argument(
        "--top-k-users",
        type=int,
        default=20,
        help="How many users to show in high-concentration bar charts.",
    )
    parser.add_argument(
        "--max-window-rows",
        type=int,
        default=5000,
        help=(
            "Maximum detailed rows written to window_action_concentration.csv. "
            "Use 0 to write all rows."
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG generation for faster sweep/debug runs.",
    )

    # Optional real-KV analysis. This path is intentionally opt-in because it
    # requires the same GPU/checkpoint environment as eval_checkpoint scripts.
    parser.add_argument("--run-kv-analysis", action="store_true")
    parser.add_argument(
        "--run-auc-impact-analysis",
        action="store_true",
        help=(
            "Run checkpoint inference with action/item/both KV reuse modes and write "
            "reuse_auc_impact_*.csv. Requires --gin-config-file and --ckpt-load-dir."
        ),
    )
    parser.add_argument(
        "--run-policy-grid-auc-analysis",
        action="store_true",
        help=(
            "Run global window/top-K AUC grid for Q3. Uses --window-sizes and --top-ks."
        ),
    )
    parser.add_argument(
        "--run-layer-policy-auc-analysis",
        action="store_true",
        help=(
            "Run per-layer AUC sensitivity for Q4. One layer is enabled per run."
        ),
    )
    parser.add_argument(
        "--layer-window-sizes",
        type=int,
        nargs="+",
        default=[64],
        help=(
            "Interleaved token window sizes for --run-layer-policy-auc-analysis. "
            "Kept separate from --window-sizes so global grid search can be broad "
            "while layer sensitivity stays cheap."
        ),
    )
    parser.add_argument(
        "--layer-top-ks",
        type=int,
        nargs="+",
        default=[1],
        help=(
            "Top-K values for --run-layer-policy-auc-analysis. "
            "Default top1 isolates layer sensitivity without running a full grid per layer."
        ),
    )
    parser.add_argument(
        "--run-user-bucket-policy-auc-analysis",
        action="store_true",
        help=(
            "Run user-length-bucket window/top-K AUC grid for Q5. "
            "One length bucket is enabled per run."
        ),
    )
    parser.add_argument("--gin-config-file", type=str, default=None)
    parser.add_argument("--ckpt-load-dir", type=str, default=None)
    parser.add_argument("--kv-max-batches", type=int, default=3)
    parser.add_argument("--kv-max-users-per-batch", type=int, default=4)
    parser.add_argument("--kv-max-actions-per-user", type=int, default=64)
    parser.add_argument(
        "--kv-max-pairs-per-token-id",
        type=int,
        default=0,
        help=(
            "If > 0, cap same-user/same-id KV pairs per layer/token id. "
            "Useful when increasing --kv-max-actions-per-user to capture repeated items."
        ),
    )
    parser.add_argument(
        "--kv-max-different-action-pairs-per-user",
        type=int,
        default=2000,
        help="Maximum different-action KV pairs sampled per user/layer for same-vs-different action analysis.",
    )
    parser.add_argument(
        "--kv-distance-buckets",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help=(
            "Interleaved-token distance bucket upper bounds for real-KV similarity. "
            "For example 64 128 256 creates <=64, 65-128, 129-256, and >256."
        ),
    )
    parser.add_argument(
        "--kv-max-pairs-per-distance-bucket",
        type=int,
        default=0,
        help=(
            "If > 0, cap sampled real-KV pairs per distance bucket and pair type. "
            "This prevents one very dense bucket from dominating distance-aware summaries."
        ),
    )
    parser.add_argument("--kv-window-size", type=int, default=512)
    parser.add_argument(
        "--write-raw-kv-pairs",
        action="store_true",
        help=(
            "Write full real-KV pair-level CSVs. By default only compact summaries "
            "and small sample CSVs are written to keep motivation outputs small."
        ),
    )
    parser.add_argument(
        "--max-raw-kv-pair-rows",
        type=int,
        default=10000,
        help=(
            "Maximum sampled pair rows to write when --write-raw-kv-pairs is not set. "
            "Use 0 to skip sampled pair CSVs."
        ),
    )
    parser.add_argument(
        "--auc-reuse-token-types",
        type=str,
        nargs="+",
        default=["action"],
        choices=["action", "item", "both"],
        help="Reuse modes evaluated by --run-auc-impact-analysis.",
    )
    parser.add_argument(
        "--auc-reuse-strategies",
        type=str,
        nargs="+",
        default=[
            "global_first_any_action_legacy",
            "same_id_max_distance",
            "same_id_max_distance_no_chain",
            "wrong_id_same_window",
            "global_same_id",
            "global_topk_same_id",
            "window_topk_same_id",
            "window_topk_same_id_max_distance",
        ],
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
        help="Reuse strategies evaluated by --run-auc-impact-analysis.",
    )
    parser.add_argument(
        "--auc-reuse-max-distances",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, -1],
        help=(
            "Distance thresholds for same_id_max_distance AUC sweeps. "
            "Values are interleaved-token distances; -1 means global."
        ),
    )
    parser.add_argument(
        "--auc-filter-baseline-threshold",
        type=float,
        default=0.6,
        help="Only metrics with baseline AUC above this threshold are included in filtered AUC summaries.",
    )
    parser.add_argument(
        "--auc-reuse-window-size",
        type=int,
        default=512,
        help="Default local interleaved-token window size for AUC impact analysis.",
    )
    parser.add_argument(
        "--auc-reuse-top-k",
        type=int,
        default=3,
        help="Default top-K frequent token IDs per user/window/layer for AUC impact analysis.",
    )
    parser.add_argument(
        "--auc-reuse-policy-json",
        type=str,
        default=None,
        help="Optional inline JSON or JSON path for layer/user-wise AUC impact reuse policy.",
    )
    parser.add_argument(
        "--auc-kv-replace-implementation",
        type=str,
        default="hidden_proxy",
        choices=["hidden_proxy", "kv_only"],
        help=(
            "Implementation used by checkpoint AUC impact analysis. "
            "hidden_proxy copies layer inputs before UVQK; kv_only copies projected K/V only."
        ),
    )
    parser.add_argument(
        "--auc-max-eval-iters",
        type=int,
        default=None,
        help="Optional eval batch cap for --run-auc-impact-analysis smoke/debug runs.",
    )
    parser.add_argument("--max-retrieval-items", type=int, default=500)
    parser.add_argument(
        "--report-command",
        type=str,
        default=None,
        help=(
            "Optional exact shell command to embed in generated reports. "
            "If omitted, the script records a reconstructed python command."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def current_report_command(args: argparse.Namespace) -> str:
    if args.report_command:
        return args.report_command
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def record_report_command(args: argparse.Namespace, stage: str) -> None:
    ensure_dir(args.output_dir)
    path = os.path.join(args.output_dir, "report_commands.json")
    rows = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except (json.JSONDecodeError, OSError):
            rows = []
    rows.append(
        {
            "stage": stage,
            "command": current_report_command(args),
            "recorded_at_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for PNG outputs.")


def load_seq(value) -> List[int]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, list):
        return value
    return list(value)


def resolve_dataset_paths(dataset_name: str, dataset_path: str) -> Tuple[str, str, str]:
    base_path = dataset_path if dataset_path else "tmp_data"
    if dataset_name in ("ml-1m", "ml-20m"):
        return os.path.join(base_path, dataset_name, "processed_seqs.csv"), "movie_id", "rating"

    kuairand_dir_map = {
        "kuairand-pure": "KuaiRand-Pure",
        "kuairand-1k": "KuaiRand-1K",
        "kuairand-27k": "KuaiRand-27K",
    }
    return (
        os.path.join(base_path, kuairand_dir_map[dataset_name], "data", "processed_seqs.csv"),
        "video_id",
        "action_weights",
    )


def load_user_sequences(
    dataset_name: str,
    dataset_path: str,
    max_users: int,
) -> Tuple[str, str, str, List[UserSequence]]:
    try:
        dp = get_common_preprocessors(dataset_path)[dataset_name]
        seq_file = dp._output_file
        item_feature_name = dp._item_feature_name
        action_feature_name = dp._action_feature_name
    except Exception:
        seq_file, item_feature_name, action_feature_name = resolve_dataset_paths(
            dataset_name, dataset_path
        )

    seq_logs_frame = pd.read_csv(
        seq_file,
        delimiter=",",
        usecols=["user_id", item_feature_name, action_feature_name],
    )
    users: List[UserSequence] = []
    seen_users = set()
    for _, row in seq_logs_frame.iterrows():
        user_id = row["user_id"]
        if user_id in seen_users:
            continue
        users.append(
            UserSequence(
                user_id=int(user_id) if not isinstance(user_id, str) else user_id,
                item_seq=load_seq(row[item_feature_name]),
                action_seq=load_seq(row[action_feature_name]),
            )
        )
        seen_users.add(user_id)
        if max_users > 0 and len(users) >= max_users:
            break
    return seq_file, item_feature_name, action_feature_name, users


def concentration_metrics(seq: Sequence[int], top_ks: Sequence[int]) -> Dict[str, float]:
    counts = Counter(seq)
    total = len(seq)
    sorted_counts = sorted(counts.values(), reverse=True)
    probs = np.asarray(sorted_counts, dtype=np.float64) / max(total, 1)
    row: Dict[str, float] = {
        "seq_len": float(total),
        "interleaved_seq_len": float(total * 2),
        "unique_actions": float(len(counts)),
        "unique_ratio": float(len(counts) / max(total, 1)),
        "hhi": float(np.square(probs).sum()) if probs.size else 0.0,
        "entropy": float(-(probs * np.log2(np.clip(probs, 1e-12, None))).sum())
        if probs.size
        else 0.0,
    }
    for k in sorted(set(top_ks) | {1, 3}):
        row[f"top{k}_count"] = float(sum(sorted_counts[:k]))
        row[f"top{k}_share"] = float(sum(sorted_counts[:k]) / max(total, 1))
    return row


def action_window_size(interleaved_window_size: int) -> int:
    return max(1, interleaved_window_size // 2)


def iter_action_windows(actions: Sequence[int], interleaved_window_size: int) -> Iterable[Tuple[int, int, Sequence[int]]]:
    step = action_window_size(interleaved_window_size)
    for start in range(0, len(actions), step):
        end = min(len(actions), start + step)
        yield start, end, actions[start:end]


def bucket_label(interleaved_len: int, bucket_starts: Sequence[int]) -> str:
    starts = sorted(set(bucket_starts))
    bucket_start = max([s for s in starts if interleaved_len >= s], default=0)
    bucket_end_candidates = [s for s in starts if s > bucket_start]
    return (
        f"{bucket_start}+"
        if not bucket_end_candidates
        else f"{bucket_start}-{bucket_end_candidates[0] - 1}"
    )


def build_user_concentration_frame(users: List[UserSequence], top_ks: Sequence[int]) -> pd.DataFrame:
    rows = []
    for user in users:
        row = concentration_metrics(user.action_seq, top_ks)
        row["user_id"] = user.user_id
        rows.append(row)
    return pd.DataFrame(rows)


def build_token_space_summary_frame(
    users: List[UserSequence],
    top_ks: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for token_type, seq_getter in (
        ("item", lambda user: user.item_seq),
        ("action", lambda user: user.action_seq),
    ):
        global_counts = Counter()
        per_user_unique = []
        per_user_unique_ratio = []
        per_user_top_share = {top_k: [] for top_k in sorted(set(top_ks) | {1, 3})}
        token_count = 0
        for user in users:
            seq = seq_getter(user)
            counts = Counter(seq)
            sorted_counts = sorted(counts.values(), reverse=True)
            total = len(seq)
            token_count += total
            global_counts.update(seq)
            per_user_unique.append(len(counts))
            per_user_unique_ratio.append(len(counts) / max(total, 1))
            for top_k in per_user_top_share:
                per_user_top_share[top_k].append(sum(sorted_counts[:top_k]) / max(total, 1))
        row = {
            "token_type": token_type,
            "num_users": len(users),
            "token_count": token_count,
            "global_unique_ids": len(global_counts),
            "global_repeat_rate": 1.0 - len(global_counts) / max(token_count, 1),
            "mean_user_unique_ids": float(np.mean(per_user_unique)) if per_user_unique else 0.0,
            "mean_user_unique_ratio": float(np.mean(per_user_unique_ratio))
            if per_user_unique_ratio
            else 0.0,
        }
        for top_k, values in per_user_top_share.items():
            row[f"mean_user_top{top_k}_share"] = float(np.mean(values)) if values else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_analysis_frames_fast(
    users: List[UserSequence],
    window_sizes: Sequence[int],
    top_ks: Sequence[int],
    bucket_starts: Sequence[int],
    max_window_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build window, policy, and bucket frames with one pass per user/window size."""
    top_ks = sorted(set(top_ks))
    top_ks_for_windows = sorted(set(top_ks) | {1, 3})
    policy_totals: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    bucket_totals: Dict[Tuple[str, int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    window_rows: List[Dict[str, object]] = []
    write_all_window_rows = max_window_rows <= 0

    for user in users:
        actions = user.action_seq
        action_tokens = len(actions)
        interleaved_len = action_tokens * 2
        user_bucket = bucket_label(interleaved_len, bucket_starts)

        for window_size in window_sizes:
            per_user_selected = {top_k: 0 for top_k in top_ks}
            per_user_candidate = {top_k: 0 for top_k in top_ks}
            per_user_selected_types = {top_k: 0 for top_k in top_ks}
            per_user_unique_refs = {top_k: 0 for top_k in top_ks}
            total_windows = 0
            eligible_windows = 0
            unique_actions_sum = 0

            for start, end, window_actions in iter_action_windows(actions, window_size):
                total_windows += 1
                if not window_actions:
                    continue
                eligible_windows += 1
                counts = Counter(window_actions)
                unique_actions_sum += len(counts)
                first_idx = {}
                for idx, action in enumerate(window_actions):
                    first_idx.setdefault(action, idx)
                ranked_actions = sorted(
                    counts.keys(), key=lambda action: (-counts[action], first_idx[action], action)
                )
                ranked_counts = [counts[action] for action in ranked_actions]
                cumulative_counts = np.cumsum(ranked_counts) if ranked_counts else np.asarray([])
                cumulative_candidates = (
                    np.cumsum([max(0, count - 1) for count in ranked_counts])
                    if ranked_counts
                    else np.asarray([])
                )

                if write_all_window_rows or len(window_rows) < max_window_rows:
                    metrics = concentration_metrics(window_actions, top_ks_for_windows)
                    metrics.update(
                        {
                            "user_id": user.user_id,
                            "user_action_len": action_tokens,
                            "user_interleaved_len": interleaved_len,
                            "window_size": window_size,
                            "window_start_action": start,
                            "window_end_action": end,
                            "window_action_count": len(window_actions),
                            "is_row_limited_sample": not write_all_window_rows,
                        }
                    )
                    window_rows.append(metrics)

                for top_k in top_ks:
                    selected_n = min(top_k, len(ranked_counts))
                    if selected_n == 0:
                        continue
                    per_user_selected[top_k] += int(cumulative_counts[selected_n - 1])
                    per_user_candidate[top_k] += int(cumulative_candidates[selected_n - 1])
                    per_user_selected_types[top_k] += selected_n
                    per_user_unique_refs[top_k] += selected_n

            for top_k in top_ks:
                key = (window_size, top_k)
                totals = policy_totals[key]
                totals["action_tokens"] += action_tokens
                totals["selected_tokens"] += per_user_selected[top_k]
                totals["candidate_tokens"] += per_user_candidate[top_k]
                totals["selected_action_types"] += per_user_selected_types[top_k]
                totals["unique_selected_refs"] += per_user_unique_refs[top_k]
                totals["total_windows"] += total_windows
                totals["eligible_windows"] += eligible_windows
                totals["unique_actions_sum"] += unique_actions_sum

                coverage = per_user_selected[top_k] / max(action_tokens, 1)
                candidate_rate = per_user_candidate[top_k] / max(action_tokens, 1)
                bucket_key = (user_bucket, window_size, top_k)
                bucket = bucket_totals[bucket_key]
                bucket["coverage_sum"] += coverage
                bucket["candidate_rate_sum"] += candidate_rate
                bucket["action_tokens"] += action_tokens
                bucket["num_users"] += 1
                bucket["user_interleaved_len_sum"] += interleaved_len

    policy_rows = []
    for (window_size, top_k), totals in sorted(policy_totals.items()):
        action_tokens = max(totals["action_tokens"], 1.0)
        eligible_windows = max(totals["eligible_windows"], 1.0)
        policy_rows.append(
            {
                "window_size": window_size,
                "top_k": top_k,
                "num_users": len(users),
                "action_tokens": int(totals["action_tokens"]),
                "selected_tokens": int(totals["selected_tokens"]),
                "candidate_tokens": int(totals["candidate_tokens"]),
                "selected_action_types": int(totals["selected_action_types"]),
                "unique_selected_refs": int(totals["unique_selected_refs"]),
                "total_windows": int(totals["total_windows"]),
                "eligible_windows": int(totals["eligible_windows"]),
                "coverage": totals["selected_tokens"] / action_tokens,
                "candidate_rate": totals["candidate_tokens"] / action_tokens,
                "kv_storage_reduction_est": totals["candidate_tokens"] / action_tokens,
                "mean_selected_action_types_per_window": totals["selected_action_types"] / eligible_windows,
                "mean_unique_actions_per_window": totals["unique_actions_sum"] / eligible_windows,
            }
        )

    bucket_rows = []
    for (bucket_name, window_size, top_k), totals in sorted(bucket_totals.items()):
        num_users = max(totals["num_users"], 1.0)
        bucket_rows.append(
            {
                "bucket": bucket_name,
                "window_size": window_size,
                "top_k": top_k,
                "coverage": totals["coverage_sum"] / num_users,
                "candidate_rate": totals["candidate_rate_sum"] / num_users,
                "action_tokens": int(totals["action_tokens"]),
                "num_users": int(totals["num_users"]),
                "mean_user_interleaved_len": totals["user_interleaved_len_sum"] / num_users,
            }
        )

    return pd.DataFrame(window_rows), pd.DataFrame(policy_rows), pd.DataFrame(bucket_rows)


def build_global_vs_window_risk_frame(
    users: List[UserSequence],
    window_sizes: Sequence[int],
    max_pairs_per_user: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2026)
    positions_by_user = []
    for user in users:
        positions_by_action: Dict[int, List[int]] = defaultdict(list)
        for pos, action in enumerate(user.action_seq):
            positions_by_action[action].append(pos)
        positions_by_user.append(positions_by_action)

    for window_size in window_sizes:
        action_step = action_window_size(window_size)
        total_pairs = 0
        same_window_pairs = 0
        sampled_distances: List[int] = []
        sampled_cross_distances: List[int] = []
        for positions_by_action in positions_by_user:
            pair_groups: List[Tuple[List[int], int]] = []
            user_pair_count = 0
            for positions in positions_by_action.values():
                n = len(positions)
                if n < 2:
                    continue
                pair_count = n * (n - 1) // 2
                user_pair_count += pair_count
                pair_groups.append((positions, pair_count))

                total_pairs += pair_count
                window_counts = Counter(pos // action_step for pos in positions)
                same_window_pairs += sum(
                    count * (count - 1) // 2 for count in window_counts.values()
                )

            if user_pair_count == 0 or max_pairs_per_user == 0:
                continue

            sample_count = min(max_pairs_per_user, user_pair_count)
            weights = np.asarray([pair_count for _, pair_count in pair_groups], dtype=np.float64)
            weights /= weights.sum()
            sampled_group_indices = rng.choice(
                len(pair_groups),
                size=sample_count,
                replace=True,
                p=weights,
            )
            for group_idx in sampled_group_indices:
                positions, _ = pair_groups[int(group_idx)]
                left_idx, right_idx = rng.choice(len(positions), size=2, replace=False)
                pos_i, pos_j = sorted((positions[int(left_idx)], positions[int(right_idx)]))
                distance = pos_j - pos_i
                sampled_distances.append(distance)
                same_window = (pos_i // action_step) == (pos_j // action_step)
                if not same_window:
                    sampled_cross_distances.append(distance)
        rows.append(
            {
                "window_size": window_size,
                "same_action_pairs": total_pairs,
                "same_window_pair_share": same_window_pairs / max(total_pairs, 1),
                "cross_window_pair_share": 1.0 - same_window_pairs / max(total_pairs, 1),
                "mean_pair_action_distance": float(np.mean(sampled_distances))
                if sampled_distances
                else 0.0,
                "p50_pair_action_distance": float(np.percentile(sampled_distances, 50))
                if sampled_distances
                else 0.0,
                "p90_cross_window_action_distance": float(np.percentile(sampled_cross_distances, 90))
                if sampled_cross_distances
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_user_concentration(user_df: pd.DataFrame, output_dir: str, top_k_users: int) -> None:
    require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(user_df["top1_share"], bins=30, color="#4c78a8", alpha=0.85)
    axes[0].set_title("User action top-1 share")
    axes[0].set_xlabel("Share")
    axes[0].set_ylabel("Users")
    axes[1].hist(user_df["top3_share"], bins=30, color="#f58518", alpha=0.85)
    axes[1].set_title("User action top-3 share")
    axes[1].set_xlabel("Share")
    axes[2].scatter(
        user_df["interleaved_seq_len"],
        user_df["top3_share"],
        s=14,
        alpha=0.6,
        color="#54a24b",
        edgecolors="none",
    )
    axes[2].set_title("Sequence length vs top-3 share")
    axes[2].set_xlabel("Interleaved sequence length")
    axes[2].set_ylabel("Top-3 share")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_topk_share_by_user.png"), dpi=160)
    plt.close(fig)

    show_df = user_df.nlargest(top_k_users, "top1_share")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(show_df["user_id"].astype(str), show_df["top1_share"], color="#4c78a8")
    ax.set_title("Users with highest action top-1 share")
    ax.set_xlabel("User ID")
    ax.set_ylabel("Top-1 share")
    ax.tick_params(axis="x", labelrotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_top1_extreme_users.png"), dpi=160)
    plt.close(fig)


def plot_policy_heatmap(policy_df: pd.DataFrame, output_dir: str, value_col: str, output_name: str) -> None:
    require_matplotlib()
    pivot = policy_df.pivot(index="top_k", columns="window_size", values=value_col)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("Interleaved window size")
    ax.set_ylabel("Top-K")
    ax.set_title(value_col.replace("_", " ").title())
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_name), dpi=160)
    plt.close(fig)


def plot_reuse_frontier(policy_df: pd.DataFrame, output_dir: str) -> None:
    require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    for top_k, group in policy_df.groupby("top_k"):
        group = group.sort_values("candidate_rate")
        ax.plot(
            group["candidate_rate"],
            group["coverage"],
            marker="o",
            label=f"topK={top_k}",
        )
        for _, row in group.iterrows():
            ax.text(
                row["candidate_rate"],
                row["coverage"],
                f" w{int(row['window_size'])}",
                fontsize=8,
            )
    ax.set_xlabel("Candidate reuse rate")
    ax.set_ylabel("Top-K action coverage")
    ax.set_title("Reuse savings frontier")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reuse_savings_frontier.png"), dpi=160)
    plt.close(fig)


def plot_user_bucket_policy(bucket_df: pd.DataFrame, output_dir: str) -> None:
    require_matplotlib()
    summary = (
        bucket_df.groupby(["bucket", "window_size", "top_k"])
        .agg({"coverage": "mean", "candidate_rate": "mean"})
        .reset_index()
    )
    default_df = summary[summary["window_size"] == 512].copy()
    if default_df.empty:
        default_window = sorted(summary["window_size"].unique())[0]
        default_df = summary[summary["window_size"] == default_window].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for top_k, group in default_df.groupby("top_k"):
        group = group.sort_values("bucket")
        axes[0].plot(group["bucket"], group["coverage"], marker="o", label=f"topK={top_k}")
        axes[1].plot(group["bucket"], group["candidate_rate"], marker="o", label=f"topK={top_k}")
    axes[0].set_title("Coverage by user length bucket")
    axes[0].set_ylabel("Mean coverage")
    axes[1].set_title("Reuse candidate rate by user length bucket")
    axes[1].set_ylabel("Mean candidate rate")
    for ax in axes:
        ax.set_xlabel("Interleaved sequence length bucket")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "policy_by_user_length_bucket.png"), dpi=160)
    plt.close(fig)


def plot_global_vs_window_risk(risk_df: pd.DataFrame, output_dir: str) -> None:
    require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        risk_df["window_size"],
        risk_df["same_window_pair_share"],
        marker="o",
        color="#4c78a8",
    )
    axes[0].set_title("Same-action pairs retained inside one window")
    axes[0].set_xlabel("Interleaved window size")
    axes[0].set_ylabel("Same-window pair share")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        risk_df["window_size"],
        risk_df["p90_cross_window_action_distance"],
        marker="o",
        color="#e45756",
    )
    axes[1].set_title("Cross-window same-action distance")
    axes[1].set_xlabel("Interleaved window size")
    axes[1].set_ylabel("P90 action-position distance")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_vs_window_position_distance.png"), dpi=160)
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def save_summary(
    *,
    output_dir: str,
    seq_file: str,
    item_feature_name: str,
    action_feature_name: str,
    user_df: pd.DataFrame,
    token_space_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> None:
    default_rows = policy_df[(policy_df["window_size"] == 512) & (policy_df["top_k"] == 3)]
    default_policy = default_rows.iloc[0].to_dict() if not default_rows.empty else {}
    summary = {
        "seq_file": seq_file,
        "item_feature_name": item_feature_name,
        "action_feature_name": action_feature_name,
        "num_users": int(len(user_df)),
        "action_concentration": {
            "mean_top1_share": float(user_df["top1_share"].mean()),
            "median_top1_share": float(user_df["top1_share"].median()),
            "p95_top1_share": float(user_df["top1_share"].quantile(0.95)),
            "mean_top3_share": float(user_df["top3_share"].mean()),
            "median_top3_share": float(user_df["top3_share"].median()),
            "mean_hhi": float(user_df["hhi"].mean()),
            "mean_entropy": float(user_df["entropy"].mean()),
        },
        "token_space": json_safe(token_space_df.to_dict(orient="records")),
        "default_512_top3_policy": {
            key: json_safe(value)
            for key, value in default_policy.items()
        },
        "best_coverage_policy": json_safe(
            policy_df.nlargest(1, "coverage").to_dict(orient="records")
        ),
        "best_candidate_rate_policy": json_safe(
            policy_df.nlargest(1, "candidate_rate").to_dict(orient="records")
        ),
        "global_vs_window_risk": json_safe(risk_df.to_dict(orient="records")),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_data_analysis(args: argparse.Namespace) -> Tuple[str, str, str, List[UserSequence]]:
    ensure_dir(args.output_dir)
    record_report_command(args, "data_analysis")
    seq_file, item_feature_name, action_feature_name, users = load_user_sequences(
        args.dataset_name,
        args.dataset_path,
        args.max_users,
    )
    if not users:
        raise ValueError(f"No users found in {seq_file}")

    top_ks = sorted(set(args.top_ks) | {1, 3})
    user_df = build_user_concentration_frame(users, top_ks)
    token_space_df = build_token_space_summary_frame(users, top_ks)
    window_df, policy_df, bucket_df = build_analysis_frames_fast(
        users,
        args.window_sizes,
        args.top_ks,
        args.long_user_buckets,
        args.max_window_rows,
    )
    risk_df = build_global_vs_window_risk_frame(
        users, args.window_sizes, args.max_risk_pairs_per_user
    )

    user_df.to_csv(os.path.join(args.output_dir, "user_action_concentration.csv"), index=False)
    token_space_df.to_csv(os.path.join(args.output_dir, "token_space_summary.csv"), index=False)
    window_df.to_csv(os.path.join(args.output_dir, "window_action_concentration.csv"), index=False)
    policy_df.to_csv(os.path.join(args.output_dir, "reuse_policy_sweep.csv"), index=False)
    bucket_df.to_csv(os.path.join(args.output_dir, "policy_by_user_length_bucket.csv"), index=False)
    risk_df.to_csv(os.path.join(args.output_dir, "global_vs_window_risk.csv"), index=False)

    if not args.skip_plots:
        plot_user_concentration(user_df, args.output_dir, args.top_k_users)
        plot_policy_heatmap(policy_df, args.output_dir, "coverage", "window_topk_coverage_heatmap.png")
        plot_policy_heatmap(
            policy_df,
            args.output_dir,
            "candidate_rate",
            "window_topk_candidate_rate_heatmap.png",
        )
        plot_reuse_frontier(policy_df, args.output_dir)
        plot_user_bucket_policy(bucket_df, args.output_dir)
        plot_global_vs_window_risk(risk_df, args.output_dir)

    save_summary(
        output_dir=args.output_dir,
        seq_file=seq_file,
        item_feature_name=item_feature_name,
        action_feature_name=action_feature_name,
        user_df=user_df,
        token_space_df=token_space_df,
        policy_df=policy_df,
        risk_df=risk_df,
    )
    return seq_file, item_feature_name, action_feature_name, users


def run_optional_kv_analysis(args: argparse.Namespace) -> None:
    if not (
        args.run_kv_analysis
        or args.run_auc_impact_analysis
        or args.run_policy_grid_auc_analysis
        or args.run_layer_policy_auc_analysis
        or args.run_user_bucket_policy_auc_analysis
    ):
        return
    if args.gin_config_file is None or args.ckpt_load_dir is None:
        raise ValueError(
            "--run-kv-analysis/--run-auc-impact-analysis/--run-*-policy-auc-analysis "
            "requires --gin-config-file and --ckpt-load-dir"
        )

    # Keep the heavyweight HSTU/GPU path isolated so the default data-side
    # analysis remains usable in lightweight environments.
    from analyze_action_reuse_motivation_kv_impl import (
        _write_motivation_insights_markdown,
        run_policy_grid_auc_analysis,
        run_auc_impact_analysis,
        run_kv_motivation_analysis,
    )

    if args.run_kv_analysis:
        record_report_command(args, "kv_analysis")
        run_kv_motivation_analysis(args)
    if args.run_auc_impact_analysis:
        record_report_command(args, "auc_impact_analysis")
        run_auc_impact_analysis(args)
    if args.run_policy_grid_auc_analysis:
        record_report_command(args, "policy_grid_auc_analysis")
        run_policy_grid_auc_analysis(args, mode="global")
    if args.run_layer_policy_auc_analysis:
        record_report_command(args, "layer_policy_auc_analysis")
        run_policy_grid_auc_analysis(args, mode="layer")
    if args.run_user_bucket_policy_auc_analysis:
        record_report_command(args, "user_bucket_policy_auc_analysis")
        run_policy_grid_auc_analysis(args, mode="user_bucket")
    if (
        args.run_kv_analysis
        and not args.run_auc_impact_analysis
        and not args.run_policy_grid_auc_analysis
        and not args.run_layer_policy_auc_analysis
        and not args.run_user_bucket_policy_auc_analysis
    ):
        _write_motivation_insights_markdown(
            args.output_dir,
            auc_summary=pd.DataFrame(),
            eligible_metrics=[],
            threshold=args.auc_filter_baseline_threshold,
        )


def main() -> None:
    args = parse_args()
    seq_file, item_feature_name, action_feature_name, users = run_data_analysis(args)
    run_optional_kv_analysis(args)
    if not (
        args.run_kv_analysis
        or args.run_auc_impact_analysis
        or args.run_policy_grid_auc_analysis
        or args.run_layer_policy_auc_analysis
        or args.run_user_bucket_policy_auc_analysis
    ):
        try:
            from analyze_action_reuse_motivation_kv_impl import _write_motivation_insights_markdown

            _write_motivation_insights_markdown(
                args.output_dir,
                auc_summary=pd.DataFrame(),
                eligible_metrics=[],
                threshold=args.auc_filter_baseline_threshold,
            )
        except ModuleNotFoundError:
            pass
    print(f"Loaded {len(users)} users from {seq_file}")
    print(f"Item feature: {item_feature_name}, action feature: {action_feature_name}")
    print(f"Saved action reuse motivation outputs to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
