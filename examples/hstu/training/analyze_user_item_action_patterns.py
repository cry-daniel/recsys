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
Analyze user-level concentration patterns for items and actions.

This script reuses the same sequence loading path as
`training/eval_checkpoint_shuffle.py`:
1. resolve `processed_seqs.csv` through `get_common_preprocessors`
2. read each user's item sequence and action sequence with `load_seq`

It focuses on:
1. Whether some users have a few items appearing disproportionately often
2. Whether some users have a few actions appearing disproportionately often
3. Whether different users have very similar action interaction distributions

Outputs:
- multiple `.png` figures
- `summary.json` with aggregate statistics

Usage:
    cd examples/hstu
    PYTHONPATH=${PYTHONPATH}:$(realpath ../) \
        python ./training/analyze_user_item_action_patterns.py \
        --dataset-name kuairand-1k \
        --output-dir ./training/user_pattern_analysis
"""

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze user item/action concentration and cross-user action similarity."
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
        default="./training/user_pattern_analysis",
        help="Directory for figures and summary.json.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="If > 0, only analyze the first N unique users in the csv.",
    )
    parser.add_argument(
        "--max-users-for-similarity",
        type=int,
        default=500,
        help="Maximum users used for pairwise action-similarity analysis.",
    )
    parser.add_argument(
        "--top-k-users",
        type=int,
        default=20,
        help="How many extreme users to display in bar charts.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required to generate .png outputs. "
            "Please install it in the current environment first."
        )


def safe_float(value) -> float:
    return float(value) if value is not None else 0.0


def load_seq(x):
    if isinstance(x, str):
        return json.loads(x)
    return x


def analyze_sequence_concentration(seq: Sequence[int]) -> Dict[str, float]:
    counts = Counter(seq)
    total = len(seq)
    sorted_counts = sorted(counts.values(), reverse=True)
    top1 = sorted_counts[0] if sorted_counts else 0
    top3 = sum(sorted_counts[:3])
    probs = np.asarray(sorted_counts, dtype=np.float64) / max(total, 1)
    hhi = float(np.square(probs).sum()) if len(probs) > 0 else 0.0
    entropy = float(-(probs * np.log2(np.clip(probs, 1e-12, None))).sum()) if len(probs) > 0 else 0.0
    return {
        "seq_len": total,
        "unique_count": len(counts),
        "top1_count": top1,
        "top1_share": top1 / max(total, 1),
        "top3_count": top3,
        "top3_share": top3 / max(total, 1),
        "unique_ratio": len(counts) / max(total, 1),
        "hhi": hhi,
        "entropy": entropy,
    }


def load_user_sequences(
    dataset_name: str,
    dataset_path: str,
    max_users: int,
) -> Tuple[str, str, str, List[Dict[str, object]]]:
    try:
        dp = get_common_preprocessors(dataset_path)[dataset_name]
        seq_file = dp._output_file
        item_feature_name = dp._item_feature_name
        action_feature_name = dp._action_feature_name
    except Exception:
        seq_file, item_feature_name, action_feature_name = resolve_dataset_paths(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )

    seq_logs_frame = pd.read_csv(seq_file, delimiter=",")

    users: List[Dict[str, object]] = []
    seen_users = set()
    for _, row in seq_logs_frame.iterrows():
        user_id = row["user_id"]
        if user_id in seen_users:
            continue
        users.append(
            {
                "user_id": int(user_id) if not isinstance(user_id, str) else user_id,
                "item_seq": load_seq(row[item_feature_name]),
                "action_seq": load_seq(row[action_feature_name]),
            }
        )
        seen_users.add(user_id)
        if max_users > 0 and len(users) >= max_users:
            break

    return seq_file, item_feature_name, action_feature_name, users


def resolve_dataset_paths(
    dataset_name: str,
    dataset_path: str,
) -> Tuple[str, str, str]:
    base_path = dataset_path if dataset_path else "tmp_data"
    if dataset_name in ("ml-1m", "ml-20m"):
        seq_file = os.path.join(base_path, dataset_name, "processed_seqs.csv")
        return seq_file, "movie_id", "rating"

    kuairand_dir_map = {
        "kuairand-pure": "KuaiRand-Pure",
        "kuairand-1k": "KuaiRand-1K",
        "kuairand-27k": "KuaiRand-27K",
    }
    seq_file = os.path.join(
        base_path,
        kuairand_dir_map[dataset_name],
        "data",
        "processed_seqs.csv",
    )
    return seq_file, "video_id", "action_weights"


def build_concentration_frame(
    users: List[Dict[str, object]],
    field_name: str,
) -> pd.DataFrame:
    rows = []
    for user in users:
        metrics = analyze_sequence_concentration(user[field_name])
        metrics["user_id"] = user["user_id"]
        rows.append(metrics)
    return pd.DataFrame(rows)


def plot_concentration_overview(
    df: pd.DataFrame,
    title_prefix: str,
    output_path: str,
) -> None:
    require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].hist(df["top1_share"], bins=30, color="#1f77b4", alpha=0.85)
    axes[0, 0].set_title(f"{title_prefix}: top-1 share")
    axes[0, 0].set_xlabel("Top-1 share")
    axes[0, 0].set_ylabel("User count")

    axes[0, 1].hist(df["top3_share"], bins=30, color="#ff7f0e", alpha=0.85)
    axes[0, 1].set_title(f"{title_prefix}: top-3 share")
    axes[0, 1].set_xlabel("Top-3 share")
    axes[0, 1].set_ylabel("User count")

    axes[1, 0].scatter(
        df["seq_len"],
        df["top1_share"],
        s=14,
        alpha=0.55,
        color="#2ca02c",
        edgecolors="none",
    )
    axes[1, 0].set_title(f"{title_prefix}: sequence length vs top-1 share")
    axes[1, 0].set_xlabel("Sequence length")
    axes[1, 0].set_ylabel("Top-1 share")

    axes[1, 1].hist(df["hhi"], bins=30, color="#d62728", alpha=0.85)
    axes[1, 1].set_title(f"{title_prefix}: concentration (HHI)")
    axes[1, 1].set_xlabel("HHI")
    axes[1, 1].set_ylabel("User count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_extreme_users(
    df: pd.DataFrame,
    title_prefix: str,
    metric_name: str,
    output_path: str,
    top_k_users: int,
) -> None:
    require_matplotlib()
    show_df = df.nlargest(top_k_users, metric_name).copy()
    show_df["user_label"] = show_df["user_id"].astype(str)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(show_df["user_label"], show_df[metric_name], color="#4c78a8")
    ax.set_title(f"{title_prefix}: users with highest {metric_name}")
    ax.set_xlabel("User ID")
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", labelrotation=75)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def select_users_for_similarity(
    users: List[Dict[str, object]],
    max_users_for_similarity: int,
) -> List[Dict[str, object]]:
    if len(users) <= max_users_for_similarity:
        return users
    return sorted(users, key=lambda x: len(x["action_seq"]), reverse=True)[:max_users_for_similarity]


def compute_action_similarity(
    users: List[Dict[str, object]],
) -> Tuple[np.ndarray, List[object], List[object], np.ndarray]:
    action_values = sorted(
        {
            action
            for user in users
            for action in user["action_seq"]
        }
    )
    action_to_idx = {action: idx for idx, action in enumerate(action_values)}

    matrix = np.zeros((len(users), len(action_values)), dtype=np.float32)
    user_ids: List[object] = []
    for row_idx, user in enumerate(users):
        counts = Counter(user["action_seq"])
        total = max(len(user["action_seq"]), 1)
        for action, count in counts.items():
            matrix[row_idx, action_to_idx[action]] = count / total
        user_ids.append(user["user_id"])

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = matrix / norms
    similarity = normalized @ normalized.T
    return similarity, user_ids, action_values, matrix


def plot_similarity_overview(
    similarity: np.ndarray,
    user_ids: List[object],
    output_dir: str,
) -> Dict[str, object]:
    require_matplotlib()
    np.fill_diagonal(similarity, -1.0)
    nearest = similarity.max(axis=1)
    nearest_idx = similarity.argmax(axis=1)
    pair_scores = []
    for i in range(similarity.shape[0]):
        j = int(nearest_idx[i])
        if i < j:
            pair_scores.append((safe_float(similarity[i, j]), i, j))
    pair_scores.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nearest, bins=30, color="#59a14f", alpha=0.85)
    ax.set_title("Cross-user action similarity: nearest-neighbor similarity")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("User count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_similarity_hist.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    top_heatmap_users = set()
    for _, i, j in pair_scores[:20]:
        top_heatmap_users.add(i)
        top_heatmap_users.add(j)
    top_heatmap_indices = sorted(top_heatmap_users)
    if top_heatmap_indices:
        heatmap = similarity[np.ix_(top_heatmap_indices, top_heatmap_indices)].copy()
        np.fill_diagonal(heatmap, 1.0)
        labels = [str(user_ids[idx]) for idx in top_heatmap_indices]
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title("Cross-user action similarity heatmap (top similar users)")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=75, fontsize=8)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "action_similarity_heatmap.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    top_pairs = pair_scores[:15]
    if top_pairs:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [f"{user_ids[i]} vs {user_ids[j]}" for _, i, j in top_pairs]
        values = [score for score, _, _ in top_pairs]
        ax.bar(labels, values, color="#e15759")
        ax.set_title("Most similar user pairs by action distribution")
        ax.set_xlabel("User pair")
        ax.set_ylabel("Cosine similarity")
        ax.tick_params(axis="x", labelrotation=75)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "action_similarity_top_pairs.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    np.fill_diagonal(similarity, 1.0)
    return {
        "num_users_for_similarity": len(user_ids),
        "mean_nearest_similarity": float(nearest.mean()) if len(nearest) > 0 else 0.0,
        "median_nearest_similarity": float(np.median(nearest)) if len(nearest) > 0 else 0.0,
        "max_pair_similarity": float(top_pairs[0][0]) if top_pairs else 0.0,
        "top_pairs": [
            {
                "user_a": str(user_ids[i]),
                "user_b": str(user_ids[j]),
                "similarity": float(score),
            }
            for score, i, j in top_pairs
        ],
    }


def save_summary(
    output_dir: str,
    seq_file: str,
    item_feature_name: str,
    action_feature_name: str,
    item_df: pd.DataFrame,
    action_df: pd.DataFrame,
    similarity_summary: Dict[str, object],
) -> None:
    summary = {
        "seq_file": seq_file,
        "item_feature_name": item_feature_name,
        "action_feature_name": action_feature_name,
        "num_users": int(len(item_df)),
        "item_concentration": {
            "mean_top1_share": float(item_df["top1_share"].mean()),
            "median_top1_share": float(item_df["top1_share"].median()),
            "p95_top1_share": float(item_df["top1_share"].quantile(0.95)),
            "mean_top3_share": float(item_df["top3_share"].mean()),
            "mean_hhi": float(item_df["hhi"].mean()),
        },
        "action_concentration": {
            "mean_top1_share": float(action_df["top1_share"].mean()),
            "median_top1_share": float(action_df["top1_share"].median()),
            "p95_top1_share": float(action_df["top1_share"].quantile(0.95)),
            "mean_top3_share": float(action_df["top3_share"].mean()),
            "mean_hhi": float(action_df["hhi"].mean()),
        },
        "action_similarity": similarity_summary,
        "top_item_concentration_users": item_df.nlargest(10, "top1_share")[
            ["user_id", "seq_len", "top1_count", "top1_share", "top3_share", "hhi"]
        ].to_dict(orient="records"),
        "top_action_concentration_users": action_df.nlargest(10, "top1_share")[
            ["user_id", "seq_len", "top1_count", "top1_share", "top3_share", "hhi"]
        ].to_dict(orient="records"),
    }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    seq_file, item_feature_name, action_feature_name, users = load_user_sequences(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        max_users=args.max_users,
    )
    if not users:
        raise ValueError(f"No users found in {seq_file}")

    item_df = build_concentration_frame(users, "item_seq")
    action_df = build_concentration_frame(users, "action_seq")

    plot_concentration_overview(
        item_df,
        title_prefix="Item concentration",
        output_path=os.path.join(args.output_dir, "item_concentration_overview.png"),
    )
    plot_extreme_users(
        item_df,
        title_prefix="Item concentration",
        metric_name="top1_share",
        output_path=os.path.join(args.output_dir, "item_concentration_top_users.png"),
        top_k_users=args.top_k_users,
    )
    plot_concentration_overview(
        action_df,
        title_prefix="Action concentration",
        output_path=os.path.join(args.output_dir, "action_concentration_overview.png"),
    )
    plot_extreme_users(
        action_df,
        title_prefix="Action concentration",
        metric_name="top1_share",
        output_path=os.path.join(args.output_dir, "action_concentration_top_users.png"),
        top_k_users=args.top_k_users,
    )

    sim_users = select_users_for_similarity(users, args.max_users_for_similarity)
    similarity, user_ids, _, _ = compute_action_similarity(sim_users)
    similarity_summary = plot_similarity_overview(similarity, user_ids, args.output_dir)

    save_summary(
        output_dir=args.output_dir,
        seq_file=seq_file,
        item_feature_name=item_feature_name,
        action_feature_name=action_feature_name,
        item_df=item_df,
        action_df=action_df,
        similarity_summary=similarity_summary,
    )

    print(f"Loaded {len(users)} users from {seq_file}")
    print(f"Item feature: {item_feature_name}, action feature: {action_feature_name}")
    print(f"Saved analysis outputs to {os.path.abspath(args.output_dir)}")
    print(
        "Key figures: "
        "item_concentration_overview.png, "
        "action_concentration_overview.png, "
        "action_similarity_hist.png, "
        "action_similarity_heatmap.png"
    )


if __name__ == "__main__":
    main()
