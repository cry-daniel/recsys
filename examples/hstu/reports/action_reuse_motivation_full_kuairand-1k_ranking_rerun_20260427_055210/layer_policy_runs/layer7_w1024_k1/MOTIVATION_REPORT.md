# Action KV Reuse Motivation Report

This report is organized around the five questions needed to motivate and tune action KV reuse.

Definitions used throughout: AUC summaries only include tasks whose baseline AUC is above the configured threshold; Pareto optimality is computed in the reuse-ratio vs AUC plane.

## Q1. Why Is Action Better Than Item For Reuse?

Variables:

- `global_unique_ids`: Number of distinct ids in the whole analyzed dataset.
- `mean_user_unique_ids`: Average number of distinct ids per user sequence.
- `mean_user_top1_share`: Average fraction of a user's tokens covered by their most frequent id.
- `mean_user_top3_share`: Average fraction covered by their three most frequent ids.

`token_space_summary.csv` was not generated in this run.

## Q2. How Similar Is Action KV At Different Distances?

`action_kv_same_vs_diff_distance_summary.csv` was not generated in this run.

## Q3. Which Global Top-K/Window Points Are Pareto Optimal?

This grid answers how top-K and window size trade reuse ratio against final inference AUC.

Variables:

- `window_size`: Interleaved-token window size used by action KV reuse.
- `top_k`: Number of high-frequency action ids reused per user/window/layer.
- `reuse_ratio_all_tokens`: Replaced action KV rows divided by all sequence tokens; higher means more compute/cache saved.
- `mean_reuse_auc`: Mean AUC after KV reuse, filtered to tasks whose baseline AUC is above the configured threshold.
- `mean_auc_diff`: Mean AUC change relative to no-reuse baseline.
- `max_auc_drop`: Worst AUC drop among filtered tasks; lower is safer.
- `policy_pareto`: True if no other point has both higher/equal reuse ratio and higher/equal AUC.

`policy_grid_auc_summary.csv` was not generated in this run.

## Q4. Which Layers Are Most Sensitive?

Each run enables reuse in one HSTU layer only. By default this uses top_k=1 and window_size=64, so the table isolates layer sensitivity without running a full top-K/window grid for every layer.

Variables:

- `window_size`: Interleaved-token window size used by action KV reuse.
- `top_k`: Number of high-frequency action ids reused per user/window/layer.
- `reuse_ratio_all_tokens`: Replaced action KV rows divided by all sequence tokens; higher means more compute/cache saved.
- `mean_reuse_auc`: Mean AUC after KV reuse, filtered to tasks whose baseline AUC is above the configured threshold.
- `mean_auc_diff`: Mean AUC change relative to no-reuse baseline.
- `max_auc_drop`: Worst AUC drop among filtered tasks; lower is safer.
- `policy_pareto`: True if no other point has both higher/equal reuse ratio and higher/equal AUC.

`layer_policy_auc_summary.csv` was not generated in this run.

## Q5. How Should Top-K/Window Be Chosen For Different Users?

Each run enables reuse for one user sequence-length bucket only, which estimates the best policy for short vs long users.

Variables:

- `window_size`: Interleaved-token window size used by action KV reuse.
- `top_k`: Number of high-frequency action ids reused per user/window/layer.
- `reuse_ratio_all_tokens`: Replaced action KV rows divided by all sequence tokens; higher means more compute/cache saved.
- `mean_reuse_auc`: Mean AUC after KV reuse, filtered to tasks whose baseline AUC is above the configured threshold.
- `mean_auc_diff`: Mean AUC change relative to no-reuse baseline.
- `max_auc_drop`: Worst AUC drop among filtered tasks; lower is safer.
- `policy_pareto`: True if no other point has both higher/equal reuse ratio and higher/equal AUC.

`user_bucket_policy_auc_summary.csv` was not generated in this run.

## Legacy/Strategy AUC Controls

AUC is averaged only over tasks with baseline AUC > 0.6: `task1.AUC`, `task2.AUC`, `task5.AUC`, `task6.AUC`.

| Mode | Max Distance | Reuse Ratio | Mean AUC | AUC Diff | Max AUC Drop | Replacements | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|
| window_topk_same_id:action | - | 3.61% | 0.698325 | +0.000107 | 0.000000 | 173,404 | yes |

Insight: the legacy first-action and wrong-action rows are negative controls: high replacement without semantic or distance constraints can damage ranking quality. The distance sweep shows how much locality is needed before reuse becomes low-risk.

These controls are useful for explaining what not to do, but Q3-Q5 are the policy-selection sections.
