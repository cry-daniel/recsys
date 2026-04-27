# Action KV Reuse Motivation Report

This report is organized around the five questions needed to motivate and tune action KV reuse.

Definitions used throughout: AUC summaries only include tasks whose baseline AUC is above the configured threshold; Pareto optimality is computed in the reuse-ratio vs AUC plane.

## Q1. Why Is Action Better Than Item For Reuse?

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name ml-20m --output-dir ./analysis_output/action_reuse_motivation_full_ml-20m_ranking_rerun_20260427_063950 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-layer-policy-auc-analysis --gin-config-file ./training/configs/movielen_ranking.gin --ckpt-load-dir ckpt_ml_20m_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

Variables:

- `global_unique_ids`: Number of distinct ids in the whole analyzed dataset.
- `mean_user_unique_ids`: Average number of distinct ids per user sequence.
- `mean_user_top1_share`: Average fraction of a user's tokens covered by their most frequent id.
- `mean_user_top3_share`: Average fraction covered by their three most frequent ids.

| Token Type | Global Unique IDs | Mean User Unique IDs | Mean User Top-1 Share | Mean User Top-3 Share |
|---|---:|---:|---:|---:|
| item | 26,744 | 144.41 | 1.87% | 5.61% |
| action | 10 | 6.37 | 38.20% | 80.23% |

Insight: action has a much smaller semantic space than item, so repeated action ids are common enough to create a real reuse opportunity.

## Q2. How Similar Is Action KV At Different Distances?

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name ml-20m --output-dir ./analysis_output/action_reuse_motivation_full_ml-20m_ranking_rerun_20260427_063950 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-kv-analysis --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/movielen_ranking.gin --ckpt-load-dir ckpt_ml_20m_ranking/iter1000 --kv-distance-buckets 64 128 256 512 1024 --kv-max-pairs-per-distance-bucket 5000 --max-raw-kv-pair-rows 1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

Variables:

- `same action`: Both KV rows come from the same action id in the same user sequence.
- `different action`: The two KV rows come from different action ids in the same user sequence.
- `distance_bucket`: Interleaved-token distance between the two positions.
- `K/V CKSim`: Mean head-wise cosine similarity of projected K or V vectors. Each head is compared along its feature dimension, then averaged across heads.
- `P10/P90`: 10th/90th percentile; these expose tail behavior hidden by the mean.
- `pair_count`: Number of sampled pairs in this bucket.

Layer 0 is excluded here so the trend reflects contextual HSTU layers rather than raw embedding identity.

| Pair Type | Distance Bucket | Mean Distance | K CKSim | K P10 | K P90 | K Centered CKSim | V CKSim | V P10 | V P90 | Pair Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same action | <= 64 | 26.4 | 0.9908 | 0.9745 | 1.0000 | 0.9903 | 0.9928 | 0.9788 | 1.0000 | 20,139 |
| same action | 65-128 | 88.7 | 0.9551 | 0.8850 | 0.9968 | 0.9522 | 0.9607 | 0.9030 | 0.9945 | 5,628 |
| same action | 129-256 | 174.2 | 0.9146 | 0.8066 | 0.9892 | 0.9057 | 0.9056 | 0.8102 | 0.9813 | 2,359 |
| same action | 257-512 | 281.6 | 0.8180 | 0.6687 | 0.9238 | 0.7996 | 0.7990 | 0.7188 | 0.8872 | 203 |
| different action | <= 64 | 26.2 | 0.5790 | 0.2923 | 0.8304 | 0.5397 | 0.5884 | 0.3117 | 0.8070 | 131,960 |
| different action | 65-128 | 89.6 | 0.4574 | 0.1126 | 0.7515 | 0.4035 | 0.4503 | 0.0945 | 0.7425 | 25,960 |
| different action | 129-256 | 171.3 | 0.3902 | 0.0386 | 0.7124 | 0.3284 | 0.3744 | 0.0582 | 0.6767 | 9,508 |
| different action | 257-512 | 276.8 | 0.4325 | 0.0736 | 0.7380 | 0.3847 | 0.4234 | 0.2314 | 0.6264 | 572 |

| Distance Bucket | K Same-Diff Gap | V Same-Diff Gap | Same Pairs | Different Pairs |
|---|---:|---:|---:|---:|
| <= 64 | +0.4118 | +0.4045 | 20,139 | 131,960 |
| 65-128 | +0.4977 | +0.5104 | 5,628 | 25,960 |
| 129-256 | +0.5243 | +0.5312 | 2,359 | 9,508 |
| 257-512 | +0.3855 | +0.3757 | 203 | 572 |

Insight: action identity is reusable mainly as a local signal: same-action KV is much closer than different-action KV in short-distance buckets, while the distance trend explains why an unbounded global action cache is not the right motivation.


Same-id action/item comparison, using the same distance buckets:

| Same-ID Token Type | Distance Bucket | K CKSim | K Centered CKSim | V CKSim | Pair Count |
|---|---:|---:|---:|---:|---:|
| action | <= 64 | 0.9908 | 0.9903 | 0.9928 | 20,139 |
| action | 65-128 | 0.9551 | 0.9522 | 0.9607 | 5,628 |
| action | 129-256 | 0.9146 | 0.9057 | 0.9056 | 2,359 |
| action | 257-512 | 0.8180 | 0.7996 | 0.7990 | 203 |

Distribution plots:
- `action_kv_same_vs_diff_by_layer.png` (same action, different action, same item, and random different-token baselines)
- `kv_identity_baselines_by_layer_summary.csv`
- `item_action_kv_similarity_boxplot.png`: each group is same-id pairs split by token type and `same_window`. The center line is the median K CKSim, the box is the interquartile range, whiskers show the non-outlier range, and dots are outlier pairs. Higher boxes mean the same id keeps more similar K vectors across positions.
- `item_vs_action_kv_similarity_by_distance.png`: action-only same-id K CKSim by distance bucket, sorted from short to long distance.

## Q3. Which Global Top-K/Window Points Are Pareto Optimal?

This grid answers how top-K and window size trade reuse ratio against final inference AUC.

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name ml-20m --output-dir ./analysis_output/action_reuse_motivation_full_ml-20m_ranking_rerun_20260427_063950 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-policy-grid-auc-analysis --gin-config-file ./training/configs/movielen_ranking.gin --ckpt-load-dir ckpt_ml_20m_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

Variables:

- `window_size`: Interleaved-token window size used by action KV reuse.
- `top_k`: Number of high-frequency action ids reused per user/window/layer.
- `reuse_ratio_all_tokens`: Replaced action KV rows divided by all sequence tokens; higher means more compute/cache saved.
- `mean_reuse_auc`: Mean AUC after KV reuse, filtered to tasks whose baseline AUC is above the configured threshold.
- `mean_auc_diff`: Mean AUC change relative to no-reuse baseline.
- `max_auc_drop`: Worst AUC drop among filtered tasks; lower is safer.
- `policy_pareto`: True if no other point has both higher/equal reuse ratio and higher/equal AUC.

All-task baseline AUC used by this grid:

| Metric | Baseline AUC | Included In Pareto |
|---|---:|---:|
| task0.AUC | 0.811706 | yes |

KV-only no-reuse sanity check:

| Metric | Original Baseline | KV-only No-Reuse | Diff |
|---|---:|---:|---:|
| task0.AUC | 0.811706 | 0.811706 | +0.000000 |

Recommended operating points:

| policy_scope | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| global | 64 | 5 | 39.72% | 0.810831 | -0.000875 | 0.000875 |

Pareto frontier preview:

| policy_scope | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| global | 512 | 5 | 44.38% | 0.808204 | -0.003502 | 0.003502 |
| global | 1024 | 5 | 44.38% | 0.808204 | -0.003502 | 0.003502 |
| global | 256 | 5 | 43.84% | 0.809095 | -0.002611 | 0.002611 |
| global | 128 | 5 | 42.46% | 0.809992 | -0.001714 | 0.001714 |
| global | 128 | 4 | 40.76% | 0.810051 | -0.001655 | 0.001655 |
| global | 64 | 5 | 39.72% | 0.810831 | -0.000875 | 0.000875 |
| global | 64 | 4 | 38.61% | 0.810861 | -0.000845 | 0.000845 |
| global | 64 | 3 | 35.88% | 0.810895 | -0.000811 | 0.000811 |
| global | 64 | 2 | 29.68% | 0.810919 | -0.000787 | 0.000787 |
| global | 64 | 1 | 18.58% | 0.810968 | -0.000738 | 0.000738 |

Full table: `policy_grid_auc_summary.csv`
All-task AUC table: `policy_grid_auc_by_task.csv`
Pareto table: `policy_grid_pareto.csv`
Scatter plot: `policy_grid_auc_reuse_scatter.png`

## Q4. Which Layers Are Most Sensitive?

Each run enables reuse in one HSTU layer only. This run uses top_k=1 and window_size=1024, so the table isolates layer sensitivity without running a full top-K/window grid for every layer.

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name ml-20m --output-dir ./analysis_output/action_reuse_motivation_full_ml-20m_ranking_rerun_20260427_063950 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-layer-policy-auc-analysis --gin-config-file ./training/configs/movielen_ranking.gin --ckpt-load-dir ckpt_ml_20m_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

Variables:

- `window_size`: Interleaved-token window size used by action KV reuse.
- `top_k`: Number of high-frequency action ids reused per user/window/layer.
- `reuse_ratio_all_tokens`: Replaced action KV rows divided by all sequence tokens; higher means more compute/cache saved.
- `mean_reuse_auc`: Mean AUC after KV reuse, filtered to tasks whose baseline AUC is above the configured threshold.
- `mean_auc_diff`: Mean AUC change relative to no-reuse baseline.
- `max_auc_drop`: Worst AUC drop among filtered tasks; lower is safer.
- `policy_pareto`: True if no other point has both higher/equal reuse ratio and higher/equal AUC.

All-task baseline AUC used by this grid:

| Metric | Baseline AUC | Included In Pareto |
|---|---:|---:|
| task0.AUC | 0.811706 | yes |

Recommended operating points:

| layer_idx | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| 0 | 1024 | 1 | 2.24% | 0.811712 | 0.000006 | 0.000000 |
| 1 | 1024 | 1 | 2.24% | 0.811607 | -0.000099 | 0.000099 |
| 2 | 1024 | 1 | 2.24% | 0.811294 | -0.000412 | 0.000412 |
| 3 | 1024 | 1 | 2.24% | 0.811451 | -0.000255 | 0.000255 |
| 4 | 1024 | 1 | 2.24% | 0.811163 | -0.000543 | 0.000543 |
| 5 | 1024 | 1 | 2.24% | 0.811596 | -0.000110 | 0.000110 |
| 6 | 1024 | 1 | 2.24% | 0.811639 | -0.000067 | 0.000067 |
| 7 | 1024 | 1 | 2.24% | 0.811145 | -0.000561 | 0.000561 |

Pareto frontier preview:

| layer_idx | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| 0 | 1024 | 1 | 2.24% | 0.811712 | 0.000006 | 0.000000 |
| 6 | 1024 | 1 | 2.24% | 0.811639 | -0.000067 | 0.000067 |
| 1 | 1024 | 1 | 2.24% | 0.811607 | -0.000099 | 0.000099 |
| 5 | 1024 | 1 | 2.24% | 0.811596 | -0.000110 | 0.000110 |
| 3 | 1024 | 1 | 2.24% | 0.811451 | -0.000255 | 0.000255 |
| 2 | 1024 | 1 | 2.24% | 0.811294 | -0.000412 | 0.000412 |
| 4 | 1024 | 1 | 2.24% | 0.811163 | -0.000543 | 0.000543 |
| 7 | 1024 | 1 | 2.24% | 0.811145 | -0.000561 | 0.000561 |

Full table: `layer_policy_auc_summary.csv`
All-task AUC table: `layer_policy_auc_by_task.csv`
Pareto table: `layer_policy_pareto.csv`

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

## Dataset-Only Reuse Opportunity Reference

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name ml-20m --output-dir ./analysis_output/action_reuse_motivation_full_ml-20m_ranking_rerun_20260427_063950 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-layer-policy-auc-analysis --gin-config-file ./training/configs/movielen_ranking.gin --ckpt-load-dir ckpt_ml_20m_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

| Signal | Value |
|---|---:|
| Mean user top-1 action share | 38.20% |
| Mean user top-3 action share | 80.23% |
| Window=512, topK=3 action coverage | 77.50% |
| Window=512, topK=3 candidate reuse rate | 74.91% |

Insight: action tokens are concentrated enough that a small local topK policy can cover most reuse candidates.

