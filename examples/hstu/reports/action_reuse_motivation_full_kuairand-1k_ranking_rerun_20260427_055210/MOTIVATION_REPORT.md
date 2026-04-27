# Action KV Reuse Motivation Report

This report is organized around the five questions needed to motivate and tune action KV reuse.

Definitions used throughout: AUC summaries only include tasks whose baseline AUC is above the configured threshold; Pareto optimality is computed in the reuse-ratio vs AUC plane.

## Q1. Why Is Action Better Than Item For Reuse?

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full_kuairand-1k_ranking_rerun_20260427_055210 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-kv-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --kv-distance-buckets 64 128 256 512 1024 --kv-max-pairs-per-distance-bucket 5000 --max-raw-kv-pair-rows 1000 --auc-kv-replace-implementation kv_only --auc-filter-baseline-threshold 0.6
```

Variables:

- `global_unique_ids`: Number of distinct ids in the whole analyzed dataset.
- `mean_user_unique_ids`: Average number of distinct ids per user sequence.
- `mean_user_top1_share`: Average fraction of a user's tokens covered by their most frequent id.
- `mean_user_top3_share`: Average fraction covered by their three most frequent ids.

| Token Type | Global Unique IDs | Mean User Unique IDs | Mean User Top-1 Share | Mean User Top-3 Share |
|---|---:|---:|---:|---:|
| item | 4,363,609 | 11536.75 | 0.06% | 0.17% |
| action | 125 | 18.56 | 58.66% | 95.58% |

Insight: action has a much smaller semantic space than item, so repeated action ids are common enough to create a real reuse opportunity.

## Q2. How Similar Is Action KV At Different Distances?

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full_kuairand-1k_ranking_rerun_20260427_055210 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-kv-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --kv-distance-buckets 64 128 256 512 1024 --kv-max-pairs-per-distance-bucket 5000 --max-raw-kv-pair-rows 1000 --auc-kv-replace-implementation kv_only --auc-filter-baseline-threshold 0.6
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
| same action | 65-128 | 118.0 | 0.9997 | 0.9996 | 1.0000 | 0.9996 | 0.9998 | 0.9998 | 1.0000 | 504 |
| same action | 129-256 | 143.0 | 0.9998 | 0.9999 | 1.0000 | 0.9998 | 0.9999 | 0.9999 | 1.0000 | 2,128 |
| same action | 257-512 | 332.7 | 0.9990 | 0.9999 | 1.0000 | 0.9989 | 0.9991 | 0.9999 | 1.0000 | 4,543 |
| same action | 513-1024 | 722.8 | 0.9995 | 0.9998 | 1.0000 | 0.9995 | 0.9996 | 0.9999 | 1.0000 | 8,988 |
| same action | > 1024 | 3355.0 | 0.9974 | 0.9983 | 1.0000 | 0.9973 | 0.9979 | 0.9991 | 1.0000 | 58,576 |
| different action | 65-128 | 118.5 | 0.3974 | -0.0562 | 0.7740 | 0.3415 | 0.3661 | 0.0009 | 0.7473 | 1,336 |
| different action | 129-256 | 147.7 | 0.3831 | -0.0236 | 0.7610 | 0.3278 | 0.3404 | 0.0753 | 0.6183 | 4,843 |
| different action | 257-512 | 335.8 | 0.3806 | -0.0248 | 0.7618 | 0.3249 | 0.3342 | 0.0555 | 0.6117 | 10,349 |
| different action | 513-1024 | 726.9 | 0.3708 | -0.0274 | 0.7606 | 0.3132 | 0.3311 | 0.0472 | 0.6178 | 20,408 |
| different action | > 1024 | 3394.9 | 0.3757 | -0.0260 | 0.7621 | 0.3191 | 0.3335 | 0.0491 | 0.6180 | 131,064 |

| Distance Bucket | K Same-Diff Gap | V Same-Diff Gap | Same Pairs | Different Pairs |
|---|---:|---:|---:|---:|
| 65-128 | +0.6023 | +0.6336 | 504 | 1,336 |
| 129-256 | +0.6167 | +0.6594 | 2,128 | 4,843 |
| 257-512 | +0.6183 | +0.6649 | 4,543 | 10,349 |
| 513-1024 | +0.6287 | +0.6685 | 8,988 | 20,408 |
| > 1024 | +0.6217 | +0.6645 | 58,576 | 131,064 |

Insight: action identity is reusable mainly as a local signal: same-action KV is much closer than different-action KV in short-distance buckets, while the distance trend explains why an unbounded global action cache is not the right motivation.


Same-id action/item comparison, using the same distance buckets:

| Same-ID Token Type | Distance Bucket | K CKSim | K Centered CKSim | V CKSim | Pair Count |
|---|---:|---:|---:|---:|---:|
| action | 65-128 | 0.9997 | 0.9996 | 0.9998 | 504 |
| action | 129-256 | 0.9998 | 0.9998 | 0.9999 | 2,128 |
| action | 257-512 | 0.9990 | 0.9989 | 0.9991 | 4,543 |
| action | 513-1024 | 0.9995 | 0.9995 | 0.9996 | 8,988 |
| action | > 1024 | 0.9974 | 0.9973 | 0.9979 | 58,576 |
| item | <= 64 | 0.8535 | 0.8386 | 0.8576 | 301 |
| item | 65-128 | 0.8390 | 0.8252 | 0.8650 | 14 |
| item | 129-256 | 0.8800 | 0.8688 | 0.8778 | 70 |
| item | 257-512 | 0.9488 | 0.9480 | 0.9604 | 14 |
| item | 513-1024 | 0.8696 | 0.8595 | 0.8840 | 21 |
| item | > 1024 | 0.9972 | 0.9971 | 0.9984 | 28 |

Distribution plots:
- `action_kv_same_vs_diff_by_layer.png` (same action, different action, same item, and random different-token baselines)
- `kv_identity_baselines_by_layer_summary.csv`
- `item_action_kv_similarity_boxplot.png`: each group is same-id pairs split by token type and `same_window`. The center line is the median K CKSim, the box is the interquartile range, whiskers show the non-outlier range, and dots are outlier pairs. Higher boxes mean the same id keeps more similar K vectors across positions.
- `item_vs_action_kv_similarity_by_distance.png`: action-only same-id K CKSim by distance bucket, sorted from short to long distance.

## Q3. Which Global Top-K/Window Points Are Pareto Optimal?

This grid answers how top-K and window size trade reuse ratio against final inference AUC.

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full_kuairand-1k_ranking_rerun_20260427_055210 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
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
| task0.AUC | 0.496216 | no |
| task1.AUC | 0.704653 | yes |
| task2.AUC | 0.648116 | yes |
| task3.AUC | 0.542228 | no |
| task4.AUC | 0.423991 | no |
| task5.AUC | 0.824978 | yes |
| task6.AUC | 0.615125 | yes |
| task7.AUC | 0.533280 | no |

KV-only no-reuse sanity check:

| Metric | Original Baseline | KV-only No-Reuse | Diff |
|---|---:|---:|---:|
| task0.AUC | 0.496216 | 0.496216 | +0.000000 |
| task1.AUC | 0.704653 | 0.704653 | +0.000000 |
| task2.AUC | 0.648116 | 0.648116 | +0.000000 |
| task3.AUC | 0.542228 | 0.542228 | +0.000000 |
| task4.AUC | 0.423991 | 0.423991 | +0.000000 |
| task5.AUC | 0.824978 | 0.824978 | +0.000000 |
| task6.AUC | 0.615125 | 0.615125 | +0.000000 |
| task7.AUC | 0.533280 | 0.533280 | +0.000000 |

Recommended operating points:

| policy_scope | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| global | 1024 | 4 | 47.91% | 0.698108 | -0.000110 | 0.000881 |

Pareto frontier preview:

| policy_scope | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| global | 1024 | 5 | 48.50% | 0.698036 | -0.000182 | 0.001080 |
| global | 512 | 5 | 48.12% | 0.698152 | -0.000066 | 0.001049 |
| global | 512 | 4 | 47.57% | 0.698257 | 0.000039 | 0.000942 |
| global | 256 | 5 | 47.38% | 0.698368 | 0.000150 | 0.000466 |
| global | 256 | 3 | 45.66% | 0.698450 | 0.000232 | 0.000188 |
| global | 256 | 1 | 29.09% | 0.698508 | 0.000290 | 0.000000 |

Full table: `policy_grid_auc_summary.csv`
All-task AUC table: `policy_grid_auc_by_task.csv`
Pareto table: `policy_grid_pareto.csv`
Scatter plot: `policy_grid_auc_reuse_scatter.png`

## Q4. Which Layers Are Most Sensitive?

Each run enables reuse in one HSTU layer only. By default this uses top_k=1 and window_size=64, so the table isolates layer sensitivity without running a full top-K/window grid for every layer.

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full_kuairand-1k_ranking_rerun_20260427_055210 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
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
| task0.AUC | 0.496216 | no |
| task1.AUC | 0.704653 | yes |
| task2.AUC | 0.648116 | yes |
| task3.AUC | 0.542228 | no |
| task4.AUC | 0.423991 | no |
| task5.AUC | 0.824978 | yes |
| task6.AUC | 0.615125 | yes |
| task7.AUC | 0.533280 | no |

Recommended operating points:

| layer_idx | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| 0 | 1024 | 1 | 3.61% | 0.698306 | 0.000088 | 0.000441 |
| 1 | 1024 | 1 | 3.61% | 0.698226 | 0.000008 | 0.000188 |
| 2 | 1024 | 1 | 3.61% | 0.698231 | 0.000013 | 0.000048 |
| 3 | 1024 | 1 | 3.61% | 0.698100 | -0.000118 | 0.000286 |
| 4 | 1024 | 1 | 3.61% | 0.698514 | 0.000296 | 0.000047 |
| 5 | 1024 | 1 | 3.61% | 0.698128 | -0.000090 | 0.000283 |
| 6 | 1024 | 1 | 3.61% | 0.698442 | 0.000224 | 0.000055 |
| 7 | 1024 | 1 | 3.61% | 0.698325 | 0.000107 | 0.000000 |

Pareto frontier preview:

| layer_idx | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| 4 | 1024 | 1 | 3.61% | 0.698514 | 0.000296 | 0.000047 |
| 6 | 1024 | 1 | 3.61% | 0.698442 | 0.000224 | 0.000055 |
| 7 | 1024 | 1 | 3.61% | 0.698325 | 0.000107 | 0.000000 |
| 0 | 1024 | 1 | 3.61% | 0.698306 | 0.000088 | 0.000441 |
| 2 | 1024 | 1 | 3.61% | 0.698231 | 0.000013 | 0.000048 |
| 1 | 1024 | 1 | 3.61% | 0.698226 | 0.000008 | 0.000188 |
| 5 | 1024 | 1 | 3.61% | 0.698128 | -0.000090 | 0.000283 |
| 3 | 1024 | 1 | 3.61% | 0.698100 | -0.000118 | 0.000286 |

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
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full_kuairand-1k_ranking_rerun_20260427_055210 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-kv-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --kv-distance-buckets 64 128 256 512 1024 --kv-max-pairs-per-distance-bucket 5000 --max-raw-kv-pair-rows 1000 --auc-kv-replace-implementation kv_only --auc-filter-baseline-threshold 0.6
```

| Signal | Value |
|---|---:|
| Mean user top-1 action share | 58.66% |
| Mean user top-3 action share | 95.58% |
| Window=512, topK=3 action coverage | 96.56% |
| Window=512, topK=3 candidate reuse rate | 95.38% |

Insight: action tokens are concentrated enough that a small local topK policy can cover most reuse candidates.

