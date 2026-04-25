# Action KV Reuse Motivation Report

This report is organized around the five questions needed to motivate and tune action KV reuse.

Definitions used throughout: AUC summaries only include tasks whose baseline AUC is above the configured threshold; Pareto optimality is computed in the reuse-ratio vs AUC plane.

## Q1. Why Is Action Better Than Item For Reuse?

Command:

```bash
docker exec gr_training bash -lc 'cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6551 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 64 --layer-top-ks 1 --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-filter-baseline-threshold 0.6'
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
docker exec gr_training bash -lc 'cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6550 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 64 --layer-top-ks 1 --run-kv-analysis --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --kv-distance-buckets 64 128 256 512 1024 --kv-max-pairs-per-distance-bucket 5000 --max-raw-kv-pair-rows 1000 --auc-kv-replace-implementation kv_only --auc-filter-baseline-threshold 0.6'
```

Variables:

- `same action`: Both KV rows come from the same action id in the same user sequence.
- `different action`: The two KV rows come from different action ids in the same user sequence.
- `distance_bucket`: Interleaved-token distance between the two positions.
- `K/V Cosine`: Mean cosine similarity of projected K or V vectors.
- `P10/P90`: 10th/90th percentile; these expose tail behavior hidden by the mean.
- `pair_count`: Number of sampled pairs in this bucket.

Layer 0 is excluded here so the trend reflects contextual HSTU layers rather than raw embedding identity.

| Pair Type | Distance Bucket | Mean Distance | K Cosine | K P10 | K P90 | K Centered | V Cosine | V P10 | V P90 | Pair Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| same action | 65-128 | 118.0 | 0.9999 | 0.9998 | 1.0000 | 0.9998 | 0.9998 | 0.9998 | 1.0000 | 504 |
| same action | 129-256 | 143.0 | 0.9999 | 1.0000 | 1.0000 | 0.9999 | 0.9999 | 1.0000 | 1.0000 | 2,128 |
| same action | 257-512 | 332.7 | 0.9992 | 0.9999 | 1.0000 | 0.9990 | 0.9990 | 0.9999 | 1.0000 | 4,543 |
| same action | 513-1024 | 722.8 | 0.9997 | 0.9999 | 1.0000 | 0.9997 | 0.9997 | 0.9999 | 1.0000 | 8,988 |
| same action | > 1024 | 3355.0 | 0.9981 | 0.9994 | 1.0000 | 0.9979 | 0.9980 | 0.9992 | 1.0000 | 58,576 |
| different action | 65-128 | 118.5 | 0.5448 | -0.0093 | 0.9027 | 0.4721 | 0.3736 | 0.0311 | 0.7911 | 1,336 |
| different action | 129-256 | 147.7 | 0.5432 | 0.0351 | 0.8863 | 0.4697 | 0.3462 | 0.0704 | 0.6617 | 4,843 |
| different action | 257-512 | 335.8 | 0.5357 | 0.0265 | 0.8867 | 0.4612 | 0.3410 | 0.0642 | 0.6195 | 10,349 |
| different action | 513-1024 | 726.9 | 0.5233 | 0.0199 | 0.8873 | 0.4472 | 0.3386 | 0.0564 | 0.6484 | 20,408 |
| different action | > 1024 | 3394.9 | 0.5327 | 0.0221 | 0.8874 | 0.4580 | 0.3407 | 0.0601 | 0.6425 | 131,064 |

| Distance Bucket | K Same-Diff Gap | V Same-Diff Gap | Same Pairs | Different Pairs |
|---|---:|---:|---:|---:|
| 65-128 | +0.4551 | +0.6262 | 504 | 1,336 |
| 129-256 | +0.4567 | +0.6537 | 2,128 | 4,843 |
| 257-512 | +0.4635 | +0.6580 | 4,543 | 10,349 |
| 513-1024 | +0.4764 | +0.6612 | 8,988 | 20,408 |
| > 1024 | +0.4654 | +0.6573 | 58,576 | 131,064 |

Insight: action identity is reusable mainly as a local signal: same-action KV is much closer than different-action KV in short-distance buckets, while the distance trend explains why an unbounded global action cache is not the right motivation.


Same-id action/item comparison, using the same distance buckets:

| Same-ID Token Type | Distance Bucket | K Cosine | K Centered | V Cosine | Pair Count |
|---|---:|---:|---:|---:|---:|
| action | 65-128 | 0.9999 | 0.9998 | 0.9998 | 504 |
| action | 129-256 | 0.9999 | 0.9999 | 0.9999 | 2,128 |
| action | 257-512 | 0.9992 | 0.9990 | 0.9990 | 4,543 |
| action | 513-1024 | 0.9997 | 0.9997 | 0.9997 | 8,988 |
| action | > 1024 | 0.9981 | 0.9979 | 0.9980 | 58,576 |

Distribution plots:
- `action_kv_same_vs_diff_by_layer.png`
- `item_action_kv_similarity_boxplot.png`
- `item_vs_action_kv_similarity_by_distance.png`

## Q3. Which Global Top-K/Window Points Are Pareto Optimal?

This grid answers how top-K and window size trade reuse ratio against final inference AUC.

Command:

```bash
docker exec gr_training bash -lc 'cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6551 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 64 --layer-top-ks 1 --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-filter-baseline-threshold 0.6'
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
| task0.AUC | 0.503938 | no |
| task1.AUC | 0.641586 | yes |
| task2.AUC | 0.609370 | yes |
| task3.AUC | 0.585510 | no |
| task4.AUC | 0.546522 | no |
| task5.AUC | 0.512018 | no |
| task6.AUC | 0.614389 | yes |
| task7.AUC | 0.521716 | no |

KV-only no-reuse sanity check:

| Metric | Original Baseline | KV-only No-Reuse | Diff |
|---|---:|---:|---:|
| task0.AUC | 0.503938 | 0.503938 | +0.000000 |
| task1.AUC | 0.641586 | 0.641586 | +0.000000 |
| task2.AUC | 0.609370 | 0.609370 | +0.000000 |
| task3.AUC | 0.585510 | 0.585510 | +0.000000 |
| task4.AUC | 0.546522 | 0.546522 | +0.000000 |
| task5.AUC | 0.512018 | 0.512018 | +0.000000 |
| task6.AUC | 0.614389 | 0.614389 | +0.000000 |
| task7.AUC | 0.521716 | 0.521716 | +0.000000 |

Recommended operating points:

| policy_scope | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| global | 1024 | 5 | 48.75% | 0.621434 | -0.000348 | 0.000954 |

Pareto frontier preview:

| policy_scope | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| global | 1024 | 5 | 48.75% | 0.621434 | -0.000348 | 0.000954 |
| global | 512 | 5 | 48.38% | 0.621692 | -0.000090 | 0.000484 |
| global | 512 | 4 | 48.03% | 0.621793 | 0.000011 | 0.000199 |
| global | 512 | 3 | 47.06% | 0.621848 | 0.000067 | 0.000089 |
| global | 256 | 3 | 46.53% | 0.621852 | 0.000070 | 0.000021 |

Full table: `policy_grid_auc_summary.csv`
All-task AUC table: `policy_grid_auc_by_task.csv`
Pareto table: `policy_grid_pareto.csv`
Scatter plot: `policy_grid_auc_reuse_scatter.png`

## Q4. Which Layers Are Most Sensitive?

Each run enables reuse in one HSTU layer only. By default this uses top_k=1 and window_size=64, so the table isolates layer sensitivity without running a full top-K/window grid for every layer.

Command:

```bash
docker exec gr_training bash -lc 'cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6551 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 64 --layer-top-ks 1 --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-filter-baseline-threshold 0.6'
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
| task0.AUC | 0.503938 | no |
| task1.AUC | 0.641586 | yes |
| task2.AUC | 0.609370 | yes |
| task3.AUC | 0.585510 | no |
| task4.AUC | 0.546522 | no |
| task5.AUC | 0.512018 | no |
| task6.AUC | 0.614389 | yes |
| task7.AUC | 0.521716 | no |

Recommended operating points:

| layer_idx | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| 0 | 64 | 1 | 3.73% | 0.621581 | -0.000200 | 0.000884 |
| 1 | 64 | 1 | 3.73% | 0.621738 | -0.000043 | 0.000078 |
| 2 | 64 | 1 | 3.73% | 0.621699 | -0.000082 | 0.000252 |
| 3 | 64 | 1 | 3.73% | 0.621847 | 0.000065 | 0.000033 |
| 4 | 64 | 1 | 3.73% | 0.621875 | 0.000093 | 0.000005 |
| 5 | 64 | 1 | 3.73% | 0.621765 | -0.000017 | 0.000044 |
| 6 | 64 | 1 | 3.73% | 0.621783 | 0.000002 | 0.000030 |
| 7 | 64 | 1 | 3.73% | 0.621817 | 0.000036 | 0.000002 |

Pareto frontier preview:

| layer_idx | window_size | top_k | reuse_ratio_all_tokens | mean_reuse_auc | mean_auc_diff | max_auc_drop |
|---|---|---|---|---|---|---|
| 4 | 64 | 1 | 3.73% | 0.621875 | 0.000093 | 0.000005 |
| 3 | 64 | 1 | 3.73% | 0.621847 | 0.000065 | 0.000033 |
| 7 | 64 | 1 | 3.73% | 0.621817 | 0.000036 | 0.000002 |
| 6 | 64 | 1 | 3.73% | 0.621783 | 0.000002 | 0.000030 |
| 5 | 64 | 1 | 3.73% | 0.621765 | -0.000017 | 0.000044 |
| 1 | 64 | 1 | 3.73% | 0.621738 | -0.000043 | 0.000078 |
| 2 | 64 | 1 | 3.73% | 0.621699 | -0.000082 | 0.000252 |
| 0 | 64 | 1 | 3.73% | 0.621581 | -0.000200 | 0.000884 |

Full table: `layer_policy_auc_summary.csv`
All-task AUC table: `layer_policy_auc_by_task.csv`
Pareto table: `layer_policy_pareto.csv`
Scatter plot: `layer_policy_auc_reuse_scatter.png`

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
docker exec gr_training bash -lc 'cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6551 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 64 --layer-top-ks 1 --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-filter-baseline-threshold 0.6'
```

| Signal | Value |
|---|---:|
| Mean user top-1 action share | 58.66% |
| Mean user top-3 action share | 95.58% |
| Window=512, topK=3 action coverage | 96.56% |
| Window=512, topK=3 candidate reuse rate | 95.38% |

Insight: action tokens are concentrated enough that a small local topK policy can cover most reuse candidates.

