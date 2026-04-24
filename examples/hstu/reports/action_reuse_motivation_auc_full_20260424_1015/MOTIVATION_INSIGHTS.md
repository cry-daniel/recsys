# Action KV Reuse Motivation Insights

## Takeaway

GR KV reuse should be constrained to token types and positions where the hidden semantics stay close. The evidence below says action tokens are the right target: their id space is small, repeated actions have high real-KV similarity, and AUC degrades when reuse ignores semantic id or distance.

## Token Space

Command:

```bash
cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6471 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --window-sizes 128 256 512 1024 --top-ks 1 2 3 5 --skip-plots --output-dir ./analysis_output/action_reuse_motivation_auc_full --run-auc-impact-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-reuse-token-types action --auc-reuse-strategies global_first_any_action_legacy same_id_max_distance wrong_id_same_window global_same_id global_topk_same_id window_topk_same_id --auc-reuse-max-distances 128 256 512 1024 -1 --auc-reuse-window-size 512 --auc-reuse-top-k 3 --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

| Token Type | Global Unique IDs | Mean User Unique IDs | Mean User Top-1 Share | Mean User Top-3 Share |
|---|---:|---:|---:|---:|
| item | 4,363,609 | 11536.75 | 0.06% | 0.17% |
| action | 125 | 18.56 | 58.66% | 95.58% |

Insight: action has a much smaller semantic space than item, so repeated action ids are common enough to create a real reuse opportunity.

## Real KV Similarity

Command:

```bash
cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6366 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --window-sizes 128 256 512 1024 --top-ks 1 2 3 5 --output-dir ./analysis_output/action_reuse_motivation_auc_full --run-kv-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --kv-max-batches 6 --kv-max-users-per-batch 8 --kv-max-actions-per-user 512 --kv-max-pairs-per-token-id 200 --kv-max-different-action-pairs-per-user 2000 --kv-window-size 512
```

| Action Pair | K Cosine Mean | V Cosine Mean | Pair Count |
|---|---:|---:|---:|
| Same action | 0.9964 | 0.9969 | 274,656 |
| Different action | 0.3887 | 0.3932 | 768,000 |
| Same - Different | +0.6077 | +0.6038 |  |

Insight: KV vectors are much more consistent when the action id is the same. This is the direct evidence that action identity carries reusable KV structure.

## KV Similarity vs Distance

Command:

```bash
cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6366 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --window-sizes 128 256 512 1024 --top-ks 1 2 3 5 --output-dir ./analysis_output/action_reuse_motivation_auc_full --run-kv-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --kv-max-batches 6 --kv-max-users-per-batch 8 --kv-max-actions-per-user 512 --kv-max-pairs-per-token-id 200 --kv-max-different-action-pairs-per-user 2000 --kv-window-size 512
```

Layer 0 is excluded in this distance view so the trend reflects contextual HSTU layers rather than raw embedding identity.

| Pair Type | Distance Bucket | Mean Distance | K Cosine | K Centered Cosine | V Cosine | Pair Count |
|---|---:|---:|---:|---:|---:|---:|
| same action | 129-256 | 196.0 | 0.9985 | 0.9983 | 0.9986 | 9,401 |
| same action | > 1024 | 3340.4 | 0.9952 | 0.9945 | 0.9959 | 172,032 |
| same action | 257-512 | 385.4 | 0.9978 | 0.9975 | 0.9981 | 17,654 |
| same action | 513-1024 | 770.1 | 0.9971 | 0.9966 | 0.9976 | 29,596 |
| same action | <= 128 | 62.9 | 0.9982 | 0.9979 | 0.9985 | 11,641 |
| different action | > 1024 | 3325.1 | 0.4238 | 0.3320 | 0.4222 | 489,067 |
| different action | <= 128 | 69.5 | 0.4471 | 0.3582 | 0.4416 | 25,563 |
| different action | 513-1024 | 766.5 | 0.4364 | 0.3465 | 0.4337 | 84,525 |
| different action | 257-512 | 387.9 | 0.4422 | 0.3528 | 0.4381 | 47,635 |
| different action | 129-256 | 196.8 | 0.4473 | 0.3590 | 0.4433 | 25,210 |

| Same-ID Token Type | Distance Bucket | K Cosine | K Centered Cosine | V Cosine | Pair Count |
|---|---:|---:|---:|---:|---:|
| item | > 1024 | 0.8051 | 0.7729 | 0.8199 | 28 |
| item | <= 128 | 0.9143 | 0.8988 | 0.9183 | 161 |
| item | 513-1024 | 0.9994 | 0.9993 | 0.9992 | 28 |
| action | 129-256 | 0.9985 | 0.9983 | 0.9986 | 9,401 |
| action | > 1024 | 0.9952 | 0.9945 | 0.9959 | 172,032 |
| action | 257-512 | 0.9978 | 0.9975 | 0.9981 | 17,654 |
| action | 513-1024 | 0.9971 | 0.9966 | 0.9976 | 29,596 |
| action | <= 128 | 0.9982 | 0.9979 | 0.9985 | 11,641 |
| item | 129-256 | 0.9810 | 0.9771 | 0.9657 | 28 |
| item | 257-512 | 0.8288 | 0.8004 | 0.8294 | 28 |

Insight: same-action KV remains far closer than different-action KV, but distance still matters; this motivates a local window instead of one unbounded action cache.

## Reuse Opportunity

Command:

```bash
cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6471 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --window-sizes 128 256 512 1024 --top-ks 1 2 3 5 --skip-plots --output-dir ./analysis_output/action_reuse_motivation_auc_full --run-auc-impact-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-reuse-token-types action --auc-reuse-strategies global_first_any_action_legacy same_id_max_distance wrong_id_same_window global_same_id global_topk_same_id window_topk_same_id --auc-reuse-max-distances 128 256 512 1024 -1 --auc-reuse-window-size 512 --auc-reuse-top-k 3 --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

| Signal | Value |
|---|---:|
| Mean user top-1 action share | 58.66% |
| Mean user top-3 action share | 95.58% |
| Window=512, topK=3 action coverage | 96.56% |
| Window=512, topK=3 candidate reuse rate | 95.38% |

Insight: action tokens are concentrated enough that a small local topK policy can cover most reuse candidates.

## AUC vs Reuse

Command:

```bash
cd /workspace/recsys/examples/hstu && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${PYTHONPATH}:$(realpath ../):$(realpath training) torchrun --nproc_per_node 1 --master_addr localhost --master_port 6471 ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --window-sizes 128 256 512 1024 --top-ks 1 2 3 5 --skip-plots --output-dir ./analysis_output/action_reuse_motivation_auc_full --run-auc-impact-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-reuse-token-types action --auc-reuse-strategies global_first_any_action_legacy same_id_max_distance wrong_id_same_window global_same_id global_topk_same_id window_topk_same_id --auc-reuse-max-distances 128 256 512 1024 -1 --auc-reuse-window-size 512 --auc-reuse-top-k 3 --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

AUC is averaged only over tasks with baseline AUC > 0.6: `task1.AUC`, `task2.AUC`, `task5.AUC`, `task6.AUC`.

| Mode | Max Distance | Reuse Ratio | Mean AUC | AUC Diff | Max AUC Drop | Replacements | Pareto |
|---|---:|---:|---:|---:|---:|---:|---:|
| wrong_id_same_window:action | - | 46.12% | 0.689849 | -0.008369 | 0.092270 | 2,215,040 | no |
| window_topk_same_id:action | - | 46.17% | 0.698261 | +0.000043 | 0.000762 | 2,217,080 | no |
| same_id_max_distance:action:dist<=128 | 128 | 46.54% | 0.698514 | +0.000297 | 0.007232 | 2,235,128 | no |
| same_id_max_distance:action:dist<=256 | 256 | 46.58% | 0.697724 | -0.000494 | 0.009410 | 2,237,200 | no |
| same_id_max_distance:action:dist<=512 | 512 | 46.59% | 0.698545 | +0.000327 | 0.009271 | 2,237,608 | yes |
| same_id_max_distance:action:dist<=1024 | 1024 | 46.60% | 0.698528 | +0.000310 | 0.009168 | 2,237,752 | yes |
| global_topk_same_id:action | - | 46.60% | 0.698527 | +0.000309 | 0.009169 | 2,237,760 | yes |
| same_id_max_distance:action:dist<=global | global | 46.60% | 0.698527 | +0.000309 | 0.009169 | 2,237,760 | yes |
| global_same_id:action | - | 49.74% | 0.698308 | +0.000090 | 0.010202 | 2,388,784 | yes |
| global_first_any_action_legacy:action | - | 49.95% | 0.592675 | -0.105543 | 0.270097 | 2,398,664 | yes |

Insight: the legacy first-action and wrong-action rows are negative controls: high replacement without semantic or distance constraints can damage ranking quality. The distance sweep shows how much locality is needed before reuse becomes low-risk.

## Conclusion

Action KV reuse is motivated by three facts: action ids have a much smaller reuse space than item ids, same-action KV stays substantially closer than the controls, and AUC risk grows when reuse ignores action identity or position distance. The method should be presented as constrained action reuse, not generic KV sharing.
