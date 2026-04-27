# Action KV Reuse Evaluation Insights

- Filtered AUC tasks: task1.AUC, task2.AUC, task5.AUC, task6.AUC
- AUC threshold: baseline > 0.6
- Implementation: kv_only

## Reuse Ratio vs AUC

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name kuairand-1k --output-dir ./analysis_output/action_reuse_motivation_full_kuairand-1k_ranking_rerun_20260427_055210 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-policy-grid-auc-analysis --run-layer-policy-auc-analysis --gin-config-file ./training/configs/kuairand_1k_ranking.gin --ckpt-load-dir ckpt_kr_1k_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

| Mode | Distance | Reuse Ratio | Mean AUC | Mean Diff | Max Drop | Replacements | Pareto |
|---|---:|---:|---:|---:|---:|---:|---|
| window_topk_same_id:action | - | 43.53% | 0.698098 | -0.000120 | 0.000404 | 2,090,272 | yes |

## Takeaways

- Aggressive or semantically wrong reuse is the negative control: `window_topk_same_id:action` reaches 43.53% reuse but has max AUC drop 0.000404.
- Constrained same-action reuse is the useful regime: `window_topk_same_id:action` reaches 43.53% reuse with mean AUC diff -0.000120.
- The evaluation is now KV-only: source rows are copied after UVQK projection, so Q/U remain position-specific.
