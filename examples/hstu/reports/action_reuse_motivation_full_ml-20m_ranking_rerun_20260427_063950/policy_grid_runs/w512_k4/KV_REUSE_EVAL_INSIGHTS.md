# Action KV Reuse Evaluation Insights

- Filtered AUC tasks: task0.AUC
- AUC threshold: baseline > 0.6
- Implementation: kv_only

## Reuse Ratio vs AUC

Command:

```bash
/usr/bin/python ./training/analyze_action_reuse_motivation.py --dataset-name ml-20m --output-dir ./analysis_output/action_reuse_motivation_full_ml-20m_ranking_rerun_20260427_063950 --max-window-rows 5000 --window-sizes 64 128 256 512 1024 --top-ks 1 2 3 4 5 --layer-window-sizes 1024 --layer-top-ks 1 --run-policy-grid-auc-analysis --gin-config-file ./training/configs/movielen_ranking.gin --ckpt-load-dir ckpt_ml_20m_ranking/iter1000 --auc-kv-replace-implementation kv_only --auc-max-eval-iters 10 --auc-filter-baseline-threshold 0.6
```

| Mode | Distance | Reuse Ratio | Mean AUC | Mean Diff | Max Drop | Replacements | Pareto |
|---|---:|---:|---:|---:|---:|---:|---|
| window_topk_same_id:action | - | 42.15% | 0.808313 | -0.003393 | 0.003393 | 837,824 | yes |

## Takeaways

- Aggressive or semantically wrong reuse is the negative control: `window_topk_same_id:action` reaches 42.15% reuse but has max AUC drop 0.003393.
- Constrained same-action reuse is the useful regime: `window_topk_same_id:action` reaches 42.15% reuse with mean AUC diff -0.003393.
- The evaluation is now KV-only: source rows are copied after UVQK projection, so Q/U remain position-specific.
