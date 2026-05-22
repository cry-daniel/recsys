# SOW 第二阶段交付文档：Action KV Reuse 实验复现与评估

本文档对应 SOW 第二阶段交付，交付目标是在 HSTU ranking 模型上完成 Action KV Reuse 的可复现实验闭环，包括数据预处理、模型训练、Action Reuse Off/On 评估、AUC 与 NKV 指标产出，以及 window size / Top-K 策略网格搜索。

## 1. 交付摘要

| 项目 | 内容 |
| --- | --- |
| 交付主题 | Action KV Cache Reuse 在 HSTU ranking 模型上的复现与评估 |
| 代码入口 | `examples/hstu` |
| 覆盖数据集 | KuaiRand-1k、MovieLens-20M |
| 覆盖模型 | HSTU-small、HSTU-middle、HSTU-large |
| 核心指标 | AUC、NKV、reuse ratio、replacement count、layer-wise reuse stats |
| 核心策略 | `window_topk_same_id`，token type 为 `action` |
| 交付形式 | 可运行命令、配置文件、评估脚本、CSV/Markdown/PNG 分析产物 |

## 2. 交付范围

本阶段包含：

- 对 KuaiRand-1k 和 MovieLens-20M 进行 HSTU ranking 训练所需的数据预处理。
- 基于 6 个 gin 配置训练 ranking checkpoint。
- 使用原始 checkpoint evaluation 作为 Action Reuse Off baseline。
- 使用 Action KV replacement evaluation 作为 Action Reuse On 结果。
- 产出 AUC、NKV、reuse ratio、layer stats、task-level AUC impact 等验收文件。
- 对 window size 与 Top-K 参数执行网格搜索，并输出帕累托点与可视化结果。

## 3. 交付产物清单

| 交付物 | 路径或生成位置 | 验收说明 |
| --- | --- | --- |
| 数据预处理脚本 | `examples/hstu/preprocessor.py` | 能生成 ranking 训练所需 `processed_seqs.csv` |
| 模型训练脚本 | `examples/hstu/training/pretrain_gr_ranking.py` | 能基于 gin 配置训练 HSTU ranking checkpoint |
| Reuse Off 评估脚本 | `examples/hstu/training/eval_checkpoint.py` | 输出 baseline AUC CSV |
| Reuse On 评估脚本 | `examples/hstu/training/eval_checkpoint_action_kv_replace.py` | 输出 KV reuse 后 AUC、NKV 相关统计和分析报告 |
| 策略分析脚本 | `examples/hstu/training/analyze_action_reuse_motivation.py` | 输出 window/top-k 网格搜索、帕累托点和可视化 |
| 模型配置 | `examples/hstu/training/configs/model_configs/*.gin` | 覆盖 2 个数据集 x 3 个模型规模 |
| 已归档分析报告 | `examples/hstu/reports/action_reuse_motivation_full_*` | 包含 `summary.json`、`MOTIVATION_REPORT.md`、`policy_grid_auc_summary.csv` 等 |

## 4. 模型与配置矩阵

所有配置默认训练 1000 iter，并在 `iter1000` 生成 checkpoint。

| 数据集 | 模型规模 | 配置文件 | checkpoint 目录 |
| --- | --- | --- | --- |
| KuaiRand-1k | HSTU-small | `training/configs/model_configs/kuairand_1k_ranking_hstu_small.gin` | `ckpt_kr_1k_ranking_hstu_small/iter1000` |
| KuaiRand-1k | HSTU-middle | `training/configs/model_configs/kuairand_1k_ranking_hstu_middle.gin` | `ckpt_kr_1k_ranking_hstu_middle/iter1000` |
| KuaiRand-1k | HSTU-large | `training/configs/model_configs/kuairand_1k_ranking_hstu_large.gin` | `ckpt_kr_1k_ranking_hstu_large/iter1000` |
| MovieLens-20M | HSTU-small | `training/configs/model_configs/movielens_20m_ranking_hstu_small.gin` | `ckpt_ml_20m_ranking_hstu_small/iter1000` |
| MovieLens-20M | HSTU-middle | `training/configs/model_configs/movielens_20m_ranking_hstu_middle.gin` | `ckpt_ml_20m_ranking_hstu_middle/iter1000` |
| MovieLens-20M | HSTU-large | `training/configs/model_configs/movielens_20m_ranking_hstu_large.gin` | `ckpt_ml_20m_ranking_hstu_large/iter1000` |

## 5. 指标口径

| 指标 | 定义 | 来源文件 |
| --- | --- | --- |
| AUC Off | 不启用 Action KV Reuse 的 checkpoint evaluation AUC | `eval_checkpoint.py --output_file` |
| AUC On | 启用 Action KV Reuse 后的 checkpoint evaluation AUC | `eval_checkpoint_action_kv_replace.py --output_file` 或 `reuse_auc_impact_by_task.csv` |
| AUC diff | `AUC On - AUC Off` | `reuse_auc_impact_by_task.csv`、`reuse_auc_impact_summary.csv` |
| selected ratio | 被 reuse 选中的 token 比例 | `kv_replace_overall_stats*.csv` |
| NKV Off | 未开启 KV Reuse 时的归一化 KV 保留量，固定为 `1.0` | baseline 口径 |
| NKV On | `1 - selected_ratio_all_tokens` | `kv_replace_overall_stats_window_topk_same_id_action.csv` |

验收时，Reuse On 必须同时报告 AUC 和 NKV。只报告 AUC 无法体现 KV cache 减少量，只报告 reuse ratio 无法体现精度影响。

## 6. 运行前准备

以下命令默认在容器或实验环境中的项目根目录为 `/workspace/recsys`。如果实际路径不同，将第一行替换为本地项目路径即可。

```bash
cd /workspace/recsys/examples/hstu
export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
```

GPU、CUDA、PyTorch、TorchRec、gin 等运行依赖沿用本仓库 HSTU 示例环境。训练和评估命令均使用单卡 `torchrun`，多实验并行运行时需要为不同任务分配不同 `GPU` 和 `PORT`。

## 7. 数据预处理

执行以下命令生成两个数据集的 ranking 序列文件：

```bash
python3 ./preprocessor.py --dataset_name kuairand-1k --training
python3 ./preprocessor.py --dataset_name ml-20m --training
```

预处理完成后，应检查以下文件存在：

| 数据集 | 预处理输出 |
| --- | --- |
| KuaiRand-1k | `tmp_data/KuaiRand-1K/data/processed_seqs.csv` |
| MovieLens-20M | `tmp_data/ml-20m/processed_seqs.csv` |

## 8. 模型训练

以 KuaiRand-1k / HSTU-small 为例：

```bash
CONFIG=./training/configs/model_configs/kuairand_1k_ranking_hstu_small.gin
PORT=6100
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} torchrun \
  --nproc_per_node 1 \
  --master_addr localhost \
  --master_port ${PORT} \
  ./training/pretrain_gr_ranking.py \
  --gin-config-file ${CONFIG}
```

训练结束后检查 checkpoint：

```bash
ls -d ckpt_kr_1k_ranking_hstu_small/iter1000
```

替换 `CONFIG` 即可训练第 4 节中的其他 5 个模型。对应 checkpoint 目录由 gin 文件中的 `TrainerArgs.ckpt_save_dir` 决定。

## 9. Action Reuse Off Baseline 评估

Reuse Off 不进行 KV Cache 替换，作为 AUC baseline。以 KuaiRand-1k / HSTU-small 为例：

```bash
CONFIG=./training/configs/model_configs/kuairand_1k_ranking_hstu_small.gin
CKPT=ckpt_kr_1k_ranking_hstu_small/iter1000
OUT=/tmp/kuairand_1k_hstu_small_reuse_off.csv
PORT=6200
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} torchrun \
  --nproc_per_node 1 \
  --master_addr localhost \
  --master_port ${PORT} \
  ./training/eval_checkpoint.py \
  --gin-config-file ${CONFIG} \
  --ckpt-load-dir ${CKPT} \
  --output_file ${OUT}
```

验收输出：

- `${OUT}`：baseline AUC CSV。
- `NKV Off = 1.0`。

## 10. Action Reuse On 评估

Reuse On 使用 `eval_checkpoint_action_kv_replace.py`，默认验收策略为：

| 参数 | 值 |
| --- | --- |
| analysis | `kv_cache_replace` |
| reuse strategy | `window_topk_same_id` |
| reuse token type | `action` |
| window size | 可配置，示例为 `1024` |
| top-k | 可配置，示例为 `4` |

以 KuaiRand-1k / HSTU-small 为例：

```bash
CONFIG=./training/configs/model_configs/kuairand_1k_ranking_hstu_small.gin
CKPT=ckpt_kr_1k_ranking_hstu_small/iter1000
WINDOW=1024
TOPK=4
OUT_DIR=/tmp/kuairand_1k_hstu_small_reuse_on
OUT=/tmp/kuairand_1k_hstu_small_reuse_on.csv
PORT=6300
GPU=0

CUDA_VISIBLE_DEVICES=${GPU} torchrun \
  --nproc_per_node 1 \
  --master_addr localhost \
  --master_port ${PORT} \
  ./training/eval_checkpoint_action_kv_replace.py \
  --gin-config-file ${CONFIG} \
  --ckpt-load-dir ${CKPT} \
  --analysis kv_cache_replace \
  --kv-reuse-window-size ${WINDOW} \
  --kv-reuse-top-k ${TOPK} \
  --kv-reuse-token-type action \
  --kv-reuse-strategy window_topk_same_id \
  --output-dir ${OUT_DIR} \
  --output_file ${OUT}
```

验收输出：

| 文件 | 说明 |
| --- | --- |
| `${OUT}` | Reuse On AUC CSV |
| `${OUT_DIR}/kv_replace_comparison_window_topk_same_id_action.csv` | baseline 与 KV replaced AUC 对比 |
| `${OUT_DIR}/kv_replace_comparison.csv` | 单策略运行时的简化副本 |
| `${OUT_DIR}/kv_replace_overall_stats_window_topk_same_id_action.csv` | reuse ratio、replacement count、layer 数等 overall stats |
| `${OUT_DIR}/kv_replace_overall_stats.csv` | 单策略运行时的简化副本 |
| `${OUT_DIR}/kv_replace_layer_stats_window_topk_same_id_action.csv` | 分 layer reuse 统计 |
| `${OUT_DIR}/reuse_auc_impact_by_task.csv` | task-level baseline、KV replaced、diff 明细 |
| `${OUT_DIR}/reuse_auc_impact_summary.csv` | 按 reuse mode 汇总后的 AUC 和 reuse ratio |
| `${OUT_DIR}/reuse_auc_impact_summary_auc_gt_0p6.csv` | 只统计 baseline AUC 大于 0.6 的 task 后得到的汇总 |
| `${OUT_DIR}/kv_replace_overall_stats_all_modes.csv` | 所有 reuse mode 的 overall stats 汇总 |
| `${OUT_DIR}/KV_REUSE_EVAL_INSIGHTS.md` | 自动生成的简要分析报告 |

NKV On 计算方式：

```text
NKV On = 1 - selected_ratio_all_tokens
```

其中 `selected_ratio_all_tokens` 来自 `${OUT_DIR}/kv_replace_overall_stats_window_topk_same_id_action.csv`。

## 11. Window Size / Top-K 网格搜索

网格搜索用于评估不同 reuse aggressiveness 下的 AUC 与 KV 减少量。以 MovieLens-20M / HSTU-small 为例：

```bash
python ./training/analyze_action_reuse_motivation.py \
  --dataset-name ml-20m \
  --output-dir ./analysis_output/action_reuse_motivation_ml20m_small \
  --max-window-rows 5000 \
  --window-sizes 64 128 256 512 \
  --top-ks 1 2 3 4 5 \
  --run-policy-grid-auc-analysis \
  --gin-config-file ./training/configs/model_configs/movielens_20m_ranking_hstu_small.gin \
  --ckpt-load-dir ckpt_ml_20m_ranking_hstu_small/iter1000 \
  --auc-kv-replace-implementation hidden_proxy \
  --auc-filter-baseline-threshold 0.6
```

验收输出：

| 文件 | 说明 |
| --- | --- |
| `policy_grid_auc_summary.csv` | 所有 window/top-k 网格点的 AUC、AUC diff、reuse ratio、是否帕累托 |
| `policy_grid_pareto.csv` | 从 `policy_grid_auc_summary.csv` 筛选出的帕累托点 |
| `policy_grid_auc_reuse_scatter.png` | AUC / reuse ratio 散点图 |
| `MOTIVATION_REPORT.md` | 自动生成的 motivation-style Markdown 报告 |