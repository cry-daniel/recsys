# Action KV Reuse 实验复现说明

本文介绍Action KV Reuse算法的相关实验复现方法，覆盖数据预处理、ranking 模型训练、模型配置选择、Action KV Reuse on/off 评估，以及网格搜索和结果可视化。

## 1. 数据预处理

使用如下命令对 kuairand-1k 和 MovieLens-20M 两个数据集进行预处理。

```bash
cd /workspace/recsys/examples/hstu
export PYTHONPATH=${PYTHONPATH}:$(realpath ../)

python3 ./preprocessor.py --dataset_name kuairand-1k --training
python3 ./preprocessor.py --dataset_name ml-20m --training
```

预处理后，模型训练时主要读取以下文件：

- KuaiRand-1k: `examples/hstu/tmp_data/KuaiRand-1K/data/processed_seqs.csv`
- MovieLens-20M: `examples/hstu/tmp_data/ml-20m/processed_seqs.csv`


## 2. Action Reuse Off/On 模型AUC与NKV评估

### 2.1 模型训练

需要先进行模型训练，然后执行模型评估。对于不同数据集（kuairand-1k 和 MovieLens-20M）、不同的模型规模（HSTU-small、HSTU-middle、HSTU-large），相应的模型配置文件均存放于`examples/hstu/training/configs/model_configs`目录下。


使用如下命令进行模型训练：

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

替换 `CONFIG` 即可训练其他 5 个模型。训练结束后检查对应 checkpoint 是否生成：

```bash
ls -d ckpt_kr_1k_ranking_hstu_small/iter1000
```

### 2.2 Action Reuse Off模型评估

不开启Action KV Cache Reuse时，使用普通 checkpoint eval脚本`./training/eval_checkpoint.py`进行评估，命令如下。替换`CONFIG`参数可评估其他五个模型。

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

由于 Action Reuse Off 不做KV Cache替换，因此：

```text
NKV Off = 1.0
```

运行完成后，`eval_checkpoint.py` 会把 AUC 自动写入 `--output_file` 指定的 CSV 文件。

### 2.3 Action Reuse On 模型评估

开启Action KV Cache Reuse时，使用 `eval_checkpoint_action_kv_replace.py`脚本进行模型评估，对所有 HSTU layer 统一应用 `window_topk_same_id` 策略。命令如下。其中可通过修改`WINDOW`和`TOPK`参数调整 KV Reuse策略的窗口大小和Top-K取值；替换`CONFIG`参数可评估其他五个模型。

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

运行完成后，脚本会自动生成两类结果。

第一类是 `--output_file` 指定的 AUC CSV：

- `${OUT}`: AUC 评估结果。

第二类是 `--output-dir` 下的 KV Reuse 详细结果。以上例为例，结果会写入：

```text
/tmp/kuairand_1k_hstu_small_reuse_on
```

主要文件包括：

- `${OUT_DIR}/kv_replace_comparison_window_topk_same_id_action.csv`: baseline 与 KV replaced AUC 对比。
- `${OUT_DIR}/kv_replace_comparison.csv`: 单一 action/window-topk 策略运行时生成的简化同名副本。
- `${OUT_DIR}/kv_replace_overall_stats_window_topk_same_id_action.csv`: reuse ratio、replacement count、layer 数等统计。
- `${OUT_DIR}/kv_replace_overall_stats.csv`: 单一 action/window-topk 策略运行时生成的简化同名副本。
- `${OUT_DIR}/kv_replace_layer_stats_window_topk_same_id_action.csv`: 分 layer 统计。
- `${OUT_DIR}/reuse_auc_impact_by_task.csv`: 每个 AUC task 的 baseline、KV replaced、diff 明细。
- `${OUT_DIR}/reuse_auc_impact_summary.csv`: 按 reuse mode 汇总后的 AUC 和 reuse ratio。
- `${OUT_DIR}/reuse_auc_impact_summary_auc_gt_0p6.csv`: 只统计 baseline AUC 大于 0.6 的 task 后得到的汇总。
- `${OUT_DIR}/kv_replace_overall_stats_all_modes.csv`: 所有 reuse mode 的 overall stats 汇总。
- `${OUT_DIR}/KV_REUSE_EVAL_INSIGHTS.md`: 脚本自动生成的简要分析报告。

NKV计算方法：

```text
NKV = 1 - selected_ratio_all_tokens
```
其中 `selected_ratio_all_tokens` 来自 `kv_replace_overall_stats_window_topk_same_id_action.csv`。


## 3. 针对window_size & Top-K 参数的网格搜索实验

以MovieLens-20M 的HSTU-small模型为例，使用脚本analyze_action_reuse_motivation.py进行实验：

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
运行结束后，实验结果会存放在 --output-dir 指定的目录下，其中包含的核心文件如下：

- `policy_grid_auc_summary.csv`: 所有 window/top-k 网格点的 AUC、AUC diff、reuse ratio、是否帕累托。
- `policy_grid_pareto.csv`: 从 `policy_grid_auc_summary.csv` 中筛出的帕累托点。
- `policy_grid_auc_reuse_scatter.png`: 脚本自动绘制的 AUC/reuse ratio 散点图。
- `MOTIVATION_REPORT.md`: 脚本自动生成的 motivation-style Markdown 报告。

