<div align="center">

# RecSys-SOW

**High-Performance Generative Recommenders Implementation**

基于 Meta 论文 ["Actions Speak Louder Than Words"](https://arxiv.org/abs/2402.17152) 的高性能生成式推荐系统实现

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[📖 论文](https://arxiv.org/abs/2402.17152) • [🚀 快速开始](#快速开始) • [📚 文档](#核心模块详解)

</div>

---

## 目录

- [项目简介](#项目简介)
- [软件架构](#软件架构)
- [代码目录结构](#代码目录结构)
- [核心模块详解](#核心模块详解)
- [依赖关系](#依赖关系)
- [快速开始](#快速开始)
- [性能基准测试（复现请直接跳到此节）](#hstu-inference-benchmark)

---

## 项目简介

本项目基于 **NVIDIA RecSys** 实现，是 **HSTU (Hierarchical Sequential Transduction Units)** 的高性能实现版本。HSTU 是一种专为大规模推荐系统设计的新型架构，源自 Meta 论文 ["Actions Speak Louder Than Words"](https://arxiv.org/abs/2402.17152)，通过将推荐任务重构为生成式建模问题，实现了检索和排序任务的统一建模。

> **项目来源**: 本项目基于 [NVIDIA/recsys](https://github.com/NVIDIA/recsys) 进行开发和优化，整合了 NVIDIA 的高性能 CUDA 内核和分布式训练能力。

### 核心特性

| 特性 | 描述 |
|:-----|:-----|
| **高性能注意力机制** | HSTU-2 (Ampere/Ada) 和 HSTU-3 (Hopper) 双版本优化，支持 FP8 量化 |
| **动态嵌入表** | 基于 HKV 的高效哈希表实现，支持 GPU/HBM 混合存储与动态扩容 |
| **分布式训练** | 支持张量并行、数据并行、序列并行等多种并行策略 |
| **高效推理** | 集成 TensorRT-LLM KV Cache，支持 Paged Attention 推理优化 |
| **多任务学习** | 支持多任务预测头，适用于点击率预估、转化率预估等场景 |

---

## 软件架构

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Application Layer                        │
    │                                                                 │
    │   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
    │   │  Training   │   │  Inference  │   │     Benchmark       │   │
    │   └─────────────┘   └─────────────┘   └─────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                          Model Layer                            │
    │                                                                 │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │  RankingGR  │  RetrievalGR  │  InferenceGR  │  Modules   │   │
    │  └──────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Core Library Layer                       │
    │                                                                 │
    │   ┌───────────────────────┐    ┌───────────────────────┐        │
    │   │      DynamicEmb       │    │         HSTU          │        │
    │   │  ┌─────────────────┐  │    │  ┌─────────────────┐  │        │
    │   │  │  Python API     │  │    │  │   HSTU-2        │  │        │
    │   │  ├─────────────────┤  │    │  │  (Ampere/Ada)   │  │        │
    │   │  │  CUDA Kernels   │  │    │  ├─────────────────┤  │        │
    │   │  │  • Lookup       │  │    │  │   HSTU-3        │  │        │
    │   │  │  • Optimizer    │  │    │  │  (Hopper+FP8)   │  │        │
    │   │  │  • Unique       │  │    │  └─────────────────┘  │        │
    │   │  └─────────────────┘  │    │                       │        │
    │   └───────────────────────┘    └───────────────────────┘        │
    └─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                       External Dependencies                     │
    │                                                                 │
    │   PyTorch  │  TorchREC  │  FBGEMM_GPU  │  TensorRT-LLM  │  HKV  │
    └─────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 描述 | 关键文件 |
|:-----|:-----|:---------|
| **DynamicEmb** | 动态嵌入表，支持哈希表后端、GPU/HBM 混合存储、动态扩容与淘汰策略 | `corelib/dynamicemb/` |
| **HSTU Attention** | 高性能注意力机制，支持多种 GPU 架构和 FP8 量化 | `corelib/hstu/` |
| **RankingGR** | 生成式排序模型，支持多任务预测 | `examples/hstu/model/ranking_gr.py` |
| **RetrievalGR** | 生成式检索模型 | `examples/hstu/model/retrieval_gr.py` |
| **KV Cache Manager** | 推理优化，支持 Paged Attention | `examples/hstu/modules/gpu_kv_cache_manager.py` |

---

## 代码目录结构

```
recsys/
├── corelib/                          # 核心库
│   ├── dynamicemb/                    # 动态嵌入表库
│   │   ├── dynamicemb/               # Python模块
│   │   │   ├── __init__.py           # 模块入口
│   │   │   ├── batched_dynamicemb_tables.py  # 批量动态嵌入表实现
│   │   │   ├── dynamicemb_config.py  # 配置类定义
│   │   │   ├── optimizer.py          # 优化器实现
│   │   │   ├── key_value_table.py    # KV表实现
│   │   │   ├── embedding_admission.py # 嵌入准入策略
│   │   │   ├── dump_load.py          # 模型保存/加载
│   │   │   └── planner/              # 分片规划器
│   │   ├── src/                      # CUDA源码
│   │   │   ├── dynamic_emb_op.cu     # 动态嵌入算子
│   │   │   ├── hkv_variable.cu       # HKV变量实现
│   │   │   ├── lookup_forward.cu     # 前向查找内核
│   │   │   ├── lookup_backward.cu    # 反向传播内核
│   │   │   ├── optimizer.cu          # 优化器内核
│   │   │   └── unique_op.cu           # 去重操作
│   │   ├── benchmark/                # 性能基准测试
│   │   ├── test/                     # 单元测试
│   │   └── example/                  # 使用示例
│   │
│   └── hstu/                         # HSTU注意力机制库
│       ├── hstu_attn/                # HSTU-2 Python接口
│       │   ├── __init__.py
│       │   └── hstu_attn_interface.py
│       ├── hopper/                   # HSTU-3 (Hopper GPU优化)
│       │   ├── __init__.py
│       │   ├── hstu_attn_interface.py # 统一接口
│       │   ├── hstu_fwd_kernel.h     # 前向内核
│       │   ├── hstu_bwd_kernel.h     # 反向内核
│       │   └── instantiations/       # 内核实例化
│       └── csrc/                     # CUDA源码
│           └── hstu_attn/            # HSTU-2 CUDA实现
│
├── examples/                         # 示例应用
│   ├── hstu/                         # HSTU训练/推理示例
│   │   ├── model/                    # 模型定义
│   │   │   ├── base_model.py         # 基础模型类
│   │   │   ├── ranking_gr.py         # 排序模型
│   │   │   ├── retrieval_gr.py       # 检索模型
│   │   │   └── inference_ranking_gr.py # 推理模型
│   │   │
│   │   ├── modules/                  # 核心模块
│   │   │   ├── embedding.py          # 嵌入层
│   │   │   ├── hstu_block.py         # HSTU块实现
│   │   │   ├── hstu_attention.py     # 注意力层
│   │   │   ├── hstu_processor.py     # 数据预/后处理
│   │   │   ├── mlp.py                # MLP预测头
│   │   │   ├── position_encoder.py   # 位置编码
│   │   │   ├── gpu_kv_cache_manager.py # GPU KV缓存管理
│   │   │   └── paged_hstu_infer_layer.py # 分页推理层
│   │   │
│   │   ├── training/                 # 训练相关
│   │   │   ├── pretrain_gr_ranking.py # 排序模型训练
│   │   │   ├── pretrain_gr_retrieval.py # 检索模型训练
│   │   │   ├── trainer/              # 训练器
│   │   │   └── configs/               # 训练配置(.gin)
│   │   │
│   │   ├── inference/                # 推理相关
│   │   │   ├── inference_gr_ranking.py # 推理脚本
│   │   │   ├── benchmark/            # 推理基准测试
│   │   │   ├── triton/               # Triton服务
│   │   │   └── configs/              # 推理配置
│   │   │
│   │   ├── dataset/                  # 数据集处理
│   │   │   ├── sequence_dataset.py   # 序列数据集
│   │   │   ├── inference_dataset.py  # 推理数据集
│   │   │   └── utils.py             # 数据工具
│   │   │
│   │   ├── distributed/              # 分布式训练
│   │   │   ├── sharding.py           # 分片策略
│   │   │   └── dmp_to_tp.py          # DMP到TP转换
│   │   │
│   │   ├── configs/                  # 模型配置
│   │   │   ├── hstu_config.py        # HSTU配置
│   │   │   └── task_config.py        # 任务配置
│   │   │
│   │   └── utils/                    # 工具函数
│   │
│   └── commons/                      # 公共模块
│       ├── checkpoint/               # 检查点工具
│       └── utils/                    # 通用工具
│
├── docker/                           # Docker配置
│   ├── Dockerfile                    # 标准Dockerfile
│   └── Dockerfile.nve                # NVE版本
│
├── third_party/                      # 第三方依赖
│   ├── cutlass/                      # CUTLASS库
│   └── HierarchicalKV/               # HKV哈希表库
│
└── README.md                         # 本文档
```

---

## 核心模块详解

### 1. DynamicEmb (动态嵌入表)

DynamicEmb提供模型并行动态嵌入表功能，专为推荐系统稀疏训练设计。

**主要特性**:
- 基于HKV哈希表后端，支持GPU HBM和Host内存混合存储
- 支持动态扩容和淘汰策略(LRU/LFU/Customized)
- 与TorchREC无缝集成
- 支持多种优化器: `EXACT_SGD`, `ADAM`, `EXACT_ADAGRAD`, `EXACT_ROWWISE_ADAGRAD`

**核心类**:
```python
from dynamicemb import (
    DynamicEmbTableOptions,    # 表配置选项
    BatchedDynamicEmbeddingTablesV2,  # 批量嵌入表
    DynamicEmbDump,            # 模型导出
    DynamicEmbLoad,            # 模型加载
)
```

**配置示例**:
```python
table_options = DynamicEmbTableOptions(
    dim=256,                          # 嵌入维度
    max_capacity=1_000_000,           # 最大容量
    evict_strategy=DynamicEmbEvictStrategy.LRU,
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
    optimizer_type=OptimizerType.Adam,
)
```

### 2. HSTU (注意力机制)

HSTU是针对推荐系统优化的注意力机制实现。

**版本对比**:

| 特性 | HSTU-2 | HSTU-3 |
|------|--------|--------|
| 支持GPU | Ampere, Ada, Hopper | Hopper only |
| 数据类型 | FP16, BF16 | FP16, BF16, FP8 |
| 量化 | 不支持 | 支持多种量化模式 |
| 性能 | 标准 | 优化(TMA, WGMMAs) |

**统一接口**:
```python
from hstu_attn import hstu_attn_varlen_func

output = hstu_attn_varlen_func(
    q,                      # (total_q, nheads, headdim)
    k,                      # (total_k, nheads, headdim)
    v,                      # (total_k, nheads, headdim)
    cu_seqlens_q,          # 累积序列长度
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_contexts=None,      # 上下文token数
    num_targets=None,       # 目标token数
    window_size=(-1, 0),    # 因果注意力
    alpha=1.0,              # RAB缩放因子
)
```

**支持的注意力掩码类型**:
- 无掩码
- 局部掩码 (滑动窗口)
- 因果掩码
- 上下文+因果掩码
- 目标+因果掩码
- Delta_q掩码

### 3. 模型架构 (RankingGR)

排序模型由三部分组成：

```
输入特征 → 嵌入表 → HSTU块 → MLP预测头 → 多任务输出
```

**数据流**:
1. **嵌入层**: 将稀疏特征转换为稠密嵌入
2. **HSTU块**: 多层注意力处理序列信息
3. **MLP预测头**: 多任务预测（点击、点赞等）

### 4. Forward 调用关系

以下是 RankingGR 模型的完整 forward 调用链：

```
RankingGR.forward(batch)
│
├── ShardedEmbedding.forward(features)           # 嵌入查找
│   └── TorchREC Embedding Lookup                # 稀疏特征 → 稠密嵌入
│
├── jt_dict_grad_scaling_and_allgather()         # 梯度缩放与全聚合
│
├── dmp_batch_to_tp(batch)                       # 数据并行 → 张量并行转换
│
├── HSTUBlock.forward(embeddings, batch)
│   │
│   ├── HSTUBlockPreprocessor.forward()          # 数据预处理
│   │   ├── 位置编码
│   │   ├── 序列拼接
│   │   └── JaggedData 构造
│   │
│   ├── FusedHSTULayer.forward(jd) × N layers    # N 层 HSTU
│   │   │
│   │   └── fused_hstu_op()                      # 融合 HSTU 算子
│   │       │
│   │       └── FusedHSTULayerFunction.apply()   # Autograd 函数
│   │           │
│   │           ├── [Forward]
│   │           │   ├── triton_weighted_layer_norm_fwd()     # LayerNorm
│   │           │   ├── triton_addmm_silu_fwd()              # Linear + SiLU
│   │           │   │   └── split → U, V, Q, K
│   │           │   ├── hstu_attn_varlen_func()              # HSTU Attention
│   │           │   │   ├── CUTLASS Backend (HSTU-2/HSTU-3)
│   │           │   │   │   └── flash_attn_cuda.varlen_fwd()
│   │           │   │   └── Triton Backend
│   │           │   │       └── triton_hstu_attention_fwd()
│   │           │   ├── triton_layer_norm_mul_dropout_fwd()  # Norm × U + Dropout
│   │           │   └── triton_addmm_silu_fwd()              # Output Projection
│   │           │
│   │           └── [Backward]
│   │               ├── triton_addmm_silu_bwd()
│   │               ├── triton_layer_norm_mul_dropout_bwd()
│   │               ├── hstu_attn_varlen_bwd()
│   │               └── triton_weighted_layer_norm_bwd()
│   │
│   └── HSTUBlockPostprocessor.forward()         # 数据后处理
│       └── 序列并行 All-Gather
│
├── MLP.forward(hidden_states)                   # 预测头
│   └── Linear → Activation → Linear → ...
│
└── MultiTaskLossModule.forward(logits, labels)  # 多任务损失
    └── BCE/CE Loss per task
```

**关键数据结构**:

| 结构 | 描述 | 形状 |
|------|------|------|
| `RankingBatch` | 输入批次数据 | features, labels, seqlen |
| `JaggedTensor` | 稀疏嵌入输出 | values, offsets |
| `JaggedData` | HSTU 内部数据 | values, seqlen, offsets, num_candidates |

**HSTU Attention 掩码逻辑**:

```
序列结构: [contextual_tokens] [historical_tokens] [candidate_tokens]
          ├───────────────────┤ ├───────────────┤ ├──────────────┤
          │   全双向注意力      │ │   因果注意力   │ │  目标组注意力  │
          └───────────────────┘ └───────────────┘ └──────────────┘

- contextual_tokens: 用户画像等，可双向 attend
- historical_tokens: 用户行为序列，因果掩码
- candidate_tokens: 候选物品，按 target_group_size 分组
```

---

## 依赖关系

### 外部依赖

| 依赖 | 版本要求 | 用途 |
|------|----------|------|
| PyTorch | CUDA版本 | 深度学习框架 |
| TorchREC | >= v1.2.0 | 模型并行嵌入 |
| FBGEMM_GPU | main分支 | 稀疏操作 |
| TensorRT-LLM | v0.19.0+ | 推理优化(可选) |

### 内部模块依赖

```
examples/hstu/model/ranking_gr.py
    ├── modules/embedding.py → corelib/dynamicemb
    ├── modules/hstu_block.py → corelib/hstu
    └── modules/mlp.py
```

---

## 快速开始

详细的训练和推理指南请参考：
- [训练文档](examples/hstu/training/README.md)
- [推理文档](examples/hstu/inference/README.md)

---

## HSTU Inference Benchmark

本文档描述了HSTU推理性能测试的标准操作流程，包括环境配置、脚本运行和结果整理三个部分。

---

## 第一部分：环境配置

### 1.0 快速环境配置（推荐）

推荐直接拉取预构建的 Docker 镜像，无需手动构建 TensorRT-LLM 和安装依赖：

```bash
# 拉取预构建镜像
~$ docker pull shijieliu01/recsys-examples:inference.2026.1.14
```

使用预构建镜像后，可直接跳过 1.1 和 1.2 步骤，前往 [1.3 数据预处理](#13-数据预处理)。

> **注意**：如果选择手动构建环境，请按顺序执行 1.1 和 1.2 步骤。

---

### 1.1 构建TensorRT-LLM (含HSTU KV Cache扩展)

HSTU推理使用来自TensorRT-LLM的定制KV缓存管理器。当前版本基于TensorRT-LLM v0.19.0的HSTU专用实现。

```bash
# 克隆TensorRT-LLM仓库
~$ cd ${WORKING_DIR}
~$ git clone -b hstu-kvcache-recsys-examples https://github.com/geoffreyQiu/TensorRT-LLM.git tensorrt-llm-kvcache && cd tensorrt-llm-kvcache

# 初始化子模块
~$ git submodule update --init --recursive

# 构建Docker镜像
~$ make -C docker release_build CUDA_ARCHS="80-real;86-real"
```

### 1.2 安装Recsys-Examples依赖

使用`INFERENCEBUILD=1`选项跳过Megatron安装(推理不需要)。

```bash
~$ cd ${WORKING_DIR}
~$ git clone --recursive -b ${TEST_BRANCH} ${TEST_REPO} recsys-examples && cd recsys-examples
~$ TRTLLM_KVCACHE_IMAGE="tensorrt_llm/release:latest" docker build \
    --build-arg BASE_IMAGE=${TRTLLM_KVCACHE_IMAGE} \
    --build-arg INFERENCEBUILD=1 \
    -t recsys-examples:inference \
    -f docker/Dockerfile .
```

### 1.3 数据预处理

在运行推理benchmark之前，需要对数据集进行预处理：

```bash
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)

# 预处理KuaiRand-1K数据集用于推理
~$ python3 ./preprocessor.py --dataset_name "kuairand-1k" --inference
```

### 1.4 验证环境

运行一个简单的推理测试来验证环境配置正确：

```bash
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ python3 ./inference/benchmark/inference_benchmark.py --batch_size 1 --num_candidates 128
```

---

## 第二部分：脚本运行

本部分介绍三个层次的性能测试脚本，分别用于端到端测试、System级别分析和算子级别分析。

### 2.0 Docker环境激活

在运行任何测试脚本之前，需要先启动Docker容器并激活环境。

#### 2.0.1 启动Docker容器

使用预构建镜像启动容器：

```bash
# 交互式启动容器（使用预构建镜像）
~$ docker run \
    --rm --shm-size 8G --cap-add SYS_NICE --net host \
    --gpus "device=0" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    -ti shijieliu01/recsys-examples:inference.2026.1.14

# 或者后台启动容器
~$ docker run \
    --rm --shm-size 8G --cap-add SYS_NICE --net host \
    --gpus "device=0" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    --name hstu-benchmark \
    -d shijieliu01/recsys-examples:inference.2026.1.14

# 进入运行中的容器
~$ docker exec -ti hstu-benchmark bash
```

#### 2.0.2 激活Python环境

```bash
# 进入容器后设置环境变量
~$ cd /workspace/recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
```

> **注意**：如果在后台启动了容器，需要确保在运行脚本前设置好`PYTHONPATH`环境变量。可以使用以下方式：
> ```bash
> # 在容器内创建环境设置脚本
> ~$ echo 'export PYTHONPATH=${PYTHONPATH}:$(realpath ../)' >> ~/.bashrc
> ```

### 2.1 端到端测试 (run_bench.sh)

**用途**：测试不同batch size和candidate数量下的端到端推理性能

**脚本位置**：`examples/hstu/run_bench.sh`

**测试参数**：
- Batch Size: 1, 2, 4, 8
- Num Candidates: 128, 256, 512, 1024

**运行方式**：

```bash
~$ cd /workspace/recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ bash ./run_bench.sh
```

**输出**：
- 日志目录：`logs/`
- 日志文件格式：`benchmark_bs{BATCH_SIZE}_nc{NUM_CANDIDATES}.log`
- 日志内容示例：
  ```
  KV Caches: 8, Candidate Embeddings: 128, Total time(ms): 123.45
  ```

### 2.2 System级别分析 (run_bench_nsys.sh)

**用途**：使用NVIDIA Nsight Systems进行系统级性能分析，获取CUDA kernel调用和NVTX事件

**脚本位置**：`examples/hstu/run_bench_nsys.sh`

**测试参数**：
- Batch Size: 8 (默认)
- Num Candidates: 1024 (默认)

**运行方式**：

```bash
~$ cd /workspace/recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ bash ./run_bench_nsys.sh
```

**输出**：
- 日志目录：`p-logs/`
- Profiling目录：`profiles/`
- Profiling文件格式：`benchmark_bs{BATCH_SIZE}_nc{NUM_CANDIDATES}.nsys-rep`

**分析方式**：

```bash
# 在NVIDIA Nsight Systems GUI中打开 .nsys-rep 文件进行可视化分析
~$ nsys stats profiles/benchmark_bs8_nc1024.nsys-rep --report gputrace
```

### 2.3 算子级别分析 (run_bench_ncu.sh)

**用途**：使用NVIDIA NCu (Nsight Compute)进行单个kernel的性能分析，获取roofline、occupancy等信息

**脚本位置**：`examples/hstu/run_bench_ncu.sh`

**测试参数**：
- Batch Size: 8 (默认)
- Num Candidates: 1024 (默认)

**分析的Kernels**：
1. `initialize_with_index_addressor_kernel` - Embedding查询算子
2. `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_64x3_tn` - Tensor Core矩阵乘法
3. `hstu_fwd_kernel` - HSTU前向传播算子

**运行方式**：

```bash
~$ cd /workspace/recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)
~$ bash ./run_bench_ncu.sh
```

**输出**：
- 日志目录：`pcu-logs/`
- Profiling目录：`profiles-cu/`
- Profiling文件格式：
  - `benchmark_bs{BATCH_SIZE}_nc{NUM_CANDIDATES}_embedding.ncu-rep`
  - `benchmark_bs{BATCH_SIZE}_nc{NUM_CANDIDATES}_tc.ncu-rep`
  - `benchmark_bs{BATCH_SIZE}_nc{NUM_CANDIDATES}_hstu_fwd.ncu-rep`

**分析方式**：

```bash
# 查看kernel的roofline分析
~$ ncu --report metrics profiles-cu/benchmark_bs8_nc1024_hstu_fwd.ncu-rep

# 查看详细信息
~$ ncu --details all profiles-cu/benchmark_bs8_nc1024_hstu_fwd.ncu-rep
```

---

## 第三部分：结果整理

### 3.1 解析端到端测试结果

使用`process_data.ipynb`来解析和整理benchmark结果。

**第一步：解析日志文件**

```python
import os
import re
from openpyxl import Workbook

LOG_DIR = "logs"
OUT_FILE = "benchmark_results.xlsx"

fname_pattern = re.compile(r"benchmark_bs(\d+)_nc(\d+)\.log")

line_pattern = re.compile(
    r"KV Caches:\s*(\d+),\s*Candidate Embeddings:\s*(\d+),\s*Total time\(ms\):\s*([0-9.]+)"
)

wb = Workbook()
ws = wb.active
ws.title = "results"

headers = [
    "file",
    "batch_size",
    "num_candidate",
    "kv_caches",
    "candidate_embeddings",
    "total_time_ms",
]
ws.append(headers)

count = 0

for fname in os.listdir(LOG_DIR):
    m = fname_pattern.match(fname)
    if not m:
        continue

    batch_size = int(m.group(1))
    num_candidate = int(m.group(2))
    path = os.path.join(LOG_DIR, fname)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m2 = line_pattern.search(line)
            if m2:
                kv_caches = int(m2.group(1))
                cand_emb = int(m2.group(2))
                total_time = float(m2.group(3))

                ws.append([
                    fname,
                    batch_size,
                    num_candidate,
                    kv_caches,
                    cand_emb,
                    total_time,
                ])
                count += 1

wb.save(OUT_FILE)
print(f"Done. Parsed {count} records.")
print(f"Saved to: {OUT_FILE}")
```

**输出文件**：`benchmark_results.xlsx`

包含字段：
| 字段 | 描述 |
|------|------|
| file | 原始日志文件名 |
| batch_size | 批处理大小 |
| num_candidate | 候选数量 |
| kv_caches | KV缓存数量 |
| candidate_embeddings | 候选embedding数量 |
| total_time_ms | 总时间(毫秒) |

### 3.2 按Batch Size分组

```python
import pandas as pd

IN_FILE = "benchmark_results.xlsx"

for batch_size in [1, 2, 4, 8]:
    OUT_FILE = f"benchmark_results_bs{batch_size}.xlsx"
    
    df = pd.read_excel(IN_FILE)
    df_bs = df[df["batch_size"] == batch_size]
    df_bs.to_excel(OUT_FILE, index=False)
    
    print(f"Done. {len(df_bs)} rows saved to {OUT_FILE}")
```

### 3.3 按Candidate数量分组

```python
import pandas as pd

IN_FILE = "benchmark_results.xlsx"

for num_candidate in [128, 256, 512, 1024]:
    OUT_FILE = f"benchmark_results_nc{num_candidate}.xlsx"
    
    df = pd.read_excel(IN_FILE)
    df_nc = df[df["num_candidate"] == num_candidate]
    df_nc.to_excel(OUT_FILE, index=False)
    
    print(f"Done. {len(df_nc)} rows saved to {OUT_FILE}")
```

### 3.4 结果分析建议

1. **端到端性能**：查看`benchmark_results.xlsx`中`total_time_ms`列，分析不同batch size和candidate数量下的性能变化

2. **System级别分析**：使用Nsight Systems打开`.nsys-rep`文件，分析：
   - CUDA kernel执行时间线
   - NVTX事件分布
   - Host-Device数据传输

3. **算子级别分析**：使用Nsight Compute打开`.ncu-rep`文件，分析：
   - Roofline性能分析
   - Kernel occupancy
   - 内存带宽使用
   - 计算吞吐量

---

## 附录：快速开始

```bash
# 1. 拉取预构建镜像（推荐）
~$ docker pull shijieliu01/recsys-examples:inference.2026.1.14

# 2. 启动Docker容器
~$ docker run \
    --rm --shm-size 8G --cap-add SYS_NICE --net host \
    --gpus "device=0" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    -ti shijieliu01/recsys-examples:inference.2026.1.14

# 3. 进入工作目录并设置环境变量
~$ cd /workspace/recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)

# 4. 运行端到端测试
~$ bash ./run_bench.sh

# 5. 解析结果
# 打开 process_data.ipynb 并运行单元格
```

---

## 注意事项

1. 确保GPU驱动版本支持CUDA 12.x和TensorRT-LLM要求
2. 运行nsys/ncu分析需要安装NVIDIA Nsight Systems/Compute工具
3. 建议在无其他GPU负载的环境下进行性能测试以获得稳定结果
4. 端到端测试可能需要较长时间(16个组合)，可根据需要修改脚本中的参数范围
5. 使用`--gpus "device=0"`指定GPU设备号，根据实际情况修改
