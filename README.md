# HSTU Inference Benchmark SOW

本文档描述了HSTU推理性能测试的标准操作流程(SOW)，包括环境配置、脚本运行和结果整理三个部分。

---

## 第一部分：环境配置

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

```bash
# 交互式启动容器
~$ docker run \
    --rm --shm-size 8G --cap-add SYS_NICE --net host \
    --gpus "device=0" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    -ti recsys-examples:inference

# 或者后台启动容器
~$ docker run \
    --rm --shm-size 8G --cap-add SYS_NICE --net host \
    --gpus "device=0" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    --name hstu-benchmark \
    -d recsys-examples:inference

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
~$ nsys stats profiles/benchmark_bs8_nc1024.qdrep --report gputrace
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
# 1. 启动Docker容器
~$ docker run \
    --rm --shm-size 8G --cap-add SYS_NICE --net host \
    --gpus "device=0" \
    --volume ${SRC_DIR}:${DST_DIR} \
    --hostname $(hostname) \
    -ti recsys-examples:inference

# 2. 进入工作目录并设置环境变量
~$ cd /workspace/recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)

# 3. 运行端到端测试
~$ bash ./run_bench.sh

# 4. 解析结果
# 打开 process_data.ipynb 并运行单元格
```

---

## 注意事项

1. 确保GPU驱动版本支持CUDA 12.x和TensorRT-LLM要求
2. 运行nsys/ncu分析需要安装NVIDIA Nsight Systems/Compute工具
3. 建议在无其他GPU负载的环境下进行性能测试以获得稳定结果
4. 端到端测试可能需要较长时间(16个组合)，可根据需要修改脚本中的参数范围
5. 使用`--gpus "device=0"`指定GPU设备号，根据实际情况修改
