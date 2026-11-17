---
title: qwen3.8-27B量化模型用ascend部署实战全流程
date: 2026-10-02 10:00:00
tags: [Qwen,模型部署]
categories: [项目]
description: 以 Atlas A3 为目标环境，使用 vLLM-Ascend 部署 Qwen3.8-27B-W8A8，覆盖环境检查、模型下载、服务启动、API 验证和性能测试。
---

# A3部署实战

这篇文章记录一套面向 Atlas A3 的 Qwen3.8-27B-W8A8 部署流程，目标是把部署工作拆成环境检查、模型准备、服务启动、接口验证和性能测试五个可以独立验收的阶段。

本文给出的版本组合是：

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 |
| Python | 3.12 |
| PyTorch | 2.10.0 |
| torch_npu | 2.10.0.post4 |
| vLLM | 0.23.0 |
| vLLM-Ascend | 0.23.0 |
| triton_ascend | 3.2.2 |
| 模型 | Qwen3.8-27B-W8A8 |

> 重要提示：启动前必须根据 `npu-smi info` 确认设备型号、NPU 数量和每卡 HBM 容量。文中的 TP=2 只是双卡示例，不能替代真实硬件适配确认。

## 一、工程目录

统一使用 `/workspace/work` 作为工作目录：

```text
/workspace/work/
├── data/
│   └── models/
│       └── Qwen3.8-27B-w8a8/
├── deploy/
│   └── qwen38/
│       ├── env.sh
│       ├── download_model.sh
│       ├── start.sh
│       ├── test_api.sh
│       └── benchmark.sh
└── logs/
    └── benchmark/
```

部署目标包括：

- 检查 Ubuntu、Python、CANN 和 NPU 状态；
- 下载并校验模型文件；
- 启动 vLLM-Ascend OpenAI 兼容服务；
- 通过 `/v1/models` 和 Chat Completions 验证服务；
- 保存 Benchmark 结果和运行日志。

## 二、检查服务器环境

### 1. 检查操作系统与 Python

```bash
cat /etc/os-release
python3 --version
```

确认系统为 Ubuntu 22.04，Python 为 3.12.x。如果服务器上存在多个 Python 版本，应明确后续脚本使用的是同一个解释器和虚拟环境。

### 2. 检查 NPU 与设备节点

```bash
npu-smi info
ls -l /dev/davinci*
```

记录以下信息：

- NPU 型号和数量；
- 每张卡的 HBM 容量及当前使用量；
- `/dev/davinci*` 设备节点是否存在；
- 当前是否有其他推理或训练任务占用设备。

### 3. 检查 CANN

```bash
ls /usr/local/Ascend/
find /usr/local/Ascend -name set_env.sh
```

如果使用标准 Toolkit 环境，可以执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

CANN 的实际安装目录可能不同，应以服务器上的 `set_env.sh` 路径为准。

## 三、创建部署目录

```bash
mkdir -p /workspace/work/data/models
mkdir -p /workspace/work/deploy/qwen38
mkdir -p /workspace/work/logs/benchmark

cd /workspace/work/deploy/qwen38
export MODEL_PATH=/workspace/work/data/models/Qwen3.8-27B-w8a8

df -h /workspace/work
```

模型权重、临时下载文件、运行日志和编译缓存都可能占用磁盘空间。正式部署前应确认磁盘余量，实际占用以模型仓库文件大小为准。

## 四、检查推理依赖

先激活 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

然后检查 Python 包版本和 NPU 可用性：

```bash
python3 - <<'PY'
import sys
import torch
import torch_npu
import vllm
import vllm_ascend
import triton

print("Python:", sys.version)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("vllm:", vllm.__version__)
print("triton:", triton.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
PY
```

如果版本已经满足部署要求并且 NPU 可用，不要重复安装或随意升级依赖。需要安装或调整版本时，应先确认 CANN、PyTorch、torch_npu、vLLM-Ascend 和 Triton 的兼容矩阵，避免混装导致运行时错误。

如果确实需要从 PyPI 安装 Python 包，可使用清华大学镜像源：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope
```

也可以配置为默认源：

```bash
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

对于 NPU 相关组件，镜像源只能解决下载问题，不能替代官方版本匹配和安装说明。安装前仍应核对对应版本的官方文档。

## 五、下载并检查模型

### 1. 安装 ModelScope CLI

先检查 ModelScope 是否已经安装：

```bash
python3 -c "import modelscope; print(modelscope.__version__)"
```

未安装时使用清华源：

```bash
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple modelscope
```

### 2. 创建下载脚本

创建 `/workspace/work/deploy/qwen38/download_model.sh`：

```bash
#!/bin/bash
set -e

MODEL_DIR="/workspace/work/data/models/Qwen3.8-27B-w8a8"
mkdir -p "$MODEL_DIR"

echo "开始下载 Qwen3.8-27B-W8A8..."

modelscope download \
    --model Eco-Tech/Qwen3.8-27B-w8a8 \
    --local_dir "$MODEL_DIR"

echo "模型下载完成：$MODEL_DIR"
du -sh "$MODEL_DIR"
ls -lh "$MODEL_DIR"
```

执行脚本：

```bash
chmod +x /workspace/work/deploy/qwen38/download_model.sh
/workspace/work/deploy/qwen38/download_model.sh
```

仓库 ID 应以所选模型的官方模型卡为准。如果仓库不可访问，应先核对模型来源、仓库名称和网络环境，不要用相似名称的权重替代目标模型。

### 3. 网络受限时使用内网传输

可以在有网络的机器上完成下载，再通过内网传输到服务器：

```bash
rsync -avP /本地模型目录/ \
  username@服务器IP:/workspace/work/data/models/Qwen3.8-27B-w8a8/
```

### 4. 检查文件和量化配置

```bash
cd /workspace/work/data/models/Qwen3.8-27B-w8a8
ls -lh
```

检查 `config.json`：

```bash
python3 - <<'PY'
import json

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print("model_type:", config.get("model_type"))
print("architectures:", config.get("architectures"))
print("quantization_config:", config.get("quantization_config"))
PY
```

确认配置文件、Tokenizer、权重索引和全部权重分片都存在。模型仓库提供校验和时，应按模型卡中的方法校验文件完整性。

## 六、配置运行环境

创建 `/workspace/work/deploy/qwen38/env.sh`：

```bash
#!/bin/bash

# CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Model
export MODEL_PATH=/workspace/work/data/models/Qwen3.8-27B-w8a8

# NPU：双卡示例，按实际设备修改
export ASCEND_RT_VISIBLE_DEVICES=0,1

# Runtime
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

# Optional
export VLLM_USE_MODELSCOPE=True

# Service
export HOST=0.0.0.0
export PORT=8000
export MODEL_NAME=qwen3.8

export LOG_DIR=/workspace/work/logs
mkdir -p "$LOG_DIR"
```

`ASCEND_RT_VISIBLE_DEVICES` 必须与实际可用设备和 TP 配置匹配。CANN 或 ATB 环境脚本路径不一致时，应改成服务器实际路径。

## 七、启动 vLLM-Ascend

创建 `/workspace/work/deploy/qwen38/start.sh`：

```bash
#!/bin/bash
set -e

source /workspace/work/deploy/qwen38/env.sh

echo "Starting Qwen3.8-27B-W8A8"
echo "Model: $MODEL_PATH"
echo "NPU: $ASCEND_RT_VISIBLE_DEVICES"

vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 2 \
    --quantization ascend \
    --trust-remote-code \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    2>&1 | tee "$LOG_DIR/server.log"
```

赋权并启动：

```bash
chmod +x /workspace/work/deploy/qwen38/start.sh
cd /workspace/work/deploy/qwen38
./start.sh
```

参数说明：

- `--tensor-parallel-size 2` 是双卡示例，需要根据真实 NPU 数量、HBM 和模型部署要求调整；
- `--max-model-len 16384`、`--max-num-seqs 32` 是初始验证参数，不代表最终性能配置；
- 出现 OOM 时，先确认模型是否适配当前卡数和 HBM，再逐步降低上下文长度、并发数或批处理 Token 数；
- 如果当前 vLLM-Ascend 版本不识别某个可选参数，应先查看 `vllm serve --help` 和对应版本的官方模型指南。

## 八、验证 OpenAI 兼容 API

服务启动后另开一个终端，先查询模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

创建 `/workspace/work/deploy/qwen38/test_api.sh`：

```bash
#!/bin/bash

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8",
    "messages": [
      {
        "role": "user",
        "content": "解释一下大模型推理中的 Prefill 和 Decode 有什么区别？"
      }
    ],
    "temperature": 0.6,
    "max_tokens": 512
  }'
```

执行：

```bash
chmod +x /workspace/work/deploy/qwen38/test_api.sh
/workspace/work/deploy/qwen38/test_api.sh
```

成功标准是接口返回合法 JSON，且 `choices` 中包含模型生成内容。若模型名称校验失败，应以 `/v1/models` 返回的 `id` 为准检查请求体中的 `model` 字段。

## 九、运行性能 Benchmark

创建 `/workspace/work/deploy/qwen38/benchmark.sh`：

```bash
#!/bin/bash
set -e

source /workspace/work/deploy/qwen38/env.sh

vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --model qwen3.8 \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 256 \
    --num-prompts 20 \
    --request-rate inf \
    --save-result \
    --result-dir /workspace/work/logs/benchmark
```

执行：

```bash
chmod +x /workspace/work/deploy/qwen38/benchmark.sh
/workspace/work/deploy/qwen38/benchmark.sh
```

重点记录以下指标：

| 指标 | 含义 |
| --- | --- |
| TTFT | 首 Token 延迟 |
| TPOT | 平均每 Token 生成时间 |
| ITL | 相邻输出 Token 间隔 |
| Output Throughput | 输出 Token/s |
| Request Throughput | 请求/s |
| HBM Usage | NPU 显存使用量 |

建议先使用小规模请求确认稳定性，再逐步提高并发，并保存每次测试的模型、上下文长度、并发数和运行参数，确保不同结果具备可比性。

## 十、日常运维命令

查看 NPU：

```bash
watch -n 1 npu-smi info
```

查看服务日志：

```bash
tail -f /workspace/work/logs/server.log
```

检查端口：

```bash
ss -lntp | grep 8000
```

查看 vLLM 进程：

```bash
ps -ef | grep vllm
```

前台启动时可以使用 `Ctrl+C` 停止。后台运行时应先确认目标 PID，再停止对应进程，不要使用 `killall python`，避免影响同机的其他任务。

## 十一、常见问题

| 现象 | 排查方向 |
| --- | --- |
| `torch.npu.is_available()` 为 `False` | 检查 CANN、驱动、设备挂载和 `torch_npu` |
| `No module named vllm_ascend` | 检查当前 Python 环境和 vLLM-Ascend 安装状态 |
| `Out of memory` | 检查 NPU HBM、TP、上下文长度、并发和批处理 Token 数 |
| `Unsupported quantization` | 检查模型量化格式与 `--quantization ascend` 的兼容性 |
| HCCL 初始化失败 | 检查可见设备、通信环境和设备拓扑 |
| 模型架构不支持 | 检查 vLLM-Ascend 版本与模型配置是否匹配 |
| 启动阶段耗时较长 | 可能正在加载权重、编译算子或执行图捕获 |

排查时优先保留完整启动日志，并把硬件信息、依赖版本、启动参数和报错上下文放在一起分析。不要在未确认原因前批量升级所有依赖。

## 十二、部署验收清单

- [ ] SSH 登录服务器并确认 Ubuntu 22.04
- [ ] `npu-smi info` 能正常显示设备
- [ ] 确认 NPU 型号、数量和 HBM
- [ ] 确认 CANN、Python 和推理依赖版本
- [ ] 创建工程目录并确认磁盘空间
- [ ] 下载模型并检查权重文件
- [ ] 检查 `config.json` 量化配置
- [ ] 创建并检查 `env.sh`
- [ ] 创建并检查 `start.sh`
- [ ] vLLM 服务成功启动
- [ ] `/v1/models` 返回模型
- [ ] Chat Completions 请求成功
- [ ] Benchmark 完成并保存结果
- [ ] 保存启动日志、测试配置和性能数据

## 十三、参考资料

- [vLLM-Ascend GitHub](https://github.com/vllm-project/vllm-ascend)
- [Qwen3.8-27B vLLM-Ascend 部署文档](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/Qwen3.8-27B.md)
- [ModelScope：Eco-Tech/Qwen3.8-27B-w8a8](https://modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8)

> 版本提示：文档链接可能随仓库分支更新。部署时应选择与 vLLM-Ascend 0.23.0 对应的文档版本，并以实际服务器硬件和模型仓库说明为准。
