---
title: vllm量化格式笔记
date: 2026-09-11 10:00:00
tags: [vLLM,大语言模型,模型量化,INT4,FP16,FP8,Qwen3.5,RTX 4060,模型部署]
description: 记录在RTX 4060 8GB上使用Qwen3.5 4B，通过INT4 权重、FP16激活和FP8 KV Cache 部署vLLM的个人实践配置与排障经验。
---

## 常见数据类型与位宽对照

| 格式 | 全称 | 位宽 | 构成 | 范围 | 精度 | 典型用途 |
| ------ | ------ | ------ | ------ | ------ | ------ | ---------- |
| INT4 | 4 位整数 | 4 | 整数 | -8～7 | 低 | AWQ/GPTQ 权重 |
| INT8 | 8 位整数 | 8 | 整数 | -128～127 | 中 | 量化权重/激活 |
| FP32 | 单精度浮点 | 32 | 1+8+23 | 大 | 高 | 训练基准 |
| FP16 | 半精度浮点 | 16 | 1+5+10 | 中 | 较高 | 推理/混合精度 |
| BF16 | Brain 浮点 | 16 | 1+8+7 | 同 FP32 | 中低 | 大模型训练/推理 |
| FP8 | 8 位浮点 | 8 | 1+4+3 或 1+5+2 | 中 | 低 | H100/4090 推理 |
| NF4 | 4 位正态浮点 | 4 | 16 级 | 小 | 低 | QLoRA |

## 我的部署目标：RTX 4060 8GB + Qwen3.5 4B

这次不再按 7B～8B 模型来规划，而是选择 **Qwen3.5 4B** 作为本地部署模型。RTX 4060 只有 8GB 显存，4B 模型更适合做个人开发、代码辅助、知识库测试和本地 API 服务。

我最终采用的配置思路是：

- **模型权重使用 INT4**：选择已经转换好的 AWQ 4-bit 模型；
- **激活和主要矩阵计算使用 FP16**：也就是 W4A16；
- **KV Cache 使用 FP8**：尽量减少多轮对话和长提示词带来的缓存占用；
- **单卡、低并发运行**：先保证不 OOM，再逐步增加上下文长度和并发数。

这里的 `W4A16` 容易被误解。它不是说模型所有计算都使用 INT4，而是权重以 4-bit 保存，读取后配合 FP16 激活完成计算。`--kv-cache-dtype fp8` 也只影响 KV Cache，不会把权重和激活一起改成 FP8。

RTX 4060 属于Ada Lovelace 架构，硬件具备 FP8 相关能力，但 FP8 KV Cache 能否正常使用，还要看当前 CUDA、PyTorch、vLLM 版本以及模型后端。我的做法是先把它作为目标配置，启动后通过日志确认；如果当前环境不支持，就临时退回 FP16 KV Cache，而不是强行修改一堆参数。

## 一、为什么选择 Qwen3.5 4B

选择 4B 模型，主要是基于显存和使用场景做的取舍。

第一，4B 模型的 INT4 权重体积比较容易控制。理论上 4B 参数使用 4-bit 存储，权重本身大约只需要 2GB 左右，实际还要加上量化 scale、zero point、模型结构、CUDA 工作区和 KV Cache，但总体上比 7B 或 8B 模型宽松很多。

第二，4B 模型更适合 RTX 4060 8GB。显存不用全部分配给权重，还可以留一部分给 FP16 激活、运行时临时空间和 KV Cache。这样在实际使用中不容易出现“模型能启动，但一输入长文本就崩溃”的情况。

第三，个人本地部署通常更在意稳定和响应速度。对于日常问答、代码解释、简单 RAG 和接口调试，4B 模型已经足够作为一个低成本实验模型。遇到特别复杂的推理任务，再切换到远程大模型，整体体验反而更合理。

需要注意的是，下面的模型目录统一写成 `./models/qwen3.5-4b-awq`，实际使用时要替换成你确认过的 **Qwen3.5 4B AWQ 模型仓库**。如果你下载的是 GPTQ 版本，启动参数中的 `--quantization awq` 必须改为 `--quantization gptq`。

不要直接下载原始 FP16 模型，然后仅仅加上 `--quantization awq`。vLLM 的这个参数是告诉框架如何读取已经量化好的权重，不会在启动时替你完成完整的 FP16 到 INT4 量化。

## 二、准备 Linux 或 WSL2 环境

我更建议在 Ubuntu 或 WSL2 Ubuntu 中运行 vLLM，不建议把 Windows 原生 Python 环境作为第一选择。vLLM 对 CUDA、PyTorch、显卡驱动和编译扩展比较敏感，Linux 环境遇到问题时更容易找到对应日志和解决方案。

首先确认驱动能识别显卡：

```bash
nvidia-smi
```

重点检查：

- GPU 型号是否为 RTX 4060；
- 显存是否识别为 8GB 左右；
- 驱动版本是否满足当前 PyTorch 和 vLLM 的要求；
- 是否有其他程序提前占用了大量显存。

然后创建独立的 Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install vllm
```

安装完成后，先不要急着下载模型，确认 vLLM 可以正常导入：

```bash
python -c "import torch, vllm; print(torch.cuda.get_device_name(0)); print(vllm.__version__)"
```

如果这里已经报 CUDA 或动态库错误，应该先处理环境问题。个人经验是，不要一开始就手动安装多套 CUDA Toolkit，也不要为了“修一个报错”连续升级 PyTorch、驱动和 vLLM。先记录当前版本，再一次只调整一个组件，后面排查会轻松很多。

## 三、下载 Qwen3.5 4B INT4 模型

8GB 显存没有必要在本地加载 FP16 原模型再做量化。我的做法是直接准备 AWQ 4-bit 版本，并将模型放在项目目录下：

```bash
mkdir -p ./models/qwen3.5-4b-awq

hf download <Qwen3.5-4B-AWQ模型仓库> \
  --local-dir ./models/qwen3.5-4b-awq
```

如果环境中没有 `hf` 命令，可以安装 Hugging Face Hub：

```bash
pip install -U huggingface_hub
```

下载完成后，模型目录至少应该包含模型配置、tokenizer、safetensors 权重和量化配置。可以简单检查：

```text
models/qwen3.5-4b-awq/
├── config.json
├── tokenizer_config.json
├── tokenizer.json 或 tokenizer.model
├── model-*.safetensors
└── quantization_config.json
```

最值得检查的是 `quantization_config.json`。里面应该能看到 AWQ 或 GPTQ 相关信息、4-bit 位宽以及 group size 等字段。如果目录中只有普通 FP16 权重，没有量化配置，就不要直接把它当作 AWQ 模型启动。

另外，不建议为了省事删除 tokenizer、generation config 或模型配置文件。vLLM 有时可以自动补全部分配置，但 Qwen 系列的 chat template、特殊 token 和生成行为最好保持模型仓库的完整状态。

## 四、第一次启动：先用保守配置验证

我第一次不会直接把上下文长度设成 16K，也不会一开始就开多个并发。先使用 4096 上下文、单请求，把模型加载、量化后端和 FP8 KV Cache 分开验证。

AWQ 版本的启动命令如下：

```bash
vllm serve ./models/qwen3.5-4b-awq \
  --served-model-name qwen3.5-4b \
  --quantization awq \
  --dtype float16 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --host 0.0.0.0 \
  --port 8000
```

这组参数的实际含义是：

- `--served-model-name qwen3.5-4b`：给 API 暴露一个简短、固定的模型名；
- `--quantization awq`：按 AWQ INT4 权重加载；
- `--dtype float16`：将激活和主要计算路径设为 FP16；
- `--kv-cache-dtype fp8`：将 KV Cache 设为 FP8；
- `--gpu-memory-utilization 0.88`：最多使用约 88% 的显存，给 CUDA 和临时工作区留余量；
- `--max-model-len 4096`：先把最大上下文控制在 4096 tokens；
- `--max-num-seqs 1`：先只允许一个序列，方便判断显存峰值；
- `--max-num-batched-tokens 2048`：限制批处理规模，避免短时间内显存突然冲高。

如果使用的是 GPTQ 4-bit 模型，只需要将量化参数改成：

```bash
--quantization gptq
```

如果模型卡已经说明可以自动识别量化格式，也可以不写 `--quantization`，但我在排查阶段更喜欢显式指定。这样从启动日志中更容易看出到底走的是 AWQ、GPTQ，还是发生了后端回退。

## 五、从启动日志确认三个配置真的生效

服务启动后，我不会只看“端口已经监听”就认为部署完成，而是重点检查终端日志。

首先确认模型是以 AWQ 或 GPTQ 4-bit 权重加载的，而不是悄悄按 FP16 加载。如果权重文件很小但启动时显存占用异常高，通常要检查量化配置是否被正确读取。

其次确认计算 dtype 是 FP16。这里的目标是 W4A16：权重低比特存储，激活和主要矩阵计算使用 FP16。不同 vLLM 版本的日志文字可能不完全一样，但不应该出现明显的 dtype 不兼容或算子回退错误。

最后确认 KV Cache dtype 是否显示为 FP8。如果日志提示当前设备或后端不支持 FP8 KV Cache，不要反复重启碰运气，可以先把参数改成：

```bash
--kv-cache-dtype auto
```

或者明确退回：

```bash
--kv-cache-dtype fp16
```

FP16 KV Cache 会多占一些显存，但 Qwen3.5 4B 在 RTX 4060 8GB 上通常仍然有较大的调参空间。先让服务稳定运行，再单独研究 FP8 KV Cache，比把模型、量化格式、CUDA 和 KV Cache 一起排查更省时间。

如果当前 vLLM 版本不接受 `fp8` 这个写法，可以根据该版本的 CLI 提示尝试 `fp8_e4m3`。参数名称以当前安装版本输出的 `vllm serve --help` 为准，不要照抄其他版本的命令。

## 六、调用 API 验证模型

启动成功后，先查询服务实际暴露的模型名：

```bash
curl http://127.0.0.1:8000/v1/models
```

如果返回的模型名是 `qwen3.5-4b`，就用 OpenAI 兼容接口发送测试请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
    "messages": [
      {
        "role": "user",
        "content": "请用三句话解释 INT4 权重、FP16 激活和 FP8 KV Cache 分别解决什么问题。"
      }
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

第一轮测试我会观察四件事：

1. 是否能正常返回内容；
2. 返回内容是否出现重复、乱码或异常截断；
3. 请求过程中显存是否突然冲到满载；
4. 日志中是否出现 CUDA kernel、量化后端或 KV Cache 的警告。

只发一个很短的问题还不够。短问题占用的 KV Cache 很少，无法验证长上下文配置。至少还要测试一段较长的中文说明、一段代码生成请求和连续两三轮对话。

## 七、4060 8GB 上的调参顺序

如果第一次启动就 OOM，我不会直接换量化格式，而是按照下面的顺序缩减资源：

第一步，把最大上下文从 4096 降到 2048：

```bash
--max-model-len 2048
```

第二步，把显存利用率从 0.88 降到 0.85：

```bash
--gpu-memory-utilization 0.85
```

第三步，把批处理 token 数降到 1024：

```bash
--max-num-batched-tokens 1024
```

第四步，排查阶段尝试启用 eager 模式：

```bash
--enforce-eager
```

`--enforce-eager` 可能牺牲一部分性能，但有时可以减少 CUDA Graph 带来的额外显存开销。我只在启动 OOM 或运行过程中显存峰值异常时使用它，正常情况下还是先使用默认执行模式。

如果模型已经能稳定运行，再尝试提高上下文和并发。我的测试顺序是：

```text
4096 上下文，单并发
        ↓
8192 上下文，单并发
        ↓
4096 上下文，双并发
        ↓
8192 上下文，双并发
```

每次只改一个变量，并记录启动参数、峰值显存、首 token 延迟和生成速度。否则多个参数一起变化，最后很难判断究竟是哪一项导致了 OOM 或速度下降。

## 八、我最后保留的配置

在我的 RTX 4060 8GB 上，如果目标是日常问答、代码辅助和简单本地 API 服务，我会先保留下面这组偏稳妥的配置：

```bash
vllm serve ./models/qwen3.5-4b-awq \
  --served-model-name qwen3.5-4b \
  --quantization awq \
  --dtype float16 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --host 0.0.0.0 \
  --port 8000
```

如果实测中 FP8 KV Cache 的日志不正常，就只把这一项替换掉：

```bash
--kv-cache-dtype fp16
```

如果 4096 上下文长期稳定、显存还有余量，再尝试将 `--max-model-len` 增加到 8192，或者将 `--max-num-seqs` 增加到 2。我的建议是优先增加上下文，不要同时增加上下文和并发，因为两者都会提高 KV Cache 的使用量。

对于 RTX 4060 8GB，我不建议把 `--gpu-memory-utilization` 一开始设成 0.98，也不建议一上来就追求 16K 上下文。个人本地服务最怕的不是少几百 tokens，而是运行一段时间后因为显存碎片、长请求或并发请求突然崩溃。

## 九、个人总结

这次选择 Qwen3.5 4B 后，配置比之前规划 7B～8B 模型轻松了不少。我的最终理解是：

1. **INT4 负责降低权重占用**。Qwen3.5 4B 使用 AWQ 4-bit 后，模型权重更容易放进 RTX 4060 8GB；
2. **FP16 负责保持计算路径稳定**。激活仍然使用 FP16，避免把权重低比特存储误认为所有计算都必须低比特；
3. **FP8 KV Cache 负责节省上下文缓存**。它在多轮对话和长提示词下更有价值，但必须以当前环境实际支持为准；
4. **4B 模型更适合个人机器**。相比盲目追求更大的参数规模，能稳定运行、响应及时、方便调试更重要；
5. **预量化模型比本地临时量化更现实**。4060 8GB 不适合在部署时再承担完整量化流程，直接下载 AWQ/GPTQ 权重更省时间；
6. **先单并发跑通，再提高利用率**。先确认短问答、长提示词、多轮对话都不 OOM，再尝试 8192 上下文或双并发；
7. **启动参数不是性能保证**。真正的效果要看启动日志、峰值显存、首 token 延迟、生成 tokens/s 和实际回答质量。

总的来说，`Qwen3.5 4B + INT4 权重 + FP16 激活 + FP8 KV Cache` 对 RTX 4060 8GB 是一套比较务实的本地部署方案。它的重点不是追求最高吞吐，而是在有限显存下获得一个能长期运行、方便调用、质量尚可的个人推理服务。

> **实操提示：** 上面的模型目录和仓库名称使用了占位写法，实际部署时请替换为真实的 Qwen3.5 4B AWQ 模型仓库。vLLM 对量化格式和 FP8 KV Cache 的支持会随版本变化，最终以当前版本的 `vllm serve --help`、模型卡和启动日志为准。
