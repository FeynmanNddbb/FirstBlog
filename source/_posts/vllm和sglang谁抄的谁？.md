---
title: vLLM和SGLang谁抄的谁？
date: 2026-10-07 10:00:00
tags: [vLLM,SGLang,大模型推理,LLM Serving]
categories: [人工智能]
description: 从共同物理约束和源码入口理解vLLM与SGLang。
---

## 先说结论：没有谁抄谁

把vLLM和SGLang放在一起比较时，最容易出现的提问就是：“到底是谁抄了谁？”

我的结论是：**这里没有一个适合直接套用的“抄袭”结论。**两个框架解决的问题和面对的瓶颈本来就高度重合，大方向自然会越来越接近；真正不同的地方，更多在实现细节、系统边界和应用场景。

大模型Serving并不是凭空设计出来的产品功能，而是在一组非常具体的物理约束下不断优化：GPU显存有限，请求长度动态变化，Decode阶段要逐Token生成，线上流量又会不断变化。只要目标都是提高吞吐、降低延迟、提高显存利用率，很多工程选择就会收敛到相似的形状。

所以，看到两个项目都实现了Continuous Batching、KV Cache、Prefix Caching、Speculative Decoding，首先应该想到的是“这些能力已经成为现代高性能LLM Serving的主流工程模式”，而不是只看类名或命令行参数猜谁复制了谁。

## 为什么vLLM和SGLang很像？

两套成熟的LLM Serving Engine面对的是同一组问题：GPU显存是有限资源，请求会随时进入和退出队列，Prefill和Decode的计算特征不同，而且模型越来越大，单卡放不下就必须做分布式并行。

这会带来一条相似的优化路线：先做动态请求批处理，再把KV Cache从普通张量管理提升为专门的内存系统；接着复用相同前缀，减少重复Prefill；最后再处理投机解码、分布式并行和Prefill/Decode资源分离。

可以把它们的能力大致对照成下面这样：

| 共同问题 | vLLM方案 | SGLang方案 |
| --- | --- | --- |
| 动态请求批处理 | Continuous Batching / Scheduler | Continuous Batching / Scheduler |
| KV显存管理 | Paged KV Cache | KV Cache Pool |
| 相同前缀避免重算 | Prefix Caching | RadixAttention / Radix Cache |
| 一次多生成Token | Speculative Decoding | Speculative Decoding |
| 大模型分卡 | TP / PP / DP / EP / CP等并行方式 | TP / PP / DP / EP / CP等并行方式 |
| Prefill和Decode资源分离 | Disaggregated Prefill/Decode | PD Disaggregation |

这里的重点不是说两边的代码完全相同，而是说它们在系统层面都需要回答类似的问题：请求下一步执行多少Token？哪些KV可以复用？显存块从哪里分配？哪个设备执行哪一部分计算？

证据可以直接看两个项目在相应版本的README，而不是只看几年前的博客或二手文章：

- [vLLM README.md，提交58b3298457dde7b4554b3b4e20b238c0ac2c3a65](https://github.com/vllm-project/vllm/tree/58b3298457dde7b4554b3b4e20b238c0ac2c3a65)
- [SGLang README.md，提交fcb080bd4007d88c76fd901a07248066b460684a](https://github.com/sgl-project/sglang/tree/fcb080bd4007d88c76fd901a07248066b460684a)

这类版本锚点很重要。两个项目迭代都很快，如果只根据旧博客比较，很容易把过去的实现方式误认为当前版本的能力边界。

## 共同问题：Scheduler和KV Cache是核心

### 动态请求批处理

线上请求不会整齐地同时到达，也不会在同一个时间结束。一个请求可能还在Prefill，另一个请求已经进入Decode，第三个请求刚刚排队。如果每个请求都独占一次完整的模型执行，GPU很快就会出现空洞，吞吐也会明显下降。

Continuous Batching的思路是：每一轮调度都重新决定参与计算的请求和Token数量。完成的请求退出，新的请求加入，正在生成的请求继续拿到少量Decode预算。Scheduler的职责，就是在延迟、吞吐、显存和公平性之间做这个决定。

### KV显存管理

Decode每生成一个Token，都要读取历史Token对应的Key和Value。如果KV Cache按照连续大块内存保存，请求长度变化和请求退出会带来严重的碎片问题，也很难动态扩容。

因此，vLLM会把KV Cache拆成可管理的Paged KV Cache块；SGLang则通过KV Cache Pool等方式管理可复用的缓存空间。实现名不同，但背后的抽象很接近：把KV Cache从“模型临时产生的一块张量”变成“由调度器和内存管理器共同维护的资源”。

### 相同前缀避免重算

很多请求共享系统提示词、工具定义、文档前缀或对话历史。如果每次都从第一个Token开始做Prefill，计算会被重复浪费。

vLLM提供Prefix Caching，SGLang使用RadixAttention和Radix Cache组织前缀。Radix Tree更适合表达不同请求之间的公共前缀，Paged KV Cache更强调块式内存管理。二者的具体数据结构和命中逻辑可以不同，但目标都是复用已有KV，减少重复计算和首Token等待时间。

## 两套方案的差别在哪里？

不能因为能力列表相似，就把vLLM和SGLang当成同一个项目。差异往往藏在“系统怎样组织这些能力”里面。

vLLM给人的整体印象是通用、高性能的模型推理服务引擎，核心执行链路围绕请求、调度、KV Cache和GPU执行展开。它适合把模型包装成稳定的OpenAI兼容服务，再通过并发、批处理、量化和并行配置逐步调优。

SGLang除了提供Serving能力，也很强调结构化生成和程序化调用场景。它的运行时会围绕Radix Cache等机制组织共享前缀，对多轮对话、Agent、工具调用和具有重复结构的请求尤其有吸引力。

这不是简单的优劣排名。对于一个普通文本生成API，两者都可能满足需求；对于大量共享Prompt前缀的工作负载，应该重点看前缀缓存命中率和调度行为；对于复杂结构化生成，则要看框架对请求编排、约束生成和模型适配的支持。最终选择应该由模型、硬件、并发模式和业务请求形态决定，而不是由“谁更像谁”决定。

另外，Speculative Decoding、TP/PP/DP/EP/CP以及Prefill/Decode分离，也不是某个项目独有的概念。它们分别对应生成速度、模型切分、专家并行、上下文处理和资源隔离等问题。只要规模继续增大，这些问题就会持续出现，解决方案相互借鉴和逐渐收敛是很自然的结果。

## 第一次读vLLM源码，先看这5个入口

如果第一次读vLLM源码，不建议一上来就扎进CUDA Kernel。Kernel当然重要，但它通常只解释“某个局部计算怎么更快”，很难先回答“一个请求是怎么走完整条链路的”。更合适的阅读顺序是：**Engine→Scheduler→KV→ModelRunner**。

下面这套入口统一锁定在提交`58b3298457dde7b4554b3b4e20b238c0ac2c3a65`，先确认版本，再开始读代码：

1. [`vllm/v1/engine/core.py`](https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/v1/engine/core.py)：看`class EngineCore`，理解请求生命周期、消息处理和Engine主循环。
2. [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/v1/core/sched/scheduler.py)：看`class Scheduler`，理解Waiting和Running请求如何变成下一步Token Budget。
3. [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/v1/core/kv_cache_manager.py)：看`class KVCacheManager`，理解Block分配、Prefix Cache和KV资源管理。
4. [`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/58b3298457dde7b4554b3b4e20b238c0ac2c3a65/vllm/v1/worker/gpu_model_runner.py)：看`class GPUModelRunner`，理解Scheduler Output最后如何变成真实的GPU Execution。
5. README中的Serving能力列表：把源码里看到的类和实际对外承诺对应起来，避免拿旧版本的实现印象去解释当前代码。

阅读时可以一直追踪一个简单的问题：一次请求从进入Engine开始，什么时候进入Waiting，什么时候被Scheduler选中，KV块何时分配，最后又如何交给ModelRunner执行。这个问题串起来以后，再回头看Prefix Cache、Speculative Decoding或者分布式并行，就不会只剩下零散的类名。

## 总结

vLLM和SGLang都在解决现代LLM Serving的共同物理问题：有限的GPU显存、动态请求、逐Token Decode、重复前缀和多卡执行。

相同的不是类名，而是系统抽象。Continuous Batching、KV Cache管理、Prefix Caching、Speculative Decoding和Prefill/Decode分离，已经逐渐成为高性能LLM Serving的主流工程模式和行业共识。

所以，这两个框架更值得比较的是：同一个抽象下，谁的实现更适合当前模型、硬件和业务流量。先把共同问题看懂，再看各自的工程取舍，通常比争论“谁抄的谁”更接近技术事实。