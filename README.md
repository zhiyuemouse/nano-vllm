<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

## 本 Fork：Prefill 与 Decode 调度实验

本仓库基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 `bb823b3e06983d71485a8e1f23715ebd87d98ef8` 开发，沿用上游的模型执行和 GPU attention 实现，保留 [MIT 许可证](LICENSE)。

上游优先调度 prefill，新请求持续到达时，已有请求的 decode 可能被推迟。本 fork 新增默认关闭的 `max_consecutive_prefills`。达到连续 prefill 轮数上限，且至少一个运行中的请求当前可追加时，本轮优先尝试 decode。若没有请求可直接追加，则允许原有 prefill 路径继续尝试。

我实现了调度开关、计数与容量检查，并补充了 [CPU 调度回归测试](tests/test_scheduler_prefill_priority.py)、GPU 调度 trace、延迟采集和统计脚本。

在 RTX 3050 Ti Laptop 上，使用 Qwen3-0.6B 和三条分别在 `0 / 10 / 50 ms` 到达的短请求，对比策略关闭与上限设为 `2`。两批采用相反的运行顺序，每种策略各测五轮。

| 延迟中位数指标 | run8（0 → 2） | run9（2 → 0） |
|---|---:|---:|
| 请求 A 的首次 token 间隔 | −39.34% | −35.21% |
| 请求 B 的首次 token 间隔 | −54.98% | −51.91% |
| 请求 A/B 的第二次 token 间隔 | +98.38% | +103.07% |
| 请求 C 的首 token 延迟 | +40.96% | +57.40% |

策略把 A/B 的一轮 decode 排到 C 的 prefill 前。A/B 更早拿到第二个 token，随后等第三个 token 的间隔拉长，C 的首 token 也更晚。三条请求的总延迟中位数在两批分别下降 **5.83%–6.64%** 和 **1.24%–1.76%**，降幅限于各批观测。

[运行与复现说明](experiments/README.md) · [结果分析与原始数据](experiments/results/README.md)

## 上游项目说明

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation (upstream)

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark (upstream)

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
