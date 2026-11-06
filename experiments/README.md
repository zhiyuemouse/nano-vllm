# 连续 Prefill 轮数限制实验

## 来源与修改范围

本实验基于 Nano-vLLM：

- 上游仓库：https://github.com/GeeeekExplorer/nano-vllm
- 开发仓库：https://github.com/zhiyuemouse/nano-vllm
- 上游基线 commit：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`
- 上游许可证：MIT，保留仓库原有 `LICENSE`

本实验新增：

- `max_consecutive_prefills` 配置。
- 连续 prefill 批次计数与 decode 容量检查。
- CPU 调度测试。
- GPU 策略开关运行脚本。

模型执行、attention 内核及 KV Cache 管理沿用上游实现。

## 实验目的

上游调度器优先处理 prefill。只要持续有请求成功进入 prefill，
已有 running 请求的 decode 就可能反复被推迟。

本实验尝试限制连续 prefill 批次数，在容量允许时插入 decode。
目标是研究已有请求的生成进度与新请求首 token 等待之间的权衡。

目前已验证指定场景中的调度行为，尚无性能收益结论。

## 已验证环境

| 项目 | 版本或配置 |
|---|---|
| 系统 | Windows 下的 WSL Linux |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU |
| 显存 | 4 GiB |
| Compute Capability | 8.6 |
| NVIDIA 驱动 | 576.88 |
| CUDA Toolkit | 12.9 |
| Python | 3.11.15 |
| PyTorch | 2.7.1+cu126 |
| PyTorch CUDA runtime | 12.6 |
| Transformers | 4.56.2 |
| Triton | 3.3.1 |
| FlashAttention | 2.8.3 |
| xxhash | 3.5.0 |
| 环境管理工具 | uv |

该表记录已实际运行的组合，不表示其他版本组合均已验证。

## 环境快照及安装限制

完整包版本快照见 `environment-wsl.txt`，生成命令为：

```bash
uv pip freeze --python "$VIRTUAL_ENV/bin/python" \
  > experiments/environment-wsl.txt
```

该文件是环境记录，不是可直接跨机器使用的安装锁文件。

其中至少包含两类机器专属路径：

1. Nano-vLLM 的 editable 源码路径。
2. FlashAttention 的本地 wheel 文件路径。

换机器时，不能直接假定这些路径存在，也不应直接原样安装整个快照。

Nano-vLLM 应从实际克隆的仓库目录安装：

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e .
```

以上命令仅安装项目本身，不负责补齐依赖。

FlashAttention 需要另行取得与目标环境匹配的 wheel，或按其官方说明编译。
选择 wheel 时需核对 Python、PyTorch、CUDA 构建标记、C++ ABI、
操作系统及 CPU 架构，并确认 GPU 架构受到支持。

本次使用的 wheel 文件名为：

```text
flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

该文件名仅用于记录本次环境，不是通用安装包推荐。
wheel 的 CUDA 标记、PyTorch CUDA runtime 与本机 CUDA Toolkit
是不同信息，不要求三个版本字符串完全相同。

安装后仍需分别验证：

- 包依赖一致性。
- Nano-vLLM 使用的 FlashAttention 接口能否导入。
- FlashAttention GPU 运算能否执行并通过数值对照。

本环境已完成上述检查。相关官方说明：

- https://github.com/Dao-AILab/flash-attention

## 源码与模型准备

在 Nano-vLLM 仓库根目录执行以下实验命令。

查看当前源码版本及修改：

```bash
git rev-parse HEAD
git status --short
git diff
```

基线 SHA 标识修改所基于的上游版本；
仅有基线 SHA 不包含本实验的本地修改。
复现实验还需取得包含策略、测试和实验脚本的开发版本。

模型使用 Qwen3-0.6B：

- https://huggingface.co/Qwen/Qwen3-0.6B

模型目录需包含完整权重、配置和 tokenizer 文件。
模型权重不提交到 Git，并保留其来源与许可证信息。

本文命令默认模型位于：

```text
../../models/Qwen3-0.6B
```

GPU 脚本可通过 `--model-path` 指定其他路径。
当前 CPU 测试中的模型路径仍按上述目录布局设置，
更换布局时需同步调整测试配置。

## 策略规则

配置项：

```python
max_consecutive_prefills: int = 0
```

- `0`：关闭限制，保留上游 prefill 优先行为。
- 正数：触发 decode 优先尝试的连续 prefill 批次数。
- 负数：配置检查抛出 `ValueError`。

启用策略时：

- 每次成功返回 prefill 批次，计数增加 1。
- 计数单位是批次，不是请求数。
- 分块 prefill 的每次调度也计为一轮。
- 成功返回 decode 批次后，计数清零。

达到上限后，只有同时满足以下条件才跳过 prefill：

1. running 队列非空。
2. 至少一个 running 请求通过 `BlockManager.can_append()` 检查。

如果 running 为空，或所有 running 请求均无法直接追加，
仍允许原有 prefill 路径继续尝试，因此计数可以超过上限。

容量检查用于避免已复现的强制 decode 故障场景。
它不保证后续 decode 遍历完全不会抢占，也不保证所有内存压力场景都能推进。

该策略不提供严格的等待轮数或等待时间上限。

## CPU 调度测试

激活已配置的 WSL 环境，在仓库根目录运行：

```bash
source ~/.venvs/minillm/bin/activate
python tests/test_scheduler_prefill_priority.py
```

测试只读取模型配置，不加载模型权重或执行 GPU 前向。
通过模拟模型输出验证：

- 默认策略下，持续到达的 prefill 请求推迟已有请求的 decode。
- 启用上限后，在容量充足时插入 decode，并重置计数。
- 默认关闭时，同批多个 prefill 请求仍可被调度。
- running 为空时，即使达到上限也允许 prefill。
- 分块 prefill 被 decode 打断后能够继续，缓存进度不丢失。
- waiting 请求无法分配时，可以自然回退到 decode。
- 无 running 请求可直接追加时，允许已有分块 prefill 继续，
  避免已复现的强制 decode 空调度断言。

预期输出：

```text
test_prefill_arrivals_delay_decode PASS
test_prefill_limit_allows_decode PASS
test_prefill_limit_with_empty_running PASS
test_chunked_prefill_resumes_after_decode PASS
test_prefill_allocation_failure_falls_back_to_decode PASS
test_prefill_continues_when_forced_decode_cannot_append PASS
```

## GPU 策略开关检查

在仓库根目录运行，两个命令分别启动独立进程：

```bash
python experiments/smoke_scheduler_policy.py \
  --model-path ../../models/Qwen3-0.6B \
  --max-consecutive-prefills 0

python experiments/smoke_scheduler_policy.py \
  --model-path ../../models/Qwen3-0.6B \
  --max-consecutive-prefills 2
```

运行配置：

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-0.6B |
| 模型 dtype | BF16 |
| enforce_eager | True |
| tensor_parallel_size | 1 |
| max_num_seqs | 2 |
| max_model_len | 128 |
| max_num_batched_tokens | 128 |
| gpu_memory_utilization | 0.70 |
| temperature | 0.6 |
| 每请求生成 token 数 | 4 |
| ignore_eos | True |
| enable_thinking | False |

`gpu_memory_utilization` 用于计算缓存预算，不是整个进程的显存硬上限。

忽略 EOS 是为了固定本次测试的生成长度，不用于评价回答质量。

请求到达顺序：

- 第 0 轮前加入 A：“你好吗”
- 第 1 轮前加入 B：“1+1等于几？”
- 第 2 轮前加入 C：“用一句话介绍北京。”

脚本记录调度类型、选中的请求 ID 和连续 prefill 计数，
并检查请求没有重复完成、每个请求返回 4 个生成 token。

## 已验证结果

CPU 测试中的调度与状态断言通过。

真实 GPU 运行中观察到：

| 策略参数 | 第 0 轮 | 第 1 轮 | 第 2 轮 |
|---|---|---|---|
| 0 | A prefill | B prefill | C prefill |
| 2 | A prefill | B prefill | A、B decode |

两种配置均正常结束，A、B、C 各完成一次，
每个请求返回 4 个生成 token，未触发重复完成断言。

本次使用随机采样。调度改变了采样调用顺序与批次，
因此不要求两种策略生成完全相同的 token IDs。

这些结果验证了调度行为及真实模型执行链路，
不构成不同调度配置下的逐 token 数值一致性证明。

## 当前限制

- 已完成短请求场景的首 token 延迟、token 间延迟和请求总延迟测量，见 [结果说明](results/README.md)；尚未测量吞吐量。
- 暂无性能收益结论。
- CPU 压力测试使用模拟输出，不等同于真实 GPU 压力测试。
- 尚未全面验证抢占、前缀共享、多卡和各种缓存耗尽场景。
- GPU 检查使用 eager 模式，未验证 CUDA Graph 模式。
- 当前请求按调度轮次到达，尚未模拟按实际时间到达的负载。
