# 调度延迟实验记录

## 数据来源

- `policy_0_run1.json`：2026-11-06 19:20:33 +08:00 开始。
- `policy_2_run1.json`：2026-11-06 19:22:11 +08:00 开始。
- 顺序为 0 → 2，每种策略独立进程运行，预热 2 轮，正式记录 5 轮。
- 两份文件包含环境、prompt token IDs、启动及采样参数。请求按 step 0、1、2 到达，各生成 4 个 token。
- `policy_0.json`、`policy_2.json` 为此前手工保存的探索数据，缺少完整元数据。本节使用带 `run1` 的两份文件。

测量时 HEAD 为 `41bc6823a0f8015b867e7c330767daf2c795cff4`，工作区为 dirty。
本次提交保存当前测量代码及原始文件，但不能追溯证明两次测量时未提交源码逐字一致；原始 metadata 保留，不改写为干净状态。

## 测量口径

到达时间在 `add_request()` 前使用 `perf_counter()` 记录。包装 `postprocess()`，通过请求 token 数增加识别新 token；后处理及 GPU 同步结束后记录批次时间，同批 token 共用时间戳。

- TTFT：到达到首个生成 token 的引擎端观测延迟。
- ITL1/2/3：第 1→2、2→3、3→4 个生成 token 的观测间隔。
- 请求总延迟：每轮先计算 TTFT + 全部 ITL，再对五轮取中位数。
- 包含主机调度、同步和记录开销，不含文本解码、网络传输。
- 每轮检查请求清空、块归还、无完整前缀缓存，重置 prefill 计数并设种子 42。

## 本批中位数

单位为 ms，变化率为 `(参数2 - 参数0) / 参数0 × 100%`。

| 请求 | 指标 | 参数0 | 参数2 | 变化率 |
|---|---|---:|---:|---:|
| A | 首次 ITL | 104.89 | 70.14 | -33.13% |
| B | 首次 ITL | 69.79 | 33.54 | -51.94% |
| A/B | 第二次 ITL | 34.83 | 70.13 | +101.35% |
| C | TTFT | 36.01 | 69.43 | +92.82% |
| C | 首次 ITL | 133.99 | 99.16 | -26.00% |
| A | 请求总延迟 | 210.51 | 211.58 | +0.51% |
| B | 请求总延迟 | 174.17 | 174.40 | +0.13% |
| C | 请求总延迟 | 231.22 | 231.16 | -0.02% |

策略将一次 A/B decode 移到 C prefill 之前，缩短 A/B 首次间隔，但把等待移到后续间隔，并推迟 C 的首 token。总延迟差异较小，当前没有稳定总延迟收益或吞吐提升结论。

## 复查与重新采集

在 Nano-vLLM 仓库根目录执行统计。无参数时读取 run8。

```bash
python experiments/summarize_scheduler_policy.py
```

复算其他批次时，用 `--policy-0` 和 `--policy-2` 指定同一批次的两份 JSON。以下命令分别复算 run1 和 run9。

```bash
python experiments/summarize_scheduler_policy.py \
  --policy-0 experiments/results/policy_0_run1.json \
  --policy-2 experiments/results/policy_2_run1.json

python experiments/summarize_scheduler_policy.py \
  --policy-0 experiments/results/policy_0_run9.json \
  --policy-2 experiments/results/policy_2_run9.json
```

单独运行统计自测时，使用默认的 run8 文件。该命令检查中位数计算、源码摘要校验和 seed 不一致的负例。

```bash
python experiments/summarize_scheduler_policy.py --self-test
```

重新采集 run1 的按轮次到达场景时，先激活已配置的 WSL 环境，再指定尚未使用的输出文件名。

```bash
python experiments/benchmark_scheduler_policy.py --model-path ../../models/Qwen3-0.6B --arrival-mode step --max-consecutive-prefills 0 --output experiments/results/policy_0_step_repeat.json
python experiments/benchmark_scheduler_policy.py --model-path ../../models/Qwen3-0.6B --arrival-mode step --max-consecutive-prefills 2 --output experiments/results/policy_2_step_repeat.json
```

run1 每种策略记录五轮，使用短序列和单一运行顺序。GPU 温度和频率未作为控制变量。

## 固定时间到达实验

TTFT 从计划到达时刻起算，包含请求到达后等待引擎下一次轮询的时间。请求总延迟先在每轮内计算 `TTFT + sum(ITL)`，再取五轮中位数。

### 调度诊断

| 批次 | A / B / C 计划到达（ms） | 策略 2 的 trace（五轮） |
|---|---|---|
| run6 | 0 / 40 / 75 | A 在 B 提交前自然 decode，计数清零。之后 `force_decode=True` 时 waiting 已空，策略没有改变所选批次。 |
| run7 | 0 / 10 / 50 | A、B 连续 prefill 后，C 已在 waiting。五轮都先执行 A/B decode，下一步才执行 C prefill。策略 0 的五轮均先执行 C prefill。 |

run6 和 run7 用于追踪批次顺序。正式延迟对照使用关闭 trace 的 run8、run9。

### 延迟对照

两批均使用 `0 / 10 / 50 ms` 到达计划，每种策略预热 2 轮、记录 5 轮。每批两份文件的配置和源码摘要一致，策略参数除外。下表比较各批中策略 2 相对策略 0 的中位数变化，两批样本分别统计。

| 指标 | run8（0 → 2） | run9（2 → 0） |
|---|---:|---:|
| A 首次 ITL | −39.34% | −35.21% |
| B 首次 ITL | −54.98% | −51.91% |
| A/B 第二次 ITL | +98.38% | +103.07% |
| C TTFT | +40.96% | +57.40% |
| A 请求总延迟 | −5.83% | −1.67% |
| B 请求总延迟 | −6.12% | −1.76% |
| C 请求总延迟 | −6.64% | −1.24% |

A/B 更早拿到第二个 token，随后等待 C prefill，拉长了到第三个 token 的间隔。C 的 prefill 推迟一轮，首 token 到得更晚。两批都重现了这个取舍。总延迟中位数在 run8 下降 5.83%–6.64%，在 run9 下降 1.24%–1.76%。降幅随批次变化，结论限于这组短请求负载。两次运行顺序提供了复查依据，但无法单独分离顺序与 GPU 状态波动的影响。

- run8 原始结果：[策略 0](policy_0_run8.json)、[策略 2](policy_2_run8.json)。
- run9 原始结果：[策略 0](policy_0_run9.json)、[策略 2](policy_2_run9.json)。
