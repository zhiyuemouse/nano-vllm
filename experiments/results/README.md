# 调度延迟实验记录

## 数据来源

- `policy_0_run1.json`：2026-11-06 19:20:33 +08:00 开始。
- `policy_2_run1.json`：2026-11-06 19:22:11 +08:00 开始。
- 顺序为 0 → 2，每种策略独立进程运行，预热 2 轮，正式记录 5 轮。
- 两份文件包含环境、prompt token IDs、启动及采样参数。请求按 step 0、1、2 到达，各生成 4 个 token。
- `policy_0.json`、`policy_2.json` 为此前手工保存的探索数据，缺少完整元数据；本说明和统计入口使用带 `run1` 的两份文件，不混合批次。

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

在 Nano-vLLM 仓库根目录执行统计，无需 GPU：

```bash
python experiments/summarize_scheduler_policy.py
```

重新采集需激活已配置的 WSL 环境，并使用新的文件名；保存函数拒绝覆盖旧文件：

```bash
python experiments/benchmark_scheduler_policy.py --model-path ../../models/Qwen3-0.6B --max-consecutive-prefills 0 --output experiments/results/policy_0_run2.json
python experiments/benchmark_scheduler_policy.py --model-path ../../models/Qwen3-0.6B --max-consecutive-prefills 2 --output experiments/results/policy_2_run2.json
```

统计脚本当前固定读取 run1；分析新批次时需显式调整输入文件。本批仅有五轮、短序列、按调度轮次到达且单一运行顺序，未控制温度/频率等随时间变化因素，不代表真实在线负载或统计显著收益。
