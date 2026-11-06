import math
import argparse
from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import torch
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess
import tempfile
from summarize_scheduler_policy import summarize_trials
from importlib.metadata import version

class RequestTiming:
    def __init__(self, arrival_time: float):
        self.arrival_time = arrival_time
        self.token_times: list[float] = []

    def record_token(self, timestamp: float) -> None:
        self.token_times.append(timestamp)

    def ttft_ms(self) -> float:
        """尚无 token 时抛出 ValueError。"""
        if self.token_times == []: raise ValueError("尚无 token")
        return (self.token_times[0] - self.arrival_time) * 1000 # ms

    def itls_ms(self) -> list[float]:
        """少于两个 token 时返回空列表。"""
        if len(self.token_times) < 2: return []
        return [(self.token_times[i] - self.token_times[i - 1]) * 1000 for i in range(1, len(self.token_times))]

def test_request_timing():
    arrival_time = 10.0
    token_times  = [10.2, 10.3, 10.5]
    request_timing = RequestTiming(arrival_time)
    for i in token_times:
        request_timing.record_token(i)
    TTFT = request_timing.ttft_ms()
    ITL = request_timing.itls_ms()
    assert math.isclose(TTFT, 200.0, abs_tol=1e-6)
    expected = [100.0, 200.0]
    assert len(ITL) == len(expected)
    for i in range(len(ITL)):
        assert math.isclose(ITL[i], expected[i], abs_tol=1e-6)
    print("test_request_timing PASS")

def test_token_timing_gpu(args):
    # 复用 smoke 脚本的模型配置与请求到达流程
    # 加入 timings 字典
    # 包装 postprocess()
    # 执行请求并检查时间戳数量
    model_path = args.model_path
    print(f"model_path : {model_path}")
    sampling_params = SamplingParams(0.6, 4, True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    A = "你好吗"
    B = "1+1等于几？"
    C = "用一句话介绍北京。"
    A_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{A}"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    B_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{B}"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    C_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{C}"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=2,
        max_model_len=128,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.70,
        max_consecutive_prefills=args.max_consecutive_prefills
    )

    original_schedule = llm.scheduler.schedule
    def traced_schedule():
        seqs, is_prefill = original_schedule()
        # print("类型 : ", "prefill" if is_prefill else "decode")
        # print("请求 : ", [seq.seq_id for seq in seqs])
        # print("连续 prefill : ", llm.scheduler.consecutive_prefills)
        return seqs, is_prefill
    llm.scheduler.schedule = traced_schedule

    timings: dict[int, RequestTiming] = {}
    original_postprocess = llm.scheduler.postprocess
    def trace_postprocess(seqs, token_ids, is_prefill):
        before_counts = {seq.seq_id: seq.num_tokens for seq in seqs}

        original_postprocess(seqs, token_ids, is_prefill)
        torch.cuda.synchronize()
        perf_counter = time.perf_counter()
        for seq in seqs:
            num_tokens_changed = seq.num_tokens - before_counts[seq.seq_id]
            assert num_tokens_changed == 0 or num_tokens_changed == 1
            if num_tokens_changed == 1:
                timings[seq.seq_id].record_token(perf_counter)
    llm.scheduler.postprocess = trace_postprocess

    completed = {}
    # 第 0 轮 step 前：加入 A
    arrival_time = time.perf_counter()
    llm.add_request(A_prompt_ids, sampling_params)
    A_seq_id = llm.scheduler.waiting[-1].seq_id
    timings[A_seq_id] = RequestTiming(arrival_time)
    outputs, num_tokens = llm.step()
    for seq_id, token_ids in outputs:
        assert seq_id not in completed, "同一请求重复完成"
        completed[seq_id] = token_ids
    # 第 1 轮 step 前：加入 B
    arrival_time = time.perf_counter()
    llm.add_request(B_prompt_ids, sampling_params)
    B_seq_id = llm.scheduler.waiting[-1].seq_id
    timings[B_seq_id] = RequestTiming(arrival_time)
    outputs, num_tokens = llm.step()
    for seq_id, token_ids in outputs:
        assert seq_id not in completed, "同一请求重复完成"
        completed[seq_id] = token_ids
    # 第 2 轮 step 前：加入 C
    arrival_time = time.perf_counter()
    llm.add_request(C_prompt_ids, sampling_params)
    C_seq_id = llm.scheduler.waiting[-1].seq_id
    timings[C_seq_id] = RequestTiming(arrival_time)
    outputs, num_tokens = llm.step()
    for seq_id, token_ids in outputs:
        assert seq_id not in completed, "同一请求重复完成"
        completed[seq_id] = token_ids
    # 之后不再加入请求，持续 step，直到全部完成

    for step_index in range(3, 30): # 设置最多 30 轮的保护，超过就报错，避免异常时无限循环。
        outputs, num_tokens = llm.step()
        for seq_id, token_ids in outputs:
            assert seq_id not in completed, "同一请求重复完成"
            completed[seq_id] = token_ids
        if llm.is_finished():
            break
    else:
        raise RuntimeError("执行 30 轮后仍未全部完成")

    assert len(completed[A_seq_id]) == 4
    assert len(completed[B_seq_id]) == 4
    assert len(completed[C_seq_id]) == 4

    for key, values in timings.items():
        assert len(values.token_times) == 4 # 时间戳数量等于实际生成数，均为 4。
        assert len(values.itls_ms()) == 3 # 每个请求有 3 个 ITL。
        assert values.token_times[0] >= values.arrival_time # 首 token 时间不早于到达时间
        for i in range(1, len(values.token_times)):
            assert values.token_times[i] >= values.token_times[i - 1] # 后续时间戳单调不减。
    assert set(completed) == set(timings) # 完成请求集合与登记的请求集合一致
    # 打印每个请求的标签、记录数、TTFT 和 ITL 列表
    seq_id2ID = {
        A_seq_id: "A",
        B_seq_id: "B",
        C_seq_id: "C",
    }
    for seq_id in [A_seq_id, B_seq_id, C_seq_id]:
        print(
            f"{seq_id2ID[seq_id]}",
            "记录数 : ", len(timings[seq_id].token_times),
            "TTFT(ms) : ", timings[seq_id].ttft_ms(),
            "ITL(ms) : ", timings[seq_id].itls_ms(),
        )
    print("token timing instrumentation PASS")

def run_trial(llm, prompt_ids_by_label, sampling_params) -> dict:
    """
    按第 0、1、2 轮分别加入 A、B、C。
    保留时间戳数量、单调性、重复完成及轮数保护检查。

    返回：
        {
            "A": {"ttft_ms": ..., "itls_ms": [...]},
            "B": {...},
            "C": {...},
        }
    """
    # 每轮开始前，由你完成这些操作
    assert list(llm.scheduler.waiting) == [] and list(llm.scheduler.running) == [] # 断言 waiting、running 均为空。
    assert list(llm.scheduler.block_manager.used_block_ids) == [] and len(list(llm.scheduler.block_manager.free_block_ids)) == len(list(llm.scheduler.block_manager.blocks)) # 断言所有物理块均已归还
    llm.scheduler.consecutive_prefills = 0 # 将 consecutive_prefills 重置为 0，清空本轮时间记录
    torch.manual_seed(42) # 设置相同随机种子，例如 torch.manual_seed(42)
    torch.cuda.synchronize() # 同步 GPU，再开始登记请求到达时间。
    assert not llm.scheduler.block_manager.hash_to_block_id # 断言这些长度条件不会形成可复用的完整前缀块
    for ids in prompt_ids_by_label.values():
        assert len(ids) + sampling_params.max_tokens < llm.scheduler.block_size # 显式检查短请求条件

    timings: dict[int, RequestTiming] = {}
    original_postprocess = llm.scheduler.postprocess
    def trace_postprocess(seqs, token_ids, is_prefill):
        before_counts = {seq.seq_id: seq.num_tokens for seq in seqs}

        original_postprocess(seqs, token_ids, is_prefill)
        torch.cuda.synchronize()
        perf_counter = time.perf_counter()
        for seq in seqs:
            num_tokens_changed = seq.num_tokens - before_counts[seq.seq_id]
            assert num_tokens_changed == 0 or num_tokens_changed == 1
            if num_tokens_changed == 1:
                timings[seq.seq_id].record_token(perf_counter)
    llm.scheduler.postprocess = trace_postprocess
    try:
        completed = {}
        # 第 0 轮 step 前：加入 A
        arrival_time = time.perf_counter()
        A_prompt_ids = prompt_ids_by_label['A']
        llm.add_request(A_prompt_ids, sampling_params)
        A_seq_id = llm.scheduler.waiting[-1].seq_id
        timings[A_seq_id] = RequestTiming(arrival_time)
        outputs, num_tokens = llm.step()
        for seq_id, token_ids in outputs:
            assert seq_id not in completed, "同一请求重复完成"
            completed[seq_id] = token_ids
        # 第 1 轮 step 前：加入 B
        arrival_time = time.perf_counter()
        B_prompt_ids = prompt_ids_by_label['B']
        llm.add_request(B_prompt_ids, sampling_params)
        B_seq_id = llm.scheduler.waiting[-1].seq_id
        timings[B_seq_id] = RequestTiming(arrival_time)
        outputs, num_tokens = llm.step()
        for seq_id, token_ids in outputs:
            assert seq_id not in completed, "同一请求重复完成"
            completed[seq_id] = token_ids
        # 第 2 轮 step 前：加入 C
        arrival_time = time.perf_counter()
        C_prompt_ids = prompt_ids_by_label['C']
        llm.add_request(C_prompt_ids, sampling_params)
        C_seq_id = llm.scheduler.waiting[-1].seq_id
        timings[C_seq_id] = RequestTiming(arrival_time)
        outputs, num_tokens = llm.step()
        for seq_id, token_ids in outputs:
            assert seq_id not in completed, "同一请求重复完成"
            completed[seq_id] = token_ids
        # 之后不再加入请求，持续 step，直到全部完成

        for step_index in range(3, 30): # 设置最多 30 轮的保护，超过就报错，避免异常时无限循环。
            outputs, num_tokens = llm.step()
            for seq_id, token_ids in outputs:
                assert seq_id not in completed, "同一请求重复完成"
                completed[seq_id] = token_ids
            if llm.is_finished():
                break
        else:
            raise RuntimeError("执行 30 轮后仍未全部完成")

        assert len(completed[A_seq_id]) == 4
        assert len(completed[B_seq_id]) == 4
        assert len(completed[C_seq_id]) == 4

        for key, values in timings.items():
            assert len(values.token_times) == 4 # 时间戳数量等于实际生成数，均为 4。
            assert len(values.itls_ms()) == 3 # 每个请求有 3 个 ITL。
            assert values.token_times[0] >= values.arrival_time # 首 token 时间不早于到达时间
            for i in range(1, len(values.token_times)):
                assert values.token_times[i] >= values.token_times[i - 1] # 后续时间戳单调不减。
        assert set(completed) == set(timings) # 完成请求集合与登记的请求集合一致
        # 打印每个请求的标签、记录数、TTFT 和 ITL 列表
        seq_id2ID = {
            A_seq_id: "A",
            B_seq_id: "B",
            C_seq_id: "C",
        }
        ans = {}
        for seq_id in [A_seq_id, B_seq_id, C_seq_id]:
            ans[seq_id2ID[seq_id]] = {"ttft_ms": timings[seq_id].ttft_ms(), "itls_ms": timings[seq_id].itls_ms()}
    finally:
        llm.scheduler.postprocess = original_postprocess # 不要在每轮把新包装器套到旧包装器上。可以只安装一次、每轮清空记录
    return ans

def save_benchmark_result(
    output_path: str,
    metadata: dict,
    trials: list[dict],
) -> None:
    """将实验信息和五轮原始结果保存为 JSON；文件已存在时拒绝覆盖。"""
    save_json = {
        "metadata": metadata,
        "trials": trials
    }
    with open(output_path, "x", encoding="utf-8") as f:
        json.dump(save_json, f, ensure_ascii=False, indent=2)

def test_save_benchmark_result():
    metadata = {"policy_limit": 2, "seed": 42}
    trials = [
        {
            label: {
                "ttft_ms": 200.0,
                "itls_ms": [100.0, 200.0, 0.0],
            }
            for label in ["A", "B", "C"]
        }
        for _ in range(5)
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "result.json"

        # 1. 保存后读回，检查内容一致
        save_benchmark_result(str(output_path), metadata, trials)

        with output_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        assert payload == {
            "metadata": metadata,
            "trials": trials,
        }

        # 2. 相同路径再次保存，必须拒绝覆盖
        try:
            save_benchmark_result(str(output_path), metadata, trials)
        except FileExistsError:
            pass
        else:
            raise AssertionError("已有结果文件不应被覆盖")

        # 3. 将新结构中的 trials 交给现有统计函数
        summary = summarize_trials(payload["trials"])
        assert summary["A"]["request_latency_median_ms"] == 500.0

    print("test_save_benchmark_result PASS")

def benchmark(args):
    started_at = datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).isoformat(timespec="seconds")
    repo_root = Path(__file__).resolve().parents[1]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()

    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        text=True,
    )
    git_dirty = bool(status.strip())


    # 在这里加载一次模型、准备 prompts 等
    model_path = args.model_path
    print(f"model_path : {model_path}")
    sampling_kwargs = {
        "temperature": 0.6,
        "max_tokens": 4,
        "ignore_eos": True,
    }

    llm_kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "max_num_seqs": 2,
        "max_model_len": 128,
        "max_num_batched_tokens": 128,
        "gpu_memory_utilization": 0.70,
        "max_consecutive_prefills": args.max_consecutive_prefills,
    }
    sampling_params = SamplingParams(**sampling_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    A = "你好吗"
    B = "1+1等于几？"
    C = "用一句话介绍北京。"
    A_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{A}"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    B_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{B}"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    C_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{C}"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids_by_label = {
        'A': A_prompt_ids,
        'B': B_prompt_ids,
        'C': C_prompt_ids,
    }
    # 创建一次 LLM
    llm = LLM(model_path, **llm_kwargs)
    # 运行 2 次 run_trial：执行全部校验，但丢弃测量结果
    for i in range(2):
        benchmark_output = run_trial(llm, prompt_ids_by_label, sampling_params)
    # 运行 5 次 run_trial：保存返回结果
    results = []
    for i in range(5):
        benchmark_output = run_trial(llm, prompt_ids_by_label, sampling_params)
        results.append(benchmark_output)
    assert len(results) == 5
    for result in results: # 检查最终保存的数据结构
        assert set(result) == {"A", "B", "C"}
        for metrics in result.values():
            assert len(metrics["itls_ms"]) == 3
    # 全部结束后，再逐轮打印 A/B/C 的 TTFT 和 ITL
    for i in range(5):
        benchmark_output = results[i]
        print(f"第 {i + 1} 轮")
        for key, values in benchmark_output.items():
            print(f"{key} : ", f"ttft_ms : {values['ttft_ms']}", f"itls_ms : {values['itls_ms']}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "started_at": started_at,
        "policy_limit": args.max_consecutive_prefills,
        "warmup": 2,
        "repeats": 5,
        "seed": 42,
        "arrival_mode": "step",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "versions": {
            name: version(name)
            for name in ["torch", "transformers", "triton", "flash-attn", "xxhash"]
        },
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "prompt_token_ids": prompt_ids_by_label,
        "llm_config": llm_kwargs.copy(),
        "sampling_config": sampling_kwargs.copy(),
    }
    save_benchmark_result(output_path, metadata, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-consecutive-prefills", type=int, default=0)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    test_request_timing()
    # test_token_timing_gpu(args)
    benchmark(args)
    # test_save_benchmark_result()