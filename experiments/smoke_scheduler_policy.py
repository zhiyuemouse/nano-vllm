import argparse
from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

def test_scheduler_policy_gpu_smoke(args):
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
        print("类型 : ", "prefill" if is_prefill else "decode")
        print("请求 : ", [seq.seq_id for seq in seqs])
        print("连续 prefill : ", llm.scheduler.consecutive_prefills)
        return seqs, is_prefill
    llm.scheduler.schedule = traced_schedule
    completed = {}

    # 第 0 轮 step 前：加入 A
    llm.add_request(A_prompt_ids, sampling_params)
    A_seq_id = llm.scheduler.waiting[-1].seq_id
    outputs, num_tokens = llm.step()
    for seq_id, token_ids in outputs:
        assert seq_id not in completed, "同一请求重复完成"
        completed[seq_id] = token_ids
    # 第 1 轮 step 前：加入 B
    llm.add_request(B_prompt_ids, sampling_params)
    B_seq_id = llm.scheduler.waiting[-1].seq_id
    outputs, num_tokens = llm.step()
    for seq_id, token_ids in outputs:
        assert seq_id not in completed, "同一请求重复完成"
        completed[seq_id] = token_ids
    # 第 2 轮 step 前：加入 C
    llm.add_request(C_prompt_ids, sampling_params)
    C_seq_id = llm.scheduler.waiting[-1].seq_id
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

    # 以下内容为我自己想看的输出，所以进行了打印
    # A_seq = A_prompt_ids + completed[A_seq_id]
    # B_seq = B_prompt_ids + completed[B_seq_id]
    # C_seq = C_prompt_ids + completed[C_seq_id]

    # A_out = tokenizer.decode(
    #     A_seq,
    #     skip_special_tokens=True
    # )
    # print(A_out)
    # print('#' * 10)
    # B_out = tokenizer.decode(
    #     B_seq,
    #     skip_special_tokens=True
    # )
    # print(B_out)
    # print('#' * 10)
    # C_out = tokenizer.decode(
    #     C_seq,
    #     skip_special_tokens=True
    # )
    # print(C_out)
    # print('#' * 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-consecutive-prefills", type=int, default=0)
    args = parser.parse_args()
    test_scheduler_policy_gpu_smoke(args)