import argparse
from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from nanovllm.utils.context import reset_context, set_context, get_context
from unittest.mock import patch

from nanovllm.models import qwen3


def run_kv_block_boundary_decode(
        llm,
        prefix_ids: list[int],
        forced_decode_token: int | None = None,
        check_triton_hits=False
    ):
    def run_stage(input_ids, positions, is_prefill):
        if not check_triton_hits or (not is_prefill and not runner.enforce_eager):
            return runner.run_model(input_ids, positions, is_prefill)

        stage = "prefill" if is_prefill else "decode"
        expected_hits = len(runner.model.model.layers)

        with patch.object(
            qwen3,
            "fused_add_rms_norm",
            wraps=qwen3.fused_add_rms_norm,
        ) as fused_norm:
            logits = runner.run_model(input_ids, positions, is_prefill)

        hits = fused_norm.call_count
        assert hits == expected_hits, (
            f"{stage}: Triton post-attention RMSNorm 命中 {hits}/{expected_hits} 层"
        )
        print(f"{stage}: Triton hits={hits}/{expected_hits}")
        return logits
    scheduler = llm.scheduler
    runner = llm.model_runner
    block_size = runner.block_size
    assert len(prefix_ids) == block_size
    llm.add_request(
        prefix_ids,
        SamplingParams(temperature=0.6, max_tokens=2, ignore_eos=True),
    )
    # prefill
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and len(seqs) == 1
    seq = seqs[0]
    assert seq.num_scheduled_tokens == block_size
    assert len(seq.block_table) == 1
    first_block_id = seq.block_table[0]

    try:
        input_ids, positions = runner.prepare_prefill(seqs)
        prefill_logits = run_stage(input_ids, positions, True)[0].float().cpu()
    finally:
        reset_context()
    prefill_top1 = prefill_logits.argmax().item()
    decode_token = prefill_top1 if forced_decode_token is None else forced_decode_token
    scheduler.postprocess(seqs, [decode_token], True)

    # decode
    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill and len(seqs) == 1 and seqs[0] is seq
    assert len(seq.block_table) == 2
    assert seq.block_table[0] == first_block_id
    second_block_id = seq.block_table[1]
    try:
        input_ids, positions = runner.prepare_decode(seqs)
        context = get_context()
        assert input_ids.tolist() == [decode_token]
        assert positions.tolist() == [block_size]
        assert context.context_lens.tolist() == [block_size + 1]
        assert context.block_tables[0, :2].tolist() == seq.block_table
        assert context.slot_mapping.tolist() == [second_block_id * block_size]

        decode_logits = run_stage(input_ids, positions, False)[0].float().cpu()
    finally:
        reset_context()

    scheduler.postprocess(seqs, [decode_logits.argmax().item()], False)
    assert seq.is_finished

    return prefill_logits, decode_logits, decode_token

def run_baseline_use_scheduler(model_path, prompt_ids, output_path, cuda_graph):
    llm = LLM(
        model_path,
        enforce_eager=not cuda_graph,
        use_triton_post_attn_norm=False,
    )
    BLOCK_SIZE = llm.model_runner.block_size
    prompt_ids = (prompt_ids * (BLOCK_SIZE // len(prompt_ids) + 1))[:BLOCK_SIZE]
    assert len(prompt_ids) == BLOCK_SIZE
    prefill_logits, decode_logits, decode_token = run_kv_block_boundary_decode(
        llm,
        prompt_ids,
        None,
    )
    record = {
        "prefix_ids": list(prompt_ids),
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
        "decode_token": decode_token,
    }

    torch.save(record, output_path)

def run_triton_use_scheduler(model_path, prompt_ids, output_path, cuda_graph):
    llm = LLM(
        model_path,
        enforce_eager=not cuda_graph,
        use_triton_post_attn_norm=True,
    )
    BLOCK_SIZE = llm.model_runner.block_size
    prompt_ids = (prompt_ids * (BLOCK_SIZE // len(prompt_ids) + 1))[:BLOCK_SIZE]
    assert len(prompt_ids) == BLOCK_SIZE

    record = torch.load(output_path, map_location="cpu")
    assert list(prompt_ids) == record["prefix_ids"]
    prefill_logits, decode_logits, _ = run_kv_block_boundary_decode(
        llm,
        prompt_ids,
        forced_decode_token=record["decode_token"],
        check_triton_hits=True,
    )

    for stage, actual in (
        ("prefill", prefill_logits),
        ("decode", decode_logits),
    ):
        expected = record[f"{stage}_logits"]
        assert expected.shape == actual.shape

        max_diff, diff_token = (expected - actual).abs().max(dim=0)
        top2 = expected.topk(2).values
        baseline_top1 = expected.argmax().item()
        triton_top1 = actual.argmax().item()
        margin = (top2[0] - top2[1]).item()

        print(
            f"{stage} max_abs_diff={max_diff.item():.8g} "
            f"max_diff_token_id={diff_token.item()} "
            f"baseline_top1={baseline_top1} triton_top1={triton_top1} "
            f"top1_match={baseline_top1 == triton_top1} "
            f"baseline_margin={margin:.8g}"
        )

def compare_rmsnorm_logits_at_kv_block_boundary(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    A = "用一句话介绍北京。"
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{A}"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    if (args.run_triton):
        run_triton_use_scheduler(args.model_path, prompt_ids, args.output_path, args.cuda_graph)
    else:
        run_baseline_use_scheduler(args.model_path, prompt_ids, args.output_path, args.cuda_graph)

@torch.inference_mode()
def last_logits_kvcache(
    llm,
    prefix_ids: list[int],
    kvcache: bool = False,
    is_prefill: bool = False,
) -> torch.Tensor:
    if not prefix_ids:
        raise ValueError("prefix_ids must contain at least one token")

    runner = llm.model_runner
    device = next(runner.model.parameters()).device
    length = len(prefix_ids)
    prefill = not kvcache or is_prefill

    block_id = 0
    slot_start = block_id * runner.block_size

    if kvcache:
        if kvcache:
            expected_length = runner.block_size if prefill else runner.block_size + 1
            if length != expected_length:
                raise ValueError(f"expected {expected_length} tokens, got {length}")

    try:
        if prefill:
            input_ids = torch.tensor(prefix_ids, dtype=torch.int64, device=device)
            positions = torch.arange(length, dtype=torch.int64, device=device)
            cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=device)

            if kvcache:
                slot_mapping = torch.arange(
                    slot_start, slot_start + length,
                    dtype=torch.int32, device=device,
                )
            else:
                slot_mapping = torch.full(
                    (length,), -1, dtype=torch.int32, device=device,
                )

            set_context(
                is_prefill=True,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=length,
                max_seqlen_k=length,
                slot_mapping=slot_mapping,
                block_tables=None,
            )
        else:
            if length < 2:
                raise ValueError("Decode expects the cached prompt plus one new token")

            # prefix_ids 包含完整 prompt 和本轮新增的一个 token。
            prompt_len = length - 1
            input_ids = torch.tensor(
                prefix_ids[-1:], dtype=torch.int64, device=device,
            )
            positions = torch.tensor(
                [prompt_len], dtype=torch.int64, device=device,
            )

            set_context(
                is_prefill=False,
                slot_mapping=torch.tensor(
                    [slot_start + prompt_len], dtype=torch.int32, device=device,
                ),
                context_lens=torch.tensor(
                    [length], dtype=torch.int32, device=device,
                ),
                block_tables=torch.tensor(
                    [[block_id, block_id + 1]], dtype=torch.int32, device=device,
                ),
            )

        logits = runner.run_model(input_ids, positions, is_prefill=prefill)
        return logits[0].detach().float().cpu()
    finally:
        reset_context()
    
def run_baseline(model_path, prompt_ids, steps, output_path):
    llm = LLM(
        model_path,
        enforce_eager=True,
        use_triton_post_attn_norm=False,
    )
    BLOCK_SIZE = llm.model_runner.block_size
    prompt_ids = (prompt_ids * (BLOCK_SIZE // len(prompt_ids) + 1))[:BLOCK_SIZE]
    assert len(prompt_ids) == BLOCK_SIZE

    prefix = list(prompt_ids)
    records = []
    for step in range(steps):
        logits = last_logits_kvcache(llm, prefix, kvcache=True, is_prefill=(step == 0))
        top1 = int(logits.argmax().item())

        records.append({
            "step": step,
            "prefix_ids": prefix.copy(),
            "logits": logits,
            "top1": top1,
        })
        prefix.append(top1)

    torch.save(records, output_path)

def run_triton(model_path, prompt_ids, steps, output_path):
    llm = LLM(
        model_path,
        enforce_eager=True,
        use_triton_post_attn_norm=True,
    )
    BLOCK_SIZE = llm.model_runner.block_size
    prompt_ids = (prompt_ids * (BLOCK_SIZE // len(prompt_ids) + 1))[:BLOCK_SIZE]
    assert len(prompt_ids) == BLOCK_SIZE

    records = torch.load(output_path)
    assert len(records) == steps
    prefix = list(prompt_ids)
    expected_hits = len(llm.model_runner.model.model.layers)

    for step, record in enumerate(records):
        assert record["step"] == step
        assert prefix == record["prefix_ids"], f"step={step}: prefix 不一致"

        with patch.object(
            qwen3, "fused_add_rms_norm", wraps=qwen3.fused_add_rms_norm
        ) as fused_norm:
            triton_logits = last_logits_kvcache(
                llm, prefix, kvcache=True, is_prefill=(step == 0),
            )

        hits = fused_norm.call_count
        assert hits == expected_hits, (
            f"step={step}: Triton post-attention RMSNorm 命中 {hits}/{expected_hits} 层"
        )

        base_logits = record["logits"]
        assert base_logits.shape == triton_logits.shape
        max_abs_diff, token_id = (base_logits - triton_logits).abs().max(dim=0)
        base_top2 = base_logits.topk(2).values
        margin = (base_top2[0] - base_top2[1]).item()
        base_top1 = record["top1"]
        triton_top1 = triton_logits.argmax().item()

        print(
            f"{'prefill' if step == 0 else 'decode'} step={step} prefix_len={len(prefix)} "
            f"hits={hits}/{expected_hits} "
            f"max_abs_diff={max_abs_diff.item():.8g} "
            f"max_diff_token_id={token_id.item()} "
            f"baseline_top1={base_top1} triton_top1={triton_top1} "
            f"top1_match={base_top1 == triton_top1} baseline_margin={margin:.8g}"
        )

        prefix.append(base_top1)

def run_prefill_logits_comparison(args):    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    A = "用一句话介绍北京。"
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{A}"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    
    if (args.run_triton):
        run_triton(args.model_path, prompt_ids, 2, args.output_path)
    else:
        run_baseline(args.model_path, prompt_ids, 2, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--run-triton", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    # run_prefill_logits_comparison(args)
    compare_rmsnorm_logits_at_kv_block_boundary(args)
    