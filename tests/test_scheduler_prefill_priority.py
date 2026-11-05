from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

def test_prefill_arrivals_delay_decode():
    config = Config(
        model="../../models/Qwen3-0.6B",
        max_num_seqs=2,
        max_num_batched_tokens=128,
        max_model_len=128,
        kvcache_block_size=256,
        num_kvcache_blocks=32,
    )
    config.eos = -1

    scheduler = Scheduler(config)
    A_sampling_params = SamplingParams(1, 16, False)
    A_token_ids = [1] * 14
    A = Sequence(token_ids=A_token_ids, sampling_params=A_sampling_params)
    scheduler.add(A)
    scheduled_A, is_prefill_A = scheduler.schedule()
    assert len(scheduled_A) == 1 and scheduled_A[0] == A and is_prefill_A == True # 确认只选中 A 且 is_prefill=True
    scheduler.postprocess(scheduled_A, [100], is_prefill_A)
    # 保存 A 的 num_tokens 和 num_cached_tokens
    A_num_tokens = A.num_tokens
    A_num_cached_tokens = A.num_cached_tokens

    for i in range(3):
        B_token_ids = [10] * 14
        B_sampling_params = SamplingParams(1, 16, False)
        B = Sequence(token_ids=B_token_ids, sampling_params=B_sampling_params)
        scheduler.add(B)
        scheduled_B, is_prefill_B = scheduler.schedule()
        assert len(scheduled_B) == 1 and scheduled_B[0] == B and is_prefill_B == True # 断言本轮只选中新 B，is_prefill=True，A 未被选中
        scheduler.postprocess(scheduled_B, [100], is_prefill_B)
        assert A_num_tokens == A.num_tokens and A_num_cached_tokens == A.num_cached_tokens # 断言 A 的两个计数仍与保存值相同

    print("test_prefill_arrivals_delay_decode PASS")

def test_prefill_limit_allows_decode():
    config = Config(
        model="../../models/Qwen3-0.6B",
        max_num_seqs=2,
        max_num_batched_tokens=128,
        max_model_len=128,
        kvcache_block_size=256,
        num_kvcache_blocks=32,
        max_consecutive_prefills=2,
    )
    config.eos = -1

    scheduler = Scheduler(config)
    # A 完成一次 prefill：计数为 1
    sampling_params = SamplingParams(1, 16, False)
    A_token_ids = [1] * 14
    A = Sequence(token_ids=A_token_ids, sampling_params=sampling_params)
    scheduler.add(A)
    scheduled_A, is_prefill_A = scheduler.schedule()
    scheduler.postprocess(scheduled_A, [100], is_prefill_A)
    assert scheduler.consecutive_prefills == 1

    # B 到达并完成 prefill：计数为 2
    B_token_ids = [1] * 14
    B = Sequence(token_ids=B_token_ids, sampling_params=sampling_params)
    scheduler.add(B)
    scheduled_B, is_prefill_B = scheduler.schedule()
    scheduler.postprocess(scheduled_B, [100], is_prefill_B)
    B_num_tokens = B.num_tokens
    B_num_cached_tokens = B.num_cached_tokens
    assert scheduler.consecutive_prefills == 2

    # C 到达
    C_token_ids = [1] * 14
    C = Sequence(token_ids=C_token_ids, sampling_params=sampling_params)
    scheduler.add(C)
    # 下一次 schedule() 应返回 decode，选中 [A, B]，C 留在 waiting
    scheduled_C, is_prefill_C = scheduler.schedule()
    assert scheduled_C == [A, B]
    assert is_prefill_C == False
    assert list(scheduler.waiting) == [C]

    # 模拟后处理后
    scheduler.postprocess(scheduled_C, [100, 100], is_prefill_C)
    assert A.num_tokens == 16 and A.num_cached_tokens == 15 # A 为 16 / 15
    assert B.num_tokens == B_num_tokens + 1 and B.num_cached_tokens == B_num_cached_tokens + 1 # B 的 num_tokens、`num_cached_tokens` 各增加 1
    assert scheduler.consecutive_prefills == 0 # 计数清零

    # 再次 schedule() 应选中 C 做 prefill，计数为 1
    scheduled_C, is_prefill_C = scheduler.schedule()
    assert scheduled_C == [C] and is_prefill_C == True and scheduler.consecutive_prefills == 1
    scheduler.postprocess(scheduled_C, [100], is_prefill_C)

    # 策略默认关闭
    config.max_consecutive_prefills = 0
    scheduler = Scheduler(config)
    A_token_ids = [1] * 14
    A = Sequence(token_ids=A_token_ids, sampling_params=sampling_params)
    scheduler.add(A)

    B_token_ids = [1] * 14
    B = Sequence(token_ids=B_token_ids, sampling_params=sampling_params)
    scheduler.add(B)
    scheduled_B, is_prefill_B = scheduler.schedule()

    assert scheduled_B == [A, B] and is_prefill_B == True
    scheduler.postprocess(scheduled_B, [100, 100], is_prefill_B)


    print("test_prefill_limit_allows_decode PASS")

def test_prefill_limit_with_empty_running():
    # 配置 max_consecutive_prefills=1，缓存和 token 预算保持充足
    config = Config(
        model="../../models/Qwen3-0.6B",
        max_num_seqs=2,
        max_num_batched_tokens=128,
        max_model_len=128,
        kvcache_block_size=256,
        num_kvcache_blocks=32,
        max_consecutive_prefills=1,
    )
    config.eos = -1

    scheduler = Scheduler(config)
    # 创建 A：短 prompt，`max_tokens=1`。
    sampling_params = SamplingParams(1, 1, False)
    A_token_ids = [1] * 14
    A = Sequence(token_ids=A_token_ids, sampling_params=sampling_params)
    # 调度 A 的 prefill，并用 [100] 调用 postprocess()
    scheduler.add(A)
    scheduled_A, is_prefill_A = scheduler.schedule()
    scheduler.postprocess(scheduled_A, [100], is_prefill_A)
    # A 达到生成上限而结束，此时断言
    assert A.is_finished == True # A.is_finished 为真
    assert not scheduler.running and not scheduler.waiting # running、`waiting` 均为空
    assert scheduler.consecutive_prefills == 1 # consecutive_prefills == 1
    # 创建 B，设置 max_tokens=16，加入调度器
    B_sampling_params = SamplingParams(1, 16, False)
    B_token_ids = [1] * 14
    B = Sequence(token_ids=B_token_ids, sampling_params=B_sampling_params)
    scheduler.add(B)
    # 再次调用 schedule()，断言
    scheduled_B, is_prefill_B = scheduler.schedule()
    assert scheduled_B == [B]
    assert is_prefill_B == True
    assert scheduler.consecutive_prefills == 2
    # 用 [100] 完成 B 的后处理，检查 B 的 token 数和缓存计数
    scheduler.postprocess(scheduled_B, [100], is_prefill_B)
    assert (B.num_tokens == len(B_token_ids) + 1) and B.num_cached_tokens == len(B_token_ids)
    print("test_prefill_limit_with_empty_running PASS")

def test_chunked_prefill_resumes_after_decode():
    config = Config(
        model="../../models/Qwen3-0.6B",
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=128,
        kvcache_block_size=256,
        num_kvcache_blocks=32,
        max_consecutive_prefills=2,
    )
    # 两个请求均设置 max_tokens=16，`eos=-1`
    config.eos = -1
    scheduler = Scheduler(config)

    sampling_params = SamplingParams(1, 16, False)
    # A：prompt 为 [1] * 4
    A_token_ids = [1] * 4
    A = Sequence(A_token_ids, sampling_params)
    # B：prompt 为 [2] * 18。
    B_token_ids = [2] * 18
    B = Sequence(B_token_ids, sampling_params)
    # 	加入 A，执行 A prefill
    scheduler.add(A)
    scheduled_A, is_prefill_A = scheduler.schedule()
    assert scheduled_A == [A]
    assert is_prefill_A == True
    assert A.num_scheduled_tokens == 4
    scheduler.postprocess(scheduled_A, [100], is_prefill_A)
    assert A.num_tokens == 5 and A.num_cached_tokens == 4
    assert scheduler.consecutive_prefills == 1
    # 加入 B，执行 B 第一块 prefill
    scheduler.add(B)
    scheduled_B, is_prefill_B = scheduler.schedule()
    assert scheduled_B == [B]
    assert is_prefill_B == True
    assert B.num_scheduled_tokens == 8
    scheduler.postprocess(scheduled_B, [100], is_prefill_B)
    assert B.status == SequenceStatus.WAITING
    assert B.completion_token_ids == []
    assert B.num_tokens == 18 and B.num_cached_tokens == 8
    assert scheduler.consecutive_prefills == 2
    B_block_table = B.block_table.copy() # 第 2 轮后保存 B 的块表副本
    B_num_cached_tokens = B.num_cached_tokens
    # 达到上限，执行 A decode
    scheduled_A, is_prefill_A = scheduler.schedule()
    assert scheduled_A == [A]
    assert is_prefill_A == False
    assert A.num_scheduled_tokens == 1
    scheduler.postprocess(scheduled_A, [100], is_prefill_A)
    assert A.num_tokens == 6 and A.num_cached_tokens == 5
    assert B.num_tokens == 18 and B.num_cached_tokens == 8
    assert B_block_table == B.block_table and B_num_cached_tokens == B.num_cached_tokens # 第 3 轮后检查它和缓存计数都没变
    assert scheduler.consecutive_prefills == 0
    # 恢复 B prefill
    scheduled_B, is_prefill_B = scheduler.schedule()
    assert scheduled_B == [B]
    assert is_prefill_B == True
    assert B.num_scheduled_tokens == 8
    scheduler.postprocess(scheduled_B, [100], is_prefill_B)
    assert B.status == SequenceStatus.WAITING
    assert B.completion_token_ids == []
    assert B.num_tokens == 18 and B.num_cached_tokens == 16
    assert scheduler.consecutive_prefills == 1
    # 	完成 B 剩余 prefill
    scheduled_B, is_prefill_B = scheduler.schedule()
    assert scheduled_B == [B]
    assert is_prefill_B == True
    assert B.num_scheduled_tokens == 2
    scheduler.postprocess(scheduled_B, [100], is_prefill_B)
    assert B.status == SequenceStatus.RUNNING
    assert len(B.completion_token_ids) == 1
    assert B.num_tokens == 19 and B.num_cached_tokens == 18
    assert scheduler.consecutive_prefills == 2

    print("test_chunked_prefill_resumes_after_decode PASS")

def test_prefill_allocation_failure_falls_back_to_decode():
    config = Config(
        model="../../models/Qwen3-0.6B",
        max_num_seqs=2,
        max_num_batched_tokens=256,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=2,
        max_consecutive_prefills=2,
    )
    # 显式设置 Sequence.block_size = config.kvcache_block_size，并保持 eos=-1
    config.eos = -1
    Sequence.block_size = config.kvcache_block_size
    sampling_params = SamplingParams(1, 16, False)
    scheduler = Scheduler(config)
    # A 的 prompt 长度为 256，`max_tokens=16`
    A_token_ids = [1] * 256
    A = Sequence(A_token_ids, sampling_params)
    # 调度并完成 A 的 prefill，模拟输出 [100]
    scheduler.add(A)
    scheduled_A, is_prefill_A = scheduler.schedule()
    assert scheduled_A == [A] and is_prefill_A == True
    scheduler.postprocess(scheduled_A, [100], is_prefill_A)
    assert A.num_tokens == 257 and A.num_cached_tokens == 256 # A：`num_tokens=257`、`num_cached_tokens=256`。
    assert A.status == SequenceStatus.RUNNING
    assert len(A.block_table) == 1 # A 在 running，占用 1 块；下次 decode 需要第 2 块。
    # 加入 B：prompt 长度为 300，`max_tokens=16`。
    B_token_ids = [2] * 300
    B = Sequence(B_token_ids, sampling_params)
    scheduler.add(B)
    # 调用 schedule()，检查：
    assert len(scheduler.block_manager.free_block_ids) == 1
    scheduled_B, is_prefill_B = scheduler.schedule()
    assert scheduled_B == [A] and is_prefill_B == False # 调度器转而执行 A 的 decode，为 A 分配剩余块。
    assert scheduler.consecutive_prefills == 0
    # 完成该轮后处理，记录 A、B 的状态和空闲块数
    scheduler.postprocess(scheduled_B, [100], is_prefill_B)
    assert A.num_tokens == 258 and A.num_cached_tokens == 257 and len(A.block_table) == 2
    assert B.status == SequenceStatus.WAITING
    assert B.block_table == [] and B.num_cached_tokens == 0
    assert len(scheduler.block_manager.free_block_ids) == 0
    print("test_prefill_allocation_failure_falls_back_to_decode PASS")

def test_prefill_continues_when_forced_decode_cannot_append():
    config = Config(
        model="../../models/Qwen3-0.6B",
        max_num_seqs=2,
        max_num_batched_tokens=256,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=3,
        max_consecutive_prefills=2,
    )
    # 保持 eos=-1，显式设置 Sequence.block_size，请求的 max_tokens=16
    config.eos = -1
    Sequence.block_size = config.kvcache_block_size
    sampling_params = SamplingParams(1, 16, False)
    scheduler = Scheduler(config)

    # A 使用 [1] * 256，完成一次 prefill，模拟输出 [100]
    A_token_ids = [1] * 256
    A = Sequence(A_token_ids, sampling_params)
    scheduler.add(A)
    scheduled_A, is_prefill_A = scheduler.schedule()
    scheduler.postprocess(scheduled_A, [100], is_prefill_A)
    assert A.num_tokens == 257 and A.num_cached_tokens == 256
    assert A.status == SequenceStatus.RUNNING and len(A.block_table) == 1 # A 为 257 / 256，在 running，占 1 块
    # B 使用 [2] * 300，加入并执行首段 prefill，模拟输出 [100]
    B_token_ids = [2] * 300
    B = Sequence(B_token_ids, sampling_params)
    scheduler.add(B)
    scheduled_B, is_prefill_B = scheduler.schedule()
    scheduler.postprocess(scheduled_B, [100], is_prefill_B)
    assert B.num_tokens == 300 and B.num_cached_tokens == 256
    assert B.status == SequenceStatus.WAITING and len(B.block_table) == 2
    assert len(scheduler.block_manager.free_block_ids) == 0
    assert scheduler.consecutive_prefills == 2
    # 保存 A 的完整 token IDs 副本，以及 B 的块表副本。
    A_token_ids = A.token_ids.copy()
    A_block_table = A.block_table.copy()
    B_block_table = B.block_table.copy()
    # 将第三轮从“预期断言”改为检查正常返回
    scheduled, is_prefill = scheduler.schedule()
    assert scheduled == [B] and is_prefill == True
    assert B.num_scheduled_tokens == 44
    assert A.status == SequenceStatus.RUNNING and A.num_tokens == 257 and A.num_cached_tokens == 256 and A.token_ids == A_token_ids and A.block_table == A_block_table
    assert B.block_table == B_block_table and scheduler.consecutive_prefills == 3
    scheduler.postprocess(scheduled, [100], is_prefill)
    assert B.num_tokens == 301 and B.num_cached_tokens == 300
    assert len(B.token_ids[len(B_token_ids):]) == 1
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [A, B]
    assert len(scheduler.block_manager.free_block_ids) == 0

    print("test_prefill_continues_when_forced_decode_cannot_append PASS")


if __name__ == "__main__":
    test_prefill_arrivals_delay_decode()

    test_prefill_limit_allows_decode()

    test_prefill_limit_with_empty_running()

    test_chunked_prefill_resumes_after_decode()

    test_prefill_allocation_failure_falls_back_to_decode()

    test_prefill_continues_when_forced_decode_cannot_append()