import statistics
from collections.abc import Callable

import torch

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.triton_rms_norm import fused_add_rms_norm


WARMUP = 30
CALLS_PER_SAMPLE = 100
SAMPLES = 10
HIDDEN_SIZE = 1024
EPS = 1e-6


def measure_us(
    call: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    start: torch.cuda.Event,
    stop: torch.cuda.Event,
) -> float:
    start.record()
    for _ in range(CALLS_PER_SAMPLE):
        call()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / CALLS_PER_SAMPLE


@torch.inference_mode()
def benchmark_shape(rows: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (rows, HIDDEN_SIZE)

    x = torch.randn(shape, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    norm = RMSNorm(HIDDEN_SIZE, eps=EPS).to(device=device, dtype=dtype).eval()
    norm.weight.uniform_(0.5, 1.5)

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        return norm(x, residual)

    def triton() -> tuple[torch.Tensor, torch.Tensor]:
        return fused_add_rms_norm(
            res=residual, x=x, weight=norm.weight, eps=norm.eps
        )

    # Allow small BF16 rounding differences in normalized output.
    baseline_y, baseline_residual = baseline()
    triton_y, triton_residual = triton()
    torch.testing.assert_close(triton_y, baseline_y, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(
        triton_residual, baseline_residual, rtol=0, atol=0
    )

    for _ in range(WARMUP):
        baseline()
        triton()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    baseline_samples = []
    triton_samples = []
    for sample in range(SAMPLES):
        if sample % 2 == 0:
            baseline_us = measure_us(baseline, start, stop)
            triton_us = measure_us(triton, start, stop)
        else:
            triton_us = measure_us(triton, start, stop)
            baseline_us = measure_us(baseline, start, stop)

        baseline_samples.append(baseline_us)
        triton_samples.append(triton_us)

    paired_deltas = [
        triton_us - baseline_us
        for baseline_us, triton_us in zip(baseline_samples, triton_samples)
    ]

    print(f"\nBF16 {rows}x{HIDDEN_SIZE} | µs/call")
    print(f"Baseline median       {statistics.median(baseline_samples):.3f}")
    print(f"Triton median         {statistics.median(triton_samples):.3f}")
    print(f"Paired delta median   {statistics.median(paired_deltas):+.3f}")
    print("Paired deltas        ", ", ".join(f"{d:+.3f}" for d in paired_deltas))


if __name__ == "__main__":
    torch.manual_seed(42)
    for rows in (1, 2, 14, 17, 18):
        benchmark_shape(rows)