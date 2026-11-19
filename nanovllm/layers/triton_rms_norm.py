import triton
import triton.language as tl
import torch

@triton.jit
def fused_add_rms_norm_kernel(Res, X, Weight, Y, ResOut, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    row_offsets = row * N + cols
    res = tl.load(Res + row_offsets, mask=mask, other=0).to(tl.float32)
    x = tl.load(X + row_offsets, mask=mask, other=0).to(tl.float32)
    w = tl.load(Weight + cols, mask=mask, other=0).to(tl.float32)
    s = res + x
    sum_sq = tl.sum(s * s, axis=0)
    inv_rms = tl.rsqrt(sum_sq / N + eps)
    y = s * inv_rms * w
    tl.store(Y + row_offsets, y, mask=mask)
    tl.store(ResOut + row_offsets, s, mask=mask)

def fused_add_rms_norm(res, x, weight, eps=1e-6):
    M, N = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    resout = torch.empty_like(res)
    fused_add_rms_norm_kernel[(M,)](res, x, weight, output, resout, N, eps, BLOCK_SIZE)
    return output, resout