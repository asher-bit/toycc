# exp04: 第 4/5 章实验 —— Triton GEMM, 对比 BLOCK / warps / stages
# 【可运行代码】需要 NVIDIA GPU + pip install triton
# 运行: python exp04_gemm_triton.py
# 对照第 5 章: BLOCK 改值 = 重新编译(num_stages 的 shared 账见第 5 章第 7 节)
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am
                    + (k + offs_k)[None, :] * stride_ak)
        b = tl.load(b_ptr + (k + offs_k)[:, None] * stride_bk
                    + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm
             + offs_n[None, :] * stride_cn, acc)


def bench(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, warps, stages):
    import torch
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (M // BLOCK_M, N // BLOCK_N)

    fn = gemm_kernel.warmup(a, b, c, M, N, K,
                            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1),
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                            num_warps=warps, num_stages=stages, grid=grid)
    ms = fn(a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=warps, num_stages=stages, grid=grid)
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


if __name__ == "__main__":
    M = N = K = 4096
    configs = [
        ("BLOCK 64,  warps 4, stages 3", dict(BLOCK_M=64, BLOCK_N=64,
         BLOCK_K=64, warps=4, stages=3)),
        ("BLOCK 128, warps 4, stages 3", dict(BLOCK_M=128, BLOCK_N=128,
         BLOCK_K=64, warps=4, stages=3)),
        ("BLOCK 128, warps 8, stages 4", dict(BLOCK_M=128, BLOCK_N=128,
         BLOCK_K=64, warps=8, stages=4)),
        ("BLOCK 128, warps 8, stages 6", dict(BLOCK_M=128, BLOCK_N=128,
         BLOCK_K=64, warps=8, stages=6)),
    ]
    print(f"GEMM {M}x{N}x{K} fp16 (每次改一个变量, 先 warmup 再计时)")
    for name, cfg in configs:
        ms, tflops = bench(M, N, K, **cfg)
        print(f"  {name:<30} {ms:8.2f} ms  {tflops:7.1f} TFLOPS")
    print("""
对照第 5 章:
  - BLOCK 128x128 + stages=4 的 shared = 128KB(第 5 章 7 节手算)
  - stages=6 → 192KB > Ampere 164KB 上限 → 预期报错或 occupancy 崩
  - 记录每组配置的编译时间: BLOCK 变化 = 新 cache key = 重新编译
""")
