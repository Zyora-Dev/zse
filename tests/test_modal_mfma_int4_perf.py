"""Phase-2 MFMA perf sanity check on MI300X.

Compares the MFMA-accelerated INT4 dequant matmul against the Phase-1 scalar
reference at realistic LLM matmul shapes. Both are ROCm HIP kernels via HIPRTC.

Reports per-kernel wall-clock over many iterations after a warm-up.
"""

import ctypes
import struct
import sys
import time
import random

sys.path.insert(0, "/root/zse-engine")
sys.path.insert(0, "/root/zse-compiler")

import zse_compiler as zse
from zse_engine.orchestrator.portable_kernels import (
    tiled_dequant_matmul_int4,
    mfma_dequant_matmul_int4,
)


def fp16_bits(f):
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bench(name, kernel, grid, block, args, iters, mem, out_dev, out_bytes):
    # warm up
    for _ in range(3):
        kernel.launch(grid=grid, block=block, args=args, backend="rocm")
    _ = mem.copy_device_to_host(out_dev)  # device sync
    t0 = time.perf_counter()
    for _ in range(iters):
        kernel.launch(grid=grid, block=block, args=args, backend="rocm")
    _ = mem.copy_device_to_host(out_dev)
    dt = (time.perf_counter() - t0) / iters * 1e6  # microseconds per launch
    print(f"  {name:30s}  {dt:8.1f} us/iter", flush=True)
    return dt


def run_shape(M, N, K, group_size, iters):
    print(f"\n=== Shape: M={M} N={N} K={K} group_size={group_size} ===", flush=True)
    num_groups = (K + group_size - 1) // group_size
    random.seed(0)

    inp_bytes = bytes(random.randint(0, 255) for _ in range(M * K * 2))
    weight_bytes = bytes(random.randint(0, 255) for _ in range(N * K // 2))
    scales_bytes = bytes(random.randint(0, 255) for _ in range(N * num_groups * 2))
    zeros_bytes = bytes(random.randint(0, 255) for _ in range(N * num_groups * 2))

    mem = zse.GPUMemory(backend="rocm")
    out_dev = mem.allocate((M * N,), zse.float16)
    weight_dev = mem.allocate((N * K // 2,), zse.uint8)
    scales_dev = mem.allocate((N * num_groups,), zse.float16)
    zeros_dev = mem.allocate((N * num_groups,), zse.float16)
    inp_dev = mem.allocate((M * K,), zse.float16)
    mem.copy_host_to_device(weight_bytes, weight_dev)
    mem.copy_host_to_device(scales_bytes, scales_dev)
    mem.copy_host_to_device(zeros_bytes, zeros_dev)
    mem.copy_host_to_device(inp_bytes, inp_dev)
    out_host = bytes(M * N * 2)

    args = (out_dev, weight_dev, scales_dev, zeros_dev, inp_dev,
            M, N, K, group_size)

    t_scalar = bench("Phase-1 scalar (32x32 tile)",
                     tiled_dequant_matmul_int4,
                     ((N + 31) // 32, (M + 31) // 32, 1), (32, 32, 1),
                     args, iters, mem, out_dev, out_host)

    t_mfma = bench("Phase-2 MFMA (16x16 tile)",
                   mfma_dequant_matmul_int4,
                   ((N + 15) // 16, (M + 15) // 16, 1), (64, 1, 1),
                   args, iters, mem, out_dev, out_host)

    speedup = t_scalar / t_mfma
    print(f"  --> MFMA speedup: {speedup:.2f}x", flush=True)

    mem.free(out_dev)
    mem.free(weight_dev)
    mem.free(scales_dev)
    mem.free(zeros_dev)
    mem.free(inp_dev)


def main():
    print("=== MFMA vs Phase-1 scalar perf on MI300X ===", flush=True)
    print("Compiling kernels...", flush=True)
    tiled_dequant_matmul_int4.compile(backend="rocm")
    mfma_dequant_matmul_int4.compile(backend="rocm")
    print("Compile OK", flush=True)

    # Realistic Qwen2.5-32B matmul shapes (per layer):
    #   QKV proj: M=batch, K=5120, N=7680 (combined Q+K+V for GQA)
    #   O proj:   M=batch, K=5120, N=5120
    #   Gate/Up:  M=batch, K=5120, N=27648
    #   Down:     M=batch, K=27648, N=5120
    # Use a few representative shapes; batch=16 (concurrent decode N=4 with 4 tokens).
    for shape in [
        (16, 5120, 5120, 128),
        (16, 7680, 5120, 128),
        (16, 27648, 5120, 128),
        (16, 5120, 27648, 128),
        (1,  5120, 5120, 128),   # single decode QKV/O size
    ]:
        run_shape(*shape, iters=20)


if __name__ == "__main__":
    main()
