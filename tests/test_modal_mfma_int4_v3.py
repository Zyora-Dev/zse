"""Phase-3 MFMA INT4 dequant matmul — parity + perf on MI300X.

Validates the v3 kernel (vectorized u32 weight load + shared-mem dequant +
hoisted scale lookup) against the Phase-1 scalar reference (already proven
bit-identical to production), then benchmarks against both Phase-1 scalar
and Phase-2 MFMA at real Qwen2.5-32B layer shapes.
"""

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
    mfma_dequant_matmul_int4_v3,
)


def fp16_bits(f):
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_fp16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


# ===== Parity =====

def parity():
    print("\n=== Phase-3 parity vs Phase-1 scalar ===", flush=True)
    M, N, K, group_size = 16, 128, 128, 64
    num_groups = (K + group_size - 1) // group_size
    random.seed(42)
    inp_fp = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    weight_bytes = bytes(random.randint(0, 255) for _ in range(N * K // 2))
    scales_fp = [random.uniform(0.001, 0.1) for _ in range(N * num_groups)]
    zeros_fp = [random.uniform(-1.0, 1.0) for _ in range(N * num_groups)]
    inp_bytes = b"".join(struct.pack("<H", fp16_bits(v)) for v in inp_fp)
    scales_bytes = b"".join(struct.pack("<H", fp16_bits(v)) for v in scales_fp)
    zeros_bytes = b"".join(struct.pack("<H", fp16_bits(v)) for v in zeros_fp)

    mem = zse.GPUMemory(backend="rocm")
    out_dev = mem.allocate((M * N,), zse.float16)
    w_dev = mem.allocate((N * K // 2,), zse.uint8)
    s_dev = mem.allocate((N * num_groups,), zse.float16)
    z_dev = mem.allocate((N * num_groups,), zse.float16)
    i_dev = mem.allocate((M * K,), zse.float16)
    mem.copy_host_to_device(weight_bytes, w_dev)
    mem.copy_host_to_device(scales_bytes, s_dev)
    mem.copy_host_to_device(zeros_bytes, z_dev)
    mem.copy_host_to_device(inp_bytes, i_dev)
    args = (out_dev, w_dev, s_dev, z_dev, i_dev, M, N, K, group_size)

    # Scalar reference
    mem.copy_host_to_device(bytes(M * N * 2), out_dev)
    tiled_dequant_matmul_int4.launch(
        grid=((N + 31) // 32, (M + 31) // 32, 1), block=(32, 32, 1),
        args=args, backend="rocm")
    ref = [bits_fp16(struct.unpack_from("<H", mem.copy_device_to_host(out_dev), i * 2)[0])
           for i in range(M * N)]

    # v3 MFMA
    mem.copy_host_to_device(bytes(M * N * 2), out_dev)
    mfma_dequant_matmul_int4_v3.launch(
        grid=((N + 15) // 16, (M + 15) // 16, 1), block=(64, 1, 1),
        args=args, backend="rocm")
    v3 = [bits_fp16(struct.unpack_from("<H", mem.copy_device_to_host(out_dev), i * 2)[0])
          for i in range(M * N)]

    n_bad = 0
    max_abs = 0.0
    max_rel = 0.0
    for i in range(M * N):
        ae = abs(v3[i] - ref[i])
        re = ae / max(abs(ref[i]), 1e-6)
        max_abs = max(max_abs, ae)
        max_rel = max(max_rel, re)
        if ae > 1e-2 and re > 1e-2:
            n_bad += 1
            if n_bad <= 5:
                print(f"  MISMATCH [{i//N},{i%N}]: v3={v3[i]:.6f} ref={ref[i]:.6f} "
                      f"ae={ae:.4f} re={re:.4f}", flush=True)

    print(f"  max abs={max_abs:.6f}  max rel={max_rel:.6f}  mismatches={n_bad}/{M*N}", flush=True)
    print(f"  v3 first row:  {[f'{x:.3f}' for x in v3[:8]]}", flush=True)
    print(f"  ref first row: {[f'{x:.3f}' for x in ref[:8]]}", flush=True)

    mem.free(out_dev); mem.free(w_dev); mem.free(s_dev); mem.free(z_dev); mem.free(i_dev)
    return n_bad == 0


# ===== Perf =====

def bench_one(name, kernel, grid, block, args, iters, mem, out_dev):
    for _ in range(3):
        kernel.launch(grid=grid, block=block, args=args, backend="rocm")
    _ = mem.copy_device_to_host(out_dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        kernel.launch(grid=grid, block=block, args=args, backend="rocm")
    _ = mem.copy_device_to_host(out_dev)
    return (time.perf_counter() - t0) / iters * 1e6


def perf_shape(M, N, K, group_size, iters):
    print(f"\n  M={M:<4} N={N:<6} K={K:<6} group_size={group_size}", flush=True)
    num_groups = (K + group_size - 1) // group_size
    random.seed(0)
    inp_bytes = bytes(random.randint(0, 255) for _ in range(M * K * 2))
    weight_bytes = bytes(random.randint(0, 255) for _ in range(N * K // 2))
    scales_bytes = bytes(random.randint(0, 255) for _ in range(N * num_groups * 2))
    zeros_bytes = bytes(random.randint(0, 255) for _ in range(N * num_groups * 2))

    mem = zse.GPUMemory(backend="rocm")
    out_dev = mem.allocate((M * N,), zse.float16)
    w_dev = mem.allocate((N * K // 2,), zse.uint8)
    s_dev = mem.allocate((N * num_groups,), zse.float16)
    z_dev = mem.allocate((N * num_groups,), zse.float16)
    i_dev = mem.allocate((M * K,), zse.float16)
    mem.copy_host_to_device(weight_bytes, w_dev)
    mem.copy_host_to_device(scales_bytes, s_dev)
    mem.copy_host_to_device(zeros_bytes, z_dev)
    mem.copy_host_to_device(inp_bytes, i_dev)
    args = (out_dev, w_dev, s_dev, z_dev, i_dev, M, N, K, group_size)

    t_s = bench_one("scalar", tiled_dequant_matmul_int4,
                    ((N + 31) // 32, (M + 31) // 32, 1), (32, 32, 1),
                    args, iters, mem, out_dev)
    t_2 = bench_one("v2", mfma_dequant_matmul_int4,
                    ((N + 15) // 16, (M + 15) // 16, 1), (64, 1, 1),
                    args, iters, mem, out_dev)
    t_3 = bench_one("v3", mfma_dequant_matmul_int4_v3,
                    ((N + 15) // 16, (M + 15) // 16, 1), (64, 1, 1),
                    args, iters, mem, out_dev)

    print(f"    scalar  {t_s:8.1f} us   (1.00x)", flush=True)
    print(f"    v2-MFMA {t_2:8.1f} us   ({t_s/t_2:.2f}x)", flush=True)
    print(f"    v3-MFMA {t_3:8.1f} us   ({t_s/t_3:.2f}x vs scalar, {t_2/t_3:.2f}x vs v2)", flush=True)

    mem.free(out_dev); mem.free(w_dev); mem.free(s_dev); mem.free(z_dev); mem.free(i_dev)


def main():
    print("=== Phase-3 MFMA INT4 (v3) — parity + perf on MI300X ===", flush=True)
    print("Compiling kernels...", flush=True)
    tiled_dequant_matmul_int4.compile(backend="rocm")
    mfma_dequant_matmul_int4.compile(backend="rocm")
    mfma_dequant_matmul_int4_v3.compile(backend="rocm")
    print("Compile OK", flush=True)

    if not parity():
        print("\n=== PARITY FAILED — skipping perf ===", flush=True)
        sys.exit(1)
    print("  PARITY OK", flush=True)

    print("\n=== Perf at real Qwen2.5-32B shapes ===", flush=True)
    for shape in [
        (16, 5120, 5120, 128),     # O proj
        (16, 7680, 5120, 128),     # QKV proj
        (16, 27648, 5120, 128),    # Gate/Up proj
        (16, 5120, 27648, 128),    # Down proj
        (1,  5120, 5120, 128),     # Single decode O size
    ]:
        perf_shape(*shape, iters=20)

    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
