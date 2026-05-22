"""Phase-2 MFMA parity test: portable MFMA tiled INT4 dequant matmul
vs Phase-1 portable tiled INT4 dequant matmul (already proven bit-identical
to the production C-string blob).

Validates that the MFMA-accelerated version produces output within fp16 ULP
of the scalar reference. Runs on AMD MI300X (gfx942) via HIPRTC.

Tolerance: MFMA accumulates in fp32 and order-of-summation differs from the
scalar k-loop, so we expect small ULP drift (≤ 1e-2 abs, ≤ 1e-2 rel).
"""

import struct
import sys
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


def bits_fp16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def main():
    print("=== Phase-2 MFMA parity test on MI300X ===", flush=True)

    # Aligned to MFMA 16x16x16 tile
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

    print(f"Test: M={M} N={N} K={K} group_size={group_size}", flush=True)

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

    # ===== Run Phase-1 scalar reference =====
    print("\nCompiling Phase-1 tiled (scalar) reference...", flush=True)
    tiled_dequant_matmul_int4.compile(backend="rocm")
    mem.copy_host_to_device(bytes(M * N * 2), out_dev)
    grid1 = ((N + 31) // 32, (M + 31) // 32, 1)
    block1 = (32, 32, 1)
    print(f"Launch scalar: grid={grid1} block={block1}", flush=True)
    tiled_dequant_matmul_int4.launch(
        grid=grid1, block=block1,
        args=(out_dev, weight_dev, scales_dev, zeros_dev, inp_dev,
              M, N, K, group_size),
        backend="rocm",
    )
    scalar_bytes = mem.copy_device_to_host(out_dev)
    scalar_out = [bits_fp16(struct.unpack_from("<H", scalar_bytes, i * 2)[0])
                  for i in range(M * N)]

    # ===== Run Phase-2 MFMA =====
    print("\nCompiling Phase-2 MFMA kernel...", flush=True)
    mfma_dequant_matmul_int4.compile(backend="rocm")
    mem.copy_host_to_device(bytes(M * N * 2), out_dev)
    grid2 = ((N + 15) // 16, (M + 15) // 16, 1)
    block2 = (64, 1, 1)
    print(f"Launch MFMA: grid={grid2} block={block2}", flush=True)
    mfma_dequant_matmul_int4.launch(
        grid=grid2, block=block2,
        args=(out_dev, weight_dev, scales_dev, zeros_dev, inp_dev,
              M, N, K, group_size),
        backend="rocm",
    )
    mfma_bytes = mem.copy_device_to_host(out_dev)
    mfma_out = [bits_fp16(struct.unpack_from("<H", mfma_bytes, i * 2)[0])
                for i in range(M * N)]

    # ===== Compare =====
    print("\n--- Comparison ---", flush=True)
    n_mismatch = 0
    max_abs = 0.0
    max_rel = 0.0
    for i in range(M * N):
        a = mfma_out[i]
        b = scalar_out[i]
        abs_err = abs(a - b)
        rel_err = abs_err / max(abs(b), 1e-6)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        # fp16 accumulator divergence: allow ≤ 1% relative or 1e-2 absolute
        if abs_err > 1e-2 and rel_err > 1e-2:
            n_mismatch += 1
            if n_mismatch <= 5:
                print(f"  MISMATCH [{i//N},{i%N}]: mfma={a:.6f} scalar={b:.6f} "
                      f"abs={abs_err:.6f} rel={rel_err:.6f}", flush=True)

    print(f"\nMax abs error: {max_abs:.6f}", flush=True)
    print(f"Max rel error: {max_rel:.6f}", flush=True)
    print(f"Mismatches:    {n_mismatch}/{M*N}", flush=True)
    print(f"\nFirst row mfma:   {[f'{x:.3f}' for x in mfma_out[:8]]}", flush=True)
    print(f"First row scalar: {[f'{x:.3f}' for x in scalar_out[:8]]}", flush=True)

    mem.free(out_dev)
    mem.free(weight_dev)
    mem.free(scales_dev)
    mem.free(zeros_dev)
    mem.free(inp_dev)

    if n_mismatch == 0:
        print("\n=== PHASE-2 PARITY: PASSED ===", flush=True)
        sys.exit(0)
    print("\n=== PHASE-2 PARITY: FAILED ===", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
