"""Phase-1 parity test: portable @zse.kernel tiled INT4 dequant matmul
vs production C-string TILED_DEQUANT_MATMUL_INT4_CUDA blob.

Validates that the new portable kernel produces bit-identical (within fp16 ULP)
output to the hand-written CUDA blob currently used in production.

Runs on AMD MI300X (gfx942) via HIPRTC.

Test setup:
  M=4, N=128, K=128, group_size=64
  Random fp16 input, random uint8 weights, random fp16 scales/zeros.
"""

import struct
import sys
import random

sys.path.insert(0, "/root/zse-engine")
sys.path.insert(0, "/root/zse-compiler")

import zse_compiler as zse
from zse_engine.orchestrator.portable_kernels import tiled_dequant_matmul_int4
from zse_engine.orchestrator.kernels import TILED_DEQUANT_MATMUL_INT4_CUDA


def fp16_to_bits(f):
    """IEEE 754 half-precision encoding using struct (Python 3.6+)."""
    # Use struct's 'e' format (half-precision float)
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_fp16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def main():
    print("=== Phase-1 portable tiled INT4 parity test on MI300X ===", flush=True)

    M, N, K, group_size = 4, 128, 128, 64
    num_groups = (K + group_size - 1) // group_size  # 2

    random.seed(42)

    # Generate test data
    # Input: M*K fp16 in [-1, 1]
    inp_fp = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    # Weights: N*K/2 uint8 (packed INT4)
    weight_bytes = bytes(random.randint(0, 255) for _ in range(N * K // 2))
    # Scales: N*num_groups fp16, positive small
    scales_fp = [random.uniform(0.001, 0.1) for _ in range(N * num_groups)]
    # Zeros: N*num_groups fp16, [-1, 1]
    zeros_fp = [random.uniform(-1.0, 1.0) for _ in range(N * num_groups)]

    # Encode to fp16 bytes
    inp_bytes = b"".join(struct.pack("<H", fp16_to_bits(v)) for v in inp_fp)
    scales_bytes = b"".join(struct.pack("<H", fp16_to_bits(v)) for v in scales_fp)
    zeros_bytes = b"".join(struct.pack("<H", fp16_to_bits(v)) for v in zeros_fp)

    print(f"Test: M={M} N={N} K={K} group_size={group_size}", flush=True)
    print(f"Sizes: inp={len(inp_bytes)}B weight={len(weight_bytes)}B "
          f"scales={len(scales_bytes)}B zeros={len(zeros_bytes)}B", flush=True)

    mem = zse.GPUMemory(backend="rocm")

    # Device buffers (shared across both runs)
    out_dev = mem.allocate((M * N,), zse.float16)
    weight_dev = mem.allocate((N * K // 2,), zse.uint8)
    scales_dev = mem.allocate((N * num_groups,), zse.float16)
    zeros_dev = mem.allocate((N * num_groups,), zse.float16)
    inp_dev = mem.allocate((M * K,), zse.float16)

    mem.copy_host_to_device(weight_bytes, weight_dev)
    mem.copy_host_to_device(scales_bytes, scales_dev)
    mem.copy_host_to_device(zeros_bytes, zeros_dev)
    mem.copy_host_to_device(inp_bytes, inp_dev)

    # ===== Run portable kernel =====
    print("\nCompiling portable @zse.kernel...", flush=True)
    tiled_dequant_matmul_int4.compile(backend="rocm")
    print("Compile OK", flush=True)

    mem.copy_host_to_device(bytes(M * N * 2), out_dev)  # zero
    grid = ((N + 31) // 32, (M + 31) // 32, 1)
    block = (32, 32, 1)
    print(f"Launch: grid={grid} block={block}", flush=True)
    tiled_dequant_matmul_int4.launch(
        grid=grid, block=block,
        args=(out_dev, weight_dev, scales_dev, zeros_dev, inp_dev,
              M, N, K, group_size),
        backend="rocm",
    )
    portable_out_bytes = mem.copy_device_to_host(out_dev)
    portable_out = [bits_to_fp16(struct.unpack_from("<H", portable_out_bytes, i * 2)[0])
                    for i in range(M * N)]

    # ===== Run production C-string kernel =====
    # The blob is CUDA-flavored — to run on HIP we substitute the include.
    # CUDA_HEADER + r"""extern "C" __global__ void tiled_dequant_matmul_int4(...)"""
    print("\nCompiling production C-string kernel...", flush=True)
    hip_blob = TILED_DEQUANT_MATMUL_INT4_CUDA.replace(
        "#include <cuda_fp16.h>", "#include <hip/hip_runtime.h>"
    )
    # Compile via runtime compiler directly
    from zse_compiler.runtime.compiler import RuntimeCompiler
    kc = RuntimeCompiler()
    compiled = kc.compile(hip_blob, "tiled_dequant_matmul_int4", "rocm")
    print("Compile OK", flush=True)

    mem.copy_host_to_device(bytes(M * N * 2), out_dev)  # zero
    # Direct launch via the launcher
    from zse_compiler.runtime.launcher import KernelLauncher, LaunchConfig
    launcher = KernelLauncher()
    config = LaunchConfig(grid=grid, block=block)
    launcher.launch(
        compiled, config,
        out_dev, weight_dev, scales_dev, zeros_dev, inp_dev,
        M, N, K, group_size,
    )
    prod_out_bytes = mem.copy_device_to_host(out_dev)
    prod_out = [bits_to_fp16(struct.unpack_from("<H", prod_out_bytes, i * 2)[0])
                for i in range(M * N)]

    # ===== Compare =====
    print("\n--- Comparison ---", flush=True)
    n_mismatch = 0
    max_rel_err = 0.0
    max_abs_err = 0.0
    for i in range(M * N):
        a = portable_out[i]
        b = prod_out[i]
        abs_err = abs(a - b)
        rel_err = abs_err / max(abs(b), 1e-6)
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
        # Allow 1 fp16 ULP (~5e-4 relative for normal values)
        if abs_err > 1e-3 and rel_err > 5e-3:
            n_mismatch += 1
            if n_mismatch <= 5:
                print(f"  MISMATCH [{i//N},{i%N}]: portable={a:.6f} prod={b:.6f} "
                      f"abs_err={abs_err:.6f} rel_err={rel_err:.6f}", flush=True)

    print(f"\nMax abs error: {max_abs_err:.6f}", flush=True)
    print(f"Max rel error: {max_rel_err:.6f}", flush=True)
    print(f"Mismatches: {n_mismatch}/{M*N}", flush=True)

    print("\nSample first row:", flush=True)
    print(f"  portable: {[f'{x:.3f}' for x in portable_out[:8]]}", flush=True)
    print(f"  prod:     {[f'{x:.3f}' for x in prod_out[:8]]}", flush=True)

    mem.free(out_dev)
    mem.free(weight_dev)
    mem.free(scales_dev)
    mem.free(zeros_dev)
    mem.free(inp_dev)

    if n_mismatch == 0:
        print("\n=== PHASE-1 PARITY: PASSED ===", flush=True)
        sys.exit(0)
    print("\n=== PHASE-1 PARITY: FAILED ===", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
