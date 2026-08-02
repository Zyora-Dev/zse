"""ZSE — Medium-concurrency INT4 GEMV (M=9..16) validation on Modal NVIDIA A100.

Validates the M=8->16 cliff fix on the NVIDIA path:
  1. PARITY: batched_dequant_gemv_int4 (now BGEMV_MAX_M=16) at M=12 vs a pure-Python
     reference (dequant = nibble*scale + zero, then matmul). Small shape.
  2. THROUGHPUT: at a real Qwen2.5-class shape and M=12, compare the medium-M
     bgemv against the tiled prefill kernel it replaces (the pre-fix path).

Run: modal run tests/test_modal_bgemv_m16.py
"""

import modal
import sys

app = modal.App("zse-bgemv-m16-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
)


@app.function(gpu="A100", image=zse_image, timeout=600)
def test_bgemv_m16():
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import time
    import struct
    import random

    print("=" * 62, flush=True)
    print("ZSE — bgemv M=9..16 medium-concurrency validation (A100)", flush=True)
    print("=" * 62, flush=True)

    # CUDA context
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    ret = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert ret == 0, f"cuCtxCreate failed: {ret}"

    from zse_compiler.runtime.memory import GPUMemory
    from zse_compiler.runtime.device import get_devices
    from zse_compiler.types.dtypes import float16, int32, uint8
    from zse_engine.orchestrator.kernels import InferenceKernels

    gpu_mem = GPUMemory(backend="cuda")
    device = get_devices("cuda")[0]
    print(f"Device: {device.name}  VRAM: {device.vram_total_gb:.1f} GB", flush=True)

    kernels = InferenceKernels(backend="cuda")

    results = {}

    # ================================================================== #
    # 1. PARITY at M=12 (small shape, pure-Python reference)
    # ================================================================== #
    print("\n[1] PARITY — batched_dequant_gemv_int4 @ M=12 (BGEMV_MAX_M=16)", flush=True)
    random.seed(42)
    M, N, K, gs = 12, 128, 256, 128
    num_groups = K // gs
    half_K = K // 2

    # Random packed INT4 weights [N, K/2] uint8 (two nibbles per byte, low first)
    w_bytes = bytes(random.randint(0, 255) for _ in range(N * half_K))
    # Random fp16 scales/zeros [N, num_groups]
    scales = [random.uniform(0.005, 0.02) for _ in range(N * num_groups)]
    zeros = [random.uniform(-0.1, 0.1) for _ in range(N * num_groups)]
    # Random fp16 input [M, K]
    inp = [random.uniform(-1.0, 1.0) for _ in range(M * K)]

    # ---- Python reference: out[m, n] = sum_k dequant(w[n,k]) * inp[m,k] ----
    def deq(n, k):
        byte = w_bytes[n * half_K + (k // 2)]
        nib = (byte & 0xF) if (k % 2 == 0) else ((byte >> 4) & 0xF)
        g = k // gs
        return nib * scales[n * num_groups + g] + zeros[n * num_groups + g]

    ref = [0.0] * (M * N)
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for k in range(K):
                acc += deq(n, k) * inp[m * K + k]
            ref[m * N + n] = acc

    # ---- GPU launch ----
    w_t = gpu_mem.allocate((N, half_K), uint8)
    gpu_mem.copy_host_to_device(w_bytes, w_t)
    sc_t = gpu_mem.allocate((N, num_groups), float16)
    gpu_mem.copy_host_to_device(struct.pack(f'<{N*num_groups}e', *scales), sc_t)
    zr_t = gpu_mem.allocate((N, num_groups), float16)
    gpu_mem.copy_host_to_device(struct.pack(f'<{N*num_groups}e', *zeros), zr_t)
    inp_t = gpu_mem.allocate((M, K), float16)
    gpu_mem.copy_host_to_device(struct.pack(f'<{M*K}e', *inp), inp_t)
    out_t = gpu_mem.allocate((M, N), float16)

    kernels.launch(
        "batched_dequant_gemv_int4",
        ((N + 7) // 8,), (256,),
        out_t, w_t, sc_t, zr_t, inp_t,
        M, N, K, gs,
    )
    out_bytes = gpu_mem.copy_device_to_host(out_t)
    got = list(struct.unpack(f'<{M*N}e', out_bytes))

    max_abs = 0.0
    max_rel = 0.0
    mism = 0
    for i in range(M * N):
        a, b = got[i], ref[i]
        d = abs(a - b)
        max_abs = max(max_abs, d)
        denom = max(abs(b), 1e-3)
        r = d / denom
        max_rel = max(max_rel, r)
        if d > 1e-2 and r > 1e-2:
            mism += 1

    print(f"    M={M} N={N} K={K} gs={gs}", flush=True)
    print(f"    mismatches (abs>1e-2 AND rel>1e-2): {mism}/{M*N}", flush=True)
    print(f"    max_abs={max_abs:.5f}  max_rel={max_rel:.5f}", flush=True)
    parity_ok = (mism == 0)
    print(f"    PARITY: {'PASS' if parity_ok else 'FAIL'}", flush=True)
    results["parity_m12"] = parity_ok
    results["parity_mismatches"] = mism

    # ================================================================== #
    # 2. THROUGHPUT at a real Qwen2.5-class shape, M=12
    #    medium-M bgemv (fix) vs tiled prefill kernel (pre-fix path)
    # ================================================================== #
    print("\n[2] THROUGHPUT — M=12, N=5120, K=5120, gs=128", flush=True)
    Mb, Nb, Kb, gsb = 12, 5120, 5120, 128
    ngb = Kb // gsb
    hKb = Kb // 2

    wb = bytes(random.randint(0, 255) for _ in range(Nb * hKb))
    scb = [random.uniform(0.005, 0.02) for _ in range(Nb * ngb)]
    zrb = [random.uniform(-0.1, 0.1) for _ in range(Nb * ngb)]
    ib = [random.uniform(-1.0, 1.0) for _ in range(Mb * Kb)]

    wb_t = gpu_mem.allocate((Nb, hKb), uint8)
    gpu_mem.copy_host_to_device(wb, wb_t)
    scb_t = gpu_mem.allocate((Nb, ngb), float16)
    gpu_mem.copy_host_to_device(struct.pack(f'<{Nb*ngb}e', *scb), scb_t)
    zrb_t = gpu_mem.allocate((Nb, ngb), float16)
    gpu_mem.copy_host_to_device(struct.pack(f'<{Nb*ngb}e', *zrb), zrb_t)
    ib_t = gpu_mem.allocate((Mb, Kb), float16)
    gpu_mem.copy_host_to_device(struct.pack(f'<{Mb*Kb}e', *ib), ib_t)
    ob_t = gpu_mem.allocate((Mb, Nb), float16)

    def bench(fn, iters=50, warmup=10):
        for _ in range(warmup):
            fn()
        libcuda.cuCtxSynchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        libcuda.cuCtxSynchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # us

    def run_bgemv():
        kernels.launch(
            "batched_dequant_gemv_int4",
            ((Nb + 7) // 8,), (256,),
            ob_t, wb_t, scb_t, zrb_t, ib_t,
            Mb, Nb, Kb, gsb,
        )

    def run_tiled():
        kernels.launch(
            "tiled_dequant_matmul_int4",
            ((Nb + 31) // 32, (Mb + 31) // 32), (32, 32),
            ob_t, wb_t, scb_t, zrb_t, ib_t,
            Mb, Nb, Kb, gsb,
        )

    us_bgemv = bench(run_bgemv)
    us_tiled = bench(run_tiled)
    speedup = us_tiled / us_bgemv if us_bgemv > 0 else 0.0
    print(f"    medium-M bgemv (fix):     {us_bgemv:8.1f} us", flush=True)
    print(f"    tiled prefill (pre-fix):  {us_tiled:8.1f} us", flush=True)
    print(f"    speedup: {speedup:.2f}x", flush=True)
    results["us_bgemv"] = round(us_bgemv, 1)
    results["us_tiled"] = round(us_tiled, 1)
    results["speedup"] = round(speedup, 2)

    print("\n" + "=" * 62, flush=True)
    print(f"RESULT: {results}", flush=True)
    print("=" * 62, flush=True)
    return results


@app.local_entrypoint()
def main():
    res = test_bgemv_m16.remote()
    print("\n=== LOCAL RESULT ===")
    print(res)
