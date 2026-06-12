"""Stage-0 NVIDIA concurrent-decode kernel probe — A10G.

Validates the portable warp-32 INT4 GEMV (`bgemv_int4_wave32`) against the
hand-written C-string `batched_dequant_gemv_int4` on real NVIDIA hardware:

  1. PARITY — bit-identical (within fp16 ULP) output at the 4 Qwen2.5-7B
     projection shapes, M=4 (concurrent decode batch).
  2. PERF   — per-kernel wall-clock (us/iter) for both kernels at each shape,
     so we know whether the portable kernel matches or beats the C-string.

Pure kernel microbenchmark: no model load, no weights download. Cheap + fast.

Run:  modal run tests/test_modal_bgemv_wave32_a10g.py
"""

import sys
import modal

app = modal.App("zse-bgemv-wave32-a10g")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine",  remote_path="/root/zse-engine",  copy=True)
)

# Qwen2.5-7B-Instruct projection shapes (hidden=3584, group_size=128):
#   QKV   : N = 3584 + 2*512 = 4608, K = 3584   (GQA: 28 q-heads, 4 kv-heads, hd=128)
#   O     : N = 3584,                 K = 3584
#   Gate/Up: N = 18944,               K = 3584
#   Down  : N = 3584,                 K = 18944
SHAPES = [
    ("O      3584x3584",  3584,  3584, 128),
    ("QKV    4608x3584",  4608,  3584, 128),
    ("GateUp 18944x3584", 18944, 3584, 128),
    ("Down   3584x18944", 3584, 18944, 128),
]
M = 4  # concurrent decode batch


@app.function(gpu="A10G", image=zse_image, timeout=1200)
def probe():
    import struct
    import time
    import random

    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import zse_compiler as zse
    from zse_compiler.runtime.compiler import RuntimeCompiler
    from zse_compiler.runtime.launcher import KernelLauncher, LaunchConfig
    from zse_engine.orchestrator.portable_kernels import bgemv_int4_wave32
    from zse_engine.orchestrator.kernels import BATCHED_DEQUANT_GEMV_INT4_CUDA

    def bits_to_fp16(bits):
        return struct.unpack("<e", struct.pack("<H", bits))[0]

    print("=== Stage-0 wave-32 INT4 GEMV probe on A10G ===", flush=True)
    dev = zse.detect_backend()
    print(f"backend={dev}", flush=True)

    mem = zse.GPUMemory(backend="cuda")
    launcher = KernelLauncher()

    # Compile portable kernel once.
    print("\nCompiling portable bgemv_int4_wave32 (CUDA)...", flush=True)
    bgemv_int4_wave32.compile(backend="cuda")
    print("OK", flush=True)

    # Compile C-string baseline once.
    print("Compiling C-string batched_dequant_gemv_int4 (CUDA)...", flush=True)
    kc = RuntimeCompiler()
    base = kc.compile(BATCHED_DEQUANT_GEMV_INT4_CUDA,
                      "batched_dequant_gemv_int4", "cuda")
    print("OK", flush=True)

    all_parity_ok = True
    perf_rows = []

    for label, N, K, gs in SHAPES:
        num_groups = (K + gs - 1) // gs
        random.seed(123)

        inp_bytes    = bytes(random.randint(0, 255) for _ in range(M * K * 2))
        weight_bytes = bytes(random.randint(0, 255) for _ in range(N * K // 2))
        scales_bytes = bytes(random.randint(0, 255) for _ in range(N * num_groups * 2))
        zeros_bytes  = bytes(random.randint(0, 255) for _ in range(N * num_groups * 2))

        out_dev    = mem.allocate((M * N,), zse.float16)
        weight_dev = mem.allocate((N * K // 2,), zse.uint8)
        scales_dev = mem.allocate((N * num_groups,), zse.float16)
        zeros_dev  = mem.allocate((N * num_groups,), zse.float16)
        inp_dev    = mem.allocate((M * K,), zse.float16)

        mem.copy_host_to_device(weight_bytes, weight_dev)
        mem.copy_host_to_device(scales_bytes, scales_dev)
        mem.copy_host_to_device(zeros_bytes,  zeros_dev)
        mem.copy_host_to_device(inp_bytes,    inp_dev)

        # ---- C-string baseline run ----
        mem.copy_host_to_device(bytes(M * N * 2), out_dev)
        base_grid = ((N + 7) // 8, 1, 1)
        base_block = (256, 1, 1)
        launcher.launch(
            base, LaunchConfig(grid=base_grid, block=base_block),
            out_dev, weight_dev, scales_dev, zeros_dev, inp_dev,
            M, N, K, gs,
        )
        base_out = mem.copy_device_to_host(out_dev)
        base_vals = [bits_to_fp16(struct.unpack_from("<H", base_out, i * 2)[0])
                     for i in range(M * N)]

        # ---- Portable wave-32 run ----
        mem.copy_host_to_device(bytes(M * N * 2), out_dev)
        w32_grid = ((N + 7) // 8, 1, 1)
        w32_block = (256, 1, 1)
        bgemv_int4_wave32.launch(
            grid=w32_grid, block=w32_block,
            args=(out_dev, weight_dev, scales_dev, zeros_dev, inp_dev, M, N, K, gs),
            backend="cuda",
        )
        w32_out = mem.copy_device_to_host(out_dev)
        w32_vals = [bits_to_fp16(struct.unpack_from("<H", w32_out, i * 2)[0])
                    for i in range(M * N)]

        # ---- Parity ----
        n_mis = 0
        max_abs = 0.0
        for i in range(M * N):
            a, b = w32_vals[i], base_vals[i]
            abs_err = abs(a - b)
            rel_err = abs_err / max(abs(b), 1e-6)
            max_abs = max(max_abs, abs_err)
            if abs_err > 1e-2 and rel_err > 1e-2:
                n_mis += 1
        parity = (n_mis == 0)
        all_parity_ok = all_parity_ok and parity

        # ---- Perf ----
        def bench(launch_fn, iters=200):
            for _ in range(5):
                launch_fn()
            mem.copy_device_to_host(out_dev)  # sync
            t0 = time.perf_counter()
            for _ in range(iters):
                launch_fn()
            mem.copy_device_to_host(out_dev)  # sync
            return (time.perf_counter() - t0) / iters * 1e6

        t_base = bench(lambda: launcher.launch(
            base, LaunchConfig(grid=base_grid, block=base_block),
            out_dev, weight_dev, scales_dev, zeros_dev, inp_dev, M, N, K, gs))
        t_w32 = bench(lambda: bgemv_int4_wave32.launch(
            grid=w32_grid, block=w32_block,
            args=(out_dev, weight_dev, scales_dev, zeros_dev, inp_dev, M, N, K, gs),
            backend="cuda"))

        speedup = t_base / t_w32 if t_w32 > 0 else 0.0
        perf_rows.append((label, t_base, t_w32, speedup, parity, max_abs))
        print(f"\n[{label}] N={N} K={K}", flush=True)
        print(f"  parity: {'PASS' if parity else 'FAIL'} (mis={n_mis}/{M*N}, max_abs={max_abs:.4f})", flush=True)
        print(f"  C-string : {t_base:8.1f} us", flush=True)
        print(f"  wave-32  : {t_w32:8.1f} us   ({speedup:.2f}x)", flush=True)

        for d in (out_dev, weight_dev, scales_dev, zeros_dev, inp_dev):
            mem.free(d)

    print("\n========== SUMMARY (M=4) ==========", flush=True)
    print(f"{'shape':20s} {'C-string':>10s} {'wave-32':>10s} {'speedup':>8s} {'parity':>7s}", flush=True)
    for label, tb, tw, sp, par, _ in perf_rows:
        print(f"{label:20s} {tb:9.1f}u {tw:9.1f}u {sp:7.2f}x {'OK' if par else 'BAD':>7s}", flush=True)

    print(f"\nALL PARITY: {'PASS' if all_parity_ok else 'FAIL'}", flush=True)
    return {"parity_ok": all_parity_ok,
            "rows": [(l, tb, tw, sp) for l, tb, tw, sp, _, _ in perf_rows]}


@app.local_entrypoint()
def main():
    res = probe.remote()
    print("\n=== LOCAL RESULT ===")
    print("parity_ok:", res["parity_ok"])
    for label, tb, tw, sp in res["rows"]:
        print(f"  {label:20s} C-string={tb:.1f}us  wave-32={tw:.1f}us  {sp:.2f}x")
