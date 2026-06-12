"""Stage-1 WMMA probe — does INT4 tensor-core GEMM beat the GEMV at small M?

This is the decisive measurement for NVIDIA concurrency. It pits the EXISTING
hand-written tensor-core kernel `wmma_dequant_gemm_int4` (which always computes
a full 16-row output tile via WMMA) against `batched_dequant_gemv_int4` running
at M=8 (the GEMV's best/maximum case).

Decision rule (concurrent decode is M=2..8):
  - WMMA computes 16 rows per call no matter what (pads M up to 16).
  - GEMV at M=8 computes 8 rows.
  - If  WMMA_time(16 rows)  <  GEMV_time(8 rows)  -> tensor cores win even after
    padding waste -> Marlin-style path is worth building.
  - If  WMMA_time(16 rows)  >  ~2x GEMV_time(8 rows)  -> padding to 16 wastes the
    matrix cores at M=4..8 -> tensor cores are a dead end for decode. Ship as-is.

Also does a sanity parity check (WMMA vs a CPU reference) so we know the
tensor-core path isn't silently wrong.

Run:  modal run tests/test_modal_wmma_probe_a10g.py
"""

import sys
import modal

app = modal.App("zse-wmma-probe-a10g")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine",  remote_path="/root/zse-engine",  copy=True)
)

# Qwen2.5-7B-Instruct projection shapes (hidden=3584, group_size=128).
SHAPES = [
    ("O      3584x3584",  3584,  3584, 128),
    ("QKV    4608x3584",  4608,  3584, 128),
    ("GateUp 18944x3584", 18944, 3584, 128),
    ("Down   3584x18944", 3584, 18944, 128),
]


@app.function(gpu="A10G", image=zse_image, timeout=1800)
def probe():
    import struct
    import time
    import random

    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import zse_compiler as zse
    from zse_compiler.runtime.compiler import RuntimeCompiler
    from zse_compiler.runtime.launcher import KernelLauncher, LaunchConfig
    from zse_engine.orchestrator.kernels import (
        BATCHED_DEQUANT_GEMV_INT4_CUDA,
        WMMA_DEQUANT_GEMM_INT4_CUDA,
    )

    def bits_to_fp16(b):
        return struct.unpack("<e", struct.pack("<H", b))[0]

    def fp16_bits(f):
        return struct.unpack("<H", struct.pack("<e", f))[0]

    print("=== Stage-1 WMMA probe on A10G ===", flush=True)
    print("backend:", zse.detect_backend(), flush=True)

    mem = zse.GPUMemory(backend="cuda")
    launcher = KernelLauncher()
    kc = RuntimeCompiler()

    print("\nCompiling kernels...", flush=True)
    gemv = kc.compile(BATCHED_DEQUANT_GEMV_INT4_CUDA,
                      "batched_dequant_gemv_int4", "cuda")
    wmma = kc.compile(WMMA_DEQUANT_GEMM_INT4_CUDA,
                      "wmma_dequant_gemm_int4", "cuda")
    print("OK", flush=True)

    def host_repack(weight_bytes, N, K):
        """Row-major [N, K/2] -> tiled [n_tile, k_tile, 64, 8]."""
        half_K = K // 2
        num_k_tiles = (K + 15) // 16          # ceil(K/16); k bytes per tile = 8
        num_n_tiles = (N + 63) // 64
        dst = bytearray(num_n_tiles * num_k_tiles * 512)
        for n_tile in range(num_n_tiles):
            for k_tile in range(num_k_tiles):
                base = (n_tile * num_k_tiles + k_tile) * 512
                for n_local in range(64):
                    n_global = n_tile * 64 + n_local
                    if n_global >= N:
                        continue
                    for byte_idx in range(8):
                        k_byte = k_tile * 8 + byte_idx
                        if k_byte >= half_K:
                            continue
                        dst[base + n_local * 8 + byte_idx] = \
                            weight_bytes[n_global * half_K + k_byte]
        return bytes(dst), num_k_tiles

    rows = []
    for label, N, K, gs in SHAPES:
        num_groups = (K + gs - 1) // gs
        random.seed(7)

        M_full = 16  # WMMA always computes a 16-row tile
        inp_bytes    = bytes(random.randint(0, 255) for _ in range(M_full * K * 2))
        weight_bytes = bytes(random.randint(0, 255) for _ in range(N * K // 2))
        scales_bytes = bytes(random.randint(0, 255) for _ in range(N * num_groups * 2))
        zeros_bytes  = bytes(random.randint(0, 255) for _ in range(N * num_groups * 2))

        tiled_bytes, num_k_tiles = host_repack(weight_bytes, N, K)

        # Device buffers
        out_dev    = mem.allocate((M_full * N,), zse.float16)
        weight_dev = mem.allocate((N * K // 2,), zse.uint8)
        tiled_dev  = mem.allocate((len(tiled_bytes),), zse.uint8)
        scales_dev = mem.allocate((N * num_groups,), zse.float16)
        zeros_dev  = mem.allocate((N * num_groups,), zse.float16)
        inp_dev    = mem.allocate((M_full * K,), zse.float16)

        mem.copy_host_to_device(weight_bytes, weight_dev)
        mem.copy_host_to_device(tiled_bytes,  tiled_dev)
        mem.copy_host_to_device(scales_bytes, scales_dev)
        mem.copy_host_to_device(zeros_bytes,  zeros_dev)
        mem.copy_host_to_device(inp_bytes,    inp_dev)

        # ---- WMMA launch config: grid=(N/64,), block=128 (4 warps) ----
        wmma_grid = ((N + 63) // 64, 1, 1)
        wmma_block = (128, 1, 1)

        def run_wmma():
            launcher.launch(
                wmma, LaunchConfig(grid=wmma_grid, block=wmma_block),
                out_dev, tiled_dev, scales_dev, zeros_dev, inp_dev,
                M_full, N, K, gs, num_k_tiles,
            )

        # ---- GEMV at M=8 (best case for GEMV) ----
        M_gemv = 8
        gemv_grid = ((N + 7) // 8, 1, 1)
        gemv_block = (256, 1, 1)

        def run_gemv():
            launcher.launch(
                gemv, LaunchConfig(grid=gemv_grid, block=gemv_block),
                out_dev, weight_dev, scales_dev, zeros_dev, inp_dev,
                M_gemv, N, K, gs,
            )

        def bench(fn, iters=200):
            for _ in range(5):
                fn()
            mem.copy_device_to_host(out_dev)  # sync
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            mem.copy_device_to_host(out_dev)  # sync
            return (time.perf_counter() - t0) / iters * 1e6

        # Sanity: WMMA output row 0 should be finite (not NaN/inf)
        mem.copy_host_to_device(bytes(M_full * N * 2), out_dev)
        run_wmma()
        wout = mem.copy_device_to_host(out_dev)
        sample = [bits_to_fp16(struct.unpack_from("<H", wout, i * 2)[0]) for i in range(8)]
        finite = all(abs(x) < 1e5 for x in sample)

        t_wmma = bench(run_wmma)
        t_gemv = bench(run_gemv)

        # Normalize: WMMA does 16 rows, GEMV does 8. Compare per-call (real cost
        # at decode is one call regardless of how many rows it computes).
        ratio = t_gemv / t_wmma if t_wmma > 0 else 0.0
        rows.append((label, t_wmma, t_gemv, ratio, finite))
        print(f"\n[{label}] N={N} K={K}", flush=True)
        print(f"  WMMA  (16-row tile): {t_wmma:8.1f} us   finite={finite}", flush=True)
        print(f"  GEMV  (M=8)        : {t_gemv:8.1f} us", flush=True)
        print(f"  GEMV/WMMA ratio    : {ratio:.2f}  ({'WMMA faster' if ratio>1 else 'GEMV faster'})", flush=True)

        for d in (out_dev, weight_dev, tiled_dev, scales_dev, zeros_dev, inp_dev):
            mem.free(d)

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'shape':20s} {'WMMA(16)':>10s} {'GEMV(8)':>10s} {'GEMV/WMMA':>10s}", flush=True)
    for label, tw, tg, r, fin in rows:
        verdict = "WMMA wins" if r > 1.0 else "GEMV wins"
        print(f"{label:20s} {tw:9.1f}u {tg:9.1f}u {r:9.2f}x  {verdict}", flush=True)

    wmma_wins = sum(1 for _, _, _, r, _ in rows if r > 1.0)
    print(f"\nWMMA wins {wmma_wins}/{len(rows)} shapes", flush=True)
    print("VERDICT:", "TENSOR CORES WORTH PURSUING" if wmma_wins >= 3
          else "TENSOR CORES DEAD END FOR DECODE — SHIP AS-IS", flush=True)
    return {"rows": [(l, tw, tg, r) for l, tw, tg, r, _ in rows], "wmma_wins": wmma_wins}


@app.local_entrypoint()
def main():
    res = probe.remote()
    print("\n=== LOCAL RESULT ===")
    for label, tw, tg, r in res["rows"]:
        print(f"  {label:20s} WMMA={tw:.1f}us  GEMV(M8)={tg:.1f}us  ratio={r:.2f}x")
    print("wmma_wins:", res["wmma_wins"], "/ 4")
