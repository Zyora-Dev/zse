"""ZSE — End-to-end concurrent throughput on Modal NVIDIA B200 (Blackwell).

Measures aggregate decode tok/s at N=1, N=4, and N=12 concurrent requests on a
real Qwen2.5-14B-Instruct INT4 model. N=12 is the band that exercises the new
medium-concurrency INT4 path (bgemv with BGEMV_MAX_M=16) — the M=8->16 cliff fix.

B200 is Blackwell (sm_100) → requires CUDA 12.8+. The ZSE compiler auto-detects
the arch (compute_100) via the CUDA driver at runtime.

Run: modal run tests/test_modal_b200_concurrent.py
"""

import sys
import modal

app = modal.App("zse-b200-concurrent")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MAX_TOKENS = 100


@app.function(
    gpu="B200",
    image=zse_image,
    timeout=2400,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def benchmark_concurrent():
    import os
    import time
    import struct

    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    print("=" * 70, flush=True)
    print("ZSE — CONCURRENT THROUGHPUT on B200 (Qwen2.5-14B INT4)", flush=True)
    print("=" * 70, flush=True)

    from zse_compiler.runtime.device import get_devices
    dev = get_devices("cuda")[0]
    print(f"Device: {dev.name}  sm_{dev.compute_capability}  VRAM {dev.vram_total_gb:.1f} GB", flush=True)

    # --- Ensure .zse exists (reuse cached, else convert) ---
    zse_path = "/root/zse_cache/qwen2_14b.zse"
    if not os.path.exists(zse_path):
        print("[0] Converting Qwen2.5-14B-Instruct → .zse ...", flush=True)
        from huggingface_hub import snapshot_download
        hf_dir = snapshot_download(MODEL_ID, cache_dir="/root/hf_cache")
        from zse_engine.format.convert import convert_hf_to_zse
        t0 = time.time()
        convert_hf_to_zse(hf_dir, zse_path)
        print(f"     Converted in {time.time()-t0:.1f}s", flush=True)
        zse_cache.commit()
    else:
        print(f"[CACHE] Using cached .zse ({os.path.getsize(zse_path)/1024**3:.2f} GB)", flush=True)

    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    t_cold = time.monotonic()
    engine = ZStreamerEngine(
        model_path=zse_path,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=4096,
            max_batch_seqs=16,   # must be >= 12 to run N=12 concurrently
        ),
        max_seq_len=512,
        quiet=False,
    )
    print(f"\nCold start: {time.monotonic()-t_cold:.2f}s", flush=True)

    tokenizer = engine._tokenizer
    results = {"device": dev.name, "sm": dev.compute_capability}

    # Distinct prompts so sequences don't dedup — keeps the decode batch full.
    base_prompts = [
        "Write about solar energy.", "Write about wind energy.",
        "Write about nuclear power.", "Write about hydro power.",
        "Write about geothermal energy.", "Write about tidal energy.",
        "Write about biomass fuel.", "Write about hydrogen fuel.",
        "Write about coal history.", "Write about oil refining.",
        "Write about the power grid.", "Write about battery storage.",
        "Write about electric cars.", "Write about solar panels.",
        "Write about wind turbines.", "Write about smart meters.",
    ]

    def run_concurrent(n, max_tokens=MAX_TOKENS):
        prompts = base_prompts[:n]
        toks = [[] for _ in range(n)]
        done = [False] * n

        def make_cbs(idx):
            def on_t(tid):
                toks[idx].append(tid)
            def on_f(out):
                done[idx] = True
            return on_t, on_f

        t0 = time.monotonic()
        for i, p in enumerate(prompts):
            on_t, on_f = make_cbs(i)
            engine.add_request(prompt=p, max_tokens=max_tokens, temperature=0.0,
                               on_token=on_t, on_finish=on_f)

        steps = 0
        for _ in range((max_tokens + 300)):
            engine.step()
            steps += 1
            if all(done):
                break
        elapsed = time.monotonic() - t0
        total = sum(len(t) for t in toks)
        tps = total / elapsed if elapsed > 0 else 0.0
        return total, elapsed, tps

    # Warmup (JIT-compiles the medium-M kernel on first M in [9..16])
    print("\n[warmup] N=12 ...", flush=True)
    run_concurrent(12, max_tokens=16)

    print("\n[BENCHMARK] aggregate decode throughput:", flush=True)
    for n in (1, 4, 12):
        total, elapsed, tps = run_concurrent(n)
        print(f"    N={n:2d}: {total:5d} tokens in {elapsed:6.2f}s = {tps:7.1f} tok/s aggregate", flush=True)
        results[f"n{n}_tps"] = round(tps, 1)
        results[f"n{n}_tokens"] = total

    # Per-request rate at N=12 for context
    if results.get("n12_tps") and results.get("n1_tps"):
        scaling = results["n12_tps"] / results["n1_tps"]
        print(f"\n    N=12 vs N=1 scaling: {scaling:.2f}x aggregate", flush=True)
        results["n12_over_n1_scaling"] = round(scaling, 2)

    print("\n" + "=" * 70, flush=True)
    print(f"RESULT: {results}", flush=True)
    print("=" * 70, flush=True)
    return results


@app.local_entrypoint()
def main():
    res = benchmark_concurrent.remote()
    print("\n=== LOCAL RESULT ===")
    print(res)
