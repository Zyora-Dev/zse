"""ZSE — A/B of the medium-M (M=8->16) cliff fix on Modal B200.

Runs N=12 concurrent decode TWICE in one process (one cold start = fair A/B):
  A) fix ON  — M=9..16 uses the widened bgemv (BGEMV_MAX_M=16)
  B) fix OFF — M=9..16 falls back to the tiled prefill kernel (pre-fix)

Reports the exact aggregate tok/s delta the cliff fix contributes end-to-end.

Run: modal run tests/test_modal_b200_ab.py
"""

import sys
import modal

app = modal.App("zse-b200-ab")

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
N = 12


@app.function(
    gpu="B200",
    image=zse_image,
    timeout=2400,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def ab_benchmark():
    import os
    import time

    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    print("=" * 70, flush=True)
    print("ZSE — medium-M cliff fix A/B on B200 (Qwen2.5-14B INT4, N=12)", flush=True)
    print("=" * 70, flush=True)

    zse_path = "/root/zse_cache/qwen2_14b.zse"
    assert os.path.exists(zse_path), "cached .zse not found; run the concurrent test first to convert"
    print(f"[CACHE] .zse ({os.path.getsize(zse_path)/1024**3:.2f} GB)", flush=True)

    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    engine = ZStreamerEngine(
        model_path=zse_path,
        scheduler_config=SchedulerConfig(max_batch_tokens=4096, max_batch_seqs=16),
        max_seq_len=512,
        quiet=True,
    )
    runner = engine._model_runner

    base_prompts = [
        "Write about solar energy.", "Write about wind energy.",
        "Write about nuclear power.", "Write about hydro power.",
        "Write about geothermal energy.", "Write about tidal energy.",
        "Write about biomass fuel.", "Write about hydrogen fuel.",
        "Write about coal history.", "Write about oil refining.",
        "Write about the power grid.", "Write about battery storage.",
    ]

    def run_n12(max_tokens=MAX_TOKENS):
        toks = [[] for _ in range(N)]
        done = [False] * N

        def make_cbs(idx):
            def on_t(tid):
                toks[idx].append(tid)
            def on_f(out):
                done[idx] = True
            return on_t, on_f

        t0 = time.monotonic()
        for i, p in enumerate(base_prompts[:N]):
            on_t, on_f = make_cbs(i)
            engine.add_request(prompt=p, max_tokens=max_tokens, temperature=0.0,
                               on_token=on_t, on_finish=on_f)
        for _ in range(max_tokens + 300):
            engine.step()
            if all(done):
                break
        elapsed = time.monotonic() - t0
        total = sum(len(t) for t in toks)
        return total, elapsed, (total / elapsed if elapsed > 0 else 0.0)

    # Warmup both kernel paths so JIT compile is out of the timed region.
    runner._disable_medium_gemv = False
    run_n12(max_tokens=16)
    runner._disable_medium_gemv = True
    run_n12(max_tokens=16)

    results = {}
    print("\n[A/B] N=12 aggregate throughput (3 runs each, best reported):", flush=True)

    def best_of(label, disable, runs=3):
        runner._disable_medium_gemv = disable
        best = 0.0
        for _ in range(runs):
            total, elapsed, tps = run_n12()
            best = max(best, tps)
        print(f"    {label:18s}: {best:7.1f} tok/s", flush=True)
        return best

    tps_on = best_of("fix ON (bgemv)", disable=False)
    tps_off = best_of("fix OFF (tiled)", disable=True)

    delta = tps_on - tps_off
    pct = (delta / tps_off * 100) if tps_off > 0 else 0.0
    results = {
        "n12_fix_on_tps": round(tps_on, 1),
        "n12_fix_off_tps": round(tps_off, 1),
        "delta_tps": round(delta, 1),
        "delta_pct": round(pct, 1),
    }

    print("\n" + "=" * 70, flush=True)
    print(f"    fix ON : {tps_on:.1f} tok/s", flush=True)
    print(f"    fix OFF: {tps_off:.1f} tok/s", flush=True)
    print(f"    DELTA  : +{delta:.1f} tok/s  (+{pct:.1f}%)", flush=True)
    print(f"RESULT: {results}", flush=True)
    print("=" * 70, flush=True)
    return results


@app.local_entrypoint()
def main():
    res = ab_benchmark.remote()
    print("\n=== LOCAL RESULT ===")
    print(res)
