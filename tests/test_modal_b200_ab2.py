"""ZSE — VALID A/B of the medium-M fix on B200 (graph captured per-config).

The earlier A/B was invalid: CUDA graphs capture kernels ONCE, so flipping the
Python flag after capture replayed the same graph. Here each config gets its OWN
engine, and the flag is set BEFORE the graph is captured (first decode).

Run: modal run tests/test_modal_b200_ab2.py
"""

import sys
import modal

app = modal.App("zse-b200-ab2")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)

N = 12
MAX_TOKENS = 100


@app.function(
    gpu="B200", image=zse_image, timeout=2400,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def ab2():
    import os, time
    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    print("=" * 70, flush=True)
    print("ZSE — VALID medium-M A/B on B200 (per-config graph capture)", flush=True)
    print("=" * 70, flush=True)

    zse_path = "/root/zse_cache/qwen2_14b.zse"
    assert os.path.exists(zse_path), "cached .zse missing"

    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    base_prompts = [f"Write a short essay number {i} about renewable energy." for i in range(N)]

    def build_engine(disable_medium):
        eng = ZStreamerEngine(
            model_path=zse_path,
            scheduler_config=SchedulerConfig(max_batch_tokens=4096, max_batch_seqs=16),
            max_seq_len=512, quiet=True,
        )
        # Set BEFORE any decode so the captured graph uses the chosen kernel.
        eng._model_runner._disable_medium_gemv = disable_medium
        return eng

    def run_n12(eng, max_tokens=MAX_TOKENS):
        toks = [[] for _ in range(N)]
        done = [False] * N
        def cbs(idx):
            def on_t(tid): toks[idx].append(tid)
            def on_f(o): done[idx] = True
            return on_t, on_f
        t0 = time.monotonic()
        for i, p in enumerate(base_prompts):
            on_t, on_f = cbs(i)
            eng.add_request(prompt=p, max_tokens=max_tokens, temperature=0.0,
                            on_token=on_t, on_finish=on_f)
        for _ in range(max_tokens + 300):
            eng.step()
            if all(done):
                break
        elapsed = time.monotonic() - t0
        total = sum(len(t) for t in toks)
        return total / elapsed if elapsed > 0 else 0.0

    def best(disable, label, runs=3):
        eng = build_engine(disable)
        run_n12(eng, max_tokens=16)  # warmup + graph capture with flag set
        b = 0.0
        for _ in range(runs):
            b = max(b, run_n12(eng))
        print(f"    {label:20s}: {b:7.1f} tok/s", flush=True)
        # free VRAM before building the next engine
        try:
            eng.shutdown()
        except Exception:
            pass
        return b

    print("\n[A/B] N=12 aggregate (separate engine per config, graph per config):", flush=True)
    tps_on = best(False, "fix ON (bgemv M16)")
    tps_off = best(True, "fix OFF (tiled)")

    delta = tps_on - tps_off
    pct = (delta / tps_off * 100) if tps_off > 0 else 0.0
    results = {
        "n12_fix_on": round(tps_on, 1),
        "n12_fix_off": round(tps_off, 1),
        "delta_tps": round(delta, 1),
        "delta_pct": round(pct, 1),
    }
    print("\n" + "=" * 70, flush=True)
    print(f"    fix ON : {tps_on:.1f} tok/s", flush=True)
    print(f"    fix OFF: {tps_off:.1f} tok/s", flush=True)
    print(f"    DELTA  : {delta:+.1f} tok/s ({pct:+.1f}%)", flush=True)
    print(f"RESULT: {results}", flush=True)
    print("=" * 70, flush=True)
    return results


@app.local_entrypoint()
def main():
    print(ab2.remote())
