"""ZSE — Decode-step decomposition profiler on Modal B200.

Answers ONE question with hard numbers: at N=12, where does the per-token time go?
  - Pure GPU decode time (batched_decode_graph) at M = 1,4,8,12,16 (synced)
  - Full engine.step() split into scheduler vs batch_runner.execute

If GPU-per-step scales ~linearly with M  -> GPU compute is the cost (fix kernels).
If GPU-per-step is ~flat but engine.step scales -> Python/scheduler is the cost.

Run: modal run tests/test_modal_b200_profile.py
"""

import sys
import modal

app = modal.App("zse-b200-profile")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)


@app.function(
    gpu="B200", image=zse_image, timeout=2400,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def profile():
    import os, time, ctypes
    os.environ["HF_HOME"] = "/root/hf_cache"
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    print("=" * 70, flush=True)
    print("ZSE — decode-step decomposition on B200 (Qwen2.5-14B INT4)", flush=True)
    print("=" * 70, flush=True)

    zse_path = "/root/zse_cache/qwen2_14b.zse"
    assert os.path.exists(zse_path), "cached .zse missing"

    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    engine = ZStreamerEngine(
        model_path=zse_path,
        scheduler_config=SchedulerConfig(max_batch_tokens=4096, max_batch_seqs=16),
        max_seq_len=512, quiet=True,
    )
    runner = engine._model_runner
    driver = engine._gpu_mem._driver
    tok = engine._tokenizer

    def sync():
        driver.cuCtxSynchronize()

    results = {}

    # ---------------------------------------------------------------- #
    # (A) Pure GPU decode time vs M (batched_decode_graph, synced)
    # ---------------------------------------------------------------- #
    print("\n[A] Pure GPU batched_decode_graph time vs M (30 steps, synced):", flush=True)
    prompt_tokens = tok.encode("Explain the theory of relativity in detail and at length")
    plen = len(prompt_tokens)

    for M in (1, 4, 8, 12, 16):
        # Fresh sequences: prefill each so KV is populated
        seq_ids = list(range(1000, 1000 + M))
        for sid in seq_ids:
            try:
                engine._kv_cache.free_sequence(sid)
            except Exception:
                pass
            runner.prefill(list(prompt_tokens), sid)

        token_ids = [prompt_tokens[-1]] * M
        positions = [plen - 1] * M

        # Warmup (captures graph for this M)
        for _ in range(3):
            runner.batched_decode_graph(token_ids, list(seq_ids),
                                        [plen + 0] * M)
        sync()

        t0 = time.perf_counter()
        steps = 30
        for s in range(steps):
            runner.batched_decode_graph(token_ids, list(seq_ids),
                                        [plen + 1 + s] * M)
        sync()
        dt = (time.perf_counter() - t0) / steps * 1000  # ms/step
        per_tok = dt / M
        print(f"    M={M:2d}: {dt:7.2f} ms/step   {per_tok:6.2f} ms/token   "
              f"({1000*M/dt:7.1f} tok/s aggregate)", flush=True)
        results[f"gpu_m{M}_ms_step"] = round(dt, 2)
        results[f"gpu_m{M}_ms_tok"] = round(per_tok, 2)

        for sid in seq_ids:
            try:
                engine._kv_cache.free_sequence(sid)
            except Exception:
                pass

    # ---------------------------------------------------------------- #
    # (B) Full engine.step() split: scheduler vs execute, at N=12
    # ---------------------------------------------------------------- #
    print("\n[B] Full engine.step() breakdown at N=12 (scheduler vs execute):", flush=True)
    prompts = [f"Write a short essay number {i} about renewable energy sources."
               for i in range(12)]
    for p in prompts:
        engine.add_request(prompt=p, max_tokens=100, temperature=0.0)

    sched = engine._scheduler
    runner_exec = engine._batch_runner

    sched_ms = 0.0
    exec_ms = 0.0
    n_steps = 0
    t_all = time.perf_counter()
    for _ in range(200):
        t0 = time.perf_counter()
        output = sched.schedule_step()
        t1 = time.perf_counter()
        if output.is_idle:
            break
        runner_exec.execute(output)
        sync()
        t2 = time.perf_counter()
        sched_ms += (t1 - t0) * 1000
        exec_ms += (t2 - t1) * 1000
        n_steps += 1
    wall = (time.perf_counter() - t_all) * 1000

    print(f"    steps={n_steps}  wall={wall:.0f} ms", flush=True)
    if n_steps:
        print(f"    scheduler: {sched_ms/n_steps:6.2f} ms/step  ({100*sched_ms/wall:4.1f}%)", flush=True)
        print(f"    execute  : {exec_ms/n_steps:6.2f} ms/step  ({100*exec_ms/wall:4.1f}%)", flush=True)
    results["sched_ms_step"] = round(sched_ms / max(n_steps, 1), 2)
    results["exec_ms_step"] = round(exec_ms / max(n_steps, 1), 2)
    results["steps"] = n_steps

    print("\n" + "=" * 70, flush=True)
    print(f"RESULT: {results}", flush=True)
    print("=" * 70, flush=True)
    return results


@app.local_entrypoint()
def main():
    res = profile.remote()
    print("\n=== LOCAL RESULT ===")
    print(res)
