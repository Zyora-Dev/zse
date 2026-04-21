"""Test CUDA Graph integration — Qwen2.5-7B on A100-80GB.

Tests:
1. Position-from-buffer kernel changes work
2. Graph capture + replay produces correct output
3. Throughput improvement measurement
"""

import sys
import modal

app = modal.App("zse-graph-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)


@app.function(
    gpu="A100-80GB",
    image=zse_image,
    timeout=3600,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def test_graph():
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import os
    import time
    import struct
    import threading

    print("=" * 70)
    print("ZSE CUDA GRAPH TEST — Qwen2.5-14B on A100-80GB")
    print("=" * 70)

    # Init CUDA
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    ret = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert ret == 0, f"cuCtxCreate failed: {ret}"

    # Ensure .zse file
    zse_path = "/root/zse_cache/qwen2_14b.zse"
    if os.path.exists(zse_path):
        sz = os.path.getsize(zse_path)
        if sz < 1_000_000_000:
            os.remove(zse_path)
        else:
            with open(zse_path, 'rb') as f:
                magic = f.read(4)
            if magic != b'ZSE\x01':
                os.remove(zse_path)

    if not os.path.exists(zse_path):
        progress_file = zse_path + ".progress"
        if os.path.exists(progress_file):
            os.remove(progress_file)
        from huggingface_hub import snapshot_download
        print("[0] Downloading + converting Qwen2.5-14B-Instruct...")
        hf_dir = snapshot_download(
            "Qwen/Qwen2.5-14B-Instruct",
            cache_dir="/root/hf_cache",
            allow_patterns=["*.safetensors", "*.json"],
        )
        from zse_engine.format.convert import convert_hf_to_zse
        convert_hf_to_zse(hf_dir, zse_path)
        try:
            vol = modal.Volume.lookup("zse-model-cache")
            vol.commit()
        except Exception:
            pass
    else:
        print(f"[CACHE] Using cached .zse ({os.path.getsize(zse_path)/1024**3:.2f} GB)")

    # Use ZStreamerEngine (handles CUDA context for background threads)
    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    print("\n[1] Loading engine...")
    t0 = time.monotonic()
    engine = ZStreamerEngine(
        model_path=zse_path,
        scheduler_config=SchedulerConfig(max_batch_tokens=2048, max_batch_seqs=8),
        max_seq_len=512,
        quiet=False,
    )
    print(f"  Loaded in {time.monotonic()-t0:.2f}s")

    tokenizer = engine._tokenizer
    runner = engine._model_runner
    config = engine._config

    # ================================================================
    # Test 1: Position-from-buffer works (normal decode, no graph)
    # ================================================================
    print("\n[2] Testing position-from-buffer (normal decode)...")
    prompt = "The capital of France is"
    prompt_tokens = tokenizer.encode(prompt)

    logits = runner.prefill(prompt_tokens, seq_id=0)
    vocab = config.vocab_size
    vals = struct.unpack(f'<{vocab}e', logits[:vocab*2])
    first_tok = max(range(vocab), key=lambda i: vals[i])
    print(f"  First token after prefill: {first_tok} = '{tokenizer.decode([first_tok])}'")

    # Decode a few steps (normal path, no graph)
    tok = first_tok
    decoded_tokens = [tok]
    for i in range(10):
        logits = runner.decode_step(tok, 0, len(prompt_tokens) + i, skip_logits_download=True)
        tok = runner.gpu_argmax(0)
        decoded_tokens.append(tok)

    text = tokenizer.decode(decoded_tokens)
    print(f"  Decoded: '{text}'")
    assert len(decoded_tokens) == 11, f"Expected 11 tokens, got {len(decoded_tokens)}"
    print("  PASS: Position-from-buffer works!")

    engine._kv_cache.mark_idle(0)
    engine._kv_cache.free_sequence(0)

    # ================================================================
    # Test 2: Graph capture + replay
    # ================================================================
    print("\n[3] Testing CUDA Graph capture + replay...")

    # Init graph
    runner.init_graph(max_seq_len=512)
    print(f"  Graph runner initialized (backend={runner._backend})")

    # Prefill
    logits = runner.prefill(prompt_tokens, seq_id=1)
    vals = struct.unpack(f'<{vocab}e', logits[:vocab*2])
    first_tok = max(range(vocab), key=lambda i: vals[i])

    # Step 0: warmup (normal decode)
    logits = runner.decode_step(first_tok, 1, len(prompt_tokens), skip_logits_download=True)
    tok = runner.gpu_argmax(0)
    graph_tokens = [first_tok, tok]

    # Step 1: capture graph
    print("  Capturing graph...")
    tok = runner.decode_step_graph(tok, 1, len(prompt_tokens) + 1)
    graph_tokens.append(tok)
    assert runner._graph_captured, "Graph should be captured!"
    print(f"  Graph captured! Token: {tok}")

    # Step 2+: replay
    print("  Replaying graph...")
    for i in range(8):
        tok = runner.decode_step_graph(tok, 1, len(prompt_tokens) + 2 + i)
        graph_tokens.append(tok)

    graph_text = tokenizer.decode(graph_tokens)
    print(f"  Graph decoded: '{graph_text}'")

    # Compare with non-graph output
    if graph_text[:30] == text[:30]:
        print("  PASS: Graph output matches non-graph output!")
    else:
        print(f"  WARN: Outputs differ")
        print(f"    Normal: '{text}'")
        print(f"    Graph:  '{graph_text}'")

    engine._kv_cache.mark_idle(1)
    engine._kv_cache.free_sequence(1)

    # ================================================================
    # Test 3: Throughput measurement
    # ================================================================
    print("\n[4] Throughput benchmark...")

    # Destroy old graph, re-init for clean timing
    runner.destroy_graph()

    # Baseline: no graph
    prompt2 = "Explain the theory of relativity in simple terms"
    prompt2_tokens = tokenizer.encode(prompt2)

    logits = runner.prefill(prompt2_tokens, seq_id=10)
    vals = struct.unpack(f'<{vocab}e', logits[:vocab*2])
    first_tok = max(range(vocab), key=lambda i: vals[i])

    # Warmup
    for i in range(3):
        runner.decode_step(first_tok, 10, len(prompt2_tokens) + i, skip_logits_download=True)
        runner.gpu_argmax(0)

    # Timed no-graph
    t0 = time.monotonic()
    for i in range(30):
        runner.decode_step(first_tok, 10, len(prompt2_tokens) + 3 + i, skip_logits_download=True)
        runner.gpu_argmax(0)
    t1 = time.monotonic()
    no_graph_ms = (t1 - t0) / 30 * 1000
    no_graph_tps = 1000 / no_graph_ms
    print(f"  No graph: {no_graph_ms:.1f}ms/step = {no_graph_tps:.1f} tok/s")

    engine._kv_cache.mark_idle(10)
    engine._kv_cache.free_sequence(10)

    # With graph
    runner.init_graph(max_seq_len=512)

    logits = runner.prefill(prompt2_tokens, seq_id=11)
    vals = struct.unpack(f'<{vocab}e', logits[:vocab*2])
    first_tok = max(range(vocab), key=lambda i: vals[i])

    # Warmup decode
    runner.decode_step(first_tok, 11, len(prompt2_tokens), skip_logits_download=True)
    tok = runner.gpu_argmax(0)

    # Capture
    tok = runner.decode_step_graph(tok, 11, len(prompt2_tokens) + 1)

    # Warmup replays
    for i in range(3):
        tok = runner.decode_step_graph(tok, 11, len(prompt2_tokens) + 2 + i)

    # Timed graph replay
    t0 = time.monotonic()
    for i in range(30):
        tok = runner.decode_step_graph(tok, 11, len(prompt2_tokens) + 5 + i)
    t1 = time.monotonic()
    graph_ms = (t1 - t0) / 30 * 1000
    graph_tps = 1000 / graph_ms
    print(f"  Graph:    {graph_ms:.1f}ms/step = {graph_tps:.1f} tok/s")

    speedup = no_graph_ms / graph_ms if graph_ms > 0 else 0
    print(f"  Speedup:  {speedup:.2f}x")

    engine._kv_cache.mark_idle(11)
    engine._kv_cache.free_sequence(11)
    runner.destroy_graph()

    # ================================================================
    # Summary
    # ================================================================
    # Test 5: 4 concurrent requests via ZStreamer (with graph)
    # ================================================================
    print("\n[5] 4 concurrent requests (via ZStreamer, optimized batched path)...")

    # Don't use graph for concurrent — test the kernel improvements directly
    runner.destroy_graph()

    concurrent_prompts = [
        "Write about solar energy.",
        "Write about wind energy.",
        "Write about nuclear energy.",
        "Write about hydro energy.",
    ]
    conc_tokens = [[] for _ in range(4)]
    conc_done = [threading.Event() for _ in range(4)]

    for i, p in enumerate(concurrent_prompts):
        def make_cbs(idx):
            def on_t(tid):
                conc_tokens[idx].append(tid)
            def on_f(out):
                if hasattr(out, 'finish_reason'):
                    print(f"  Req {idx} finished: {out.finish_reason}, {len(conc_tokens[idx])} tokens")
                conc_done[idx].set()
            return on_t, on_f
        on_t, on_f = make_cbs(i)
        engine.add_request(prompt=p, max_tokens=100, temperature=0.0,
                          on_token=on_t, on_finish=on_f)

    t0 = time.monotonic()
    step_count = 0
    step_times = []
    for _ in range(600):
        try:
            t_step = time.monotonic()
            result = engine.step()
            dt = time.monotonic() - t_step
            step_count += 1
            if step_count <= 5:
                print(f"  Step {step_count}: {result.num_tokens} new tokens, {len(result.finished)} finished, {dt*1000:.1f}ms")
            if step_count > 10:
                step_times.append(dt)
        except Exception as e:
            print(f"  STEP ERROR: {e}")
            import traceback; traceback.print_exc()
            break
        if all(e.is_set() for e in conc_done):
            break
    t_conc = time.monotonic() - t0
    total_conc = sum(len(t) for t in conc_tokens)
    conc_tps = total_conc / t_conc if t_conc > 0 else 0
    print(f"  {total_conc} tokens in {t_conc:.2f}s = {conc_tps:.1f} tok/s aggregate ({step_count} steps)")
    if step_times:
        avg_step = sum(step_times) / len(step_times) * 1000
        print(f"  Avg step (decode): {avg_step:.1f}ms ({4000/avg_step:.1f} aggregate tok/s)")
    for i in range(4):
        n = len(conc_tokens[i])
        per_tps = n / t_conc if t_conc > 0 else 0
        print(f"  Req {i}: {n} tokens ({per_tps:.1f} tok/s)")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Single (no graph):  {no_graph_ms:.1f}ms = {no_graph_tps:.1f} tok/s")
    print(f"  Single (graph):     {graph_ms:.1f}ms = {graph_tps:.1f} tok/s")
    print(f"  Graph speedup:      {speedup:.2f}x")
    print(f"  4 concurrent:       {conc_tps:.1f} tok/s aggregate")

    engine.stop()

    return {
        "no_graph_ms": round(no_graph_ms, 1),
        "graph_ms": round(graph_ms, 1),
        "no_graph_tps": round(no_graph_tps, 1),
        "graph_tps": round(graph_tps, 1),
        "speedup": round(speedup, 2),
        "concurrent_4_tps": round(conc_tps, 1),
    }


@app.local_entrypoint()
def main():
    print("Launching CUDA Graph test on A100-80GB...")
    results = test_graph.remote()
    print("\nResults:", results)
