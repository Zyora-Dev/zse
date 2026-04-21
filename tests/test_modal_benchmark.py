"""ZSE vs vLLM Benchmark — Qwen2.5-7B-Instruct on A100.

Head-to-head comparison:
- Cold start: time from zero to first token ready
- Throughput: tok/s for greedy decode (128 tokens)
- VRAM: total GPU memory used after model load
- Output quality: verify coherent text

Note: ZSE uses INT4 quantization, vLLM uses FP16.
"""

import sys
import modal

app = modal.App("zse-vs-vllm-benchmark")

# ============================================================================
# Images
# ============================================================================

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install("vllm", "torch")
)

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)

PROMPT = "Write a detailed essay about the history of artificial intelligence, covering its origins, major milestones, and future prospects."
MAX_TOKENS = 128
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


# ============================================================================
# vLLM Benchmark
# ============================================================================

@app.function(
    gpu="A100",
    image=vllm_image,
    timeout=1800,
    volumes={"/root/hf_cache": hf_cache},
)
def benchmark_vllm():
    """Benchmark vLLM: cold start, throughput, VRAM."""
    import time
    import os
    import torch

    os.environ["HF_HOME"] = "/root/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/root/hf_cache"

    results = {"engine": "vLLM"}

    print("=" * 70)
    print("vLLM BENCHMARK — Qwen2.5-7B-Instruct (FP16)")
    print("=" * 70)

    # --- Cold Start ---
    print("\n[1] Cold Start...")
    t_cold = time.monotonic()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        max_model_len=2048,
        download_dir="/root/hf_cache",
        trust_remote_code=True,
        enforce_eager=True,  # Skip CUDA graph capture for fair cold start
    )

    cold_start_s = time.monotonic() - t_cold
    print(f"     Cold start: {cold_start_s:.2f}s")
    results["cold_start_s"] = round(cold_start_s, 2)

    # --- VRAM ---
    print("\n[2] VRAM Usage...")
    vram_allocated = torch.cuda.memory_allocated() / 1024**3
    vram_reserved = torch.cuda.memory_reserved() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"     Allocated: {vram_allocated:.2f} GB")
    print(f"     Reserved:  {vram_reserved:.2f} GB")
    print(f"     GPU Total: {vram_total:.1f} GB")
    results["vram_allocated_gb"] = round(vram_allocated, 2)
    results["vram_reserved_gb"] = round(vram_reserved, 2)
    results["vram_total_gb"] = round(vram_total, 1)

    # --- Throughput ---
    print(f"\n[3] Throughput ('{PROMPT}' → {MAX_TOKENS} tokens, greedy)...")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )

    # Warmup
    print("     Warmup...")
    _ = llm.generate([PROMPT], sampling_params)

    # Timed run
    print("     Timed run...")
    t0 = time.monotonic()
    outputs = llm.generate([PROMPT], sampling_params)
    t_gen = time.monotonic() - t0

    output = outputs[0]
    output_text = output.outputs[0].text
    num_tokens = len(output.outputs[0].token_ids)
    tps = num_tokens / t_gen if t_gen > 0 else 0

    print(f"     Tokens: {num_tokens}")
    print(f"     Time: {t_gen*1000:.1f}ms")
    print(f"     Throughput: {tps:.1f} tok/s")
    print(f"     Output: '{output_text[:100]}...'")

    results["num_tokens"] = num_tokens
    results["gen_time_ms"] = round(t_gen * 1000, 1)
    results["tps"] = round(tps, 1)
    results["output"] = output_text[:150]

    # Multi-run average
    print("     Running 3 more iterations for average...")
    times = []
    for _ in range(3):
        t0 = time.monotonic()
        out = llm.generate([PROMPT], sampling_params)
        times.append(time.monotonic() - t0)
    avg_time = sum(times) / len(times)
    avg_tps = num_tokens / avg_time
    print(f"     Avg: {avg_time*1000:.1f}ms, {avg_tps:.1f} tok/s")
    results["avg_tps"] = round(avg_tps, 1)

    print("\nvLLM results:", results)
    return results


# ============================================================================
# ZSE Benchmark
# ============================================================================

@app.function(
    gpu="A100",
    image=zse_image,
    timeout=3600,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def benchmark_zse():
    """Benchmark ZSE: cold start, throughput, VRAM."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import os
    import time
    import threading

    results = {"engine": "ZSE"}

    print("=" * 70)
    print("ZSE BENCHMARK — Qwen2.5-7B-Instruct (INT4)")
    print("=" * 70)

    # Init CUDA
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    ret = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert ret == 0, f"cuCtxCreate failed: {ret}"

    # --- Ensure .zse file exists ---
    zse_path = "/root/zse_cache/qwen2_7b.zse"

    if os.path.exists(zse_path) and os.path.getsize(zse_path) > 1_000_000_000:
        with open(zse_path, 'rb') as f:
            magic = f.read(4)
        if magic != b'ZSE\x01':
            print(f"[CACHE] Corrupt .zse, reconverting...")
            os.remove(zse_path)
            progress_file = zse_path + ".progress"
            if os.path.exists(progress_file):
                os.remove(progress_file)
        else:
            # Validate eos_id — old caches had wrong eos_id=2 for Qwen2
            try:
                from zse_engine.format.loader import ZSELoader
                _loader = ZSELoader(zse_path)
                _eos = _loader.tokenizer.special_tokens.eos_id
                _loader.close()
                if _eos < 100:
                    print(f"[CACHE] Stale .zse (eos_id={_eos}), reconverting...")
                    os.remove(zse_path)
                    progress_file = zse_path + ".progress"
                    if os.path.exists(progress_file):
                        os.remove(progress_file)
            except Exception:
                pass

    if not os.path.exists(zse_path):
        progress_file = zse_path + ".progress"
        if os.path.exists(progress_file):
            os.remove(progress_file)

        from huggingface_hub import snapshot_download
        print("[0] Downloading + converting Qwen2.5-7B-Instruct...")
        hf_dir = snapshot_download(
            MODEL_ID,
            cache_dir="/root/hf_cache",
            allow_patterns=["*.safetensors", "*.json"],
        )
        from zse_engine.format.convert import convert_hf_to_zse
        t0 = time.time()
        convert_hf_to_zse(hf_dir, zse_path)
        print(f"     Converted in {time.time()-t0:.1f}s")

        try:
            vol = modal.Volume.lookup("zse-model-cache")
            vol.commit()
        except Exception:
            pass
    else:
        print(f"[CACHE] Using cached .zse ({os.path.getsize(zse_path)/1024**3:.2f} GB)")

    # --- Cold Start ---
    print("\n[1] Cold Start...")
    t_cold = time.monotonic()

    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    engine = ZStreamerEngine(
        model_path=zse_path,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=2048,
            max_batch_seqs=8,
        ),
        max_seq_len=512,
        quiet=False,
    )

    cold_start_s = time.monotonic() - t_cold
    print(f"\n     Cold start: {cold_start_s:.2f}s")
    results["cold_start_s"] = round(cold_start_s, 2)

    # --- VRAM ---
    print("\n[2] VRAM Usage...")
    plan = engine._vram_plan
    try:
        actual = engine._allocator.utilization()
        vram_used = actual['used_bytes'] / 1024**3
        vram_free = actual['free_bytes'] / 1024**3
    except Exception:
        vram_used = (plan.weight_bytes + plan.kv_cache_bytes + plan.scratch_bytes) / 1024**3
        vram_free = 0

    vram_total = plan.total_vram / 1024**3
    vram_weights = plan.weight_bytes / 1024**3
    print(f"     Weights:   {vram_weights:.2f} GB")
    print(f"     Total used:{vram_used:.2f} GB")
    print(f"     GPU Total: {vram_total:.1f} GB")
    results["vram_weights_gb"] = round(vram_weights, 2)
    results["vram_used_gb"] = round(vram_used, 2)
    results["vram_total_gb"] = round(vram_total, 1)

    # --- Throughput ---
    print(f"\n[3] Throughput ('{PROMPT[:50]}...' → {MAX_TOKENS} tokens, greedy)...")

    tokenizer = engine._tokenizer
    print(f"     eos_id={tokenizer.special_tokens.eos_id}, vocab={engine._config.vocab_size}")

    # Helper: run a single request to completion, return (tokens, time_s)
    def run_request(prompt, max_tokens, label=""):
        toks = []
        done = threading.Event()
        first_t = [None]
        finish_info = [None]

        def _on_token(tid):
            if first_t[0] is None:
                first_t[0] = time.monotonic()
            toks.append(tid)

        def _on_finish(output):
            finish_info[0] = output.finish_reason
            done.set()

        t0 = time.monotonic()
        req_id = engine.add_request(
            prompt=prompt, max_tokens=max_tokens, temperature=0.0,
            on_token=_on_token, on_finish=_on_finish,
        )

        # Step until done — generous limit
        for _ in range(max_tokens + 100):
            engine.step()
            if done.is_set():
                break

        elapsed = time.monotonic() - t0
        ttft = (first_t[0] - t0) * 1000 if first_t[0] else 0

        if label:
            n = len(toks)
            tps_val = n / elapsed if elapsed > 0 and n > 1 else 0
            reason = finish_info[0]
            print(f"     {label}: {n} tokens, {tps_val:.1f} tok/s, {ttft:.0f}ms TTFT, reason={reason}")
            if n > 0:
                text = tokenizer.decode(toks)
                print(f"     Output: '{text[:80]}...'")

        return toks, elapsed, ttft, finish_info[0]

    # Warmup
    run_request(PROMPT, 32, label="Warmup")

    # Timed run
    tokens_received, t_gen, ttft, finish_reason = run_request(PROMPT, MAX_TOKENS, label="Timed")
    num_tokens = len(tokens_received)
    tps = num_tokens / t_gen if t_gen > 0 and num_tokens > 1 else 0

    try:
        output_text = tokenizer.decode(tokens_received)
    except Exception:
        output_text = str(tokens_received[:10])

    results["num_tokens"] = num_tokens
    results["ttft_ms"] = round(ttft, 1)
    results["gen_time_ms"] = round(t_gen * 1000, 1)
    results["tps"] = round(tps, 1)
    results["output"] = output_text[:150]

    # Raw decode timing (direct model_runner, bypass scheduler)
    print("     Raw decode timing...")
    import time as _time
    runner = engine._model_runner
    try:
        raw_prompt = "Explain the theory of relativity in detail"
        raw_tokens = tokenizer.encode(raw_prompt)
        raw_seq = 8888
        logits = runner.prefill(raw_tokens, raw_seq)

        import struct as _struct
        vocab = engine._config.vocab_size
        vals = _struct.unpack(f'<{vocab}e', logits[:vocab*2])
        first_tok = max(range(vocab), key=lambda i: vals[i])

        for i in range(3):
            runner.decode_step(first_tok, raw_seq, len(raw_tokens) + i)

        t0 = _time.monotonic()
        for i in range(20):
            runner.decode_step(first_tok, raw_seq, len(raw_tokens) + 3 + i)
        t1 = _time.monotonic()
        raw_ms = (t1 - t0) / 20 * 1000
        raw_tps = 1000 / raw_ms
        print(f"     Raw decode_step: {raw_ms:.1f}ms = {raw_tps:.1f} tok/s")
        results["raw_step_ms"] = round(raw_ms, 1)
        results["raw_tps"] = round(raw_tps, 1)

        kv = engine._kv_cache
        try:
            kv.mark_idle(raw_seq)
            kv.free_sequence(raw_seq)
        except Exception:
            pass
    except Exception as e:
        print(f"     Raw timing failed: {e}")
        import traceback
        traceback.print_exc()

    # Multi-run average
    print("     Running 2 more iterations for average...")
    run_tps_list = [tps]  # Include the timed run
    for run_i in range(2):
        toks_r, t_r, _, _ = run_request(PROMPT, MAX_TOKENS)
        n = len(toks_r)
        r_tps = n / t_r if t_r > 0 and n > 1 else 0
        run_tps_list.append(r_tps)
        print(f"     Run {run_i+1}: {n} tokens, {r_tps:.1f} tok/s")

    avg_tps = sum(run_tps_list) / len(run_tps_list) if run_tps_list else tps
    print(f"     Avg: {avg_tps:.1f} tok/s")
    results["avg_tps"] = round(avg_tps, 1)

    print("\nZSE results:", results)
    return results


# ============================================================================
# Main — Run both and compare
# ============================================================================

@app.local_entrypoint()
def main():
    print("=" * 70)
    print("ZSE vs vLLM BENCHMARK — Qwen2.5-7B-Instruct on A100")
    print("=" * 70)
    print()

    print("Running vLLM benchmark...")
    vllm_results = benchmark_vllm.remote()
    print("\nvLLM done.\n")

    print("Running ZSE benchmark...")
    zse_results = benchmark_zse.remote()
    print("\nZSE done.\n")

    # --- Comparison Table ---
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS — Qwen2.5-7B-Instruct on A100")
    print("=" * 70)

    v = vllm_results
    z = zse_results

    def ratio(a, b, better="lower"):
        if a == 0 or b == 0:
            return "N/A"
        if better == "lower":
            return f"{a/b:.1f}x" if a > b else f"{b/a:.1f}x"
        else:
            return f"{a/b:.1f}x" if a > b else f"{b/a:.1f}x"

    print(f"\n  {'Metric':<25} {'vLLM (FP16)':<20} {'ZSE (INT4)':<20} {'Winner':<15}")
    print(f"  {'-'*25} {'-'*20} {'-'*20} {'-'*15}")

    # Cold start
    vc = v.get('cold_start_s', 0)
    zc = z.get('cold_start_s', 0)
    winner = "ZSE ✅" if zc < vc else "vLLM"
    speedup = f"({vc/zc:.1f}x faster)" if zc > 0 and zc < vc else ""
    print(f"  {'Cold start':<25} {vc}s{'':<14} {zc}s{'':<14} {winner} {speedup}")

    # Throughput (avg)
    vt = v.get('avg_tps', v.get('tps', 0))
    zt = z.get('avg_tps', z.get('tps', 0))
    winner = "ZSE ✅" if zt > vt else "vLLM ✅" if vt > zt else "Tie"
    print(f"  {'Throughput (avg tok/s)':<25} {vt:<20} {zt:<20} {winner}")

    # Raw throughput (ZSE only)
    zr = z.get('raw_tps', 0)
    if zr:
        print(f"  {'Raw decode tok/s':<25} {'N/A':<20} {zr:<20} {'(ZSE only)'}")

    # VRAM
    vv = v.get('vram_allocated_gb', v.get('vram_reserved_gb', 0))
    zv = z.get('vram_used_gb', 0)
    winner = "ZSE ✅" if zv < vv else "vLLM"
    savings = f"({vv/zv:.1f}x less)" if zv > 0 and zv < vv else ""
    print(f"  {'VRAM used':<25} {vv} GB{'':<13} {zv} GB{'':<13} {winner} {savings}")

    # Weights
    zw = z.get('vram_weights_gb', 0)
    print(f"  {'Weight VRAM':<25} {'~14 GB (FP16)':<20} {f'{zw} GB (INT4)':<20} {'ZSE ✅'}")

    # Dependencies
    print(f"  {'Dependencies':<25} {'PyTorch+Triton':<20} {'Zero':<20} {'ZSE ✅'}")

    # Output
    print(f"\n  vLLM output: {v.get('output', '?')[:80]}...")
    print(f"  ZSE output:  {z.get('output', '?')[:80]}...")

    # Overall
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if zc < vc:
        print(f"  🚀 ZSE cold start: {vc/zc:.1f}x faster than vLLM ({zc}s vs {vc}s)")
    if zv < vv and zv > 0:
        print(f"  💾 ZSE VRAM: {vv/zv:.1f}x less than vLLM ({zv}GB vs {vv}GB)")
    if zt > 0 and vt > 0:
        pct = (zt / vt) * 100
        print(f"  ⚡ ZSE throughput: {pct:.0f}% of vLLM ({zt} vs {vt} tok/s)")
    print(f"  📦 ZSE dependencies: ZERO (vLLM needs PyTorch + Triton + CUDA toolkit)")
    print()
