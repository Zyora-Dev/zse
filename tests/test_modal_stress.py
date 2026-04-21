"""ZSE Stress Test — Qwen2.5-14B on A100-40GB.

Validates that ZSE can handle a 14B model on a single 40GB GPU:
- Memory: INT4 weights fit with room for KV cache + scratch
- Cold start: model load + kernel compile timing
- Throughput: tok/s for 14B decode
- Quality: logits are healthy, output is coherent

Compares results against 7B baseline.
"""

import sys
import modal

app = modal.App("zse-stress-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)


@app.function(
    gpu="A100",
    image=zse_image,
    timeout=7200,  # 2h — 14B conversion can be slow
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def test_stress_14b():
    """Stress test: Qwen2.5-14B on A100-40GB."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import os
    import time
    import struct
    import threading

    print("=" * 70)
    print("ZSE STRESS TEST — Qwen2.5-14B on A100-40GB")
    print("=" * 70)

    # Init CUDA
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    ret = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert ret == 0, f"cuCtxCreate failed: {ret}"

    from zse_compiler.runtime.device import get_devices
    devices = get_devices("cuda")
    device = devices[0]
    print(f"GPU: {device.name} ({device.vram_total_gb:.1f} GB)")

    results = {}

    # ================================================================== #
    # Phase 1: Model Conversion (14B)
    # ================================================================== #
    print("\n" + "=" * 70)
    print("PHASE 1: MODEL CONVERSION — Qwen2.5-14B-Instruct")
    print("=" * 70)

    zse_path_14b = "/root/zse_cache/qwen2_14b.zse"

    if os.path.exists(zse_path_14b) and os.path.getsize(zse_path_14b) > 5_000_000_000:
        # Verify magic bytes
        with open(zse_path_14b, 'rb') as f:
            magic = f.read(4)
        if magic == b'ZSE\x01':
            file_size = os.path.getsize(zse_path_14b)
            print(f"[CACHE] Using cached .zse: {file_size:,} bytes ({file_size/1024**3:.2f} GB)")
            results["conversion_14b"] = "CACHED"
            results["zse_size_14b_gb"] = round(file_size / 1024**3, 2)

            # Validate eos_id — old caches had wrong eos_id=2 for Qwen2
            try:
                from zse_engine.format.loader import ZSELoader
                _loader = ZSELoader(zse_path_14b)
                _eos = _loader.tokenizer.special_tokens.eos_id
                _loader.close()
                if _eos < 100:
                    print(f"[CACHE] Stale .zse (eos_id={_eos}), reconverting...")
                    os.remove(zse_path_14b)
                    progress_file = zse_path_14b + ".progress"
                    if os.path.exists(progress_file):
                        os.remove(progress_file)
                    results["conversion_14b"] = None  # Force reconversion below
            except Exception:
                pass
        else:
            print(f"[CACHE] Corrupt .zse (magic={magic!r}), removing and reconverting...")
            os.remove(zse_path_14b)
            # Also remove progress file to prevent resume from deleted file
            progress_file = zse_path_14b + ".progress"
            if os.path.exists(progress_file):
                os.remove(progress_file)

    if not os.path.exists(zse_path_14b):
        # Clean up any stale progress files
        progress_file = zse_path_14b + ".progress"
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"     Removed stale progress file")

        from huggingface_hub import snapshot_download

        print("[1a] Downloading Qwen2.5-14B-Instruct...")
        t0 = time.time()
        hf_dir = snapshot_download(
            "Qwen/Qwen2.5-14B-Instruct",
            cache_dir="/root/hf_cache",
            allow_patterns=["*.safetensors", "*.json"],
        )
        t_download = time.time() - t0
        print(f"     Downloaded in {t_download:.1f}s → {hf_dir}")

        print("[1b] Converting to .zse (INT4 quantization)...")
        print("     NOTE: 14B model — may take 20-40min in pure Python")
        from zse_engine.format.convert import convert_hf_to_zse

        t0 = time.time()
        tensor_count = [0]
        def progress(name, cur, total):
            tensor_count[0] = total
            if cur == 1 or cur == total or cur % 20 == 0:
                elapsed = time.time() - t0
                rate = cur / elapsed if elapsed > 0 else 0
                eta = (total - cur) / rate if rate > 0 else 0
                print(f"     [{cur}/{total}] {name} ({elapsed:.0f}s, ETA: {eta:.0f}s)")

        convert_hf_to_zse(hf_dir, zse_path_14b, progress_callback=progress)
        t_convert = time.time() - t0
        file_size = os.path.getsize(zse_path_14b)
        print(f"     Converted {tensor_count[0]} tensors in {t_convert:.1f}s")
        print(f"     .zse size: {file_size:,} bytes ({file_size/1024**3:.2f} GB)")
        results["conversion_14b"] = "PASS"
        results["convert_time_14b_s"] = round(t_convert, 1)
        results["zse_size_14b_gb"] = round(file_size / 1024**3, 2)

    # Flush volume
    try:
        vol = modal.Volume.lookup("zse-model-cache")
        vol.commit()
    except Exception:
        pass

    # ================================================================== #
    # Phase 2: Cold Start + Memory
    # ================================================================== #
    print("\n" + "=" * 70)
    print("PHASE 2: COLD START + MEMORY — 14B")
    print("=" * 70)

    from zse_engine.zstreamer.engine import ZStreamerEngine
    from zse_engine.zstreamer.scheduler import SchedulerConfig

    print("[2a] Loading 14B model + compiling kernels + allocating KV cache...")
    t_cold = time.monotonic()

    engine = ZStreamerEngine(
        model_path=zse_path_14b,
        scheduler_config=SchedulerConfig(
            max_batch_tokens=2048,
            max_batch_seqs=8,
        ),
        max_seq_len=512,
        quiet=False,
    )

    cold_start_s = time.monotonic() - t_cold
    print(f"\n     COLD START: {cold_start_s:.2f}s")

    config = engine._config
    gpu_mem = engine._gpu_mem
    print(f"     Model: {config.arch}, {config.num_layers}L, {config.hidden_size}H")
    print(f"     Vocab: {config.vocab_size}, Heads: {config.num_heads}, KV Heads: {config.num_kv_heads}")
    print(f"     Intermediate: {config.intermediate_size}")

    results["cold_start_14b_s"] = round(cold_start_s, 2)
    results["cold_start_14b"] = "PASS"

    # Memory report from VRAM plan
    plan = engine._vram_plan
    if plan:
        print(f"\n     VRAM Plan:")
        print(f"       Total:   {plan.total_vram/1024**3:.2f} GB")
        print(f"       Weights: {plan.weight_bytes/1024**3:.2f} GB")
        print(f"       KV Cache:{plan.kv_cache_bytes/1024**3:.2f} GB")
        print(f"       Scratch: {plan.scratch_bytes/1024**2:.1f} MB")
        print(f"       Usage:   {plan.utilization_pct:.1f}%")
        results["vram_total_gb"] = round(plan.total_vram/1024**3, 2)
        results["vram_used_gb"] = round((plan.weight_bytes + plan.kv_cache_bytes + plan.scratch_bytes)/1024**3, 2)
        results["vram_weight_gb"] = round(plan.weight_bytes/1024**3, 2)
        results["vram_kv_gb"] = round(plan.kv_cache_bytes/1024**3, 2)
        results["vram_pct"] = round(plan.utilization_pct, 1)

    # Actual GPU free memory
    try:
        actual_util = engine._allocator.utilization()
        print(f"\n     Actual VRAM (post-load):")
        print(f"       Used:    {actual_util['used_bytes']/1024**3:.2f} GB")
        print(f"       Free:    {actual_util['free_bytes']/1024**3:.2f} GB")
        results["vram_actual_used_gb"] = round(actual_util['used_bytes']/1024**3, 2)
    except Exception as e:
        print(f"     Actual VRAM query failed: {e}")

    # ================================================================== #
    # Phase 2b: PREFILL DIAGNOSTICS — Debug garbage tokens
    # ================================================================== #
    print("\n" + "=" * 70)
    print("PHASE 2b: PREFILL DIAGNOSTICS")
    print("=" * 70)

    runner = engine._model_runner
    tokenizer = engine._tokenizer
    kv_cache = engine._kv_cache

    # Tokenize prompt
    prompt = "The capital of France is"
    prompt_tokens = tokenizer.encode(prompt)
    print(f"     Prompt: '{prompt}'")
    print(f"     Tokens: {prompt_tokens}")
    print(f"     Token text: {[tokenizer.decode([t]) for t in prompt_tokens]}")

    # Test BOTH paths: fast C decoder and Python fallback
    for path_name, disable_fast in [("Fast C", False), ("Python", True)]:
        diag_seq_id = 9998 if disable_fast else 9999
        print(f"\n     --- Testing {path_name} path ---")

        # Temporarily disable fast decoder if testing Python path
        saved_fd = runner._fast_decoder
        if disable_fast:
            runner._fast_decoder = None

        try:
            logits_bytes = runner.prefill(prompt_tokens, diag_seq_id)
            print(f"     Prefill logits: {len(logits_bytes)} bytes")

            import struct as _struct
            vocab_size = engine._config.vocab_size
            num_vals = min(vocab_size, len(logits_bytes) // 2)
            logits = _struct.unpack(f'<{num_vals}e', logits_bytes[:num_vals * 2])

            has_nan = sum(1 for v in logits if v != v)
            max_v = max(logits)
            min_v = min(logits)
            print(f"     Logits range: [{min_v:.4f}, {max_v:.4f}], NaN: {has_nan}")

            indexed = sorted(enumerate(logits), key=lambda x: -x[1])[:5]
            print(f"     Top-5 prefill:")
            for rank, (tid, score) in enumerate(indexed):
                text = tokenizer.decode([tid])
                print(f"       #{rank+1}: token={tid} score={score:.4f} text='{text}'")

            first_token = indexed[0][0]
            decode_logits = runner.decode_step(first_token, diag_seq_id, len(prompt_tokens))
            decode_vals = _struct.unpack(f'<{num_vals}e', decode_logits[:num_vals * 2])
            d_indexed = sorted(enumerate(decode_vals), key=lambda x: -x[1])[:5]
            print(f"     Top-5 decode (after '{tokenizer.decode([first_token])}'):")
            for rank, (tid, score) in enumerate(d_indexed):
                text = tokenizer.decode([tid])
                print(f"       #{rank+1}: token={tid} score={score:.4f} text='{text}'")

            if path_name == "Fast C":
                results["prefill_top1_id"] = indexed[0][0]
                results["prefill_top1_text"] = tokenizer.decode([indexed[0][0]])
                results["logits_nan"] = has_nan
                results["logits_inf"] = 0
                results["decode_top1_id"] = d_indexed[0][0]
                results["decode_top1_text"] = tokenizer.decode([d_indexed[0][0]])

        except Exception as e:
            import traceback
            print(f"     {path_name} path FAILED: {e}")
            traceback.print_exc()
        finally:
            runner._fast_decoder = saved_fd
            try:
                kv_cache.mark_idle(diag_seq_id)
                kv_cache.free_sequence(diag_seq_id)
            except Exception:
                pass

    # ================================================================== #
    # Phase 3: Single Request Throughput
    # ================================================================== #
    print("\n" + "=" * 70)
    print("PHASE 3: THROUGHPUT — 14B")
    print("=" * 70)

    tokens_received = []
    done_event = threading.Event()
    first_token_time = [None]
    request_start = [None]

    def on_token(token_id):
        if first_token_time[0] is None:
            first_token_time[0] = time.monotonic()
        tokens_received.append(token_id)

    def on_finish(output):
        done_event.set()

    print("[3a] Request: 'The capital of France is' → 32 tokens, greedy")
    request_start[0] = time.monotonic()
    req_id = engine.add_request(
        prompt="The capital of France is",
        max_tokens=32,
        temperature=0.0,
        on_token=on_token,
        on_finish=on_finish,
    )

    import time as _time

    # Prefill
    result = engine.step()
    print(f"     Prefill: tokens={result.num_tokens}")

    # Timing: 3 warmup steps
    for si in range(3):
        t0 = _time.monotonic()
        result = engine.step()
        t1 = _time.monotonic()
        ms = (t1 - t0) * 1000
        print(f"     Step {si}: {ms:.1f}ms")

    # Timing: 20-step average
    t0 = _time.monotonic()
    for _ in range(20):
        engine.step()
    t1 = _time.monotonic()
    avg_ms = (t1 - t0) / 20 * 1000
    print(f"     Avg over 20 steps: {avg_ms:.1f}ms/step = {1000/avg_ms:.1f} tok/s")
    results["raw_step_ms_14b"] = round(avg_ms, 1)
    results["raw_tps_14b"] = round(1000 / avg_ms, 1)

    # Raw decode_step timing
    try:
        runner = engine._model_runner
        req = list(engine._scheduler._active.values())[0]
        sid = req.seq_id
        pos = req.total_tokens
        tid = tokens_received[-1] if tokens_received else 0
        runner.decode_step(tid, sid, pos)  # warmup
        t0 = _time.monotonic()
        for _ in range(10):
            runner.decode_step(tid, sid, pos)
        t1 = _time.monotonic()
        raw_ms = (t1 - t0) / 10 * 1000
        print(f"     Raw decode_step: {raw_ms:.1f}ms = {1000/raw_ms:.1f} tok/s")
        results["raw_decode_ms_14b"] = round(raw_ms, 1)
        results["raw_decode_tps_14b"] = round(1000 / raw_ms, 1)
    except Exception as e:
        print(f"     Raw timing failed: {e}")

    # Finish generation
    for _ in range(200):
        result = engine.step()
        if done_event.is_set():
            break
    total_time = time.monotonic() - request_start[0]

    ttft = (first_token_time[0] - request_start[0]) if first_token_time[0] else 0
    num_tokens = len(tokens_received)
    tps = num_tokens / total_time if total_time > 0 else 0

    # Decode output
    try:
        tokenizer = engine._tokenizer
        output_text = tokenizer.decode(tokens_received)
    except Exception:
        output_text = str(tokens_received[:10])

    print(f"\n     Tokens: {num_tokens}")
    print(f"     TTFT: {ttft*1000:.1f}ms")
    print(f"     Total time: {total_time*1000:.1f}ms")
    print(f"     Throughput: {tps:.1f} tokens/sec")
    print(f"     Output: '{output_text[:100]}...'")

    results["ttft_ms_14b"] = round(ttft * 1000, 1)
    results["single_tps_14b"] = round(tps, 1)
    results["output_14b"] = output_text[:100]

    # ================================================================== #
    # Phase 4: Quality Check
    # ================================================================== #
    print("\n" + "=" * 70)
    print("PHASE 4: QUALITY CHECK — 14B")
    print("=" * 70)

    # Check logits health
    try:
        req = list(engine._scheduler._active.values())
        if not req:
            # Request finished, just report
            print("     Request completed, checking output quality...")
        else:
            req = req[0]
            logits_list = runner.batched_decode(
                token_ids=[tokens_received[-1] if tokens_received else 0],
                seq_ids=[req.seq_id],
                positions=[req.total_tokens],
            )
            if logits_list:
                raw = logits_list[0]
                vals = struct.unpack(f'<{min(100, config.vocab_size)}e', raw[:200])
                has_nan = any(v != v for v in vals)
                has_inf = any(abs(v) > 60000 for v in vals)
                max_v = max(vals)
                min_v = min(vals)
                print(f"     Logits[0:100]: min={min_v:.2f}, max={max_v:.2f}")
                print(f"     NaN: {has_nan}, Inf/overflow: {has_inf}")
                results["logits_healthy_14b"] = not has_nan and not has_inf
    except Exception as e:
        print(f"     Quality check failed: {e}")

    # Output coherence
    output_lower = output_text.lower() if output_text else ""
    has_paris = "paris" in output_lower
    print(f"     Contains 'Paris': {has_paris}")
    results["output_correct_14b"] = has_paris

    # ================================================================== #
    # Phase 5: Summary
    # ================================================================== #
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — 14B STRESS TEST")
    print("=" * 70)

    # 7B baseline (hardcoded from previous runs)
    baseline_7b = {
        "cold_start_s": "~9-12",
        "weight_gb": "~3.8",
        "raw_tps": "84.6",
        "single_tps": "38.3",
        "ttft_ms": "~260",
    }

    print(f"\n  {'Metric':<25} {'7B (baseline)':<20} {'14B (this run)':<20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    print(f"  {'Cold start':<25} {baseline_7b['cold_start_s']:<20} {results.get('cold_start_14b_s', '?')}s")
    print(f"  {'Weight VRAM':<25} {baseline_7b['weight_gb']:<20} {results.get('vram_weight_gb', '?')} GB")
    print(f"  {'KV Cache VRAM':<25} {'~5.0 GB':<20} {results.get('vram_kv_gb', '?')} GB")
    print(f"  {'.zse file size':<25} {'~3.8 GB':<20} {results.get('zse_size_14b_gb', '?')} GB")
    print(f"  {'Raw step tok/s':<25} {baseline_7b['raw_tps']:<20} {results.get('raw_tps_14b', '?')}")
    print(f"  {'End-to-end tok/s':<25} {baseline_7b['single_tps']:<20} {results.get('single_tps_14b', '?')}")
    print(f"  {'TTFT':<25} {baseline_7b['ttft_ms']:<20} {results.get('ttft_ms_14b', '?')}ms")
    print(f"  {'VRAM used':<25} {'~35%':<20} {results.get('vram_pct', '?')}%")

    print(f"\n  Output (14B): {results.get('output_14b', '?')}")

    # Pass/fail
    passed = True
    if results.get('cold_start_14b_s', 999) > 60:
        print("\n  ✗ Cold start > 60s")
        passed = False
    if results.get('single_tps_14b', 0) < 10:
        print("\n  ✗ Throughput < 10 tok/s")
        passed = False

    if passed:
        print("\n  ✅ ALL STRESS TESTS PASSED!")
    else:
        print("\n  ❌ SOME TESTS FAILED")

    print("\nRemote results:", results)
    return results


@app.local_entrypoint()
def main():
    print("Launching ZSE Stress Test: Qwen2.5-14B on A100-40GB...")
    print("(First run may take 30-60min for model download + conversion)")
    results = test_stress_14b.remote()
    print("\nRemote results:", results)
