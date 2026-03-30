"""
ZSE Tensor Parallelism - Multi-GPU Test

Tests tensor parallelism with 2x A10G GPUs on Modal.
Compares single-GPU vs 2-GPU TP for a 7B model.

Usage:
    modal run deploy/test_tensor_parallel.py
"""

import modal
import os
import sys
import time

app = modal.App("zse-tensor-parallel-test")

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(DEPLOY_DIR)

test_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "accelerate>=0.25.0",
        "huggingface_hub>=0.20.0",
        "bitsandbytes>=0.41.0",
        "pynvml",
        "rich",
        "typer",
        "sentencepiece",
        "protobuf",
        "psutil",
    )
)

test_image_with_code = test_image.add_local_dir(
    ZSE_ROOT, remote_path="/root/zse"
)


@app.function(
    image=test_image_with_code,
    gpu="a10g:2",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_tensor_parallel():
    """Test tensor parallelism with 2x A10G GPUs."""
    import traceback
    try:
        return _run_test()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n❌ TEST FAILED: {e}\n{tb}", flush=True)
        return {"error": str(e), "traceback": tb}


def _run_test():
    """Test tensor parallelism with 2x A10G GPUs."""
    print(">>> _run_test() entered", flush=True)
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        cwd="/root/zse",
    )
    if "/root/zse" not in sys.path:
        sys.path.insert(0, "/root/zse")

    import torch
    from huggingface_hub import hf_hub_download

    gpu_count = torch.cuda.device_count()
    print(f"\n{'='*60}", flush=True)
    print(f"ZSE Tensor Parallelism Test", flush=True)
    print(f"GPUs: {gpu_count}", flush=True)
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({vram:.1f} GB)", flush=True)
    print(f"{'='*60}\n", flush=True)

    assert gpu_count >= 2, f"Need >= 2 GPUs, got {gpu_count}"

    # ── Download .zse model ─────────────────────────────────────
    TARGET_REPO = "zse-zllm/Qwen2.5-7B-Instruct-zse-int4"
    TARGET_FILE = "Qwen2.5-7B-Instruct-zse-int4.zse"
    PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes."
    MAX_TOKENS = 200

    print(f"Downloading: {TARGET_REPO}", flush=True)
    target_path = hf_hub_download(repo_id=TARGET_REPO, filename=TARGET_FILE)
    print(f"  Size: {os.path.getsize(target_path) / 1024**3:.2f} GB", flush=True)
    print(flush=True)

    # ── Test 1: Single GPU baseline ─────────────────────────────
    print(f"{'─'*60}", flush=True)
    print("TEST 1: Single GPU Baseline (GPU 0 only)", flush=True)
    print(f"{'─'*60}", flush=True)

    from zse.format.reader_v2 import load_zse_model

    model_1gpu, tokenizer, info = load_zse_model(target_path, device="cuda:0")
    model_1gpu.eval()
    vram_1gpu = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  VRAM: {vram_1gpu:.2f} GB", flush=True)

    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")
    prompt_len = input_ids.shape[1]
    print(f"  Prompt tokens: {prompt_len}", flush=True)

    # Warmup
    with torch.no_grad():
        _ = model_1gpu.generate(input_ids, max_new_tokens=5, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out_1gpu = model_1gpu.generate(
            input_ids, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=True,
        )
    torch.cuda.synchronize()
    time_1gpu = time.perf_counter() - start
    tokens_1gpu = out_1gpu.shape[1] - prompt_len
    tps_1gpu = tokens_1gpu / time_1gpu
    text_1gpu = tokenizer.decode(out_1gpu[0, prompt_len:], skip_special_tokens=True)

    print(f"  Tokens:  {tokens_1gpu}", flush=True)
    print(f"  Time:    {time_1gpu:.2f}s", flush=True)
    print(f"  Speed:   {tps_1gpu:.1f} tok/s", flush=True)
    print(f"  Output:  {text_1gpu[:150]}...", flush=True)
    print(flush=True)

    # Free single GPU model
    del model_1gpu
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ── Test 2: Tensor Parallel (2 GPUs) ────────────────────────
    print(f"{'─'*60}", flush=True)
    print("TEST 2: Tensor Parallel (2x GPU, TP=2)", flush=True)
    print(f"{'─'*60}", flush=True)

    from zse.core.zdistributed.worker import TPCoordinator

    print("  Spawning 2 TP worker processes...", flush=True)
    coord = TPCoordinator(
        model_path=target_path,
        tp_size=2,
    )
    coord.start(verbose=True)
    
    # Use HF generate through the coordinator
    input_ids_tp = tokenizer(prompt_text, return_tensors="pt").input_ids
    
    # Warmup
    print("  Warming up...", flush=True)
    warmup_ids = coord.generate(input_ids_tp, max_new_tokens=5, do_sample=False, use_cache=True)
    print(f"  Warmup output shape: {warmup_ids.shape}", flush=True)

    # Benchmark
    print("  Benchmarking...", flush=True)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    start = time.perf_counter()
    out_tp = coord.generate(
        input_ids_tp, max_new_tokens=MAX_TOKENS, do_sample=False, use_cache=True,
    )
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    time_tp = time.perf_counter() - start
    tokens_tp = out_tp.shape[1] - prompt_len
    tps_tp = tokens_tp / time_tp
    text_tp = tokenizer.decode(out_tp[0, prompt_len:], skip_special_tokens=True)

    print(f"  Tokens:  {tokens_tp}", flush=True)
    print(f"  Time:    {time_tp:.2f}s", flush=True)
    print(f"  Speed:   {tps_tp:.1f} tok/s", flush=True)
    print(f"  Output:  {text_tp[:150]}...", flush=True)
    
    # Report per-GPU VRAM (child processes held memory, read from parent)
    for i in range(2):
        vram = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i} VRAM (visible from parent): {vram:.2f} GB", flush=True)
    print(flush=True)
    
    coord.shutdown()

    # ── Summary ─────────────────────────────────────────────────
    speedup = tps_tp / tps_1gpu if tps_1gpu > 0 else 0

    print(f"{'='*60}", flush=True)
    print(f"RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Single GPU:   {tps_1gpu:.1f} tok/s  |  {vram_1gpu:.2f} GB VRAM", flush=True)
    print(f"  TP=2:         {tps_tp:.1f} tok/s", flush=True)
    print(f"  Speedup:      {speedup:.2f}x", flush=True)
    print(f"{'='*60}", flush=True)

    # Verify output is coherent
    assert tokens_tp > 10, f"Too few tokens generated: {tokens_tp}"
    assert len(text_tp) > 50, f"Output too short: {text_tp}"
    print(f"\n✅ Tensor parallelism test PASSED", flush=True)

    return {
        "single_gpu_tps": tps_1gpu,
        "tp2_tps": tps_tp,
        "speedup": speedup,
        "vram_1gpu": vram_1gpu,
    }


@app.local_entrypoint()
def main():
    result = test_tensor_parallel.remote()
    print(f"\nRemote result: {result}")
