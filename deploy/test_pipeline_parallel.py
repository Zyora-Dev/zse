"""
ZSE Pipeline Parallelism - Multi-GPU Test

Tests pipeline parallelism with 2x A10G GPUs on Modal.
Compares single-GPU vs 2-GPU PP for a 7B model.

Usage:
    modal run deploy/test_pipeline_parallel.py
"""

import modal
import os
import sys
import time

app = modal.App("zse-pipeline-parallel-test")

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

test_image_with_code = test_image.add_local_dir(ZSE_ROOT, remote_path="/root/zse")


@app.function(
    image=test_image_with_code,
    gpu="a10g:2",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_pipeline_parallel():
    """Test pipeline parallelism with 2x A10G GPUs."""
    import traceback

    try:
        return _run_test()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n❌ TEST FAILED: {e}\n{tb}", flush=True)
        return {"error": str(e), "traceback": tb}


def _run_test():
    print(">>> PP test entered", flush=True)
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
    print(f"\n{'=' * 60}", flush=True)
    print(f"ZSE Pipeline Parallelism Test", flush=True)
    print(f"GPUs: {gpu_count}", flush=True)
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({vram:.1f} GB)", flush=True)
    print(f"{'=' * 60}\n", flush=True)

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
    print(f"{'─' * 60}", flush=True)
    print("TEST 1: Single GPU Baseline (GPU 0 only)", flush=True)
    print(f"{'─' * 60}", flush=True)

    from zse.format.reader_v2 import load_zse_model

    model_1gpu, tokenizer, info = load_zse_model(target_path, device="cuda:0")
    model_1gpu.eval()
    vram_1gpu = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  VRAM: {vram_1gpu:.2f} GB", flush=True)

    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
            input_ids,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            use_cache=True,
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
    import gc

    gc.collect()

    # ── Test 2: Pipeline Parallel (2 GPUs) ──────────────────────
    print(f"{'─' * 60}", flush=True)
    print("TEST 2: Pipeline Parallel (2x GPU, PP=2)", flush=True)
    print(f"{'─' * 60}", flush=True)

    from zse.core.zdistributed.pipeline_parallel import PPCoordinator

    print("  Spawning 2 PP stage processes...", flush=True)
    coord = PPCoordinator(
        model_path=target_path,
        pp_size=2,
    )
    coord.start(verbose=True)

    # Use generate through the coordinator
    input_ids_pp = tokenizer(prompt_text, return_tensors="pt").input_ids

    # Warmup
    print("  Warming up...", flush=True)
    warmup_ids = coord.generate(input_ids_pp, max_new_tokens=5, do_sample=False)
    print(f"  Warmup output shape: {warmup_ids.shape}", flush=True)

    # Benchmark
    print("  Benchmarking...", flush=True)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    start = time.perf_counter()
    out_pp = coord.generate(
        input_ids_pp,
        max_new_tokens=MAX_TOKENS,
        do_sample=False,
    )
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    time_pp = time.perf_counter() - start
    tokens_pp = out_pp.shape[1] - prompt_len
    tps_pp = tokens_pp / time_pp
    text_pp = tokenizer.decode(out_pp[0, prompt_len:], skip_special_tokens=True)

    print(f"  Tokens:  {tokens_pp}", flush=True)
    print(f"  Time:    {time_pp:.2f}s", flush=True)
    print(f"  Speed:   {tps_pp:.1f} tok/s", flush=True)
    print(f"  Output:  {text_pp[:150]}...", flush=True)

    for i in range(2):
        vram = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i} VRAM (parent view): {vram:.2f} GB", flush=True)
    print(flush=True)

    coord.shutdown()

    # ── Summary ─────────────────────────────────────────────────
    speedup = tps_pp / tps_1gpu if tps_1gpu > 0 else 0

    print(f"{'=' * 60}", flush=True)
    print(f"RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Single GPU:   {tps_1gpu:.1f} tok/s  |  {vram_1gpu:.2f} GB VRAM", flush=True)
    print(f"  PP=2:         {tps_pp:.1f} tok/s", flush=True)
    print(f"  Speedup:      {speedup:.2f}x", flush=True)
    print(f"{'=' * 60}", flush=True)

    assert tokens_pp > 10, f"Too few tokens: {tokens_pp}"
    assert len(text_pp) > 50, f"Output too short: {text_pp}"
    print(f"\n✅ Pipeline parallelism test PASSED", flush=True)

    return {
        "single_gpu_tps": tps_1gpu,
        "pp2_tps": tps_pp,
        "speedup": speedup,
        "vram_1gpu": vram_1gpu,
    }


@app.local_entrypoint()
def main():
    result = test_pipeline_parallel.remote()
    print(f"\nRemote result: {result}")
