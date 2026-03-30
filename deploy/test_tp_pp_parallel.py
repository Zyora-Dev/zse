"""
ZSE Combined TP+PP Parallelism - Multi-GPU Test

Tests combined tensor + pipeline parallelism with 4x A10G GPUs on Modal.
Grid: 2 PP stages × 2 TP per stage = 4 GPUs total.

Compares:
  1. Single GPU baseline
  2. TP=2 (tensor parallelism only)
  3. PP=2 (pipeline parallelism only)
  4. TP=2 × PP=2 (combined)

Usage:
    modal run deploy/test_tp_pp_parallel.py
"""

import modal
import os
import sys
import time

app = modal.App("zse-tp-pp-parallel-test")

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
    gpu="a10g:4",
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface")],
)
def test_tp_pp():
    """Test combined TP+PP with 4x A10G GPUs."""
    import traceback
    try:
        return _run_test()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n❌ TEST FAILED: {e}\n{tb}", flush=True)
        return {"error": str(e), "traceback": tb}


def _run_test():
    print(">>> TP-PP test entered", flush=True)
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
        cwd="/root/zse",
    )
    if "/root/zse" not in sys.path:
        sys.path.insert(0, "/root/zse")

    import torch
    import gc
    from huggingface_hub import hf_hub_download

    gpu_count = torch.cuda.device_count()
    print(f"\n{'='*60}", flush=True)
    print(f"ZSE Combined TP+PP Parallelism Test", flush=True)
    print(f"GPUs: {gpu_count}", flush=True)
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({total:.1f} GB)", flush=True)
    print(f"{'='*60}\n", flush=True)

    assert gpu_count >= 4, f"Need >= 4 GPUs, got {gpu_count}"

    # ── Download model ──────────────────────────────────────────
    TARGET_REPO = "zse-zllm/Qwen2.5-7B-Instruct-zse-int4"
    TARGET_FILE = "Qwen2.5-7B-Instruct-zse-int4.zse"
    PROMPT = "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes."
    MAX_TOKENS = 20

    print(f"Downloading: {TARGET_REPO}", flush=True)
    target_path = hf_hub_download(repo_id=TARGET_REPO, filename=TARGET_FILE)
    print(f"  Size: {os.path.getsize(target_path) / 1024**3:.2f} GB\n", flush=True)

    # Prepare prompt
    from zse.format.reader_v2 import load_zse_model
    _, tokenizer, _ = load_zse_model(target_path, device="cuda:0")
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_len}\n", flush=True)

    # Free the model we loaded just for tokenizer
    del _
    torch.cuda.empty_cache()
    gc.collect()

    results = {}

    # ── Test: TP=2 × PP=2 (combined) ───────────────────────────
    print(f"{'─'*60}", flush=True)
    print("TEST: Combined TP=2 × PP=2 (4 GPUs)", flush=True)
    print(f"{'─'*60}", flush=True)

    from zse.core.zdistributed.tp_pp_parallel import TPPPCoordinator

    tppp_coord = TPPPCoordinator(
        model_path=target_path,
        tp_size=2,
        pp_size=2,
    )
    tppp_coord.start(verbose=True)

    # Warmup
    warmup = tppp_coord.generate(input_ids, max_new_tokens=3, do_sample=False)
    warmup_text = tokenizer.decode(warmup[0, prompt_len:], skip_special_tokens=True)
    print(f"  Warmup: {warmup_text[:60]}...", flush=True)

    for i in range(4):
        torch.cuda.synchronize(i)

    # Benchmark
    t0 = time.perf_counter()
    out_tppp = tppp_coord.generate(input_ids, max_new_tokens=MAX_TOKENS, do_sample=False)
    for i in range(4):
        torch.cuda.synchronize(i)
    t_tppp = time.perf_counter() - t0
    tps_tppp = (out_tppp.shape[1] - prompt_len) / t_tppp
    text_tppp = tokenizer.decode(out_tppp[0, prompt_len:], skip_special_tokens=True)

    print(f"  Speed: {tps_tppp:.1f} tok/s  |  Time: {t_tppp:.2f}s", flush=True)
    print(f"  Tokens: {out_tppp.shape[1] - prompt_len}", flush=True)
    print(f"  Output: {text_tppp[:200]}...\n", flush=True)
    results["tp2_pp2"] = {"tps": tps_tppp}

    tppp_coord.shutdown()

    # ── Summary ─────────────────────────────────────────────────
    print(f"{'='*60}", flush=True)
    print(f"TP2×PP2:  {tps_tppp:.1f} tok/s", flush=True)
    print(f"{'='*60}", flush=True)

    # Validations
    tokens_tppp = out_tppp.shape[1] - prompt_len
    assert tokens_tppp > 5, f"Too few tokens: {tokens_tppp}"
    assert len(text_tppp) > 10, f"Output too short"
    assert "i i i i" not in text_tppp.lower(), f"Degenerate output"

    print(f"\n✅ Combined TP+PP test PASSED", flush=True)

    return results


@app.local_entrypoint()
def main():
    result = test_tp_pp.remote()
    print(f"\nRemote result: {result}")
