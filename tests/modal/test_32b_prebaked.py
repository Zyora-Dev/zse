"""
Qwen 32B Cold Start Test - Model Pre-baked in Image

Downloads model DURING IMAGE BUILD (not runtime).
This avoids heartbeat timeout issues.
"""

import modal

app = modal.App("zse-32b-prebaked")

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
MODEL_DIR = "/models/qwen32b"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Download model DURING IMAGE BUILD
def download_model():
    from huggingface_hub import snapshot_download
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_DIR,
        token=HF_TOKEN,
    )

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
    )
    .env({"HF_TOKEN": HF_TOKEN})
    .run_function(download_model)  # <-- Download happens HERE during build
)


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=1800,
)
def test_cold_start():
    """
    Cold start test with model PRE-DOWNLOADED in image.
    No download = no heartbeat timeout.
    """
    import torch
    import time
    import glob
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from safetensors.torch import load_file
    
    results = {}
    
    print("=" * 70)
    print("COLD START TEST - QWEN 32B (MODEL PRE-BAKED)")
    print("=" * 70)
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.1f} GB")
    print(f"Model: {MODEL_NAME}")
    print(f"Model dir: {MODEL_DIR}")
    print("=" * 70)
    
    # Verify model exists
    safetensor_files = glob.glob(f"{MODEL_DIR}/*.safetensors")
    print(f"\nModel files found: {len(safetensor_files)} safetensors")
    if not safetensor_files:
        raise RuntimeError(f"Model not found in {MODEL_DIR}!")
    
    # =========================================================================
    # TEST 1: bitsandbytes NF4 Cold Start (from local files)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] bitsandbytes NF4 - Cold Start (local files)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # COLD START (model already on disk)
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    
    # Verify ready - generate one token
    inputs = tokenizer("def hello():", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)
    
    bnb_cold_start = time.time() - start
    bnb_vram = torch.cuda.memory_allocated() / 1e9
    bnb_peak = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n✅ bitsandbytes NF4 COLD START: {bnb_cold_start:.1f}s")
    print(f"   VRAM: {bnb_vram:.2f} GB (peak: {bnb_peak:.2f} GB)")
    
    results['bnb_cold_start'] = bnb_cold_start
    results['bnb_vram'] = bnb_vram
    results['bnb_peak_vram'] = bnb_peak
    
    # Generation benchmark
    print("\n   Running generation benchmark...")
    inputs = tokenizer("def fibonacci(n):", return_tensors="pt").to("cuda")
    start_gen = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    gen_time = time.time() - start_gen
    tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tok_per_sec = tokens / gen_time
    print(f"   Generation: {tokens} tokens in {gen_time:.2f}s ({tok_per_sec:.1f} tok/s)")
    
    results['bnb_tok_per_sec'] = tok_per_sec
    
    # Clean up GPU
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TEST 2: Raw safetensors load (simulates .zse)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] Raw safetensors load (what .zse does)")
    print("=" * 70)
    
    start_raw = time.time()
    all_weights = {}
    for sf in safetensor_files:
        weights = load_file(sf)
        all_weights.update(weights)
    raw_load_time = time.time() - start_raw
    
    total_params = sum(w.numel() for w in all_weights.values())
    print(f"\n⚡ Raw safetensors load: {raw_load_time:.2f}s")
    print(f"   Total tensors: {len(all_weights)}")
    print(f"   Total params: {total_params/1e9:.1f}B")
    
    results['raw_load_time'] = raw_load_time
    
    del all_weights
    
    # =========================================================================
    # Estimate .zse cold start
    # =========================================================================
    model_init_overhead = 15  # ~15s for 32B architecture init + GPU transfer
    zse_estimated = raw_load_time + model_init_overhead
    speedup = bnb_cold_start / zse_estimated
    
    results['zse_estimated'] = zse_estimated
    results['speedup'] = speedup
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - QWEN 32B (A100-80GB)")
    print("=" * 70)
    print(f"""
┌─────────────────────────────────┬────────────┬──────────┐
│ Method                          │ Cold Start │ Speedup  │
├─────────────────────────────────┼────────────┼──────────┤
│ bitsandbytes NF4 (local)        │ {bnb_cold_start:>7.1f}s   │    -     │
│ Raw safetensors load            │ {raw_load_time:>7.2f}s   │    -     │
│ .zse format (estimated)         │ {zse_estimated:>7.1f}s   │ {speedup:>5.1f}×   │
└─────────────────────────────────┴────────────┴──────────┘

VERIFIED MEASUREMENTS (Qwen 32B, A100-80GB):
- bitsandbytes NF4: {bnb_cold_start:.1f}s cold start ✅
- VRAM: {bnb_vram:.2f} GB (peak: {bnb_peak:.2f} GB) ✅  
- Throughput: {tok_per_sec:.1f} tok/s ✅
- Raw safetensor load: {raw_load_time:.2f}s ✅

ESTIMATED .zse:
- {zse_estimated:.1f}s ({speedup:.1f}× faster than bitsandbytes)
""")
    
    return results


@app.local_entrypoint()
def main():
    print("Testing Qwen 32B cold start (model pre-baked in image)...")
    print("First run will build image (~15 min to download 65GB)...")
    print("Subsequent runs use cached image.\n")
    result = test_cold_start.remote()
    print(f"\nFinal results: {result}")
