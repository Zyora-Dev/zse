"""
Qwen 32B Cold Start Test with Cached Model

Uses Modal Volume to persist model (one-time download).
Then measures actual cold start from cached model.
"""

import modal
import time
import os

app = modal.App("zse-32b-cached")

# Persistent volume for model cache
model_volume = modal.Volume.from_name("qwen-32b-cache", create_if_missing=True)

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
    .env({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
CACHE_DIR = "/cache/models"


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours
    volumes={CACHE_DIR: model_volume},
)
def download_model():
    """Download model to persistent volume (run once)."""
    from huggingface_hub import snapshot_download
    import os
    
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    
    print("=" * 70)
    print("DOWNLOADING QWEN 32B TO PERSISTENT CACHE")
    print("=" * 70)
    
    start = time.time()
    path = snapshot_download(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    download_time = time.time() - start
    
    print(f"\n✅ Download complete: {download_time:.1f}s")
    print(f"   Path: {path}")
    
    # List files
    import glob
    files = glob.glob(f"{path}/*")
    total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
    print(f"   Files: {len(files)}")
    print(f"   Total size: {total_size / 1e9:.2f} GB")
    
    # Commit the volume
    model_volume.commit()
    
    return {"download_time": download_time, "path": path, "size_gb": total_size / 1e9}


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={CACHE_DIR: model_volume},
)
def test_cold_start_cached():
    """
    Test cold start with model already cached.
    This is what users experience after first download.
    """
    import torch
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from safetensors.torch import load_file
    import glob
    
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode - use cache only
    
    results = {}
    
    print("=" * 70)
    print("COLD START TEST - QWEN 32B (FROM CACHE)")
    print("=" * 70)
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.1f} GB")
    print(f"Model: {MODEL_NAME}")
    print(f"Cache: {CACHE_DIR}")
    print("=" * 70)
    
    # Verify model is cached
    import glob
    cached_files = glob.glob(f"{CACHE_DIR}/**/*.safetensors", recursive=True)
    print(f"\nCached safetensor files: {len(cached_files)}")
    if not cached_files:
        raise RuntimeError("Model not cached! Run download_model first.")
    
    # =========================================================================
    # TEST 1: bitsandbytes NF4 - Cold Start (from cache)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] bitsandbytes NF4 - Cold Start (cached model)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # COLD START from cache
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        cache_dir=CACHE_DIR,
        local_files_only=True,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
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
    
    print(f"\n✅ bitsandbytes NF4 COLD START (cached): {bnb_cold_start:.1f}s")
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
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TEST 2: Raw safetensors load (simulates .zse)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] Raw safetensors load (what .zse does)")
    print("=" * 70)
    
    # Find safetensor files
    safetensor_files = glob.glob(f"{CACHE_DIR}/**/*.safetensors", recursive=True)
    print(f"   Found {len(safetensor_files)} safetensor files")
    
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
    # TEST 3: Estimate .zse full cold start
    # =========================================================================
    # .zse cold start = raw load + model init + GPU transfer
    # From 7B test: raw=0.02s, init+GPU=3.9s total
    # For 32B: raw is measured, init scales ~4x
    
    model_init_overhead = 15  # ~15s for 32B architecture init + GPU transfer
    zse_estimated = raw_load_time + model_init_overhead
    speedup = bnb_cold_start / zse_estimated
    
    results['zse_estimated'] = zse_estimated
    results['speedup'] = speedup
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - QWEN 32B (A100-80GB, CACHED)")
    print("=" * 70)
    print(f"""
┌─────────────────────────────────┬────────────┬──────────┐
│ Method                          │ Cold Start │ Speedup  │
├─────────────────────────────────┼────────────┼──────────┤
│ bitsandbytes NF4 (cached)       │ {bnb_cold_start:>7.1f}s   │    -     │
│ Raw safetensors load            │ {raw_load_time:>7.2f}s   │    -     │
│ .zse format (estimated)         │ {zse_estimated:>7.1f}s   │ {speedup:>5.1f}×   │
└─────────────────────────────────┴────────────┴──────────┘

VERIFIED MEASUREMENTS:
- bitsandbytes NF4 (from cache): {bnb_cold_start:.1f}s
- VRAM: {bnb_vram:.2f} GB (peak: {bnb_peak:.2f} GB)
- Throughput: {tok_per_sec:.1f} tok/s
- Raw safetensor load: {raw_load_time:.2f}s

ESTIMATED .zse:
- Raw load ({raw_load_time:.2f}s) + init overhead (~15s) = {zse_estimated:.1f}s
- Speedup vs bitsandbytes: {speedup:.1f}×
""")
    
    print("=" * 70)
    print("HONEST NUMBERS FOR README:")
    print("=" * 70)
    print(f"Qwen 32B on A100-80GB (model cached):")
    print(f"  - bitsandbytes NF4: {bnb_cold_start:.1f}s ✅ MEASURED")
    print(f"  - VRAM: {bnb_vram:.2f} GB ✅")
    print(f"  - .zse estimated: {zse_estimated:.1f}s ({speedup:.1f}× faster)")
    print("=" * 70)
    
    return results


@app.local_entrypoint()
def main():
    print("Step 1: Downloading model to persistent cache...")
    result = download_model.remote()
    print(f"Download complete: {result}")
    
    print("\nStep 2: Testing cold start from cache...")
    result = test_cold_start_cached.remote()
    print(f"\nFinal results: {result}")
