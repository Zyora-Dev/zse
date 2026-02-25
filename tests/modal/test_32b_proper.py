"""
Qwen 32B Cold Start Test - PROPER approach

Downloads model during IMAGE BUILD (not runtime).
Model is baked into the container image.
Cold start test runs instantly from cached image.
"""

import modal
import time

app = modal.App("zse-32b-proper")

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Download model DURING IMAGE BUILD
def download_model():
    """This runs at image build time, not runtime."""
    import os
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    from huggingface_hub import snapshot_download
    snapshot_download(
        MODEL_NAME,
        token=HF_TOKEN,
    )
    print(f"✅ Model {MODEL_NAME} downloaded and cached in image")

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
    .run_function(download_model)  # Download at build time!
)


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=1800,
)
def test_cold_start():
    """
    Test cold start with model already in image.
    No download needed - instant start.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from safetensors.torch import load_file
    import glob
    import os
    
    results = {}
    
    print("=" * 70)
    print("COLD START TEST - QWEN 32B (MODEL PRE-CACHED IN IMAGE)")
    print("=" * 70)
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.1f} GB")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)
    
    # =========================================================================
    # TEST 1: bitsandbytes NF4 - Cold Start (from cached image)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] bitsandbytes NF4 - Cold Start (model in image)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # COLD START - model already cached
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer_time = time.time() - start
    print(f"   Tokenizer loaded: {tokenizer_time:.2f}s")
    
    model_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model_load_time = time.time() - model_start
    print(f"   Model loaded + quantized: {model_load_time:.2f}s")
    
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
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TEST 2: Raw safetensors load (simulates .zse)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] Raw safetensors load (what .zse does)")
    print("=" * 70)
    
    # Find HF cache
    from huggingface_hub import snapshot_download
    cache_path = snapshot_download(MODEL_NAME, local_files_only=True)
    
    safetensor_files = glob.glob(f"{cache_path}/*.safetensors")
    print(f"   Found {len(safetensor_files)} safetensor files in {cache_path}")
    
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
    # ESTIMATE .zse cold start
    # =========================================================================
    # .zse = raw load + model init + GPU transfer
    # For 32B: raw is measured, model init ~15s overhead
    
    model_init_overhead = 15
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
│ bitsandbytes NF4 (cached)       │ {bnb_cold_start:>7.1f}s   │    -     │
│ Raw safetensors load            │ {raw_load_time:>7.2f}s   │    -     │
│ .zse format (estimated)         │ {zse_estimated:>7.1f}s   │ {speedup:>5.1f}×   │
└─────────────────────────────────┴────────────┴──────────┘

VERIFIED MEASUREMENTS:
- bitsandbytes NF4: {bnb_cold_start:.1f}s cold start ✅
- VRAM: {bnb_vram:.2f} GB (peak: {bnb_peak:.2f} GB) ✅
- Throughput: {tok_per_sec:.1f} tok/s ✅

ESTIMATED .zse:
- Raw load ({raw_load_time:.2f}s) + init (~15s) = {zse_estimated:.1f}s
- Speedup: {speedup:.1f}×
""")
    
    print("=" * 70)
    print("FOR README (VERIFIED):")
    print("=" * 70)
    print(f"Qwen 32B on A100-80GB:")
    print(f"  - bitsandbytes NF4: {bnb_cold_start:.1f}s ✅")
    print(f"  - VRAM: {bnb_vram:.2f} GB ✅")
    print(f"  - Throughput: {tok_per_sec:.1f} tok/s ✅")
    print(f"  - .zse estimated: ~{zse_estimated:.0f}s ({speedup:.1f}× faster)")
    print("=" * 70)
    
    return results


@app.local_entrypoint()
def main():
    print("Running Qwen 32B cold start test...")
    print("(Model is pre-cached in image - no download at runtime)\n")
    result = test_cold_start.remote()
    print(f"\nResults: {result}")
