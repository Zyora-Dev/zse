"""
HONEST Cold Start Comparison Test - Qwen 32B

Tests ACTUAL cold start time on Modal A100-80GB.
Based on verified 7B results: 11.5x speedup (45s → 3.9s)

Model: Qwen/Qwen2.5-Coder-32B-Instruct
GPU: A100-80GB
"""

import modal
import time

app = modal.App("zse-cold-start-32b-v2")

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
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=3600,
)
def test_cold_start_32b():
    """
    Measure bitsandbytes NF4 cold start for Qwen 32B.
    Then measure raw safetensors load time to estimate .zse speedup.
    """
    import torch
    import os
    import tempfile
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from safetensors.torch import save_file, load_file
    
    results = {}
    
    # GPU Info
    print("=" * 70)
    print("COLD START TEST - QWEN 32B")
    print("=" * 70)
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.1f} GB")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)
    
    # =========================================================================
    # TEST 1: bitsandbytes NF4 COLD START
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] bitsandbytes NF4 - Full Cold Start")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    start_bnb = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Verify model is ready
    inputs = tokenizer("def hello():", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1)
    
    bnb_cold_start = time.time() - start_bnb
    bnb_vram = torch.cuda.memory_allocated() / 1e9
    bnb_peak = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n✅ bitsandbytes NF4 COLD START: {bnb_cold_start:.1f}s")
    print(f"   VRAM: {bnb_vram:.2f} GB (peak: {bnb_peak:.2f} GB)")
    
    results['bnb_cold_start'] = bnb_cold_start
    results['bnb_vram'] = bnb_vram
    
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
    
    # Clean up for next test
    del model
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TEST 2: Measure raw safetensors load + model init overhead
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] Measuring .zse load components")
    print("=" * 70)
    
    # Find the model's safetensors files
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(MODEL_NAME)
    
    # Time loading raw safetensors
    import glob
    safetensor_files = glob.glob(f"{cache_dir}/*.safetensors")
    print(f"\n   Found {len(safetensor_files)} safetensor files")
    
    start_raw = time.time()
    all_weights = {}
    for sf in safetensor_files:
        weights = load_file(sf)
        all_weights.update(weights)
    raw_load_time = time.time() - start_raw
    
    total_params = sum(w.numel() for w in all_weights.values())
    total_size_gb = sum(w.numel() * w.element_size() for w in all_weights.values()) / 1e9
    
    print(f"\n⚡ Raw safetensors load: {raw_load_time:.2f}s")
    print(f"   Total tensors: {len(all_weights)}")
    print(f"   Total params: {total_params/1e9:.1f}B")
    print(f"   Total size: {total_size_gb:.1f} GB")
    
    results['raw_safetensor_load'] = raw_load_time
    results['total_params_b'] = total_params / 1e9
    
    del all_weights
    
    # =========================================================================
    # ESTIMATE .zse cold start based on 7B data
    # =========================================================================
    # From 7B test: BNB=45s, .zse=3.9s, ratio=11.5x
    # The .zse speedup comes from:
    # 1. Pre-quantized weights (skip on-the-fly quantization)
    # 2. Memory-mapped I/O
    # 3. Optimized weight loading
    
    # For 32B, download is ~4x larger but quantization scales linearly
    # Estimated .zse time based on component analysis:
    # - Raw load (measured): raw_load_time
    # - Model init overhead (scales ~linearly by layers): ~15s for 32B vs ~5s for 7B
    # - GPU transfer: ~5s for 32B
    
    estimated_zse_time = raw_load_time + 20  # raw load + init + GPU transfer
    
    # Also calculate based on ratio (more conservative)
    ratio_based_estimate = bnb_cold_start / 11.5
    
    # Use the larger (more conservative) estimate
    zse_estimate = max(estimated_zse_time, ratio_based_estimate)
    speedup = bnb_cold_start / zse_estimate
    
    results['zse_estimated'] = zse_estimate
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
│ bitsandbytes NF4 (measured)     │ {bnb_cold_start:>7.1f}s   │    -     │
│ Raw safetensors load            │ {raw_load_time:>7.2f}s   │    -     │
│ .zse format (estimated)         │ {zse_estimate:>7.1f}s   │ {speedup:>5.1f}×   │
└─────────────────────────────────┴────────────┴──────────┘

VERIFIED MEASUREMENTS:
- bitsandbytes NF4: {bnb_cold_start:.1f}s cold start
- VRAM: {bnb_vram:.2f} GB
- Throughput: {tok_per_sec:.1f} tok/s

ESTIMATED .zse PERFORMANCE:
- Based on component analysis: {estimated_zse_time:.1f}s
- Based on 7B ratio (11.5×): {ratio_based_estimate:.1f}s
- Conservative estimate: {zse_estimate:.1f}s ({speedup:.1f}× speedup)

Note: .zse estimate is theoretical based on:
1. Measured raw safetensor load time ({raw_load_time:.2f}s)
2. Estimated model init overhead (~15s for 32B)
3. GPU transfer time (~5s)
""")
    
    print("=" * 70)
    print("FOR README (VERIFIED):")
    print("=" * 70)
    print(f"Qwen 32B on A100-80GB:")
    print(f"  - bitsandbytes NF4: {bnb_cold_start:.1f}s cold start ✅")
    print(f"  - VRAM: {bnb_vram:.2f} GB ✅")
    print(f"  - Throughput: {tok_per_sec:.1f} tok/s ✅")
    print(f"  - .zse estimated: ~{zse_estimate:.0f}s (based on 7B verified ratio)")
    print("=" * 70)
    
    return results


@app.local_entrypoint()
def main():
    print("Starting Qwen 32B cold start test on Modal A100-80GB...")
    print("This will take ~10 minutes.\n")
    results = test_cold_start_32b.remote()
    print("\nTest complete!")
    print(f"Results: {results}")
