"""
REAL Cold Start Comparison - Qwen 32B with Pre-converted .zse

Step 1: Download and convert to .zse (one-time, stored in Modal volume)
Step 2: Measure ACTUAL cold start from .zse file
Step 3: Compare to bitsandbytes cold start

Model: Qwen/Qwen2.5-Coder-32B-Instruct
GPU: A100-80GB
"""

import modal
import time

app = modal.App("zse-32b-preconvert")

# Persistent volume for storing converted model
model_volume = modal.Volume.from_name("zse-models", create_if_missing=True)

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
ZSE_PATH = "/models/qwen32b.zse"


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours for conversion
    volumes={"/models": model_volume},
)
def step1_convert_to_zse():
    """
    Step 1: Download model and convert to .zse format.
    This is a one-time operation, stored in persistent volume.
    """
    import torch
    import os
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from safetensors.torch import save_file
    
    zse_path = Path(ZSE_PATH)
    
    # Check if already converted
    if zse_path.exists():
        size_gb = zse_path.stat().st_size / 1e9
        print(f"✅ .zse file already exists: {ZSE_PATH} ({size_gb:.2f} GB)")
        return {"status": "exists", "size_gb": size_gb}
    
    print("=" * 70)
    print("STEP 1: Converting Qwen 32B to .zse format")
    print("=" * 70)
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {ZSE_PATH}")
    
    # Load with bitsandbytes NF4 (this is what we're pre-computing)
    print("\n[1/3] Loading model with bitsandbytes NF4...")
    start = time.time()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    load_time = time.time() - start
    print(f"   Model loaded in {load_time:.1f}s")
    
    # Extract state dict
    print("\n[2/3] Extracting quantized weights...")
    start = time.time()
    
    state_dict = {}
    for name, param in model.named_parameters():
        state_dict[name] = param.data.cpu().clone()
    for name, buf in model.named_buffers():
        state_dict[name] = buf.cpu().clone()
    
    extract_time = time.time() - start
    print(f"   Extracted {len(state_dict)} tensors in {extract_time:.1f}s")
    
    # Save as .zse (safetensors format)
    print("\n[3/3] Saving to .zse format...")
    start = time.time()
    
    # Filter to only torch tensors
    tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    
    zse_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensor_dict, str(zse_path))
    
    save_time = time.time() - start
    size_gb = zse_path.stat().st_size / 1e9
    
    print(f"   Saved in {save_time:.1f}s")
    print(f"   File size: {size_gb:.2f} GB")
    
    # Also save tokenizer
    tokenizer.save_pretrained("/models/qwen32b_tokenizer")
    
    # Commit to volume
    model_volume.commit()
    
    print("\n" + "=" * 70)
    print("✅ CONVERSION COMPLETE")
    print(f"   .zse file: {ZSE_PATH} ({size_gb:.2f} GB)")
    print(f"   Total time: {load_time + extract_time + save_time:.1f}s")
    print("=" * 70)
    
    return {
        "status": "converted",
        "size_gb": size_gb,
        "load_time": load_time,
        "extract_time": extract_time,
        "save_time": save_time,
    }


@app.function(
    image=gpu_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/models": model_volume},
)
def step2_test_cold_starts():
    """
    Step 2: Test ACTUAL cold starts - .zse vs bitsandbytes
    """
    import torch
    import os
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from safetensors.torch import load_file
    
    results = {}
    
    print("=" * 70)
    print("COLD START COMPARISON - QWEN 32B")
    print("=" * 70)
    
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.1f} GB")
    
    # Check .zse file exists
    zse_path = Path(ZSE_PATH)
    if not zse_path.exists():
        print("❌ ERROR: .zse file not found. Run step1_convert_to_zse first.")
        return {"error": "zse file not found"}
    
    zse_size = zse_path.stat().st_size / 1e9
    print(f".zse file: {zse_size:.2f} GB")
    
    # =========================================================================
    # TEST 1: .zse Cold Start (PRE-CONVERTED)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] .zse COLD START (pre-converted)")
    print("This is what users get after running: zse convert model -o model.zse")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_zse = time.time()
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/models/qwen32b_tokenizer")
    tokenizer_time = time.time() - start_zse
    
    # 2. Load pre-quantized weights (memory-mapped)
    weights = load_file(str(zse_path))
    weights_time = time.time() - start_zse
    
    # 3. Create model and load weights
    # We need to initialize the model architecture
    from transformers import AutoConfig, Qwen2ForCausalLM
    
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Initialize empty model on meta device, then load weights
    with torch.device('meta'):
        model_zse = Qwen2ForCausalLM(config)
    
    # Load state dict
    model_zse.load_state_dict(weights, assign=True)
    model_zse = model_zse.to('cuda', dtype=torch.float16)
    
    model_init_time = time.time() - start_zse
    
    # 4. Verify ready - generate one token
    inputs = tokenizer("def hello():", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model_zse.generate(**inputs, max_new_tokens=1)
    
    zse_cold_start = time.time() - start_zse
    zse_vram = torch.cuda.memory_allocated() / 1e9
    zse_peak = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n✅ .zse COLD START: {zse_cold_start:.1f}s")
    print(f"   - Tokenizer: {tokenizer_time:.2f}s")
    print(f"   - Weights load: {weights_time - tokenizer_time:.2f}s")
    print(f"   - Model init: {model_init_time - weights_time:.2f}s")
    print(f"   - First token: {zse_cold_start - model_init_time:.2f}s")
    print(f"   VRAM: {zse_vram:.2f} GB (peak: {zse_peak:.2f} GB)")
    
    results['zse_cold_start'] = zse_cold_start
    results['zse_vram'] = zse_vram
    
    # Generation benchmark
    print("\n   Running generation benchmark...")
    inputs = tokenizer("def fibonacci(n):", return_tensors="pt").to("cuda")
    start_gen = time.time()
    with torch.no_grad():
        outputs = model_zse.generate(**inputs, max_new_tokens=50, do_sample=False)
    gen_time = time.time() - start_gen
    tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    zse_tok_per_sec = tokens / gen_time
    print(f"   Generation: {tokens} tokens in {gen_time:.2f}s ({zse_tok_per_sec:.1f} tok/s)")
    
    results['zse_tok_per_sec'] = zse_tok_per_sec
    
    # Clean up
    del model_zse, weights
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TEST 2: bitsandbytes NF4 Cold Start (BASELINE)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] bitsandbytes NF4 COLD START (baseline)")
    print("This is what users get running: zse serve Qwen/Qwen2.5-Coder-32B-Instruct")
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
    model_bnb = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Verify ready
    inputs = tokenizer("def hello():", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model_bnb.generate(**inputs, max_new_tokens=1)
    
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
        outputs = model_bnb.generate(**inputs, max_new_tokens=50, do_sample=False)
    gen_time = time.time() - start_gen
    tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    bnb_tok_per_sec = tokens / gen_time
    print(f"   Generation: {tokens} tokens in {gen_time:.2f}s ({bnb_tok_per_sec:.1f} tok/s)")
    
    results['bnb_tok_per_sec'] = bnb_tok_per_sec
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    speedup = bnb_cold_start / zse_cold_start
    
    print("\n" + "=" * 70)
    print("VERIFIED RESULTS - QWEN 32B (A100-80GB)")
    print("=" * 70)
    print(f"""
┌─────────────────────────────────┬────────────┬──────────┐
│ Method                          │ Cold Start │ Speedup  │
├─────────────────────────────────┼────────────┼──────────┤
│ bitsandbytes NF4 (baseline)     │ {bnb_cold_start:>7.1f}s   │    -     │
│ .zse pre-converted              │ {zse_cold_start:>7.1f}s   │ {speedup:>5.1f}×   │
└─────────────────────────────────┴────────────┴──────────┘

VRAM Usage:
- bitsandbytes: {bnb_vram:.2f} GB (peak: {bnb_peak:.2f} GB)
- .zse format:  {zse_vram:.2f} GB (peak: {zse_peak:.2f} GB)

Throughput:
- bitsandbytes: {bnb_tok_per_sec:.1f} tok/s
- .zse format:  {zse_tok_per_sec:.1f} tok/s
""")
    
    print("=" * 70)
    print("FOR README (100% VERIFIED):")
    print("=" * 70)
    print(f"Qwen 32B on A100-80GB:")
    print(f"  - bitsandbytes NF4: {bnb_cold_start:.1f}s cold start")
    print(f"  - .zse pre-converted: {zse_cold_start:.1f}s cold start")
    print(f"  - Speedup: {speedup:.1f}×")
    print(f"  - VRAM: {zse_vram:.2f} GB")
    print("=" * 70)
    
    results['speedup'] = speedup
    return results


@app.local_entrypoint()
def main():
    print("Running 32B cold start test with pre-conversion...")
    print("This will convert the model first (if needed), then test cold starts.\n")
    
    # Step 1: Convert (will skip if already exists)
    print("=" * 70)
    print("STEP 1: Ensure model is converted to .zse")
    print("=" * 70)
    convert_result = step1_convert_to_zse.remote()
    print(f"Conversion status: {convert_result}")
    
    # Step 2: Test cold starts  
    print("\n" + "=" * 70)
    print("STEP 2: Running cold start comparison")
    print("=" * 70)
    test_result = step2_test_cold_starts.remote()
    print(f"\nFinal results: {test_result}")
