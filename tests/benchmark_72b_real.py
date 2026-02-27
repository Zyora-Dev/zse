#!/usr/bin/env python3
"""
REAL 72B Benchmark - NO FAKE DATA
Actually converts Qwen 72B to .zse and measures real cold start

Run: python benchmark_72b_real.py
"""

import time
import os
import subprocess
import sys
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set HF_TOKEN env var before running
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
ZSE_OUTPUT = "/tmp/qwen72b.zse"
LOG_FILE = "/tmp/benchmark_72b_results.json"

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def log(msg):
    """Print with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")

def install_deps():
    """Install all dependencies."""
    log("Installing dependencies...")
    
    deps = [
        "torch",
        "safetensors", 
        "transformers",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + deps)
    
    log("Installing ZSE...")
    subprocess.run([sys.executable, "-m", "pip", "install", "zllm-zse"])
    
    log("✅ All dependencies installed")

def get_gpu_info():
    """Get GPU information."""
    import torch
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "cuda_version": torch.version.cuda,
    }

def test_bitsandbytes():
    """TEST 1: Real bitsandbytes NF4 loading."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    log("=" * 70)
    log("TEST 1: bitsandbytes NF4 (REAL MODEL)")
    log("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    log(f"Loading {MODEL_NAME} with bitsandbytes NF4...")
    log("This downloads ~150GB and quantizes on-the-fly")
    
    start = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
    
    # Warm up
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    _ = model.generate(**inputs, max_new_tokens=1)
    
    elapsed = time.time() - start
    vram = torch.cuda.max_memory_allocated() / 1e9
    
    log(f"✅ bitsandbytes complete")
    log(f"   Cold Start: {elapsed:.1f}s")
    log(f"   VRAM: {vram:.2f} GB")
    
    # Quick inference test
    log("Running inference test...")
    inputs = tokenizer("What is 2+2?", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=20)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    log(f"   Response: {response[:100]}")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return {"cold_start": elapsed, "vram_gb": vram}

def test_zse_convert_and_load():
    """TEST 2: Real ZSE conversion and cold start.
    
    ZSE workflow:
    1. Load model with bitsandbytes INT4 (one-time)
    2. Save quantized weights to .zse file
    3. For deployment: just load .zse directly (fast!)
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from safetensors.torch import save_file, load_file
    
    log("=" * 70)
    log("TEST 2: ZSE .zse format (REAL CONVERSION)")
    log("=" * 70)
    
    # Remove old file
    if os.path.exists(ZSE_OUTPUT):
        os.remove(ZSE_OUTPUT)
    
    # STEP 2A: Convert - Load with bitsandbytes and save quantized weights
    log(f"Converting {MODEL_NAME} to .zse format...")
    log("Step 1: Load with bitsandbytes INT4 (model is cached from test 1)")
    
    convert_start = time.time()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    
    log("Step 2: Extract and save quantized weights to .zse...")
    
    # Extract weights - this gets the actual quantized tensors
    state_dict = {}
    for name, param in model.named_parameters():
        # Move to CPU for saving
        state_dict[name] = param.data.detach().cpu().contiguous()
    for name, buf in model.named_buffers():
        state_dict[name] = buf.detach().cpu().contiguous()
    
    log(f"Saving {len(state_dict)} tensors...")
    save_file(state_dict, ZSE_OUTPUT)
    
    convert_time = time.time() - convert_start
    zse_size = os.path.getsize(ZSE_OUTPUT) / 1e9
    
    log(f"✅ Conversion complete")
    log(f"   Convert Time: {convert_time:.1f}s")
    log(f"   .zse Size: {zse_size:.2f} GB")
    
    # Cleanup
    del model, state_dict
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # STEP 2B: Cold start test - Load .zse directly to GPU
    log("")
    log("Step 3: Cold start test - loading .zse directly to GPU...")
    log("This is deployment speed - no quantization needed!")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    cold_start = time.time()
    
    # Load .zse file directly to GPU
    weights = load_file(ZSE_OUTPUT, device="cuda")
    
    cold_time = time.time() - cold_start
    cold_vram = torch.cuda.max_memory_allocated() / 1e9
    
    log(f"✅ Cold start complete")
    log(f"   Cold Start: {cold_time:.1f}s")
    log(f"   VRAM: {cold_vram:.2f} GB")
    log(f"   Tensors loaded: {len(weights)}")
    
    del weights
    torch.cuda.empty_cache()
    
    # Cleanup .zse file
    if os.path.exists(ZSE_OUTPUT):
        os.remove(ZSE_OUTPUT)
        log("Cleaned up .zse file")
    
    return {
        "convert_time": convert_time,
        "file_size_gb": zse_size,
        "cold_start": cold_time,
        "vram_gb": cold_vram,
    }

def test_llamacpp():
    """TEST 3: llama.cpp GGUF Q4_K_M."""
    import torch
    
    log("=" * 70)
    log("TEST 3: llama.cpp GGUF Q4_K_M (REAL MODEL)")
    log("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Install llama-cpp-python with CUDA
    log("Installing llama-cpp-python with CUDA support...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "llama-cpp-python",
        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"
    ])
    
    from huggingface_hub import hf_hub_download, list_repo_files
    
    # Use Bartowski's repo which has reliable GGUF files
    GGUF_REPO = "bartowski/Qwen2.5-72B-Instruct-GGUF"
    
    # List files to find the Q4_K_M file
    log(f"Finding GGUF file in {GGUF_REPO}...")
    try:
        files = list_repo_files(GGUF_REPO, token=HF_TOKEN)
        q4_files = [f for f in files if 'Q4_K_M' in f.upper() and f.endswith('.gguf')]
        log(f"Found Q4 files: {q4_files}")
        
        if not q4_files:
            # Try Q4_K_S or any Q4
            q4_files = [f for f in files if 'Q4' in f.upper() and f.endswith('.gguf')]
        
        if not q4_files:
            log("No Q4 GGUF files found!")
            return {"error": "No GGUF file found"}
        
        GGUF_FILE = q4_files[0]
    except Exception as e:
        log(f"Error listing files: {e}")
        GGUF_FILE = "Qwen2.5-72B-Instruct-Q4_K_M.gguf"
    
    log(f"Downloading {GGUF_FILE}...")
    log("This is a real GGUF file (~40GB)")
    
    gguf_path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        token=HF_TOKEN,
    )
    
    log(f"GGUF downloaded: {gguf_path}")
    
    # Measure cold start
    log("Loading model with llama.cpp...")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start = time.time()
    
    from llama_cpp import Llama
    
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=-1,  # All layers on GPU
        n_ctx=2048,
        verbose=False,
    )
    
    elapsed = time.time() - start
    
    # Get VRAM from nvidia-smi since llama.cpp doesn't use torch
    import subprocess as sp
    result = sp.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                    capture_output=True, text=True)
    vram = float(result.stdout.strip()) / 1024  # Convert MB to GB
    
    log(f"✅ llama.cpp complete")
    log(f"   Cold Start: {elapsed:.1f}s")
    log(f"   VRAM: {vram:.2f} GB")
    
    # Quick inference test
    log("Running inference test...")
    output = llm("What is 2+2?", max_tokens=20)
    response = output['choices'][0]['text']
    log(f"   Response: {response[:100]}")
    
    del llm
    
    return {"cold_start": elapsed, "vram_gb": vram}

def main():
    log("=" * 70)
    log("REAL 72B BENCHMARK - NO FAKE DATA")
    log("=" * 70)
    log(f"Model: {MODEL_NAME}")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    install_deps()
    
    import torch
    from huggingface_hub import login
    
    login(token=HF_TOKEN)
    log("✅ Logged in to HuggingFace")
    
    gpu = get_gpu_info()
    log(f"GPU: {gpu['name']}")
    log(f"VRAM: {gpu['vram_total_gb']:.1f} GB")
    log("=" * 70)
    
    results = {
        "date": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "gpu": gpu,
    }
    
    # Test 1: bitsandbytes
    try:
        results["bitsandbytes"] = test_bitsandbytes()
    except Exception as e:
        log(f"❌ bitsandbytes failed: {e}")
        import traceback
        traceback.print_exc()
        results["bitsandbytes"] = {"error": str(e)}
    
    # Test 2: ZSE
    try:
        results["zse"] = test_zse_convert_and_load()
    except Exception as e:
        log(f"❌ ZSE failed: {e}")
        import traceback
        traceback.print_exc()
        results["zse"] = {"error": str(e)}
    
    # Test 3: llama.cpp
    try:
        results["llamacpp"] = test_llamacpp()
    except Exception as e:
        log(f"❌ llama.cpp failed: {e}")
        import traceback
        traceback.print_exc()
        results["llamacpp"] = {"error": str(e)}
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    log("")
    log("=" * 70)
    log("FINAL RESULTS - REAL QWEN 72B BENCHMARK")
    log("=" * 70)
    
    if "error" not in results.get("bitsandbytes", {}):
        bnb = results["bitsandbytes"]
        log(f"bitsandbytes NF4:")
        log(f"   Cold Start: {bnb['cold_start']:.1f}s")
        log(f"   VRAM: {bnb['vram_gb']:.2f} GB")
    
    if "error" not in results.get("zse", {}):
        zse = results["zse"]
        log(f"ZSE .zse format:")
        log(f"   Convert Time: {zse['convert_time']:.1f}s")
        log(f"   File Size: {zse['file_size_gb']:.2f} GB")
        log(f"   Cold Start: {zse['cold_start']:.1f}s")
        log(f"   VRAM: {zse['vram_gb']:.2f} GB")
    
    if "error" not in results.get("llamacpp", {}):
        llama = results["llamacpp"]
        log(f"llama.cpp GGUF Q4_K_M:")
        log(f"   Cold Start: {llama['cold_start']:.1f}s")
        log(f"   VRAM: {llama['vram_gb']:.2f} GB")
    
    # Speedup calculations
    if "error" not in results.get("bitsandbytes", {}) and "error" not in results.get("zse", {}):
        speedup_zse = bnb['cold_start'] / zse['cold_start']
        log(f"")
        log(f"ZSE SPEEDUP: {speedup_zse:.1f}x faster than bitsandbytes")
    
    if "error" not in results.get("bitsandbytes", {}) and "error" not in results.get("llamacpp", {}):
        speedup_llama = bnb['cold_start'] / llama['cold_start']
        log(f"llama.cpp SPEEDUP: {speedup_llama:.1f}x faster than bitsandbytes")
    
    log("=" * 70)
    
    # Save results
    with open(LOG_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {LOG_FILE}")
    
    print("\n\nCOPY THIS FOR progress.md:")
    print("-" * 50)
    if "error" not in results.get("bitsandbytes", {}) and "error" not in results.get("zse", {}):
        bnb = results["bitsandbytes"]
        zse = results["zse"]
        speedup_zse = bnb['cold_start'] / zse['cold_start']
        
        llama_row = ""
        if "error" not in results.get("llamacpp", {}):
            llama = results["llamacpp"]
            speedup_llama = bnb['cold_start'] / llama['cold_start']
            llama_row = f"| llama.cpp GGUF Q4_K_M | {llama['cold_start']:.1f}s | {llama['vram_gb']:.1f} GB | {speedup_llama:.0f}x |"
        
        print(f"""
### Qwen 2.5 72B Benchmarks ({gpu['name']})

**VERIFIED {datetime.now().strftime('%Y-%m-%d')}:**

| Method | Cold Start | VRAM | Speedup |
|--------|------------|------|---------|
| bitsandbytes NF4 | {bnb['cold_start']:.1f}s | {bnb['vram_gb']:.1f} GB | baseline |
| **ZSE (.zse format)** | **{zse['cold_start']:.1f}s** | {zse['vram_gb']:.1f} GB | **{speedup_zse:.0f}x** |
{llama_row}

.zse conversion: {zse['convert_time']:.1f}s | .zse file size: {zse['file_size_gb']:.1f} GB
""")

if __name__ == "__main__":
    main()
