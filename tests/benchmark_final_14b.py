"""
ZSE Comprehensive Benchmark - 14B Model
Tests: bitsandbytes, .zse format, llama.cpp

Metrics: Load Time, VRAM, Cold Start, Token Speed
"""
import time
import torch
import gc
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Config
MODEL_HF = "Qwen/Qwen2.5-14B-Instruct"
ZSE_DIR = Path("/home/ionet/qwen14b")
GGUF_PATH = Path("/home/ionet/Qwen2.5-14B-Instruct-Q4_K_M.gguf")
RESULTS_FILE = Path(f"/home/ionet/benchmark_14b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

def get_gpu_memory():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**3

def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

# ============================================================================
# TEST 1: ZSE bitsandbytes (on-the-fly quantization)
# ============================================================================
def test_bitsandbytes():
    print_header("TEST 1: ZSE bitsandbytes (on-the-fly INT4)")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    clear_gpu()
    baseline = get_gpu_memory()
    
    # Cold start - measure full load time
    print("Loading model with bitsandbytes INT4...")
    start_load = time.perf_counter()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    load_time = time.perf_counter() - start_load
    vram = get_gpu_memory() - baseline
    
    print(f"  Load time: {load_time:.2f}s")
    print(f"  VRAM: {vram:.2f} GB")
    
    # Warmup
    print("Warmup...")
    prompt = "Explain quantum computing:"
    for _ in range(2):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=32, do_sample=False)
    
    # Benchmark
    print("Benchmarking (5 iterations, 128 tokens)...")
    times = []
    for i in range(5):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False, 
                                 pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = out.shape[1] - inputs['input_ids'].shape[1]
        speed = tokens / elapsed
        times.append(speed)
        print(f"    Iter {i+1}: {speed:.1f} tok/s")
    
    avg_speed = sum(times) / len(times)
    
    # Cleanup
    del model, tokenizer
    clear_gpu()
    
    return {
        "method": "bitsandbytes",
        "load_time_sec": round(load_time, 2),
        "vram_gb": round(vram, 2),
        "tokens_per_sec": round(avg_speed, 1),
    }

# ============================================================================
# TEST 2: ZSE .zse format (pre-converted)
# ============================================================================
def test_zse_format():
    print_header("TEST 2: ZSE .zse Format (pre-converted INT4)")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    clear_gpu()
    baseline = get_gpu_memory()
    
    model_dir = ZSE_DIR / "model"
    tokenizer_dir = ZSE_DIR / "tokenizer"
    
    if not model_dir.exists():
        print("  ERROR: .zse format not found. Run: zse convert Qwen/Qwen2.5-14B-Instruct -o qwen14b")
        return None
    
    # Cold start from pre-converted
    print(f"Loading from {ZSE_DIR}...")
    start_load = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    load_time = time.perf_counter() - start_load
    vram = get_gpu_memory() - baseline
    
    print(f"  Load time: {load_time:.2f}s")
    print(f"  VRAM: {vram:.2f} GB")
    
    # Warmup
    print("Warmup...")
    prompt = "Explain quantum computing:"
    for _ in range(2):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=32, do_sample=False)
    
    # Benchmark
    print("Benchmarking (5 iterations, 128 tokens)...")
    times = []
    for i in range(5):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tokens = out.shape[1] - inputs['input_ids'].shape[1]
        speed = tokens / elapsed
        times.append(speed)
        print(f"    Iter {i+1}: {speed:.1f} tok/s")
    
    avg_speed = sum(times) / len(times)
    
    # Cleanup
    del model, tokenizer
    clear_gpu()
    
    return {
        "method": "zse_format",
        "load_time_sec": round(load_time, 2),
        "vram_gb": round(vram, 2),
        "tokens_per_sec": round(avg_speed, 1),
    }

# ============================================================================
# TEST 3: llama.cpp (GGUF)
# ============================================================================
def test_llamacpp():
    print_header("TEST 3: llama.cpp (GGUF Q4_K_M)")
    
    if not GGUF_PATH.exists():
        print(f"  ERROR: GGUF not found at {GGUF_PATH}")
        print("  Download: huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GGUF qwen2.5-14b-instruct-q4_k_m.gguf --local-dir .")
        return None
    
    # Check llama-bench
    llama_bench = subprocess.run(["which", "llama-bench"], capture_output=True, text=True)
    if llama_bench.returncode != 0:
        print("  ERROR: llama-bench not found. Install llama.cpp")
        return None
    
    clear_gpu()
    
    # Run llama-bench
    print(f"Running llama-bench on {GGUF_PATH}...")
    start_load = time.perf_counter()
    
    result = subprocess.run(
        ["llama-bench", "-m", str(GGUF_PATH), "-p", "128", "-n", "128", "-r", "5", "-ngl", "99"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    
    total_time = time.perf_counter() - start_load
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:500])
    
    # Parse output
    lines = result.stdout.strip().split('\n')
    tg_speed = None
    for line in lines:
        if 'tg' in line.lower() and 't/s' in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                if 't/s' in p.lower() and i > 0:
                    try:
                        tg_speed = float(parts[i-1].replace(',', ''))
                    except:
                        pass
    
    # Get VRAM from nvidia-smi during run (approximate)
    vram_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    vram_mb = int(vram_result.stdout.strip().split('\n')[0])
    vram_gb = vram_mb / 1024
    
    return {
        "method": "llama_cpp",
        "load_time_sec": round(total_time, 2),
        "vram_gb": round(vram_gb, 2),
        "tokens_per_sec": round(tg_speed, 1) if tg_speed else None,
    }

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  ZSE COMPREHENSIVE BENCHMARK - Qwen2.5-14B-Instruct")
    print("  Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    results = {
        "model": "Qwen2.5-14B-Instruct",
        "date": datetime.now().isoformat(),
        "tests": []
    }
    
    # Test 1: bitsandbytes
    try:
        r1 = test_bitsandbytes()
        if r1:
            results["tests"].append(r1)
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 2: .zse format
    try:
        r2 = test_zse_format()
        if r2:
            results["tests"].append(r2)
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 3: llama.cpp
    try:
        r3 = test_llamacpp()
        if r3:
            results["tests"].append(r3)
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Summary
    print_header("BENCHMARK SUMMARY")
    print(f"{'Method':<20} {'Load Time':>12} {'VRAM':>10} {'Speed':>12}")
    print("-" * 56)
    for t in results["tests"]:
        print(f"{t['method']:<20} {t['load_time_sec']:>10.2f}s {t['vram_gb']:>8.2f} GB {t['tokens_per_sec']:>10.1f} tok/s")
    
    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    return results

if __name__ == "__main__":
    main()
