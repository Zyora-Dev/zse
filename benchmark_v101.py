#!/usr/bin/env python3
"""
ZSE v1.0.1 Comprehensive Benchmark

Compares:
1. ZSE INT4 (bitsandbytes) 
2. ZSE FP16
3. Standard bitsandbytes (transformers)
4. Cold start times
5. Throughput (tokens/sec)
6. Memory efficiency

Run on H200 server:
    python benchmark_v101.py
"""

import torch
import time
import gc
import os

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
PROMPT = "Write a Python function to calculate fibonacci numbers recursively with memoization. Include docstring and type hints."
MAX_TOKENS = 200
WARMUP_TOKENS = 20

def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def get_peak_memory():
    """Get peak GPU memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0

def benchmark_zse_int4():
    """Benchmark ZSE with INT4 quantization."""
    print("\n" + "="*70)
    print("🔹 BENCHMARK: ZSE INT4 (bitsandbytes NF4)")
    print("="*70)
    
    clear_gpu()
    
    # Cold start
    cold_start = time.perf_counter()
    
    from zse.engine.orchestrator import IntelligenceOrchestrator
    orch = IntelligenceOrchestrator.min_memory(MODEL_ID)
    orch.load(verbose=False)
    
    cold_start_time = time.perf_counter() - cold_start
    load_memory = get_gpu_memory()
    
    print(f"   Cold Start: {cold_start_time:.2f}s")
    print(f"   VRAM After Load: {load_memory:.2f} GB")
    
    # Warmup
    print("   Warming up...")
    for _ in orch.generate("Hello", max_tokens=WARMUP_TOKENS, stream=True):
        pass
    
    # Throughput benchmark
    print("   Running throughput test...")
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    tokens = 0
    output = []
    for chunk in orch.generate(PROMPT, max_tokens=MAX_TOKENS, stream=True):
        output.append(chunk)
        tokens += 1
    
    elapsed = time.perf_counter() - start
    throughput = tokens / elapsed
    peak_memory = get_peak_memory()
    
    print(f"   Tokens Generated: {tokens}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.2f} tok/s")
    print(f"   Peak VRAM: {peak_memory:.2f} GB")
    
    # Cleanup
    del orch
    clear_gpu()
    
    return {
        "name": "ZSE INT4",
        "cold_start": cold_start_time,
        "vram_load": load_memory,
        "vram_peak": peak_memory,
        "throughput": throughput,
        "tokens": tokens,
    }

def benchmark_zse_fp16():
    """Benchmark ZSE with FP16."""
    print("\n" + "="*70)
    print("🔹 BENCHMARK: ZSE FP16")
    print("="*70)
    
    clear_gpu()
    
    # Cold start
    cold_start = time.perf_counter()
    
    from zse.engine.orchestrator import IntelligenceOrchestrator
    orch = IntelligenceOrchestrator.max_speed(MODEL_ID)
    orch.load(verbose=False)
    
    cold_start_time = time.perf_counter() - cold_start
    load_memory = get_gpu_memory()
    
    print(f"   Cold Start: {cold_start_time:.2f}s")
    print(f"   VRAM After Load: {load_memory:.2f} GB")
    
    # Warmup
    print("   Warming up...")
    for _ in orch.generate("Hello", max_tokens=WARMUP_TOKENS, stream=True):
        pass
    
    # Throughput benchmark
    print("   Running throughput test...")
    torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    tokens = 0
    for chunk in orch.generate(PROMPT, max_tokens=MAX_TOKENS, stream=True):
        tokens += 1
    
    elapsed = time.perf_counter() - start
    throughput = tokens / elapsed
    peak_memory = get_peak_memory()
    
    print(f"   Tokens Generated: {tokens}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.2f} tok/s")
    print(f"   Peak VRAM: {peak_memory:.2f} GB")
    
    # Cleanup
    del orch
    clear_gpu()
    
    return {
        "name": "ZSE FP16",
        "cold_start": cold_start_time,
        "vram_load": load_memory,
        "vram_peak": peak_memory,
        "throughput": throughput,
        "tokens": tokens,
    }

def benchmark_raw_bnb_int4():
    """Benchmark raw bitsandbytes INT4 (no ZSE)."""
    print("\n" + "="*70)
    print("🔹 BENCHMARK: Raw bitsandbytes INT4 (transformers only)")
    print("="*70)
    
    clear_gpu()
    
    # Cold start
    cold_start = time.perf_counter()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    cold_start_time = time.perf_counter() - cold_start
    load_memory = get_gpu_memory()
    
    print(f"   Cold Start: {cold_start_time:.2f}s")
    print(f"   VRAM After Load: {load_memory:.2f} GB")
    
    # Warmup
    print("   Warming up...")
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)
    
    # Throughput benchmark  
    print("   Running throughput test...")
    torch.cuda.reset_peak_memory_stats()
    
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_TOKENS, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    elapsed = time.perf_counter() - start
    
    tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    throughput = tokens / elapsed
    peak_memory = get_peak_memory()
    
    print(f"   Tokens Generated: {tokens}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.2f} tok/s")
    print(f"   Peak VRAM: {peak_memory:.2f} GB")
    
    # Cleanup
    del model, tokenizer
    clear_gpu()
    
    return {
        "name": "Raw BNB INT4",
        "cold_start": cold_start_time,
        "vram_load": load_memory,
        "vram_peak": peak_memory,
        "throughput": throughput,
        "tokens": tokens,
    }

def benchmark_raw_fp16():
    """Benchmark raw FP16 (transformers only, no ZSE)."""
    print("\n" + "="*70)
    print("🔹 BENCHMARK: Raw FP16 (transformers only)")
    print("="*70)
    
    clear_gpu()
    
    # Cold start
    cold_start = time.perf_counter()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    cold_start_time = time.perf_counter() - cold_start
    load_memory = get_gpu_memory()
    
    print(f"   Cold Start: {cold_start_time:.2f}s")
    print(f"   VRAM After Load: {load_memory:.2f} GB")
    
    # Warmup
    print("   Warming up...")
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)
    
    # Throughput benchmark  
    print("   Running throughput test...")
    torch.cuda.reset_peak_memory_stats()
    
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_TOKENS, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    elapsed = time.perf_counter() - start
    
    tokens = outputs.shape[1] - inputs.input_ids.shape[1]
    throughput = tokens / elapsed
    peak_memory = get_peak_memory()
    
    print(f"   Tokens Generated: {tokens}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.2f} tok/s")
    print(f"   Peak VRAM: {peak_memory:.2f} GB")
    
    # Cleanup
    del model, tokenizer
    clear_gpu()
    
    return {
        "name": "Raw FP16",
        "cold_start": cold_start_time,
        "vram_load": load_memory,
        "vram_peak": peak_memory,
        "throughput": throughput,
        "tokens": tokens,
    }

def print_summary(results):
    """Print benchmark summary table."""
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY: Qwen2.5-14B-Instruct")
    print("="*70)
    
    # Header
    print(f"\n{'Backend':<20} {'Cold Start':<12} {'VRAM Load':<12} {'VRAM Peak':<12} {'Throughput':<12}")
    print("-"*70)
    
    for r in results:
        print(f"{r['name']:<20} {r['cold_start']:.2f}s{'':<6} {r['vram_load']:.2f} GB{'':<4} {r['vram_peak']:.2f} GB{'':<4} {r['throughput']:.1f} tok/s")
    
    print("-"*70)
    
    # Analysis
    print("\n📈 ANALYSIS:")
    
    # Find best in each category
    best_cold = min(results, key=lambda x: x['cold_start'])
    best_vram = min(results, key=lambda x: x['vram_peak'])
    best_throughput = max(results, key=lambda x: x['throughput'])
    
    print(f"   Fastest Cold Start: {best_cold['name']} ({best_cold['cold_start']:.2f}s)")
    print(f"   Lowest VRAM: {best_vram['name']} ({best_vram['vram_peak']:.2f} GB)")
    print(f"   Highest Throughput: {best_throughput['name']} ({best_throughput['throughput']:.1f} tok/s)")
    
    # ZSE vs Raw comparison
    zse_int4 = next((r for r in results if r['name'] == 'ZSE INT4'), None)
    raw_int4 = next((r for r in results if r['name'] == 'Raw BNB INT4'), None)
    
    if zse_int4 and raw_int4:
        print(f"\n   ZSE INT4 vs Raw BNB INT4:")
        cold_diff = raw_int4['cold_start'] / zse_int4['cold_start']
        vram_diff = (raw_int4['vram_peak'] - zse_int4['vram_peak']) / raw_int4['vram_peak'] * 100
        throughput_ratio = zse_int4['throughput'] / raw_int4['throughput']
        print(f"     Cold Start: ZSE is {cold_diff:.1f}x faster" if cold_diff > 1 else f"     Cold Start: Raw is {1/cold_diff:.1f}x faster")
        print(f"     VRAM: ZSE uses {abs(vram_diff):.1f}% {'less' if vram_diff > 0 else 'more'}")
        print(f"     Throughput: ZSE is {throughput_ratio:.1%} of Raw")

def main():
    print("="*70)
    print("🚀 ZSE v1.0.1 Comprehensive Benchmark")
    print("="*70)
    print(f"Model: {MODEL_ID}")
    print(f"Prompt tokens: ~{len(PROMPT.split())*1.3:.0f}")
    print(f"Max new tokens: {MAX_TOKENS}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {total_vram:.1f} GB")
    
    results = []
    
    # Run benchmarks
    try:
        results.append(benchmark_zse_int4())
    except Exception as e:
        print(f"   ❌ ZSE INT4 failed: {e}")
    
    try:
        results.append(benchmark_zse_fp16())
    except Exception as e:
        print(f"   ❌ ZSE FP16 failed: {e}")
    
    try:
        results.append(benchmark_raw_bnb_int4())
    except Exception as e:
        print(f"   ❌ Raw BNB INT4 failed: {e}")
    
    try:
        results.append(benchmark_raw_fp16())
    except Exception as e:
        print(f"   ❌ Raw FP16 failed: {e}")
    
    # Print summary
    if results:
        print_summary(results)
    
    print("\n✅ Benchmark complete!")

if __name__ == "__main__":
    main()
