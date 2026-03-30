"""
ZSE Memory Optimization Test

Demonstrates the memory/speed trade-off:
- INT4: 3.5-4 GB (minimum memory)
- INT8: ~8 GB (balanced)
- FP16: ~14 GB (maximum speed)

Usage:
    modal run deploy/test_memory_optimization.py
"""

import modal
import os

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(DEPLOY_DIR)

app = modal.App("zse-memory-optimization")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "pynvml",
    )
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)


def format_bytes(size_bytes: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


@app.function(
    image=image,
    gpu="A10G",
    timeout=2400,
)
def test_memory_optimization():
    """
    Test all quantization levels and compare memory/speed.
    """
    import torch
    import time
    import gc
    import sys
    sys.path.insert(0, "/root/zse")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from zse.engine.generation import TextGenerator, SamplingParams
    from zse.efficiency.quantization import quantize_model, QuantType, get_model_memory
    
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    prompt = "Write a binary search function in Python"
    max_tokens = 100
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print("=" * 70)
    print("ZSE Memory Optimization Test")
    print("=" * 70)
    print(f"\n📊 GPU: {gpu_name} ({gpu_total:.1f} GB)")
    print(f"📦 Model: {model_name}")
    print()
    
    results = {}
    
    # Common sampling params
    params = SamplingParams(max_new_tokens=max_tokens, temperature=0.7)
    
    # =========================================================================
    # TEST 1: FP16 (Maximum Speed)
    # =========================================================================
    print("=" * 70)
    print("TEST 1: FP16 (Maximum Speed, Maximum Memory)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    print("\n📥 Loading FP16 model...")
    start = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    load_time = time.perf_counter() - start
    fp16_memory = torch.cuda.memory_allocated() / (1024**3)
    
    print(f"✅ Loaded in {load_time:.1f}s")
    print(f"   Memory: {fp16_memory:.2f} GB")
    
    # Generate
    generator = TextGenerator(model_fp16, tokenizer, device="cuda")
    
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    
    output = []
    for chunk in generator.generate_stream(prompt, params):
        output.append(chunk.text)
    
    gen_time = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
    results["fp16"] = {
        "memory_gb": fp16_memory,
        "peak_memory_gb": peak_memory,
        "tokens_per_sec": max_tokens / gen_time,
        "latency_ms": (gen_time / max_tokens) * 1000,
    }
    
    print(f"\n📊 FP16 Results:")
    print(f"   Model Memory: {fp16_memory:.2f} GB")
    print(f"   Peak Memory: {peak_memory:.2f} GB")
    print(f"   Speed: {results['fp16']['tokens_per_sec']:.1f} tok/s")
    print(f"   Latency: {results['fp16']['latency_ms']:.1f} ms/tok")
    
    # Cleanup
    del model_fp16
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    
    # =========================================================================
    # TEST 2: INT8 (Balanced)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: INT8 (Balanced Memory/Speed)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    print("\n📥 Loading and quantizing to INT8...")
    start = time.perf_counter()
    
    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    model_int8 = quantize_model(
        model_int8,
        quant_type=QuantType.INT8,
        skip_layers=["embed", "lm_head", "norm", "layernorm"],
    )
    
    load_time = time.perf_counter() - start
    int8_memory = torch.cuda.memory_allocated() / (1024**3)
    
    print(f"✅ Loaded in {load_time:.1f}s")
    print(f"   Memory: {int8_memory:.2f} GB")
    
    # Generate
    generator = TextGenerator(model_int8, tokenizer, device="cuda")
    
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    
    output = []
    for chunk in generator.generate_stream(prompt, params):
        output.append(chunk.text)
    
    gen_time = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
    results["int8"] = {
        "memory_gb": int8_memory,
        "peak_memory_gb": peak_memory,
        "tokens_per_sec": max_tokens / gen_time,
        "latency_ms": (gen_time / max_tokens) * 1000,
    }
    
    print(f"\n📊 INT8 Results:")
    print(f"   Model Memory: {int8_memory:.2f} GB")
    print(f"   Peak Memory: {peak_memory:.2f} GB")
    print(f"   Speed: {results['int8']['tokens_per_sec']:.1f} tok/s")
    print(f"   Latency: {results['int8']['latency_ms']:.1f} ms/tok")
    print(f"   Memory Saved: {(1 - int8_memory/fp16_memory)*100:.1f}%")
    
    # Cleanup
    del model_int8
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    
    # =========================================================================
    # TEST 3: INT4 (Minimum Memory)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: INT4 (Minimum Memory ~3.5-4 GB)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    print("\n📥 Loading and quantizing to INT4...")
    start = time.perf_counter()
    
    model_int4 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    model_int4 = quantize_model(
        model_int4,
        quant_type=QuantType.INT4,
        skip_layers=["embed", "lm_head", "norm", "layernorm"],
        group_size=128,
    )
    
    load_time = time.perf_counter() - start
    int4_memory = torch.cuda.memory_allocated() / (1024**3)
    
    print(f"✅ Loaded in {load_time:.1f}s")
    print(f"   Memory: {int4_memory:.2f} GB")
    
    # Generate
    generator = TextGenerator(model_int4, tokenizer, device="cuda")
    
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    
    output = []
    for chunk in generator.generate_stream(prompt, params):
        output.append(chunk.text)
    
    gen_time = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    
    results["int4"] = {
        "memory_gb": int4_memory,
        "peak_memory_gb": peak_memory,
        "tokens_per_sec": max_tokens / gen_time,
        "latency_ms": (gen_time / max_tokens) * 1000,
    }
    
    print(f"\n📊 INT4 Results:")
    print(f"   Model Memory: {int4_memory:.2f} GB")
    print(f"   Peak Memory: {peak_memory:.2f} GB")
    print(f"   Speed: {results['int4']['tokens_per_sec']:.1f} tok/s")
    print(f"   Latency: {results['int4']['latency_ms']:.1f} ms/tok")
    print(f"   Memory Saved: {(1 - int4_memory/fp16_memory)*100:.1f}%")
    
    # Cleanup
    del model_int4
    del generator
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("MEMORY/SPEED TRADE-OFF SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Mode':<10} {'Memory':<12} {'Peak':<12} {'Speed':<12} {'Latency':<12} {'Savings':<10}")
    print("-" * 70)
    
    for mode in ["fp16", "int8", "int4"]:
        r = results[mode]
        savings = (1 - r["memory_gb"] / results["fp16"]["memory_gb"]) * 100
        print(f"{mode.upper():<10} {r['memory_gb']:.2f} GB     {r['peak_memory_gb']:.2f} GB     "
              f"{r['tokens_per_sec']:.1f} tok/s   {r['latency_ms']:.1f} ms      {savings:.0f}%")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│ Your VRAM    │ Recommended  │ Expected Memory │ Expected Speed     │
├─────────────────────────────────────────────────────────────────────┤
│ 4 GB GPU     │ INT4         │ ~{results['int4']['memory_gb']:.1f} GB          │ ~{results['int4']['tokens_per_sec']:.0f} tok/s           │
│ 8 GB GPU     │ INT8         │ ~{results['int8']['memory_gb']:.1f} GB          │ ~{results['int8']['tokens_per_sec']:.0f} tok/s            │
│ 16+ GB GPU   │ FP16         │ ~{results['fp16']['memory_gb']:.1f} GB         │ ~{results['fp16']['tokens_per_sec']:.0f} tok/s           │
└─────────────────────────────────────────────────────────────────────┘
    """)
    
    # Check if INT4 meets target
    target_gb = 4.0
    if results["int4"]["memory_gb"] <= target_gb:
        print(f"✅ INT4 achieves target: {results['int4']['memory_gb']:.2f} GB <= {target_gb} GB")
    else:
        print(f"⚠️ INT4 above target: {results['int4']['memory_gb']:.2f} GB > {target_gb} GB")
        print(f"   (May need additional optimizations or layer skipping)")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    
    return results


@app.local_entrypoint()
def main():
    results = test_memory_optimization.remote()
    print("\nFinal Results:", results)
