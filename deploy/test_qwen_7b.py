"""
ZSE Full Pipeline Test: Qwen 2.5 Coder 7B

Tests the complete ZSE stack:
1. Memory profiling (before/after loading)
2. INT8 quantization (50% memory savings)
3. Paged KV cache (efficient generation)
4. Streaming text output
5. Memory efficiency comparison

Usage:
    modal run deploy/test_qwen_7b.py
"""

import modal
import os

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(DEPLOY_DIR)

app = modal.App("zse-qwen-7b-test")

# Image with all dependencies + ZSE code
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
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM - enough for 7B model
    timeout=1800,  # 30 min timeout for model download
)
def test_qwen_full_pipeline():
    """
    Full ZSE pipeline test with Qwen 2.5 Coder 7B.
    
    Demonstrates:
    - Memory profiling
    - INT8 quantization 
    - Streaming generation
    - KV cache efficiency
    """
    import torch
    import torch.nn as nn
    import time
    import gc
    import sys
    sys.path.insert(0, "/root/zse")
    
    from zse.engine.generation import TextGenerator, SamplingParams, StreamChunk
    from zse.efficiency.quantization import quantize_model, get_model_memory, QuantType
    
    print("=" * 70)
    print("ZSE Full Pipeline Test: Qwen 2.5 Coder 7B")
    print("=" * 70)
    
    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    print(f"\n📊 GPU: {gpu_name}")
    print(f"   Total VRAM: {format_bytes(gpu_total)}")
    
    # Initial memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    print(f"   Initial allocated: {format_bytes(initial_allocated)}")
    
    # =========================================================================
    # PHASE 1: Load Model (FP16)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Loading Qwen 2.5 Coder 7B (FP16)")
    print("=" * 70)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    
    print(f"\n📥 Loading model: {model_name}")
    print("   This may take a few minutes on first run...")
    
    load_start = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load in FP16 to measure baseline
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    
    load_time = time.perf_counter() - load_start
    
    # Memory after FP16 load
    fp16_allocated = torch.cuda.memory_allocated()
    fp16_reserved = torch.cuda.memory_reserved()
    fp16_peak = torch.cuda.max_memory_allocated()
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_count_b = param_count / 1e9
    
    print(f"\n✅ Model loaded in {load_time:.1f}s")
    print(f"   Parameters: {param_count:,} ({param_count_b:.2f}B)")
    print(f"   FP16 Memory: {format_bytes(fp16_allocated)}")
    print(f"   Peak Memory: {format_bytes(fp16_peak)}")
    print(f"   Theoretical FP16: {format_bytes(param_count * 2)}")  # 2 bytes per param
    
    # Memory efficiency
    theoretical = param_count * 2
    overhead = (fp16_allocated - theoretical) / theoretical * 100
    print(f"   Memory Overhead: {overhead:.1f}%")
    
    # =========================================================================
    # PHASE 2: Test FP16 Generation (Baseline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FP16 Baseline Generation")
    print("=" * 70)
    
    generator = TextGenerator(model, tokenizer, device="cuda")
    
    # Test prompt - coding task
    prompt = """<|im_start|>system
You are a helpful coding assistant.<|im_end|>
<|im_start|>user
Write a Python function to find the longest common subsequence of two strings. Include type hints and a docstring.<|im_end|>
<|im_start|>assistant
"""
    
    params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_new_tokens=200,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print("\n📝 Prompt: Write a LCS function in Python")
    print("-" * 50)
    print("🔄 Streaming output (FP16):\n")
    
    torch.cuda.reset_peak_memory_stats()
    gen_start = time.perf_counter()
    
    fp16_tokens = []
    fp16_text = ""
    latencies = []
    
    for chunk in generator.generate_stream(prompt, params):
        fp16_tokens.append(chunk.token_id)
        fp16_text += chunk.text
        latencies.append(chunk.latency_ms)
        print(chunk.text, end="", flush=True)
    
    fp16_gen_time = time.perf_counter() - gen_start
    fp16_gen_peak = torch.cuda.max_memory_allocated()
    
    print("\n" + "-" * 50)
    print(f"\n📊 FP16 Generation Stats:")
    print(f"   Tokens generated: {len(fp16_tokens)}")
    print(f"   Total time: {fp16_gen_time:.2f}s")
    print(f"   Throughput: {len(fp16_tokens)/fp16_gen_time:.1f} tokens/sec")
    print(f"   Avg latency: {sum(latencies)/len(latencies):.1f}ms per token")
    print(f"   Peak memory during generation: {format_bytes(fp16_gen_peak)}")
    
    # =========================================================================
    # PHASE 3: INT8 Quantization
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: INT8 Quantization")
    print("=" * 70)
    
    # Clear caches
    del generator
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n🗜️ Quantizing model to INT8...")
    print("   Skipping: embed_tokens, lm_head, layernorm")
    
    quant_start = time.perf_counter()
    
    # Quantize the model
    quantized_model = quantize_model(
        model,
        quant_type=QuantType.INT8,
        skip_layers=["embed", "lm_head", "norm", "layernorm"],
    )
    
    quant_time = time.perf_counter() - quant_start
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    # Measure INT8 memory
    int8_allocated = torch.cuda.memory_allocated()
    
    print(f"\n✅ Quantization complete in {quant_time:.1f}s")
    print(f"   INT8 Memory: {format_bytes(int8_allocated)}")
    print(f"   Memory saved: {format_bytes(fp16_allocated - int8_allocated)}")
    print(f"   Reduction: {(1 - int8_allocated/fp16_allocated)*100:.1f}%")
    
    # =========================================================================
    # PHASE 4: INT8 Generation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: INT8 Generation Test")
    print("=" * 70)
    
    generator_int8 = TextGenerator(quantized_model, tokenizer, device="cuda")
    
    print("\n🔄 Streaming output (INT8):\n")
    
    torch.cuda.reset_peak_memory_stats()
    gen_start = time.perf_counter()
    
    int8_tokens = []
    int8_text = ""
    latencies_int8 = []
    
    for chunk in generator_int8.generate_stream(prompt, params):
        int8_tokens.append(chunk.token_id)
        int8_text += chunk.text
        latencies_int8.append(chunk.latency_ms)
        print(chunk.text, end="", flush=True)
    
    int8_gen_time = time.perf_counter() - gen_start
    int8_gen_peak = torch.cuda.max_memory_allocated()
    
    print("\n" + "-" * 50)
    print(f"\n📊 INT8 Generation Stats:")
    print(f"   Tokens generated: {len(int8_tokens)}")
    print(f"   Total time: {int8_gen_time:.2f}s")
    print(f"   Throughput: {len(int8_tokens)/int8_gen_time:.1f} tokens/sec")
    print(f"   Avg latency: {sum(latencies_int8)/len(latencies_int8):.1f}ms per token")
    print(f"   Peak memory during generation: {format_bytes(int8_gen_peak)}")
    
    # =========================================================================
    # PHASE 5: Summary & Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Summary & Comparison")
    print("=" * 70)
    
    print("\n📊 Memory Efficiency:")
    print(f"   {'Metric':<30} {'FP16':>15} {'INT8':>15} {'Savings':>15}")
    print(f"   {'-'*75}")
    print(f"   {'Model Memory':<30} {format_bytes(fp16_allocated):>15} {format_bytes(int8_allocated):>15} {(1-int8_allocated/fp16_allocated)*100:>14.1f}%")
    print(f"   {'Peak Generation':<30} {format_bytes(fp16_gen_peak):>15} {format_bytes(int8_gen_peak):>15} {(1-int8_gen_peak/fp16_gen_peak)*100:>14.1f}%")
    
    print("\n⚡ Performance:")
    print(f"   {'Metric':<30} {'FP16':>15} {'INT8':>15} {'Change':>15}")
    print(f"   {'-'*75}")
    print(f"   {'Tokens/sec':<30} {len(fp16_tokens)/fp16_gen_time:>15.1f} {len(int8_tokens)/int8_gen_time:>15.1f} {(int8_gen_time/fp16_gen_time-1)*100:>+14.1f}%")
    print(f"   {'Avg Latency (ms)':<30} {sum(latencies)/len(latencies):>15.1f} {sum(latencies_int8)/len(latencies_int8):>15.1f} {(sum(latencies_int8)/len(latencies_int8))/(sum(latencies)/len(latencies))*100-100:>+14.1f}%")
    
    print("\n🎯 ZSE Efficiency Targets:")
    target_7b = 3.5 * 1024**3  # 3.5 GB target
    print(f"   Target for 7B model: {format_bytes(target_7b)}")
    print(f"   Achieved with INT8: {format_bytes(int8_allocated)}")
    if int8_allocated <= target_7b:
        print(f"   ✅ Target MET! {format_bytes(target_7b - int8_allocated)} under budget")
    else:
        print(f"   ⚠️ {format_bytes(int8_allocated - target_7b)} over target")
        print(f"   💡 Use INT4 quantization for 75% reduction to meet target")
    
    print("\n" + "=" * 70)
    print("ZSE Full Pipeline Test Complete!")
    print("=" * 70)
    
    return {
        "model": model_name,
        "parameters": param_count,
        "fp16_memory_gb": fp16_allocated / 1024**3,
        "int8_memory_gb": int8_allocated / 1024**3,
        "memory_reduction_pct": (1 - int8_allocated/fp16_allocated) * 100,
        "fp16_tokens_per_sec": len(fp16_tokens) / fp16_gen_time,
        "int8_tokens_per_sec": len(int8_tokens) / int8_gen_time,
        "gpu": gpu_name,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
def test_memory_advisor():
    """Test the memory advisor/orchestrator concept."""
    import torch
    import sys
    sys.path.insert(0, "/root/zse")
    
    print("=" * 70)
    print("ZSE Intelligence Orchestrator Demo")
    print("=" * 70)
    
    # Get hardware info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    gpu_free = gpu_total - torch.cuda.memory_allocated()
    
    print(f"\n💻 Hardware Profile:")
    print(f"   GPU: {gpu_name}")
    print(f"   Total VRAM: {format_bytes(gpu_total)}")
    print(f"   Free VRAM: {format_bytes(gpu_free)}")
    
    # Model analysis
    models = [
        ("Qwen2.5-Coder-7B", 7.6e9),
        ("Qwen2.5-Coder-14B", 14.2e9),
        ("Qwen2.5-Coder-32B", 32.5e9),
        ("Llama-3-70B", 70e9),
    ]
    
    print(f"\n🎯 ZSE Memory Recommendations:")
    print(f"   {'Model':<25} {'FP16':>12} {'INT8':>12} {'INT4':>12} {'Fits?':>10}")
    print(f"   {'-'*75}")
    
    for name, params in models:
        fp16_size = params * 2  # 2 bytes per param
        int8_size = params * 1.1  # ~1.1 bytes (with scales)
        int4_size = params * 0.6  # ~0.6 bytes (with scales)
        
        # Check if fits in free VRAM (with 20% buffer for KV cache)
        fits_fp16 = "✅" if fp16_size * 1.2 < gpu_free else "❌"
        fits_int8 = "✅" if int8_size * 1.2 < gpu_free else "❌"
        fits_int4 = "✅" if int4_size * 1.2 < gpu_free else "❌"
        
        best_fit = fits_int4 if fits_int4 == "✅" else (fits_int8 if fits_int8 == "✅" else fits_fp16)
        
        print(f"   {name:<25} {format_bytes(fp16_size):>12} {format_bytes(int8_size):>12} {format_bytes(int4_size):>12} {best_fit:>10}")
    
    print(f"\n💡 Recommendations for {format_bytes(gpu_free)} free VRAM:")
    
    for name, params in models:
        fp16_size = params * 2
        int8_size = params * 1.1
        int4_size = params * 0.6
        
        if fp16_size * 1.2 < gpu_free:
            print(f"   {name}: Use FP16 (best quality)")
        elif int8_size * 1.2 < gpu_free:
            print(f"   {name}: Use INT8 (50% smaller, minimal quality loss)")
        elif int4_size * 1.2 < gpu_free:
            print(f"   {name}: Use INT4 (75% smaller, some quality loss)")
        else:
            print(f"   {name}: ❌ Won't fit - need {format_bytes(int4_size * 1.2 - gpu_free)} more VRAM")
    
    return {"status": "success"}


@app.local_entrypoint()
def main():
    print("\n🚀 Running ZSE Full Pipeline Test with Qwen 2.5 Coder 7B...")
    print("   This will take ~5-10 minutes for model download + testing\n")
    
    result = test_qwen_full_pipeline.remote()
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nModel: {result['model']}")
    print(f"Parameters: {result['parameters']:,}")
    print(f"FP16 Memory: {result['fp16_memory_gb']:.2f} GB")
    print(f"INT8 Memory: {result['int8_memory_gb']:.2f} GB")
    print(f"Memory Reduction: {result['memory_reduction_pct']:.1f}%")
    print(f"FP16 Throughput: {result['fp16_tokens_per_sec']:.1f} tokens/sec")
    print(f"INT8 Throughput: {result['int8_tokens_per_sec']:.1f} tokens/sec")
