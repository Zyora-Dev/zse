"""
Modal test for fused quantized matmul performance.

Benchmarks:
1. Fused vs unfused kernel performance (synthetic)
2. End-to-end token generation with Qwen 7B

Run: modal run deploy/test_fused_kernels.py
"""

import modal
import os

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(DEPLOY_DIR)

# Create the Modal app
app = modal.App("zse-fused-kernel-benchmark")

# GPU image with ZSE and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "triton==3.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "safetensors>=0.4.0",
        "huggingface_hub",
    ])
    .run_commands("pip install packaging")
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,  # 20 mins
)
def benchmark_fused_kernels():
    """
    Benchmark fused vs unfused quantized matmul kernels.
    """
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import time
    
    print("=" * 70)
    print("ZSE FUSED KERNEL BENCHMARK")
    print("=" * 70)
    
    # Show GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Import ZSE modules
    try:
        from zse.efficiency.triton_quant_kernels import (
            int8_fused_matmul,
            int4_fused_matmul,
            FusedQuantizedLinear,
            benchmark_fused_vs_unfused,
        )
        from zse.efficiency.quantization import (
            QuantizedLinear,
            QuantType,
            dequantize_tensor_int8,
            dequantize_tensor_int4,
        )
        print("\n✓ Fused kernels loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load fused kernels: {e}")
        return {"success": False, "error": str(e)}
    
    results = {}
    
    # ==========================================================================
    # BENCHMARK 1: Synthetic matmul operations
    # ==========================================================================
    print("-" * 70)
    print("BENCHMARK 1: Synthetic Matmul (4096 x 4096)")
    print("-" * 70)
    
    try:
        bench_results = benchmark_fused_vs_unfused(
            in_features=4096,
            out_features=4096,
            batch_size=1,
            seq_len=512,
            num_warmup=10,
            num_iters=50,
        )
        
        print(f"  FP16:        {bench_results['fp16_ms']:.3f} ms")
        print(f"  INT8 fused:  {bench_results['int8_fused_ms']:.3f} ms ({bench_results['int8_speedup']:.2f}x vs FP16)")
        print(f"  INT4 fused:  {bench_results['int4_fused_ms']:.3f} ms ({bench_results['int4_speedup']:.2f}x vs FP16)")
        
        results["synthetic"] = bench_results
    except Exception as e:
        print(f"  ✗ Synthetic benchmark failed: {e}")
        results["synthetic_error"] = str(e)
    
    # ==========================================================================
    # BENCHMARK 2: Fused vs Unfused comparison
    # ==========================================================================
    print("\n" + "-" * 70)
    print("BENCHMARK 2: Fused vs Unfused Comparison")
    print("-" * 70)
    
    device = "cuda"
    dtype = torch.float16
    
    # Test different sizes
    sizes = [(4096, 4096), (4096, 11008), (11008, 4096)]
    
    for in_feat, out_feat in sizes:
        print(f"\n  Size: [{in_feat} x {out_feat}]")
        
        try:
            # Create weights
            x = torch.randn(1, 512, in_feat, dtype=dtype, device=device)
            linear_fp16 = torch.nn.Linear(in_feat, out_feat, bias=True).to(device).half()
            
            # INT8: Compare fused vs unfused
            q_linear_unfused = QuantizedLinear.from_float(linear_fp16, QuantType.INT8)
            q_linear_fused = FusedQuantizedLinear.from_float(linear_fp16, "int8")
            
            # Warmup
            for _ in range(5):
                _ = q_linear_unfused(x)
                _ = q_linear_fused(x)
            torch.cuda.synchronize()
            
            # Benchmark unfused INT8
            start = time.perf_counter()
            for _ in range(20):
                _ = q_linear_unfused(x)
            torch.cuda.synchronize()
            unfused_int8_ms = (time.perf_counter() - start) / 20 * 1000
            
            # Benchmark fused INT8
            start = time.perf_counter()
            for _ in range(20):
                _ = q_linear_fused(x)
            torch.cuda.synchronize()
            fused_int8_ms = (time.perf_counter() - start) / 20 * 1000
            
            speedup_int8 = unfused_int8_ms / fused_int8_ms
            print(f"    INT8 unfused: {unfused_int8_ms:.3f} ms")
            print(f"    INT8 fused:   {fused_int8_ms:.3f} ms ({speedup_int8:.2f}x faster)")
            
            # INT4: Compare fused vs unfused
            q_linear_unfused_int4 = QuantizedLinear.from_float(linear_fp16, QuantType.INT4, group_size=128)
            q_linear_fused_int4 = FusedQuantizedLinear.from_float(linear_fp16, "int4", group_size=128)
            
            # Warmup
            for _ in range(5):
                _ = q_linear_unfused_int4(x)
                _ = q_linear_fused_int4(x)
            torch.cuda.synchronize()
            
            # Benchmark unfused INT4
            start = time.perf_counter()
            for _ in range(20):
                _ = q_linear_unfused_int4(x)
            torch.cuda.synchronize()
            unfused_int4_ms = (time.perf_counter() - start) / 20 * 1000
            
            # Benchmark fused INT4
            start = time.perf_counter()
            for _ in range(20):
                _ = q_linear_fused_int4(x)
            torch.cuda.synchronize()
            fused_int4_ms = (time.perf_counter() - start) / 20 * 1000
            
            speedup_int4 = unfused_int4_ms / fused_int4_ms
            print(f"    INT4 unfused: {unfused_int4_ms:.3f} ms")
            print(f"    INT4 fused:   {fused_int4_ms:.3f} ms ({speedup_int4:.2f}x faster)")
            
            results[f"size_{in_feat}x{out_feat}"] = {
                "int8_unfused_ms": unfused_int8_ms,
                "int8_fused_ms": fused_int8_ms,
                "int8_speedup": speedup_int8,
                "int4_unfused_ms": unfused_int4_ms,
                "int4_fused_ms": fused_int4_ms,
                "int4_speedup": speedup_int4,
            }
            
            # Cleanup
            del linear_fp16, q_linear_unfused, q_linear_fused
            del q_linear_unfused_int4, q_linear_fused_int4
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results[f"size_{in_feat}x{out_feat}_error"] = str(e)
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if "synthetic" in results:
        s = results["synthetic"]
        print(f"\nSynthetic Benchmark (4096x4096 matmul):")
        print(f"  FP16 baseline:  {s['fp16_ms']:.3f} ms")
        print(f"  INT8 fused:     {s['int8_fused_ms']:.3f} ms ({1/s['int8_speedup']*100:.0f}% of FP16 speed)")
        print(f"  INT4 fused:     {s['int4_fused_ms']:.3f} ms ({1/s['int4_speedup']*100:.0f}% of FP16 speed)")
    
    # Calculate average speedup
    int8_speedups = []
    int4_speedups = []
    for key, val in results.items():
        if isinstance(val, dict):
            if "int8_speedup" in val:
                int8_speedups.append(val["int8_speedup"])
            if "int4_speedup" in val:
                int4_speedups.append(val["int4_speedup"])
    
    if int8_speedups:
        avg_int8 = sum(int8_speedups) / len(int8_speedups)
        print(f"\nAverage Fused Speedup:")
        print(f"  INT8: {avg_int8:.2f}x faster than unfused")
    if int4_speedups:
        avg_int4 = sum(int4_speedups) / len(int4_speedups)
        print(f"  INT4: {avg_int4:.2f}x faster than unfused")
    
    results["success"] = True
    return results


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,  # 30 mins
)
def benchmark_qwen_generation():
    """
    Benchmark token generation with Qwen 7B using fused kernels.
    """
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import time
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print("=" * 70)
    print("QWEN 7B TOKEN GENERATION BENCHMARK")
    print("=" * 70)
    
    # Show GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Load model
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Check memory
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"FP16 Model Memory: {allocated:.2f} GB")
    
    # Benchmark FP16 generation
    prompt = "Write a Python function to calculate fibonacci numbers:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("\n" + "-" * 70)
    print("FP16 Generation Benchmark")
    print("-" * 70)
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize()
    
    # Measure
    tokens_to_generate = 50
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=tokens_to_generate, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    fp16_tps = tokens_to_generate / elapsed
    print(f"FP16 Speed: {fp16_tps:.1f} tok/s")
    
    # Now quantize the model with fused kernels
    print("\n" + "-" * 70)
    print("Quantizing with Fused INT8 Kernels")
    print("-" * 70)
    
    try:
        from zse.efficiency.quantization import quantize_model, QuantType
        
        # Create a fresh model for INT8
        del model
        torch.cuda.empty_cache()
        
        model_int8 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        quantize_model(model_int8, QuantType.INT8)
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"INT8 Model Memory: {allocated:.2f} GB")
        
        # Warmup
        with torch.no_grad():
            _ = model_int8.generate(**inputs, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()
        
        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model_int8.generate(**inputs, max_new_tokens=tokens_to_generate, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        int8_tps = tokens_to_generate / elapsed
        print(f"INT8 Fused Speed: {int8_tps:.1f} tok/s ({int8_tps/fp16_tps*100:.0f}% of FP16)")
        
        del model_int8
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"INT8 benchmark failed: {e}")
        int8_tps = 0
    
    # Quantize with INT4
    print("\n" + "-" * 70)
    print("Quantizing with Fused INT4 Kernels")
    print("-" * 70)
    
    try:
        model_int4 = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        quantize_model(model_int4, QuantType.INT4, group_size=128)
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"INT4 Model Memory: {allocated:.2f} GB")
        
        # Warmup  
        with torch.no_grad():
            _ = model_int4.generate(**inputs, max_new_tokens=10, do_sample=False)
        torch.cuda.synchronize()
        
        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model_int4.generate(**inputs, max_new_tokens=tokens_to_generate, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        int4_tps = tokens_to_generate / elapsed
        print(f"INT4 Fused Speed: {int4_tps:.1f} tok/s ({int4_tps/fp16_tps*100:.0f}% of FP16)")
        
    except Exception as e:
        print(f"INT4 benchmark failed: {e}")
        int4_tps = 0
    
    # Summary
    print("\n" + "=" * 70)
    print("QWEN 7B GENERATION SUMMARY")
    print("=" * 70)
    print(f"FP16:       {fp16_tps:.1f} tok/s (baseline)")
    print(f"INT8 Fused: {int8_tps:.1f} tok/s ({int8_tps/fp16_tps*100:.0f}% of FP16)")
    print(f"INT4 Fused: {int4_tps:.1f} tok/s ({int4_tps/fp16_tps*100:.0f}% of FP16)")
    print("=" * 70)
    
    return {
        "fp16_tps": fp16_tps,
        "int8_tps": int8_tps,
        "int4_tps": int4_tps,
    }


@app.local_entrypoint()
def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("RUNNING FUSED KERNEL BENCHMARKS ON MODAL A10G")
    print("=" * 70 + "\n")
    
    # Run synthetic kernel benchmarks
    print("Starting kernel benchmarks...")
    kernel_results = benchmark_fused_kernels.remote()
    print(f"\nKernel benchmark complete: {kernel_results.get('success', False)}")
    
    # Run Qwen generation benchmark
    print("\nStarting Qwen generation benchmark...")
    gen_results = benchmark_qwen_generation.remote()
    print(f"\nGeneration benchmark complete")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
    print(f"Kernel Results: {kernel_results}")
    print(f"Generation Results: {gen_results}")
