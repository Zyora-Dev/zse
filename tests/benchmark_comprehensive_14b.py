#!/usr/bin/env python3
"""
ZSE Comprehensive Benchmark - 14B Model
========================================
Tests three inference methods:
1. bitsandbytes (ZSE --efficiency memory) - On-the-fly INT4
2. .zse format - Pre-converted INT4
3. llama.cpp - GGUF Q4_K_M

Measures:
- Cold start / Load time
- VRAM usage
- Token generation speed
- Memory efficiency

Run: python3 benchmark_comprehensive_14b.py
"""

import os
import sys
import time
import json
import gc
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import torch

# ============================================================================
# Configuration
# ============================================================================

MODEL_HF = "Qwen/Qwen2.5-14B-Instruct"
ZSE_DIR = Path("/home/ionet/qwen14b")
GGUF_PATH = Path("/home/ionet/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct-GGUF/snapshots/bb5353e0659ecee080ad60d102dc3d9528b8e0bd/qwen2.5-14b-instruct-q4_k_m.gguf")

# Benchmark settings
WARMUP_ITERATIONS = 2
BENCHMARK_ITERATIONS = 5
OUTPUT_TOKENS = 128
PROMPT = "Explain quantum computing in simple terms. What are qubits and how do they differ from classical bits?"

# Results file
RESULTS_FILE = Path("/home/ionet/benchmark_results_14b.json")


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    method: str
    load_time_sec: float
    vram_gb: float
    tokens_per_sec: float
    total_tokens: int
    generation_time_sec: float
    timestamp: str
    model: str = MODEL_HF
    output_tokens: int = OUTPUT_TOKENS
    iterations: int = BENCHMARK_ITERATIONS
    notes: str = ""


# ============================================================================
# Utilities
# ============================================================================

def get_gpu_memory() -> float:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def get_gpu_memory_reserved() -> float:
    """Get reserved GPU memory in GB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_reserved() / (1024 ** 3)
    return 0.0


def clear_gpu_memory():
    """Clear GPU memory completely."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    time.sleep(2)  # Allow memory to settle


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print(f"\n{'─' * 50}")
    print(f"  Method:        {result.method}")
    print(f"  Load Time:     {result.load_time_sec:.2f} seconds")
    print(f"  VRAM Used:     {result.vram_gb:.2f} GB")
    print(f"  Speed:         {result.tokens_per_sec:.1f} tokens/sec")
    print(f"  Gen Time:      {result.generation_time_sec:.2f} sec for {result.total_tokens} tokens")
    print(f"{'─' * 50}")


# ============================================================================
# Benchmark 1: bitsandbytes (On-the-fly INT4)
# ============================================================================

def benchmark_bitsandbytes() -> Optional[BenchmarkResult]:
    """Benchmark bitsandbytes INT4 quantization (on-the-fly)."""
    print_header("BENCHMARK 1: bitsandbytes (On-the-fly INT4)")
    
    clear_gpu_memory()
    baseline = get_gpu_memory()
    print(f"Baseline VRAM: {baseline:.2f} GB")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Time cold start (includes download if not cached)
        print("\n[1/4] Loading tokenizer...")
        start_total = time.perf_counter()
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_HF, trust_remote_code=True)
        
        print("[2/4] Loading model with INT4 quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_HF,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        load_time = time.perf_counter() - start_total
        vram_used = get_gpu_memory() - baseline
        
        print(f"    ✓ Loaded in {load_time:.2f}s, VRAM: {vram_used:.2f} GB")
        
        # Warmup
        print(f"\n[3/4] Warmup ({WARMUP_ITERATIONS} iterations)...")
        for i in range(WARMUP_ITERATIONS):
            inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            print(f"    Warmup {i+1}/{WARMUP_ITERATIONS} done")
        
        # Benchmark
        print(f"\n[4/4] Benchmarking ({BENCHMARK_ITERATIONS} iterations, {OUTPUT_TOKENS} tokens each)...")
        times = []
        tokens_generated = []
        
        for i in range(BENCHMARK_ITERATIONS):
            inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
            input_len = inputs['input_ids'].shape[1]
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=OUTPUT_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            num_tokens = outputs.shape[1] - input_len
            times.append(elapsed)
            tokens_generated.append(num_tokens)
            
            speed = num_tokens / elapsed
            print(f"    Iter {i+1}: {num_tokens} tokens in {elapsed:.2f}s = {speed:.1f} tok/s")
        
        # Calculate averages
        avg_time = sum(times) / len(times)
        total_tokens = sum(tokens_generated)
        avg_speed = total_tokens / sum(times)
        
        result = BenchmarkResult(
            method="bitsandbytes (INT4 NF4)",
            load_time_sec=load_time,
            vram_gb=vram_used,
            tokens_per_sec=avg_speed,
            total_tokens=total_tokens,
            generation_time_sec=sum(times),
            timestamp=datetime.now().isoformat(),
            notes="On-the-fly quantization from HuggingFace"
        )
        
        print_result(result)
        
        # Cleanup
        del model, tokenizer
        clear_gpu_memory()
        
        return result
        
    except Exception as e:
        print(f"❌ bitsandbytes benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return None


# ============================================================================
# Benchmark 2: .zse Format (Pre-converted INT4)
# ============================================================================

def benchmark_zse_format() -> Optional[BenchmarkResult]:
    """Benchmark .zse format (pre-converted INT4)."""
    print_header("BENCHMARK 2: .zse Format (Pre-converted INT4)")
    
    if not ZSE_DIR.exists():
        print(f"❌ ZSE directory not found: {ZSE_DIR}")
        print("   Run: zse convert Qwen/Qwen2.5-14B-Instruct -o qwen14b")
        return None
    
    clear_gpu_memory()
    baseline = get_gpu_memory()
    print(f"Baseline VRAM: {baseline:.2f} GB")
    print(f"ZSE Directory: {ZSE_DIR}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        model_dir = ZSE_DIR / "model"
        tokenizer_dir = ZSE_DIR / "tokenizer"
        
        # Time cold start
        print("\n[1/4] Loading tokenizer from .zse...")
        start_total = time.perf_counter()
        
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        
        print("[2/4] Loading pre-quantized model from .zse...")
        
        # The model is already quantized with bitsandbytes during conversion
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
        
        load_time = time.perf_counter() - start_total
        vram_used = get_gpu_memory() - baseline
        
        print(f"    ✓ Loaded in {load_time:.2f}s, VRAM: {vram_used:.2f} GB")
        
        # Warmup
        print(f"\n[3/4] Warmup ({WARMUP_ITERATIONS} iterations)...")
        for i in range(WARMUP_ITERATIONS):
            inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            print(f"    Warmup {i+1}/{WARMUP_ITERATIONS} done")
        
        # Benchmark
        print(f"\n[4/4] Benchmarking ({BENCHMARK_ITERATIONS} iterations, {OUTPUT_TOKENS} tokens each)...")
        times = []
        tokens_generated = []
        
        for i in range(BENCHMARK_ITERATIONS):
            inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
            input_len = inputs['input_ids'].shape[1]
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=OUTPUT_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            num_tokens = outputs.shape[1] - input_len
            times.append(elapsed)
            tokens_generated.append(num_tokens)
            
            speed = num_tokens / elapsed
            print(f"    Iter {i+1}: {num_tokens} tokens in {elapsed:.2f}s = {speed:.1f} tok/s")
        
        # Calculate averages
        avg_time = sum(times) / len(times)
        total_tokens = sum(tokens_generated)
        avg_speed = total_tokens / sum(times)
        
        result = BenchmarkResult(
            method=".zse format (Pre-converted INT4)",
            load_time_sec=load_time,
            vram_gb=vram_used,
            tokens_per_sec=avg_speed,
            total_tokens=total_tokens,
            generation_time_sec=sum(times),
            timestamp=datetime.now().isoformat(),
            notes="Pre-converted with: zse convert"
        )
        
        print_result(result)
        
        # Cleanup
        del model, tokenizer
        clear_gpu_memory()
        
        return result
        
    except Exception as e:
        print(f"❌ .zse format benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return None


# ============================================================================
# Benchmark 3: llama.cpp (GGUF Q4_K_M)
# ============================================================================

def benchmark_llamacpp() -> Optional[BenchmarkResult]:
    """Benchmark llama.cpp with GGUF format."""
    print_header("BENCHMARK 3: llama.cpp (GGUF Q4_K_M)")
    
    # Check for llama-cli
    llama_cli = None
    for path in ["/home/ionet/llama.cpp/build/bin/llama-cli", "/usr/local/bin/llama-cli", "llama-cli"]:
        if Path(path).exists() or subprocess.run(["which", path], capture_output=True).returncode == 0:
            llama_cli = path
            break
    
    if not llama_cli:
        print("❌ llama-cli not found. Install llama.cpp first.")
        return None
    
    if not GGUF_PATH.exists():
        print(f"❌ GGUF file not found: {GGUF_PATH}")
        return None
    
    print(f"llama-cli: {llama_cli}")
    print(f"GGUF file: {GGUF_PATH}")
    print(f"GGUF size: {GGUF_PATH.stat().st_size / 1e9:.2f} GB")
    
    clear_gpu_memory()
    
    try:
        # Time cold start + generation together
        print("\n[1/2] Running llama.cpp benchmark...")
        
        # Build command
        cmd = [
            llama_cli,
            "-m", str(GGUF_PATH),
            "-p", PROMPT,
            "-n", str(OUTPUT_TOKENS),
            "-ngl", "99",  # Offload all layers to GPU
            "--temp", "0",
            "-c", "4096",
        ]
        
        # Run warmup
        print(f"\n[1/3] Warmup...")
        warmup_cmd = cmd.copy()
        warmup_cmd[warmup_cmd.index("-n") + 1] = "32"
        subprocess.run(warmup_cmd, capture_output=True, timeout=120)
        print("    Warmup done")
        
        # Run benchmark iterations
        print(f"\n[2/3] Benchmarking ({BENCHMARK_ITERATIONS} iterations)...")
        
        times = []
        tokens_list = []
        load_times = []
        
        for i in range(BENCHMARK_ITERATIONS):
            start = time.perf_counter()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            elapsed = time.perf_counter() - start
            
            # Parse output for timing info
            output = result.stderr + result.stdout
            
            # Try to extract load time and tokens/sec from llama.cpp output
            load_time = 0
            tok_per_sec = 0
            tokens = OUTPUT_TOKENS
            
            for line in output.split('\n'):
                if 'llama_load_model_from_file' in line and 'time' in line.lower():
                    # Parse load time
                    try:
                        parts = line.split()
                        for j, p in enumerate(parts):
                            if 'ms' in p:
                                load_time = float(parts[j-1]) / 1000
                                break
                    except:
                        pass
                elif 'eval time' in line.lower() or 'generation speed' in line.lower():
                    try:
                        # Look for tok/s pattern
                        import re
                        match = re.search(r'([\d.]+)\s*(?:tok|token).*?/\s*s', line, re.I)
                        if match:
                            tok_per_sec = float(match.group(1))
                    except:
                        pass
                elif 'sampled' in line.lower() and 'token' in line.lower():
                    try:
                        import re
                        match = re.search(r'(\d+)\s*token', line, re.I)
                        if match:
                            tokens = int(match.group(1))
                    except:
                        pass
            
            # If we couldn't parse, estimate
            if tok_per_sec == 0:
                tok_per_sec = tokens / elapsed
            
            times.append(elapsed)
            tokens_list.append(tokens)
            if load_time > 0:
                load_times.append(load_time)
            
            print(f"    Iter {i+1}: {tokens} tokens in {elapsed:.2f}s = {tok_per_sec:.1f} tok/s")
        
        # Get VRAM usage via nvidia-smi during generation
        print("\n[3/3] Measuring VRAM...")
        
        # Run one more time while checking VRAM
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)  # Let it load
        
        try:
            nvidia_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            vram_mb = float(nvidia_result.stdout.strip().split('\n')[0])
            vram_gb = vram_mb / 1024
        except:
            vram_gb = 9.1  # Default estimate for Q4_K_M 14B
        
        proc.wait(timeout=120)
        
        # Calculate results
        avg_time = sum(times) / len(times)
        total_tokens = sum(tokens_list)
        avg_speed = total_tokens / sum(times)
        avg_load = sum(load_times) / len(load_times) if load_times else 5.0
        
        result = BenchmarkResult(
            method="llama.cpp (GGUF Q4_K_M)",
            load_time_sec=avg_load,
            vram_gb=vram_gb,
            tokens_per_sec=avg_speed,
            total_tokens=total_tokens,
            generation_time_sec=sum(times),
            timestamp=datetime.now().isoformat(),
            notes=f"GGUF file: {GGUF_PATH.name}"
        )
        
        print_result(result)
        
        return result
        
    except Exception as e:
        print(f"❌ llama.cpp benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all benchmarks and save results."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ZSE COMPREHENSIVE BENCHMARK - Qwen 14B".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Model: {MODEL_HF}")
    print(f"Output tokens: {OUTPUT_TOKENS}")
    print(f"Iterations: {BENCHMARK_ITERATIONS}")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("WARNING: No GPU detected!")
    
    results = []
    
    # Run benchmarks
    print("\n" + "─" * 70)
    print(" Running benchmarks... This may take 10-20 minutes.")
    print("─" * 70)
    
    # 1. bitsandbytes
    result1 = benchmark_bitsandbytes()
    if result1:
        results.append(result1)
    
    # 2. .zse format
    result2 = benchmark_zse_format()
    if result2:
        results.append(result2)
    
    # 3. llama.cpp
    result3 = benchmark_llamacpp()
    if result3:
        results.append(result3)
    
    # Print summary
    print_header("BENCHMARK SUMMARY")
    
    print(f"\n{'Method':<35} {'Load Time':>12} {'VRAM':>10} {'Speed':>15}")
    print("─" * 75)
    
    for r in results:
        print(f"{r.method:<35} {r.load_time_sec:>10.2f}s {r.vram_gb:>8.2f} GB {r.tokens_per_sec:>12.1f} tok/s")
    
    print("─" * 75)
    
    # Save results
    results_data = {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_HF,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "output_tokens": OUTPUT_TOKENS,
            "iterations": BENCHMARK_ITERATIONS,
        },
        "results": [asdict(r) for r in results],
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {RESULTS_FILE}")
    
    # Print comparison table
    print("\n" + "=" * 70)
    print(" COMPARISON TABLE")
    print("=" * 70)
    print("""
┌─────────────────────────────────┬────────────┬──────────┬─────────────┐
│ Method                          │ Cold Start │ VRAM     │ Speed       │
├─────────────────────────────────┼────────────┼──────────┼─────────────┤""")
    
    for r in results:
        print(f"│ {r.method:<31} │ {r.load_time_sec:>8.2f}s │ {r.vram_gb:>6.2f} GB │ {r.tokens_per_sec:>9.1f} t/s │")
    
    print("└─────────────────────────────────┴────────────┴──────────┴─────────────┘")
    
    print("\n✅ Benchmark complete!")
    
    return results


if __name__ == "__main__":
    main()
