#!/usr/bin/env python3
"""
ZSE Real Benchmark v2 - Qwen2.5-14B-Instruct
=============================================
HONEST benchmark using FIXED ZSE implementation.

Tests:
1. bitsandbytes NF4 (baseline)
2. ZSE .zse format with REAL INT4 quantization
3. llama.cpp GGUF Q4_K_M

Metrics: Cold start time, VRAM usage, inference speed, file size

Hardware: NVIDIA H200 (144GB VRAM)
Date: 2026-02-27

NO FAKE DATA. Real measurements only.
"""

import os
import sys
import time
import json
import gc
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
GGUF_REPO = "bartowski/Qwen2.5-14B-Instruct-GGUF"
GGUF_FILE = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
ZSE_FILE = "qwen14b.zse"
TEST_PROMPT = "Write a Python function to calculate fibonacci numbers efficiently."
RESULTS_FILE = Path("benchmark_14b_v2_results.json")


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip().split('\n')[0]) / 1024
    except:
        return -1


def get_gpu_total():
    """Get total GPU memory in GB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip().split('\n')[0]) / 1024
    except:
        return -1


def clear_gpu():
    """Clear GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    time.sleep(2)


def log_result(test_name, status, data):
    """Log result to file."""
    results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    
    results.append({
        "timestamp": datetime.now().isoformat(),
        "test": test_name,
        "status": status,
        "data": data
    })
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[LOG] {test_name}: {status}")


def test_bitsandbytes():
    """Test bitsandbytes NF4 as baseline."""
    print("\n" + "="*60)
    print("TEST 1: bitsandbytes NF4 (BASELINE)")
    print("="*60)
    
    clear_gpu()
    vram_before = get_gpu_memory()
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print(f"VRAM before: {vram_before:.2f} GB")
        print(f"Loading {MODEL_ID} with NF4...")
        
        start = time.perf_counter()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        load_time = time.perf_counter() - start
        vram_after = get_gpu_memory()
        
        print(f"Cold start: {load_time:.2f}s")
        print(f"VRAM used: {vram_after - vram_before:.2f} GB")
        
        # Inference test
        print("Running inference...")
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
        
        inf_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        inf_time = time.perf_counter() - inf_start
        
        tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tok_s = tokens / inf_time
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated {tokens} tokens in {inf_time:.2f}s ({tok_s:.1f} tok/s)")
        print(f"Output: {output_text[:150]}...")
        
        result = {
            "cold_start_s": round(load_time, 2),
            "vram_gb": round(vram_after - vram_before, 2),
            "inference_s": round(inf_time, 2),
            "tokens": tokens,
            "tok_s": round(tok_s, 1),
        }
        
        log_result("bitsandbytes_nf4", "SUCCESS", result)
        del model, tokenizer
        clear_gpu()
        return result
        
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()
        log_result("bitsandbytes_nf4", "FAILED", {"error": str(e)})
        return None


def test_zse_format():
    """Test ZSE .zse format with proper INT4 quantization."""
    print("\n" + "="*60)
    print("TEST 2: ZSE .zse Format (INT4)")
    print("="*60)
    
    clear_gpu()
    
    try:
        import torch
        
        # Make sure ZSE modules are importable
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        zse_path = Path(ZSE_FILE)
        
        # Step 1: Convert if doesn't exist
        if not zse_path.exists():
            print("Converting model to .zse format...")
            print("(One-time cost, not counted in cold start)")
            
            from zse.format.writer_v2 import ZSEWriterV2, ConversionConfig
            
            config = ConversionConfig(
                quantization="int4",
                group_size=128,
            )
            writer = ZSEWriterV2(ZSE_FILE, config)
            writer.convert_from_hf(MODEL_ID)
            
            print(f"Created {ZSE_FILE}: {zse_path.stat().st_size / 1e9:.2f} GB")
            clear_gpu()
        
        # Step 2: Cold start test
        vram_before = get_gpu_memory()
        print(f"\nVRAM before: {vram_before:.2f} GB")
        print(f"Loading from .zse ({zse_path.stat().st_size / 1e9:.2f} GB)...")
        
        start = time.perf_counter()
        
        from zse.format.reader_v2 import load_zse_model
        
        model, tokenizer, info = load_zse_model(ZSE_FILE, device="cuda")
        
        load_time = time.perf_counter() - start
        vram_after = get_gpu_memory()
        
        print(f"Cold start: {load_time:.2f}s")
        print(f"VRAM used: {vram_after - vram_before:.2f} GB")
        print(f"File size: {info['size_gb']:.2f} GB")
        
        # Inference test
        print("Running inference...")
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to("cuda")
        
        inf_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        inf_time = time.perf_counter() - inf_start
        
        tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tok_s = tokens / inf_time
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated {tokens} tokens in {inf_time:.2f}s ({tok_s:.1f} tok/s)")
        print(f"Output: {output_text[:150]}...")
        
        result = {
            "cold_start_s": round(load_time, 2),
            "vram_gb": round(vram_after - vram_before, 2),
            "file_size_gb": round(info['size_gb'], 2),
            "inference_s": round(inf_time, 2),
            "tokens": tokens,
            "tok_s": round(tok_s, 1),
        }
        
        log_result("zse_int4", "SUCCESS", result)
        del model, tokenizer
        clear_gpu()
        return result
        
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()
        log_result("zse_int4", "FAILED", {"error": str(e)})
        return None


def test_llamacpp():
    """Test llama.cpp GGUF."""
    print("\n" + "="*60)
    print("TEST 3: llama.cpp GGUF (Q4_K_M)")
    print("="*60)
    
    clear_gpu()
    vram_before = get_gpu_memory()
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download GGUF
        gguf_path = Path(GGUF_FILE)
        if not gguf_path.exists():
            print(f"Downloading {GGUF_FILE}...")
            hf_hub_download(
                repo_id=GGUF_REPO,
                filename=GGUF_FILE,
                local_dir=".",
            )
        
        file_size = gguf_path.stat().st_size / 1e9
        print(f"GGUF file: {file_size:.2f} GB")
        
        # Import or install llama-cpp
        try:
            from llama_cpp import Llama
        except ImportError:
            print("Installing llama-cpp-python...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "llama-cpp-python",
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"
            ], check=True)
            from llama_cpp import Llama
        
        print(f"\nVRAM before: {vram_before:.2f} GB")
        print("Loading GGUF...")
        
        start = time.perf_counter()
        
        llm = Llama(
            model_path=str(gguf_path),
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        
        load_time = time.perf_counter() - start
        vram_after = get_gpu_memory()
        
        print(f"Cold start: {load_time:.2f}s")
        print(f"VRAM used: {vram_after - vram_before:.2f} GB")
        
        # Inference test
        print("Running inference...")
        
        inf_start = time.perf_counter()
        output = llm(
            TEST_PROMPT,
            max_tokens=100,
            temperature=0,
            echo=False,
        )
        inf_time = time.perf_counter() - inf_start
        
        tokens = output["usage"]["completion_tokens"]
        tok_s = tokens / inf_time
        output_text = output["choices"][0]["text"]
        
        print(f"Generated {tokens} tokens in {inf_time:.2f}s ({tok_s:.1f} tok/s)")
        print(f"Output: {output_text[:150]}...")
        
        result = {
            "cold_start_s": round(load_time, 2),
            "vram_gb": round(vram_after - vram_before, 2),
            "file_size_gb": round(file_size, 2),
            "inference_s": round(inf_time, 2),
            "tokens": tokens,
            "tok_s": round(tok_s, 1),
        }
        
        log_result("llamacpp_gguf", "SUCCESS", result)
        del llm
        clear_gpu()
        return result
        
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()
        log_result("llamacpp_gguf", "FAILED", {"error": str(e)})
        return None


def print_summary(results):
    """Print comparison."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY - Qwen2.5-14B-Instruct on H200")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total GPU: {get_gpu_total():.0f} GB")
    print()
    
    print(f"{'Method':<20} {'Cold Start':>12} {'VRAM':>10} {'File Size':>12} {'Speed':>12}")
    print("-" * 70)
    
    baseline = None
    
    for name, data in results.items():
        if data is None:
            print(f"{name:<20} {'FAILED':>12}")
            continue
        
        cold = data.get("cold_start_s", -1)
        vram = data.get("vram_gb", -1)
        fsize = data.get("file_size_gb", "-")
        speed = data.get("tok_s", -1)
        
        if name == "bitsandbytes_nf4":
            baseline = cold
        
        speedup = ""
        if baseline and cold > 0 and name != "bitsandbytes_nf4":
            ratio = baseline / cold
            if ratio > 1:
                speedup = f" ({ratio:.1f}× faster)"
            else:
                speedup = f" ({1/ratio:.1f}× slower)"
        
        fsize_str = f"{fsize:.1f} GB" if isinstance(fsize, float) else str(fsize)
        
        print(f"{name:<20} {cold:>10.1f}s{speedup:<15} {vram:>8.1f} GB {fsize_str:>10} {speed:>10.1f} tok/s")
    
    print("-" * 70)
    print(f"\nResults saved to: {RESULTS_FILE}")
    print("\nNOTE: These are REAL measurements. No fake data.")


def main():
    """Run all benchmarks."""
    print("="*70)
    print("ZSE REAL BENCHMARK v2 - Qwen2.5-14B-Instruct")
    print("="*70)
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {get_gpu_total():.0f} GB")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis tests REAL ZSE INT4 quantization.")
    print("No fake data. Results logged for proof.\n")
    
    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)
    
    results = {}
    
    print("\n[1/3] bitsandbytes NF4 (baseline)...")
    results["bitsandbytes_nf4"] = test_bitsandbytes()
    
    print("\n[2/3] ZSE .zse INT4...")
    results["zse_int4"] = test_zse_format()
    
    print("\n[3/3] llama.cpp GGUF...")
    results["llamacpp_gguf"] = test_llamacpp()
    
    print_summary(results)
    
    log_result("benchmark_complete", "DONE", {
        "model": MODEL_ID,
        "gpu_gb": get_gpu_total(),
        "summary": {k: v for k, v in results.items() if v}
    })


if __name__ == "__main__":
    main()
