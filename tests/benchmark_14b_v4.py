#!/usr/bin/env python3
"""
ZSE REAL BENCHMARK v4 - Qwen2.5-14B-Instruct
=============================================
PROPER benchmark using:
1. bitsandbytes NF4 (baseline)
2. ZSE FP16 with safetensors direct GPU load (device='cuda')
3. llama.cpp GGUF Q4_K_M

Key insight: Use safetensors' device="cuda" for zero-copy GPU loading.
HuggingFace's from_pretrained with low_cpu_mem_usage=True uses this internally.
"""

import os
import sys
import time
import json
import gc
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Config
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
ZSE_DIR = Path("zse_qwen14b_fp16")
GGUF_REPO = "bartowski/Qwen2.5-14B-Instruct-GGUF"
GGUF_FILE = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
TEST_PROMPT = "Write a Python function to calculate fibonacci numbers efficiently."
RESULTS_FILE = Path("benchmark_14b_v4_results.json")
GROUP_SIZE = 128


def get_gpu_memory():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip().split('\n')[0]) / 1024
    except:
        return -1


def get_gpu_total():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return int(result.stdout.strip().split('\n')[0]) / 1024
    except:
        return -1


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)


def log_result(test_name, status, data):
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


# ============================================================================
# INT4 QUANTIZATION (for ZSE format)
# ============================================================================

def quantize_int4(tensor: torch.Tensor, group_size: int = 128):
    """Quantize FP16 to INT4 (packed into uint8)."""
    out_features, in_features = tensor.shape
    
    # Pad to group_size
    padded_in = ((in_features + group_size - 1) // group_size) * group_size
    if padded_in != in_features:
        tensor = F.pad(tensor, (0, padded_in - in_features))
    
    # Reshape for group quantization
    num_groups = padded_in // group_size
    tensor_grouped = tensor.view(out_features, num_groups, group_size)
    
    # Compute scales
    group_max = tensor_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scales = group_max / 7.0
    
    # Quantize to [-7, 7]
    quantized = torch.round(tensor_grouped / scales).clamp(-7, 7).to(torch.int8)
    quantized = quantized.view(out_features, padded_in)
    
    # Pack pairs into uint8
    shifted = (quantized + 8).to(torch.uint8)
    packed = (shifted[:, 0::2] & 0x0F) | ((shifted[:, 1::2] & 0x0F) << 4)
    
    return packed, scales.squeeze(-1).half()


def dequantize_int4_cuda(packed: torch.Tensor, scales: torch.Tensor, 
                          group_size: int, orig_in: int = None):
    """Dequantize INT4 on GPU - FAST."""
    out_features = packed.shape[0]
    packed_in = packed.shape[1]
    in_features = packed_in * 2
    
    # Unpack on GPU
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    
    unpacked = torch.zeros(out_features, in_features, dtype=torch.float16, device=packed.device)
    unpacked[:, 0::2] = low.half()
    unpacked[:, 1::2] = high.half()
    
    # Dequantize with scales
    num_groups = in_features // group_size
    unpacked_grouped = unpacked.view(out_features, num_groups, group_size)
    result = (unpacked_grouped * scales.unsqueeze(-1)).view(out_features, in_features)
    
    # Trim padding
    if orig_in and orig_in < in_features:
        result = result[:, :orig_in]
    
    return result


# ============================================================================
# QUANTIZED LINEAR LAYER (matches ZSE QuantizedLinear)
# ============================================================================

class QuantizedLinearINT4(nn.Module):
    """INT4 quantized linear - keeps weights in INT4, dequants on forward."""
    
    def __init__(self, in_features, out_features, bias=True, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Packed INT4 weights
        padded_in = ((in_features + group_size - 1) // group_size) * group_size
        self.register_buffer('weight_packed', torch.zeros(out_features, padded_in // 2, dtype=torch.uint8))
        
        # Scales
        num_groups = padded_in // group_size
        self.register_buffer('scales', torch.zeros(out_features, num_groups, dtype=torch.float16))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
    
    def forward(self, x):
        # Dequantize on-the-fly (GPU)
        weight = dequantize_int4_cuda(
            self.weight_packed, self.scales, 
            self.group_size, self.in_features
        )
        return F.linear(x, weight, self.bias)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128):
        q = cls(linear.in_features, linear.out_features, 
                linear.bias is not None, group_size)
        
        packed, scales = quantize_int4(linear.weight.data.float(), group_size)
        q.weight_packed.copy_(packed)
        q.scales.copy_(scales)
        
        if linear.bias is not None:
            q.bias.copy_(linear.bias.data.half())
        
        return q


# ============================================================================
# TEST 1: BITSANDBYTES NF4
# ============================================================================

def test_bitsandbytes():
    print("\n" + "="*60)
    print("TEST 1: bitsandbytes NF4 (BASELINE)")
    print("="*60)
    
    clear_gpu()
    vram_before = get_gpu_memory()
    
    try:
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
        
        # Inference
        print("Running inference...")
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to(model.device)
        
        inf_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        inf_time = time.perf_counter() - inf_start
        
        tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tok_s = tokens / inf_time
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Generated {tokens} tokens in {inf_time:.2f}s ({tok_s:.1f} tok/s)")
        print(f"Output: {output_text[:150]}...")
        
        result = {
            "cold_start_s": round(load_time, 2),
            "vram_gb": round(vram_after - vram_before, 2),
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


# ============================================================================
# TEST 2: ZSE FP16 (Pre-saved with direct GPU load)
# ============================================================================

def convert_to_zse_fp16():
    """Convert to ZSE FP16 safetensors format (fast loading)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Converting to ZSE FP16 safetensors format...")
    print("(One-time cost)")
    
    ZSE_DIR.mkdir(exist_ok=True)
    
    # Load FP16 and save with safetensors
    print("  Loading and saving model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.save_pretrained(ZSE_DIR, safe_serialization=True)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.save_pretrained(ZSE_DIR)
    
    total_size = sum(f.stat().st_size for f in ZSE_DIR.glob("*") if f.is_file())
    print(f"  Done! Size: {total_size / 1e9:.2f} GB")
    
    del model
    clear_gpu()
    return total_size / 1e9


def test_zse_fp16():
    """
    Test ZSE FP16: Pre-saved safetensors with HuggingFace's optimized GPU loader.
    
    This uses safetensors' zero-copy GPU loading under the hood.
    """
    print("\n" + "="*60)
    print("TEST 2: ZSE FP16 (Safetensors Direct GPU)")
    print("="*60)
    
    clear_gpu()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Convert if needed
        if not ZSE_DIR.exists():
            convert_to_zse_fp16()
            clear_gpu()
        
        total_size = sum(f.stat().st_size for f in ZSE_DIR.glob("*") if f.is_file())
        file_size = total_size / 1e9
        
        vram_before = get_gpu_memory()
        print(f"\nVRAM before: {vram_before:.2f} GB")
        print(f"Loading from {ZSE_DIR} ({file_size:.2f} GB)...")
        
        start = time.perf_counter()
        
        # HuggingFace's optimized loader uses safetensors direct GPU loading
        tokenizer = AutoTokenizer.from_pretrained(ZSE_DIR, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            ZSE_DIR,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Use accelerate's efficient loading
        )
        
        load_time = time.perf_counter() - start
        vram_after = get_gpu_memory()
        
        print(f"Cold start: {load_time:.2f}s")
        print(f"VRAM used: {vram_after - vram_before:.2f} GB")
        
        # Inference
        print("Running inference...")
        inputs = tokenizer(TEST_PROMPT, return_tensors="pt").to("cuda")
        
        inf_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        inf_time = time.perf_counter() - inf_start
        
        tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tok_s = tokens / inf_time
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Generated {tokens} tokens in {inf_time:.2f}s ({tok_s:.1f} tok/s)")
        print(f"Output: {output_text[:150]}...")
        
        result = {
            "cold_start_s": round(load_time, 2),
            "vram_gb": round(vram_after - vram_before, 2),
            "file_size_gb": round(file_size, 2),
            "tok_s": round(tok_s, 1),
        }
        
        log_result("zse_fp16", "SUCCESS", result)
        del model
        clear_gpu()
        return result
        
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()
        log_result("zse_fp16", "FAILED", {"error": str(e)})
        return None


# ============================================================================
# TEST 3: LLAMA.CPP GGUF
# ============================================================================

def test_llamacpp():
    print("\n" + "="*60)
    print("TEST 3: llama.cpp GGUF (Q4_K_M)")
    print("="*60)
    
    clear_gpu()
    vram_before = get_gpu_memory()
    
    try:
        from huggingface_hub import hf_hub_download
        
        gguf_path = Path(GGUF_FILE)
        if not gguf_path.exists():
            print(f"Downloading {GGUF_FILE}...")
            hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE, local_dir=".")
        
        file_size = gguf_path.stat().st_size / 1e9
        print(f"GGUF file: {file_size:.2f} GB")
        
        try:
            from llama_cpp import Llama
        except ImportError:
            print("Installing llama-cpp-python...")
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python",
                          "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"],
                         check=True)
            from llama_cpp import Llama
        
        print(f"\nVRAM before: {vram_before:.2f} GB")
        print("Loading GGUF...")
        
        start = time.perf_counter()
        llm = Llama(model_path=str(gguf_path), n_ctx=4096, n_gpu_layers=-1, verbose=False)
        load_time = time.perf_counter() - start
        
        vram_after = get_gpu_memory()
        print(f"Cold start: {load_time:.2f}s")
        print(f"VRAM used: {vram_after - vram_before:.2f} GB")
        
        # Inference
        print("Running inference...")
        inf_start = time.perf_counter()
        output = llm(TEST_PROMPT, max_tokens=100, temperature=0, echo=False)
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("ZSE REAL BENCHMARK v4 - Qwen2.5-14B-Instruct")
    print("="*60)
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {get_gpu_total():.0f} GB")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("This tests ZSE with:")
    print("- safetensors direct GPU loading (device='cuda')")
    print("- HuggingFace optimized loader with low_cpu_mem_usage=True")
    print()
    
    with open(RESULTS_FILE, "w") as f:
        json.dump([], f)
    
    results = {}
    
    print("\n[1/3] bitsandbytes NF4...")
    results["bitsandbytes_nf4"] = test_bitsandbytes()
    
    print("\n[2/3] ZSE FP16 (safetensors direct GPU)...")
    results["zse_fp16"] = test_zse_fp16()
    
    print("\n[3/3] llama.cpp GGUF...")
    results["llamacpp_gguf"] = test_llamacpp()
    
    # Summary
    print("\n" + "="*60)
    print(f"BENCHMARK SUMMARY - {MODEL_ID}")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {get_gpu_total():.0f} GB")
    print()
    
    baseline = results.get("bitsandbytes_nf4", {}).get("cold_start_s", 0)
    
    print(f"{'Method':<20} {'Cold Start':>12} {'VRAM':>10} {'File':>10} {'Speed':>12}")
    print("-" * 70)
    
    for name, data in results.items():
        if data is None:
            print(f"{name:<20} {'FAILED':>12}")
            continue
        
        cold = data.get("cold_start_s", -1)
        vram = data.get("vram_gb", -1)
        fsize = data.get("file_size_gb", 0)
        tok_s = data.get("tok_s", -1)
        
        speedup = ""
        if baseline and cold > 0 and name != "bitsandbytes_nf4":
            ratio = baseline / cold
            if ratio > 1:
                speedup = f" ({ratio:.1f}× faster)"
            else:
                speedup = f" ({1/ratio:.1f}× slower)"
        
        fsize_str = f"{fsize:.1f} GB" if fsize else "-"
        print(f"{name:<20} {cold:>8.1f}s{speedup:<12} {vram:>8.1f} GB {fsize_str:>10} {tok_s:>8.1f} tok/s")
    
    print("-" * 70)
    print(f"\nResults saved to: {RESULTS_FILE}")
    log_result("benchmark_complete", "DONE", {k: v for k, v in results.items() if v})


if __name__ == "__main__":
    main()
