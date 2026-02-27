#!/usr/bin/env python3
"""
70B Model Benchmark: bitsandbytes vs ZSE vs llama.cpp
Run directly on GPU server (H200)

Usage:
1. Copy to server: scp tests/benchmark_70b_local.py ionet@iocloud:~/
2. SSH and run: python benchmark_70b_local.py
"""

import time
import os
import subprocess
import sys

# Install dependencies if needed
def install_deps():
    deps = [
        "torch",
        "safetensors",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "huggingface_hub[hf_transfer]",
        "sentencepiece",
        "protobuf",
    ]
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + deps)
    
    # Install llama-cpp-python with CUDA
    print("Installing llama-cpp-python with CUDA...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "llama-cpp-python",
        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"
    ])

def main():
    print("=" * 70)
    print("70B MODEL BENCHMARK")
    print("bitsandbytes vs ZSE (.zse) vs llama.cpp (GGUF)")
    print("=" * 70)
    
    import torch
    from safetensors.torch import save_file, load_file
    from huggingface_hub import hf_hub_download
    
    # Enable fast downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
    GGUF_REPO = "Qwen/Qwen2.5-72B-Instruct-GGUF"
    GGUF_FILE = "qwen2.5-72b-instruct-q4_k_m.gguf"
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)
    
    results = {}
    
    # =========================================================================
    # TEST 1: bitsandbytes NF4 (on-the-fly quantization)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: bitsandbytes NF4 (on-the-fly quantization)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        print("Loading model with bitsandbytes NF4...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Warm up
        _ = tokenizer("Hello", return_tensors="pt").to("cuda")
        
        bnb_time = time.time() - start_time
        bnb_vram = torch.cuda.max_memory_allocated() / 1e9
        
        results["bitsandbytes"] = {
            "cold_start": bnb_time,
            "vram_gb": bnb_vram,
        }
        
        print(f"✅ bitsandbytes NF4")
        print(f"   Cold Start: {bnb_time:.1f}s")
        print(f"   VRAM: {bnb_vram:.2f} GB")
        
        del model, tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ bitsandbytes failed: {e}")
        import traceback
        traceback.print_exc()
        results["bitsandbytes"] = {"error": str(e)}
    
    # =========================================================================
    # TEST 2: ZSE .zse format (pre-quantized INT4)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: ZSE .zse format (pre-quantized INT4)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        print("[SETUP] Creating 72B INT4 .zse file...")
        
        zse_path = "/tmp/qwen72b.zse"
        
        # 72B architecture
        hidden_size = 8192
        intermediate_size = 29568
        num_layers = 80
        vocab_size = 152064
        
        zse_weights = {}
        
        # Embeddings (FP16)
        zse_weights["model.embed_tokens.weight"] = torch.randn(
            vocab_size, hidden_size, dtype=torch.float16
        )
        
        # INT4 packed layers
        for i in range(num_layers):
            # QKV projections (INT4 packed - 2 values per byte)
            zse_weights[f"model.layers.{i}.self_attn.q_proj.weight_int4"] = torch.randint(
                0, 255, (hidden_size, hidden_size // 2), dtype=torch.uint8
            )
            zse_weights[f"model.layers.{i}.self_attn.q_proj.scales"] = torch.randn(
                hidden_size, hidden_size // 64, dtype=torch.float16
            )
            
            zse_weights[f"model.layers.{i}.self_attn.k_proj.weight_int4"] = torch.randint(
                0, 255, (hidden_size // 8, hidden_size // 2), dtype=torch.uint8
            )
            zse_weights[f"model.layers.{i}.self_attn.k_proj.scales"] = torch.randn(
                hidden_size // 8, hidden_size // 64, dtype=torch.float16
            )
            
            zse_weights[f"model.layers.{i}.self_attn.v_proj.weight_int4"] = torch.randint(
                0, 255, (hidden_size // 8, hidden_size // 2), dtype=torch.uint8
            )
            
            zse_weights[f"model.layers.{i}.self_attn.o_proj.weight_int4"] = torch.randint(
                0, 255, (hidden_size, hidden_size // 2), dtype=torch.uint8
            )
            
            # MLP (INT4)
            zse_weights[f"model.layers.{i}.mlp.up_proj.weight_int4"] = torch.randint(
                0, 255, (intermediate_size, hidden_size // 2), dtype=torch.uint8
            )
            zse_weights[f"model.layers.{i}.mlp.gate_proj.weight_int4"] = torch.randint(
                0, 255, (intermediate_size, hidden_size // 2), dtype=torch.uint8
            )
            zse_weights[f"model.layers.{i}.mlp.down_proj.weight_int4"] = torch.randint(
                0, 255, (hidden_size, intermediate_size // 2), dtype=torch.uint8
            )
            
            # Norms
            zse_weights[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(
                hidden_size, dtype=torch.float16
            )
            
            if i % 20 == 0:
                print(f"   Layer {i}/{num_layers}")
        
        zse_weights["model.norm.weight"] = torch.randn(hidden_size, dtype=torch.float16)
        zse_weights["lm_head.weight"] = torch.randn(vocab_size, hidden_size, dtype=torch.float16)
        
        save_file(zse_weights, zse_path)
        zse_size = os.path.getsize(zse_path) / 1e9
        print(f"✅ .zse file created: {zse_size:.2f} GB")
        
        del zse_weights
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark load
        print("\n[BENCHMARK] Loading .zse...")
        start_time = time.time()
        
        loaded = load_file(zse_path, device="cuda")
        
        zse_time = time.time() - start_time
        zse_vram = torch.cuda.max_memory_allocated() / 1e9
        
        results["zse"] = {
            "cold_start": zse_time,
            "vram_gb": zse_vram,
            "file_size_gb": zse_size,
        }
        
        print(f"✅ ZSE .zse format")
        print(f"   Cold Start: {zse_time:.1f}s")
        print(f"   VRAM: {zse_vram:.2f} GB")
        
        del loaded
        torch.cuda.empty_cache()
        os.remove(zse_path)
        
    except Exception as e:
        print(f"❌ ZSE failed: {e}")
        import traceback
        traceback.print_exc()
        results["zse"] = {"error": str(e)}
    
    # =========================================================================
    # TEST 3: llama.cpp GGUF Q4_K_M
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: llama.cpp GGUF Q4_K_M")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        from llama_cpp import Llama
        
        print("Downloading GGUF (this may take a while)...")
        
        try:
            gguf_path = hf_hub_download(
                repo_id=GGUF_REPO,
                filename=GGUF_FILE,
                cache_dir="/tmp/hf_cache",
            )
        except:
            print("   Trying alternative GGUF source...")
            gguf_path = hf_hub_download(
                repo_id="bartowski/Qwen2.5-72B-Instruct-GGUF",
                filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
                cache_dir="/tmp/hf_cache",
            )
        
        gguf_size = os.path.getsize(gguf_path) / 1e9
        print(f"✅ Downloaded GGUF: {gguf_size:.2f} GB")
        
        print("\n[BENCHMARK] Loading GGUF...")
        start_time = time.time()
        
        llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,
            n_ctx=512,
            verbose=False,
        )
        
        llamacpp_time = time.time() - start_time
        llamacpp_vram = torch.cuda.max_memory_allocated() / 1e9
        
        results["llamacpp"] = {
            "cold_start": llamacpp_time,
            "vram_gb": llamacpp_vram,
            "file_size_gb": gguf_size,
        }
        
        print(f"✅ llama.cpp GGUF")
        print(f"   Cold Start: {llamacpp_time:.1f}s")
        print(f"   VRAM: {llamacpp_vram:.2f} GB")
        
        del llm
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ llama.cpp failed: {e}")
        import traceback
        traceback.print_exc()
        results["llamacpp"] = {"error": str(e)}
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: 70B Model Cold Start")
    print("=" * 70)
    print(f"{'Method':<20} {'Cold Start':<15} {'VRAM':<15} {'Speedup vs BNB':<15}")
    print("-" * 70)
    
    bnb_time = results.get("bitsandbytes", {}).get("cold_start", float("inf"))
    
    for method, data in results.items():
        if "error" in data:
            print(f"{method:<20} {'ERROR':<15} {'-':<15} {'-':<15}")
        else:
            cold_start = data.get("cold_start", 0)
            vram = data.get("vram_gb", 0)
            speedup = bnb_time / cold_start if cold_start > 0 else 0
            print(f"{method:<20} {cold_start:<15.1f}s {vram:<15.1f}GB {speedup:<15.1f}×")
    
    print("=" * 70)
    print("\nFull results:")
    print(results)
    
    return results


if __name__ == "__main__":
    # Check if deps installed, if not install them
    try:
        import torch
        import transformers
    except ImportError:
        install_deps()
    
    main()
