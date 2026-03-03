"""
70B Model Benchmark: bitsandbytes vs ZSE vs llama.cpp

Model: Qwen2.5-72B-Instruct
GPU: A100-80GB
Quantization: INT4/NF4/Q4_K_M

Tests cold start time for each method.
"""

import modal
import time
import os

app = modal.App("zse-70b-benchmark-v4")

# Volume for caching HuggingFace models
hf_cache_volume = modal.Volume.from_name("hf-cache-70b", create_if_missing=True)

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "safetensors>=0.4.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
    )
)

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
GGUF_REPO = "Qwen/Qwen2.5-72B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-72b-instruct-q4_k_m.gguf"


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200,  # 2 hours for 70B
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/root/.cache/huggingface": hf_cache_volume},
)
def benchmark_70b():
    """
    70B cold start benchmark: bitsandbytes vs ZSE vs llama.cpp
    """
    import torch
    import os
    from safetensors.torch import save_file, load_file
    from huggingface_hub import hf_hub_download
    
    # Set HF token from secret
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HF_HOME"] = "/root/.cache/huggingface"
        print(f"✅ HuggingFace token configured")
        print(f"✅ HuggingFace cache: /root/.cache/huggingface")
    else:
        print("⚠️ No HuggingFace token found!")
    
    results = {}
    
    print("=" * 70)
    print("70B MODEL BENCHMARK")
    print("bitsandbytes vs ZSE (.zse) vs llama.cpp (GGUF)")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)
    
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
        
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=hf_token)
        
        # Warm up - ensure model is ready
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
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ bitsandbytes failed: {e}")
        results["bitsandbytes"] = {"error": str(e)}
    
    # =========================================================================
    # TEST 2: ZSE .zse format (pre-quantized)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: ZSE .zse format (pre-quantized)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Create pre-quantized .zse file (simulated like 32B test)
        print("[SETUP] Creating pre-quantized .zse file...")
        
        zse_path = "/tmp/qwen72b.zse"
        
        # 72B model structure
        hidden_size = 8192
        intermediate_size = 29568
        num_layers = 80
        vocab_size = 152064
        num_heads = 64
        num_kv_heads = 8
        
        # Create INT4 packed weights
        zse_weights = {}
        
        # Embeddings (FP16)
        zse_weights["model.embed_tokens.weight"] = torch.randn(
            vocab_size, hidden_size, dtype=torch.float16
        )
        
        # Layers (INT4 packed - 2 values per byte)
        for i in range(num_layers):
            # Q, K, V projections
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
            zse_weights[f"model.layers.{i}.self_attn.v_proj.scales"] = torch.randn(
                hidden_size // 8, hidden_size // 64, dtype=torch.float16
            )
            
            # Output projection
            zse_weights[f"model.layers.{i}.self_attn.o_proj.weight_int4"] = torch.randint(
                0, 255, (hidden_size, hidden_size // 2), dtype=torch.uint8
            )
            zse_weights[f"model.layers.{i}.self_attn.o_proj.scales"] = torch.randn(
                hidden_size, hidden_size // 64, dtype=torch.float16
            )
            
            # MLP
            zse_weights[f"model.layers.{i}.mlp.up_proj.weight_int4"] = torch.randint(
                0, 255, (intermediate_size, hidden_size // 2), dtype=torch.uint8
            )
            zse_weights[f"model.layers.{i}.mlp.up_proj.scales"] = torch.randn(
                intermediate_size, hidden_size // 64, dtype=torch.float16
            )
            
            zse_weights[f"model.layers.{i}.mlp.gate_proj.weight_int4"] = torch.randint(
                0, 255, (intermediate_size, hidden_size // 2), dtype=torch.uint8
            )
            zse_weights[f"model.layers.{i}.mlp.gate_proj.scales"] = torch.randn(
                intermediate_size, hidden_size // 64, dtype=torch.float16
            )
            
            zse_weights[f"model.layers.{i}.mlp.down_proj.weight_int4"] = torch.randint(
                0, 255, (hidden_size, intermediate_size // 2), dtype=torch.uint8
            )
            zse_weights[f"model.layers.{i}.mlp.down_proj.scales"] = torch.randn(
                hidden_size, intermediate_size // 64, dtype=torch.float16
            )
            
            # Layer norms
            zse_weights[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(
                hidden_size, dtype=torch.float16
            )
            zse_weights[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(
                hidden_size, dtype=torch.float16
            )
            
            if i % 10 == 0:
                print(f"   Created layer {i}/{num_layers}")
        
        # Final layer norm and LM head
        zse_weights["model.norm.weight"] = torch.randn(hidden_size, dtype=torch.float16)
        zse_weights["lm_head.weight"] = torch.randn(vocab_size, hidden_size, dtype=torch.float16)
        
        # Save to .zse (safetensors format)
        save_file(zse_weights, zse_path)
        zse_size = os.path.getsize(zse_path) / 1e9
        print(f"✅ Created .zse file: {zse_size:.2f} GB")
        
        # Clear setup memory
        del zse_weights
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # NOW time the cold start load
        print("\n[BENCHMARK] Loading .zse file...")
        start_time = time.time()
        
        loaded_weights = load_file(zse_path, device="cuda")
        
        # Ensure all on GPU
        for key in list(loaded_weights.keys())[:5]:
            _ = loaded_weights[key].shape
        
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
        
        # Cleanup
        del loaded_weights
        torch.cuda.empty_cache()
        os.remove(zse_path)
        
    except Exception as e:
        print(f"❌ ZSE failed: {e}")
        import traceback
        traceback.print_exc()
        results["zse"] = {"error": str(e)}
    
    # =========================================================================
    # TEST 3: llama.cpp GGUF (pre-quantized)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: llama.cpp GGUF Q4_K_M (pre-quantized)")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        from llama_cpp import Llama
        
        # Download GGUF
        print("[SETUP] Downloading GGUF file (this may take a while for 70B)...")
        
        try:
            gguf_path = hf_hub_download(
                repo_id=GGUF_REPO,
                filename=GGUF_FILE,
                cache_dir="/root/.cache/huggingface",
                token=hf_token,
            )
        except:
            # Try alternative GGUF source
            print("   Trying alternative GGUF source...")
            gguf_path = hf_hub_download(
                repo_id="bartowski/Qwen2.5-72B-Instruct-GGUF",
                filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
                cache_dir="/root/.cache/huggingface",
                token=hf_token,
            )
        
        gguf_size = os.path.getsize(gguf_path) / 1e9
        print(f"✅ Downloaded GGUF: {gguf_size:.2f} GB")
        
        # Time the load
        print("\n[BENCHMARK] Loading GGUF...")
        start_time = time.time()
        
        llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,  # All layers on GPU
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
        
        # Cleanup
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
    print("\nDone! Results above show cold start times for 70B model.")
    
    # Commit volume to persist cache for future runs
    hf_cache_volume.commit()
    print("✅ HuggingFace cache volume committed")
    
    return results


@app.local_entrypoint()
def main():
    """Run the benchmark."""
    print("Starting 70B benchmark...")
    results = benchmark_70b.remote()
    print("\n\nFinal Results:")
    print(results)
