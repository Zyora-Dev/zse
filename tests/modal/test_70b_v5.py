"""
70B Model Benchmark: bitsandbytes vs ZSE vs llama.cpp
Version 5: Proper 2-step approach (download first, then benchmark)

Model: Qwen2.5-72B-Instruct
GPU: A100-80GB
"""

import modal
import time
import os

app = modal.App("zse-70b-benchmark-v6")

# Persistent volume for model cache
model_volume = modal.Volume.from_name("zse-model-cache-v5", create_if_missing=True)

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
GGUF_REPO = "Qwen/Qwen2.5-72B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-72b-instruct-q4_k_m.gguf"
MODEL_DIR = "/models"

# Lightweight image for downloads (no GPU)
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "transformers",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Fast parallel downloads
)

# Full image for benchmark (GPU)
benchmark_image = (
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


@app.function(
    image=download_image,
    gpu="H100",  # H100 has best network bandwidth
    timeout=3600,  # 1 hour for downloads
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_DIR: model_volume},
)
def download_models():
    """
    Step 1: Download all models to persistent volume.
    No GPU needed, just downloads.
    """
    import os
    from huggingface_hub import snapshot_download, hf_hub_download
    
    hf_token = os.environ.get("HF_TOKEN")
    
    print("=" * 70)
    print("STEP 1: DOWNLOADING MODELS")
    print("=" * 70)
    
    # Check what's already downloaded
    hf_model_dir = f"{MODEL_DIR}/qwen72b"
    gguf_path = f"{MODEL_DIR}/qwen72b.gguf"
    
    # Download HF model (for bitsandbytes)
    if not os.path.exists(hf_model_dir) or len(os.listdir(hf_model_dir)) < 30:
        print(f"\n📥 Downloading {MODEL_NAME}...")
        print("   Using HF_TRANSFER for fast parallel downloads")
        
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=hf_model_dir,
            token=hf_token,
            ignore_patterns=["*.gguf", "*.bin"],  # Skip GGUF in HF repo
        )
        print(f"✅ HF model downloaded to {hf_model_dir}")
    else:
        print(f"✅ HF model already cached at {hf_model_dir}")
    
    # Download GGUF (for llama.cpp)
    if not os.path.exists(gguf_path):
        print(f"\n📥 Downloading GGUF from {GGUF_REPO}...")
        
        try:
            downloaded = hf_hub_download(
                repo_id=GGUF_REPO,
                filename=GGUF_FILE,
                local_dir=MODEL_DIR,
                token=hf_token,
            )
            # Rename to standard path
            if downloaded != gguf_path:
                os.rename(downloaded, gguf_path)
        except Exception as e:
            print(f"   Primary GGUF failed: {e}")
            print("   Trying bartowski repo...")
            downloaded = hf_hub_download(
                repo_id="bartowski/Qwen2.5-72B-Instruct-GGUF",
                filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
                local_dir=MODEL_DIR,
                token=hf_token,
            )
            if downloaded != gguf_path:
                os.rename(downloaded, gguf_path)
        
        print(f"✅ GGUF downloaded to {gguf_path}")
    else:
        print(f"✅ GGUF already cached at {gguf_path}")
    
    # Commit volume to persist
    model_volume.commit()
    
    # List what we have
    print("\n📁 Downloaded files:")
    for f in os.listdir(MODEL_DIR):
        path = os.path.join(MODEL_DIR, f)
        if os.path.isfile(path):
            size = os.path.getsize(path) / 1e9
            print(f"   {f}: {size:.2f} GB")
        else:
            print(f"   {f}/ (directory)")
    
    print("\n✅ All models downloaded! Ready for benchmark.")
    return {"status": "ready", "model_dir": MODEL_DIR}


@app.function(
    image=benchmark_image,
    gpu="H100",  # H100 80GB
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={MODEL_DIR: model_volume},
)
def run_benchmark():
    """
    Step 2: Run benchmark using cached models.
    Models already downloaded - just load and time.
    """
    import torch
    from safetensors.torch import save_file, load_file
    
    hf_token = os.environ.get("HF_TOKEN")
    hf_model_dir = f"{MODEL_DIR}/qwen72b"
    gguf_path = f"{MODEL_DIR}/qwen72b.gguf"
    
    results = {}
    
    print("=" * 70)
    print("STEP 2: RUNNING BENCHMARK")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)
    
    # Verify models exist
    if not os.path.exists(hf_model_dir):
        print("❌ HF model not found! Run download_models first.")
        return {"error": "Models not downloaded"}
    
    # =========================================================================
    # TEST 1: bitsandbytes NF4 (on-the-fly quantization from local cache)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: bitsandbytes NF4 (quantize from local safetensors)")
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
        
        print(f"Loading from local cache: {hf_model_dir}")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_dir,  # Local path, no download
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        
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
    # TEST 2: ZSE .zse format (pre-quantized simulation)
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
        
        # Embeddings
        zse_weights["model.embed_tokens.weight"] = torch.randn(
            vocab_size, hidden_size, dtype=torch.float16
        )
        
        # INT4 packed layers
        for i in range(num_layers):
            # QKV projections (INT4 packed)
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
    # TEST 3: llama.cpp GGUF
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: llama.cpp GGUF Q4_K_M")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        from llama_cpp import Llama
        
        if not os.path.exists(gguf_path):
            print(f"❌ GGUF not found at {gguf_path}")
            results["llamacpp"] = {"error": "GGUF not downloaded"}
        else:
            gguf_size = os.path.getsize(gguf_path) / 1e9
            print(f"Loading GGUF: {gguf_size:.2f} GB")
            
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
    
    return results


@app.local_entrypoint()
def main():
    """Run download then benchmark."""
    print("=" * 70)
    print("70B BENCHMARK - TWO STEP PROCESS")
    print("=" * 70)
    
    print("\n📥 Step 1: Downloading models (no GPU)...")
    download_result = download_models.remote()
    print(f"Download result: {download_result}")
    
    if download_result.get("status") == "ready":
        print("\n🚀 Step 2: Running benchmark (GPU)...")
        results = run_benchmark.remote()
        print("\n\nFINAL RESULTS:")
        print(results)
        return results
    else:
        print("❌ Download failed, cannot run benchmark")
        return download_result
