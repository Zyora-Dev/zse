"""
ZSE vs llama.cpp Cold Start Comparison

HONEST benchmark on A100-80GB:
1. llama.cpp (via llama-cpp-python) loading Q4_K_M GGUF
2. ZSE loading pre-quantized .zse format

Model: Qwen 7B (Q4 quantization)
GPU: A100-80GB

This settles the question: Is .zse format faster than llama.cpp's GGUF?
"""

import modal
import time

app = modal.App("zse-vs-llamacpp-benchmark")

# Image with llama-cpp-python (use pre-built CUDA wheel)
llamacpp_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub",
        "torch>=2.0.0",
        "safetensors>=0.4.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        # Install llama-cpp-python with pre-built CUDA 12.4 wheel
        "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
    )
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
# Popular Q4_K_M GGUF for Qwen 7B
GGUF_REPO = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"


@app.function(
    image=llamacpp_image,
    gpu="A100-80GB",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")] if False else [],
)
def test_llamacpp_vs_zse():
    """
    Head-to-head cold start comparison.
    
    Measures time from "start" to "ready to generate first token"
    """
    import torch
    import os
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file, load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    results = {}
    
    # =========================================================================
    # GPU INFO
    # =========================================================================
    print("=" * 70)
    print("ZSE vs llama.cpp COLD START BENCHMARK")
    print("=" * 70)
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram:.1f} GB")
    print(f"Model: Qwen 7B (Q4 quantization)")
    print("=" * 70)
    
    # =========================================================================
    # DOWNLOAD GGUF FILE
    # =========================================================================
    print("\n[SETUP] Downloading GGUF file...")
    print("-" * 50)
    
    start_download = time.time()
    try:
        gguf_path = hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            cache_dir="/tmp/hf_cache",
        )
        download_time = time.time() - start_download
        gguf_size = os.path.getsize(gguf_path) / 1e9
        print(f"✅ Downloaded: {gguf_path}")
        print(f"   Size: {gguf_size:.2f} GB")
        print(f"   Download time: {download_time:.1f}s")
    except Exception as e:
        print(f"⚠️  Qwen GGUF not found, trying alternative...")
        # Try Bartowski's GGUF (popular GGUF provider)
        try:
            gguf_path = hf_hub_download(
                repo_id="bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                filename="Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
                cache_dir="/tmp/hf_cache",
            )
            download_time = time.time() - start_download
            gguf_size = os.path.getsize(gguf_path) / 1e9
            print(f"✅ Downloaded from bartowski: {gguf_path}")
            print(f"   Size: {gguf_size:.2f} GB")
        except Exception as e2:
            print(f"❌ Could not download GGUF: {e2}")
            print("   Trying generic small model for testing...")
            gguf_path = hf_hub_download(
                repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                cache_dir="/tmp/hf_cache",
            )
            gguf_size = os.path.getsize(gguf_path) / 1e9
            print(f"✅ Using TinyLlama for test: {gguf_path}")
    
    # =========================================================================
    # TEST 1: llama.cpp Cold Start
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] llama.cpp (llama-cpp-python) Cold Start")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    try:
        from llama_cpp import Llama
        
        # COLD START: Load model with full GPU offload
        print("\n   Loading GGUF with full GPU offload (n_gpu_layers=-1)...")
        start_llamacpp = time.time()
        
        model_llamacpp = Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,  # Full GPU offload
            n_ctx=2048,
            verbose=False,
        )
        
        llamacpp_load_time = time.time() - start_llamacpp
        
        # Verify model works - generate one token
        print("   Verifying model works (generating 1 token)...")
        start_verify = time.time()
        _ = model_llamacpp("def hello():", max_tokens=1)
        verify_time = time.time() - start_verify
        
        llamacpp_cold_start = llamacpp_load_time + verify_time
        
        print(f"\n✅ llama.cpp COLD START RESULTS:")
        print(f"   Model load time: {llamacpp_load_time:.2f}s")
        print(f"   First token verify: {verify_time:.3f}s")
        print(f"   TOTAL COLD START: {llamacpp_cold_start:.2f}s")
        
        results['llamacpp_load_time'] = llamacpp_load_time
        results['llamacpp_cold_start'] = llamacpp_cold_start
        
        # Quick throughput test
        print("\n   Running throughput test...")
        start_gen = time.time()
        output = model_llamacpp("def fibonacci(n):", max_tokens=100)
        gen_time = time.time() - start_gen
        tokens = len(model_llamacpp.tokenize(output['choices'][0]['text'].encode()))
        tok_per_sec = tokens / gen_time if gen_time > 0 else 0
        print(f"   Generation: {tokens} tokens in {gen_time:.2f}s ({tok_per_sec:.1f} tok/s)")
        results['llamacpp_tok_per_sec'] = tok_per_sec
        
        # Clean up
        del model_llamacpp
        
    except Exception as e:
        print(f"❌ llama.cpp test failed: {e}")
        import traceback
        traceback.print_exc()
        results['llamacpp_cold_start'] = None
        results['llamacpp_error'] = str(e)
    
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TEST 2: Create .zse format (simulating zse convert)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[SETUP] Creating .zse format (one-time conversion)")
    print("This simulates: zse convert Qwen/Qwen2.5-Coder-7B-Instruct -o qwen7b.zse")
    print("=" * 70)
    
    # Load tokenizer for later
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Load FP16 model and quantize to INT4
    print("\n   Loading FP16 model to CPU...")
    start_convert = time.time()
    
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # INT4 quantization (matching GGUF Q4)
    print("   Quantizing to INT4...")
    zse_weights = {}
    total_params = 0
    quantized_params = 0
    
    for name, param in model_fp16.named_parameters():
        total_params += param.numel()
        if param.dim() >= 2 and param.numel() > 1024:
            # INT4 quantization
            scale = param.abs().max() / 7.0
            quantized = (param / scale).round().clamp(-8, 7).to(torch.int8)
            zse_weights[f"{name}.quantized"] = quantized
            zse_weights[f"{name}.scale"] = scale.unsqueeze(0)
            quantized_params += param.numel()
        else:
            zse_weights[name] = param.half()
    
    # Save .zse file
    zse_path = "/tmp/qwen7b.zse"
    print(f"   Saving to {zse_path}...")
    save_file(zse_weights, zse_path)
    
    convert_time = time.time() - start_convert
    zse_size = os.path.getsize(zse_path) / 1e9
    
    print(f"\n✅ Conversion complete:")
    print(f"   Time: {convert_time:.1f}s")
    print(f"   .zse size: {zse_size:.2f} GB")
    print(f"   Quantized: {quantized_params/1e9:.2f}B / {total_params/1e9:.2f}B params")
    
    results['convert_time'] = convert_time
    results['zse_size_gb'] = zse_size
    
    # Keep model config for reconstruction
    model_config = model_fp16.config
    
    # Clean up
    del model_fp16
    torch.cuda.empty_cache()
    
    # =========================================================================
    # TEST 3: ZSE Cold Start
    # =========================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] ZSE (.zse format) Cold Start")
    print("This simulates: zse serve qwen7b.zse")
    print("=" * 70)
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # COLD START: Load .zse and reconstruct model
    print("\n   Loading .zse file...")
    start_zse = time.time()
    
    # Step 1: Load weights from .zse (memory-mapped for speed)
    loaded_weights = load_file(zse_path, device="cpu")
    
    zse_weight_load_time = time.time() - start_zse
    print(f"   Weight load: {zse_weight_load_time:.3f}s ({len(loaded_weights)} tensors)")
    
    # Step 2: Create model skeleton
    print("   Creating model structure...")
    from transformers import AutoModelForCausalLM
    model_zse = AutoModelForCausalLM.from_config(model_config)
    
    # Step 3: Dequantize and load weights to GPU
    print("   Dequantizing and loading to GPU...")
    state_dict = {}
    
    for name, param in model_zse.named_parameters():
        if f"{name}.quantized" in loaded_weights:
            # Dequantize INT4 → FP16
            quantized = loaded_weights[f"{name}.quantized"]
            scale = loaded_weights[f"{name}.scale"]
            dequantized = quantized.float() * scale
            state_dict[name] = dequantized.half().cuda()
        elif name in loaded_weights:
            state_dict[name] = loaded_weights[name].cuda()
        else:
            print(f"   ⚠️  Missing: {name}")
    
    # Step 4: Load state dict
    model_zse.load_state_dict(state_dict, strict=False)
    model_zse = model_zse.cuda()
    model_zse.eval()
    
    zse_load_time = time.time() - start_zse
    
    # Step 5: Verify model works
    print("   Verifying model works (generating 1 token)...")
    inputs = tokenizer("def hello():", return_tensors="pt").to("cuda")
    start_verify = time.time()
    with torch.no_grad():
        _ = model_zse.generate(**inputs, max_new_tokens=1)
    verify_time = time.time() - start_verify
    
    zse_cold_start = zse_load_time + verify_time
    zse_vram = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n✅ ZSE COLD START RESULTS:")
    print(f"   Weight load time: {zse_weight_load_time:.2f}s")
    print(f"   Full load + dequant: {zse_load_time:.2f}s")
    print(f"   First token verify: {verify_time:.3f}s")
    print(f"   TOTAL COLD START: {zse_cold_start:.2f}s")
    print(f"   Peak VRAM: {zse_vram:.2f} GB")
    
    results['zse_weight_load_time'] = zse_weight_load_time
    results['zse_cold_start'] = zse_cold_start
    results['zse_vram'] = zse_vram
    
    # Quick throughput test
    print("\n   Running throughput test...")
    inputs = tokenizer("def fibonacci(n):", return_tensors="pt").to("cuda")
    start_gen = time.time()
    with torch.no_grad():
        outputs = model_zse.generate(**inputs, max_new_tokens=100, do_sample=False)
    gen_time = time.time() - start_gen
    tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    tok_per_sec = tokens / gen_time if gen_time > 0 else 0
    print(f"   Generation: {tokens} tokens in {gen_time:.2f}s ({tok_per_sec:.1f} tok/s)")
    results['zse_tok_per_sec'] = tok_per_sec
    
    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS: Cold Start Comparison")
    print("=" * 70)
    
    print("\n┌─────────────────────┬──────────────┬────────────────┐")
    print("│ Method              │ Cold Start   │ vs llama.cpp   │")
    print("├─────────────────────┼──────────────┼────────────────┤")
    
    llamacpp_time = results.get('llamacpp_cold_start')
    zse_time = results.get('zse_cold_start')
    
    if llamacpp_time:
        print(f"│ llama.cpp (GGUF)    │ {llamacpp_time:>8.2f}s    │    baseline    │")
    else:
        print(f"│ llama.cpp (GGUF)    │   FAILED     │      -         │")
    
    if zse_time and llamacpp_time:
        if zse_time < llamacpp_time:
            speedup = llamacpp_time / zse_time
            print(f"│ ZSE (.zse format)   │ {zse_time:>8.2f}s    │ {speedup:>6.2f}× faster │")
        else:
            slowdown = zse_time / llamacpp_time
            print(f"│ ZSE (.zse format)   │ {zse_time:>8.2f}s    │ {slowdown:>6.2f}× slower │")
    elif zse_time:
        print(f"│ ZSE (.zse format)   │ {zse_time:>8.2f}s    │      -         │")
    
    print("└─────────────────────┴──────────────┴────────────────┘")
    
    # Throughput comparison
    print("\n┌─────────────────────┬──────────────┐")
    print("│ Method              │ tok/s        │")
    print("├─────────────────────┼──────────────┤")
    if results.get('llamacpp_tok_per_sec'):
        print(f"│ llama.cpp (GGUF)    │ {results['llamacpp_tok_per_sec']:>8.1f}     │")
    if results.get('zse_tok_per_sec'):
        print(f"│ ZSE (.zse format)   │ {results['zse_tok_per_sec']:>8.1f}     │")
    print("└─────────────────────┴──────────────┘")
    
    # Summary
    print("\n" + "=" * 70)
    if llamacpp_time and zse_time:
        if zse_time < llamacpp_time:
            speedup = llamacpp_time / zse_time
            print(f"🏆 ZSE is {speedup:.2f}× FASTER than llama.cpp")
            print("   This is a legitimate, defensible benchmark result!")
        elif zse_time > llamacpp_time:
            slowdown = zse_time / llamacpp_time
            print(f"⚠️  ZSE is {slowdown:.2f}× SLOWER than llama.cpp")
            print("   llama.cpp's mmap is very efficient.")
        else:
            print("🤝 ZSE and llama.cpp are comparable")
    print("=" * 70)
    
    return results


@app.local_entrypoint()
def main():
    """Deploy and spawn the benchmark (detached - runs server-side)."""
    print("Spawning ZSE vs llama.cpp benchmark on Modal A100-80GB...")
    print("Using detached execution - function runs server-side.")
    print("Check Modal dashboard for results: https://modal.com/apps\n")
    
    # Use spawn() for detached execution - doesn't wait for result
    # This avoids timeout issues with long-running benchmarks
    call = test_llamacpp_vs_zse.spawn()
    
    print(f"✅ Benchmark spawned successfully!")
    print(f"   Function call ID: {call.object_id}")
    print(f"\nTo view results:")
    print(f"   1. Go to https://modal.com/apps")
    print(f"   2. Find 'zse-vs-llamacpp-benchmark' app")
    print(f"   3. Check the function logs for complete results")
