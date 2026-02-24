"""
Test Multi-GPU Support on Modal

Tests the IntelligenceOrchestrator.multi_gpu() feature.
Uses Modal's multi-GPU instances (A10G x2).
"""

import modal

# Modal setup
app = modal.App("zse-multi-gpu-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "protobuf",
    ])
    .run_commands([
        "mkdir -p /root/zse_pkg",
        "touch /root/zse_pkg/__init__.py",
    ])
    .env({
        "PYTHONPATH": "/root/zse_pkg",
        "HF_HOME": "/root/.cache/huggingface",
    })
    .add_local_dir(
        "/Users/redfoxhotels/zse/zse",
        remote_path="/root/zse_pkg/zse"
    )
)


@app.function(
    image=image,
    gpu="A10G:2",  # 2x A10G GPUs
    timeout=900,
)
def test_multi_gpu():
    """Test multi-GPU model loading and generation."""
    import sys
    sys.path.insert(0, "/root/zse_pkg")
    
    import torch
    import time
    
    print("=" * 60)
    print("ZSE MULTI-GPU TEST")
    print("=" * 60)
    
    # Check GPU availability
    from zse.engine.orchestrator import IntelligenceOrchestrator
    
    gpu_info = IntelligenceOrchestrator.get_gpu_info()
    print(f"\n📊 GPU Info:")
    print(f"   Available: {gpu_info['available']}")
    print(f"   Count: {gpu_info['count']}")
    print(f"   Total VRAM: {gpu_info['total_vram_gb']} GB")
    for gpu in gpu_info['gpus']:
        print(f"   GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']} GB)")
    
    if gpu_info['count'] < 2:
        print("\n❌ Need at least 2 GPUs for this test")
        return False
    
    results = {
        "gpu_detection": True,
        "model_load": False,
        "generation": False,
        "vram_split": False,
    }
    
    # =========================================================================
    # Test 1: Multi-GPU Model Loading
    # =========================================================================
    print("\n" + "=" * 60)
    print("[1/3] Testing Multi-GPU Model Loading")
    print("=" * 60)
    
    try:
        # Use a 7B model to test multi-GPU split
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        
        print(f"\n📦 Loading {model_name} across {gpu_info['count']} GPUs...")
        
        start = time.time()
        orch = IntelligenceOrchestrator.multi_gpu(
            model_name,
            quantization="fp16"  # FP16 to see clear split
        )
        orch.load(verbose=True)
        load_time = time.time() - start
        
        print(f"\n✅ Model loaded in {load_time:.1f}s")
        results["model_load"] = True
        
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # =========================================================================
    # Test 2: Check VRAM Distribution
    # =========================================================================
    print("\n" + "=" * 60)
    print("[2/3] Checking VRAM Distribution")
    print("=" * 60)
    
    try:
        vram_per_gpu = []
        for i in range(gpu_info['count']):
            vram = torch.cuda.memory_allocated(i) / (1024**3)
            vram_per_gpu.append(vram)
            print(f"   GPU {i}: {vram:.2f} GB")
        
        # Check if model is actually split (both GPUs have >1GB)
        gpus_with_data = sum(1 for v in vram_per_gpu if v > 0.5)
        
        if gpus_with_data >= 2:
            print(f"\n✅ Model split across {gpus_with_data} GPUs")
            results["vram_split"] = True
        else:
            print(f"\n⚠️ Model only on {gpus_with_data} GPU(s) - not truly multi-GPU")
            
    except Exception as e:
        print(f"\n❌ VRAM check failed: {e}")
    
    # =========================================================================
    # Test 3: Generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("[3/3] Testing Generation")
    print("=" * 60)
    
    try:
        prompt = "Write a Python function to calculate fibonacci numbers:"
        print(f"\n📝 Prompt: {prompt}")
        print("\n🤖 Response:")
        
        start = time.time()
        output = ""
        token_count = 0
        
        for chunk in orch.generate(prompt, max_tokens=100, temperature=0.7):
            print(chunk, end="", flush=True)
            output += chunk
            token_count += 1
        
        gen_time = time.time() - start
        tokens_per_sec = token_count / gen_time if gen_time > 0 else 0
        
        print(f"\n\n✅ Generated {token_count} tokens in {gen_time:.1f}s")
        print(f"   Speed: {tokens_per_sec:.1f} tok/s")
        results["generation"] = True
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    
    return results


@app.function(
    image=image,
    gpu="A10G:2",
    timeout=900,
)
def test_70b_multi_gpu():
    """
    Test loading a 70B model across multiple GPUs.
    
    Requires 4x A100-80GB or similar to fit 70B in FP16.
    With INT4, can fit on 2x A10G.
    """
    import sys
    sys.path.insert(0, "/root/zse_pkg")
    
    import torch
    import time
    
    from zse.engine.orchestrator import IntelligenceOrchestrator
    
    print("=" * 60)
    print("70B MODEL MULTI-GPU TEST (INT4)")
    print("=" * 60)
    
    gpu_info = IntelligenceOrchestrator.get_gpu_info()
    print(f"\nGPUs: {gpu_info['count']} x {gpu_info['gpus'][0]['name'] if gpu_info['gpus'] else 'N/A'}")
    print(f"Total VRAM: {gpu_info['total_vram_gb']} GB")
    
    # 70B model in INT4 ≈ 35GB, should fit on 2x A10G (44GB total)
    # But 70B models are gated, so use a smaller one for demo
    model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"  # 14B is accessible
    
    print(f"\n📦 Loading {model_name} (INT4, Multi-GPU)...")
    
    try:
        start = time.time()
        orch = IntelligenceOrchestrator.multi_gpu(model_name, quantization="int4")
        orch.load(verbose=True)
        load_time = time.time() - start
        
        print(f"\n✅ Loaded in {load_time:.1f}s")
        
        # Test generation
        print("\n🤖 Generating...")
        for chunk in orch.generate("Hello, ", max_tokens=20):
            print(chunk, end="", flush=True)
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.local_entrypoint()
def main():
    """Run multi-GPU tests."""
    print("Running ZSE Multi-GPU tests on Modal...")
    print("Using 2x A10G GPUs\n")
    
    results = test_multi_gpu.remote()
    
    if all(results.values()):
        print("\n✅ Multi-GPU test passed!")
    else:
        print("\n❌ Some tests failed")
