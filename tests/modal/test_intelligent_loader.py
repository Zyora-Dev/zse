"""
Test IntelligenceOrchestrator - Does it actually work?
"""

import modal

app = modal.App("test-intelligent-loader")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "safetensors",
        "sentencepiece",
        "triton>=2.1.0",
    ])
    .add_local_dir("/Users/redfoxhotels/zse/zse", remote_path="/root/zse/zse")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    memory=32768,
)
def test_intelligent_loader():
    """Test if IntelligenceOrchestrator works."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import time
    
    print("=" * 60)
    print("TESTING INTELLIGENCE ORCHESTRATOR")
    print("=" * 60)
    
    # Check GPU
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test 1: Import
    print("\n[1] Importing IntelligenceOrchestrator...")
    try:
        from zse.engine.orchestrator import IntelligenceOrchestrator
        print("  ✅ Import successful")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return {"success": False, "error": str(e)}
    
    # Test 2: Auto-detect config
    print("\n[2] Testing auto-detection...")
    try:
        orch = IntelligenceOrchestrator.auto("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        config = orch.get_config()
        print(f"  ✅ Auto-selected: {config.quantization.upper()}")
        print(f"  ✅ Estimated VRAM: {config.estimated_vram_gb:.1f} GB")
    except Exception as e:
        print(f"  ❌ Auto-detect failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Load model with min_memory (INT4)
    print("\n[3] Loading model with INT4 (min_memory)...")
    try:
        start = time.time()
        orch = IntelligenceOrchestrator.min_memory("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        orch.load(verbose=True)
        load_time = time.time() - start
        print(f"  ✅ Loaded in {load_time:.1f}s")
        
        # Check actual memory
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  ✅ Actual GPU memory: {mem_gb:.2f} GB")
    except Exception as e:
        print(f"  ❌ Load failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
    # Test 4: Generate text
    print("\n[4] Testing generation...")
    try:
        prompt = "Write a hello world in Python:"
        print(f"  Prompt: {prompt}")
        print("  Output: ", end="")
        
        start = time.time()
        output = ""
        for chunk in orch.generate(prompt, max_tokens=50, stream=True):
            print(chunk, end="", flush=True)
            output += chunk
        gen_time = time.time() - start
        
        print()
        print(f"\n  ✅ Generated in {gen_time:.1f}s")
        
        # Estimate tokens
        tokens = len(output.split()) * 1.3  # rough estimate
        tps = tokens / gen_time if gen_time > 0 else 0
        print(f"  ✅ Speed: ~{tps:.1f} tok/s (estimated)")
    except Exception as e:
        print(f"\n  ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
    # Test 5: Benchmark
    print("\n[5] Running benchmark...")
    try:
        stats = orch.benchmark(tokens=50)
        print(f"  ✅ Tokens: {stats.tokens_generated}")
        print(f"  ✅ Time: {stats.total_time_sec:.2f}s")
        print(f"  ✅ Speed: {stats.tokens_per_sec:.1f} tok/s")
        print(f"  ✅ Peak Memory: {stats.peak_memory_gb:.2f} GB")
    except Exception as e:
        print(f"  ❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    orch.unload()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
  IntelligenceOrchestrator STATUS:
  - Import: ✅ Works
  - Auto-detect: Need to verify
  - Model loading: Need to verify  
  - Generation: Need to verify
  - Streaming: Need to verify
""")
    
    return {"success": True}

@app.local_entrypoint()
def main():
    print("Testing IntelligenceOrchestrator...")
    result = test_intelligent_loader.remote()
    print(f"\nResult: {result}")
