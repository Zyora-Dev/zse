"""Test IntelligenceOrchestrator with Qwen 2.5 Coder 32B"""

import modal

app = modal.App("test-qwen32b")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "safetensors",
        "sentencepiece",
        "protobuf",
    ])
    .env({"HF_TOKEN": "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"})
    .add_local_dir("/Users/redfoxhotels/zse/zse", remote_path="/root/zse/zse")
)

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=600,
    memory=65536,
)
def test_qwen32b():
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    print("=" * 60)
    print("TESTING: Qwen 2.5 Coder 32B with IntelligenceOrchestrator")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    from zse.engine.orchestrator import IntelligenceOrchestrator
    
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    # Test 1: Auto-detect
    print("[1] Auto-detecting optimal configuration...")
    orchestrator = IntelligenceOrchestrator.auto(model_name)
    config = orchestrator.get_config()
    print(f"  ✅ Auto-selected: {config.quantization.upper()}")
    print(f"  ✅ Estimated VRAM: {config.estimated_vram_gb:.1f} GB")
    print()
    
    # Test 2: Load with INT4 (min memory)
    print("[2] Loading with INT4 (min_memory mode)...")
    print("   Expected: ~16-18 GB VRAM for 32B model")
    orchestrator = IntelligenceOrchestrator.min_memory(model_name)
    orchestrator.load()
    
    actual_vram = torch.cuda.memory_allocated() / (1024**3)
    print(f"  ✅ Actual GPU memory: {actual_vram:.2f} GB")
    print()
    
    # Test 3: Generate code
    print("[3] Testing code generation...")
    prompt = "Write a Python function to check if a number is prime:"
    print(f"  Prompt: {prompt}")
    
    import time
    start = time.perf_counter()
    output = ""
    tokens = 0
    for chunk in orchestrator.generate(prompt, max_tokens=100, temperature=0.1):
        output += chunk
        tokens += 1
    elapsed = time.perf_counter() - start
    
    print(f"  Output:\n{output}")
    print()
    print(f"  ✅ Tokens: {tokens}")
    print(f"  ✅ Time: {elapsed:.2f}s")
    print(f"  ✅ Speed: {tokens/elapsed:.1f} tok/s")
    print()
    
    # Test 4: Benchmark
    print("[4] Running benchmark...")
    stats = orchestrator.benchmark(prompt="def fibonacci(n):", tokens=50)
    print(f"  ✅ Speed: {stats.tokens_per_sec:.1f} tok/s")
    print(f"  ✅ Peak Memory: {stats.peak_memory_gb:.2f} GB")
    print()
    
    print("=" * 60)
    print("SUMMARY: Qwen 2.5 Coder 32B")
    print("=" * 60)
    print(f"  - Model: {model_name}")
    print(f"  - Quantization: INT4/NF4")
    print(f"  - VRAM Used: {actual_vram:.2f} GB")
    print(f"  - Speed: {stats.tokens_per_sec:.1f} tok/s")
    print(f"  - Generation: {'✅ Working' if len(output) > 10 else '❌ Failed'}")

@app.local_entrypoint()
def main():
    test_qwen32b.remote()
