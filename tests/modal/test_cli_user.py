"""
Test ZSE CLI from user perspective on Modal GPU.
"""

import modal

app = modal.App("test-zse-cli-user")

cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "psutil>=5.9.0",
        "pynvml>=11.5.0",
        "uvicorn>=0.25.0",
        "fastapi>=0.109.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "protobuf",
    )
    .pip_install("triton>=2.1.0")
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
    image=cuda_image,
    gpu="A10G",
    timeout=600,
)
def test_cli_user_experience():
    """Test CLI from a user's perspective."""
    import subprocess
    import sys
    import os
    
    os.chdir("/root/zse_pkg")
    sys.path.insert(0, "/root/zse_pkg")
    
    results = []
    
    def run_cmd(cmd, desc):
        """Run command and capture output."""
        print(f"\n{'='*60}")
        print(f"TEST: {desc}")
        print(f"CMD: {cmd}")
        print('='*60)
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/root/zse_pkg"
        )
        
        output = result.stdout + result.stderr
        # Filter warnings
        lines = [l for l in output.split('\n') 
                 if 'Warning' not in l and 'pynvml' not in l and 'frozen' not in l]
        clean_output = '\n'.join(lines)
        
        print(clean_output[:2000])
        
        success = result.returncode == 0
        results.append((desc, success))
        return success, clean_output
    
    # Test 1: Version
    print("\n" + "🧪 TEST 1: Version Check")
    run_cmd("python -m zse.api.cli.main --version", "Show version")
    
    # Test 2: Help
    print("\n" + "🧪 TEST 2: Help Command")
    run_cmd("python -m zse.api.cli.main --help", "Show help")
    
    # Test 3: Hardware Detection
    print("\n" + "🧪 TEST 3: Hardware Detection (Real GPU)")
    success, output = run_cmd("python -m zse.api.cli.main hardware", "Detect hardware")
    assert "A10G" in output or "GPU" in output, "Should detect A10G GPU"
    
    # Test 4: Model Info
    print("\n" + "🧪 TEST 4: Model Info")
    success, output = run_cmd(
        'python -m zse.api.cli.main info "Qwen/Qwen2.5-Coder-7B-Instruct"',
        "Get model info"
    )
    assert "7.0B" in output or "7B" in output, "Should show 7B params"
    assert "INT4" in output, "Should show INT4 option"
    
    # Test 5: Model Info JSON
    print("\n" + "🧪 TEST 5: Model Info (JSON format)")
    success, output = run_cmd(
        'python -m zse.api.cli.main info "meta-llama/Llama-3-8B" --format json',
        "Get model info as JSON"
    )
    assert '"estimated_params_b"' in output, "Should have JSON output"
    
    # Test 6: Serve Help
    print("\n" + "🧪 TEST 6: Serve Command Help")
    run_cmd("python -m zse.api.cli.main serve --help", "Serve help")
    
    # Test 7: Chat Help  
    print("\n" + "🧪 TEST 7: Chat Command Help")
    run_cmd("python -m zse.api.cli.main chat --help", "Chat help")
    
    # Test 8: Benchmark Help
    print("\n" + "🧪 TEST 8: Benchmark Command Help")
    run_cmd("python -m zse.api.cli.main benchmark --help", "Benchmark help")
    
    # Test 9: Convert Help
    print("\n" + "🧪 TEST 9: Convert Command Help")
    run_cmd("python -m zse.api.cli.main convert --help", "Convert help")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 CLI USER EXPERIENCE TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for desc, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {desc}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return {"passed": passed, "total": total, "results": results}


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=900,
)
def test_cli_real_inference():
    """Test CLI with real model inference."""
    import subprocess
    import sys
    import os
    import time
    
    os.chdir("/root/zse_pkg")
    sys.path.insert(0, "/root/zse_pkg")
    
    print("="*60)
    print("🚀 TESTING REAL INFERENCE VIA CLI")
    print("="*60)
    
    # Test actual model loading and inference via orchestrator
    print("\n📥 Loading small model for quick test...")
    
    from zse.engine.orchestrator.core import IntelligenceOrchestrator
    
    # Use a small model for speed
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    print(f"Model: {model_name}")
    print("Quantization: INT4 (minimum memory)")
    
    start = time.perf_counter()
    orch = IntelligenceOrchestrator.min_memory(model_name)
    orch.load(verbose=True)
    load_time = time.perf_counter() - start
    
    print(f"\n✅ Model loaded in {load_time:.1f}s")
    
    config = orch.get_config()
    print(f"   Quantization: {config.quantization}")
    print(f"   VRAM: {config.estimated_vram_gb:.2f} GB")
    
    # Test generation
    print("\n📝 Testing generation...")
    prompt = "Write a Python function to calculate fibonacci:"
    
    print(f"Prompt: {prompt}")
    print("Response: ", end="")
    
    start = time.perf_counter()
    tokens = 0
    response = []
    for chunk in orch.generate(prompt, max_tokens=100, stream=True):
        print(chunk, end="", flush=True)
        response.append(chunk)
        tokens += 1
    
    gen_time = time.perf_counter() - start
    print(f"\n\n⚡ Generated {tokens} tokens in {gen_time:.2f}s ({tokens/gen_time:.1f} tok/s)")
    
    # Cleanup
    orch.unload()
    
    print("\n" + "="*60)
    print("✅ CLI REAL INFERENCE TEST PASSED")
    print("="*60)
    
    return {
        "model": model_name,
        "load_time": load_time,
        "tokens": tokens,
        "gen_time": gen_time,
        "tok_per_sec": tokens / gen_time,
    }


@app.local_entrypoint()
def main():
    """Run CLI tests."""
    print("🧪 Running ZSE CLI User Experience Tests on Modal GPU...")
    print()
    
    # Test 1: Basic CLI commands
    result1 = test_cli_user_experience.remote()
    print(f"\n📊 Basic CLI: {result1['passed']}/{result1['total']} passed")
    
    # Test 2: Real inference
    print("\n" + "="*60)
    result2 = test_cli_real_inference.remote()
    print(f"\n📊 Real Inference: {result2['tok_per_sec']:.1f} tok/s")
    
    print("\n" + "="*60)
    print("✅ ALL CLI TESTS COMPLETE")
    print("="*60)
