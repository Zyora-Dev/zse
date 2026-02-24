"""
ZSE Batching Engine Test (Modal GPU)

Tests request batching for high-throughput inference.

Run with:
    modal run tests/modal/test_batching.py
"""

import modal
import os
import sys

# Modal app setup
app = modal.App("zse-batching-test")

# Get ZSE root for mounting
ZSE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPU image with ZSE code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "httpx>=0.25.0",
        "aiohttp>=3.9.0",
    ])
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
async def test_batching_engine():
    """Test the batching engine with concurrent requests."""
    import asyncio
    import time
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("=" * 60)
    print("ZSE BATCHING ENGINE TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Basic imports
    print("\n🔧 Test 1: Import batching engine...")
    try:
        from zse.engine.batching import BatchingEngine, BatchConfig
        print("  ✅ Import successful")
        results["imports"] = True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        results["imports"] = False
        return results
    
    # Test 2: Load a small model
    print("\n📦 Test 2: Loading TinyLlama model...")
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / (1024**3)
            print(f"  ✅ Model loaded: {mem:.2f} GB VRAM")
        else:
            print("  ✅ Model loaded (CPU)")
        
        results["model_load"] = True
    except Exception as e:
        print(f"  ❌ Model load failed: {e}")
        results["model_load"] = False
        return results
    
    # Test 3: Create batching engine
    print("\n⚙️ Test 3: Creating batching engine...")
    try:
        config = BatchConfig(
            max_batch_size=8,
            max_tokens_per_batch=2048,
            batch_wait_timeout_ms=100,
        )
        
        engine = BatchingEngine(model, tokenizer, config)
        await engine.start()
        
        print(f"  ✅ Engine started with config:")
        print(f"     Max batch size: {config.max_batch_size}")
        print(f"     Max tokens/batch: {config.max_tokens_per_batch}")
        print(f"     Batch wait: {config.batch_wait_timeout_ms}ms")
        
        results["engine_create"] = True
    except Exception as e:
        print(f"  ❌ Engine creation failed: {e}")
        import traceback
        traceback.print_exc()
        results["engine_create"] = False
        return results
    
    # Test 4: Single request
    print("\n🎯 Test 4: Single request...")
    try:
        start = time.time()
        output = await engine.generate(
            "What is 2+2?",
            max_tokens=20,
            temperature=0.7,
        )
        elapsed = time.time() - start
        
        print(f"  ✅ Single request completed in {elapsed:.2f}s")
        print(f"     Output: {output[:100]}...")
        
        results["single_request"] = True
    except Exception as e:
        print(f"  ❌ Single request failed: {e}")
        import traceback
        traceback.print_exc()
        results["single_request"] = False
    
    # Test 5: Concurrent requests (batching test)
    print("\n🚀 Test 5: Concurrent requests (batching)...")
    try:
        prompts = [
            "What is the capital of France?",
            "What is 10 * 5?",
            "Name a color.",
            "What is Python?",
            "Say hello.",
            "What is AI?",
            "Count to 5.",
            "What is water?",
        ]
        
        start = time.time()
        
        # Submit all requests concurrently
        tasks = [
            engine.generate(prompt, max_tokens=30, temperature=0.7)
            for prompt in prompts
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start
        
        print(f"  ✅ {len(prompts)} concurrent requests completed in {elapsed:.2f}s")
        print(f"     Throughput: {len(prompts) / elapsed:.2f} req/s")
        
        # Show sample outputs
        for i, (prompt, output) in enumerate(zip(prompts[:3], results_list[:3])):
            print(f"\n     [{i+1}] {prompt}")
            print(f"         → {output[:60]}...")
        
        results["concurrent_requests"] = True
        results["concurrent_time"] = elapsed
        results["concurrent_throughput"] = len(prompts) / elapsed
        
    except Exception as e:
        print(f"  ❌ Concurrent requests failed: {e}")
        import traceback
        traceback.print_exc()
        results["concurrent_requests"] = False
    
    # Test 6: Sequential baseline (for comparison)
    print("\n📊 Test 6: Sequential baseline (for comparison)...")
    try:
        prompts_small = prompts[:4]  # Use fewer for speed
        
        start = time.time()
        
        # Process sequentially
        for prompt in prompts_small:
            await engine.generate(prompt, max_tokens=30, temperature=0.7)
        
        elapsed = time.time() - start
        
        print(f"  ✅ {len(prompts_small)} sequential requests completed in {elapsed:.2f}s")
        print(f"     Throughput: {len(prompts_small) / elapsed:.2f} req/s")
        
        results["sequential_time"] = elapsed
        results["sequential_throughput"] = len(prompts_small) / elapsed
        
        # Calculate speedup
        if results.get("concurrent_throughput") and results.get("sequential_throughput"):
            speedup = results["concurrent_throughput"] / results["sequential_throughput"]
            print(f"\n  📈 Batching speedup: {speedup:.2f}x")
            results["speedup"] = speedup
        
    except Exception as e:
        print(f"  ❌ Sequential baseline failed: {e}")
        results["sequential_baseline"] = False
    
    # Test 7: Streaming
    print("\n🌊 Test 7: Streaming generation...")
    try:
        start = time.time()
        tokens = []
        
        async for token in engine.generate_stream(
            "Count from 1 to 5:",
            max_tokens=30,
            temperature=0.7,
        ):
            tokens.append(token)
        
        elapsed = time.time() - start
        full_output = "".join(tokens)
        
        print(f"  ✅ Streaming completed in {elapsed:.2f}s")
        print(f"     Tokens streamed: {len(tokens)}")
        print(f"     Output: {full_output[:100]}")
        
        results["streaming"] = True
    except Exception as e:
        print(f"  ❌ Streaming failed: {e}")
        import traceback
        traceback.print_exc()
        results["streaming"] = False
    
    # Test 8: Engine stats
    print("\n📊 Test 8: Engine stats...")
    try:
        stats = engine.stats()
        print(f"  ✅ Stats:")
        print(f"     Total requests: {stats['total_requests']}")
        print(f"     Total tokens: {stats['total_tokens_generated']}")
        print(f"     Total batches: {stats['total_batches']}")
        print(f"     Avg batch size: {stats['avg_batch_size']:.2f}")
        
        results["stats"] = True
    except Exception as e:
        print(f"  ❌ Stats failed: {e}")
        results["stats"] = False
    
    # Cleanup
    print("\n🧹 Cleanup...")
    await engine.stop()
    print("  ✅ Engine stopped")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for k, v in results.items() if v is True)
    total = sum(1 for k, v in results.items() if isinstance(v, bool))
    
    for key, value in results.items():
        if isinstance(value, bool):
            status = "✅ PASSED" if value else "❌ FAILED"
            print(f"  {key}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if results.get("speedup"):
        print(f"\n🚀 Batching speedup: {results['speedup']:.2f}x")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED")
    
    return results


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
async def test_api_server_batching():
    """Test batching through the API server."""
    import asyncio
    import time
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import httpx
    from multiprocessing import Process
    import uvicorn
    
    print("=" * 60)
    print("ZSE API SERVER BATCHING TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Start server
    print("\n🚀 Test 1: Starting API server...")
    try:
        from zse.api.server.app import create_app
        
        app = create_app()
        
        # Run server in background
        config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="warning")
        server = uvicorn.Server(config)
        
        # Start in background task
        server_task = asyncio.create_task(server.serve())
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        print("  ✅ Server started on port 8080")
        results["server_start"] = True
    except Exception as e:
        print(f"  ❌ Server start failed: {e}")
        import traceback
        traceback.print_exc()
        results["server_start"] = False
        return results
    
    base_url = "http://127.0.0.1:8080"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Test 2: Health check
        print("\n❤️ Test 2: Health check...")
        try:
            resp = await client.get(f"{base_url}/health")
            assert resp.status_code == 200
            print(f"  ✅ Health check passed: {resp.json()['status']}")
            results["health"] = True
        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
            results["health"] = False
        
        # Test 3: Load model
        print("\n📦 Test 3: Loading model...")
        try:
            resp = await client.post(
                f"{base_url}/api/models/load",
                json={
                    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "quantization": "auto",
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                print(f"  ✅ Model loaded: {data['model_name']}")
                print(f"     Quantization: {data['quantization']}")
                print(f"     VRAM: {data['vram_used_gb']:.2f} GB")
                results["model_load"] = True
            else:
                print(f"  ❌ Model load failed: {resp.text}")
                results["model_load"] = False
        except Exception as e:
            print(f"  ❌ Model load failed: {e}")
            import traceback
            traceback.print_exc()
            results["model_load"] = False
        
        # Test 4: Enable batching
        print("\n⚙️ Test 4: Enable batching...")
        try:
            resp = await client.post(f"{base_url}/api/batching/enable")
            assert resp.status_code == 200
            print(f"  ✅ Batching enabled: {resp.json()}")
            results["batching_enable"] = True
        except Exception as e:
            print(f"  ❌ Enable batching failed: {e}")
            results["batching_enable"] = False
        
        # Test 5: Batching status
        print("\n📊 Test 5: Batching status...")
        try:
            resp = await client.get(f"{base_url}/api/batching")
            data = resp.json()
            print(f"  ✅ Batching status: enabled={data['enabled']}")
            results["batching_status"] = True
        except Exception as e:
            print(f"  ❌ Batching status failed: {e}")
            results["batching_status"] = False
        
        # Disable auth for testing
        print("\n🔓 Disabling authentication for test...")
        try:
            resp = await client.post(f"{base_url}/api/keys/auth", json={"enabled": False})
            if resp.status_code == 200:
                print(f"  ✅ Auth disabled: {resp.json()}")
            else:
                print(f"  ⚠️ Auth disable returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"  ⚠️ Could not disable auth: {e}")
        
        if results.get("model_load"):
            # Test 6: Concurrent chat completions
            print("\n🚀 Test 6: Concurrent chat completions...")
            try:
                prompts = [
                    "What is 2+2?",
                    "Name a fruit.",
                    "What color is the sky?",
                    "Say hello.",
                ]
                
                start = time.time()
                
                async def make_request(prompt):
                    resp = await client.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 30,
                        }
                    )
                    return resp.json()
                
                # Submit concurrently
                tasks = [make_request(p) for p in prompts]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                elapsed = time.time() - start
                
                success_count = 0
                for i, resp in enumerate(responses):
                    if isinstance(resp, Exception):
                        print(f"     [{i+1}] Exception: {resp}")
                    elif isinstance(resp, dict) and "choices" in resp:
                        success_count += 1
                        content = resp["choices"][0]["message"]["content"][:50]
                        print(f"     [{i+1}] ✅ {content}...")
                    else:
                        print(f"     [{i+1}] ❌ Unexpected response: {type(resp)}: {str(resp)[:100]}")
                
                print(f"\n  Summary: {success_count}/{len(prompts)} requests completed in {elapsed:.2f}s")
                if success_count > 0:
                    print(f"     Throughput: {success_count / elapsed:.2f} req/s")
                
                results["concurrent_chat"] = success_count == len(prompts)
                results["chat_throughput"] = success_count / elapsed if success_count > 0 else 0
                
            except Exception as e:
                print(f"  ❌ Concurrent chat failed: {e}")
                import traceback
                traceback.print_exc()
                results["concurrent_chat"] = False
    
    # Cleanup
    print("\n🧹 Cleanup...")
    server.should_exit = True
    await asyncio.sleep(1)
    print("  ✅ Server stopped")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for k, v in results.items() if v is True)
    total = sum(1 for k, v in results.items() if isinstance(v, bool))
    
    for key, value in results.items():
        if isinstance(value, bool):
            status = "✅ PASSED" if value else "❌ FAILED"
            print(f"  {key}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if results.get("chat_throughput"):
        print(f"\n🚀 Chat throughput: {results['chat_throughput']:.2f} req/s")
    
    return results


@app.local_entrypoint()
def main():
    """Run all batching tests."""
    import asyncio
    
    print("=" * 60)
    print("RUNNING BATCHING ENGINE TESTS")
    print("=" * 60)
    
    # Test 1: Direct engine test
    print("\n[1/2] Testing batching engine directly...")
    result1 = test_batching_engine.remote()
    
    # Test 2: API server test
    print("\n[2/2] Testing batching via API server...")
    result2 = test_api_server_batching.remote()
    
    print("\n" + "=" * 60)
    print("ALL BATCHING TESTS COMPLETED")
    print("=" * 60)
