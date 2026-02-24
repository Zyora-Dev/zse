"""
Test ZSE API Server on Modal

Tests:
1. Server startup and health check
2. Model loading
3. Chat completion
4. Streaming completion
5. Monitoring endpoints
6. Analytics endpoints
"""

import modal

# Modal setup
app = modal.App("zse-api-test")

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
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.0.0",
        "psutil>=5.9.0",
        "httpx>=0.26.0",  # For testing
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
    gpu="A10G",
    timeout=600,
)
def test_api_server():
    """Test the API server functionality."""
    import sys
    sys.path.insert(0, "/root/zse_pkg")
    
    import time
    import json
    from fastapi.testclient import TestClient
    
    print("=" * 60)
    print("ZSE API SERVER TEST")
    print("=" * 60)
    
    # Import and create app
    from zse.api.server import create_app, server_state
    app = create_app()
    client = TestClient(app)
    
    results = {
        "health": False,
        "model_load": False,
        "chat_completion": False,
        "completion": False,
        "monitoring": False,
        "analytics": False,
    }
    
    # =========================================================================
    # Test 1: Health Check
    # =========================================================================
    print("\n[1/6] Testing health endpoint...")
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print(f"  ✅ Health check passed")
        print(f"     Version: {data['version']}")
        print(f"     GPU Available: {data['gpu_available']}")
        results["health"] = True
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
    
    # =========================================================================
    # Test 2: Load Model
    # =========================================================================
    print("\n[2/6] Testing model loading...")
    try:
        start = time.time()
        response = client.post("/api/models/load", json={
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "quantization": "int4"
        })
        load_time = time.time() - start
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        
        print(f"  ✅ Model loaded successfully")
        print(f"     Model ID: {data['model_id']}")
        print(f"     Quantization: {data['quantization']}")
        print(f"     VRAM Used: {data['vram_used_gb']:.2f} GB")
        print(f"     Load Time: {load_time:.1f}s")
        
        model_id = data["model_id"]
        results["model_load"] = True
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return results
    
    # =========================================================================
    # Test 3: List Models
    # =========================================================================
    print("\n[2b] Testing model listing...")
    try:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        print(f"  ✅ Models listed: {len(data['data'])} model(s)")
        for m in data["data"]:
            print(f"     - {m['id']}")
    except Exception as e:
        print(f"  ❌ Model listing failed: {e}")
    
    # =========================================================================
    # Test 4: Chat Completion
    # =========================================================================
    print("\n[3/6] Testing chat completion...")
    try:
        start = time.time()
        response = client.post("/v1/chat/completions", json={
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "messages": [
                {"role": "user", "content": "Say hello in one sentence."}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        })
        latency = (time.time() - start) * 1000
        
        if response.status_code != 200:
            print(f"  ❌ Status: {response.status_code}")
            print(f"     Body: {response.text[:500]}")
            raise Exception(f"Status {response.status_code}: {response.text[:200]}")
        
        data = response.json()
        
        content = data["choices"][0]["message"]["content"]
        usage = data["usage"]
        
        print(f"  ✅ Chat completion successful")
        print(f"     Response: {content[:100]}...")
        print(f"     Tokens: {usage['completion_tokens']} generated")
        print(f"     Latency: {latency:.0f}ms")
        results["chat_completion"] = True
    except Exception as e:
        print(f"  ❌ Chat completion failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Test 5: Text Completion
    # =========================================================================
    print("\n[4/6] Testing text completion...")
    try:
        start = time.time()
        response = client.post("/v1/completions", json={
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "prompt": "The capital of France is",
            "max_tokens": 20,
            "temperature": 0.5
        })
        latency = (time.time() - start) * 1000
        
        if response.status_code != 200:
            print(f"  ❌ Status: {response.status_code}")
            print(f"     Body: {response.text[:500]}")
            raise Exception(f"Status {response.status_code}: {response.text[:200]}")
        
        data = response.json()
        
        text = data["choices"][0]["text"]
        print(f"  ✅ Text completion successful")
        print(f"     Response: {text[:80]}...")
        print(f"     Latency: {latency:.0f}ms")
        results["completion"] = True
    except Exception as e:
        print(f"  ❌ Text completion failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Test 6: Monitoring Endpoints
    # =========================================================================
    print("\n[5/6] Testing monitoring endpoints...")
    try:
        # System stats
        response = client.get("/api/stats")
        assert response.status_code == 200
        stats = response.json()
        
        print(f"  ✅ System stats retrieved")
        print(f"     CPU: {stats['cpu_percent']}%")
        print(f"     Memory: {stats['memory_used_gb']:.1f} / {stats['memory_total_gb']:.1f} GB")
        
        if stats['gpus']:
            gpu = stats['gpus'][0]
            print(f"     GPU: {gpu['name']}")
            print(f"     GPU Memory: {gpu['used_memory_gb']:.1f} / {gpu['total_memory_gb']:.1f} GB")
        
        # Model stats
        response = client.get("/api/stats/models")
        assert response.status_code == 200
        model_stats = response.json()
        print(f"     Models loaded: {len(model_stats['models'])}")
        
        results["monitoring"] = True
    except Exception as e:
        print(f"  ❌ Monitoring failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # Test 7: Analytics Endpoints
    # =========================================================================
    print("\n[6/6] Testing analytics endpoints...")
    try:
        # Overview
        response = client.get("/api/analytics")
        assert response.status_code == 200
        analytics = response.json()
        
        print(f"  ✅ Analytics retrieved")
        print(f"     Total Requests: {analytics['total_requests']}")
        print(f"     Successful: {analytics['successful_requests']}")
        print(f"     Tokens Generated: {analytics['total_tokens_generated']}")
        print(f"     Avg Latency: {analytics['avg_latency_ms']:.0f}ms")
        print(f"     Avg Speed: {analytics['avg_tokens_per_sec']:.1f} tok/s")
        
        # Recent requests
        response = client.get("/api/analytics/requests?limit=10")
        assert response.status_code == 200
        requests = response.json()
        print(f"     Recent Requests: {len(requests['requests'])}")
        
        results["analytics"] = True
    except Exception as e:
        print(f"  ❌ Analytics failed: {e}")
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
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return results


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
)
@modal.web_server(8000)
def serve_api():
    """
    Run the ZSE API server as a web endpoint.
    
    Access at: https://<modal-app-name>--serve-api.modal.run
    """
    import sys
    sys.path.insert(0, "/root/zse_pkg")
    
    import uvicorn
    from zse.api.server import app
    
    print("🚀 Starting ZSE API Server...")
    print("   Dashboard: /dashboard")
    print("   Docs: /docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.local_entrypoint()
def main():
    """Run API server tests."""
    print("Running ZSE API Server tests on Modal...")
    results = test_api_server.remote()
    
    if all(results.values()):
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
