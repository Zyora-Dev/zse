"""ZSE Server E2E Test — OpenAI-compatible API on A100.

Starts ZSEServer with Qwen2.5-7B, sends HTTP requests, validates responses.
Tests: health, models, completions (sync), chat (sync), chat (streaming SSE).
"""

import sys
import modal

app = modal.App("zse-server-e2e")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)


@app.function(
    gpu="A100",
    image=zse_image,
    timeout=3600,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def test_server_e2e():
    """E2E test: start server, send requests, validate responses."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import os
    import time
    import json
    import asyncio
    import threading
    import http.client

    results = {}

    print("=" * 70)
    print("ZSE SERVER E2E TEST — Qwen2.5-7B on A100")
    print("=" * 70)

    # Init CUDA
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    ret = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert ret == 0, f"cuCtxCreate failed: {ret}"

    # Ensure .zse file exists
    zse_path = "/root/zse_cache/qwen2_7b.zse"

    if os.path.exists(zse_path) and os.path.getsize(zse_path) > 1_000_000_000:
        with open(zse_path, 'rb') as f:
            magic = f.read(4)
        if magic != b'ZSE\x01':
            os.remove(zse_path)
        else:
            # Validate eos_id
            try:
                from zse_engine.format.loader import ZSELoader
                _loader = ZSELoader(zse_path)
                _eos = _loader.tokenizer.special_tokens.eos_id
                _loader.close()
                if _eos < 100:
                    print(f"[CACHE] Stale .zse (eos_id={_eos}), reconverting...")
                    os.remove(zse_path)
                    progress_file = zse_path + ".progress"
                    if os.path.exists(progress_file):
                        os.remove(progress_file)
            except Exception:
                pass

    if not os.path.exists(zse_path):
        progress_file = zse_path + ".progress"
        if os.path.exists(progress_file):
            os.remove(progress_file)
        from huggingface_hub import snapshot_download
        print("[0] Downloading + converting Qwen2.5-7B-Instruct...")
        hf_dir = snapshot_download(
            "Qwen/Qwen2.5-7B-Instruct",
            cache_dir="/root/hf_cache",
            allow_patterns=["*.safetensors", "*.json"],
        )
        from zse_engine.format.convert import convert_hf_to_zse
        convert_hf_to_zse(hf_dir, zse_path)
        try:
            vol = modal.Volume.lookup("zse-model-cache")
            vol.commit()
        except Exception:
            pass
    else:
        print(f"[CACHE] Using cached .zse ({os.path.getsize(zse_path)/1024**3:.2f} GB)")

    # ================================================================
    # Start server
    # ================================================================
    print("\n[1] Starting ZSE Server...")
    PORT = 8321
    ADMIN_KEY = "sk-zse-test-admin-key"

    from zse_engine.server.app import ZSEServer

    server = ZSEServer(
        model_path=zse_path,
        host="127.0.0.1",
        port=PORT,
        admin_key=ADMIN_KEY,
        db_path="/tmp/zse_test.db",
        model_name="qwen2.5-7b",
        quiet=False,
    )

    # Run server in background thread with its own event loop
    server_loop = asyncio.new_event_loop()
    server_ready = threading.Event()

    def run_server():
        asyncio.set_event_loop(server_loop)
        server_loop.run_until_complete(server.start())
        server_ready.set()
        server_loop.run_forever()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    server_ready.wait(timeout=30)
    print(f"     Server running on 127.0.0.1:{PORT}")

    # Wait a moment for engine loop to start
    time.sleep(2)

    # Helper: send HTTP request
    def http_request(method, path, body=None, headers=None, timeout=120):
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)
        body_bytes = json.dumps(body).encode() if body else None
        conn.request(method, path, body=body_bytes, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, json.loads(data) if data else {}

    def http_raw(method, path, timeout=10):
        """Raw HTTP request, returns (status, content_type, body_bytes)."""
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
        conn.request(method, path)
        resp = conn.getresponse()
        data = resp.read()
        ct = resp.getheader("Content-Type", "")
        conn.close()
        return resp.status, ct, data

    def http_stream(path, body, headers=None, timeout=120):
        """Send request and read SSE stream."""
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)
        conn.request("POST", path, body=json.dumps(body).encode(), headers=hdrs)
        resp = conn.getresponse()

        chunks = []
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            # Parse SSE events
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                line = event.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        chunks.append("[DONE]")
                    else:
                        chunks.append(json.loads(data))
        conn.close()
        return resp.status, chunks

    passed = 0
    failed = 0

    # ================================================================
    # Test 1: Health check
    # ================================================================
    print("\n[2] Testing GET /health...")
    try:
        status, data = http_request("GET", "/health")
        assert status == 200, f"Expected 200, got {status}"
        assert data["status"] == "ok", f"Expected ok, got {data}"
        print(f"     ✅ PASS: {data}")
        passed += 1
    except Exception as e:
        print(f"     ❌ FAIL: {e}")
        failed += 1

    # ================================================================
    # Test 1b: Dashboard UI
    # ================================================================
    print("\n[2b] Testing Dashboard (GET /, /static/style.css, /static/app.js)...")
    try:
        # Index page
        status, ct, body = http_raw("GET", "/")
        assert status == 200, f"/ returned {status}"
        assert "text/html" in ct, f"/ content-type: {ct}"
        html = body.decode("utf-8")
        assert "<title>ZSE</title>" in html, "Missing title"
        assert "chat-input" in html, "Missing chat input"
        print(f"     index.html: {status} OK ({len(body)} bytes)")

        # CSS
        status, ct, body = http_raw("GET", "/static/style.css")
        assert status == 200, f"/static/style.css returned {status}"
        assert "text/css" in ct, f"CSS content-type: {ct}"
        css = body.decode("utf-8")
        assert "--bg-main" in css, "Missing CSS vars"
        print(f"     style.css:  {status} OK ({len(body)} bytes)")

        # JS
        status, ct, body = http_raw("GET", "/static/app.js")
        assert status == 200, f"/static/app.js returned {status}"
        js = body.decode("utf-8")
        assert "sendMessage" in js, "Missing sendMessage function"
        print(f"     app.js:     {status} OK ({len(body)} bytes)")

        print(f"     ✅ PASS: Dashboard serving correctly")
        passed += 1
    except Exception as e:
        print(f"     ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        failed += 1
    print("\n[3] Testing GET /v1/models...")
    try:
        status, data = http_request("GET", "/v1/models")
        assert status == 200, f"Expected 200, got {status}"
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        model_id = data["data"][0]["id"]
        print(f"     ✅ PASS: models={[m['id'] for m in data['data']]}")
        passed += 1
    except Exception as e:
        print(f"     ❌ FAIL: {e}")
        failed += 1

    # ================================================================
    # Test 3: POST /v1/completions (non-streaming)
    # ================================================================
    print("\n[4] Testing POST /v1/completions (sync)...")
    try:
        status, data = http_request("POST", "/v1/completions", body={
            "prompt": "The capital of France is",
            "max_tokens": 32,
            "temperature": 0.0,
        })
        assert status == 200, f"Expected 200, got {status}: {data}"
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        text = data["choices"][0]["text"]
        assert len(text) > 0, "Empty output"
        print(f"     ✅ PASS: '{text[:80]}...'")
        print(f"     Usage: {data.get('usage', {})}")
        results["completion_text"] = text[:100]
        passed += 1
    except Exception as e:
        print(f"     ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ================================================================
    # Test 4: POST /v1/chat/completions (non-streaming)
    # ================================================================
    print("\n[5] Testing POST /v1/chat/completions (sync)...")
    try:
        status, data = http_request("POST", "/v1/chat/completions", body={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2? Answer in one word."},
            ],
            "max_tokens": 16,
            "temperature": 0.0,
        })
        assert status == 200, f"Expected 200, got {status}: {data}"
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        msg = data["choices"][0]["message"]
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert len(content) > 0, "Empty response"
        print(f"     ✅ PASS: '{content[:80]}'")
        print(f"     Usage: {data.get('usage', {})}")
        results["chat_text"] = content[:100]
        passed += 1
    except Exception as e:
        print(f"     ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ================================================================
    # Test 5: POST /v1/chat/completions (streaming SSE)
    # ================================================================
    print("\n[6] Testing POST /v1/chat/completions (streaming)...")
    try:
        status, chunks = http_stream("/v1/chat/completions", body={
            "messages": [
                {"role": "user", "content": "Say hello in French."},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": True,
        })
        assert status == 200, f"Expected 200, got {status}"
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
        assert chunks[-1] == "[DONE]", f"Expected [DONE], got {chunks[-1]}"

        # Reconstruct text from deltas
        text_parts = []
        for chunk in chunks:
            if chunk == "[DONE]":
                break
            if "choices" in chunk:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    text_parts.append(delta["content"])
        full_text = "".join(text_parts)
        print(f"     ✅ PASS: {len(chunks)} chunks, text='{full_text[:80]}'")
        results["stream_text"] = full_text[:100]
        results["stream_chunks"] = len(chunks)
        passed += 1
    except Exception as e:
        print(f"     ❌ FAIL: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ================================================================
    # Test 6: Admin stats
    # ================================================================
    print("\n[7] Testing GET /v1/admin/stats...")
    try:
        status, data = http_request("GET", "/v1/admin/stats", headers={
            "Authorization": f"Bearer {ADMIN_KEY}",
        })
        assert status == 200, f"Expected 200, got {status}: {data}"
        print(f"     ✅ PASS: engine={data.get('engine', {})}")
        passed += 1
    except Exception as e:
        print(f"     ❌ FAIL: {e}")
        failed += 1

    # ================================================================
    # Shutdown
    # ================================================================
    print("\n[8] Shutting down server...")
    server_loop.call_soon_threadsafe(server_loop.stop)
    time.sleep(1)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    results["passed"] = passed
    results["failed"] = failed

    if results.get("completion_text"):
        print(f"  Completion: {results['completion_text']}")
    if results.get("chat_text"):
        print(f"  Chat:       {results['chat_text']}")
    if results.get("stream_text"):
        print(f"  Stream:     {results['stream_text']} ({results['stream_chunks']} chunks)")

    if failed == 0:
        print("\n  ✅ ALL SERVER TESTS PASSED!")
    else:
        print(f"\n  ❌ {failed} TESTS FAILED")

    return results


@app.local_entrypoint()
def main():
    print("Launching ZSE Server E2E Test...")
    results = test_server_e2e.remote()
    print("\nResults:", results)
