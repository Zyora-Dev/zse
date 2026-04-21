"""ZSE Format — Real model conversion test on Modal.

Downloads Qwen2.5-0.5B and SmolLM-135M (tiny Llama-arch), converts to .zse,
loads back, and verifies weights on GPU via cuMemcpy.

Run: modal run tests/test_modal_convert.py

Note: Qwen2.5-0.5B has 494M params → pure Python quantization takes ~5-10 min.
SmolLM-135M is tiny → finishes in seconds.
"""

import modal
import sys

app = modal.App("zse-format-convert-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

# Cache model downloads across runs
hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)


@app.function(
    gpu="A100",
    image=zse_image,
    timeout=1800,  # 30 min for quantization
    volumes={"/root/hf_cache": hf_cache},
)
def test_real_model_convert():
    """Download, convert, load, and verify real models."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import os
    import time
    import struct
    from huggingface_hub import snapshot_download

    from zse_engine.format.convert import convert_hf_to_zse
    from zse_engine.format.loader import ZSELoader
    from zse_engine.format.spec import should_quantize

    print("=" * 60)
    print("ZSE FORMAT — REAL MODEL CONVERSION TEST")
    print("=" * 60)

    results = {}

    # ------------------------------------------------------------------ #
    # Test 1: SmolLM-135M (Llama architecture, tiny, fast)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[TEST 1] SmolLM-135M (Llama arch, 135M params)")
    print("=" * 60)

    print("\n[1a] Downloading SmolLM-135M...")
    t0 = time.time()
    smol_dir = snapshot_download(
        "HuggingFaceTB/SmolLM-135M",
        cache_dir="/root/hf_cache",
        allow_patterns=["*.safetensors", "*.json"],
    )
    t_download = time.time() - t0
    print(f"     Downloaded in {t_download:.1f}s")
    print(f"     Path: {smol_dir}")

    # List files
    for f in sorted(os.listdir(smol_dir)):
        fpath = os.path.join(smol_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"     {f}: {size:,} bytes")

    print("\n[1b] Converting to .zse...")
    smol_zse = "/tmp/smol_135m.zse"
    t0 = time.time()

    tensor_count = [0]
    def progress(name, cur, total):
        tensor_count[0] = total
        if cur == 1 or cur == total or cur % 20 == 0:
            print(f"     [{cur}/{total}] {name}")

    convert_hf_to_zse(smol_dir, smol_zse, progress_callback=progress)
    t_convert = time.time() - t0
    file_size = os.path.getsize(smol_zse)
    print(f"     Converted in {t_convert:.1f}s")
    print(f"     Output: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

    print("\n[1c] Loading and verifying .zse...")
    with ZSELoader(smol_zse) as loader:
        c = loader.config
        print(f"     Arch: {c.arch}")
        print(f"     Layers: {c.num_layers}, Heads: {c.num_heads}, KV Heads: {c.num_kv_heads}")
        print(f"     Hidden: {c.hidden_size}, Intermediate: {c.intermediate_size}")
        print(f"     Vocab: {c.vocab_size}")
        print(f"     Quant: {c.quant.bits}-bit, group={c.quant.group_size}")

        assert c.num_layers > 0
        assert c.hidden_size > 0

        wi = loader.weight_index
        print(f"     Tensors: {len(wi)}")
        q_count = sum(1 for e in wi if e.dtype == "int4")
        fp_count = sum(1 for e in wi if e.dtype == "float16")
        print(f"       INT4: {q_count}, FP16: {fp_count}")

        # Verify tokenizer
        if loader.tokenizer:
            print(f"     Tokenizer: {loader.tokenizer.vocab_size} tokens")

        # Spot-check: read a quantized tensor
        for entry in wi:
            if entry.dtype == "int4":
                data = loader.get_weight_data(entry)
                assert len(data) == entry.data_nbytes
                scales = loader.get_weight_scales(entry)
                assert len(scales) == entry.scale_nbytes
                # Dequantize and check values are reasonable
                vals = loader.get_weight_as_float(entry)
                assert len(vals) == entry.num_elements
                # Weights should be small (typical init is ~0.02 std)
                avg_abs = sum(abs(v) for v in vals[:1000]) / 1000
                print(f"     Spot check '{entry.name}': shape={entry.shape}, "
                      f"avg_abs={avg_abs:.4f}")
                assert avg_abs < 5.0, f"Weights look unreasonable: avg_abs={avg_abs}"
                break

    results["SmolLM-135M"] = {
        "status": "PASS",
        "download_time": t_download,
        "convert_time": t_convert,
        "file_size": file_size,
        "tensors": tensor_count[0],
    }
    print(f"\n     ✅ SmolLM-135M: PASS")

    # ------------------------------------------------------------------ #
    # Test 2: Qwen2.5-0.5B
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[TEST 2] Qwen2.5-0.5B (Qwen2 arch, 494M params)")
    print("=" * 60)

    print("\n[2a] Downloading Qwen2.5-0.5B...")
    t0 = time.time()
    qwen_dir = snapshot_download(
        "Qwen/Qwen2.5-0.5B",
        cache_dir="/root/hf_cache",
        allow_patterns=["*.safetensors", "*.json"],
    )
    t_download = time.time() - t0
    print(f"     Downloaded in {t_download:.1f}s")

    for f in sorted(os.listdir(qwen_dir)):
        fpath = os.path.join(qwen_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"     {f}: {size:,} bytes")

    print("\n[2b] Converting to .zse (this takes a few minutes)...")
    qwen_zse = "/tmp/qwen2.5_0.5b.zse"
    t0 = time.time()

    tensor_count[0] = 0
    def progress2(name, cur, total):
        tensor_count[0] = total
        if cur == 1 or cur == total or cur % 50 == 0:
            elapsed = time.time() - t0
            rate = cur / elapsed if elapsed > 0 else 0
            eta = (total - cur) / rate if rate > 0 else 0
            print(f"     [{cur}/{total}] {name} ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    convert_hf_to_zse(qwen_dir, qwen_zse, progress_callback=progress2)
    t_convert = time.time() - t0
    file_size = os.path.getsize(qwen_zse)
    print(f"     Converted in {t_convert:.1f}s")
    print(f"     Output: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

    print("\n[2c] Loading and verifying .zse...")
    with ZSELoader(qwen_zse) as loader:
        c = loader.config
        print(f"     Arch: {c.arch}")
        print(f"     Layers: {c.num_layers}, Heads: {c.num_heads}, KV Heads: {c.num_kv_heads}")
        print(f"     Hidden: {c.hidden_size}, Intermediate: {c.intermediate_size}")
        print(f"     Vocab: {c.vocab_size}")

        # Qwen2-specific checks
        assert c.arch == "qwen2"
        assert c.num_layers == 24
        assert c.hidden_size == 896
        assert c.num_kv_heads == 2
        assert c.vocab_size == 151936
        assert c.tie_word_embeddings is True

        wi = loader.weight_index
        print(f"     Tensors: {len(wi)}")
        q_count = sum(1 for e in wi if e.dtype == "int4")
        fp_count = sum(1 for e in wi if e.dtype == "float16")
        print(f"       INT4: {q_count}, FP16: {fp_count}")

        # Verify biases exist and are fp16
        bias_count = 0
        for entry in wi:
            if entry.name.endswith(".bias"):
                assert entry.dtype == "float16", f"Bias {entry.name} should be fp16"
                bias_count += 1
        print(f"       Biases (fp16): {bias_count}")
        assert bias_count == 24 * 3, f"Expected {24*3} biases, got {bias_count}"

        # Verify no lm_head (tied embeddings)
        assert wi.find("lm_head.weight") is None, "lm_head should not exist (tied)"

        # Tokenizer
        if loader.tokenizer:
            print(f"     Tokenizer: {loader.tokenizer.vocab_size} tokens")
            # Test encode/decode
            ids = loader.tokenizer.encode("Hello world", add_bos=True)
            decoded = loader.tokenizer.decode(ids, skip_special=True)
            print(f"     Encode test: 'Hello world' → {ids[:10]}... → '{decoded}'")

        # Spot check quantized tensor
        for entry in wi:
            if entry.dtype == "int4" and "q_proj" in entry.name:
                vals = loader.get_weight_as_float(entry)
                avg_abs = sum(abs(v) for v in vals[:1000]) / 1000
                print(f"     Spot check '{entry.name}': shape={entry.shape}, "
                      f"avg_abs={avg_abs:.4f}")
                assert avg_abs < 5.0
                break

        print(f"\n{loader.summary()}")

    results["Qwen2.5-0.5B"] = {
        "status": "PASS",
        "download_time": t_download,
        "convert_time": t_convert,
        "file_size": file_size,
        "tensors": tensor_count[0],
    }
    print(f"\n     ✅ Qwen2.5-0.5B: PASS")

    # ------------------------------------------------------------------ #
    # Test 3: GPU weight transfer (verify mmap → GPU works)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[TEST 3] GPU Weight Transfer")
    print("=" * 60)

    import ctypes

    # Use the SmolLM .zse file (smaller, faster)
    with ZSELoader(smol_zse) as loader:
        # Init CUDA
        libcuda = ctypes.CDLL("libcuda.so.1")
        libcuda.cuInit(0)

        ctx = ctypes.c_void_p()
        libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)

        # Pick first quantized weight
        for entry in loader.weight_index:
            if entry.dtype == "int4" and entry.data_nbytes > 0:
                break

        print(f"     Tensor: {entry.name}")
        print(f"     Shape: {entry.shape}, dtype: {entry.dtype}")
        print(f"     Data size: {entry.data_nbytes:,} bytes")

        # Get mmap pointer
        mm, offset = loader.get_mmap_pointer_and_offset(entry)
        host_data = loader.get_weight_data(entry)

        # Allocate GPU memory
        gpu_ptr = ctypes.c_uint64()
        ret = libcuda.cuMemAlloc_v2(ctypes.byref(gpu_ptr), entry.data_nbytes)
        assert ret == 0, f"cuMemAlloc failed: {ret}"
        print(f"     GPU alloc: {gpu_ptr.value:#x} ({entry.data_nbytes:,} bytes)")

        # Copy host → device
        t0 = time.time()
        ret = libcuda.cuMemcpyHtoD_v2(
            gpu_ptr,
            ctypes.c_char_p(host_data),
            entry.data_nbytes,
        )
        assert ret == 0, f"cuMemcpyHtoD failed: {ret}"
        t_copy = time.time() - t0
        bw = entry.data_nbytes / t_copy / 1e9 if t_copy > 0 else float('inf')
        print(f"     H→D copy: {t_copy*1000:.2f}ms ({bw:.1f} GB/s)")

        # Copy back device → host and verify
        verify_buf = ctypes.create_string_buffer(entry.data_nbytes)
        ret = libcuda.cuMemcpyDtoH_v2(verify_buf, gpu_ptr, entry.data_nbytes)
        assert ret == 0, f"cuMemcpyDtoH failed: {ret}"

        # Compare
        roundtrip = bytes(verify_buf)
        assert roundtrip == host_data, "GPU roundtrip data mismatch!"
        print(f"     Roundtrip verify: {len(roundtrip):,} bytes match ✓")

        # Also copy scales and zeros
        if entry.scale_nbytes > 0:
            scales_data = loader.get_weight_scales(entry)
            scales_gpu = ctypes.c_uint64()
            ret = libcuda.cuMemAlloc_v2(ctypes.byref(scales_gpu), entry.scale_nbytes)
            assert ret == 0
            ret = libcuda.cuMemcpyHtoD_v2(scales_gpu, ctypes.c_char_p(scales_data), entry.scale_nbytes)
            assert ret == 0
            print(f"     Scales uploaded: {entry.scale_nbytes:,} bytes ✓")
            libcuda.cuMemFree_v2(scales_gpu)

        if entry.zeros_nbytes > 0:
            zeros_data = loader.get_weight_zeros(entry)
            zeros_gpu = ctypes.c_uint64()
            ret = libcuda.cuMemAlloc_v2(ctypes.byref(zeros_gpu), entry.zeros_nbytes)
            assert ret == 0
            ret = libcuda.cuMemcpyHtoD_v2(zeros_gpu, ctypes.c_char_p(zeros_data), entry.zeros_nbytes)
            assert ret == 0
            print(f"     Zeros uploaded: {entry.zeros_nbytes:,} bytes ✓")
            libcuda.cuMemFree_v2(zeros_gpu)

        # Cleanup
        libcuda.cuMemFree_v2(gpu_ptr)
        libcuda.cuCtxDestroy_v2(ctx)

    results["GPU_transfer"] = {"status": "PASS"}
    print(f"\n     ✅ GPU weight transfer: PASS")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, r in results.items():
        status = r["status"]
        if status != "PASS":
            all_pass = False
        extra = ""
        if "convert_time" in r:
            extra = (f" | convert={r['convert_time']:.1f}s "
                     f"| size={r['file_size']/1024/1024:.1f}MB "
                     f"| tensors={r['tensors']}")
        symbol = "✅" if status == "PASS" else "❌"
        print(f"  {symbol} {name}: {status}{extra}")

    print("=" * 60)
    if all_pass:
        print("✅ ALL REAL MODEL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

    return "PASS" if all_pass else "FAIL"


@app.local_entrypoint()
def main():
    result = test_real_model_convert.remote()
    print(f"\nModal result: {result}")
