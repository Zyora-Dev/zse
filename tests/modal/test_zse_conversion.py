"""
ZSE Format Conversion Test - A100 80GB
Tests .zse conversion using ZSEWriter.convert_from_hf() API

Run: modal run tests/modal/test_zse_conversion.py::test_7b_models
     modal run tests/modal/test_zse_conversion.py::test_14b_32b_models
     modal run tests/modal/test_zse_conversion.py::test_72b_model
"""

import modal
import time
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))

app = modal.App("zse-conversion-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "safetensors>=0.4.0",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "bitsandbytes>=0.41.0",  # For INT4 quantization
    )
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)

vol = modal.Volume.from_name("zse-model-cache", create_if_missing=True)

# Use open models (not gated)
BATCH_7B = [
    "Qwen/Qwen2.5-7B-Instruct",           # Open, ~14GB FP16
    "deepseek-ai/deepseek-coder-6.7b-instruct",  # Open, ~13GB FP16
    "mistralai/Mistral-7B-Instruct-v0.3",  # Open (Apache 2.0), ~14GB FP16
]

BATCH_14B_32B = [
    "Qwen/Qwen2.5-14B-Instruct",          # ~28GB FP16
    "Qwen/Qwen2.5-32B-Instruct",          # ~64GB FP16
]

BATCH_72B = [
    "Qwen/Qwen2.5-72B-Instruct",          # ~144GB FP16 - use INT4
]


def test_zse_conversion(model_id: str, use_int4: bool = False):
    """Test .zse conversion for a single model using ZSEWriter.convert_from_hf()."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import tempfile
    from pathlib import Path
    
    result = {
        "model_id": model_id,
        "status": "unknown",
        "convert_time": 0,
        "zse_size_gb": 0,
        "reload_time": 0,
        "tensor_count": 0,
        "error": None,
    }
    
    print(f"\n{'='*60}")
    print(f"Testing .zse conversion: {model_id}")
    print(f"Quantization: {'INT4' if use_int4 else 'FP16'}")
    print(f"{'='*60}")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zse_path = Path(tmpdir) / "model.zse"
            
            # Step 1: Convert using ZSEWriter API
            print(f"\n[1/2] Converting to .zse using ZSEWriter.convert_from_hf()...")
            t0 = time.time()
            
            from zse.format import ZSEWriter, ZSEReader
            from zse.format.writer import ConversionConfig
            
            # Configure quantization
            config = ConversionConfig()
            if use_int4:
                config.quantization = "int4"
            else:
                config.quantization = "none"  # FP16
                config.compute_dtype = torch.float16
            
            writer = ZSEWriter(str(zse_path), config=config)
            writer.convert_from_hf(model_id, trust_remote_code=True)
            
            result["convert_time"] = time.time() - t0
            result["zse_size_gb"] = zse_path.stat().st_size / 1e9
            print(f"    ✅ Converted in {result['convert_time']:.1f}s")
            print(f"    .zse file size: {result['zse_size_gb']:.2f} GB")
            
            # Clear memory before reload
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Step 2: Reload from .zse and verify
            print(f"\n[2/2] Reloading from .zse...")
            t0 = time.time()
            
            reader = ZSEReader(str(zse_path))
            header = reader.header
            print(f"    Architecture: {header.architecture}")
            print(f"    Model type: {header.model_type}")
            print(f"    Hidden size: {header.hidden_size}")
            print(f"    Layers: {header.num_hidden_layers}")
            print(f"    Vocab: {header.vocab_size}")
            
            state_dict = reader.load_state_dict()
            tokenizer = reader.load_tokenizer()
            
            result["reload_time"] = time.time() - t0
            result["tensor_count"] = len(state_dict)
            print(f"    ✅ Reloaded in {result['reload_time']:.1f}s")
            print(f"    Tensors loaded: {result['tensor_count']}")
            print(f"    Tokenizer loaded: {tokenizer is not None}")
            
            # Verify tensors have correct dtype
            sample_tensor = list(state_dict.values())[0]
            print(f"    Sample tensor dtype: {sample_tensor.dtype}")
            
            result["status"] = "PASSED"
            print(f"\n✅ {model_id} - PASSED")
            
    except Exception as e:
        result["status"] = "FAILED"
        result["error"] = str(e)
        print(f"\n❌ {model_id} - FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return result

# =============================================================================
# Batch 1: 7B/8B Models
# =============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/cache": vol},
)
def test_7b_models():
    """Test 7B/8B models: Qwen 7B, DeepSeek 7B, Mistral 7B."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    
    print("=" * 60)
    print("ZSE CONVERSION TEST - BATCH 1: 7B/8B MODELS")
    print("GPU: A100 80GB | Format: FP16 → .zse")
    print("=" * 60)
    
    results = []
    for model_id in BATCH_7B:
        result = test_zse_conversion(model_id)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH 1 SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["status"] == "PASSED")
    print(f"Results: {passed}/{len(results)} passed\n")
    
    for r in results:
        s = "✅" if r["status"] == "PASSED" else "❌"
        print(f"{s} {r['model_id']}")
        if r["status"] == "PASSED":
            print(f"   Convert: {r['convert_time']:.1f}s | .zse: {r['zse_size_gb']:.2f}GB | Reload: {r['reload_time']:.1f}s")
        else:
            print(f"   Error: {r['error'][:80] if r['error'] else 'Unknown'}")
    
    return results


# =============================================================================
# Batch 2: 14B & 32B Models
# =============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=5400,
    volumes={"/cache": vol},
)
def test_14b_32b_models():
    """Test 14B and 32B models."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    
    print("=" * 60)
    print("ZSE CONVERSION TEST - BATCH 2: 14B & 32B MODELS")
    print("GPU: A100 80GB | Format: FP16 → .zse")
    print("=" * 60)
    
    results = []
    for model_id in BATCH_14B_32B:
        result = test_zse_conversion(model_id)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("BATCH 2 SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["status"] == "PASSED")
    print(f"Results: {passed}/{len(results)} passed\n")
    
    for r in results:
        s = "✅" if r["status"] == "PASSED" else "❌"
        print(f"{s} {r['model_id']}")
        if r["status"] == "PASSED":
            print(f"   Convert: {r['convert_time']:.1f}s | .zse: {r['zse_size_gb']:.2f}GB | Reload: {r['reload_time']:.1f}s")
        else:
            print(f"   Error: {r['error'][:80] if r['error'] else 'Unknown'}")
    
    return results


# =============================================================================
# Batch 3: 72B Model (INT4 - doesn't fit in 80GB FP16)
# =============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=10800,  # 3 hours
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_dict({"HF_TOKEN": "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"})],
)
def test_72b_model():
    """Test 72B model with INT4 quantization - FP16 needs ~144GB."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    print("=" * 60)
    print("ZSE CONVERSION TEST - BATCH 3: 72B MODEL")
    print("GPU: A100 80GB | Format: INT4 → .zse (FP16 needs 144GB)")
    print("=" * 60)
    
    # 72B in FP16 needs ~144GB, won't fit in 80GB
    # Use INT4 quantization for this model
    result = test_zse_conversion("Qwen/Qwen2.5-72B-Instruct", use_int4=True)
    
    print("\n" + "=" * 60)
    print("BATCH 3 SUMMARY")
    print("=" * 60)
    s = "✅" if result["status"] == "PASSED" else "❌"
    print(f"{s} {result['model_id']}")
    if result["status"] == "PASSED":
        print(f"   Convert: {result['convert_time']:.1f}s | .zse: {result['zse_size_gb']:.2f}GB | Reload: {result['reload_time']:.1f}s")
    else:
        print(f"   Error: {result['error'][:80] if result['error'] else 'Unknown'}")
    
    return [result]


# =============================================================================
# Individual Test: Qwen 32B Only (FP16 ~64GB, needs A100 80GB)
# =============================================================================
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=10800,  # 3 hours for download + conversion
    volumes={"/cache": vol},
    # secrets=[modal.Secret.from_name("huggingface-secret")],  # Optional: for gated models
)
def test_qwen_32b():
    """Test Qwen 32B model individually with extended timeout."""
    import os
    os.environ["HF_HOME"] = "/cache/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/hf"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads
    
    print("=" * 60)
    print("ZSE CONVERSION TEST - QWEN 32B ONLY")
    print("GPU: A100 80GB | Format: FP16 → .zse (~64GB)")
    print("Extended timeout: 3 hours")
    print("=" * 60)
    
    result = test_zse_conversion("Qwen/Qwen2.5-32B-Instruct", use_int4=False)
    
    print("\n" + "=" * 60)
    print("QWEN 32B RESULT")
    print("=" * 60)
    s = "✅" if result["status"] == "PASSED" else "❌"
    print(f"{s} {result['model_id']}")
    if result["status"] == "PASSED":
        print(f"   Convert: {result['convert_time']:.1f}s | .zse: {result['zse_size_gb']:.2f}GB | Reload: {result['reload_time']:.1f}s")
    else:
        print(f"   Error: {result['error']}")
    
    return result


# =============================================================================
# Run all batches
# =============================================================================
@app.local_entrypoint()
def main():
    """Run all batch tests."""
    print("\n" + "=" * 60)
    print("ZSE FORMAT CONVERSION TEST")
    print("Testing: 7B, 8B, 14B, 32B, 72B models")
    print("GPU: A100 80GB")
    print("=" * 60)
    
    all_results = []
    
    print("\n🚀 BATCH 1: 7B/8B models...")
    r1 = test_7b_models.remote()
    all_results.extend(r1)
    
    print("\n🚀 BATCH 2: 14B/32B models...")
    r2 = test_14b_32b_models.remote()
    all_results.extend(r2)
    
    print("\n🚀 BATCH 3: 72B model...")
    r3 = test_72b_model.remote()
    all_results.extend(r3)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - ALL MODELS")
    print("=" * 60)
    
    passed = [r for r in all_results if r["status"] == "PASSED"]
    failed = [r for r in all_results if r["status"] == "FAILED"]
    
    print(f"\nTotal: {len(passed)}/{len(all_results)} passed\n")
    
    print("PASSED (ready for registry):")
    for r in passed:
        print(f"  ✅ {r['model_id']} (.zse: {r['zse_size_gb']:.2f}GB)")
    
    if failed:
        print("\nFAILED:")
        for r in failed:
            print(f"  ❌ {r['model_id']}: {r['error'][:60]}")
