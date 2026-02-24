"""
Test all models in the ZSE registry on Modal GPU.

Tests each curated model to verify it can:
1. Load successfully
2. Generate text
3. Produce valid output

Run all small models (fits in A10G 24GB):
    modal run tests/modal/test_model_registry.py::test_small_models

Run specific model:
    modal run tests/modal/test_model_registry.py::test_single_model --model "Qwen/Qwen2.5-3B-Instruct"

Run tiny models only (quick test):
    modal run tests/modal/test_model_registry.py::test_tiny_models
"""

import os
import modal

app = modal.App("zse-model-registry-test")

# Get ZSE root for mounting
ZSE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Base image with ZSE and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.40.0", 
        "safetensors>=0.4.0",
        "accelerate>=0.25.0",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "bitsandbytes>=0.43.0",  # For INT4/INT8 quantization
    ])
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)


# Models grouped by VRAM requirement
TINY_MODELS = [
    # < 4GB VRAM - can test many on A10G
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

SMALL_MODELS = [
    # 4-10GB VRAM - fits on A10G with INT8 (non-gated only)
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
]

MEDIUM_MODELS = [
    # 10-16GB VRAM - fits on A10G with care (non-gated only)
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
]

# These need A100 or larger (non-gated)
LARGE_MODELS = [
    "Qwen/Qwen2.5-14B-Instruct",
]

# Gated models - require HF authentication
GATED_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# These need A100 80GB
XLARGE_MODELS = [
    "Qwen/Qwen2.5-32B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

# Test prompt
TEST_PROMPT = "What is 2 + 2? Answer with just the number."


def test_model(model_id: str, quantization: str = "auto") -> dict:
    """Test a single model - load and generate."""
    import sys
    import time
    import torch
    
    # Add ZSE to path
    sys.path.insert(0, "/root/zse")
    
    result = {
        "model_id": model_id,
        "status": "unknown",
        "load_time": None,
        "generate_time": None,
        "output": None,
        "vram_gb": None,
        "error": None,
    }
    
    try:
        from zse.engine.orchestrator import IntelligenceOrchestrator
        
        print(f"\n{'='*60}")
        print(f"Testing: {model_id}")
        print(f"{'='*60}")
        
        # Load model using the intelligent loader
        start = time.time()
        if quantization == "auto":
            orch = IntelligenceOrchestrator.auto(model_id)
        elif quantization == "int8":
            orch = IntelligenceOrchestrator.min_memory(model_id)
        else:
            orch = IntelligenceOrchestrator.auto(model_id)
        
        load_time = time.time() - start
        result["load_time"] = round(load_time, 2)
        
        # Get VRAM usage
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            result["vram_gb"] = round(vram, 2)
            print(f"  Loaded in {load_time:.1f}s, VRAM: {vram:.2f} GB")
        
        # Generate using streaming
        start = time.time()
        output = ""
        for chunk in orch.generate(TEST_PROMPT, max_tokens=20, temperature=0.1, stream=True):
            output += chunk
        generate_time = time.time() - start
        result["generate_time"] = round(generate_time, 2)
        result["output"] = output[:100] if output else None
        
        print(f"  Generated in {generate_time:.2f}s")
        print(f"  Output: {output[:80]}...")
        
        # Verify output contains something reasonable
        if output and len(output) > 0:
            result["status"] = "passed"
            print(f"  ✅ PASSED")
        else:
            result["status"] = "failed"
            result["error"] = "Empty output"
            print(f"  ❌ FAILED - Empty output")
        
        # Cleanup
        del orch
        torch.cuda.empty_cache()
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:200]
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def print_summary(results: list):
    """Print test summary."""
    print("\n" + "="*70)
    print("MODEL REGISTRY TEST SUMMARY")
    print("="*70)
    
    passed = [r for r in results if r["status"] == "passed"]
    failed = [r for r in results if r["status"] == "failed"]
    errors = [r for r in results if r["status"] == "error"]
    
    print(f"\n✅ Passed: {len(passed)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    print(f"⚠️  Errors: {len(errors)}/{len(results)}")
    
    print("\nDetailed Results:")
    print("-"*70)
    print(f"{'Model':<45} {'Status':<10} {'Load':<8} {'VRAM':<8}")
    print("-"*70)
    
    for r in results:
        model = r["model_id"][:44]
        status = "✅" if r["status"] == "passed" else ("❌" if r["status"] == "failed" else "⚠️")
        load = f"{r['load_time']:.1f}s" if r["load_time"] else "-"
        vram = f"{r['vram_gb']:.1f}GB" if r["vram_gb"] else "-"
        print(f"{model:<45} {status:<10} {load:<8} {vram:<8}")
        
        if r["error"]:
            print(f"    Error: {r['error'][:60]}...")
    
    print("-"*70)
    
    # Return pass/fail for CI
    return len(failed) == 0 and len(errors) == 0


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,  # 10 min for tiny models
)
def test_tiny_models():
    """Test tiny models (< 2GB VRAM each)."""
    print("="*70)
    print("TESTING TINY MODELS (< 2GB VRAM)")
    print("="*70)
    
    results = []
    for model_id in TINY_MODELS:
        result = test_model(model_id, quantization="auto")
        results.append(result)
    
    success = print_summary(results)
    return {"success": success, "results": results}


@app.function(
    image=image,
    gpu="A10G",
    timeout=900,  # 15 min for small models
)
def test_small_models():
    """Test small models (2-6GB VRAM each)."""
    print("="*70)
    print("TESTING SMALL MODELS (2-6GB VRAM)")
    print("="*70)
    
    # First test tiny models
    results = []
    for model_id in TINY_MODELS:
        result = test_model(model_id, quantization="auto")
        results.append(result)
    
    # Then small models
    for model_id in SMALL_MODELS:
        result = test_model(model_id, quantization="int8")
        results.append(result)
    
    success = print_summary(results)
    return {"success": success, "results": results}


@app.function(
    image=image,
    gpu="A10G",
    timeout=1200,  # 20 min for medium models
)
def test_medium_models():
    """Test medium models (6-16GB VRAM) with INT8."""
    print("="*70)
    print("TESTING MEDIUM MODELS (6-16GB VRAM)")
    print("="*70)
    
    results = []
    for model_id in MEDIUM_MODELS:
        result = test_model(model_id, quantization="int8")
        results.append(result)
    
    success = print_summary(results)
    return {"success": success, "results": results}


@app.function(
    image=image,
    gpu="A100",  # Need A100 for large models
    timeout=1800,  # 30 min
)
def test_large_models():
    """Test large models (16-26GB VRAM) - requires A100."""
    import os
    
    print("="*70)
    print("TESTING LARGE MODELS (16-26GB VRAM)")
    print("="*70)
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("✓ HuggingFace token available for gated models")
    else:
        print("⚠️  No HF token - gated models (Llama) will be skipped")
    
    results = []
    for model_id in LARGE_MODELS:
        # Skip gated models if no token
        if "meta-llama" in model_id and not hf_token:
            print(f"\n⚠️  Skipping {model_id} (gated, no HF token)")
            results.append({
                "model_id": model_id,
                "status": "skipped",
                "error": "Gated model - HF token required"
            })
            continue
        
        result = test_model(model_id, quantization="int8")
        results.append(result)
    
    success = print_summary(results)
    return {"success": success, "results": results}


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
def test_single_model(model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", quant: str = "auto"):
    """Test a single specific model."""
    print("="*70)
    print(f"TESTING SINGLE MODEL: {model}")
    print("="*70)
    
    result = test_model(model, quantization=quant)
    print_summary([result])
    return result


@app.function(
    image=image,
    gpu="A10G", 
    timeout=1800,  # 30 min for all
)
def test_all_compatible():
    """
    Test all models that fit on A10G (24GB).
    
    This tests TINY + SMALL + MEDIUM models with appropriate quantization.
    Large models (A100 required) are skipped.
    """
    print("="*70)
    print("TESTING ALL A10G-COMPATIBLE MODELS")
    print("="*70)
    
    results = []
    
    # Tiny models - FP16/auto
    print("\n📦 TINY MODELS (auto quantization)")
    for model_id in TINY_MODELS:
        result = test_model(model_id, quantization="auto")
        results.append(result)
    
    # Small models - INT8
    print("\n📦 SMALL MODELS (INT8 quantization)")
    for model_id in SMALL_MODELS:
        result = test_model(model_id, quantization="int8")
        results.append(result)
    
    # Medium models - INT8 with care
    print("\n📦 MEDIUM MODELS (INT8 quantization)")
    for model_id in MEDIUM_MODELS:
        result = test_model(model_id, quantization="int8")
        results.append(result)
    
    success = print_summary(results)
    
    # Mark verified models
    verified_models = [r["model_id"] for r in results if r["status"] == "passed"]
    print(f"\n✅ Verified models ({len(verified_models)}):")
    for m in verified_models:
        print(f"   - {m}")
    
    return {"success": success, "results": results, "verified": verified_models}


@app.local_entrypoint()
def main(
    tier: str = "tiny",
    model: str = None,
):
    """
    Test models from registry.
    
    Args:
        tier: tiny, small, medium, large, or all
        model: specific model ID to test
    """
    if model:
        result = test_single_model.remote(model=model)
        print(f"\nResult: {result['status']}")
    elif tier == "tiny":
        result = test_tiny_models.remote()
    elif tier == "small":
        result = test_small_models.remote()
    elif tier == "medium":
        result = test_medium_models.remote()
    elif tier == "large":
        result = test_large_models.remote()
    elif tier == "all":
        result = test_all_compatible.remote()
    else:
        print(f"Unknown tier: {tier}")
        print("Valid: tiny, small, medium, large, all")
        return
    
    if result.get("success"):
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n❌ SOME TESTS FAILED")
