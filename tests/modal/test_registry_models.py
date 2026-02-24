"""
Test all curated registry models on Modal GPU.

Run specific batches:
  modal run tests/modal/test_registry_models.py::test_7b_models
  modal run tests/modal/test_registry_models.py::test_8b_models
  modal run tests/modal/test_registry_models.py::test_14b_models
  modal run tests/modal/test_registry_models.py::test_32b_models
  modal run tests/modal/test_registry_models.py::test_70b_models
"""

import modal
import time
from pathlib import Path

# Modal app
app = modal.App("zse-registry-model-tests")

# Get ZSE root
ZSE_ROOT = str(Path(__file__).parent.parent.parent)

# Base image with dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "protobuf",
        "bitsandbytes>=0.42.0",
        "httpx",
    )
    .run_commands("pip install flash-attn --no-build-isolation || true")
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)


def test_model(model_id: str, quantization: str = "auto", max_new_tokens: int = 20) -> dict:
    """Test a single model: load + generate."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    result = {
        "model_id": model_id,
        "status": "unknown",
        "load_time": 0,
        "generate_time": 0,
        "output": "",
        "error": None,
        "vram_gb": 0,
    }
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print(f"\n{'='*60}")
        print(f"Testing: {model_id}")
        print(f"Quantization: {quantization}")
        print(f"{'='*60}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Load model with appropriate settings
        print("Loading model...")
        load_start = time.time()
        
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if quantization == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        
        load_time = time.time() - load_start
        result["load_time"] = round(load_time, 2)
        print(f"✓ Model loaded in {load_time:.1f}s")
        
        # Check VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1024**3
            result["vram_gb"] = round(vram_gb, 2)
            print(f"✓ VRAM used: {vram_gb:.2f} GB")
        
        # Generate
        print("Generating...")
        prompt = "Hello, my name is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - gen_start
        result["generate_time"] = round(gen_time, 2)
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result["output"] = output_text[:100]
        print(f"✓ Generated in {gen_time:.2f}s: {output_text[:50]}...")
        
        result["status"] = "passed"
        print(f"\n✅ {model_id} PASSED")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)[:200]
        print(f"\n❌ {model_id} FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    return result


# =============================================================================
# 7B Models (A10G - 24GB)
# =============================================================================
@app.function(
    image=base_image,
    gpu="A10G",
    timeout=1800,
)
def test_7b_models():
    """Test 7B parameter models."""
    models = [
        ("Qwen/Qwen2.5-7B-Instruct", "auto"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "auto"),
        ("Qwen/Qwen2.5-Coder-7B-Instruct", "auto"),
        ("deepseek-ai/deepseek-coder-6.7b-instruct", "auto"),
    ]
    
    print("="*60)
    print("ZSE REGISTRY MODEL TEST - 7B MODELS")
    print("="*60)
    
    results = []
    for model_id, quant in models:
        result = test_model(model_id, quant)
        results.append(result)
        
        # Clear GPU memory between models
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - 7B MODELS")
    print("="*60)
    passed = sum(1 for r in results if r["status"] == "passed")
    print(f"Total: {passed}/{len(results)} passed\n")
    
    for r in results:
        status = "✅" if r["status"] == "passed" else "❌"
        print(f"{status} {r['model_id']}")
        if r["status"] == "passed":
            print(f"   Load: {r['load_time']}s, Gen: {r['generate_time']}s, VRAM: {r['vram_gb']}GB")
        else:
            print(f"   Error: {r['error'][:80]}...")
    
    return results


# =============================================================================
# 8B Models (A10G - 24GB)
# =============================================================================
@app.function(
    image=base_image,
    gpu="A10G",
    timeout=1800,
)
def test_8b_models():
    """Test 8B parameter models."""
    models = [
        ("meta-llama/Llama-3.1-8B-Instruct", "auto"),
        ("meta-llama/Llama-3.2-3B-Instruct", "auto"),  # Including 3B Llama here
        ("google/gemma-2-9b-it", "auto"),
    ]
    
    print("="*60)
    print("ZSE REGISTRY MODEL TEST - 8B MODELS")
    print("="*60)
    
    results = []
    for model_id, quant in models:
        result = test_model(model_id, quant)
        results.append(result)
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - 8B MODELS")
    print("="*60)
    passed = sum(1 for r in results if r["status"] == "passed")
    print(f"Total: {passed}/{len(results)} passed\n")
    
    for r in results:
        status = "✅" if r["status"] == "passed" else "❌"
        print(f"{status} {r['model_id']}")
        if r["status"] == "passed":
            print(f"   Load: {r['load_time']}s, Gen: {r['generate_time']}s, VRAM: {r['vram_gb']}GB")
        else:
            print(f"   Error: {r['error'][:80]}...")
    
    return results


# =============================================================================
# 14B Models (A10G with INT8)
# =============================================================================
@app.function(
    image=base_image,
    gpu="A10G",
    timeout=1800,
)
def test_14b_models():
    """Test 14B parameter models with INT8 quantization."""
    models = [
        ("Qwen/Qwen2.5-14B-Instruct", "int8"),
    ]
    
    print("="*60)
    print("ZSE REGISTRY MODEL TEST - 14B MODELS")
    print("="*60)
    
    results = []
    for model_id, quant in models:
        result = test_model(model_id, quant)
        results.append(result)
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - 14B MODELS")
    print("="*60)
    passed = sum(1 for r in results if r["status"] == "passed")
    print(f"Total: {passed}/{len(results)} passed\n")
    
    for r in results:
        status = "✅" if r["status"] == "passed" else "❌"
        print(f"{status} {r['model_id']}")
        if r["status"] == "passed":
            print(f"   Load: {r['load_time']}s, Gen: {r['generate_time']}s, VRAM: {r['vram_gb']}GB")
        else:
            print(f"   Error: {r['error'][:80]}...")
    
    return results


# =============================================================================
# 32B Models (A100 40GB with INT8)
# =============================================================================
@app.function(
    image=base_image,
    gpu="A100",
    timeout=2400,
)
def test_32b_models():
    """Test 32B parameter models."""
    models = [
        ("Qwen/Qwen2.5-32B-Instruct", "int8"),
    ]
    
    print("="*60)
    print("ZSE REGISTRY MODEL TEST - 32B MODELS")
    print("="*60)
    
    results = []
    for model_id, quant in models:
        result = test_model(model_id, quant)
        results.append(result)
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - 32B MODELS")
    print("="*60)
    passed = sum(1 for r in results if r["status"] == "passed")
    print(f"Total: {passed}/{len(results)} passed\n")
    
    for r in results:
        status = "✅" if r["status"] == "passed" else "❌"
        print(f"{status} {r['model_id']}")
        if r["status"] == "passed":
            print(f"   Load: {r['load_time']}s, Gen: {r['generate_time']}s, VRAM: {r['vram_gb']}GB")
        else:
            print(f"   Error: {r['error'][:80]}...")
    
    return results


# =============================================================================
# 70B+ Models (A100 80GB with INT4)
# =============================================================================
@app.function(
    image=base_image,
    gpu="A100-80GB",
    timeout=3600,
)
def test_70b_models():
    """Test 70B+ parameter models with INT4 quantization."""
    models = [
        ("meta-llama/Llama-3.1-70B-Instruct", "int4"),
        ("Qwen/Qwen2.5-72B-Instruct", "int4"),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "int4"),
    ]
    
    print("="*60)
    print("ZSE REGISTRY MODEL TEST - 70B+ MODELS")
    print("="*60)
    
    results = []
    for model_id, quant in models:
        result = test_model(model_id, quant)
        results.append(result)
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - 70B+ MODELS")
    print("="*60)
    passed = sum(1 for r in results if r["status"] == "passed")
    print(f"Total: {passed}/{len(results)} passed\n")
    
    for r in results:
        status = "✅" if r["status"] == "passed" else "❌"
        print(f"{status} {r['model_id']}")
        if r["status"] == "passed":
            print(f"   Load: {r['load_time']}s, Gen: {r['generate_time']}s, VRAM: {r['vram_gb']}GB")
        else:
            print(f"   Error: {r['error'][:80]}...")
    
    return results
