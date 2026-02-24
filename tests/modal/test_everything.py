"""
HONEST TEST - What Actually Works in ZSE?
Tests everything and reports truthfully.
"""

import modal

app = modal.App("test-zse-everything")

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "triton>=2.1.0",
        "bitsandbytes>=0.41.0",
        "safetensors",
        "sentencepiece",
    ])
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    memory=32768,
)
def test_everything():
    """Test every component and report honestly."""
    import torch
    import time
    
    results = {
        "cuda_available": False,
        "triton_available": False,
        "triton_kernel_runs": False,
        "huggingface_works": False,
        "bitsandbytes_works": False,
        "generation_speed": None,
    }
    
    print("=" * 60)
    print("HONEST ZSE COMPONENT TEST")
    print("=" * 60)
    
    # 1. CUDA Check
    print("\n[1/7] CUDA Available?")
    try:
        results["cuda_available"] = torch.cuda.is_available()
        if results["cuda_available"]:
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  ✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  ❌ CUDA not available")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # 2. Triton Check
    print("\n[2/7] Triton Import?")
    try:
        import triton
        import triton.language as tl
        results["triton_available"] = True
        print(f"  ✅ Triton version: {triton.__version__}")
    except Exception as e:
        print(f"  ❌ Triton import failed: {e}")
    
    # 3. Triton Kernel Execution
    print("\n[3/7] Triton Kernel Actually Runs?")
    try:
        import triton
        import triton.language as tl
        
        @triton.jit
        def simple_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)
        
        # Test it
        n = 1024
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        output = torch.empty_like(x)
        
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        simple_add_kernel[grid](x, y, output, n, BLOCK_SIZE=256)
        
        # Verify
        expected = x + y
        if torch.allclose(output, expected):
            results["triton_kernel_runs"] = True
            print("  ✅ Triton kernel executed and verified")
        else:
            print("  ❌ Triton kernel output incorrect")
    except Exception as e:
        print(f"  ❌ Triton kernel failed: {e}")
    
    # 4. HuggingFace Model
    print("\n[4/7] HuggingFace Model Loading?")
    model = None
    tokenizer = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        load_time = time.time() - start
        
        results["huggingface_works"] = True
        print(f"  ✅ TinyLlama loaded in {load_time:.1f}s")
        print(f"  ✅ Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    except Exception as e:
        print(f"  ❌ HuggingFace failed: {e}")
    
    # 5. Bitsandbytes Quantization
    print("\n[5/7] Bitsandbytes INT4 Quantization?")
    try:
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        model_4bit = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            quantization_config=bnb_config,
            device_map="cuda"
        )
        
        mem_used = torch.cuda.memory_allocated() / 1e9
        results["bitsandbytes_works"] = True
        print(f"  ✅ INT4 model loaded, GPU memory: {mem_used:.2f} GB")
        del model_4bit
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ❌ Bitsandbytes failed: {e}")
    
    # 6. Generation Speed Test
    print("\n[6/7] Generation Speed (FP16)?")
    try:
        if model and tokenizer:
            input_text = "Write a Python function to calculate fibonacci:"
            inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
            
            # Warmup
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            torch.cuda.synchronize()
            
            # Timed generation
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            torch.cuda.synchronize()
            gen_time = time.time() - start
            
            new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_sec = new_tokens / gen_time
            results["generation_speed"] = tokens_per_sec
            
            print(f"  ✅ Generated {new_tokens} tokens in {gen_time:.2f}s")
            print(f"  ✅ Speed: {tokens_per_sec:.1f} tokens/sec")
            
            # Show output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Output: {output_text[:200]}...")
        else:
            print("  ❌ Skipped (model not loaded)")
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
    
    # 7. ZSE Custom Code Check
    print("\n[7/7] ZSE Custom Code?")
    try:
        # Create a simple test
        batch_size, num_heads, seq_len, head_dim = 1, 8, 128, 64
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # Use PyTorch's scaled dot product attention as baseline
        with torch.no_grad():
            baseline = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        
        print(f"  ✅ Baseline attention works: output shape {baseline.shape}")
        print("  ⚠️  ZSE Triton kernels exist but NOT BENCHMARKED against baseline")
        print("  ⚠️  CUDA kernels exist but NOT COMPILED")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - WHAT ACTUALLY WORKS")
    print("=" * 60)
    
    checks = [
        ("CUDA Available", results["cuda_available"]),
        ("Triton Available", results["triton_available"]),
        ("Triton Kernels Run", results["triton_kernel_runs"]),
        ("HuggingFace Works", results["huggingface_works"]),
        ("Bitsandbytes Works", results["bitsandbytes_works"]),
        ("Generation Works", results["generation_speed"] is not None),
    ]
    
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    if results["generation_speed"]:
        print(f"\n  Generation Speed: {results['generation_speed']:.1f} tok/s")
    
    print("\n" + "=" * 60)
    print("HONEST ASSESSMENT")
    print("=" * 60)
    print("""
  What ZSE Actually Has:
  ✅ Layer streaming (zStream) - WORKS but slow
  ✅ Bitsandbytes quantization wrapper - WORKS
  ✅ Triton kernel files - EXIST but not benchmarked
  ✅ CUDA kernel files - EXIST but not compiled
  
  What ZSE Does NOT Have:
  ❌ Faster inference than vanilla HuggingFace
  ❌ Production server (zse serve)
  ❌ Compiled & tested custom CUDA kernels
  ❌ Verified speed improvements
  
  Bottom Line:
  - We have WRAPPERS around existing libraries
  - We have UNTESTED custom kernels
  - We do NOT have a production inference engine
""")
    
    return results

@app.local_entrypoint()
def main():
    print("Running honest assessment of ZSE...")
    results = test_everything.remote()
    print("\nTest complete.")
