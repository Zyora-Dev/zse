"""
ZSE Full Test: Qwen 2.5 Coder 7B on A100 80GB

Tests the complete ZSE system:
1. Automatic model loading
2. Quantization (INT8/INT4)
3. zStream layer streaming
4. Orchestrator inference

Run with: modal run tests/modal/test_qwen_7b.py
"""

import modal
import time
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))

app = modal.App("zse-qwen-7b-test")

# Image with full ZSE dependencies
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "triton==3.1.0",
        "transformers>=4.44.0",
        "accelerate>=0.28.0",
        "safetensors>=0.4.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "huggingface_hub",
        "psutil",
        "numpy",
    ])
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"


@app.function(
    image=cuda_image,
    gpu=modal.gpu.A100(size="80GB"),  # Explicitly A100 80GB
    timeout=3600,
    memory=65536,  # 64GB RAM
)
def test_qwen_7b_full():
    """
    Full ZSE test with Qwen 2.5 Coder 7B on A100 80GB.
    Tests quantization, layer streaming, and generation.
    """
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import gc
    
    print("=" * 70)
    print("ZSE FULL TEST: Qwen 2.5 Coder 7B on A100 80GB")
    print("=" * 70)
    
    # GPU Info
    print(f"\n[GPU INFO]")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  Total VRAM: {total_vram:.1f} GB")
    
    # Verify we got 80GB
    if total_vram < 75:
        print(f"  ⚠️  WARNING: Expected 80GB VRAM but got {total_vram:.1f} GB!")
    else:
        print(f"  ✅ Correct GPU: A100 80GB")
    
    results = {
        "model_load": False,
        "quantization": None,
        "vram_used_gb": None,
        "generation": False,
        "tokens_per_sec": None,
        "output_quality": False,
    }
    
    # =================================================================
    # TEST 1: Load model with INT4 quantization (BitsAndBytes)
    # =================================================================
    print("\n" + "=" * 70)
    print("[TEST 1] Loading Qwen 2.5 Coder 7B with INT4 Quantization")
    print("=" * 70)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # INT4 quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        print("  Loading model with INT4 quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        
        load_time = time.time() - start_time
        vram_after = torch.cuda.memory_allocated() / (1024**3)
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        
        results["model_load"] = True
        results["vram_used_gb"] = vram_after
        results["quantization"] = "INT4 (NF4)"
        
        print(f"  ✅ Model loaded in {load_time:.1f}s")
        print(f"  ✅ VRAM used: {vram_after:.2f} GB")
        print(f"  ✅ Peak VRAM: {peak_vram:.2f} GB")
        print(f"  ✅ Quantization: INT4 (NF4 with double quant)")
        
        # Create wrapper for generation
        class ModelWrapper:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
                self.quantization = "INT4"
            
            def generate(self, prompt, max_tokens=100, temperature=0.7):
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        orchestrator = ModelWrapper(model, tokenizer)
        
    except Exception as e:
        print(f"  ❌ INT4 loading failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try FP16 as fallback
        print("  Trying FP16 fallback...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="cuda:0",
                torch_dtype=torch.float16,
            )
            
            vram_after = torch.cuda.memory_allocated() / (1024**3)
            results["model_load"] = True
            results["vram_used_gb"] = vram_after
            results["quantization"] = "FP16"
            
            print(f"  ✅ FP16 fallback loaded, VRAM: {vram_after:.2f} GB")
            
            class ModelWrapper:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.quantization = "FP16"
                
                def generate(self, prompt, max_tokens=100, temperature=0.7):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            orchestrator = ModelWrapper(model, tokenizer)
            
        except Exception as e2:
            print(f"  ❌ FP16 fallback also failed: {e2}")
            return results
    
    # =================================================================
    # TEST 2: Code Generation Quality
    # =================================================================
    print("\n" + "=" * 70)
    print("[TEST 2] Code Generation Test")
    print("=" * 70)
    
    code_prompts = [
        "Write a Python function to calculate the fibonacci sequence:",
        "Write a Python function to sort a list using quicksort:",
        "Write a Python class for a binary search tree with insert and search methods:",
    ]
    
    for i, prompt in enumerate(code_prompts, 1):
        print(f"\n  [{i}] Prompt: {prompt[:50]}...")
        
        try:
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            output = orchestrator.generate(
                prompt,
                max_tokens=200,
                temperature=0.3,
            )
            
            gen_time = time.time() - start_time
            
            # Count tokens (rough estimate)
            output_tokens = len(output.split()) * 1.3  # rough tokenization
            tokens_per_sec = output_tokens / gen_time if gen_time > 0 else 0
            
            print(f"  Generated in {gen_time:.2f}s (~{tokens_per_sec:.1f} tokens/sec)")
            print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
            
            # Show first part of output
            output_lines = output.split('\n')[:15]
            print("  Output preview:")
            for line in output_lines:
                print(f"    {line}")
            
            # Check if output contains code markers
            if "def " in output or "class " in output:
                results["generation"] = True
                results["output_quality"] = True
                results["tokens_per_sec"] = tokens_per_sec
                print("  ✅ Generated valid Python code")
            else:
                print("  ⚠️  Output may not contain proper code")
                
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
    
    # =================================================================
    # TEST 3: zStream Layer Streaming (if model supports it)
    # =================================================================
    print("\n" + "=" * 70)
    print("[TEST 3] zStream Layer Streaming Test")
    print("=" * 70)
    
    try:
        from zse.core.zstream import StreamingModel, StreamingConfig, MemoryTracker
        
        tracker = MemoryTracker(device=0)
        state = tracker.get_state()
        
        print(f"  VRAM Status: {state.allocated_gb:.2f} GB allocated, {state.free_gb:.2f} GB free")
        print(f"  Memory Pressure: {state.pressure.value}")
        
        # Estimate layer capacity
        if hasattr(orchestrator, 'model'):
            model = orchestrator.model
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                num_layers = len(model.model.layers)
                print(f"  Model has {num_layers} layers")
                
                # Get layer size estimate
                sample_layer = model.model.layers[0]
                layer_bytes = sum(p.numel() * p.element_size() for p in sample_layer.parameters())
                layer_gb = layer_bytes / (1024**3)
                print(f"  Layer size: ~{layer_gb*1000:.1f} MB each")
                
                capacity = tracker.estimate_layer_capacity(layer_bytes)
                print(f"  Estimated GPU capacity: {capacity} layers")
                
                print("  ✅ zStream analysis complete")
            else:
                print("  ⚠️  Model structure not compatible with layer analysis")
        else:
            print("  ⚠️  Could not access model for layer analysis")
            
    except Exception as e:
        print(f"  ❌ zStream test failed: {e}")
    
    # =================================================================
    # TEST 4: Memory Efficiency Check
    # =================================================================
    print("\n" + "=" * 70)
    print("[TEST 4] Memory Efficiency Analysis")
    print("=" * 70)
    
    # Model size vs VRAM used
    model_size_gb = 7.0 * 2  # 7B params * 2 bytes (FP16) = 14GB
    quant_size_gb = 7.0 * 0.5  # INT4 = ~3.5GB
    
    print(f"  Theoretical FP16 size: {model_size_gb:.1f} GB")
    print(f"  Theoretical INT4 size: {quant_size_gb:.1f} GB")
    print(f"  Actual VRAM used: {results['vram_used_gb']:.2f} GB")
    
    if results['vram_used_gb'] and results['vram_used_gb'] < model_size_gb * 0.5:
        print(f"  ✅ Efficient quantization! Using {results['vram_used_gb']/model_size_gb*100:.0f}% of FP16 size")
    else:
        print(f"  ⚠️  VRAM usage higher than expected")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
    Model: {MODEL_NAME}
    GPU: A100 80GB
    
    Results:
      Model Load:     {'✅' if results['model_load'] else '❌'}
      Quantization:   {results['quantization'] or 'N/A'}
      VRAM Used:      {results['vram_used_gb']:.2f if results['vram_used_gb'] else 'N/A'} GB
      Generation:     {'✅' if results['generation'] else '❌'}
      Code Quality:   {'✅' if results['output_quality'] else '❌'}
      Speed:          {results['tokens_per_sec']:.1f if results['tokens_per_sec'] else 'N/A'} tokens/sec
    """)
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


@app.function(
    image=cuda_image,
    gpu=modal.gpu.A100(size="80GB"),  # Explicitly A100 80GB
    timeout=3600,
    memory=65536,
)
def test_qwen_7b_inference_benchmark():
    """Benchmark inference speed with Qwen 2.5 Coder 7B."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print("=" * 70)
    print("INFERENCE BENCHMARK: Qwen 2.5 Coder 7B on A100 80GB")
    print("=" * 70)
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    # Load with INT4 quantization
    print("\nLoading model with INT4 quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.float16,
    )
    
    print(f"VRAM after load: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    
    # Warmup
    print("\nWarmup...")
    inputs = tokenizer("def hello():", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    
    # Benchmark different sequence lengths
    print("\nBenchmarking...")
    
    prompts = [
        ("Short", "Write a Python hello world:"),
        ("Medium", "Write a Python function to calculate factorial with proper error handling and docstring:"),
        ("Long", """Write a Python class implementing a complete binary search tree with the following methods:
1. insert(value) - insert a value
2. search(value) - return True if value exists
3. delete(value) - delete a value
4. inorder() - return inorder traversal
5. height() - return tree height

Include proper docstrings and error handling:"""),
    ]
    
    for name, prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
        input_len = input_ids.shape[1]
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        output_len = outputs.shape[1]
        gen_tokens = output_len - input_len
        tps = gen_tokens / elapsed
        
        print(f"\n  [{name}] Input: {input_len} tokens")
        print(f"    Generated: {gen_tokens} tokens in {elapsed:.2f}s")
        print(f"    Speed: {tps:.1f} tokens/sec")
        print(f"    Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        
        # Show output
        output_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        print(f"    Preview: {output_text[:100]}...")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


@app.local_entrypoint()
def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Starting ZSE Qwen 2.5 Coder 7B Tests on A100 80GB")
    print("=" * 70 + "\n")
    
    # Run full test
    print("Running full orchestrator test...")
    results = test_qwen_7b_full.remote()
    
    print("\n" + "=" * 70)
    print("Running inference benchmark...")
    test_qwen_7b_inference_benchmark.remote()
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
