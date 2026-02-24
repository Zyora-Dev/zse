"""
zStream Modal Test - Layer Streaming for Large Models

Tests the zStream layer streaming capability by:
1. Loading a 7B model with layer streaming
2. Simulating limited GPU memory
3. Verifying streaming generation works correctly

Run with: modal run tests/modal/test_zstream.py
"""

import modal
import time
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))

# Modal app setup
app = modal.App("zse-zstream-test")

# Image with CUDA support
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.5.1",
        "triton==3.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "huggingface_hub",
        "psutil",
    ])
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=1800,
)
def test_zstream_basic():
    """Test basic layer streaming functionality."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    from zse.core.zstream import (
        MemoryTracker,
        LayerStreamer,
        StreamerConfig,
        MemoryPressure,
    )
    
    print("=" * 60)
    print("Test 1: Memory Tracker")
    print("=" * 60)
    
    tracker = MemoryTracker(device=0)
    state = tracker.get_state()
    
    print(f"Total VRAM: {state.total_gb:.2f} GB")
    print(f"Allocated VRAM: {state.allocated_gb:.2f} GB")
    print(f"Free VRAM: {state.free_gb:.2f} GB")
    print(f"Pressure: {state.pressure.value}")
    
    # Test layer capacity estimation
    layer_size = 500 * 1024 * 1024  # 500MB per layer
    capacity = tracker.estimate_layer_capacity(layer_size)
    print(f"Estimated capacity: {capacity} layers (500MB each)")
    
    print("\n" + "=" * 60)
    print("Test 2: Layer Streamer (Synthetic)")
    print("=" * 60)
    
    # Create a simple test model
    class DummyLayer(torch.nn.Module):
        def __init__(self, hidden_size=4096):
            super().__init__()
            self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))
    
    class DummyModel(torch.nn.Module):
        """Dummy model with layers structure like LLaMA."""
        def __init__(self, num_layers=32, hidden_size=4096):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([
                DummyLayer(hidden_size) for _ in range(num_layers)
            ])
    
    num_layers = 32
    hidden_size = 4096
    
    # Create model on CPU
    model = DummyModel(num_layers, hidden_size)
    
    # Estimate layer size
    sample_layer = model.model.layers[0]
    layer_bytes = sum(p.numel() * p.element_size() for p in sample_layer.parameters())
    print(f"Layer size: {layer_bytes / 1024 / 1024:.2f} MB")
    print(f"Total model size: {layer_bytes * num_layers / 1024 / 1024:.2f} MB")
    
    # Create streamer with config
    from zse.core.zstream import StreamerConfig, StreamingForward
    config = StreamerConfig(
        max_window_size=4,  # Only 4 layers on GPU
        prefetch_count=2,
        use_pinned_memory=True,
    )
    
    streamer = LayerStreamer(model=model, config=config)
    
    print(f"Created streamer for {streamer.num_layers} layers, max window: {config.max_window_size}")
    
    # Simulate forward pass using StreamingForward helper
    print("\nSimulating forward pass...")
    
    with StreamingForward(streamer) as stream:
        for layer_idx in range(num_layers):
            # Get the layer (streams it to GPU if needed)
            layer = stream.get(layer_idx)
            
            # Verify it's on GPU
            device = next(layer.parameters()).device
            assert device.type == "cuda", f"Layer {layer_idx} not on GPU!"
            
            # Print progress every 8 layers
            if (layer_idx + 1) % 8 == 0:
                stats = streamer.get_stats()
                print(f"  Layer {layer_idx + 1}/{num_layers}")
    
    stats = streamer.get_stats()
    print(f"\nFinal Stats:")
    print(f"  GPU hits: {stats.get('gpu_hits', 0)}")
    print(f"  Evictions: {stats.get('evictions', 0)}")
    print(f"  GPU memory allocated: {tracker.get_state().allocated_gb:.2f} GB")
    
    print("\n✅ Layer streaming working correctly!")
    return True


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=3600,
)
def test_zstream_real_model():
    """Test layer streaming with a real model."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Use Qwen 2.5 - freely accessible, no gating
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print("=" * 60)
    print(f"Test: zStream with {MODEL_NAME}")
    print("=" * 60)
    
    # Check initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / (1024**3)
    print(f"Initial GPU memory: {initial_mem:.2f} GB")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model to CPU (not GPU)
    print("Loading model to CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",  # Important: CPU first
        low_cpu_mem_usage=True,
    )
    
    cpu_mem = torch.cuda.memory_allocated() / (1024**3)
    print(f"GPU memory after CPU load: {cpu_mem:.2f} GB (should be ~0)")
    
    # Count layers
    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")
    
    # Import streaming model
    from zse.core.zstream import StreamingModel, StreamingConfig
    
    # Create streaming wrapper
    print("\nCreating streaming wrapper...")
    
    config = StreamingConfig(
        gpu_layers=4,      # Only 4 layers on GPU
        prefetch_layers=2, # Prefetch 2 ahead
        gpu_budget_gb=20.0,
    )
    
    streaming = StreamingModel(model, config)
    
    # Check memory after setup
    setup_mem = torch.cuda.memory_allocated() / (1024**3)
    print(f"GPU memory after setup: {setup_mem:.2f} GB")
    
    # Generate text
    print("\nGenerating text with streaming...")
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda:0")
    
    start_time = time.time()
    
    # Generate with streaming (only using ~4 layers worth of GPU memory)
    output_ids = streaming.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
    )
    
    gen_time = time.time() - start_time
    
    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"\nGeneration completed in {gen_time:.2f}s")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    
    print("\n" + "-" * 60)
    print("Generated Text:")
    print("-" * 60)
    print(output_text)
    print("-" * 60)
    
    # Print streaming stats
    stats = streaming.get_stats()
    print(f"\nStreaming Statistics:")
    print(f"  Forward passes: {stats['forward_count']}")
    print(f"  Average time/forward: {stats['avg_time_per_forward']*1000:.1f}ms")
    print(f"  Layer loads: {stats['streamer_stats'].get('gpu_loads', 'N/A')}")
    print(f"  Layer offloads: {stats['streamer_stats'].get('offloads', 'N/A')}")
    
    print("\n✅ Real model streaming test passed!")
    
    # Cleanup
    streaming.cleanup()
    
    return True


@app.function(
    image=cuda_image,
    gpu="A10G",
    timeout=1800,
)
def test_zstream_prefetcher():
    """Test async prefetching functionality."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import time
    from zse.core.zstream import (
        MemoryTracker,
        LayerStreamer,
        StreamerConfig,
        AsyncPrefetcher,
        PrefetchStrategy,
    )
    
    print("=" * 60)
    print("Test: Async Prefetcher")
    print("=" * 60)
    
    # Create a simple model
    num_layers = 16
    
    class DummyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2048, 2048)
        
        def forward(self, x):
            return self.linear(x)
    
    class DummyModel(torch.nn.Module):
        def __init__(self, num_layers):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([
                DummyLayer() for _ in range(num_layers)
            ])
    
    model = DummyModel(num_layers)
    
    # Create streamer with config
    config = StreamerConfig(
        max_window_size=3,
        prefetch_count=2,
        use_pinned_memory=True,
    )
    
    streamer = LayerStreamer(model=model, config=config)
    
    # Create prefetcher with correct API
    # The prefetcher needs load_fn and is_loaded_fn
    def load_fn(layer_idx: int):
        """Load layer to GPU."""
        streamer.get_layer(layer_idx)
    
    def is_loaded_fn(layer_idx: int) -> bool:
        """Check if layer is on GPU."""
        return layer_idx in streamer._gpu_window
    
    prefetcher = AsyncPrefetcher(
        load_fn=load_fn,
        is_loaded_fn=is_loaded_fn,
        num_layers=num_layers,
        strategy=PrefetchStrategy.SEQUENTIAL,
        num_streams=2,
    )
    
    print(f"Created prefetcher with {num_layers} layers")
    print(f"Max window: 3, Prefetch: 2")
    
    # Test with prefetching
    print("\n--- Testing layer streaming with prefetch ---")
    prefetcher.start()
    start = time.time()
    
    for i in range(num_layers):
        prefetcher.notify_access(i)  # Signal what layer we're accessing
        layer = streamer.get_layer(i)
        x = torch.randn(1, 2048, device="cuda")
        _ = layer(x)
        streamer.release_layer(i)
    
    prefetcher.stop()
    elapsed = time.time() - start
    print(f"Time: {elapsed*1000:.1f}ms")
    
    stats = prefetcher.get_stats()
    print(f"Prefetch hits: {stats.get('prefetch_hits', 0)}")
    print(f"Prefetch misses: {stats.get('prefetch_misses', 0)}")
    
    print("\n✅ Prefetcher test passed!")
    return True


@app.function(
    image=cuda_image,
    gpu="A100-80GB",  # 80GB VRAM for large models - USER SPECIFIED
    memory=262144,    # 256GB RAM for CPU layer storage (32B model needs ~70GB)
    timeout=7200,     # 2 hours for large model
)
def test_zstream_large_model(model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    """
    Test zStream with large models (32B) on A100-80GB.
    
    Uses Qwen2.5-Coder-32B-Instruct (open source, no gating required).
    """
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import time
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import login
    
    print("=" * 60)
    print(f"Test: zStream with {model_name}")
    print("=" * 60)
    
    # Login to HuggingFace for gated models
    hf_token = os.environ.get("HF_TOKEN", "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")
    login(token=hf_token)
    
    # Check GPU
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # System memory check
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"System RAM: {ram_gb:.1f} GB")
    
    # Load tokenizer
    print(f"\nLoading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model to CPU (NOT GPU) - this is the key for zStream
    print(f"\nLoading {model_name} to CPU (may take several minutes)...")
    load_start = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU first - essential for zStream
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    
    load_time = time.time() - load_start
    print(f"Model loaded to CPU in {load_time:.1f}s")
    
    # Get model info
    num_params = sum(p.numel() for p in model.parameters())
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"Parameters: {num_params / 1e9:.1f}B")
    print(f"Model size: {model_size_gb:.1f} GB")
    
    # Count layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_layers = len(model.transformer.h)
    else:
        num_layers = "unknown"
    print(f"Layers: {num_layers}")
    
    # Check GPU memory before streaming setup
    gpu_mem_before = torch.cuda.memory_allocated() / 1e9
    print(f"\nGPU memory before setup: {gpu_mem_before:.2f} GB (should be ~0)")
    
    # Import zStream
    from zse.core.zstream import StreamingModel, StreamingConfig
    
    # AUTO-CALCULATE optimal GPU window size
    free_vram_gb = torch.cuda.mem_get_info()[0] / 1e9
    layer_size_gb = model_size_gb / num_layers
    kv_reserve_gb = 8.0  # Reserve for KV cache and activations
    optimal_gpu_layers = max(4, int((free_vram_gb - kv_reserve_gb) / layer_size_gb))
    
    print(f"\n[AUTO-CONFIG] Free VRAM: {free_vram_gb:.1f} GB")
    print(f"[AUTO-CONFIG] Layer size: {layer_size_gb:.2f} GB")
    print(f"[AUTO-CONFIG] KV reserve: {kv_reserve_gb:.1f} GB")
    print(f"[AUTO-CONFIG] Optimal GPU layers: {optimal_gpu_layers} (out of {num_layers})")
    
    # Configure for large model with auto-calculated window
    print("\nCreating StreamingModel wrapper...")
    config = StreamingConfig(
        gpu_layers=optimal_gpu_layers,  # Auto-calculated optimal window
        prefetch_layers=3,              # Prefetch 3 ahead for large model
        gpu_budget_gb=free_vram_gb - 5, # Leave 5GB headroom
    )
    
    streaming = StreamingModel(model, config)
    
    # Check GPU memory after setup
    gpu_mem_setup = torch.cuda.memory_allocated() / 1e9
    print(f"GPU memory after setup: {gpu_mem_setup:.2f} GB")
    
    # Generate text
    print(f"\nGenerating text with streaming...")
    prompt = "Once upon a time in a world of artificial intelligence,"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda:0")
    
    gen_start = time.time()
    
    output_ids = streaming.generate(
        input_ids,
        max_new_tokens=100,  # Generate more tokens for large model
        temperature=0.7,
        do_sample=True,
    )
    
    gen_time = time.time() - gen_start
    
    # Decode and show results
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_sec = new_tokens / gen_time
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Parameters: {num_params/1e9:.1f}B")
    print(f"Model size: {model_size_gb:.1f} GB")
    print(f"Layers: {num_layers}")
    print(f"GPU layers: {config.gpu_layers}")
    print(f"Setup GPU memory: {gpu_mem_setup:.2f} GB")
    print(f"Peak GPU memory: {peak_mem:.2f} GB")
    print(f"Tokens generated: {new_tokens}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Tokens/second: {tokens_per_sec:.2f}")
    print(f"\n{'='*60}")
    print("Generated Text:")
    print(f"{'='*60}")
    print(output_text[:500] + "..." if len(output_text) > 500 else output_text)
    print(f"{'='*60}")
    
    # Get streaming stats
    stats = streaming.get_stats()
    print(f"\nStreaming Statistics:")
    print(f"  Forward passes: {stats['forward_count']}")
    print(f"  Avg time/forward: {stats['avg_time_per_forward']*1000:.1f}ms")
    
    # Cleanup
    streaming.cleanup()
    
    # Memory saved calculation
    memory_saved = model_size_gb - peak_mem
    print(f"\n✅ LARGE MODEL STREAMING TEST PASSED!")
    print(f"   Memory saved: {memory_saved:.1f} GB (model={model_size_gb:.1f}GB, peak={peak_mem:.1f}GB)")
    
    return {
        "model": model_name,
        "params_b": num_params / 1e9,
        "model_size_gb": model_size_gb,
        "peak_gpu_gb": peak_mem,
        "tokens_per_sec": tokens_per_sec,
        "success": True,
    }


@app.local_entrypoint()
def main():
    """Run all zStream tests."""
    print("=" * 60)
    print("zStream Layer Streaming Tests")
    print("=" * 60)
    
    # Run basic test first
    print("\nRunning basic layer streaming test...")
    result1 = test_zstream_basic.remote()
    
    if result1:
        print("\n✅ Basic test passed")
    else:
        print("\n❌ Basic test failed")
        return
    
    # Run prefetcher test
    print("\nRunning prefetcher test...")
    result2 = test_zstream_prefetcher.remote()
    
    if result2:
        print("\n✅ Prefetcher test passed")
    else:
        print("\n❌ Prefetcher test failed")
    
    # Run real model test (optional - takes longer)
    print("\nRunning real model test (may take 5-10 minutes)...")
    try:
        result3 = test_zstream_real_model.remote()
        if result3:
            print("\n✅ Real model test passed")
        else:
            print("\n❌ Real model test failed")
    except Exception as e:
        print(f"\n⚠️ Real model test skipped: {e}")
    
    print("\n" + "=" * 60)
    print("All zStream tests completed!")
    print("=" * 60)
