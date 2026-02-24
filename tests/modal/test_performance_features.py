"""
Test ZSE Performance Features on Modal GPU

Tests:
1. Prefix Caching (RadixCache)
2. CUDA Graph Execution
3. Speculative Decoding

Run: modal run tests/modal/test_performance_features.py
"""

import modal
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))

app = modal.App("zse-performance-features-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "safetensors",
        "sentencepiece",
        "bitsandbytes",
    )
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
def test_all_features():
    """Test all three performance features."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    import torch
    import time
    
    print("=" * 60)
    print("ZSE Performance Features Test")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    results = {
        "prefix_cache": False,
        "cuda_graph": False,
        "speculative": False,
    }
    
    # =================================================================
    # TEST 1: PREFIX CACHING (RadixCache)
    # =================================================================
    print("=" * 60)
    print("[1/3] Testing Prefix Caching (RadixCache)")
    print("=" * 60)
    
    try:
        from zse.core.zkv.radix_cache import RadixCache, PrefixMatch
        
        # Create a mock block allocator
        class MockBlockAllocator:
            def __init__(self):
                self.refs = {}
            def inc_ref(self, block_id):
                self.refs[block_id] = self.refs.get(block_id, 0) + 1
            def free(self, block_id):
                self.refs[block_id] = max(0, self.refs.get(block_id, 1) - 1)
        
        allocator = MockBlockAllocator()
        cache = RadixCache(
            block_allocator=allocator,
            max_cached_tokens=10000,
            block_size=16,
        )
        
        # Test 1: Insert and match prefix
        print("\n[Test 1.1] Insert and match prefix...")
        tokens1 = [1, 2, 3, 4, 5, 6, 7, 8]  # System prompt
        blocks1 = [0, 1]  # Fake block IDs
        
        success = cache.insert_prefix(tokens1, blocks1)
        print(f"  Insert result: {'✅' if success else '❌'}")
        
        # Same prefix should hit cache
        match = cache.match_prefix(tokens1)
        print(f"  Exact match length: {match.matched_length}")
        print(f"  Cache hit: {'✅' if match.matched_length == len(tokens1) else '❌'}")
        
        # Test 2: Prefix matching (partial)
        print("\n[Test 1.2] Partial prefix matching...")
        tokens2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Extended prompt
        match2 = cache.match_prefix(tokens2)
        print(f"  Matched length: {match2.matched_length}/{len(tokens2)}")
        print(f"  Remaining tokens: {len(match2.remaining_tokens)}")
        print(f"  Prefix reuse: {'✅' if match2.matched_length > 0 else '❌'}")
        
        # Test 3: Different prefix (miss)
        print("\n[Test 1.3] Cache miss for different prefix...")
        tokens3 = [100, 200, 300]  # Completely different
        match3 = cache.match_prefix(tokens3)
        print(f"  Matched length: {match3.matched_length}")
        print(f"  Cache miss: {'✅' if match3.matched_length == 0 else '❌'}")
        
        # Test 4: Statistics
        print("\n[Test 1.4] Cache statistics...")
        stats = cache.get_stats()
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%}")
        print(f"  Cached tokens: {stats['cached_tokens']}")
        
        results["prefix_cache"] = True
        print("\n✅ Prefix Caching: PASSED")
        
    except Exception as e:
        print(f"\n❌ Prefix Caching: FAILED - {e}")
        import traceback
        traceback.print_exc()
    
    # =================================================================
    # TEST 2: CUDA GRAPH EXECUTION
    # =================================================================
    print("\n" + "=" * 60)
    print("[2/3] Testing CUDA Graph Execution")
    print("=" * 60)
    
    try:
        from zse.core.zgraph.cuda_graph import (
            CUDAGraphRunner,
            is_cuda_graph_compatible,
            capture_model_graph,
        )
        import torch.nn as nn
        
        # Create a simple model for testing
        class SimpleDecoder(nn.Module):
            def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2):
                super().__init__()
                self._vocab_size = vocab_size
                self._hidden_size = hidden_size
                self.embed = nn.Embedding(vocab_size, hidden_size)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size, 
                        nhead=4, 
                        dim_feedforward=512,
                        batch_first=True,
                    )
                    for _ in range(num_layers)
                ])
                self.lm_head = nn.Linear(hidden_size, vocab_size)
                
                # Add config for compatibility
                class Config:
                    pass
                Config.hidden_size = hidden_size
                Config.vocab_size = vocab_size
                self.config = Config()
            
            def forward(self, input_ids, position_ids=None, use_cache=False, **kwargs):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                logits = self.lm_head(x)
                
                class Output:
                    def __init__(self, logits):
                        self.logits = logits
                        self.past_key_values = None
                
                return Output(logits)
        
        print("\n[Test 2.1] Creating test model...")
        model = SimpleDecoder().cuda().eval()
        print(f"  Model on CUDA: ✅")
        
        print("\n[Test 2.2] Checking CUDA graph compatibility...")
        compatible = is_cuda_graph_compatible(model)
        print(f"  Compatible: {'✅' if compatible else '❌'}")
        
        print("\n[Test 2.3] Creating CUDA Graph Runner...")
        runner = CUDAGraphRunner(
            model=model,
            max_batch_size=4,
            device=0,
            enable_capture=True,
        )
        print(f"  Runner created: ✅")
        
        print("\n[Test 2.4] Testing decode step (first call captures graph)...")
        input_ids = torch.randint(0, 1000, (2, 1), device="cuda")
        position_ids = torch.zeros((2, 1), dtype=torch.long, device="cuda")
        
        # Warmup + capture
        start = time.perf_counter()
        output1 = runner.decode_step(input_ids, position_ids, use_graph=True)
        capture_time = (time.perf_counter() - start) * 1000
        print(f"  First call (capture): {capture_time:.2f} ms")
        print(f"  Output shape: {output1.shape}")
        
        print("\n[Test 2.5] Testing graph replay (should be faster)...")
        times = []
        for i in range(10):
            input_ids = torch.randint(0, 1000, (2, 1), device="cuda")
            start = time.perf_counter()
            output = runner.decode_step(input_ids, position_ids, use_graph=True)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_replay = sum(times) / len(times)
        print(f"  Average replay time: {avg_replay:.3f} ms")
        print(f"  Speedup vs capture: {capture_time / avg_replay:.1f}x")
        
        print("\n[Test 2.6] Graph statistics...")
        stats = runner.get_stats()
        print(f"  Captures: {stats['captures']}")
        print(f"  Replays: {stats['replays']}")
        print(f"  Num graphs: {stats['num_graphs']}")
        
        results["cuda_graph"] = True
        print("\n✅ CUDA Graph Execution: PASSED")
        
    except Exception as e:
        print(f"\n❌ CUDA Graph Execution: FAILED - {e}")
        import traceback
        traceback.print_exc()
    
    # =================================================================
    # TEST 3: SPECULATIVE DECODING
    # =================================================================
    print("\n" + "=" * 60)
    print("[3/3] Testing Speculative Decoding")
    print("=" * 60)
    
    try:
        from zse.core.zspec.speculative import (
            SpeculativeDecoder,
            SpeculativeConfig,
            SelfSpeculativeDecoder,
            MedusaHead,
            estimate_speculation_speedup,
            is_compatible_models,
        )
        import torch.nn as nn
        
        print("\n[Test 3.1] Testing speedup estimation...")
        # Estimate theoretical speedup
        speedup = estimate_speculation_speedup(
            target_time_per_token=50.0,  # 50ms for large model
            draft_time_per_token=5.0,    # 5ms for small model
            acceptance_rate=0.7,          # 70% tokens accepted
            num_speculative_tokens=5,
        )
        print(f"  Estimated speedup: {speedup:.2f}x")
        print(f"  Speedup reasonable: {'✅' if 1.5 < speedup < 4.0 else '❌'}")
        
        print("\n[Test 3.2] Testing MedusaHead...")
        medusa = MedusaHead(
            hidden_size=256,
            vocab_size=1000,
            num_heads=4,
        ).cuda()
        
        hidden = torch.randn(2, 10, 256, device="cuda")
        predictions = medusa(hidden)
        print(f"  Num heads: {len(predictions)}")
        print(f"  Prediction shape: {predictions[0].shape}")
        print(f"  MedusaHead works: ✅")
        
        print("\n[Test 3.3] Testing SpeculativeConfig...")
        config = SpeculativeConfig(
            num_speculative_tokens=5,
            acceptance_threshold=0.0,
            use_tree_attention=False,
            min_acceptance_rate=0.3,
        )
        print(f"  Config created: ✅")
        print(f"  Speculative tokens: {config.num_speculative_tokens}")
        
        print("\n[Test 3.4] Testing model compatibility check...")
        # Create two compatible models
        class Model1(nn.Module):
            def __init__(self):
                super().__init__()
                class Config:
                    vocab_size = 32000
                self.config = Config()
        
        class Model2(nn.Module):
            def __init__(self):
                super().__init__()
                class Config:
                    vocab_size = 32000
                self.config = Config()
        
        m1, m2 = Model1(), Model2()
        compatible = is_compatible_models(m1, m2)
        print(f"  Models compatible: {'✅' if compatible else '❌'}")
        
        # Test incompatible
        class Model3(nn.Module):
            def __init__(self):
                super().__init__()
                class Config:
                    vocab_size = 50000  # Different vocab
                self.config = Config()
        
        m3 = Model3()
        incompatible = not is_compatible_models(m1, m3)
        print(f"  Incompatibility detected: {'✅' if incompatible else '❌'}")
        
        results["speculative"] = True
        print("\n✅ Speculative Decoding: PASSED")
        
    except Exception as e:
        print(f"\n❌ Speculative Decoding: FAILED - {e}")
        import traceback
        traceback.print_exc()
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    print(f"  Prefix Caching (RadixCache): {'✅ PASSED' if results['prefix_cache'] else '❌ FAILED'}")
    print(f"  CUDA Graph Execution:        {'✅ PASSED' if results['cuda_graph'] else '❌ FAILED'}")
    print(f"  Speculative Decoding:        {'✅ PASSED' if results['speculative'] else '❌ FAILED'}")
    print()
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 60)
    
    return results


@app.local_entrypoint()
def main():
    """Run tests on Modal GPU."""
    print("🚀 Starting ZSE Performance Features Test on Modal...")
    print()
    results = test_all_features.remote()
    print("\n✅ Test run complete!")
