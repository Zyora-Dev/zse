"""Tests for ZSE Orchestrator — VRAM allocator, weight loader, sampler, engine API.

All tests run CPU-only with mocked GPU — no actual GPU required.
GPU tests are in test_modal_orchestrator.py.
"""

import struct
import math
import os
import sys
import pytest

# Ensure zse packages are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zse-compiler'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zse-engine'))


# ============================================================================
# Test VRAMAllocator
# ============================================================================

class TestVRAMAllocator:
    def _make_config(self, **overrides):
        from zse_engine.format.config import ModelConfig
        defaults = dict(
            arch="llama", num_layers=4, num_heads=8, num_kv_heads=8,
            head_dim=64, hidden_size=512, intermediate_size=1376,
            vocab_size=32000, max_seq_len=2048,
        )
        defaults.update(overrides)
        return ModelConfig(**defaults)

    def test_plan_allocation_basic(self):
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator
        config = self._make_config()

        alloc = VRAMAllocator()  # CPU-only mode
        plan = alloc.plan_allocation(
            model_size_bytes=500 * 1024 * 1024,  # 500MB
            config=config,
        )

        assert plan.weight_bytes == 500 * 1024 * 1024
        assert plan.kv_cache_bytes > 0
        assert plan.scratch_bytes > 0
        assert plan.reserved_bytes > 0
        assert plan.utilization_pct > 0
        assert plan.total_vram == 16 * 1024**3  # Default 16GB

    def test_plan_summary(self):
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator
        config = self._make_config()
        alloc = VRAMAllocator()
        plan = alloc.plan_allocation(500 * 1024**2, config)
        summary = plan.summary()
        assert "VRAM Plan" in summary
        assert "Weights" in summary
        assert "KV Cache" in summary

    def test_scratch_bytes_estimate(self):
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator
        config = self._make_config()
        alloc = VRAMAllocator()
        scratch_bytes = alloc._estimate_scratch_bytes(config, max_seq_len=1024)
        # Should be > 0 and reasonable
        assert scratch_bytes > 0
        # For 512 hidden, 1024 seq_len: hidden alone is 1024*512*2 = 1MB
        assert scratch_bytes > 1024 * 1024

    def test_plan_kv_budget_after_weights(self):
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator
        config = self._make_config()
        alloc = VRAMAllocator()

        # With huge demand, KV must be capped by remaining VRAM after weights
        # Small model leaves more remaining → can satisfy more of the demand
        plan_small = alloc.plan_allocation(
            100 * 1024**2, config, max_seq_len=8192, max_batch_seqs=256,
        )
        plan_large = alloc.plan_allocation(
            10 * 1024**3, config, max_seq_len=8192, max_batch_seqs=256,
        )

        assert plan_small.kv_cache_bytes >= plan_large.kv_cache_bytes

    def test_scratch_buffers_cpu_mode(self):
        """In CPU mode (no gpu_mem), scratch buffers are None but total_bytes computed."""
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator
        config = self._make_config()
        alloc = VRAMAllocator()
        scratch = alloc.allocate_scratch(config, max_seq_len=512)
        assert scratch.hidden is None  # No GPU to allocate on
        assert scratch.total_bytes > 0

    def test_track_weight_upload(self):
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator
        alloc = VRAMAllocator()
        assert alloc.weight_bytes == 0
        alloc.track_weight_upload(1000)
        alloc.track_weight_upload(2000)
        assert alloc.weight_bytes == 3000
        assert alloc.allocated_bytes == 3000

    def test_max_batch_tokens(self):
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator
        config = self._make_config()
        alloc = VRAMAllocator()
        plan = alloc.plan_allocation(100 * 1024**2, config)
        # Should compute how many tokens fit in KV budget
        assert plan.max_batch_tokens > 0
        # Verify: max_tokens * bytes_per_token <= kv_budget
        assert (plan.max_batch_tokens * config.total_kv_cache_bytes_per_token
                <= plan.kv_cache_bytes)


# ============================================================================
# Test WeightStore
# ============================================================================

class TestWeightStore:
    def test_basic(self):
        from zse_engine.orchestrator.weight_loader import WeightStore, GPUWeight
        store = WeightStore()
        w = GPUWeight(
            name="embed_tokens.weight", shape=(32000, 512),
            dtype="float16", data_ptr=0x1000, data_nbytes=32000 * 512 * 2,
        )
        store.add(w)
        assert store.num_weights == 1
        assert store.has("embed_tokens.weight")
        assert store.get("embed_tokens.weight").data_ptr == 0x1000
        assert "float16" in store.summary()

    def test_find_missing(self):
        from zse_engine.orchestrator.weight_loader import WeightStore
        store = WeightStore()
        assert store.find("nonexistent") is None

    def test_total_bytes(self):
        from zse_engine.orchestrator.weight_loader import GPUWeight, WeightStore
        store = WeightStore()
        store.add(GPUWeight(name="a", shape=(10,), dtype="float16",
                            data_ptr=1, data_nbytes=100,
                            scales_ptr=2, scales_nbytes=20))
        assert store.total_bytes == 120  # 100 + 20

    def test_contains(self):
        from zse_engine.orchestrator.weight_loader import GPUWeight, WeightStore
        store = WeightStore()
        store.add(GPUWeight(name="x", shape=(1,), dtype="float16",
                            data_ptr=1, data_nbytes=2))
        assert "x" in store
        assert "y" not in store

    def test_iter(self):
        from zse_engine.orchestrator.weight_loader import GPUWeight, WeightStore
        store = WeightStore()
        store.add(GPUWeight(name="a", shape=(1,), dtype="float16",
                            data_ptr=1, data_nbytes=2))
        store.add(GPUWeight(name="b", shape=(2,), dtype="int4",
                            data_ptr=3, data_nbytes=4))
        names = [w.name for w in store]
        assert set(names) == {"a", "b"}


# ============================================================================
# Test Sampler
# ============================================================================

class TestSampler:
    def _make_logits(self, values):
        """Pack float values as fp16 bytes."""
        return struct.pack(f'<{len(values)}e', *values)

    def test_greedy(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler()
        logits = self._make_logits([0.1, 0.5, 0.3, 0.9, 0.2])
        token = s.greedy(logits, 5)
        assert token == 3  # argmax at index 3

    def test_greedy_negative(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler()
        logits = self._make_logits([-1.0, -0.5, -2.0])
        token = s.greedy(logits, 3)
        assert token == 1  # -0.5 is highest

    def test_temperature_zero_is_greedy(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler(seed=42)
        logits = self._make_logits([0.1, 0.5, 0.3, 0.9, 0.2])
        token = s.sample(logits, 5, temperature=0.0)
        assert token == 3

    def test_temperature_high_increases_randomness(self):
        """High temperature should spread probability more evenly."""
        from zse_engine.orchestrator.sampler import Sampler
        logits = self._make_logits([10.0, 0.0, 0.0, 0.0, 0.0])

        # Low temp: almost always picks 0
        counts = {0: 0, 1: 0}
        for i in range(100):
            s = Sampler(seed=i)
            t = s.sample(logits, 5, temperature=0.1, top_p=1.0, top_k=0)
            if t == 0:
                counts[0] += 1
            else:
                counts[1] += 1
        assert counts[0] >= 95  # Should pick 0 almost always

        # High temp: more spread
        counts = {0: 0, 1: 0}
        for i in range(100):
            s = Sampler(seed=i + 1000)
            t = s.sample(logits, 5, temperature=5.0, top_p=1.0, top_k=0)
            if t == 0:
                counts[0] += 1
            else:
                counts[1] += 1
        assert counts[1] > 10  # Should pick non-zero sometimes

    def test_top_k(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler(seed=42)
        # 5 tokens, top_k=2 means only top 2 are considered
        logits = self._make_logits([0.1, 5.0, 0.2, 4.0, 0.0])
        tokens = set()
        for i in range(50):
            s = Sampler(seed=i)
            t = s.sample(logits, 5, temperature=1.0, top_k=2, top_p=1.0)
            tokens.add(t)
        # Should only sample from indices 1 and 3
        assert tokens.issubset({1, 3})

    def test_top_p(self):
        from zse_engine.orchestrator.sampler import Sampler
        # One dominant token
        logits = self._make_logits([10.0, -10.0, -10.0, -10.0])
        s = Sampler(seed=42)
        t = s.sample(logits, 4, temperature=1.0, top_p=0.5, top_k=0)
        assert t == 0  # Only token 0 has > 50% prob

    def test_repetition_penalty(self):
        from zse_engine.orchestrator.sampler import Sampler
        # Token 0 has highest logit but is in past_tokens
        logits = self._make_logits([5.0, 4.9, 0.1])

        # Without penalty: always token 0
        s = Sampler(seed=42)
        t = s.sample(logits, 3, temperature=0.0)
        assert t == 0

        # With penalty: token 0's logit reduced, token 1 wins
        s = Sampler(seed=42)
        t = s.sample(logits, 3, temperature=0.0,
                     repetition_penalty=2.0, past_tokens={0})
        assert t == 1

    def test_categorical_sample_deterministic(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler(seed=123)
        logits = self._make_logits([1.0, 1.0, 1.0, 1.0])
        # With seed, should be deterministic
        t1 = s.sample(logits, 4, temperature=1.0, top_p=1.0, top_k=0)
        s2 = Sampler(seed=123)
        t2 = s2.sample(logits, 4, temperature=1.0, top_p=1.0, top_k=0)
        assert t1 == t2

    def test_decode_fp16_error(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler()
        with pytest.raises(ValueError, match="Expected"):
            s.greedy(b'\x00', 5)  # Too few bytes


# ============================================================================
# Test InferenceKernels (structure only — no GPU)
# ============================================================================

class TestInferenceKernels:
    def test_kernel_names(self):
        from zse_engine.orchestrator.kernels import InferenceKernels
        k = InferenceKernels(backend="cuda")
        names = k.kernel_names
        assert "rmsnorm" in names
        assert "silu_mul" in names
        assert "paged_attention" in names
        assert "tiled_dequant_matmul_int4" in names
        assert "tiled_dequant_matmul_int8" in names
        assert "embedding_lookup_f32out" in names
        assert "kv_cache_write" in names
        assert len(names) == 32  # 24 core + 8 Gemma 4 dedicated-path kernels

    def test_not_compiled_initially(self):
        from zse_engine.orchestrator.kernels import InferenceKernels
        k = InferenceKernels(backend="cuda")
        assert k.num_compiled == 0
        assert not k.is_compiled("rmsnorm")

    def test_unknown_kernel(self):
        from zse_engine.orchestrator.kernels import InferenceKernels
        k = InferenceKernels(backend="cuda")
        with pytest.raises(ValueError, match="Unknown kernel"):
            k.compile_kernel("nonexistent")

    def test_kernel_sources_valid_strings(self):
        """All kernel sources should be non-empty CUDA C strings."""
        from zse_engine.orchestrator.kernels import InferenceKernels
        for name, source in InferenceKernels.KERNEL_SOURCES.items():
            assert isinstance(source, str), f"{name} source is not a string"
            assert len(source) > 50, f"{name} source too short"
            assert "extern \"C\"" in source, f"{name} missing extern C"
            assert "__global__" in source, f"{name} missing __global__"


# ============================================================================
# Test GenerateConfig
# ============================================================================

class TestGenerateConfig:
    def test_defaults(self):
        from zse_engine.orchestrator.engine import GenerateConfig
        cfg = GenerateConfig()
        assert cfg.max_tokens == 128
        assert cfg.temperature == 1.0
        assert cfg.top_p == 0.9
        assert cfg.top_k == 50


# ============================================================================
# Test GPUWeight
# ============================================================================

class TestGPUWeight:
    def test_total_bytes(self):
        from zse_engine.orchestrator.weight_loader import GPUWeight
        w = GPUWeight(
            name="test", shape=(10, 10), dtype="int4",
            data_ptr=1, data_nbytes=100,
            scales_ptr=2, scales_nbytes=20,
            zeros_ptr=3, zeros_nbytes=10,
        )
        assert w.total_gpu_bytes == 130


# ============================================================================
# Integration: VRAMPlan + ModelConfig
# ============================================================================

class TestVRAMPlanIntegration:
    def test_7b_model_plan(self):
        """Plan for a ~7B parameter Llama model."""
        from zse_engine.format.config import ModelConfig
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator

        config = ModelConfig(
            arch="llama", num_layers=32, num_heads=32, num_kv_heads=32,
            head_dim=128, hidden_size=4096, intermediate_size=11008,
            vocab_size=32000, max_seq_len=4096,
        )

        alloc = VRAMAllocator()
        model_size = config.estimate_model_size_bytes()
        plan = alloc.plan_allocation(model_size, config)

        # 7B INT4 should be ~3.5GB weights
        assert 2 * 1024**3 < plan.weight_bytes < 6 * 1024**3
        # KV should get substantial budget
        assert plan.kv_cache_bytes > 0
        # Scratch should be reasonable
        assert plan.scratch_bytes > 0

    def test_small_model_gets_more_kv(self):
        """Smaller model = more VRAM for KV cache."""
        from zse_engine.format.config import ModelConfig
        from zse_engine.orchestrator.vram_allocator import VRAMAllocator

        small = ModelConfig(
            num_layers=4, num_heads=8, num_kv_heads=8,
            head_dim=64, hidden_size=512, intermediate_size=1376, vocab_size=1000,
        )
        big = ModelConfig(
            num_layers=32, num_heads=32, num_kv_heads=32,
            head_dim=128, hidden_size=4096, intermediate_size=11008, vocab_size=32000,
        )

        alloc = VRAMAllocator()
        # Use a workload large enough that the big model's KV gets clipped by
        # remaining VRAM (demand-based sizing).
        plan_s = alloc.plan_allocation(
            small.estimate_model_size_bytes(), small,
            max_seq_len=8192, max_batch_seqs=256,
        )
        plan_b = alloc.plan_allocation(
            big.estimate_model_size_bytes(), big,
            max_seq_len=8192, max_batch_seqs=256,
        )

        assert plan_s.kv_cache_bytes >= plan_b.kv_cache_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
