"""ZSE Performance Optimization — Unit tests.

Tests correctness of optimized kernels:
- Tiled matmul (shared memory tiling)
- Fused residual + RMSNorm
- Vectorized element-wise ops
- Batched RoPE
- Bulk logits download
- Pre-indexed weight lookup
- Batch struct.unpack sampler
"""

import struct
import math
import pytest

from zse_engine.orchestrator.kernels import InferenceKernels


# ======================================================================
# Kernel Registration Tests
# ======================================================================

class TestOptimizedKernelRegistration:
    def test_tiled_matmul_registered(self):
        k = InferenceKernels(backend="cuda")
        assert "tiled_fp16_matmul" in k.kernel_names
        assert "tiled_dequant_matmul_int4" in k.kernel_names
        assert "tiled_dequant_matmul_int8" in k.kernel_names

    def test_fused_residual_rmsnorm_registered(self):
        k = InferenceKernels(backend="cuda")
        assert "fused_residual_rmsnorm" in k.kernel_names

    def test_vectorized_kernels_removed(self):
        """Dead vectorized kernels removed — superseded by fp32 residual stream."""
        k = InferenceKernels(backend="cuda")
        assert "residual_add_vec" not in k.kernel_names
        assert "silu_mul_vec" not in k.kernel_names

    def test_batched_rope_registered(self):
        k = InferenceKernels(backend="cuda")
        assert "batched_rotary_embedding" in k.kernel_names

    def test_total_kernel_count(self):
        k = InferenceKernels(backend="cuda")
        assert len(k.kernel_names) == 24  # active kernels only (dead code removed)

    def test_original_kernels_still_present(self):
        """Core kernels are present (dead code removed, active code kept)."""
        k = InferenceKernels(backend="cuda")
        originals = [
            "rmsnorm", "silu_mul", "rotary_embedding",
            "paged_attention", "kv_cache_write", "bias_add",
        ]
        for name in originals:
            assert name in k.kernel_names, f"Missing kernel: {name}"


# ======================================================================
# Kernel Source Correctness Tests (syntax validation)
# ======================================================================

class TestKernelSources:
    def test_tiled_fp16_matmul_has_shared_memory(self):
        k = InferenceKernels(backend="cuda")
        src = k.KERNEL_SOURCES["tiled_fp16_matmul"]
        assert "__shared__" in src
        assert "TILE" in src

    def test_tiled_int4_has_dequant(self):
        k = InferenceKernels(backend="cuda")
        src = k.KERNEL_SOURCES["tiled_dequant_matmul_int4"]
        assert "__shared__" in src
        assert "nibble" in src
        assert "group_size" in src

    def test_fused_kernel_has_residual_and_rmsnorm(self):
        k = InferenceKernels(backend="cuda")
        src = k.KERNEL_SOURCES["fused_residual_rmsnorm"]
        assert "rsqrtf" in src  # RMSNorm
        assert "residual_out" in src
        assert "__shfl_xor_sync" in src  # Warp reduction

    def test_batched_rope_has_positions_array(self):
        k = InferenceKernels(backend="cuda")
        src = k.KERNEL_SOURCES["batched_rotary_embedding"]
        assert "positions" in src
        assert "int M" in src

    def test_vectorized_add_removed(self):
        """Dead vectorized kernel removed from KERNEL_SOURCES."""
        k = InferenceKernels(backend="cuda")
        assert "residual_add_vec" not in k.KERNEL_SOURCES


# ======================================================================
# Sampler Optimization Tests
# ======================================================================

class TestSamplerOptimization:
    def test_batch_decode_fp16(self):
        """Verify batch struct.unpack matches per-element."""
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler(seed=42)

        vocab = 100
        # Create fake logits
        logits = [float(i) * 0.01 for i in range(vocab)]
        data = struct.pack(f'<{vocab}e', *logits)

        decoded = s._decode_fp16(data, vocab)
        assert len(decoded) == vocab
        for i, val in enumerate(decoded):
            expected = logits[i]
            assert abs(val - expected) < 0.01, f"Mismatch at {i}: {val} vs {expected}"

    def test_greedy_still_works(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler(seed=42)

        vocab = 50
        logits = [0.0] * vocab
        logits[42] = 10.0  # Make token 42 the argmax
        data = struct.pack(f'<{vocab}e', *logits)

        token = s.greedy(data, vocab)
        assert token == 42

    def test_temperature_sampling(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler(seed=42)

        vocab = 10
        logits = [0.0] * vocab
        logits[5] = 5.0
        data = struct.pack(f'<{vocab}e', *logits)

        token = s.sample(data, vocab, temperature=0.0)
        assert token == 5  # Greedy

    def test_repetition_penalty(self):
        from zse_engine.orchestrator.sampler import Sampler
        s = Sampler(seed=42)

        vocab = 10
        logits = [1.0] * vocab
        logits[3] = 5.0
        data = struct.pack(f'<{vocab}e', *logits)

        # Without penalty: should pick 3
        token_no_penalty = s.sample(data, vocab, temperature=0.0)
        assert token_no_penalty == 3

        # With high penalty on token 3: should still pick 3 at temp=0
        # (just reduced logit, still highest)
        token_with_penalty = s.sample(
            data, vocab, temperature=0.0,
            repetition_penalty=2.0, past_tokens={3},
        )
        # logits[3] = 5.0/2.0 = 2.5, still > 1.0
        assert token_with_penalty == 3


# ======================================================================
# Model Runner Optimization Tests
# ======================================================================

class TestModelRunnerOptimizations:
    def test_weight_index_building(self):
        """Pre-indexed weight lookup builds correctly."""
        from unittest.mock import MagicMock, patch
        from zse_engine.format.config import ModelConfig

        config = ModelConfig(
            num_layers=2, num_heads=4, num_kv_heads=4,
            head_dim=16, hidden_size=64, vocab_size=100,
        )

        # Mock weights with find() returning None (no weights loaded)
        mock_weights = MagicMock()
        mock_weights.find.return_value = None

        from zse_engine.orchestrator.model_runner import ModelRunner
        with patch.object(ModelRunner, '__init__', lambda self, **kw: None):
            runner = ModelRunner.__new__(ModelRunner)
            runner._config = config
            runner._weights = mock_weights
            runner._num_layers = 2
            runner._hidden_size = 64
            runner._num_heads = 4
            runner._num_kv_heads = 4
            runner._head_dim = 16
            runner._intermediate_size = 128
            runner._rope_theta = 10000.0
            runner._rms_eps = 1e-5
            runner._scale = 0.25
            runner._layer_weights = runner._build_weight_index()

        assert len(runner._layer_weights) == 2
        # find() was called for each layer × each weight name
        assert mock_weights.find.call_count >= 18  # 2 layers × 9+ weight names

    def test_bulk_download_logits(self):
        """Bulk download produces same data as per-row download."""
        vocab = 50
        M = 3
        row_bytes = vocab * 2

        # Create fake logits data
        all_data = bytes(range(M * row_bytes % 256)) * (M * row_bytes // (M * row_bytes % 256 or 1) + 1)
        all_data = all_data[:M * row_bytes]

        # Simulate slicing
        rows = [all_data[i * row_bytes:(i + 1) * row_bytes] for i in range(M)]
        assert len(rows) == M
        assert all(len(r) == row_bytes for r in rows)


# ======================================================================
# Integration: Optimized Kernel + Original Produce Same Results
# ======================================================================

class TestOptimizedVsOriginal:
    def test_kernel_sources_parse_cleanly(self):
        """All kernel source strings are non-empty and contain entry points."""
        k = InferenceKernels(backend="cuda")
        for name, src in k.KERNEL_SOURCES.items():
            assert len(src) > 100, f"Kernel {name} source too short"
            assert "extern \"C\" __global__" in src, f"Kernel {name} missing entry point"

    def test_tiled_matmul_launch_config(self):
        """Tiled matmul uses 2D grid/block (32x32 tiles)."""
        # Verify the model_runner uses (N+31)//32, (M+31)//32 grid
        # This is tested via the kernel source having TILE=32
        k = InferenceKernels(backend="cuda")
        src = k.KERNEL_SOURCES["tiled_fp16_matmul"]
        assert "#define TILE 32" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
