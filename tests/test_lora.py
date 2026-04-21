"""ZSE LoRA Serving — Unit tests.

Tests LoRA weight storage, adapter lifecycle, format I/O,
request routing, and LoRA-aware matmul integration.

No GPU needed — tests the algorithm and data structures.
"""

import struct
import math
import os
import tempfile
import pytest

from zse_engine.orchestrator.lora_weights import (
    LoRAWeight, LoRAAdapter, LoRAWeightStore,
)
from zse_engine.orchestrator.kernels import InferenceKernels


# ======================================================================
# LoRA Weight Tests
# ======================================================================

class TestLoRAWeight:
    def test_basic_creation(self):
        w = LoRAWeight(
            layer_name="model.layers.0.self_attn.q_proj",
            rank=16, in_features=4096, out_features=4096,
        )
        assert w.a_shape == (16, 4096)
        assert w.b_shape == (4096, 16)
        assert w.rank == 16

    def test_gpu_bytes(self):
        w = LoRAWeight(
            layer_name="test", rank=16, in_features=4096, out_features=4096,
            a_nbytes=16 * 4096 * 2, b_nbytes=4096 * 16 * 2,
        )
        # A: 16*4096*2 = 131072, B: 4096*16*2 = 131072
        assert w.total_gpu_bytes == 262144

    def test_tensor_creation(self):
        w = LoRAWeight(
            layer_name="test", rank=8, in_features=512, out_features=256,
            a_ptr=0x1000, a_nbytes=8 * 512 * 2,
            b_ptr=0x2000, b_nbytes=256 * 8 * 2,
        )
        a_t = w.make_a_tensor()
        assert a_t.shape == (8, 512)
        b_t = w.make_b_tensor()
        assert b_t.shape == (256, 8)


# ======================================================================
# LoRA Adapter Tests
# ======================================================================

class TestLoRAAdapter:
    def test_basic_creation(self):
        adapter = LoRAAdapter(
            adapter_id="test-adapter",
            rank=16, alpha=16.0,
            target_modules=["q_proj", "v_proj"],
            num_layers=2,
        )
        assert adapter.adapter_id == "test-adapter"
        assert adapter.scaling == 1.0  # alpha/rank = 16/16
        assert adapter.num_weight_pairs == 0

    def test_scaling_factor(self):
        adapter = LoRAAdapter(adapter_id="a", rank=16, alpha=32.0)
        assert adapter.scaling == 2.0

        adapter2 = LoRAAdapter(adapter_id="b", rank=8, alpha=8.0)
        assert adapter2.scaling == 1.0

    def test_add_and_get_weight(self):
        adapter = LoRAAdapter(adapter_id="a", rank=16, alpha=16.0, num_layers=2)
        w = LoRAWeight("test", rank=16, in_features=4096, out_features=4096)
        adapter.add_weight(0, "q_proj", w)

        assert adapter.has_weight(0, "q_proj")
        assert not adapter.has_weight(0, "v_proj")
        assert not adapter.has_weight(1, "q_proj")
        assert adapter.get_weight(0, "q_proj") is w
        assert adapter.num_weight_pairs == 1

    def test_summary(self):
        adapter = LoRAAdapter(
            adapter_id="customer-a", rank=16, alpha=16.0,
            target_modules=["q_proj", "v_proj"], num_layers=32,
        )
        s = adapter.summary()
        assert "customer-a" in s
        assert "Rank: 16" in s


# ======================================================================
# LoRA Weight Store Tests
# ======================================================================

class TestLoRAWeightStore:
    def test_add_and_get(self):
        store = LoRAWeightStore()
        adapter = LoRAAdapter(adapter_id="a", rank=16, alpha=16.0)
        store.add(adapter)

        assert store.has("a")
        assert store.get("a") is adapter
        assert store.num_adapters == 1

    def test_remove(self):
        store = LoRAWeightStore()
        adapter = LoRAAdapter(adapter_id="a", rank=16, alpha=16.0)
        store.add(adapter)

        removed = store.remove("a")
        assert removed is adapter
        assert not store.has("a")
        assert store.num_adapters == 0

    def test_remove_nonexistent(self):
        store = LoRAWeightStore()
        assert store.remove("missing") is None

    def test_multiple_adapters(self):
        store = LoRAWeightStore()
        for i in range(5):
            store.add(LoRAAdapter(adapter_id=f"adapter-{i}", rank=16, alpha=16.0))

        assert store.num_adapters == 5
        assert "adapter-3" in store.list_adapters()

    def test_summary(self):
        store = LoRAWeightStore()
        store.add(LoRAAdapter(adapter_id="a", rank=16, alpha=16.0))
        store.add(LoRAAdapter(adapter_id="b", rank=8, alpha=8.0))
        s = store.summary()
        assert "2 adapters" in s


# ======================================================================
# LoRA Format Tests
# ======================================================================

class TestLoRAFormat:
    def test_save_and_load(self):
        from zse_engine.format.lora_format import save_lora, load_lora

        # Create adapter
        adapter = LoRAAdapter(
            adapter_id="test", rank=4, alpha=4.0,
            target_modules=["q_proj"], num_layers=2,
        )

        # Create fake weight data
        in_features, out_features = 64, 32
        weight_data = {}
        for layer in range(2):
            a_bytes = struct.pack(f'<{4 * in_features}e', *[0.1] * (4 * in_features))
            b_bytes = struct.pack(f'<{out_features * 4}e', *[0.2] * (out_features * 4))
            weight_data[(layer, "q_proj")] = (a_bytes, b_bytes)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.zse-lora', delete=False) as f:
            path = f.name
        try:
            save_lora(path, adapter, weight_data)

            # Load
            shapes = {"q_proj": (in_features, out_features)}
            loaded_adapter, loaded_data = load_lora(path, shapes)

            assert loaded_adapter.rank == 4
            assert loaded_adapter.alpha == pytest.approx(4.0)
            assert loaded_adapter.num_layers == 2
            assert loaded_adapter.target_modules == ["q_proj"]
            assert len(loaded_data) == 2
            assert (0, "q_proj") in loaded_data
            assert (1, "q_proj") in loaded_data

            # Verify data matches
            for key in weight_data:
                orig_a, orig_b = weight_data[key]
                load_a, load_b = loaded_data[key]
                assert orig_a == load_a
                assert orig_b == load_b
        finally:
            os.unlink(path)

    def test_multiple_targets(self):
        from zse_engine.format.lora_format import save_lora, load_lora

        adapter = LoRAAdapter(
            adapter_id="multi", rank=8, alpha=8.0,
            target_modules=["q_proj", "v_proj"], num_layers=1,
        )

        weight_data = {}
        for target in ["q_proj", "v_proj"]:
            a_bytes = struct.pack(f'<{8 * 32}e', *[0.01] * (8 * 32))
            b_bytes = struct.pack(f'<{32 * 8}e', *[0.02] * (32 * 8))
            weight_data[(0, target)] = (a_bytes, b_bytes)

        with tempfile.NamedTemporaryFile(suffix='.zse-lora', delete=False) as f:
            path = f.name
        try:
            save_lora(path, adapter, weight_data)

            shapes = {"q_proj": (32, 32), "v_proj": (32, 32)}
            loaded, data = load_lora(path, shapes)
            assert len(data) == 2
            assert loaded.target_modules == ["q_proj", "v_proj"]
        finally:
            os.unlink(path)

    def test_estimate_lora_size(self):
        from zse_engine.format.lora_format import estimate_lora_size

        # rank=16, 32 layers, q_proj + v_proj on a 4096-hidden model
        size = estimate_lora_size(
            rank=16, num_layers=32,
            target_modules=["q_proj", "v_proj"],
            hidden_size=4096, num_heads=32, num_kv_heads=32, head_dim=128,
        )
        # Per target per layer: A = 16*4096*2 + B = 4096*16*2 = 262144 bytes
        # 2 targets * 32 layers * 262144 = 16,777,216 = 16MB
        assert size == 16 * 1024 * 1024

    def test_invalid_magic(self):
        from zse_engine.format.lora_format import load_lora

        with tempfile.NamedTemporaryFile(suffix='.zse-lora', delete=False) as f:
            f.write(b'BAAD' + b'\x00' * 28)
            path = f.name
        try:
            with pytest.raises(ValueError, match="bad magic"):
                load_lora(path, {})
        finally:
            os.unlink(path)


# ======================================================================
# Request LoRA ID Tests
# ======================================================================

class TestRequestLoRA:
    def test_lora_id_in_params(self):
        from zse_engine.zstreamer.request import GenerationParams
        params = GenerationParams(lora_id="customer-a")
        assert params.lora_id == "customer-a"

    def test_lora_id_default_none(self):
        from zse_engine.zstreamer.request import GenerationParams
        params = GenerationParams()
        assert params.lora_id is None

    def test_request_lora_id_property(self):
        from zse_engine.zstreamer.request import InferenceRequest, GenerationParams
        req = InferenceRequest(
            request_id="r1",
            prompt_tokens=[1, 2, 3],
            params=GenerationParams(lora_id="adapter-1"),
        )
        assert req.lora_id == "adapter-1"

    def test_request_no_lora(self):
        from zse_engine.zstreamer.request import InferenceRequest
        req = InferenceRequest(request_id="r2", prompt_tokens=[1, 2])
        assert req.lora_id is None


# ======================================================================
# Kernel Registration Tests
# ======================================================================

class TestLoRAKernel:
    def test_lora_scaled_add_registered(self):
        k = InferenceKernels(backend="cuda")
        assert "lora_scaled_add" in k.kernel_names

    def test_lora_scaled_add_source(self):
        k = InferenceKernels(backend="cuda")
        src = k.KERNEL_SOURCES["lora_scaled_add"]
        assert "scaling" in src
        assert "__half2float" in src

    def test_total_kernel_count(self):
        k = InferenceKernels(backend="cuda")
        assert len(k.kernel_names) == 24  # active kernels only (dead code removed)


# ======================================================================
# LoRA Manager Tests (no GPU)
# ======================================================================

class TestLoRAManager:
    def test_load_random_adapter(self):
        from unittest.mock import MagicMock
        from zse_engine.orchestrator.lora_manager import LoRAManager

        mock_gpu = MagicMock()
        mock_kernels = MagicMock()

        manager = LoRAManager(mock_gpu, mock_kernels)
        adapter = manager.load_adapter_random(
            adapter_id="test",
            rank=16, alpha=16.0,
            num_layers=2,
            target_modules=["q_proj", "v_proj"],
            hidden_size=64,
            num_heads=4, num_kv_heads=4, head_dim=16,
        )

        assert adapter.adapter_id == "test"
        assert adapter.rank == 16
        assert adapter.num_weight_pairs == 4  # 2 layers × 2 targets
        assert manager.has_adapter("test")
        assert manager.num_adapters == 1

    def test_unload_adapter(self):
        from unittest.mock import MagicMock
        from zse_engine.orchestrator.lora_manager import LoRAManager

        mock_gpu = MagicMock()
        mock_kernels = MagicMock()

        manager = LoRAManager(mock_gpu, mock_kernels)
        manager.load_adapter_random(
            adapter_id="test", rank=8, alpha=8.0,
            num_layers=1, target_modules=["q_proj"],
            hidden_size=64, num_heads=4, num_kv_heads=4, head_dim=16,
        )

        assert manager.has_adapter("test")
        result = manager.unload_adapter("test")
        assert result is True
        assert not manager.has_adapter("test")
        assert manager.num_adapters == 0

    def test_unload_nonexistent(self):
        from unittest.mock import MagicMock
        from zse_engine.orchestrator.lora_manager import LoRAManager

        manager = LoRAManager(MagicMock(), MagicMock())
        assert manager.unload_adapter("missing") is False

    def test_multiple_adapters(self):
        from unittest.mock import MagicMock
        from zse_engine.orchestrator.lora_manager import LoRAManager

        manager = LoRAManager(MagicMock(), MagicMock())
        for i in range(3):
            manager.load_adapter_random(
                adapter_id=f"adapter-{i}", rank=16, alpha=16.0,
                num_layers=1, target_modules=["q_proj"],
                hidden_size=64, num_heads=4, num_kv_heads=4, head_dim=16,
            )

        assert manager.num_adapters == 3
        ids = manager.list_adapters()
        assert "adapter-0" in ids
        assert "adapter-2" in ids

    def test_stats(self):
        from unittest.mock import MagicMock
        from zse_engine.orchestrator.lora_manager import LoRAManager

        manager = LoRAManager(MagicMock(), MagicMock())
        manager.load_adapter_random(
            adapter_id="a", rank=8, alpha=8.0,
            num_layers=2, target_modules=["q_proj", "v_proj"],
            hidden_size=64, num_heads=4, num_kv_heads=4, head_dim=16,
        )
        s = manager.stats()
        assert s["num_adapters"] == 1
        assert "a" in s["adapter_ids"]


# ======================================================================
# Integration: LoRA-aware ModelRunner
# ======================================================================

class TestModelRunnerLoRA:
    def test_model_runner_accepts_lora_manager(self):
        """ModelRunner init should accept optional lora_manager."""
        from unittest.mock import MagicMock, patch
        from zse_engine.format.config import ModelConfig
        from zse_engine.orchestrator.model_runner import ModelRunner

        config = ModelConfig(
            num_layers=1, num_heads=4, num_kv_heads=4,
            head_dim=16, hidden_size=64, vocab_size=100,
        )

        mock_weights = MagicMock()
        mock_weights.find.return_value = None

        runner = ModelRunner(
            config=config,
            weights=mock_weights,
            kv_cache=MagicMock(),
            scratch=MagicMock(),
            gpu_mem=MagicMock(),
            kernels=MagicMock(),
            lora_manager=MagicMock(),
        )
        assert runner._lora_manager is not None

    def test_model_runner_works_without_lora(self):
        """ModelRunner should work fine with lora_manager=None (default)."""
        from unittest.mock import MagicMock
        from zse_engine.format.config import ModelConfig
        from zse_engine.orchestrator.model_runner import ModelRunner

        config = ModelConfig(
            num_layers=1, num_heads=4, num_kv_heads=4,
            head_dim=16, hidden_size=64, vocab_size=100,
        )

        mock_weights = MagicMock()
        mock_weights.find.return_value = None

        runner = ModelRunner(
            config=config,
            weights=mock_weights,
            kv_cache=MagicMock(),
            scratch=MagicMock(),
            gpu_mem=MagicMock(),
            kernels=MagicMock(),
        )
        assert runner._lora_manager is None


# ======================================================================
# BatchRunner LoRA Integration Tests
# ======================================================================

class TestBatchRunnerLoRA:
    def test_batch_runner_accepts_lora_manager(self):
        from unittest.mock import MagicMock
        from zse_engine.zstreamer.batch_runner import BatchRunner

        br = BatchRunner(
            model_runner=MagicMock(),
            sampler=MagicMock(),
            kv_cache=MagicMock(),
            scheduler=MagicMock(),
            vocab_size=100,
            lora_manager=MagicMock(),
        )
        assert br._lora_manager is not None

    def test_batch_runner_works_without_lora(self):
        from unittest.mock import MagicMock
        from zse_engine.zstreamer.batch_runner import BatchRunner

        br = BatchRunner(
            model_runner=MagicMock(),
            sampler=MagicMock(),
            kv_cache=MagicMock(),
            scheduler=MagicMock(),
            vocab_size=100,
        )
        assert br._lora_manager is None

    def test_resolve_lora_returns_none_when_no_manager(self):
        from unittest.mock import MagicMock
        from zse_engine.zstreamer.batch_runner import BatchRunner
        from zse_engine.zstreamer.request import InferenceRequest, GenerationParams

        br = BatchRunner(
            model_runner=MagicMock(),
            sampler=MagicMock(),
            kv_cache=MagicMock(),
            scheduler=MagicMock(),
            vocab_size=100,
        )
        req = InferenceRequest(
            request_id="r1", prompt_tokens=[1, 2],
            params=GenerationParams(lora_id="adapter-1"),
        )
        assert br._resolve_lora(req) is None

    def test_resolve_lora_returns_adapter(self):
        from unittest.mock import MagicMock
        from zse_engine.zstreamer.batch_runner import BatchRunner
        from zse_engine.zstreamer.request import InferenceRequest, GenerationParams

        mock_manager = MagicMock()
        mock_adapter = MagicMock()
        mock_manager.get_adapter.return_value = mock_adapter

        br = BatchRunner(
            model_runner=MagicMock(),
            sampler=MagicMock(),
            kv_cache=MagicMock(),
            scheduler=MagicMock(),
            vocab_size=100,
            lora_manager=mock_manager,
        )
        req = InferenceRequest(
            request_id="r1", prompt_tokens=[1, 2],
            params=GenerationParams(lora_id="adapter-1"),
        )
        result = br._resolve_lora(req)
        assert result is mock_adapter
        mock_manager.get_adapter.assert_called_once_with("adapter-1")

    def test_resolve_lora_returns_none_no_lora_id(self):
        from unittest.mock import MagicMock
        from zse_engine.zstreamer.batch_runner import BatchRunner
        from zse_engine.zstreamer.request import InferenceRequest

        br = BatchRunner(
            model_runner=MagicMock(),
            sampler=MagicMock(),
            kv_cache=MagicMock(),
            scheduler=MagicMock(),
            vocab_size=100,
            lora_manager=MagicMock(),
        )
        req = InferenceRequest(request_id="r1", prompt_tokens=[1, 2])
        assert br._resolve_lora(req) is None


# ======================================================================
# LoRA Manager Thread Safety Tests
# ======================================================================

class TestLoRAManagerThreadSafety:
    def test_has_lock(self):
        from unittest.mock import MagicMock
        from zse_engine.orchestrator.lora_manager import LoRAManager
        import threading

        manager = LoRAManager(MagicMock(), MagicMock())
        assert isinstance(manager._lock, type(threading.Lock()))

    def test_concurrent_load_unload(self):
        """Load and unload adapters from multiple threads."""
        from unittest.mock import MagicMock
        from zse_engine.orchestrator.lora_manager import LoRAManager
        import threading

        manager = LoRAManager(MagicMock(), MagicMock())
        errors = []

        def load_and_unload(i):
            try:
                adapter_id = f"adapter-{i}"
                manager.load_adapter_random(
                    adapter_id=adapter_id, rank=8, alpha=8.0,
                    num_layers=1, target_modules=["q_proj"],
                    hidden_size=64, num_heads=4, num_kv_heads=4, head_dim=16,
                )
                assert manager.has_adapter(adapter_id)
                manager.unload_adapter(adapter_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=load_and_unload, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ======================================================================
# LoRA Manager load_adapter_from_file Tests
# ======================================================================

class TestLoRAManagerFromFile:
    def test_load_from_file(self):
        from unittest.mock import MagicMock
        from zse_engine.orchestrator.lora_manager import LoRAManager
        from zse_engine.orchestrator.lora_weights import LoRAAdapter
        from zse_engine.format.lora_format import save_lora
        import tempfile
        import struct

        # Create a test .zse-lora file
        adapter = LoRAAdapter(
            adapter_id="test", rank=4, alpha=4.0,
            target_modules=["q_proj"], num_layers=1,
        )
        in_features, out_features = 32, 32
        weight_data = {
            (0, "q_proj"): (
                struct.pack(f'<{4 * in_features}e', *[0.1] * (4 * in_features)),
                struct.pack(f'<{out_features * 4}e', *[0.2] * (out_features * 4)),
            ),
        }

        with tempfile.NamedTemporaryFile(suffix='.zse-lora', delete=False) as f:
            path = f.name
        try:
            save_lora(path, adapter, weight_data)

            # Mock GPU — load_adapter_from_dict will try to allocate
            mock_gpu = MagicMock()
            mock_tensor = MagicMock()
            mock_tensor.data_ptr = 0x1000
            mock_gpu.allocate.return_value = mock_tensor

            manager = LoRAManager(mock_gpu, MagicMock())
            loaded = manager.load_adapter_from_file(
                adapter_id="from-file",
                path=path,
                weight_shapes={"q_proj": (in_features, out_features)},
            )

            assert loaded.rank == 4
            assert manager.has_adapter("from-file")
            assert loaded.has_weight(0, "q_proj")
        finally:
            import os
            os.unlink(path)


# ======================================================================
# Speculative Runner LoRA Tests
# ======================================================================

class TestSpecRunnerLoRA:
    def test_spec_runner_accepts_lora_manager(self):
        from unittest.mock import MagicMock
        from zse_engine.zstreamer.spec_runner import SpeculativeRunner

        spec = SpeculativeRunner(
            model_runner=MagicMock(),
            kv_cache=MagicMock(),
            draft_model=MagicMock(),
            verifier=MagicMock(),
            vocab_size=100,
            lora_manager=MagicMock(),
        )
        assert spec._lora_manager is not None

    def test_spec_runner_works_without_lora(self):
        from unittest.mock import MagicMock
        from zse_engine.zstreamer.spec_runner import SpeculativeRunner

        spec = SpeculativeRunner(
            model_runner=MagicMock(),
            kv_cache=MagicMock(),
            draft_model=MagicMock(),
            verifier=MagicMock(),
            vocab_size=100,
        )
        assert spec._lora_manager is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
