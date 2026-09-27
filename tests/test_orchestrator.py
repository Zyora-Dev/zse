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

class TestConcurrentBenchmark:
    @pytest.mark.parametrize("model", ["qwen2.5-7b", "unsupported"])
    def test_model_selector_spawn(self, model):
        import importlib.util
        from unittest.mock import Mock, patch

        path = os.path.join(os.path.dirname(__file__), "test_modal_b200_ab2.py")
        spec = importlib.util.spec_from_file_location("benchmark_model_test", path)
        module = importlib.util.module_from_spec(spec)
        modal = Mock()
        modal.App.return_value.local_entrypoint.return_value = lambda function: function
        with patch.dict(sys.modules, {"modal": modal}):
            spec.loader.exec_module(module)
        if model == "unsupported":
            with pytest.raises(ValueError, match="Unsupported benchmark model"):
                module.main(model=model, sampler_ab=True, spawn=True)
            module.ab2.spawn.assert_not_called()
        else:
            module.main(model=model, sampler_ab=True, spawn=True)
            module.ab2.spawn.assert_called_once_with(12, False, False, False, True, model, False, False)

    @pytest.mark.parametrize("failure", [None, "request_error", "parity"])
    def test_overall_matrix_and_checkpoints(self, failure):
        import importlib.util
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from zse_engine.orchestrator.sampler import Sampler

        path = os.path.join(os.path.dirname(__file__), "test_modal_b200_ab2.py")
        spec = importlib.util.spec_from_file_location("benchmark_overall_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": Mock()}):
            spec.loader.exec_module(module)
        engines = []

        def build_engine(disable_medium, max_prefill_per_step):
            runner = SimpleNamespace(_graph_runner=object(),
                                     _batched_graph_runners={max_prefill_per_step: object()})

            def destroy_graph():
                runner._graph_runner = None
                runner._batched_graph_runners = {}

            runner.destroy_graph = destroy_graph
            engine = SimpleNamespace(_model_runner=runner, _sampler=Sampler(),
                                     _tokenizer=SimpleNamespace(decode=lambda tokens: "sample"),
                                     destroy=Mock())
            engines.append(engine)
            return engine

        def measure(engine, prompts, max_tokens, output_tokens=None, **params):
            if failure == "request_error":
                raise RuntimeError("request failed")
            if output_tokens is not None:
                token = 9 if failure == "parity" and engine._model_runner._graph_runner else 7
                for row in output_tokens:
                    row.extend([token] * max_tokens)
            return dict(aggregate_tps=10.0, generated_tokens=len(prompts) * max_tokens,
                        ttft_median_ms=2.0, itl_median_ms=3.0, ttft_p95_ms=4.0,
                        itl_p95_ms=5.0, completed_requests=len(prompts), failed_requests=0)

        persist = Mock()
        with patch.object(module, "measure_batch", side_effect=measure):
            result = module.benchmark_overall(build_engine, ["prompt"] * 12,
                                             lambda engine: {"device_used_bytes": 123}, persist)
        assert len(result["cases"]) == len(engines) == persist.call_count == 12
        for engine in engines:
            engine.destroy.assert_called_once_with()
        for case in result["cases"]:
            if failure == "request_error":
                assert case["status"] == "failed"
                assert "request failed" in case["error"]
            else:
                assert case["status"] == "completed"
                assert len(case["runs"]) == 3
                assert case["median"]["aggregate_tps"] == 10.0
                for sample in case["runs"]:
                    assert sample["full_length"] is True
                    assert sample["matches_graph_off"] == (failure != "parity" or not case["graphs"])

    @pytest.mark.parametrize("failure", [None, "parity", "early"])
    def test_sampler_comparison_gates_and_cleanup(self, failure):
        import importlib.util
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from zse_engine.orchestrator.sampler import Sampler

        path = os.path.join(os.path.dirname(__file__), "test_modal_b200_ab2.py")
        spec = importlib.util.spec_from_file_location("benchmark_sampler_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": Mock()}):
            spec.loader.exec_module(module)
        engines = []

        def build_engine(disable_medium, max_prefill_per_step):
            assert disable_medium is False
            assert max_prefill_per_step == 2
            engine = SimpleNamespace(_sampler=Sampler(), destroy=Mock(),
                                     _model_runner=SimpleNamespace(_batched_graph_runners={2: object()}))
            engines.append(engine)
            return engine

        def measure(engine, prompts, max_tokens, output_tokens=None, **params):
            assert params == dict(temperature=0.8, repetition_penalty=1.0, top_k=50, top_p=0.9)
            if output_tokens is not None:
                token = 9 if failure == "parity" and len(engines) == 2 else 7
                for row in output_tokens:
                    row.extend([token] * max_tokens)
            count = len(prompts) * max_tokens
            return dict(aggregate_tps=10.0, generated_tokens=count - (failure == "early"))

        with patch.object(module, "measure_batch", side_effect=measure):
            if failure:
                with pytest.raises(AssertionError):
                    module.compare_samplers(build_engine, ["one", "two"], max_tokens=4)
            else:
                result = module.compare_samplers(build_engine, ["one", "two"], max_tokens=4)
                assert result["token_parity"] is True
                assert len(engines) == 2
                assert result["compact"]["median_tps"] == 10.0
                assert len(result["reference"]["runs"]) == 3
        for engine in engines:
            engine.destroy.assert_called_once_with()

    def test_stage_timings_preserve_results_and_restore(self):
        import importlib.util
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        path = os.path.join(os.path.dirname(__file__), "test_modal_b200_ab2.py")
        spec = importlib.util.spec_from_file_location("benchmark_timings_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": Mock()}):
            spec.loader.exec_module(module)
        original = Mock(return_value=7)
        owner = SimpleNamespace(sample=original)
        clock = iter([1.0, 3.0, 4.0, 7.0])
        timings = module.StageTimings(clock=lambda: next(clock))
        timings.wrap(owner, "sample", "sampling")
        assert owner.sample(b"logits", temperature=0.8) == 7
        original.assert_called_once_with(b"logits", temperature=0.8)
        assert timings.seconds == {"sampling": 2.0}
        timings.reset()
        assert timings.seconds == timings.calls == {}
        original.side_effect = ValueError("failed")
        with pytest.raises(ValueError, match="failed"):
            owner.sample()
        assert timings.seconds == {"sampling": 3.0}
        assert timings.calls == {"sampling": 1}
        timings.restore()
        assert owner.sample is original

    @pytest.mark.parametrize("failure", [None, "incomplete", "request_error", "finish_error", "empty"])
    def test_metrics_and_failure_gates(self, failure):
        import importlib.util
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        path = os.path.join(os.path.dirname(__file__), "test_modal_b200_ab2.py")
        spec = importlib.util.spec_from_file_location("benchmark_metrics_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modal": Mock()}):
            spec.loader.exec_module(module)

        callbacks = []
        now = [0.0]

        def add_request(**kwargs):
            callbacks.append((kwargs["on_token"], kwargs["on_finish"]))

        def step():
            now[0] += 1.0
            for on_token, on_finish in callbacks:
                if failure != "empty":
                    on_token(7)
                if now[0] == 2.0 and failure != "incomplete":
                    reason = "ERROR" if failure == "finish_error" else "LENGTH"
                    on_finish(SimpleNamespace(finish_reason=SimpleNamespace(name=reason)))
            return SimpleNamespace(errors={"request": "failed"} if failure == "request_error" else {})

        engine = SimpleNamespace(add_request=add_request, step=step)
        if failure:
            with pytest.raises(RuntimeError):
                module.measure_batch(engine, ["one", "two"], 2, clock=lambda: now[0])
        else:
            metrics = module.measure_batch(engine, ["one", "two"], 2, clock=lambda: now[0])
            assert metrics == dict(aggregate_tps=2.0, ttft_median_ms=1000.0,
                                   itl_median_ms=1000.0, generated_tokens=4, elapsed_s=2.0,
                                   ttft_p95_ms=1000.0, itl_p95_ms=1000.0,
                                   completed_requests=2, failed_requests=0)


class TestBatchedDecodeGraph:
    @pytest.mark.parametrize("return_logits", [False, True])
    def test_capture_only_requested_batch_and_reuse(self, return_logits):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from zse_engine.orchestrator.model_runner import ModelRunner

        runner = ModelRunner.__new__(ModelRunner)
        runner._kv_cache = Mock()
        runner._kv_cache.get_attention_metadata.return_value = SimpleNamespace(
            block_tables=[[0], [1]], seq_lengths=[2, 2],
        )
        runner._gpu_mem = Mock()
        runner._gpu_mem.copy_device_to_host.return_value = struct.pack("<2i", 7, 9)
        runner._decode_token_buf = object()
        runner._decode_pos_buf = object()
        runner._decode_bt_buf = object()
        runner._decode_seqlens_buf = object()
        runner._graph_max_blocks = 4
        runner._graph_argmax_buf = SimpleNamespace(data_ptr=123)
        runner._batched_graph_runners = {}
        runner._bulk_download_logits = Mock(return_value=[b"row-one", b"row-two"])
        graph = Mock()

        def capture(batch_size, metadata):
            runner._batched_graph_runners[batch_size] = (graph, None)

        runner._capture_batched_graph = Mock(side_effect=capture)
        for _ in range(2):
            result = runner.batched_decode_graph(
                [1, 2], [10, 11], [1, 1], return_logits=return_logits,
            )
            assert result == ([b"row-one", b"row-two"] if return_logits else [7, 9])

        assert runner._capture_batched_graph.call_count == 1
        assert set(runner._batched_graph_runners) == {2}
        assert graph.replay.call_count == graph.sync.call_count == 2
        assert runner._kv_cache.extend_sequence.call_count == 4
        assert runner._bulk_download_logits.call_count == (2 if return_logits else 0)
        assert runner._gpu_mem.copy_device_to_host.call_count == (0 if return_logits else 2)


class TestInt4Dispatch:
    @pytest.mark.parametrize("backend,batch_size,group_size,disabled,expected", [
        ("rocm", 8, 12, False, "batched_dequant_gemv_int4"),
        ("rocm", 9, 12, False, "batched_dequant_gemv_int4"),
        ("rocm", 16, 12, False, "batched_dequant_gemv_int4"),
        ("rocm", 8, 24, False, "bgemv_int4_wave64"),
        ("rocm", 8, 128, False, "bgemv_int4_wave64_v2"),
        ("rocm", 9, 128, False, "bgemv_int4_wave64_m16"),
        ("rocm", 12, 128, False, "bgemv_int4_wave64_m16"),
        ("rocm", 16, 128, False, "bgemv_int4_wave64_m16"),
        ("rocm", 17, 128, False, "mfma_dequant_matmul_int4_v3"),
        ("rocm", 12, 128, True, "mfma_dequant_matmul_int4_v3"),
        ("cuda", 8, 128, False, "batched_dequant_gemv_int4"),
        ("cuda", 9, 128, False, "batched_dequant_gemv_int4"),
        ("cuda", 12, 128, False, "batched_dequant_gemv_int4"),
        ("cuda", 16, 128, False, "batched_dequant_gemv_int4"),
        ("cuda", 17, 128, False, "tiled_dequant_matmul_int4"),
        ("cuda", 12, 128, True, "tiled_dequant_matmul_int4"),
    ])
    def test_group_and_batch_boundaries(self, backend, batch_size, group_size,
                                       disabled, expected):
        from unittest.mock import Mock
        from zse_engine.orchestrator.model_runner import ModelRunner
        from zse_engine.orchestrator.weight_loader import GPUWeight

        runner = ModelRunner.__new__(ModelRunner)
        runner._backend = backend
        runner._disable_medium_gemv = disabled
        runner._kernels = Mock()
        runner._make_tensor_from_ptr = Mock()
        weight = GPUWeight(
            name="projection", shape=(128, 384), dtype="int4",
            data_ptr=1, data_nbytes=24576, scales_ptr=2, zeros_ptr=3,
            group_size=group_size,
        )
        runner._launch_matmul(None, None, weight, batch_size, 128, 384)
        assert runner._kernels.launch.call_count == 1
        assert runner._kernels.launch.call_args.args[0] == expected


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

    @pytest.mark.parametrize("top_k", [0, 1, 3, 50, 257, 300])
    @pytest.mark.parametrize("top_p", [0.0, 0.5, 0.9, 1.0])
    @pytest.mark.parametrize("temperature", [0.0, 0.8, 1.0, 1.7])
    def test_compact_sampling_matches_reference(self, top_k, top_p, temperature):
        import random
        from zse_engine.orchestrator.sampler import Sampler

        generator = random.Random(123)
        cases = [
            [generator.uniform(-9, 9) for token in range(257)],
            [float(token % 5) for token in range(257)],
            [1.0] * 257,
        ]
        for values in cases:
            data = self._make_logits(values)
            for past_tokens in (None, {1, 3, 99}, {1: 8, 3: 2, 99: 1, -1: 4, 500: 2}):
                optimized = Sampler(seed=20260927)
                reference = Sampler(seed=20260927)
                reference._sample_top_k = lambda *args: None
                params = dict(temperature=temperature, top_k=top_k, top_p=top_p,
                              repetition_penalty=1.1, past_tokens=past_tokens)
                for draw in range(12):
                    assert optimized.sample(data, len(values), **params) == reference.sample(data, len(values), **params)
                assert optimized._rng.getstate() == reference._rng.getstate()

    def test_compact_sampling_cutoff_ties_and_tail_fallback(self):
        from zse_engine.orchestrator.sampler import Sampler

        for draw in (0.0, 0.25, 0.5, 0.75, 0.9999999999999999, 1.0):
            optimized = Sampler()
            reference = Sampler()
            optimized._rng.random = lambda: draw
            reference._rng.random = lambda: draw
            reference._sample_top_k = lambda *args: None
            data = self._make_logits([-10, 2, 2, 2, 2, -10])
            assert optimized.sample(data, 6, top_k=1) == reference.sample(data, 6, top_k=1)

    def test_compact_sampling_nonfinite_fallback(self):
        from zse_engine.orchestrator.sampler import Sampler

        sampler = Sampler(seed=42)
        for value in (float('-inf'), float('inf'), float('nan')):
            state = sampler._rng.getstate()
            assert sampler._sample_top_k([1.0, value, 0.0], 1, 0.9) is None
            assert sampler._rng.getstate() == state

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
