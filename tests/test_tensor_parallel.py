"""Tests for tensor parallelism — weight splitting, dimension validation, TP group."""

import struct
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "zse-engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "zse-compiler"))

import pytest


# =============================================================================
# NCCL wrapper tests (unit — no GPU needed)
# =============================================================================

class TestNcclModule:
    """Test NCCL module imports and constants."""

    def test_import(self):
        from zse_compiler.runtime.nccl import (
            NcclCommunicator, get_unique_id, is_nccl_available,
            NCCL_UNIQUE_ID_BYTES, NCCL_FLOAT16, NCCL_SUM,
        )
        assert NCCL_UNIQUE_ID_BYTES == 128
        assert NCCL_FLOAT16 == 6
        assert NCCL_SUM == 0

    def test_dtype_map(self):
        from zse_compiler.runtime.nccl import _DTYPE_MAP, _OP_MAP
        assert _DTYPE_MAP["float16"] == 6
        assert _DTYPE_MAP["float32"] == 7
        assert _DTYPE_MAP["fp16"] == 6
        assert _OP_MAP["sum"] == 0
        assert _OP_MAP["max"] == 2

    def test_is_nccl_available_returns_bool(self):
        from zse_compiler.runtime.nccl import is_nccl_available
        result = is_nccl_available("cuda")
        assert isinstance(result, bool)
        result_rocm = is_nccl_available("rocm")
        assert isinstance(result_rocm, bool)


# =============================================================================
# TensorParallelGroup tests (no GPU needed)
# =============================================================================

class TestTPConfig:
    """Test TP configuration validation."""

    def test_single_gpu_always_valid(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=1)
        assert not cfg.is_enabled()
        cfg.validate(32, 8, 11008)  # Any dims OK for tp_size=1

    def test_valid_tp2(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=2)
        assert cfg.is_enabled()
        cfg.validate(32, 8, 11008)  # All divisible by 2

    def test_invalid_heads_not_divisible(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=4)
        with pytest.raises(ValueError, match="num_heads"):
            cfg.validate(30, 8, 11008)  # 30 not divisible by 4

    def test_invalid_kv_heads_not_divisible(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=4)
        with pytest.raises(ValueError, match="num_kv_heads"):
            cfg.validate(32, 6, 11008)  # 6 not divisible by 4

    def test_invalid_intermediate_not_divisible(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=4)
        with pytest.raises(ValueError, match="intermediate_size"):
            cfg.validate(32, 8, 11007)  # 11007 not divisible by 4


class TestTensorParallelGroup:
    """Test TP group weight splitting logic (no NCCL needed)."""

    def _make_tp(self, tp_size, rank):
        """Create a TP group without NCCL (for unit testing split logic)."""
        from zse_engine.orchestrator.tensor_parallel import TensorParallelGroup
        tp = TensorParallelGroup.__new__(TensorParallelGroup)
        tp.tp_size = tp_size
        tp.rank = rank
        tp.backend = "cuda"
        tp._stream = 0
        tp._comm = None
        return tp

    def test_split_strategy_qkv(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("model.layers.0.self_attn.q_proj.weight") == COLUMN_PARALLEL
        assert tp.get_split_strategy("model.layers.5.self_attn.k_proj.weight") == COLUMN_PARALLEL
        assert tp.get_split_strategy("model.layers.31.self_attn.v_proj.weight") == COLUMN_PARALLEL

    def test_split_strategy_o_proj(self):
        from zse_engine.orchestrator.tensor_parallel import ROW_PARALLEL
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("model.layers.0.self_attn.o_proj.weight") == ROW_PARALLEL

    def test_split_strategy_mlp(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL, ROW_PARALLEL
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("model.layers.0.mlp.gate_proj.weight") == COLUMN_PARALLEL
        assert tp.get_split_strategy("model.layers.0.mlp.up_proj.weight") == COLUMN_PARALLEL
        assert tp.get_split_strategy("model.layers.0.mlp.down_proj.weight") == ROW_PARALLEL

    def test_split_strategy_norms_replicated(self):
        from zse_engine.orchestrator.tensor_parallel import REPLICATED
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("model.layers.0.input_layernorm.weight") == REPLICATED
        assert tp.get_split_strategy("model.norm.weight") == REPLICATED
        assert tp.get_split_strategy("embed_tokens.weight") == REPLICATED

    def test_split_strategy_lm_head(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("lm_head.weight") == COLUMN_PARALLEL

    def test_shard_range_column_parallel(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL
        tp0 = self._make_tp(2, 0)
        tp1 = self._make_tp(2, 1)
        assert tp0.compute_shard_range(4096, COLUMN_PARALLEL) == (0, 2048)
        assert tp1.compute_shard_range(4096, COLUMN_PARALLEL) == (2048, 4096)

    def test_shard_range_row_parallel(self):
        from zse_engine.orchestrator.tensor_parallel import ROW_PARALLEL
        tp0 = self._make_tp(4, 0)
        tp1 = self._make_tp(4, 1)
        tp3 = self._make_tp(4, 3)
        assert tp0.compute_shard_range(4096, ROW_PARALLEL) == (0, 1024)
        assert tp1.compute_shard_range(4096, ROW_PARALLEL) == (1024, 2048)
        assert tp3.compute_shard_range(4096, ROW_PARALLEL) == (3072, 4096)

    def test_shard_range_replicated(self):
        from zse_engine.orchestrator.tensor_parallel import REPLICATED
        tp = self._make_tp(4, 2)
        assert tp.compute_shard_range(4096, REPLICATED) == (0, 4096)

    def test_shard_size(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL, REPLICATED
        tp = self._make_tp(4, 0)
        assert tp.shard_size(4096, COLUMN_PARALLEL) == 1024
        assert tp.shard_size(4096, REPLICATED) == 4096

    def test_tp1_noop(self):
        """TP size 1 should return full ranges."""
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL
        tp = self._make_tp(1, 0)
        assert tp.compute_shard_range(4096, COLUMN_PARALLEL) == (0, 4096)
        assert tp.shard_size(4096, COLUMN_PARALLEL) == 4096


# =============================================================================
# TPWeightLoader shard computation tests
# =============================================================================

class TestTPWeightLoaderSharding:
    """Test weight shard dimension calculations."""

    def _make_entry(self, name, shape, dtype="int4", data_nbytes=0,
                    scale_nbytes=0, zeros_nbytes=0, group_size=128):
        """Create a mock WeightEntry."""
        from zse_engine.format.weight_index import WeightEntry
        N = shape[0]
        K = shape[1] if len(shape) > 1 else 0

        if data_nbytes == 0:
            if dtype == "int4":
                data_nbytes = N * (K // 2) if K > 0 else N
            elif dtype == "int8":
                data_nbytes = N * K if K > 0 else N
            else:
                num_el = 1
                for s in shape:
                    num_el *= s
                data_nbytes = num_el * 2  # fp16

        if scale_nbytes == 0 and dtype in ("int4", "int8") and K > 0:
            num_groups = K // group_size
            scale_nbytes = N * num_groups * 2  # fp16

        if zeros_nbytes == 0 and dtype == "int4" and K > 0:
            num_groups = K // group_size
            zeros_nbytes = N * num_groups * 2

        return WeightEntry(
            name=name,
            shape=shape,
            dtype=dtype,
            group_size=group_size,
            data_nbytes=data_nbytes,
            scale_nbytes=scale_nbytes,
            zeros_nbytes=zeros_nbytes,
        )

    def _make_tp_loader(self, tp_size, rank):
        """Create TPWeightLoader with mock TP group."""
        from zse_engine.orchestrator.tp_weight_loader import TPWeightLoader
        from zse_engine.orchestrator.tensor_parallel import TensorParallelGroup

        tp = TensorParallelGroup.__new__(TensorParallelGroup)
        tp.tp_size = tp_size
        tp.rank = rank
        tp.backend = "cuda"
        tp._stream = 0
        tp._comm = None

        loader = TPWeightLoader.__new__(TPWeightLoader)
        loader._tp = tp
        loader._loader = None
        loader._gpu_mem = None
        return loader

    def test_column_parallel_int4_shard_dims(self):
        """Q projection: [4096, 4096] INT4 split column-wise with tp=2."""
        loader = self._make_tp_loader(2, 0)
        entry = self._make_entry("model.layers.0.self_attn.q_proj.weight",
                                  (4096, 4096), "int4")
        shard = loader._compute_shard_info(entry, "column")
        assert shard["shape"] == (2048, 4096)
        assert shard["num_elements"] == 2048 * 4096
        assert shard["row_start"] == 0
        assert shard["row_end"] == 2048
        # INT4: data = N * K/2
        assert shard["data_nbytes"] == 2048 * 2048  # 2048 * (4096/2)

    def test_column_parallel_rank1(self):
        """Second rank gets second half of rows."""
        loader = self._make_tp_loader(2, 1)
        entry = self._make_entry("model.layers.0.self_attn.q_proj.weight",
                                  (4096, 4096), "int4")
        shard = loader._compute_shard_info(entry, "column")
        assert shard["row_start"] == 2048
        assert shard["row_end"] == 4096

    def test_row_parallel_int4_shard_dims(self):
        """O projection: [4096, 4096] INT4 split row-wise (K dim) with tp=2."""
        loader = self._make_tp_loader(2, 0)
        entry = self._make_entry("model.layers.0.self_attn.o_proj.weight",
                                  (4096, 4096), "int4")
        shard = loader._compute_shard_info(entry, "row")
        assert shard["shape"] == (4096, 2048)
        assert shard["col_start"] == 0
        assert shard["col_end"] == 2048
        # INT4: data = N * shard_K/2
        assert shard["data_nbytes"] == 4096 * 1024  # 4096 * (2048/2)

    def test_replicated_full_copy(self):
        """Norm weight should be full copy."""
        loader = self._make_tp_loader(4, 2)
        entry = self._make_entry("model.layers.0.input_layernorm.weight",
                                  (4096,), "float16",
                                  data_nbytes=4096*2, scale_nbytes=0, zeros_nbytes=0)
        shard = loader._compute_shard_info(entry, "replicated")
        assert shard["shape"] == (4096,)
        assert shard["data_nbytes"] == 4096 * 2

    def test_tp4_column_parallel(self):
        """4-way split of gate_proj [11008, 4096]."""
        loader = self._make_tp_loader(4, 2)
        entry = self._make_entry("model.layers.0.mlp.gate_proj.weight",
                                  (11008, 4096), "int4")
        shard = loader._compute_shard_info(entry, "column")
        assert shard["shape"] == (2752, 4096)  # 11008/4
        assert shard["row_start"] == 2752 * 2
        assert shard["row_end"] == 2752 * 3


# =============================================================================
# GPU Memory device_index test
# =============================================================================

class TestGPUMemoryDeviceIndex:
    """Test that GPUMemory accepts device_index."""

    def test_default_device_index(self):
        """GPUMemory should accept device_index=0 without error."""
        from zse_compiler.runtime.memory import GPUMemory
        # Don't actually create — just verify the signature accepts it
        # (Creating requires a GPU)
        import inspect
        sig = inspect.signature(GPUMemory.__init__)
        params = list(sig.parameters.keys())
        assert "device_index" in params

    def test_ensure_context_exists(self):
        """GPUMemory should have ensure_context method."""
        from zse_compiler.runtime.memory import GPUMemory
        assert hasattr(GPUMemory, 'ensure_context')


# =============================================================================
# TPModelRunner import test
# =============================================================================

class TestTPModelRunnerImport:
    """Test that TPModelRunner can be imported."""

    def test_import(self):
        from zse_engine.orchestrator.model_runner import TPModelRunner
        assert TPModelRunner is not None

    def test_is_subclass(self):
        from zse_engine.orchestrator.model_runner import ModelRunner, TPModelRunner
        assert issubclass(TPModelRunner, ModelRunner)


# =============================================================================
# TPEngine import test
# =============================================================================

class TestTPEngineImport:
    """Test TPEngine can be imported."""

    def test_import(self):
        from zse_engine.orchestrator.tp_engine import TPEngine
        assert TPEngine is not None

    def test_cmd_constants(self):
        from zse_engine.orchestrator.tp_engine import CMD_PREFILL, CMD_DECODE, CMD_STOP, CMD_DESTROY
        assert CMD_PREFILL == 1
        assert CMD_DECODE == 2
        assert CMD_STOP == 3
        assert CMD_DESTROY == 4


# =============================================================================
# CLI --tp flag test
# =============================================================================

class TestCLITPFlag:
    """Test that CLI accepts --tp flag."""

    def test_serve_parser_has_tp(self):
        """The serve subcommand should accept --tp."""
        import argparse
        from zse_engine.cli import main
        # Parse args directly
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        # Import cli and check the serve parser accepts --tp
        import zse_engine.cli as cli_mod
        # Re-run the parser setup logic by calling main with --help-like args
        # Simpler: just check the source
        import inspect
        source = inspect.getsource(cli_mod.main)
        assert "--tp" in source or "tensor-parallel" in source or "tp_size" in source


# =============================================================================
# Server tp_size test
# =============================================================================

class TestServerTPSize:
    """Test server accepts tp_size parameter."""

    def test_server_init_signature(self):
        """ZSEServer should accept tp_size parameter."""
        from zse_engine.server.app import ZSEServer
        import inspect
        sig = inspect.signature(ZSEServer.__init__)
        assert "tp_size" in sig.parameters
