"""Tests for Tier 1 compiler bug fixes.

Covers:
1. NVRTC arch detection helper (RuntimeCompiler._detect_cuda_arch)
2. Block-reduce shared-mem name collision (multiple block_reduce in one kernel)
3. WMMA fragment counter is per-codegen-instance (not class-level)
4. Fusion input/output detection — clear error on multi-input kernels, honors chain hint
5. Tile load/store boundary predication (bound_row / bound_col)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import zse_compiler as zse
from zse_compiler.codegen.cuda import CUDACodegen
from zse_compiler.codegen.rocm import ROCmCodegen
from zse_compiler.codegen.metal import MetalCodegen
from zse_compiler.ir.fusion import FusionPass
from zse_compiler.ir.nodes import IRTileLoad, IRTileStore, IRConst, IRVar


# --- Fix 1: NVRTC arch detection ---

def test_arch_cache_fallback():
    """When driver is missing/broken, helper falls back to compute_80."""
    from zse_compiler.runtime.compiler import RuntimeCompiler
    # Clear cache so we hit the detection path
    RuntimeCompiler._arch_cache.pop(99, None)

    class FakeDevice:
        value = 99

    class BrokenDriver:
        def cuDeviceGetAttribute(self, *a, **kw):
            raise RuntimeError("no driver")

    arch = RuntimeCompiler._detect_cuda_arch(BrokenDriver(), FakeDevice())
    assert arch == "compute_80"


def test_arch_cache_returns_per_device():
    """Different compute capabilities produce different arch strings."""
    from zse_compiler.runtime.compiler import RuntimeCompiler
    import ctypes

    class FakeDevice:
        def __init__(self, ordinal):
            self.value = ordinal

    class FakeDriver:
        def __init__(self, major, minor):
            self._major, self._minor = major, minor

        def cuDeviceGetAttribute(self, ptr, attr_id, device):
            # ptr is a ctypes pointer; write the right value
            target = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))
            if attr_id == 75:
                target[0] = self._major
            elif attr_id == 76:
                target[0] = self._minor
            return 0

    RuntimeCompiler._arch_cache.pop(7, None)
    RuntimeCompiler._arch_cache.pop(8, None)
    assert RuntimeCompiler._detect_cuda_arch(FakeDriver(7, 5), FakeDevice(7)) == "compute_75"
    assert RuntimeCompiler._detect_cuda_arch(FakeDriver(9, 0), FakeDevice(8)) == "compute_90"


# --- Fix 2: Block-reduce shared-mem name collision ---

@zse.kernel
def two_block_reductions(x: zse.Tensor, out: zse.Tensor, n: int):
    """Kernel that uses block_reduce twice — must produce distinct shared arrays."""
    tid = zse.thread_id(0)
    val = x[tid]
    s1 = zse.block_reduce_sum(val)
    s2 = zse.block_reduce_max(val)
    out[tid] = s1 + s2


def test_block_reduce_unique_smem_names_cuda():
    src = two_block_reductions.source("cuda")
    # Both shared arrays must appear with distinct suffixes
    assert "_zse_bsmem_1[32]" in src
    assert "_zse_bsmem_2[32]" in src


def test_block_reduce_unique_smem_names_rocm():
    src = two_block_reductions.source("rocm")
    assert "_zse_bsmem_1[" in src
    assert "_zse_bsmem_2[" in src


def test_block_reduce_unique_smem_names_metal():
    src = two_block_reductions.source("metal")
    assert "_zse_bsmem_1[32]" in src
    assert "_zse_bsmem_2[32]" in src


def test_block_reduce_resets_per_codegen():
    """Running generate() twice must restart counter at 0 (no leak across kernels)."""
    cg = CUDACodegen()
    s1 = cg.generate(two_block_reductions.ir)
    s2 = cg.generate(two_block_reductions.ir)
    assert s1 == s2  # deterministic — would differ if counter leaked


# --- Fix 3: WMMA counter is instance-level ---

def test_wmma_counter_is_instance_level():
    """Two CUDACodegen instances must not share fragment counter state."""
    cg1 = CUDACodegen()
    cg2 = CUDACodegen()
    # No class-level _wmma_counter attribute survives the fix
    assert not hasattr(CUDACodegen, "_wmma_counter") or \
           getattr(CUDACodegen, "_wmma_counter", None) is None
    assert cg1._wmma_frag_counter == 0
    assert cg2._wmma_frag_counter == 0
    cg1._wmma_frag_counter = 42
    assert cg2._wmma_frag_counter == 0  # independent


# --- Fix 4: Fusion role detection ---

@zse.kernel
def k_unary_a(x: zse.Tensor, out: zse.Tensor):
    idx = zse.global_id(0)
    out[idx] = x[idx] * 2.0


@zse.kernel
def k_unary_b(x: zse.Tensor, out: zse.Tensor):
    idx = zse.global_id(0)
    out[idx] = x[idx] + 1.0


@zse.kernel
def k_two_inputs(x: zse.Tensor, residual: zse.Tensor, out: zse.Tensor):
    idx = zse.global_id(0)
    out[idx] = x[idx] + residual[idx]


def test_fusion_unambiguous_single_input():
    fused = zse.fuse([k_unary_a, k_unary_b], name="fab")
    assert fused.ir.name == "fab"


def test_fusion_rejects_ambiguous_multi_input():
    with pytest.raises(ValueError, match="multiple input tensors"):
        zse.fuse([k_unary_a, k_two_inputs], name="bad")


def test_fusion_honors_chain_hint():
    """With explicit chain, multi-input kernel is fusable."""
    fused = zse.fuse([k_unary_a, k_two_inputs], name="good", chain=["x"])
    # Should produce a valid IR
    assert fused.ir.name == "good"
    # And the codegen must succeed
    src = fused.source("cuda")
    assert "good" in src


def test_fusion_rejects_bad_chain_name():
    with pytest.raises(ValueError, match="not a tensor parameter"):
        zse.fuse([k_unary_a, k_two_inputs], name="bad", chain=["does_not_exist"])


def test_fusion_chain_wrong_length():
    with pytest.raises(ValueError, match="chain must have length"):
        zse.fuse([k_unary_a, k_unary_b], chain=["x", "y"])  # too long for 2 kernels


# --- Fix 5: Tile load/store boundary predication ---

def _make_tile_load(with_bounds):
    return IRTileLoad(
        tensor=IRVar(name="A"),
        tile_row=IRConst(value=0),
        tile_col=IRConst(value=0),
        tile_size=IRConst(value=16),
        shared_buf=IRVar(name="tile"),
        bound_row=IRConst(value=100) if with_bounds else None,
        bound_col=IRConst(value=200) if with_bounds else None,
    )


def _make_tile_store(with_bounds):
    return IRTileStore(
        shared_buf=IRVar(name="tile"),
        tensor=IRVar(name="C"),
        tile_row=IRConst(value=0),
        tile_col=IRConst(value=0),
        tile_size=IRConst(value=16),
        bound_row=IRConst(value=100) if with_bounds else None,
        bound_col=IRConst(value=200) if with_bounds else None,
    )


def test_tile_load_unbounded_cuda_unchanged():
    cg = CUDACodegen()
    cg._var_types = {}
    out = cg._emit_tile_load(_make_tile_load(with_bounds=False))
    assert "_zse_gr" not in out  # no boundary guard
    assert "tile[" in out


def test_tile_load_bounded_cuda_emits_guard():
    cg = CUDACodegen()
    cg._var_types = {}
    out = cg._emit_tile_load(_make_tile_load(with_bounds=True))
    assert "_zse_gr" in out
    assert "_zse_gc" in out
    assert "0.0f" in out  # zero-fill for OOB threads


def test_tile_store_bounded_cuda_emits_guard():
    cg = CUDACodegen()
    cg._var_types = {}
    out = cg._emit_tile_store(_make_tile_store(with_bounds=True))
    assert "if (_zse_gr <" in out


def test_tile_load_bounded_rocm_emits_guard():
    cg = ROCmCodegen()
    cg._var_types = {}
    out = cg._emit_tile_load(_make_tile_load(with_bounds=True))
    assert "_zse_gr" in out
    assert "0.0f" in out


def test_tile_load_bounded_metal_emits_guard():
    cg = MetalCodegen()
    cg._var_types = {}
    out = cg._emit_tile_load(_make_tile_load(with_bounds=True))
    assert "_zse_gr" in out
    assert "0.0f" in out


def test_tile_load_parser_accepts_bounds():
    """zse.tile_load(...) with 7 args parses bound_row + bound_col."""
    @zse.kernel
    def k_bounded(A: zse.Tensor, tile: zse.Tensor, out: zse.Tensor, M: int, N: int):
        zse.tile_load(A, 0, 0, 16, tile, M, N)
        out[0] = tile[0]

    src = k_bounded.source("cuda")
    # The boundary guard must appear in generated code
    assert "_zse_gr" in src
