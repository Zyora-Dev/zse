"""Tier-2: zse.unpack_int4 intrinsic — codegen + parser tests across all 3 backends.

This is the foundational primitive for hand-tuned INT4 dequant matmul kernels.
One call replaces 8 separate shift/mask/sign-extend expressions and gives the
backend compiler a tight, recognizable pattern (NVRTC -> bfe.s32,
HIPRTC -> v_bfe_i32, Metal -> scalar loop).
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import zse_compiler as zse
from zse_compiler.ast_parser.parser import KernelParser
from zse_compiler.codegen.cuda import CUDACodegen
from zse_compiler.codegen.rocm import ROCmCodegen
from zse_compiler.codegen.metal import MetalCodegen
from zse_compiler.ir.nodes import IRUnpackInt4


# ---------- Parser ----------

def test_parser_recognizes_unpack_int4():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_int4(packed[0], out, 0)

    ir = KernelParser().parse(k)
    # IRUnpackInt4 should appear in the body
    found = any(isinstance(s, IRUnpackInt4) for s in ir.body)
    assert found, "IRUnpackInt4 not emitted by parser"


def test_parser_rejects_wrong_arity():
    def k_bad(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_int4(packed[0], out)  # missing base_idx

    with pytest.raises(SyntaxError, match="unpack_int4 requires 3 args"):
        KernelParser().parse(k_bad)


# ---------- Codegen helpers ----------

def _kernel():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_int4(packed[0], out, 0)
    return k


def _codegen(backend_cls):
    ir = KernelParser().parse(_kernel())
    return backend_cls().generate(ir)


# ---------- CUDA ----------

def test_unpack_int4_cuda_emits_shift_mask_sign_extend():
    src = _codegen(CUDACodegen)
    # Block scope opened
    assert "unsigned int _zse_u4_p_1" in src
    # Unrolled loop hint
    assert "#pragma unroll" in src
    # Shift-mask pattern
    assert ">> (_zse_u4_i_1 * 4)) & 0xF" in src
    # Sign-extend trick: (n ^ 0x8) - 0x8
    assert "(_zse_u4_n_1 ^ 0x8) - 0x8" in src


# ---------- ROCm ----------

def test_unpack_int4_rocm_emits_shift_mask_sign_extend():
    src = _codegen(ROCmCodegen)
    assert "unsigned int _zse_u4_p_1" in src
    assert "#pragma unroll" in src
    assert ">> (_zse_u4_i_1 * 4)) & 0xF" in src
    assert "(_zse_u4_n_1 ^ 0x8) - 0x8" in src


# ---------- Metal ----------

def test_unpack_int4_metal_emits_scalar_loop():
    src = _codegen(MetalCodegen)
    # Metal uses uint, not "unsigned int"
    assert "uint _zse_u4_p_1" in src
    assert ">> (_zse_u4_i_1 * 4)) & 0xF" in src
    assert "(_zse_u4_n_1 ^ 0x8) - 0x8" in src


# ---------- Counter isolation ----------

def test_unpack_int4_counter_increments_per_call():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_int4(packed[0], out, 0)
        zse.unpack_int4(packed[1], out, 8)

    ir = KernelParser().parse(k)
    src = CUDACodegen().generate(ir)
    # Two distinct counter suffixes
    assert "_zse_u4_p_1" in src
    assert "_zse_u4_p_2" in src
    assert "_zse_u4_i_1" in src
    assert "_zse_u4_i_2" in src


def test_unpack_int4_counter_resets_per_generate():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_int4(packed[0], out, 0)

    ir = KernelParser().parse(k)
    cg = CUDACodegen()
    src1 = cg.generate(ir)
    src2 = cg.generate(ir)
    # Both runs should produce identical output (counter reset)
    assert src1 == src2
    # And both should use _zse_u4_p_1 (not _2 in the second run)
    assert "_zse_u4_p_1" in src2
    assert "_zse_u4_p_2" not in src2


# ---------- ROCm counter isolation ----------

def test_unpack_int4_rocm_counter_isolation():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_int4(packed[0], out, 0)
        zse.unpack_int4(packed[1], out, 8)

    ir = KernelParser().parse(k)
    src = ROCmCodegen().generate(ir)
    assert "_zse_u4_p_1" in src and "_zse_u4_p_2" in src


# ---------- Metal counter isolation ----------

def test_unpack_int4_metal_counter_isolation():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_int4(packed[0], out, 0)
        zse.unpack_int4(packed[1], out, 8)

    ir = KernelParser().parse(k)
    src = MetalCodegen().generate(ir)
    assert "_zse_u4_p_1" in src and "_zse_u4_p_2" in src


# ---------- Variable base index ----------

def test_unpack_int4_accepts_expr_base_idx():
    def k(packed: zse.Tensor, out: zse.Tensor):
        i = zse.thread_id(0)
        zse.unpack_int4(packed[i], out, i * 8)

    ir = KernelParser().parse(k)
    src = CUDACodegen().generate(ir)
    # Variable expression in base index should appear in the cast
    assert "(int)((i * 8))" in src or "(int)(i * 8)" in src or "(int)((i*8))" in src


# ---------- Semantic correctness via host Python simulation ----------

def _unpack_int4_python(packed: int):
    """Reference: unpack 8 sign-extended nibbles from a u32."""
    out = []
    for i in range(8):
        n = (packed >> (i * 4)) & 0xF
        n = ((n ^ 0x8) - 0x8)  # sign-extend 4-bit -> int
        out.append(n)
    return out


def test_unpack_int4_reference_semantics():
    # The emitted C uses the same formula. Sanity-check the Python reference
    # so anyone reading the test sees what the kernel produces.
    assert _unpack_int4_python(0x00000000) == [0, 0, 0, 0, 0, 0, 0, 0]
    assert _unpack_int4_python(0xFFFFFFFF) == [-1, -1, -1, -1, -1, -1, -1, -1]
    # Nibbles 0..7 little-endian: byte0=0x10 -> nibble0=0, nibble1=1; byte1=0x32 -> 2,3; ...
    assert _unpack_int4_python(0x76543210) == [0, 1, 2, 3, 4, 5, 6, 7]
    # Mixed: 0x89ABCDEF -> [-1,-2,-3,-4,-5,-6,-7,-8] (high nibbles are negative)
    assert _unpack_int4_python(0x89ABCDEF) == [-1, -2, -3, -4, -5, -6, -7, -8]


# ---------- Public API export ----------

def test_unpack_int4_exported_from_zse_compiler():
    assert hasattr(zse, "unpack_int4")
    # Calling outside a kernel should not raise (stub returns None)
    assert zse.unpack_int4(0, None, 0) is None
