"""Tier-2.5: zse.unpack_uint4 intrinsic — sibling of unpack_int4 without
sign extension. Matches the asymmetric INT4 quant format used by .zse
weights (unsigned nibbles 0..15 with separately-stored fp16 zero point).

Lowering: shift+mask only — NVRTC -> bfe.u32, HIPRTC -> v_bfe_u32,
Metal -> scalar shift+mask loop.
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
from zse_compiler.ir.nodes import IRUnpackUint4, IRUnpackInt4


# ---------- Parser ----------

def test_parser_recognizes_unpack_uint4():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_uint4(packed[0], out, 0)

    ir = KernelParser().parse(k)
    found = any(isinstance(s, IRUnpackUint4) for s in ir.body)
    assert found, "IRUnpackUint4 not emitted by parser"


def test_parser_rejects_wrong_arity():
    def k_bad(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_uint4(packed[0], out)  # missing base_idx

    with pytest.raises(SyntaxError, match="unpack_uint4 requires 3 args"):
        KernelParser().parse(k_bad)


def test_parser_distinguishes_int4_vs_uint4():
    """Both intrinsics must be recognized as distinct IR nodes."""
    def k(packed: zse.Tensor, out_s: zse.Tensor, out_u: zse.Tensor):
        zse.unpack_int4(packed[0], out_s, 0)
        zse.unpack_uint4(packed[0], out_u, 0)

    ir = KernelParser().parse(k)
    has_signed = any(isinstance(s, IRUnpackInt4) for s in ir.body)
    has_unsigned = any(isinstance(s, IRUnpackUint4) for s in ir.body)
    assert has_signed and has_unsigned


# ---------- Codegen (CUDA) ----------

def _make_kernel():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_uint4(packed[0], out, 0)
    return KernelParser().parse(k)


def test_cuda_codegen_emits_shift_mask_no_sign_extend():
    src = CUDACodegen().generate(_make_kernel())
    # Must contain shift + mask
    assert ">> (" in src and "& 0xF" in src
    # Must NOT contain sign-extend XOR trick (that's unpack_int4)
    assert "_zse_uu4_n_" in src
    # Make sure no occurrence of ^ 0x8 referencing the uu4 var
    assert "_zse_uu4_n_1 = (_zse_uu4_n_1 ^ 0x8)" not in src


def test_rocm_codegen_emits_shift_mask_no_sign_extend():
    src = ROCmCodegen().generate(_make_kernel())
    assert ">> (" in src and "& 0xF" in src
    assert "_zse_uu4_n_" in src
    assert "_zse_uu4_n_1 = (_zse_uu4_n_1 ^ 0x8)" not in src


def test_metal_codegen_emits_shift_mask_no_sign_extend():
    src = MetalCodegen().generate(_make_kernel())
    assert ">> (" in src and "& 0xF" in src
    assert "_zse_uu4_n_" in src
    assert "_zse_uu4_n_1 = (_zse_uu4_n_1 ^ 0x8)" not in src


# ---------- Counter collision isolation ----------

def test_counter_increments_on_multiple_calls_cuda():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_uint4(packed[0], out, 0)
        zse.unpack_uint4(packed[1], out, 8)

    src = CUDACodegen().generate(KernelParser().parse(k))
    # Two distinct suffixes
    assert "_zse_uu4_p_1" in src
    assert "_zse_uu4_p_2" in src


def test_counter_resets_between_generate_calls_cuda():
    def k(packed: zse.Tensor, out: zse.Tensor):
        zse.unpack_uint4(packed[0], out, 0)

    cg = CUDACodegen()
    src1 = cg.generate(KernelParser().parse(k))
    src2 = cg.generate(KernelParser().parse(k))
    # Counter must reset — both should produce _1, not _1 then _2
    assert src1.count("_zse_uu4_p_1") == src2.count("_zse_uu4_p_1")
    assert "_zse_uu4_p_2" not in src1
    assert "_zse_uu4_p_2" not in src2


def test_int4_and_uint4_counters_independent():
    """Signed and unsigned counters must not share namespace."""
    def k(packed: zse.Tensor, out_s: zse.Tensor, out_u: zse.Tensor):
        zse.unpack_int4(packed[0], out_s, 0)
        zse.unpack_uint4(packed[0], out_u, 0)

    src = CUDACodegen().generate(KernelParser().parse(k))
    # Both should start at index 1 in their own namespace
    assert "_zse_u4_p_1" in src       # signed
    assert "_zse_uu4_p_1" in src      # unsigned


# ---------- Variable base_idx ----------

def test_variable_base_idx_cuda():
    def k(packed: zse.Tensor, out: zse.Tensor, base: int):
        zse.unpack_uint4(packed[0], out, base)

    src = CUDACodegen().generate(KernelParser().parse(k))
    assert "_zse_uu4_o_1 = (int)(base)" in src


# ---------- Reference semantics ----------

def _ref_unpack_uint4(packed_u32: int) -> list:
    """Python reference: unsigned 4-bit nibbles 0..15."""
    out = []
    for i in range(8):
        out.append((packed_u32 >> (i * 4)) & 0xF)
    return out


def test_reference_semantics():
    # All zeros
    assert _ref_unpack_uint4(0x00000000) == [0] * 8
    # All ones
    assert _ref_unpack_uint4(0xFFFFFFFF) == [15] * 8
    # Sequential 0..7 → little-endian within u32
    assert _ref_unpack_uint4(0x76543210) == [0, 1, 2, 3, 4, 5, 6, 7]
    # High nibbles 8..15 must remain UNSIGNED (NOT sign-extended)
    assert _ref_unpack_uint4(0xFEDCBA98) == [8, 9, 10, 11, 12, 13, 14, 15]


# ---------- Public API export ----------

def test_public_api_export():
    assert hasattr(zse, "unpack_uint4")
    assert "unpack_uint4" in zse.__all__
