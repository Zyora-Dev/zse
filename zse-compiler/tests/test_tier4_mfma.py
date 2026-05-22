"""Tier-4: AMD CDNA MFMA matrix-core intrinsics.

mfma_f32_16x16x16_f16 and mfma_f32_32x32x8_f16 — single-instruction tile
multiply-accumulate on AMD CDNA matrix cores. ROCm-only; CUDA/Metal raise.
"""

import pytest

import zse_compiler as zse
from zse_compiler.codegen.rocm import ROCmCodegen
from zse_compiler.ir.nodes import IRMfmaOp


# ---------- parser ----------

def test_parser_recognises_16x16x16():
    @zse.kernel
    def k(out: "fp32_tensor"):
        a_buf = zse.local_array(4, zse.float16)
        b_buf = zse.local_array(4, zse.float16)
        c_buf = zse.local_array(4, zse.float32)
        c_buf[0] = 0.0
        zse.mfma_f32_16x16x16_f16(a_buf, b_buf, c_buf)
        out[0] = c_buf[0]

    found = [n for n in k.ir.body if isinstance(n, IRMfmaOp)]
    assert len(found) == 1
    assert found[0].shape == "16x16x16_f16"


def test_parser_recognises_32x32x8():
    @zse.kernel
    def k(out: "fp32_tensor"):
        a_buf = zse.local_array(4, zse.float16)
        b_buf = zse.local_array(4, zse.float16)
        c_buf = zse.local_array(16, zse.float32)
        c_buf[0] = 0.0
        zse.mfma_f32_32x32x8_f16(a_buf, b_buf, c_buf)
        out[0] = c_buf[0]

    found = [n for n in k.ir.body if isinstance(n, IRMfmaOp)]
    assert len(found) == 1
    assert found[0].shape == "32x32x8_f16"


def test_parser_rejects_wrong_arity():
    with pytest.raises((SyntaxError, TypeError)):
        @zse.kernel
        def k(out: "fp32_tensor"):
            a_buf = zse.local_array(4, zse.float16)
            b_buf = zse.local_array(4, zse.float16)
            zse.mfma_f32_16x16x16_f16(a_buf, b_buf)   # missing c_buf
            out[0] = 0.0


# ---------- ROCm codegen ----------

def _k_16():
    @zse.kernel
    def k(out: "fp32_tensor"):
        a_buf = zse.local_array(4, zse.float16)
        b_buf = zse.local_array(4, zse.float16)
        c_buf = zse.local_array(4, zse.float32)
        c_buf[0] = 0.0
        c_buf[1] = 0.0
        c_buf[2] = 0.0
        c_buf[3] = 0.0
        zse.mfma_f32_16x16x16_f16(a_buf, b_buf, c_buf)
        out[0] = c_buf[0]
    return k


def _k_32():
    @zse.kernel
    def k(out: "fp32_tensor"):
        a_buf = zse.local_array(4, zse.float16)
        b_buf = zse.local_array(4, zse.float16)
        c_buf = zse.local_array(16, zse.float32)
        c_buf[0] = 0.0
        zse.mfma_f32_32x32x8_f16(a_buf, b_buf, c_buf)
        out[0] = c_buf[0]
    return k


def test_rocm_emits_16x16x16_builtin():
    src = _k_16().source(backend="rocm")
    assert "__builtin_amdgcn_mfma_f32_16x16x16f16" in src
    assert "ext_vector_type(4)" in src
    assert "_zse_mfma_a_1" in src
    assert "_zse_mfma_b_1" in src
    assert "_zse_mfma_c_1" in src
    assert "(a_buf)[0]" in src
    assert "_zse_mfma_i_1 < 4" in src


def test_rocm_emits_32x32x8_builtin_with_float32_acc():
    src = _k_32().source(backend="rocm")
    assert "__builtin_amdgcn_mfma_f32_32x32x8f16" in src
    assert "ext_vector_type(16)" in src
    assert "_zse_mfma_i_1 < 16" in src


def test_rocm_multiple_mfma_calls_independent_names():
    @zse.kernel
    def k(out: "fp32_tensor"):
        a = zse.local_array(4, zse.float16)
        b = zse.local_array(4, zse.float16)
        c1 = zse.local_array(4, zse.float32)
        c2 = zse.local_array(4, zse.float32)
        c1[0] = 0.0
        c2[0] = 0.0
        zse.mfma_f32_16x16x16_f16(a, b, c1)
        zse.mfma_f32_16x16x16_f16(a, b, c2)
        out[0] = c1[0] + c2[0]
    src = k.source(backend="rocm")
    assert "_zse_mfma_a_1" in src and "_zse_mfma_a_2" in src
    assert "_zse_mfma_c_1" in src and "_zse_mfma_c_2" in src


def test_rocm_counter_resets_across_generate():
    cg = ROCmCodegen()
    k1 = _k_16()
    s1 = cg.generate(k1.ir)
    s2 = cg.generate(k1.ir)
    assert "_zse_mfma_a_1" in s1
    assert "_zse_mfma_a_1" in s2
    assert "_zse_mfma_a_2" not in s1
    assert "_zse_mfma_a_2" not in s2


# ---------- CUDA / Metal — must raise clear error ----------

def test_cuda_raises_on_mfma():
    k = _k_16()
    with pytest.raises(NotImplementedError) as exc:
        k.source(backend="cuda")
    msg = str(exc.value).lower()
    assert "mfma" in msg
    assert "amd" in msg or "rocm" in msg
    assert "wmma" in msg


def test_metal_raises_on_mfma():
    k = _k_16()
    with pytest.raises(NotImplementedError) as exc:
        k.source(backend="metal")
    assert "mfma" in str(exc.value).lower()


# ---------- public API surface ----------

def test_public_api_exports():
    assert hasattr(zse, "mfma_f32_16x16x16_f16")
    assert hasattr(zse, "mfma_f32_32x32x8_f16")
    assert callable(zse.mfma_f32_16x16x16_f16)
    assert callable(zse.mfma_f32_32x32x8_f16)
