"""Tier-3 — local_array + reinterpret + tensor-type uniformity.

Codegen-only tests (no GPU required). Verifies:
1. zse.local_array(N, dtype) — emits stack array in CUDA/HIP, `thread T arr[N]` in Metal.
2. zse.reinterpret(ptr, dtype) — emits typed pointer cast; LHS gets explicit pointer type.
3. New tensor type aliases (uint32_tensor, uint16_tensor, fp32_tensor, bf16_tensor) parse correctly.
"""

import pytest
import zse_compiler as zse


# ============================================================
# Fix 1 — zse.local_array
# ============================================================

@zse.kernel
def k_local_array(out: "int32_tensor"):
    tid = zse.global_id(0)
    buf = zse.local_array(8, zse.int32)
    buf[0] = tid
    buf[1] = tid + 1
    out[tid] = buf[0] + buf[1]


def test_local_array_cuda():
    src = k_local_array.source("cuda")
    assert "int buf[8];" in src
    assert "buf[0] = " in src
    assert "buf[1] = " in src


def test_local_array_rocm():
    src = k_local_array.source("rocm")
    assert "int buf[8];" in src


def test_local_array_metal():
    src = k_local_array.source("metal")
    assert "thread int buf[8];" in src


def test_local_array_float16():
    @zse.kernel
    def k(out: "half_tensor"):
        tid = zse.global_id(0)
        scratch = zse.local_array(4, zse.float16)
        scratch[0] = out[tid]
        out[tid] = scratch[0]
    src = k.source("cuda")
    assert "half scratch[4];" in src

    msrc = k.source("metal")
    assert "thread half scratch[4];" in msrc


def test_local_array_requires_int_literal_size():
    with pytest.raises(SyntaxError, match="integer literal"):
        @zse.kernel
        def k_bad(out: "int32_tensor", N: int):
            buf = zse.local_array(N, zse.int32)  # not a literal
            out[0] = buf[0]
        k_bad.source("cuda")


# ============================================================
# Fix 2 — zse.reinterpret
# ============================================================

@zse.kernel
def k_reinterpret(weights: "uint8_tensor", out: "int32_tensor", N: int):
    tid = zse.global_id(0)
    if tid >= N:
        return
    qp = zse.reinterpret(weights, zse.uint32)
    packed = qp[tid]
    out[tid] = packed


def test_reinterpret_cuda():
    src = k_reinterpret.source("cuda")
    # LHS should be explicit pointer type, not `auto` — guarantees Metal parity
    assert "unsigned int* qp = " in src
    assert "((unsigned int*)(weights))" in src


def test_reinterpret_rocm():
    src = k_reinterpret.source("rocm")
    assert "unsigned int* qp = " in src
    assert "((unsigned int*)(weights))" in src


def test_reinterpret_metal_preserves_device_addr_space():
    src = k_reinterpret.source("metal")
    # Metal kernel args are `device` — cast must preserve that address space
    assert "device uint* qp = " in src
    assert "((device uint*)(weights))" in src


def test_reinterpret_dtype_choices():
    @zse.kernel
    def k(buf: "uint8_tensor", out: "int32_tensor"):
        tid = zse.global_id(0)
        as_u16 = zse.reinterpret(buf, zse.uint16)
        out[tid] = as_u16[tid]
    assert "unsigned short* as_u16" in k.source("cuda")
    assert "device ushort* as_u16" in k.source("metal")


def test_reinterpret_arity_check():
    with pytest.raises(SyntaxError, match="requires 2 args"):
        @zse.kernel
        def k_bad(buf: "uint8_tensor", out: "int32_tensor"):
            qp = zse.reinterpret(buf)  # missing dtype
            out[0] = qp[0]
        k_bad.source("cuda")


# ============================================================
# Fix 5 — tensor type aliases (uniformity)
# ============================================================

def test_uint32_tensor_cuda():
    @zse.kernel
    def k(x: "uint32_tensor", out: "int32_tensor"):
        tid = zse.global_id(0)
        out[tid] = x[tid]
    src = k.source("cuda")
    assert "unsigned int* __restrict__ x" in src

    rsrc = k.source("rocm")
    assert "unsigned int* __restrict__ x" in rsrc

    msrc = k.source("metal")
    assert "device uint* x" in msrc


def test_uint16_tensor_all_backends():
    @zse.kernel
    def k(x: "uint16_tensor", out: "int32_tensor"):
        tid = zse.global_id(0)
        out[tid] = x[tid]
    assert "unsigned short* __restrict__ x" in k.source("cuda")
    assert "unsigned short* __restrict__ x" in k.source("rocm")
    assert "device ushort* x" in k.source("metal")


def test_fp32_tensor_alias():
    @zse.kernel
    def k(x: "fp32_tensor", out: "fp32_tensor"):
        tid = zse.global_id(0)
        out[tid] = x[tid] + 1.0
    assert "float* __restrict__ x" in k.source("cuda")
    assert "device float* x" in k.source("metal")


def test_bfloat16_tensor():
    @zse.kernel
    def k(x: "bf16_tensor", out: "bf16_tensor"):
        tid = zse.global_id(0)
        out[tid] = x[tid]
    assert "__nv_bfloat16* __restrict__ x" in k.source("cuda")
    assert "hip_bfloat16* __restrict__ x" in k.source("rocm")
    assert "device bfloat* x" in k.source("metal")


# ============================================================
# Integration: all three primitives in one INT4-style kernel
# ============================================================

@zse.kernel
def k_int4_style(weights: "uint8_tensor", out: "int32_tensor", N: int):
    """Mimics the inner loop of an INT4 dequant matmul:
       - reinterpret packed weights as uint32
       - allocate per-thread nibble scratch
       - unpack 8 nibbles into scratch
       - sum them and write out.
    """
    tid = zse.global_id(0)
    if tid >= N:
        return
    qp = zse.reinterpret(weights, zse.uint32)
    packed = qp[tid]
    buf = zse.local_array(8, zse.int32)
    zse.unpack_uint4(packed, buf, 0)
    acc = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7]
    out[tid] = acc


def test_integration_kernel_cuda():
    src = k_int4_style.source("cuda")
    assert "unsigned int* qp = " in src         # reinterpret LHS
    assert "((unsigned int*)(weights))" in src  # reinterpret expr
    assert "int buf[8];" in src                  # local array
    assert "_zse_uu4_" in src                    # unpack_uint4 lowering
    assert "buf[0]" in src and "buf[7]" in src   # buffer indexed


def test_integration_kernel_rocm():
    src = k_int4_style.source("rocm")
    assert "unsigned int* qp = " in src
    assert "int buf[8];" in src
    assert "_zse_uu4_" in src


def test_integration_kernel_metal():
    src = k_int4_style.source("metal")
    assert "device uint* qp = " in src
    assert "thread int buf[8];" in src
    assert "_zse_uu4_" in src


# ============================================================
# Public API surface
# ============================================================

def test_public_exports():
    assert hasattr(zse, "local_array")
    assert hasattr(zse, "reinterpret")
    assert hasattr(zse, "uint32")
    assert hasattr(zse, "uint16")
    assert hasattr(zse, "int16")
    assert callable(zse.local_array)
    assert callable(zse.reinterpret)
