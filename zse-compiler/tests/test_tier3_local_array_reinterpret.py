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


@pytest.mark.parametrize("backend,unsigned_type", [
    ("cuda", "unsigned int"), ("rocm", "unsigned int"), ("metal", "uint"),
])
def test_typed_load_and_unsigned_arithmetic(backend, unsigned_type):
    @zse.kernel
    def typed_load(words: "uint32_tensor", indices: "int32_tensor", out: "uint32_tensor"):
        position = indices[0]
        packed = words[position]
        shifted = packed >> 4
        masked = shifted & 15
        selected = packed if position > 0 else masked
        out[0] = selected

    source = typed_load.source(backend)
    assert "int position = indices[0];" in source
    for name in ("packed", "shifted", "masked", "selected"):
        assert f"{unsigned_type} {name} = " in source


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_sibling_loop_temporaries(backend):
    @zse.kernel
    def sibling_loops(out: "int32_tensor"):
        for first in range(4):
            temporary = first + 1
            out[first] = temporary
        for second in range(4):
            temporary = second + 2
            out[second] = temporary

    source = sibling_loops.source(backend)
    assert source.count("int temporary") == 2


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_branch_value_visible_after_join(backend):
    @zse.kernel
    def branch_join(out: "int32_tensor", flag: int):
        if flag:
            chosen = 1
        else:
            chosen = 2
        out[0] = chosen

    source = branch_join.source(backend)
    assert source.index("int chosen;") < source.index("if (")
    assert "chosen = 1;" in source
    assert "chosen = 2;" in source


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_reject_possibly_unassigned_branch_value(backend):
    @zse.kernel
    def incomplete_branch(out: "int32_tensor", flag: int):
        if flag:
            chosen = 1
        out[0] = chosen

    with pytest.raises(ValueError, match="chosen.*before assignment"):
        incomplete_branch.source(backend)


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_scalar_promotions(backend):
    @zse.kernel
    def promotions(small: "uint16_tensor", halves: "half_tensor", out: "fp32_tensor", flag: int):
        narrow = small[0]
        widened = narrow + 65536
        value = halves[0]
        total = value + halves[1]
        out[flag] = total + widened

    source = promotions.source(backend)
    assert "int widened = " in source
    assert "float value = " in source
    assert "float total = " in source


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_scalar_parameter_reassignment(backend):
    @zse.kernel
    def parameter_update(out: "int32_tensor", flag: int):
        flag = flag + 1
        out[0] = flag

    source = parameter_update.source(backend)
    assert "int flag = " not in source
    assert "flag = (flag + 1);" in source


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_nested_branch_and_early_return(backend):
    @zse.kernel
    def nested(out: "fp32_tensor", flag: int):
        if flag < 0:
            return
        else:
            if flag > 1:
                value = 1.5
            else:
                value = 2
        total = 0.0
        for position in range(4):
            total += value
        out[0] = total

    source = nested.source(backend)
    assert source.index("float value;") < source.index("if (")
    assert source.count("float total") == 1
    assert "float value = " not in source


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_reject_value_from_zero_iteration_loop(backend):
    @zse.kernel
    def loop_value(out: "int32_tensor", count: int):
        for position in range(count):
            value = position
        out[0] = value

    with pytest.raises(ValueError, match="value.*before assignment"):
        loop_value.source(backend)


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_branch_pointer_preserves_backend_type(backend):
    @zse.kernel
    def pointer_branch(words: "uint8_tensor", out: "uint32_tensor", flag: int):
        if flag:
            pointer = zse.reinterpret(words, zse.uint32)
        else:
            pointer = zse.reinterpret(words, zse.uint32)
        out[0] = pointer[0]

    source = pointer_branch.source(backend)
    declaration = "device uint* pointer;" if backend == "metal" else "unsigned int* pointer;"
    assert source.index(declaration) < source.index("if (")


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_reject_branch_local_array_escape(backend):
    @zse.kernel
    def array_branch(out: "int32_tensor", flag: int):
        if flag:
            scratch = zse.local_array(4, zse.int32)
            scratch[0] = 1
        else:
            scratch = zse.local_array(4, zse.int32)
            scratch[0] = 2
        out[0] = scratch[0]

    with pytest.raises(ValueError, match="scratch.*before assignment"):
        array_branch.source(backend)


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_promoted_variable_propagates_to_dependents(backend):
    @zse.kernel
    def promotion_chain(out: "fp32_tensor", count: int):
        value = 0
        for position in range(count):
            copied = value
            result = copied + 1
            out[position] = result
            value = 1.5

    source = promotion_chain.source(backend)
    assert "float value = " in source
    assert "float copied = " in source
    assert "float result = " in source


def test_generated_scalar_cuda_syntax():
    import shutil
    import subprocess

    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ unavailable for generated scalar-source syntax check")

    @zse.kernel
    def scalar_syntax(words: "uint32_tensor", indices: "int32_tensor", out: "uint32_tensor", flag: int):
        position = indices[0]
        packed = words[position]
        if flag:
            chosen = packed >> 4
        else:
            chosen = packed & 15
        for first in range(4):
            temporary = chosen + first
            out[first] = temporary
        for second in range(4):
            temporary = chosen + second
            out[second] = temporary

    source = scalar_syntax.source("cuda").replace("__global__ ", "")
    result = subprocess.run(
        [compiler, "-x", "c++", "-std=c++17", "-fsyntax-only", "-"],
        input=source, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_sibling_loops_have_independent_types(backend):
    @zse.kernel
    def independent(words: "uint32_tensor", values: "fp32_tensor", out: "fp32_tensor"):
        for first in range(4):
            temporary = words[first]
            shifted = temporary >> 4
            out[first] = shifted
        for second in range(4):
            temporary = values[second]
            out[second] = temporary

    source = independent.source(backend)
    unsigned_type = "uint" if backend == "metal" else "unsigned int"
    assert f"{unsigned_type} temporary = words[first];" in source
    assert f"{unsigned_type} shifted = " in source
    assert "float temporary = values[second];" in source


@pytest.mark.parametrize("backend", ["cuda", "rocm", "metal"])
def test_shared_integer_load_inside_loop(backend):
    @zse.kernel
    def shared_loop(out: "uint32_tensor"):
        scratch = zse.shared_memory((4,), zse.uint32)
        scratch[0] = 255
        for position in range(1):
            packed = scratch[position]
            shifted = packed >> 4
            out[position] = shifted

    source = shared_loop.source(backend)
    unsigned_type = "uint" if backend == "metal" else "unsigned int"
    assert f"{unsigned_type} packed = scratch[position];" in source
    assert f"{unsigned_type} shifted = " in source
