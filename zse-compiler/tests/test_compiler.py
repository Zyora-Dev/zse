"""Test ZSE Kernel Compiler — end-to-end: Python → IR → CUDA/ROCm/Metal source."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import zse_compiler as zse


# --- Test 1: Simple vector add ---

@zse.kernel
def vector_add(a: zse.Tensor, b: zse.Tensor, out: zse.Tensor):
    idx = zse.global_id(0)
    out[idx] = a[idx] + b[idx]


# --- Test 2: Dot product with shared memory ---

@zse.kernel
def dot_product(a: zse.Tensor, b: zse.Tensor, out: zse.Tensor):
    tid = zse.thread_id(0)
    bid = zse.block_id(0)
    bdim = zse.block_dim(0)
    idx = bid * bdim + tid
    temp = a[idx] * b[idx]
    zse.atomic_add(out[0], temp)


# --- Test 3: Matrix multiply (naive) ---

@zse.kernel
def matmul_naive(a: zse.Tensor, b: zse.Tensor, out: zse.Tensor, n: int):
    row = zse.block_id(0)
    col = zse.block_id(1)
    acc: float = 0.0
    for k in range(n):
        acc = acc + a[row * n + k] * b[k * n + col]
    out[row * n + col] = acc


# --- Test 4: RMSNorm (needed for LLM inference) ---

@zse.kernel
def rmsnorm(x: zse.Tensor, weight: zse.Tensor, out: zse.Tensor, n: int):
    tid = zse.thread_id(0)
    bid = zse.block_id(0)
    offset = bid * n + tid
    # Compute sum of squares (simplified — real version needs reduction)
    val = x[offset]
    ss = val * val
    # Normalize
    out[offset] = weight[tid] * (val * zse.rsqrt(ss + 0.00001))


def test_codegen_all_backends():
    """Test code generation for all three backends."""
    kernels = [
        ("vector_add", vector_add),
        ("dot_product", dot_product),
        ("matmul_naive", matmul_naive),
        ("rmsnorm", rmsnorm),
    ]

    backends = ["cuda", "rocm", "metal"]

    for name, k in kernels:
        print(f"\n{'='*60}")
        print(f"KERNEL: {name}")
        print(f"{'='*60}")

        for backend in backends:
            print(f"\n--- {backend.upper()} ---")
            src = k.source(backend)
            print(src)

        # Verify IR was parsed
        assert k.ir is not None
        assert k.ir.name == name
        print(f"\n✓ {name}: IR parsed, all backends generated")


def test_device_detection():
    """Test GPU device detection."""
    backend = zse.detect_backend()
    print(f"\nDetected backend: {backend}")

    devices = zse.get_devices()
    print(f"Devices found: {len(devices)}")
    for d in devices:
        print(f"  {d}")


def test_tensor_basics():
    """Test Tensor creation and metadata."""
    t = zse.Tensor(shape=(1024,), dtype=zse.float32)
    assert t.numel == 1024
    assert t.nbytes == 4096
    assert t.ndim == 1

    t2 = zse.Tensor(shape=(32, 64), dtype=zse.float16)
    assert t2.numel == 2048
    assert t2.nbytes == 4096
    assert t2.ndim == 2

    # INT4 packed
    t3 = zse.Tensor(shape=(1024,), dtype=zse.int4)
    assert t3.nbytes == 512  # 1024 * 4 bits / 8

    print("✓ Tensor basics passed")


if __name__ == "__main__":
    test_tensor_basics()
    test_device_detection()
    test_codegen_all_backends()
    print("\n\n✅ ALL TESTS PASSED")
