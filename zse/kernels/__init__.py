"""
ZSE Custom Kernels

High-performance kernels for INT4 inference.
Uses Triton for JIT compilation (no Python.h needed).

Two kernel versions:
- v1 (triton_int4.py): Basic Triton kernel, fixed tile sizes
- v2 (triton_int4_v2.py): Optimized with autotuning, GEMV for decode, coalesced layout
"""

# =============================================================================
# Triton v1 (basic kernel)
# =============================================================================
_TRITON_INT4_AVAILABLE = False
int4_matmul_triton = None
TritonInt4Linear = None

try:
    from .triton_int4 import (
        int4_matmul_triton,
        TritonInt4Linear,
        is_triton_available,
    )
    _TRITON_INT4_AVAILABLE = is_triton_available()
except ImportError:
    pass

def is_triton_available():
    return _TRITON_INT4_AVAILABLE


# =============================================================================
# Triton v2 (optimized kernel with autotuning)
# =============================================================================
_TRITON_V2_AVAILABLE = False
int4_matmul_triton_v2 = None
TritonInt4LinearV2 = None
repack_weights_for_v2 = None
repack_scales_for_v2 = None
_triton_v2_error = None

try:
    from .triton_int4_v2 import (
        int4_matmul_triton_v2,
        TritonInt4LinearV2,
        repack_weights_for_v2,
        repack_scales_for_v2,
        is_triton_v2_available,
        get_triton_v2_error,
    )
    _TRITON_V2_AVAILABLE = is_triton_v2_available()
    _triton_v2_error = get_triton_v2_error()
except ImportError as e:
    _triton_v2_error = f"Import error: {e}"
except Exception as e:
    _triton_v2_error = f"Error: {e}"

def is_triton_v2_available():
    """Check if optimized Triton v2 kernels are available."""
    return _TRITON_V2_AVAILABLE

def get_triton_v2_error():
    """Get error message if Triton v2 isn't available."""
    return _triton_v2_error


# =============================================================================
# Legacy CUDA (disabled)
# =============================================================================
int4_matmul = None
Int4Linear = None
int4_matmul_pytorch = None

def is_kernel_available():
    return False


__all__ = [
    # V1 kernel
    "int4_matmul_triton",
    "TritonInt4Linear",
    "is_triton_available",
    # V2 kernel (optimized)
    "int4_matmul_triton_v2",
    "TritonInt4LinearV2",
    "repack_weights_for_v2",
    "repack_scales_for_v2",
    "is_triton_v2_available",
    "get_triton_v2_error",
]
