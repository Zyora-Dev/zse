"""
ZSE Compiler — Pure Python GPU Kernel Compiler

Write GPU kernels in Python, compile to CUDA / ROCm / Metal.
Zero dependencies. Zero vendor lock-in.

Usage:
    import zse_compiler as zse

    @zse.kernel
    def vector_add(a: zse.Tensor, b: zse.Tensor, out: zse.Tensor):
        idx = zse.global_id(0)
        out[idx] = a[idx] + b[idx]

    # Compile and run
    vector_add.compile(backend="cuda")
    vector_add.launch(grid=(n // 256,), block=(256,), args=(a, b, out))
"""

__version__ = "2.0.1"

# Types
from zse_compiler.types.dtypes import (
    DType, float16, float32, bfloat16,
    int32, int16, int8, uint32, uint16, uint8,
    int4, uint4,
)
from zse_compiler.types.tensor import Tensor, empty, zeros

# Primitives (for use inside @zse.kernel functions)
from zse_compiler.types.primitives import (
    thread_id, block_id, block_dim, grid_dim, global_id,
    shared_memory, dynamic_shared_memory, syncthreads,
    atomic_add, atomic_max, atomic_min, atomic_cas,
    exp, log, sqrt, rsqrt, max_val, min_val, fma,
    pow, cos, sin,
    half_to_float, float_to_half,
    # Warp primitives
    lane_id, warp_id,
    warp_shuffle_down, warp_shuffle_up, warp_shuffle_xor, warp_shuffle,
    warp_ballot, warp_all, warp_any,
    warp_reduce_sum, warp_reduce_max, warp_reduce_min,
    # Block reductions
    block_reduce_sum, block_reduce_max, block_reduce_min,
    # Vectorized memory
    load_float4, store_float4, load_half2, store_half2,
    # Tiling
    tile_load, tile_store,
    # INT4 nibble unpack
    unpack_int4,
    unpack_uint4,
    # Local register array + pointer reinterpret (Tier-3)
    local_array,
    reinterpret,
    # AMD CDNA MFMA matrix cores (Tier-4)
    mfma_f32_16x16x16_f16,
    mfma_f32_32x32x8_f16,
)

# Runtime
from zse_compiler.runtime.device import detect_backend, get_devices, DeviceInfo
from zse_compiler.runtime.memory import GPUMemory

# Kernel decorator
from zse_compiler.kernel import kernel, fuse, KernelFunction


__all__ = [
    # Core
    "kernel", "KernelFunction",
    # Types
    "DType", "Tensor", "empty", "zeros",
    "float16", "float32", "bfloat16",
    "int32", "int16", "int8", "uint32", "uint16", "uint8", "int4", "uint4",
    # Primitives
    "thread_id", "block_id", "block_dim", "grid_dim", "global_id",
    "shared_memory", "syncthreads",
    "atomic_add", "atomic_max", "atomic_min", "atomic_cas",
    "exp", "log", "sqrt", "rsqrt", "max_val", "min_val", "fma",
    "lane_id", "warp_id",
    "warp_shuffle_down", "warp_shuffle_up", "warp_shuffle_xor", "warp_shuffle",
    "warp_ballot", "warp_all", "warp_any",
    "warp_reduce_sum", "warp_reduce_max", "warp_reduce_min",
    "block_reduce_sum", "block_reduce_max", "block_reduce_min",
    "load_float4", "store_float4", "load_half2", "store_half2",
    "tile_load", "tile_store",
    "unpack_int4",
    "unpack_uint4",
    "local_array",
    "reinterpret",
    "mfma_f32_16x16x16_f16",
    "mfma_f32_32x32x8_f16",
    # Runtime
    "detect_backend", "get_devices", "DeviceInfo", "GPUMemory",
]
