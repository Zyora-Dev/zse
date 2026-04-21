"""ZSE Fast Quantizer — C-accelerated INT4 quantization via runtime compilation.

Compiles a tiny C library at first use via the system C compiler (cc/gcc/clang).
Same philosophy as ZSE Kernel Compiler: zero dependencies, just system tools.

Falls back to pure Python quantize.py if no C compiler is available.

Performance:
    Pure Python: ~1.16s per 1M elements (574s for Qwen 0.5B)
    C-accelerated: ~0.003s per 1M elements (0.9s for Qwen 0.5B) — 645x faster
"""

import ctypes
import os
import struct
import subprocess
import tempfile
from typing import Optional, Tuple

# --------------------------------------------------------------------------- #
# C source for the quantizer
# --------------------------------------------------------------------------- #

_C_SOURCE = r"""
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* BF16 / F16 -> float32 conversion                                   */
/* ------------------------------------------------------------------ */

void bf16_to_f32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t val = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &val, 4);
    }
}

void f16_to_f32(const uint16_t* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h & 0x8000) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            /* Subnormal or zero */
            if (mant == 0) {
                uint32_t val = sign;
                memcpy(&dst[i], &val, 4);
            } else {
                /* Subnormal: normalize */
                exp = 1;
                while (!(mant & 0x400)) { mant <<= 1; exp--; }
                mant &= 0x3FF;
                uint32_t val = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                memcpy(&dst[i], &val, 4);
            }
        } else if (exp == 31) {
            /* Inf / NaN */
            uint32_t val = sign | 0x7F800000 | (mant << 13);
            memcpy(&dst[i], &val, 4);
        } else {
            uint32_t val = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            memcpy(&dst[i], &val, 4);
        }
    }
}

/* ------------------------------------------------------------------ */
/* float32 -> float16 conversion (for scales/zeros output)            */
/* ------------------------------------------------------------------ */

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0) {
        return sign;  /* flush to zero */
    } else if (exp >= 31) {
        return sign | 0x7C00;  /* infinity */
    }
    return sign | (exp << 10) | (mant >> 13);
}

/* ------------------------------------------------------------------ */
/* INT4 asymmetric quantization (per-group)                           */
/* ------------------------------------------------------------------ */

void quantize_int4(
    const float* weights,
    int n,
    int group_size,
    uint8_t* packed,     /* n/2 bytes output: 2 uint4 per byte, low nibble first */
    uint16_t* scales,    /* num_groups float16 values (as raw uint16) */
    uint16_t* zeros      /* num_groups float16 values (as raw uint16) */
) {
    int num_groups = n / group_size;

    for (int g = 0; g < num_groups; g++) {
        const float* grp = weights + g * group_size;

        /* Find min/max */
        float gmin = grp[0], gmax = grp[0];
        for (int i = 1; i < group_size; i++) {
            if (grp[i] < gmin) gmin = grp[i];
            if (grp[i] > gmax) gmax = grp[i];
        }

        float range = gmax - gmin;
        float scale = (range > 1e-10f) ? range / 15.0f : 1.0f;
        float inv_scale = (range > 1e-10f) ? 15.0f / range : 0.0f;

        scales[g] = f32_to_f16(scale);
        zeros[g]  = f32_to_f16(gmin);

        /* Quantize and pack two values per byte */
        uint8_t* out = packed + g * (group_size / 2);
        for (int i = 0; i < group_size; i += 2) {
            float v0 = (grp[i]     - gmin) * inv_scale;
            float v1 = (grp[i + 1] - gmin) * inv_scale;
            int q0 = (int)(v0 + 0.5f);
            int q1 = (int)(v1 + 0.5f);
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            out[i / 2] = (uint8_t)((q1 << 4) | q0);
        }
    }
}

/* ------------------------------------------------------------------ */
/* INT8 symmetric quantization (per-group)                            */
/* ------------------------------------------------------------------ */

void quantize_int8(
    const float* weights,
    int n,
    int group_size,
    int8_t* packed,      /* n bytes output: 1 int8 per element */
    uint16_t* scales     /* num_groups float16 values (as raw uint16) */
) {
    int num_groups = n / group_size;

    for (int g = 0; g < num_groups; g++) {
        const float* grp = weights + g * group_size;

        /* Find max absolute value */
        float amax = 0.0f;
        for (int i = 0; i < group_size; i++) {
            float a = grp[i] < 0 ? -grp[i] : grp[i];
            if (a > amax) amax = a;
        }

        float scale = (amax > 1e-10f) ? amax / 127.0f : 1.0f;
        float inv_scale = (amax > 1e-10f) ? 127.0f / amax : 0.0f;

        scales[g] = f32_to_f16(scale);

        /* Quantize to [-128, 127] */
        int8_t* out = packed + g * group_size;
        for (int i = 0; i < group_size; i++) {
            float v = grp[i] * inv_scale;
            int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            out[i] = (int8_t)q;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Float32 -> Float16 bulk conversion (for fp16 tensors)              */
/* ------------------------------------------------------------------ */

void f32_to_f16_bulk(const float* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = f32_to_f16(src[i]);
    }
}

/* ------------------------------------------------------------------ */
/* INT4 row-major → tiled repack for WMMA tensor core coalesced access */
/* ------------------------------------------------------------------ */
/* Row-major: packed[n * half_K + k/2], 2 nibbles per byte             */
/* Tiled:     tiles[n_tile][k_tile][64][8]                              */
/*   n_tile = n / 64, k_tile = k / 16                                  */
/*   Each tile = 64 rows × 8 bytes (16 INT4 values) = 512 bytes        */
/* Scales/zeros stay row-major (unchanged).                             */

void repack_int4_to_tiled(
    const uint8_t* src,      /* [N, K/2] row-major packed INT4 */
    uint8_t* dst,            /* [num_n_tiles * num_k_tiles * 512] tiled output */
    int N, int K
) {
    int half_K = K / 2;
    int num_k_tiles = (K + 15) / 16;
    int num_n_tiles = (N + 63) / 64;

    /* Zero-fill output (handles padding for partial tiles) */
    memset(dst, 0, (long long)num_n_tiles * num_k_tiles * 512);

    for (int n = 0; n < N; n++) {
        int n_tile = n / 64;
        int n_local = n % 64;

        for (int kt = 0; kt < num_k_tiles; kt++) {
            int k_start_byte = kt * 8;  /* 16 INT4 values = 8 bytes */
            long long tile_offset = ((long long)n_tile * num_k_tiles + kt) * 512
                                  + n_local * 8;

            /* Copy 8 bytes (one row's K-tile) */
            int bytes_to_copy = 8;
            if (k_start_byte + bytes_to_copy > half_K) {
                bytes_to_copy = half_K - k_start_byte;
                if (bytes_to_copy <= 0) continue;
            }
            memcpy(dst + tile_offset,
                   src + (long long)n * half_K + k_start_byte,
                   bytes_to_copy);
        }
    }
}
"""

# --------------------------------------------------------------------------- #
# Runtime compilation + caching
# --------------------------------------------------------------------------- #

_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[str] = None


def _get_cache_dir() -> str:
    """Get or create cache directory for compiled libraries."""
    cache = os.path.join(tempfile.gettempdir(), "zse_cache")
    os.makedirs(cache, exist_ok=True)
    return cache


def _compile_lib() -> Optional[ctypes.CDLL]:
    """Compile the C quantizer library. Returns None if no compiler available."""
    cache_dir = _get_cache_dir()

    # Use a hash-based filename so recompilation only happens if source changes
    import hashlib
    src_hash = hashlib.md5(_C_SOURCE.encode()).hexdigest()[:12]

    if os.name == 'nt':
        so_name = f"zse_quant_{src_hash}.dll"
    elif os.uname().sysname == 'Darwin':
        so_name = f"zse_quant_{src_hash}.dylib"
    else:
        so_name = f"zse_quant_{src_hash}.so"

    so_path = os.path.join(cache_dir, so_name)

    # Check cache
    if os.path.exists(so_path):
        try:
            return ctypes.CDLL(so_path)
        except OSError:
            os.unlink(so_path)  # corrupted, recompile

    # Write source
    c_path = os.path.join(cache_dir, f"zse_quant_{src_hash}.c")
    with open(c_path, 'w') as f:
        f.write(_C_SOURCE)

    # Try compilers in order
    compilers = ['cc', 'gcc', 'clang']
    for compiler in compilers:
        try:
            cmd = [compiler, '-O3', '-shared', '-fPIC', '-o', so_path, c_path]
            if os.uname().sysname == 'Darwin':
                cmd.insert(1, '-arch')
                cmd.insert(2, os.uname().machine)  # arm64 or x86_64
            result = subprocess.run(
                cmd, capture_output=True, timeout=30,
            )
            if result.returncode == 0 and os.path.exists(so_path):
                return ctypes.CDLL(so_path)
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

    return None


def _ensure_lib() -> Optional[ctypes.CDLL]:
    """Get or compile the C library (cached)."""
    global _lib
    if _lib is not None:
        return _lib
    _lib = _compile_lib()
    return _lib


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def is_available() -> bool:
    """Check if the C-accelerated quantizer is available."""
    return _ensure_lib() is not None


def quantize_tensor_fast(
    raw_data: bytes,
    dtype: str,
    num_elements: int,
    group_size: int = 128,
) -> Tuple[bytes, bytes, bytes]:
    """Quantize raw weight bytes directly to INT4.

    Args:
        raw_data: Raw tensor bytes (BF16, F16, or F32 from safetensors)
        dtype: Source dtype string ("BF16", "F16", "F32")
        num_elements: Number of elements in the tensor
        group_size: Quantization group size

    Returns:
        (packed_data, scales_data, zeros_data) as bytes
    """
    lib = _ensure_lib()
    if lib is None:
        raise RuntimeError("C quantizer not available — no C compiler found")

    # Pad to group_size multiple
    padded = num_elements
    if padded % group_size != 0:
        pad_count = group_size - (padded % group_size)
        padded += pad_count
        # Extend raw data with zeros
        elem_size = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
        raw_data = raw_data + b'\x00' * (pad_count * elem_size)

    num_groups = padded // group_size

    # Convert to float32
    f32_buf = (ctypes.c_float * padded)()

    if dtype == "BF16":
        src = (ctypes.c_uint16 * padded).from_buffer_copy(raw_data[:padded * 2])
        lib.bf16_to_f32(src, f32_buf, padded)
    elif dtype == "F16":
        src = (ctypes.c_uint16 * padded).from_buffer_copy(raw_data[:padded * 2])
        lib.f16_to_f32(src, f32_buf, padded)
    elif dtype == "F32":
        ctypes.memmove(f32_buf, raw_data[:padded * 4], padded * 4)
    else:
        raise ValueError(f"Unsupported dtype for fast quantizer: {dtype}")

    # Quantize
    packed_buf = (ctypes.c_uint8 * (padded // 2))()
    scales_buf = (ctypes.c_uint16 * num_groups)()
    zeros_buf = (ctypes.c_uint16 * num_groups)()

    lib.quantize_int4(f32_buf, padded, group_size, packed_buf, scales_buf, zeros_buf)

    return (
        bytes(packed_buf),
        bytes(scales_buf),
        bytes(zeros_buf),
    )


def convert_to_fp16_fast(
    raw_data: bytes,
    dtype: str,
    num_elements: int,
) -> bytes:
    """Convert raw weight bytes to float16 (for non-quantized tensors).

    Args:
        raw_data: Raw tensor bytes (BF16, F16, or F32)
        dtype: Source dtype string
        num_elements: Number of elements

    Returns:
        Float16 bytes
    """
    lib = _ensure_lib()
    if lib is None:
        raise RuntimeError("C quantizer not available")

    if dtype == "F16":
        # Already fp16
        return raw_data[:num_elements * 2]

    # Convert to float32 first
    f32_buf = (ctypes.c_float * num_elements)()

    if dtype == "BF16":
        src = (ctypes.c_uint16 * num_elements).from_buffer_copy(raw_data[:num_elements * 2])
        lib.bf16_to_f32(src, f32_buf, num_elements)
    elif dtype == "F32":
        ctypes.memmove(f32_buf, raw_data[:num_elements * 4], num_elements * 4)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Convert to fp16
    f16_buf = (ctypes.c_uint16 * num_elements)()
    lib.f32_to_f16_bulk(f32_buf, f16_buf, num_elements)

    return bytes(f16_buf)


def quantize_tensor_int8_fast(
    raw_data: bytes,
    dtype: str,
    num_elements: int,
    group_size: int = 128,
) -> Tuple[bytes, bytes]:
    """Quantize raw weight bytes to INT8 symmetric.

    Args:
        raw_data: Raw tensor bytes (BF16, F16, or F32)
        dtype: Source dtype string
        num_elements: Number of elements
        group_size: Quantization group size

    Returns:
        (packed_data, scales_data) as bytes
    """
    lib = _ensure_lib()
    if lib is None:
        raise RuntimeError("C quantizer not available")

    # Pad to group_size multiple
    padded = num_elements
    if padded % group_size != 0:
        pad_count = group_size - (padded % group_size)
        padded += pad_count
        elem_size = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
        raw_data = raw_data + b'\x00' * (pad_count * elem_size)

    num_groups = padded // group_size

    # Convert to float32
    f32_buf = (ctypes.c_float * padded)()

    if dtype == "BF16":
        src = (ctypes.c_uint16 * padded).from_buffer_copy(raw_data[:padded * 2])
        lib.bf16_to_f32(src, f32_buf, padded)
    elif dtype == "F16":
        src = (ctypes.c_uint16 * padded).from_buffer_copy(raw_data[:padded * 2])
        lib.f16_to_f32(src, f32_buf, padded)
    elif dtype == "F32":
        ctypes.memmove(f32_buf, raw_data[:padded * 4], padded * 4)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Quantize
    packed_buf = (ctypes.c_int8 * padded)()
    scales_buf = (ctypes.c_uint16 * num_groups)()

    lib.quantize_int8(f32_buf, padded, group_size, packed_buf, scales_buf)

    return bytes(packed_buf), bytes(scales_buf)


def repack_int4_tiled(
    packed_data: bytes,
    N: int,
    K: int,
) -> bytes:
    """Repack INT4 weights from row-major [N, K/2] to tiled format for WMMA.

    Tiled layout: [num_n_tiles, num_k_tiles, 64, 8]
    Each tile = 64 rows × 8 bytes (16 INT4 values) = 512 contiguous bytes.
    128 GPU threads load one tile in a single coalesced read.

    Args:
        packed_data: Row-major packed INT4 bytes [N * K/2]
        N: Number of output rows
        K: Number of input columns (full, not packed)

    Returns:
        Tiled packed bytes
    """
    lib = _ensure_lib()
    if lib is None:
        raise RuntimeError("C quantizer not available for tiled repack")

    half_K = K // 2
    num_k_tiles = (K + 15) // 16
    num_n_tiles = (N + 63) // 64
    tiled_size = num_n_tiles * num_k_tiles * 512

    src_buf = (ctypes.c_uint8 * len(packed_data)).from_buffer_copy(packed_data)
    dst_buf = (ctypes.c_uint8 * tiled_size)()

    lib.repack_int4_to_tiled(src_buf, dst_buf, N, K)

    return bytes(dst_buf)
