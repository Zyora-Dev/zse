"""ZSE INT4 Quantization — Pure Python, zero numpy dependency.

Asymmetric INT4 quantization with per-group scales and zero-points.
Uses only Python stdlib (struct module for packing).

Each group of `group_size` weights is quantized independently:
    scale = (max_val - min_val) / 15.0
    zero  = min_val
    quant[i] = clamp(round((weight[i] - zero) / scale), 0, 15)

Packed: 2 uint4 values per byte, low nibble first.
"""

import struct
from typing import List, Tuple


def quantize_tensor(
    weights: List[float],
    group_size: int = 128,
) -> Tuple[bytes, bytes, bytes]:
    """Quantize a flat weight list to INT4.

    Args:
        weights: Flat list of float32 weight values
        group_size: Number of weights per quantization group

    Returns:
        (packed_data, scales_data, zeros_data)
        packed_data: uint4 packed (2 per byte)
        scales_data: float16 scale per group
        zeros_data: float16 zero-point per group
    """
    n = len(weights)
    # Pad to group_size multiple
    pad_count = 0
    if n % group_size != 0:
        pad_count = group_size - (n % group_size)
        weights = list(weights) + [0.0] * pad_count

    num_groups = len(weights) // group_size

    packed_parts = []
    scales_parts = []
    zeros_parts = []

    for g in range(num_groups):
        start = g * group_size
        group = weights[start:start + group_size]

        # Find min/max
        g_min = group[0]
        g_max = group[0]
        for v in group:
            if v < g_min:
                g_min = v
            if v > g_max:
                g_max = v

        # Compute scale and zero
        val_range = g_max - g_min
        if val_range < 1e-10:
            scale = 1.0
        else:
            scale = val_range / 15.0
        zero = g_min

        # Pack scale and zero as float16
        scales_parts.append(struct.pack('<e', scale))
        zeros_parts.append(struct.pack('<e', zero))

        # Quantize values to [0, 15]
        inv_scale = 1.0 / scale if scale > 1e-10 else 0.0
        quantized = []
        for v in group:
            q = round((v - zero) * inv_scale)
            q = max(0, min(15, int(q)))
            quantized.append(q)

        # Pack 2 uint4 per byte (low nibble first)
        for i in range(0, group_size, 2):
            low = quantized[i]
            high = quantized[i + 1] if (i + 1) < group_size else 0
            packed_parts.append(struct.pack('B', (high << 4) | low))

    return b''.join(packed_parts), b''.join(scales_parts), b''.join(zeros_parts)


def dequantize_tensor(
    packed_data: bytes,
    scales_data: bytes,
    zeros_data: bytes,
    shape: Tuple[int, ...],
    group_size: int = 128,
) -> List[float]:
    """Dequantize INT4 packed data back to float32.

    Used for verification. The GPU kernel does this on-device.
    """
    total_elements = 1
    for d in shape:
        total_elements *= d

    # Pad to group_size
    padded = total_elements
    if padded % group_size != 0:
        padded += group_size - (padded % group_size)

    num_groups = padded // group_size
    result = []

    for g in range(num_groups):
        # Read scale and zero
        scale = struct.unpack_from('<e', scales_data, g * 2)[0]
        zero = struct.unpack_from('<e', zeros_data, g * 2)[0]

        # Unpack uint4 values
        byte_offset = g * (group_size // 2)
        for i in range(0, group_size, 2):
            byte_val = packed_data[byte_offset + i // 2]
            low = byte_val & 0x0F
            high = (byte_val >> 4) & 0x0F

            result.append(low * scale + zero)
            if i + 1 < group_size:
                result.append(high * scale + zero)

    return result[:total_elements]


def quantize_fp16(weights: List[float]) -> bytes:
    """Convert float32 weights to float16 bytes. For unquantized tensors."""
    parts = []
    for v in weights:
        parts.append(struct.pack('<e', v))
    return b''.join(parts)


def quantize_tensor_int8(
    weights: List[float],
    group_size: int = 128,
) -> Tuple[bytes, bytes]:
    """Quantize a flat weight list to INT8 symmetric.

    Symmetric: scale = max(abs(group)) / 127, zero = 0
    Each value: q = clamp(round(v / scale), -128, 127)

    Args:
        weights: Flat list of float32 weight values
        group_size: Number of weights per quantization group

    Returns:
        (packed_data, scales_data)
        packed_data: int8 values (1 byte each)
        scales_data: float16 scale per group
    """
    n = len(weights)
    if n % group_size != 0:
        pad_count = group_size - (n % group_size)
        weights = list(weights) + [0.0] * pad_count

    num_groups = len(weights) // group_size

    packed_parts = []
    scales_parts = []

    for g in range(num_groups):
        start = g * group_size
        group = weights[start:start + group_size]

        # Find max absolute value
        amax = 0.0
        for v in group:
            a = abs(v)
            if a > amax:
                amax = a

        scale = amax / 127.0 if amax > 1e-10 else 1.0
        inv_scale = 127.0 / amax if amax > 1e-10 else 0.0

        scales_parts.append(struct.pack('<e', scale))

        # Quantize to [-128, 127]
        for v in group:
            q = round(v * inv_scale)
            q = max(-128, min(127, int(q)))
            packed_parts.append(struct.pack('b', q))

    return b''.join(packed_parts), b''.join(scales_parts)


def dequantize_tensor_int8(
    packed_data: bytes,
    scales_data: bytes,
    shape: Tuple[int, ...],
    group_size: int = 128,
) -> List[float]:
    """Dequantize INT8 symmetric data back to float32."""
    total_elements = 1
    for d in shape:
        total_elements *= d

    padded = total_elements
    if padded % group_size != 0:
        padded += group_size - (padded % group_size)

    num_groups = padded // group_size
    result = []

    for g in range(num_groups):
        scale = struct.unpack_from('<e', scales_data, g * 2)[0]
        byte_offset = g * group_size
        for i in range(group_size):
            q = struct.unpack_from('b', packed_data, byte_offset + i)[0]
            result.append(q * scale)

    return result[:total_elements]


def compute_packed_size(num_elements: int, group_size: int = 128) -> Tuple[int, int, int]:
    """Compute sizes for packed data, scales, and zeros.

    Returns: (packed_bytes, scales_bytes, zeros_bytes)
    """
    # Pad to group_size
    padded = num_elements
    if padded % group_size != 0:
        padded += group_size - (padded % group_size)

    num_groups = padded // group_size
    packed_bytes = padded // 2  # 2 values per byte
    scales_bytes = num_groups * 2  # float16 per group
    zeros_bytes = num_groups * 2   # float16 per group

    return packed_bytes, scales_bytes, zeros_bytes
