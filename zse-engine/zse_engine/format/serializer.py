"""ZSE Binary Serializer — Zero-dependency replacement for msgpack.

Encodes Python dicts/lists/strings/ints/floats/bytes/None into a compact
binary format. Used for CONFIG, TOKENIZER, and WEIGHT_INDEX sections.

Wire format (type byte prefix):
    0x00 = None
    0x01 = bool False
    0x02 = bool True
    0x10 = int (int64_le)
    0x11 = float (float64_le)
    0x20 = bytes (uint32_le length + raw)
    0x21 = string (uint32_le length + utf-8)
    0x30 = list (uint32_le count + elements)
    0x40 = dict (uint32_le count + key-value pairs)
"""

import struct
from typing import Any

# Type tags
TAG_NONE = 0x00
TAG_FALSE = 0x01
TAG_TRUE = 0x02
TAG_INT = 0x10
TAG_FLOAT = 0x11
TAG_BYTES = 0x20
TAG_STR = 0x21
TAG_LIST = 0x30
TAG_DICT = 0x40


def encode(obj: Any) -> bytes:
    """Encode a Python object to ZSE binary format."""
    parts = []
    _encode_into(obj, parts)
    return b''.join(parts)


def _encode_into(obj: Any, parts: list):
    if obj is None:
        parts.append(struct.pack('B', TAG_NONE))
    elif obj is False:
        parts.append(struct.pack('B', TAG_FALSE))
    elif obj is True:
        parts.append(struct.pack('B', TAG_TRUE))
    elif isinstance(obj, int) and not isinstance(obj, bool):
        parts.append(struct.pack('<Bq', TAG_INT, obj))
    elif isinstance(obj, float):
        parts.append(struct.pack('<Bd', TAG_FLOAT, obj))
    elif isinstance(obj, bytes):
        parts.append(struct.pack('<BI', TAG_BYTES, len(obj)))
        parts.append(obj)
    elif isinstance(obj, str):
        encoded = obj.encode('utf-8')
        parts.append(struct.pack('<BI', TAG_STR, len(encoded)))
        parts.append(encoded)
    elif isinstance(obj, (list, tuple)):
        parts.append(struct.pack('<BI', TAG_LIST, len(obj)))
        for item in obj:
            _encode_into(item, parts)
    elif isinstance(obj, dict):
        parts.append(struct.pack('<BI', TAG_DICT, len(obj)))
        for key, value in obj.items():
            _encode_into(key, parts)
            _encode_into(value, parts)
    else:
        raise TypeError(f"ZSE serializer: unsupported type {type(obj).__name__}")


def decode(data: bytes) -> Any:
    """Decode ZSE binary format to Python object."""
    obj, _ = _decode_at(data, 0)
    return obj


def _decode_at(data: bytes, pos: int) -> tuple:
    """Decode object at position, return (object, new_position)."""
    tag = data[pos]
    pos += 1

    if tag == TAG_NONE:
        return None, pos
    elif tag == TAG_FALSE:
        return False, pos
    elif tag == TAG_TRUE:
        return True, pos
    elif tag == TAG_INT:
        val = struct.unpack_from('<q', data, pos)[0]
        return val, pos + 8
    elif tag == TAG_FLOAT:
        val = struct.unpack_from('<d', data, pos)[0]
        return val, pos + 8
    elif tag == TAG_BYTES:
        length = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        return data[pos:pos + length], pos + length
    elif tag == TAG_STR:
        length = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        return data[pos:pos + length].decode('utf-8'), pos + length
    elif tag == TAG_LIST:
        count = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        result = []
        for _ in range(count):
            item, pos = _decode_at(data, pos)
            result.append(item)
        return result, pos
    elif tag == TAG_DICT:
        count = struct.unpack_from('<I', data, pos)[0]
        pos += 4
        result = {}
        for _ in range(count):
            key, pos = _decode_at(data, pos)
            value, pos = _decode_at(data, pos)
            result[key] = value
        return result, pos
    else:
        raise ValueError(f"ZSE serializer: unknown tag 0x{tag:02x} at position {pos - 1}")
