"""ZSE Format Specification — Constants and types for the .zse binary format."""

import enum

# File identification
MAGIC = b"ZSE\x01"
VERSION = 1

# Alignment
PAGE_SIZE = 4096       # Weight data section starts at page boundary (mmap-ready)
TENSOR_ALIGN = 256     # Each tensor within weight data aligned to 256 bytes (GPU cache-line)

# Header size
HEADER_SIZE = 64
SECTION_ENTRY_SIZE = 32

# Quantization defaults
DEFAULT_GROUP_SIZE = 128
DEFAULT_QUANT_BITS = 4


class SectionType(enum.IntEnum):
    CONFIG = 0
    TOKENIZER = 1
    WEIGHT_INDEX = 2
    WEIGHT_DATA = 3
    KERNELS = 4


class Flags(enum.IntFlag):
    HAS_TOKENIZER = 1 << 0
    HAS_KERNELS = 1 << 1
    MMAP_READY = 1 << 2


class QuantMethod(enum.IntEnum):
    NONE = 0       # Unquantized (fp16/fp32)
    INT4_ASYM = 1  # Asymmetric INT4 with group scales + zeros
    INT8_SYM = 2   # Symmetric INT8 (future)


# Tensors that should NOT be quantized (quality-critical, small size)
NO_QUANTIZE_PATTERNS = [
    "embed_tokens",
    "lm_head",
    "norm",        # Catches input_layernorm, post_attention_layernorm, model.norm
    "layernorm",
    ".bias",       # All biases stay fp16 (small, quality-critical)
]


def should_quantize(tensor_name: str) -> bool:
    """Check if a tensor should be quantized or kept in fp16."""
    name_lower = tensor_name.lower()
    for pattern in NO_QUANTIZE_PATTERNS:
        if pattern in name_lower:
            return False
    return True


def align_offset(offset: int, alignment: int) -> int:
    """Round up offset to next alignment boundary."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def pad_to_alignment(offset: int, alignment: int) -> bytes:
    """Return zero bytes needed to pad from offset to next alignment boundary."""
    aligned = align_offset(offset, alignment)
    return b'\x00' * (aligned - offset)
