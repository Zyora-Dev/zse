"""
ZSE Native Format Specification

Defines the binary format for .zse model files.

Format Overview:
- Magic + Version: 8 bytes
- Header: Length-prefixed JSON with model config
- Tokenizer: Length-prefixed msgpack with tokenizer data
- Tensor Index: Offset table for all tensors
- Tensor Data: Raw tensor data (memory-mappable)

Version History:
- v1: Initial format with INT4/INT8 quantization support
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
import json


# Magic bytes: "ZSE" + null + version (major.minor.patch.reserved)
ZSE_MAGIC = b"ZSE\x00"
ZSE_VERSION = (1, 0, 0, 0)  # Major, Minor, Patch, Reserved

# Maximum sizes
MAX_HEADER_SIZE = 10 * 1024 * 1024  # 10 MB max header
MAX_TOKENIZER_SIZE = 100 * 1024 * 1024  # 100 MB max tokenizer


class TensorDType(IntEnum):
    """Supported tensor data types."""
    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2
    INT8 = 3
    INT4 = 4  # Packed (2 values per byte)
    UINT8 = 5
    UINT4 = 6  # Packed
    FP8_E4M3 = 7
    FP8_E5M2 = 8


class QuantizationType(IntEnum):
    """Quantization methods."""
    NONE = 0
    ABSMAX = 1  # Simple absmax quantization
    ZEROP = 2   # Zero-point quantization
    GPTQ = 3    # GPTQ quantization
    AWQ = 4     # AWQ quantization
    HQQ = 5     # Half-Quadratic Quantization


@dataclass
class TensorInfo:
    """Information about a stored tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: TensorDType
    offset: int  # Byte offset in file
    size: int    # Size in bytes
    
    # Quantization info (if applicable)
    quant_type: QuantizationType = QuantizationType.NONE
    quant_bits: int = 0
    group_size: int = 0
    scale_offset: int = 0  # Offset to scales tensor
    zeros_offset: int = 0  # Offset to zeros tensor
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "offset": self.offset,
            "size": self.size,
            "quant_type": self.quant_type.value,
            "quant_bits": self.quant_bits,
            "group_size": self.group_size,
            "scale_offset": self.scale_offset,
            "zeros_offset": self.zeros_offset,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorInfo":
        """Deserialize from dictionary."""
        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=TensorDType(d["dtype"]),
            offset=d["offset"],
            size=d["size"],
            quant_type=QuantizationType(d.get("quant_type", 0)),
            quant_bits=d.get("quant_bits", 0),
            group_size=d.get("group_size", 0),
            scale_offset=d.get("scale_offset", 0),
            zeros_offset=d.get("zeros_offset", 0),
        )


@dataclass
class LayerGroup:
    """Group of tensors belonging to a layer."""
    layer_idx: int
    tensor_names: List[str]
    offset: int  # Start offset for this layer's tensors
    size: int    # Total size of this layer's tensors
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": self.layer_idx,
            "tensor_names": self.tensor_names,
            "offset": self.offset,
            "size": self.size,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerGroup":
        return cls(
            layer_idx=d["layer_idx"],
            tensor_names=d["tensor_names"],
            offset=d["offset"],
            size=d["size"],
        )


@dataclass
class ZSEHeader:
    """
    Header containing all metadata for a .zse file.
    """
    # Version info
    version: Tuple[int, int, int, int] = ZSE_VERSION
    
    # Model architecture
    architecture: str = ""  # e.g., "LlamaForCausalLM"
    model_type: str = ""    # e.g., "llama"
    
    # Model config
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    vocab_size: int = 0
    max_position_embeddings: int = 0
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    
    # Quantization info
    quantization: str = "none"  # "none", "int8", "int4"
    quant_method: str = ""      # "gptq", "awq", "hqq", ""
    
    # ZSE-specific
    zse_config: Dict[str, Any] = field(default_factory=dict)
    
    # Tensor info
    tensors: List[TensorInfo] = field(default_factory=list)
    
    # Layer groups (for streaming)
    layer_groups: List[LayerGroup] = field(default_factory=list)
    
    # Offsets
    tokenizer_offset: int = 0
    tokenizer_size: int = 0
    tensor_data_offset: int = 0
    
    # Original source
    source_model: str = ""
    source_revision: str = ""
    
    def to_json(self) -> str:
        """Serialize header to JSON."""
        data = {
            "version": list(self.version),
            "architecture": self.architecture,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "quantization": self.quantization,
            "quant_method": self.quant_method,
            "zse_config": self.zse_config,
            "tensors": [t.to_dict() for t in self.tensors],
            "layer_groups": [g.to_dict() for g in self.layer_groups],
            "tokenizer_offset": self.tokenizer_offset,
            "tokenizer_size": self.tokenizer_size,
            "tensor_data_offset": self.tensor_data_offset,
            "source_model": self.source_model,
            "source_revision": self.source_revision,
        }
        return json.dumps(data, separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str) -> "ZSEHeader":
        """Deserialize header from JSON."""
        data = json.loads(json_str)
        header = cls(
            version=tuple(data.get("version", ZSE_VERSION)),
            architecture=data.get("architecture", ""),
            model_type=data.get("model_type", ""),
            hidden_size=data.get("hidden_size", 0),
            intermediate_size=data.get("intermediate_size", 0),
            num_hidden_layers=data.get("num_hidden_layers", 0),
            num_attention_heads=data.get("num_attention_heads", 0),
            num_key_value_heads=data.get("num_key_value_heads", 0),
            vocab_size=data.get("vocab_size", 0),
            max_position_embeddings=data.get("max_position_embeddings", 0),
            rope_theta=data.get("rope_theta", 10000.0),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            quantization=data.get("quantization", "none"),
            quant_method=data.get("quant_method", ""),
            zse_config=data.get("zse_config", {}),
            tensors=[TensorInfo.from_dict(t) for t in data.get("tensors", [])],
            layer_groups=[LayerGroup.from_dict(g) for g in data.get("layer_groups", [])],
            tokenizer_offset=data.get("tokenizer_offset", 0),
            tokenizer_size=data.get("tokenizer_size", 0),
            tensor_data_offset=data.get("tensor_data_offset", 0),
            source_model=data.get("source_model", ""),
            source_revision=data.get("source_revision", ""),
        )
        return header
    
    def get_tensor(self, name: str) -> Optional[TensorInfo]:
        """Get tensor info by name."""
        for t in self.tensors:
            if t.name == name:
                return t
        return None
    
    def get_layer_tensors(self, layer_idx: int) -> List[TensorInfo]:
        """Get all tensors for a specific layer."""
        prefix = f"model.layers.{layer_idx}."
        return [t for t in self.tensors if t.name.startswith(prefix)]
    
    def get_layer_group(self, layer_idx: int) -> Optional[LayerGroup]:
        """Get layer group by index."""
        for g in self.layer_groups:
            if g.layer_idx == layer_idx:
                return g
        return None


def encode_header(header: ZSEHeader) -> bytes:
    """
    Encode header to bytes.
    
    Format:
    - Magic (4 bytes): "ZSE\x00"
    - Version (4 bytes): major, minor, patch, reserved
    - Header length (4 bytes, uint32)
    - Header JSON (variable length)
    """
    json_bytes = header.to_json().encode('utf-8')
    
    if len(json_bytes) > MAX_HEADER_SIZE:
        raise ValueError(f"Header too large: {len(json_bytes)} > {MAX_HEADER_SIZE}")
    
    result = bytearray()
    result.extend(ZSE_MAGIC)
    result.extend(struct.pack('<BBBB', *header.version))
    result.extend(struct.pack('<I', len(json_bytes)))
    result.extend(json_bytes)
    
    return bytes(result)


def decode_header(data: bytes) -> Tuple[ZSEHeader, int]:
    """
    Decode header from bytes.
    
    Returns:
        (header, bytes_consumed)
    """
    if len(data) < 12:
        raise ValueError("Invalid .zse file: too short")
    
    # Check magic
    if data[:4] != ZSE_MAGIC:
        raise ValueError("Invalid .zse file: bad magic bytes")
    
    # Read version
    version = struct.unpack('<BBBB', data[4:8])
    
    # Read header length
    header_len = struct.unpack('<I', data[8:12])[0]
    
    if header_len > MAX_HEADER_SIZE:
        raise ValueError(f"Header too large: {header_len}")
    
    if len(data) < 12 + header_len:
        raise ValueError("Invalid .zse file: truncated header")
    
    # Read header JSON
    json_str = data[12:12 + header_len].decode('utf-8')
    header = ZSEHeader.from_json(json_str)
    header.version = version
    
    return header, 12 + header_len


# Utility functions for dtype conversion
DTYPE_SIZES = {
    TensorDType.FLOAT32: 4,
    TensorDType.FLOAT16: 2,
    TensorDType.BFLOAT16: 2,
    TensorDType.INT8: 1,
    TensorDType.INT4: 0.5,
    TensorDType.UINT8: 1,
    TensorDType.UINT4: 0.5,
    TensorDType.FP8_E4M3: 1,
    TensorDType.FP8_E5M2: 1,
}


def calculate_tensor_size(shape: Tuple[int, ...], dtype: TensorDType) -> int:
    """Calculate byte size for a tensor."""
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    byte_size = num_elements * DTYPE_SIZES[dtype]
    return int(byte_size)


def torch_dtype_to_zse(torch_dtype) -> TensorDType:
    """Convert PyTorch dtype to ZSE dtype."""
    import torch
    
    mapping = {
        torch.float32: TensorDType.FLOAT32,
        torch.float16: TensorDType.FLOAT16,
        torch.bfloat16: TensorDType.BFLOAT16,
        torch.int8: TensorDType.INT8,
        torch.uint8: TensorDType.UINT8,
    }
    
    return mapping.get(torch_dtype, TensorDType.FLOAT16)


def zse_dtype_to_torch(zse_dtype: TensorDType):
    """Convert ZSE dtype to PyTorch dtype."""
    import torch
    
    mapping = {
        TensorDType.FLOAT32: torch.float32,
        TensorDType.FLOAT16: torch.float16,
        TensorDType.BFLOAT16: torch.bfloat16,
        TensorDType.INT8: torch.int8,
        TensorDType.UINT8: torch.uint8,
        TensorDType.INT4: torch.int8,  # Packed as int8
        TensorDType.UINT4: torch.uint8,  # Packed as uint8
        TensorDType.FP8_E4M3: torch.float16,  # Fallback
        TensorDType.FP8_E5M2: torch.float16,  # Fallback
    }
    
    return mapping.get(zse_dtype, torch.float16)
