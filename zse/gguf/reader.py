"""
GGUF File Reader

Parses GGUF (GPT-Generated Unified Format) files.
GGUF is the successor to GGML, used by llama.cpp.

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, BinaryIO, Union
from enum import IntEnum


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLQuantType(IntEnum):
    """GGML quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23


# Quantization block sizes
GGML_BLOCK_SIZES = {
    GGMLQuantType.F32: 1,
    GGMLQuantType.F16: 1,
    GGMLQuantType.Q4_0: 32,
    GGMLQuantType.Q4_1: 32,
    GGMLQuantType.Q5_0: 32,
    GGMLQuantType.Q5_1: 32,
    GGMLQuantType.Q8_0: 32,
    GGMLQuantType.Q8_1: 32,
    GGMLQuantType.Q2_K: 256,
    GGMLQuantType.Q3_K: 256,
    GGMLQuantType.Q4_K: 256,
    GGMLQuantType.Q5_K: 256,
    GGMLQuantType.Q6_K: 256,
    GGMLQuantType.Q8_K: 256,
}

# Bytes per block for each quantization type
GGML_TYPE_SIZES = {
    GGMLQuantType.F32: 4,
    GGMLQuantType.F16: 2,
    GGMLQuantType.Q4_0: 18,  # 32 values: 2 bytes scale + 16 bytes data
    GGMLQuantType.Q4_1: 20,  # 32 values: 2 bytes scale + 2 bytes min + 16 bytes data
    GGMLQuantType.Q5_0: 22,
    GGMLQuantType.Q5_1: 24,
    GGMLQuantType.Q8_0: 34,  # 32 values: 2 bytes scale + 32 bytes data
    GGMLQuantType.Q8_1: 36,
    GGMLQuantType.Q2_K: 84,
    GGMLQuantType.Q3_K: 110,
    GGMLQuantType.Q4_K: 144,
    GGMLQuantType.Q5_K: 176,
    GGMLQuantType.Q6_K: 210,
    GGMLQuantType.Q8_K: 292,
}


@dataclass
class GGUFTensorInfo:
    """Information about a tensor in the GGUF file."""
    name: str
    n_dims: int
    dims: List[int]
    dtype: GGMLQuantType
    offset: int  # Offset from start of tensor data section
    
    @property
    def n_elements(self) -> int:
        """Total number of elements."""
        result = 1
        for d in self.dims:
            result *= d
        return result
    
    @property
    def n_bytes(self) -> int:
        """Total bytes for this tensor."""
        block_size = GGML_BLOCK_SIZES.get(self.dtype, 1)
        type_size = GGML_TYPE_SIZES.get(self.dtype, 4)
        n_blocks = (self.n_elements + block_size - 1) // block_size
        return n_blocks * type_size


@dataclass
class GGUFHeader:
    """GGUF file header."""
    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int


@dataclass 
class GGUFFile:
    """Parsed GGUF file."""
    path: Path
    header: GGUFHeader
    metadata: Dict[str, Any]
    tensors: Dict[str, GGUFTensorInfo]
    tensor_data_offset: int  # Offset where tensor data starts
    
    # Common metadata accessors
    @property
    def architecture(self) -> str:
        return self.metadata.get("general.architecture", "unknown")
    
    @property
    def name(self) -> str:
        return self.metadata.get("general.name", self.path.stem)
    
    @property
    def context_length(self) -> int:
        arch = self.architecture
        return self.metadata.get(f"{arch}.context_length", 2048)
    
    @property
    def embedding_length(self) -> int:
        arch = self.architecture
        return self.metadata.get(f"{arch}.embedding_length", 4096)
    
    @property
    def block_count(self) -> int:
        """Number of transformer blocks/layers."""
        arch = self.architecture
        return self.metadata.get(f"{arch}.block_count", 32)
    
    @property
    def head_count(self) -> int:
        arch = self.architecture
        return self.metadata.get(f"{arch}.attention.head_count", 32)
    
    @property
    def head_count_kv(self) -> int:
        """KV heads for GQA models."""
        arch = self.architecture
        return self.metadata.get(f"{arch}.attention.head_count_kv", self.head_count)
    
    @property
    def vocab_size(self) -> int:
        return len(self.metadata.get("tokenizer.ggml.tokens", []))
    
    @property
    def quantization_type(self) -> str:
        """Determine primary quantization type from tensors."""
        # Check a weight tensor
        for name, tensor in self.tensors.items():
            if "weight" in name and tensor.dtype != GGMLQuantType.F32:
                return tensor.dtype.name
        return "F16"


class GGUFReader:
    """
    Reader for GGUF files.
    
    Usage:
        reader = GGUFReader("model.gguf")
        gguf = reader.read()
        print(f"Architecture: {gguf.architecture}")
        print(f"Layers: {gguf.block_count}")
        
        # Read a specific tensor
        tensor_data = reader.read_tensor("blk.0.attn_q.weight")
    """
    
    GGUF_MAGIC = b"GGUF"
    SUPPORTED_VERSIONS = {2, 3}
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._file: Optional[BinaryIO] = None
        self._gguf: Optional[GGUFFile] = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def open(self) -> None:
        """Open the GGUF file for reading."""
        if self._file is None:
            self._file = open(self.path, "rb")
    
    def close(self) -> None:
        """Close the GGUF file."""
        if self._file is not None:
            self._file.close()
            self._file = None
    
    def read(self) -> GGUFFile:
        """Read and parse the GGUF file."""
        if self._gguf is not None:
            return self._gguf
        
        self.open()
        
        # Read header
        header = self._read_header()
        
        # Read metadata
        metadata = {}
        for _ in range(header.metadata_kv_count):
            key, value = self._read_kv()
            metadata[key] = value
        
        # Read tensor infos
        tensors = {}
        for _ in range(header.tensor_count):
            tensor_info = self._read_tensor_info()
            tensors[tensor_info.name] = tensor_info
        
        # Tensor data starts after alignment
        current_pos = self._file.tell()
        alignment = metadata.get("general.alignment", 32)
        tensor_data_offset = (current_pos + alignment - 1) // alignment * alignment
        
        self._gguf = GGUFFile(
            path=self.path,
            header=header,
            metadata=metadata,
            tensors=tensors,
            tensor_data_offset=tensor_data_offset,
        )
        
        return self._gguf
    
    def _read_header(self) -> GGUFHeader:
        """Read the GGUF header."""
        magic = self._file.read(4)
        if magic != self.GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic!r}, expected {self.GGUF_MAGIC!r}")
        
        version = struct.unpack("<I", self._file.read(4))[0]
        if version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported GGUF version: {version}")
        
        tensor_count = struct.unpack("<Q", self._file.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", self._file.read(8))[0]
        
        return GGUFHeader(
            magic=magic,
            version=version,
            tensor_count=tensor_count,
            metadata_kv_count=metadata_kv_count,
        )
    
    def _read_string(self) -> str:
        """Read a length-prefixed string."""
        length = struct.unpack("<Q", self._file.read(8))[0]
        return self._file.read(length).decode("utf-8")
    
    def _read_value(self, value_type: GGUFValueType) -> Any:
        """Read a value of the given type."""
        if value_type == GGUFValueType.UINT8:
            return struct.unpack("<B", self._file.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack("<b", self._file.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack("<H", self._file.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack("<h", self._file.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack("<I", self._file.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack("<i", self._file.read(4))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack("<f", self._file.read(4))[0]
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack("<Q", self._file.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack("<q", self._file.read(8))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack("<d", self._file.read(8))[0]
        elif value_type == GGUFValueType.BOOL:
            return struct.unpack("<B", self._file.read(1))[0] != 0
        elif value_type == GGUFValueType.STRING:
            return self._read_string()
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array()
        else:
            raise ValueError(f"Unknown value type: {value_type}")
    
    def _read_array(self) -> List[Any]:
        """Read an array value."""
        element_type = GGUFValueType(struct.unpack("<I", self._file.read(4))[0])
        length = struct.unpack("<Q", self._file.read(8))[0]
        return [self._read_value(element_type) for _ in range(length)]
    
    def _read_kv(self) -> tuple:
        """Read a key-value pair."""
        key = self._read_string()
        value_type = GGUFValueType(struct.unpack("<I", self._file.read(4))[0])
        value = self._read_value(value_type)
        return key, value
    
    def _read_tensor_info(self) -> GGUFTensorInfo:
        """Read tensor info."""
        name = self._read_string()
        n_dims = struct.unpack("<I", self._file.read(4))[0]
        dims = [struct.unpack("<Q", self._file.read(8))[0] for _ in range(n_dims)]
        dtype = GGMLQuantType(struct.unpack("<I", self._file.read(4))[0])
        offset = struct.unpack("<Q", self._file.read(8))[0]
        
        return GGUFTensorInfo(
            name=name,
            n_dims=n_dims,
            dims=dims,
            dtype=dtype,
            offset=offset,
        )
    
    def read_tensor(self, name: str) -> np.ndarray:
        """
        Read tensor data from the file.
        
        Note: Returns raw bytes for quantized tensors.
        Use dequantize() to convert to float.
        """
        if self._gguf is None:
            self.read()
        
        tensor_info = self._gguf.tensors.get(name)
        if tensor_info is None:
            raise KeyError(f"Tensor not found: {name}")
        
        # Seek to tensor data
        self._file.seek(self._gguf.tensor_data_offset + tensor_info.offset)
        
        # Read raw data
        data = self._file.read(tensor_info.n_bytes)
        
        # For F32/F16, convert to numpy array
        if tensor_info.dtype == GGMLQuantType.F32:
            arr = np.frombuffer(data, dtype=np.float32)
            return arr.reshape(tensor_info.dims)
        elif tensor_info.dtype == GGMLQuantType.F16:
            arr = np.frombuffer(data, dtype=np.float16)
            return arr.reshape(tensor_info.dims)
        else:
            # Return raw bytes for quantized data
            return np.frombuffer(data, dtype=np.uint8)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get a summary of the model."""
        if self._gguf is None:
            self.read()
        
        gguf = self._gguf
        
        # Calculate total size
        total_bytes = sum(t.n_bytes for t in gguf.tensors.values())
        
        return {
            "path": str(gguf.path),
            "architecture": gguf.architecture,
            "name": gguf.name,
            "quantization": gguf.quantization_type,
            "context_length": gguf.context_length,
            "embedding_length": gguf.embedding_length,
            "layers": gguf.block_count,
            "heads": gguf.head_count,
            "kv_heads": gguf.head_count_kv,
            "vocab_size": gguf.vocab_size,
            "tensor_count": len(gguf.tensors),
            "total_size_gb": total_bytes / (1024**3),
            "metadata_keys": list(gguf.metadata.keys()),
        }


def is_gguf_file(path: Union[str, Path]) -> bool:
    """Check if a file is a GGUF file."""
    path = Path(path)
    if not path.exists():
        return False
    
    # Check extension
    if path.suffix.lower() in (".gguf", ".ggml"):
        return True
    
    # Check magic bytes
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            return magic == GGUFReader.GGUF_MAGIC
    except Exception:
        return False
