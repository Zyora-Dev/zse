"""
ZSE Format Reader

Reads .zse format files with memory-mapping for efficient access.

Features:
- Memory-mapped file access (zero-copy)
- Layer-by-layer streaming (for zStream)
- Lazy loading (only load what's needed)
- Embedded tokenizer extraction
"""

import json
import mmap
import struct
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
import io
import base64

import torch
import numpy as np

from .spec import (
    ZSEHeader,
    TensorInfo,
    LayerGroup,
    TensorDType,
    decode_header,
    zse_dtype_to_torch,
    ZSE_MAGIC,
)


class ZSEReader:
    """
    Reader for .zse format files.
    
    Supports efficient memory-mapped access and layer streaming.
    
    Usage:
        # Load full model
        reader = ZSEReader("model.zse")
        state_dict = reader.load_state_dict()
        
        # Stream layers
        for layer_idx in range(reader.num_layers):
            layer_tensors = reader.load_layer(layer_idx)
            # Process layer...
            reader.unload_layer(layer_idx)  # Free memory
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        use_mmap: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize reader.
        
        Args:
            path: Path to .zse file
            use_mmap: Use memory mapping (recommended)
            device: Target device for loaded tensors
        """
        self.path = Path(path)
        self.use_mmap = use_mmap
        self.device = device
        
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        # Open file
        self._file = open(self.path, 'rb')
        
        # Memory map if requested
        if use_mmap:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._data = self._mmap
        else:
            self._mmap = None
            self._data = self._file.read()
        
        # Read header
        self.header, self._header_end = decode_header(self._data)
        
        # Cache for loaded tensors
        self._tensor_cache: Dict[str, torch.Tensor] = {}
        self._layer_cache: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def close(self) -> None:
        """Close the file and release resources."""
        self._tensor_cache.clear()
        self._layer_cache.clear()
        
        if self._mmap:
            self._mmap.close()
        self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def architecture(self) -> str:
        """Model architecture name."""
        return self.header.architecture
    
    @property
    def model_type(self) -> str:
        """Model type (e.g., 'llama')."""
        return self.header.model_type
    
    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        return self.header.num_hidden_layers
    
    @property
    def hidden_size(self) -> int:
        """Hidden dimension size."""
        return self.header.hidden_size
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.header.vocab_size
    
    @property
    def quantization(self) -> str:
        """Quantization type used."""
        return self.header.quantization
    
    @property
    def tensor_names(self) -> List[str]:
        """List of all tensor names."""
        return [t.name for t in self.header.tensors]
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        file_size = self.path.stat().st_size
        
        return {
            "path": str(self.path),
            "file_size_gb": file_size / 1e9,
            "architecture": self.architecture,
            "model_type": self.model_type,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "quantization": self.quantization,
            "num_tensors": len(self.header.tensors),
            "source_model": self.header.source_model,
        }
    
    # =========================================================================
    # Tokenizer
    # =========================================================================
    
    def load_tokenizer(self, output_dir: Optional[Path] = None):
        """
        Load the embedded tokenizer.
        
        Args:
            output_dir: Directory to extract tokenizer files (temp if None)
            
        Returns:
            Tokenizer instance
        """
        from transformers import AutoTokenizer
        
        # Read tokenizer data
        offset = self.header.tokenizer_offset
        size_bytes = self._data[offset:offset + 4]
        tokenizer_size = struct.unpack('<I', size_bytes)[0]
        
        tokenizer_json = self._data[offset + 4:offset + 4 + tokenizer_size].decode('utf-8')
        tokenizer_data = json.loads(tokenizer_json)
        
        # Extract to directory
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="zse_tokenizer_")
            output_dir = Path(temp_dir)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write tokenizer files
        for filename, content_b64 in tokenizer_data.items():
            content = base64.b64decode(content_b64)
            with open(output_dir / filename, 'wb') as f:
                f.write(content)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
        
        return tokenizer
    
    # =========================================================================
    # Tensor Loading
    # =========================================================================
    
    def load_tensor(
        self,
        name: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Load a single tensor by name.
        
        Args:
            name: Tensor name
            dtype: Override dtype (None = use stored dtype)
            device: Override device (None = use reader default)
            
        Returns:
            Loaded tensor
        """
        # Check cache
        if name in self._tensor_cache:
            return self._tensor_cache[name]
        
        # Find tensor info
        tensor_info = self.header.get_tensor(name)
        if tensor_info is None:
            raise KeyError(f"Tensor not found: {name}")
        
        # Load from file
        tensor = self._read_tensor(tensor_info, dtype, device)
        
        return tensor
    
    def load_state_dict(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Load complete state dict.
        
        Args:
            dtype: Override dtype for all tensors
            device: Override device for all tensors
            progress: Show progress bar
            
        Returns:
            State dictionary
        """
        from tqdm import tqdm
        
        state_dict = {}
        
        tensors = self.header.tensors
        if progress:
            tensors = tqdm(tensors, desc="Loading tensors")
        
        for tensor_info in tensors:
            tensor = self._read_tensor(tensor_info, dtype, device)
            state_dict[tensor_info.name] = tensor
        
        return state_dict
    
    # =========================================================================
    # Layer Streaming
    # =========================================================================
    
    def load_layer(
        self,
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
        cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Load a single layer's tensors.
        
        Optimized for layer streaming - loads all tensors for one layer.
        
        Args:
            layer_idx: Layer index to load
            dtype: Override dtype
            device: Override device
            cache: Cache in memory
            
        Returns:
            Dictionary of layer tensors
        """
        # Check cache
        if cache and layer_idx in self._layer_cache:
            return self._layer_cache[layer_idx]
        
        # Get layer group
        layer_group = self.header.get_layer_group(layer_idx)
        
        if layer_group is not None:
            # Use optimized grouped read
            layer_tensors = self._read_layer_group(layer_group, dtype, device)
        else:
            # Fallback: find tensors by name pattern
            layer_tensors = {}
            tensor_infos = self.header.get_layer_tensors(layer_idx)
            
            for tensor_info in tensor_infos:
                tensor = self._read_tensor(tensor_info, dtype, device)
                layer_tensors[tensor_info.name] = tensor
        
        if cache:
            self._layer_cache[layer_idx] = layer_tensors
        
        return layer_tensors
    
    def unload_layer(self, layer_idx: int) -> None:
        """
        Unload a layer from cache to free memory.
        
        Args:
            layer_idx: Layer index to unload
        """
        if layer_idx in self._layer_cache:
            del self._layer_cache[layer_idx]
    
    def iter_layers(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> Iterator[Tuple[int, Dict[str, torch.Tensor]]]:
        """
        Iterate over layers one at a time.
        
        Memory-efficient: only one layer in memory at a time.
        
        Yields:
            (layer_idx, layer_tensors) tuples
        """
        for layer_idx in range(self.num_layers):
            layer_tensors = self.load_layer(layer_idx, dtype, device, cache=False)
            yield layer_idx, layer_tensors
    
    def load_non_layer_tensors(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load tensors that don't belong to any layer.
        
        Includes: embeddings, lm_head, final layernorm, etc.
        
        Returns:
            Dictionary of non-layer tensors
        """
        result = {}
        
        # Get layer tensor names
        layer_names = set()
        for group in self.header.layer_groups:
            layer_names.update(group.tensor_names)
        
        # Load tensors not in any layer
        for tensor_info in self.header.tensors:
            if tensor_info.name not in layer_names:
                tensor = self._read_tensor(tensor_info, dtype, device)
                result[tensor_info.name] = tensor
        
        return result
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _read_tensor(
        self,
        info: TensorInfo,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Read a tensor from file using memory mapping."""
        device = device or self.device
        
        # Get raw bytes
        raw_bytes = self._data[info.offset:info.offset + info.size]
        
        # Determine dtype
        stored_dtype = zse_dtype_to_torch(info.dtype)
        target_dtype = dtype or stored_dtype
        
        # Convert to numpy, then torch
        np_dtype = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.float16,  # Numpy doesn't have bfloat16
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }.get(stored_dtype, np.float16)
        
        array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(info.shape)
        tensor = torch.from_numpy(array.copy())
        
        # Handle bfloat16 conversion
        if info.dtype == TensorDType.BFLOAT16:
            tensor = tensor.view(torch.bfloat16)
        
        # Convert dtype if needed
        if target_dtype != stored_dtype:
            tensor = tensor.to(target_dtype)
        
        # Move to device
        if device != "cpu":
            tensor = tensor.to(device)
        
        return tensor
    
    def _read_layer_group(
        self,
        group: LayerGroup,
        dtype: Optional[torch.dtype] = None,
        device: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Read all tensors in a layer group efficiently."""
        result = {}
        
        for name in group.tensor_names:
            tensor_info = self.header.get_tensor(name)
            if tensor_info:
                tensor = self._read_tensor(tensor_info, dtype, device)
                result[name] = tensor
        
        return result


def load_zse(
    path: Union[str, Path],
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Dict[str, torch.Tensor], Any, Dict[str, Any]]:
    """
    Convenience function to load a .zse file.
    
    Args:
        path: Path to .zse file
        device: Target device
        dtype: Override dtype
        
    Returns:
        (state_dict, tokenizer, info)
    """
    with ZSEReader(path, device=device) as reader:
        state_dict = reader.load_state_dict(dtype=dtype, device=device)
        tokenizer = reader.load_tokenizer()
        info = reader.get_info()
    
    return state_dict, tokenizer, info


class ZSEStreamLoader:
    """
    Streaming loader for large models with zStream integration.
    
    Usage:
        loader = ZSEStreamLoader("model.zse", gpu_layers=4)
        
        # Automatically manages layer streaming
        for output in loader.generate(input_ids):
            print(output)
    """
    
    def __init__(
        self,
        path: Union[str, Path],
        gpu_layers: int = 4,
        device: int = 0,
    ):
        """
        Initialize streaming loader.
        
        Args:
            path: Path to .zse file
            gpu_layers: Number of layers to keep on GPU
            device: CUDA device ID
        """
        self.reader = ZSEReader(path, use_mmap=True, device="cpu")
        self.gpu_layers = gpu_layers
        self.device = f"cuda:{device}"
        
        # Track which layers are on GPU
        self._gpu_layer_ids: List[int] = []
    
    def get_layer(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a layer, loading to GPU if needed.
        
        Implements LRU eviction if GPU is full.
        """
        # Already on GPU?
        if layer_idx in self._gpu_layer_ids:
            self._gpu_layer_ids.remove(layer_idx)
            self._gpu_layer_ids.append(layer_idx)  # Move to end (most recent)
            return self.reader._layer_cache.get(layer_idx, {})
        
        # Need to evict?
        while len(self._gpu_layer_ids) >= self.gpu_layers:
            evict_idx = self._gpu_layer_ids.pop(0)  # Remove oldest
            self.reader.unload_layer(evict_idx)
        
        # Load to GPU
        layer_tensors = self.reader.load_layer(
            layer_idx,
            device=self.device,
            cache=True,
        )
        self._gpu_layer_ids.append(layer_idx)
        
        return layer_tensors
    
    def release_layer(self, layer_idx: int) -> None:
        """Release a layer (hint that it's no longer needed)."""
        # For now, let LRU handle eviction
        pass
    
    def close(self) -> None:
        """Close reader and release resources."""
        self.reader.close()
