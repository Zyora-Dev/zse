"""
Safetensors Model Loader

Efficient loading from safetensors files with:
- Memory-mapped I/O for large models
- Streaming weight iteration
- Direct GPU loading
- Parallel shard loading
"""

import os
import json
import mmap
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any
import logging

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from .base import BaseModelLoader, LoadConfig, ModelInfo, ShardInfo, LoadProgress

logger = logging.getLogger(__name__)


class MMapTensorLoader:
    """
    Memory-mapped tensor loader for large files.
    
    Loads tensors directly from disk without copying entire file to memory.
    Essential for loading 70B+ models on systems with limited RAM.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        self._mmap = None
        self._header = None
        self._tensors_offset = 0
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def open(self):
        """Open file and parse header."""
        self._file = open(self.file_path, 'rb')
        
        # Safetensors format:
        # - 8 bytes: header size (little endian u64)
        # - N bytes: JSON header
        # - Rest: tensor data
        
        header_size_bytes = self._file.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        
        header_json = self._file.read(header_size).decode('utf-8')
        self._header = json.loads(header_json)
        
        self._tensors_offset = 8 + header_size
        
        # Memory-map the file
        self._file.seek(0)
        self._mmap = mmap.mmap(
            self._file.fileno(),
            0,
            access=mmap.ACCESS_READ
        )
    
    def close(self):
        """Close file and release mmap."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
    
    def get_tensor_names(self) -> List[str]:
        """Get all tensor names in file."""
        if self._header is None:
            raise RuntimeError("File not opened")
        return [k for k in self._header.keys() if k != "__metadata__"]
    
    def get_tensor_info(self, name: str) -> Dict[str, Any]:
        """Get tensor metadata (dtype, shape, offsets)."""
        if self._header is None:
            raise RuntimeError("File not opened")
        return self._header.get(name, {})
    
    def load_tensor(
        self,
        name: str,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Load a single tensor from the memory-mapped file.
        
        This is much more memory-efficient than loading the entire file,
        as only the requested tensor's bytes are read.
        """
        if self._header is None or self._mmap is None:
            raise RuntimeError("File not opened")
        
        info = self._header.get(name)
        if info is None:
            raise KeyError(f"Tensor '{name}' not found in file")
        
        dtype_str = info["dtype"]
        shape = info["shape"]
        data_offsets = info["data_offsets"]
        start, end = data_offsets
        
        # Map dtype string to torch dtype
        dtype_map = {
            "F16": torch.float16,
            "F32": torch.float32,
            "BF16": torch.bfloat16,
            "I8": torch.int8,
            "I16": torch.int16,
            "I32": torch.int32,
            "I64": torch.int64,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        torch_dtype = dtype_map.get(dtype_str, torch.float32)
        
        # Read tensor data from mmap - ZERO COPY with memoryview
        # Critical: Don't use bytearray() which copies data!
        tensor_start = self._tensors_offset + start
        tensor_end = self._tensors_offset + end
        
        # Create tensor directly from mmap buffer (zero-copy)
        # Use memoryview to avoid copying
        tensor = torch.frombuffer(
            memoryview(self._mmap)[tensor_start:tensor_end],
            dtype=torch_dtype
        ).reshape(shape).clone()  # clone() to own the data after mmap closes
        
        if device != "cpu":
            # Use non_blocking for async GPU transfer
            tensor = tensor.to(device, non_blocking=True)
        
        return tensor
    
    def iterate_tensors(
        self,
        device: str = "cpu",
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over all tensors in file."""
        for name in self.get_tensor_names():
            yield name, self.load_tensor(name, device)


class SafetensorsLoader(BaseModelLoader):
    """
    Load models from safetensors files.
    
    Features:
    - Memory-mapped loading for large models
    - Direct GPU loading to avoid CPU memory
    - Parallel shard loading
    - Progress tracking
    """
    
    def __init__(self, config: Optional[LoadConfig] = None):
        super().__init__(config)
        
        if not SAFETENSORS_AVAILABLE:
            logger.warning(
                "safetensors package not installed. "
                "Install with: pip install safetensors"
            )
    
    def _find_weight_files(self, model_path: str) -> List[str]:
        """Find all safetensors files in model directory."""
        model_path = Path(model_path)
        
        if model_path.is_file():
            return [str(model_path)]
        
        # Look for safetensors files
        patterns = [
            "*.safetensors",
            "model*.safetensors",
        ]
        
        files = []
        for pattern in patterns:
            files.extend(model_path.glob(pattern))
        
        # Sort by shard index if present
        def sort_key(f):
            name = f.stem
            # Handle patterns like model-00001-of-00010
            if "-of-" in name:
                parts = name.split("-")
                for i, p in enumerate(parts):
                    if p.isdigit():
                        return int(p)
            return name
        
        return [str(f) for f in sorted(set(files), key=sort_key)]
    
    def load_model_info(self, model_path: str) -> ModelInfo:
        """Load model info from config.json."""
        model_path = Path(model_path)
        
        if model_path.is_file():
            model_path = model_path.parent
        
        config_file = model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        with open(config_file) as f:
            config = json.load(f)
        
        info = ModelInfo.from_config(config, name=model_path.name)
        info.config_file = str(config_file)
        info.weight_files = self._find_weight_files(str(model_path))
        
        # Find tokenizer files
        tokenizer_files = []
        for pattern in ["tokenizer*.json", "*.model", "vocab.json", "merges.txt"]:
            tokenizer_files.extend(model_path.glob(pattern))
        info.tokenizer_files = [str(f) for f in tokenizer_files]
        
        self._model_info = info
        return info
    
    def load_weights(
        self,
        model_path: str,
        model: torch.nn.Module,
        progress_callback=None,
    ) -> torch.nn.Module:
        """
        Load weights into a model instance.
        
        Uses memory-mapping when enabled for efficient loading.
        """
        weight_files = self._find_weight_files(model_path)
        
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found in {model_path}")
        
        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in weight_files)
        progress = LoadProgress(total_size, progress_callback)
        
        # Build state dict key mapping
        model_state = model.state_dict()
        
        for weight_file in weight_files:
            logger.info(f"Loading {weight_file}")
            
            if self.config.use_mmap:
                self._load_file_mmap(weight_file, model, model_state, progress)
            else:
                self._load_file_direct(weight_file, model, model_state, progress)
        
        return model
    
    def _load_file_mmap(
        self,
        file_path: str,
        model: torch.nn.Module,
        model_state: Dict[str, torch.Tensor],
        progress: LoadProgress,
    ):
        """Load weights using safetensors direct GPU loading (fastest)."""
        progress.update(0, os.path.basename(file_path))
        
        # Use safetensors' optimized direct GPU loading when available
        target_device = self.config.device if self.config.device != "cpu" else "cpu"
        
        if SAFETENSORS_AVAILABLE and target_device != "cpu":
            # FAST PATH: Direct disk → GPU with safetensors
            # This uses memory-mapping internally and avoids CPU copy
            with safe_open(file_path, framework="pt", device=target_device) as f:
                for name in f.keys():
                    param_name = self._map_param_name(name, model_state)
                    if param_name is None:
                        logger.debug(f"Skipping unmapped tensor: {name}")
                        continue
                    
                    # Load directly to GPU - no CPU copy!
                    tensor = f.get_tensor(name)
                    tensor = self._quantize_tensor(tensor, name)
                    self._set_parameter(model, param_name, tensor)
                    
                    progress.update(tensor.numel() * tensor.element_size())
        else:
            # FALLBACK: Manual mmap loading
            with MMapTensorLoader(file_path) as loader:
                for name in loader.get_tensor_names():
                    param_name = self._map_param_name(name, model_state)
                    if param_name is None:
                        logger.debug(f"Skipping unmapped tensor: {name}")
                        continue
                    
                    param = model_state.get(param_name)
                    device = str(param.device) if param is not None else self.config.device
                    
                    tensor = loader.load_tensor(name, device)
                    tensor = self._quantize_tensor(tensor, name)
                    self._set_parameter(model, param_name, tensor)
                    
                    info = loader.get_tensor_info(name)
                    start, end = info["data_offsets"]
                    progress.update(end - start)
    
    def _load_file_direct(
        self,
        file_path: str,
        model: torch.nn.Module,
        model_state: Dict[str, torch.Tensor],
        progress: LoadProgress,
    ):
        """Load weights directly using safetensors library."""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors package required")
        
        # Load all tensors from file
        tensors = load_file(file_path, device="cpu")
        
        progress.update(0, os.path.basename(file_path))
        
        for name, tensor in tensors.items():
            param_name = self._map_param_name(name, model_state)
            
            if param_name is None:
                continue
            
            tensor = self._quantize_tensor(tensor, name)
            
            if self.config.device != "cpu":
                tensor = tensor.to(self.config.device, non_blocking=True)
            
            self._set_parameter(model, param_name, tensor)
            
            progress.update(tensor.numel() * tensor.element_size())
    
    def iterate_weights(
        self,
        model_path: str,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Iterate over weights without loading all at once.
        
        Useful for streaming to GPU or custom processing.
        """
        weight_files = self._find_weight_files(model_path)
        
        for weight_file in weight_files:
            if self.config.use_mmap:
                with MMapTensorLoader(weight_file) as loader:
                    for name, tensor in loader.iterate_tensors():
                        yield name, self._quantize_tensor(tensor, name)
            else:
                if SAFETENSORS_AVAILABLE:
                    tensors = load_file(weight_file, device="cpu")
                    for name, tensor in tensors.items():
                        yield name, self._quantize_tensor(tensor, name)
    
    def _map_param_name(
        self,
        tensor_name: str,
        model_state: Dict[str, torch.Tensor],
    ) -> Optional[str]:
        """
        Map safetensors tensor name to model parameter name.
        
        Handles different naming conventions between HF and ZSE.
        """
        # Direct match
        if tensor_name in model_state:
            return tensor_name
        
        # Common transformations
        transforms = [
            # HF -> ZSE naming
            (r"model\.", ""),
            (r"layers\.", "layers."),
            (r"self_attn\.", "attention."),
            (r"mlp\.", "feed_forward."),
            (r"q_proj", "wq"),
            (r"k_proj", "wk"),
            (r"v_proj", "wv"),
            (r"o_proj", "wo"),
            (r"gate_proj", "w1"),
            (r"up_proj", "w3"),
            (r"down_proj", "w2"),
            (r"input_layernorm", "attention_norm"),
            (r"post_attention_layernorm", "ffn_norm"),
        ]
        
        name = tensor_name
        for pattern, replacement in transforms:
            import re
            name = re.sub(pattern, replacement, name)
        
        if name in model_state:
            return name
        
        # Try with/without "model." prefix
        if name.startswith("model."):
            name = name[6:]
        else:
            name = "model." + name
        
        if name in model_state:
            return name
        
        return None
    
    def _set_parameter(
        self,
        model: torch.nn.Module,
        name: str,
        tensor: torch.Tensor,
    ):
        """Set a parameter in the model."""
        parts = name.split(".")
        module = model
        
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        param_name = parts[-1]
        
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            if isinstance(param, torch.nn.Parameter):
                param.data.copy_(tensor)
            else:
                setattr(module, param_name, tensor)


class ShardedLoader(SafetensorsLoader):
    """
    Load sharded models efficiently across multiple GPUs.
    
    Distributes shards across GPUs based on memory constraints.
    """
    
    def __init__(self, config: Optional[LoadConfig] = None):
        super().__init__(config)
        self._shard_info: List[ShardInfo] = []
    
    def analyze_shards(self, model_path: str) -> List[ShardInfo]:
        """Analyze shards and plan distribution."""
        weight_files = self._find_weight_files(model_path)
        
        shards = []
        for file_path in weight_files:
            size = os.path.getsize(file_path)
            
            # Get tensor names in shard
            if self.config.use_mmap:
                with MMapTensorLoader(file_path) as loader:
                    tensor_names = loader.get_tensor_names()
            else:
                with safe_open(file_path, framework="pt") as f:
                    tensor_names = list(f.keys())
            
            shards.append(ShardInfo(
                filename=file_path,
                tensor_names=tensor_names,
                size_bytes=size,
            ))
        
        self._shard_info = shards
        return shards
    
    def load_to_devices(
        self,
        model_path: str,
        model: torch.nn.Module,
        device_map: Dict[str, int],
        progress_callback=None,
    ) -> torch.nn.Module:
        """
        Load model with weights distributed to specific devices.
        
        Args:
            model_path: Path to model
            model: Model instance
            device_map: Mapping of parameter names to device IDs
            progress_callback: Optional progress callback
        """
        if not self._shard_info:
            self.analyze_shards(model_path)
        
        total_size = sum(s.size_bytes for s in self._shard_info)
        progress = LoadProgress(total_size, progress_callback)
        
        model_state = model.state_dict()
        
        def get_device(param_name: str) -> str:
            """Get device for parameter."""
            # Check exact match
            if param_name in device_map:
                return f"cuda:{device_map[param_name]}"
            
            # Check prefix match
            for prefix, device_id in device_map.items():
                if param_name.startswith(prefix):
                    return f"cuda:{device_id}"
            
            return self.config.device
        
        for shard in self._shard_info:
            logger.info(f"Loading shard: {shard.filename}")
            
            with MMapTensorLoader(shard.filename) as loader:
                progress.update(0, os.path.basename(shard.filename))
                
                for name in loader.get_tensor_names():
                    param_name = self._map_param_name(name, model_state)
                    
                    if param_name is None:
                        continue
                    
                    device = get_device(param_name)
                    
                    # Load directly to target device
                    tensor = loader.load_tensor(name, "cpu")
                    tensor = self._quantize_tensor(tensor, name)
                    tensor = tensor.to(device, non_blocking=True)
                    
                    self._set_parameter(model, param_name, tensor)
                    
                    info = loader.get_tensor_info(name)
                    start, end = info["data_offsets"]
                    progress.update(end - start)
        
        return model
