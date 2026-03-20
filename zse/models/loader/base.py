"""
Base Model Loader

Provides the foundation for loading models with:
- Memory estimation
- Progress tracking
- Device mapping
- Quantization during load
"""

import os
import json
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Quantization types supported during loading."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP8 = "fp8"
    NF4 = "nf4"  # QLoRA-style


@dataclass
class LoadConfig:
    """Configuration for model loading."""
    
    # Device settings
    device: str = "cuda"
    device_map: Optional[Union[str, Dict[str, int]]] = "auto"
    max_memory: Optional[Dict[int, str]] = None
    
    # Quantization
    quantization: QuantizationType = QuantizationType.NONE
    quantize_layers: Optional[List[str]] = None  # Specific layers to quantize
    
    # Loading behavior
    use_mmap: bool = True  # Memory-map large files
    streaming: bool = True  # Load weights on-demand
    low_cpu_memory: bool = True  # Minimize CPU memory usage
    
    # Performance
    num_threads: int = 4  # Parallel loading threads
    pin_memory: bool = False  # Pin memory for faster GPU transfer
    
    # Precision
    dtype: torch.dtype = torch.float16
    compute_dtype: torch.dtype = torch.float16
    
    # Trust settings
    trust_remote_code: bool = False
    
    # Cache
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.quantization, str):
            self.quantization = QuantizationType(self.quantization)


@dataclass
class ModelInfo:
    """Information about a loaded or to-be-loaded model."""
    
    name: str
    architecture: str
    num_parameters: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: bool = False
    
    # Memory estimates
    fp16_memory_gb: float = 0.0
    int8_memory_gb: float = 0.0
    int4_memory_gb: float = 0.0
    
    # Files
    weight_files: List[str] = field(default_factory=list)
    config_file: Optional[str] = None
    tokenizer_files: List[str] = field(default_factory=list)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], name: str = "unknown") -> "ModelInfo":
        """Create ModelInfo from HuggingFace config dict."""
        # Extract architecture type
        arch = config.get("architectures", ["unknown"])[0]
        
        # Handle different config formats
        hidden_size = config.get("hidden_size", config.get("d_model", 4096))
        num_layers = config.get("num_hidden_layers", config.get("n_layers", 32))
        num_heads = config.get("num_attention_heads", config.get("n_heads", 32))
        
        # GQA support
        num_kv_heads = config.get(
            "num_key_value_heads",
            config.get("num_kv_heads", num_heads)
        )
        
        vocab_size = config.get("vocab_size", 32000)
        max_pos = config.get("max_position_embeddings", config.get("max_seq_len", 4096))
        
        # Calculate parameters
        head_dim = hidden_size // num_heads
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        
        # Rough parameter count
        # Embedding: vocab * hidden
        # Each layer: 4 * hidden^2 (QKV+O) + 3 * hidden * intermediate (MLP)
        # LM head: vocab * hidden (unless tied)
        embed_params = vocab_size * hidden_size
        layer_params = (4 * hidden_size * hidden_size + 
                       3 * hidden_size * intermediate_size)
        lm_head_params = 0 if config.get("tie_word_embeddings", False) else vocab_size * hidden_size
        
        total_params = embed_params + (num_layers * layer_params) + lm_head_params
        
        # Memory estimates (in GB)
        bytes_per_param_fp16 = 2
        bytes_per_param_int8 = 1
        bytes_per_param_int4 = 0.5
        
        fp16_gb = (total_params * bytes_per_param_fp16) / (1024**3)
        int8_gb = (total_params * bytes_per_param_int8) / (1024**3)
        int4_gb = (total_params * bytes_per_param_int4) / (1024**3)
        
        return cls(
            name=name,
            architecture=arch,
            num_parameters=total_params,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            vocab_size=vocab_size,
            max_position_embeddings=max_pos,
            rope_theta=config.get("rope_theta", 10000.0),
            rope_scaling=config.get("rope_scaling"),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
            fp16_memory_gb=fp16_gb,
            int8_memory_gb=int8_gb,
            int4_memory_gb=int4_gb,
        )
    
    def __str__(self) -> str:
        return (
            f"Model: {self.name}\n"
            f"  Architecture: {self.architecture}\n"
            f"  Parameters: {self.num_parameters / 1e9:.2f}B\n"
            f"  Layers: {self.num_layers}\n"
            f"  Hidden size: {self.hidden_size}\n"
            f"  Attention heads: {self.num_attention_heads} (KV: {self.num_key_value_heads})\n"
            f"  Memory (FP16): {self.fp16_memory_gb:.1f} GB\n"
            f"  Memory (INT8): {self.int8_memory_gb:.1f} GB\n"
            f"  Memory (INT4): {self.int4_memory_gb:.1f} GB"
        )


@dataclass
class ShardInfo:
    """Information about weight shards."""
    
    filename: str
    tensor_names: List[str]
    size_bytes: int
    device: int = 0  # GPU device ID


class LoadProgress:
    """Track loading progress."""
    
    def __init__(self, total_bytes: int, callback=None):
        self.total_bytes = total_bytes
        self.loaded_bytes = 0
        self.current_file = ""
        self.callback = callback
    
    def update(self, bytes_loaded: int, filename: str = ""):
        """Update progress."""
        self.loaded_bytes += bytes_loaded
        if filename:
            self.current_file = filename
        
        if self.callback:
            progress = self.loaded_bytes / self.total_bytes if self.total_bytes > 0 else 0
            self.callback(progress, self.current_file)
    
    @property
    def progress(self) -> float:
        """Get progress as fraction."""
        return self.loaded_bytes / self.total_bytes if self.total_bytes > 0 else 0


class BaseModelLoader(ABC):
    """
    Abstract base class for model loaders.
    
    Subclasses implement loading from different sources:
    - HuggingFaceLoader: Load from HuggingFace Hub
    - SafetensorsLoader: Load from local safetensors
    - ZSEFormatLoader: Load from native ZSE format
    """
    
    def __init__(self, config: Optional[LoadConfig] = None):
        self.config = config or LoadConfig()
        self._model_info: Optional[ModelInfo] = None
    
    @abstractmethod
    def load_model_info(self, model_path: str) -> ModelInfo:
        """Load model metadata without loading weights."""
        pass
    
    @abstractmethod
    def load_weights(
        self,
        model_path: str,
        model: torch.nn.Module,
        progress_callback=None,
    ) -> torch.nn.Module:
        """Load weights into a model."""
        pass
    
    @abstractmethod
    def iterate_weights(
        self,
        model_path: str,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate over weights without loading all at once."""
        pass
    
    def estimate_memory(self, model_path: str) -> Dict[str, float]:
        """Estimate memory requirements for different quantization levels."""
        info = self.load_model_info(model_path)
        return {
            "fp16_gb": info.fp16_memory_gb,
            "int8_gb": info.int8_memory_gb,
            "int4_gb": info.int4_memory_gb,
        }
    
    def get_device_map(
        self,
        model_info: ModelInfo,
        available_memory: Optional[Dict[int, int]] = None,
    ) -> Dict[str, int]:
        """
        Calculate optimal device mapping for multi-GPU.
        
        Returns a dict mapping layer names to device IDs.
        """
        if self.config.device == "cpu":
            return {"": "cpu"}
        
        if self.config.device_map == "auto":
            return self._auto_device_map(model_info, available_memory)
        elif isinstance(self.config.device_map, dict):
            return self.config.device_map
        else:
            return {"": self.config.device}
    
    def _auto_device_map(
        self,
        model_info: ModelInfo,
        available_memory: Optional[Dict[int, int]] = None,
    ) -> Dict[str, int]:
        """Automatically distribute model across available GPUs."""
        if not torch.cuda.is_available():
            return {"": "cpu"}
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            return {"": 0}
        
        # Get available memory per GPU
        if available_memory is None:
            available_memory = {}
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                # Leave 2GB headroom
                available_memory[i] = props.total_memory - 2 * (1024**3)
        
        # Estimate memory per layer
        quant = self.config.quantization
        if quant == QuantizationType.INT4:
            model_memory = model_info.int4_memory_gb * (1024**3)
        elif quant == QuantizationType.INT8:
            model_memory = model_info.int8_memory_gb * (1024**3)
        else:
            model_memory = model_info.fp16_memory_gb * (1024**3)
        
        memory_per_layer = model_memory / (model_info.num_layers + 2)  # +2 for embed/lm_head
        
        # Distribute layers
        device_map = {}
        current_device = 0
        current_memory = 0
        
        # Embedding layer
        device_map["model.embed_tokens"] = 0
        current_memory += memory_per_layer
        
        # Transformer layers
        for i in range(model_info.num_layers):
            if current_memory + memory_per_layer > available_memory.get(current_device, 0):
                current_device = min(current_device + 1, num_gpus - 1)
                current_memory = 0
            
            device_map[f"model.layers.{i}"] = current_device
            current_memory += memory_per_layer
        
        # Final layers
        device_map["model.norm"] = current_device
        device_map["lm_head"] = current_device
        
        return device_map
    
    def _quantize_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Quantize a tensor during loading."""
        quant = self.config.quantization
        
        if quant == QuantizationType.NONE:
            return tensor.to(self.config.dtype)
        
        # Check if this layer should be quantized
        if self.config.quantize_layers:
            should_quantize = any(
                pattern in name for pattern in self.config.quantize_layers
            )
            if not should_quantize:
                return tensor.to(self.config.dtype)
        
        if quant == QuantizationType.INT8:
            return self._quantize_int8(tensor)
        elif quant == QuantizationType.INT4:
            return self._quantize_int4(tensor)
        else:
            return tensor.to(self.config.dtype)
    
    def _quantize_int8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize to INT8 with per-channel scaling."""
        # Per-channel absmax quantization
        if tensor.dim() < 2:
            return tensor.to(self.config.dtype)
        
        scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 127.0
        scale = scale.clamp(min=1e-5)
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        
        # Store scale for dequantization (in practice, fuse with compute)
        # For now, return dequantized for compatibility
        return (quantized.float() * scale).to(self.config.dtype)
    
    def _quantize_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize to INT4 with group-wise scaling."""
        if tensor.dim() < 2:
            return tensor.to(self.config.dtype)
        
        # Group-wise quantization (group size 128)
        group_size = 128
        original_shape = tensor.shape
        
        # Reshape for grouping
        if tensor.shape[-1] % group_size != 0:
            return tensor.to(self.config.dtype)  # Can't group evenly
        
        tensor = tensor.reshape(-1, group_size)
        scale = tensor.abs().max(dim=-1, keepdim=True)[0] / 7.0
        scale = scale.clamp(min=1e-5)
        
        quantized = (tensor / scale).round().clamp(-8, 7)
        dequantized = (quantized * scale).reshape(original_shape)
        
        return dequantized.to(self.config.dtype)
