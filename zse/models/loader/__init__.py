"""
Model Loader

Supports loading models from multiple sources:
- HuggingFace Hub (transformers format)
- Local safetensors files
- Local PyTorch checkpoints
- ZSE native format (.zse)

Features:
- Memory-mapped loading for large models
- On-the-fly quantization during load
- Memory estimation before loading
- Multi-GPU weight sharding
- Progress reporting
"""

from .base import (
    BaseModelLoader,
    LoadConfig,
    ModelInfo,
    ShardInfo,
    LoadProgress,
    QuantizationType,
)

from .safetensors_loader import (
    SafetensorsLoader,
    ShardedLoader,
    MMapTensorLoader,
)

from .huggingface_loader import (
    HuggingFaceLoader,
    ModelHub,
)


__all__ = [
    "BaseModelLoader",
    "LoadConfig",
    "ModelInfo",
    "ShardInfo",
    "LoadProgress",
    "QuantizationType",
    "SafetensorsLoader",
    "ShardedLoader",
    "MMapTensorLoader",
    "HuggingFaceLoader",
    "ModelHub",
]
