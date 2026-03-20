"""
ZSE Models Module

Handles model loading and architecture implementations:
- loader: Load models from HuggingFace, safetensors, .zse format
- architectures: Model implementations (LLaMA, Mistral, Qwen, etc.)

Usage:
    from zse.models import LlamaModel, ModelLoader
    
    # Load from HuggingFace
    model = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")
    
    # Or use manual loading
    loader = ModelLoader()
    info = loader.load_model_info("path/to/model")
    print(info)  # Shows memory requirements
"""

from .loader.base import LoadConfig, ModelInfo, QuantizationType
from .loader.safetensors_loader import SafetensorsLoader, ShardedLoader, MMapTensorLoader
from .loader.huggingface_loader import HuggingFaceLoader, ModelHub

from .architectures.base import ModelConfig, BaseModel, RMSNorm
from .architectures.llama import LlamaModel, LlamaConfig
from .architectures.mistral import MistralModel, MistralConfig


# Convenience alias
ModelLoader = ModelHub


def get_model_architecture(model_type: str):
    """Get model class by type name."""
    architectures = {
        "llama": LlamaModel,
        "mistral": MistralModel,
        "LlamaForCausalLM": LlamaModel,
        "MistralForCausalLM": MistralModel,
    }
    return architectures.get(model_type)


# Registry and Discovery
from .registry import (
    ModelRegistry, ModelSpec, ModelCategory, ModelSize,
    get_registry, REGISTRY
)
from .discovery import (
    ModelDiscovery, HFModelInfo, SUPPORTED_ARCHITECTURES,
    get_discovery, search_models, check_model
)


__all__ = [
    # Loaders
    "LoadConfig",
    "ModelInfo",
    "QuantizationType",
    "SafetensorsLoader",
    "ShardedLoader",
    "MMapTensorLoader",
    "HuggingFaceLoader",
    "ModelHub",
    "ModelLoader",
    
    # Base classes
    "ModelConfig",
    "BaseModel",
    "RMSNorm",
    
    # Models
    "LlamaModel",
    "LlamaConfig",
    "MistralModel",
    "MistralConfig",
    
    # Registry & Discovery
    "ModelRegistry",
    "ModelSpec", 
    "ModelCategory",
    "ModelSize",
    "get_registry",
    "REGISTRY",
    "ModelDiscovery",
    "HFModelInfo",
    "SUPPORTED_ARCHITECTURES",
    "get_discovery",
    "search_models",
    "check_model",    
    # Utilities
    "get_model_architecture",
]