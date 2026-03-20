"""
Model Architectures

Native ZSE implementations of popular model architectures:
- LLaMA (1, 2, 3, 3.1, 3.2, 3.3)
- Mistral (7B, Instruct)
- More coming: Qwen, Phi, Gemma

Each architecture is optimized for:
- zAttention integration
- zQuantize compatibility
- zKV cache management
- Efficient generation
"""

from .base import (
    ModelConfig,
    BaseModel,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .llama import (
    LlamaConfig,
    LlamaModel,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
)

from .mistral import (
    MistralConfig,
    MistralModel,
    MistralAttention,
    MistralDecoderLayer,
)


def get_model_class(model_type: str):
    """Get model class by architecture type."""
    models = {
        "llama": LlamaModel,
        "mistral": MistralModel,
        "LlamaForCausalLM": LlamaModel,
        "MistralForCausalLM": MistralModel,
    }
    return models.get(model_type)


__all__ = [
    # Base
    "ModelConfig",
    "BaseModel",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "repeat_kv",
    
    # Llama
    "LlamaConfig",
    "LlamaModel",
    "LlamaAttention",
    "LlamaMLP",
    "LlamaDecoderLayer",
    
    # Mistral
    "MistralConfig",
    "MistralModel",
    "MistralAttention",
    "MistralDecoderLayer",
    
    # Utilities
    "get_model_class",
]
