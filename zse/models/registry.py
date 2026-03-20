"""
ZSE Model Registry

Curated list of tested and recommended models for ZSE.
Provides model metadata including VRAM requirements, capabilities, and recommended settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelCategory(str, Enum):
    """Model categories/capabilities."""
    CHAT = "chat"
    INSTRUCT = "instruct"
    CODE = "code"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    BASE = "base"


class ModelSize(str, Enum):
    """Model size tiers for VRAM estimation."""
    TINY = "tiny"       # < 1B params, < 2GB VRAM
    SMALL = "small"     # 1-3B params, 2-6GB VRAM
    MEDIUM = "medium"   # 3-8B params, 6-16GB VRAM
    LARGE = "large"     # 8-13B params, 16-26GB VRAM
    XLARGE = "xlarge"   # 13-34B params, 26-70GB VRAM
    XXL = "xxl"         # 34B+ params, 70GB+ VRAM


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    # Identity
    model_id: str                    # HuggingFace model ID
    name: str                        # Display name
    description: str                 # Short description
    
    # Metadata
    parameters: str                  # e.g., "7B", "1.1B", "70B"
    architecture: str                # e.g., "LlamaForCausalLM"
    categories: List[ModelCategory]  # What the model is good for
    size: ModelSize                  # Size tier
    
    # Requirements
    vram_fp16_gb: float             # VRAM needed for FP16
    vram_int8_gb: float             # VRAM needed for INT8
    vram_int4_gb: float             # VRAM needed for INT4
    
    # Recommendations
    recommended_quant: str = "int8"  # Recommended quantization
    context_length: int = 4096       # Max context length
    
    # Status
    tested: bool = True              # Tested with ZSE
    zse_optimized: bool = False      # Has .zse optimizations
    
    # Additional info
    license: str = "unknown"
    provider: str = "community"      # meta, mistral, qwen, google, etc.
    homepage: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "architecture": self.architecture,
            "categories": [c.value for c in self.categories],
            "size": self.size.value,
            "vram": {
                "fp16_gb": self.vram_fp16_gb,
                "int8_gb": self.vram_int8_gb,
                "int4_gb": self.vram_int4_gb,
            },
            "recommended_quant": self.recommended_quant,
            "context_length": self.context_length,
            "tested": self.tested,
            "zse_optimized": self.zse_optimized,
            "license": self.license,
            "provider": self.provider,
            "homepage": self.homepage,
            "tags": self.tags,
        }


# =============================================================================
# Curated Model Registry
# =============================================================================

REGISTRY: List[ModelSpec] = [
    # =========================================================================
    # TINY MODELS (< 1B, < 2GB VRAM) - Great for testing & edge devices
    # =========================================================================
    ModelSpec(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        name="TinyLlama 1.1B Chat",
        description="Compact chat model, great for testing and edge deployment",
        parameters="1.1B",
        architecture="LlamaForCausalLM",
        categories=[ModelCategory.CHAT],
        size=ModelSize.TINY,
        vram_fp16_gb=2.2,
        vram_int8_gb=1.2,
        vram_int4_gb=0.7,
        recommended_quant="int8",
        context_length=2048,
        license="apache-2.0",
        provider="TinyLlama",
        tags=["fast", "edge", "testing"],
    ),
    
    # =========================================================================
    # SMALL MODELS (1-3B, 2-6GB VRAM) - Good balance of quality and speed
    # =========================================================================
    ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        name="Qwen2.5 0.5B Instruct",
        description="Ultra-compact instruction-following model from Alibaba",
        parameters="0.5B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT],
        size=ModelSize.TINY,
        vram_fp16_gb=1.0,
        vram_int8_gb=0.6,
        vram_int4_gb=0.4,
        recommended_quant="fp16",
        context_length=32768,
        license="apache-2.0",
        provider="Qwen",
        tags=["fast", "multilingual"],
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        name="Qwen2.5 1.5B Instruct",
        description="Compact yet capable instruction model with long context",
        parameters="1.5B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT],
        size=ModelSize.SMALL,
        vram_fp16_gb=3.0,
        vram_int8_gb=1.8,
        vram_int4_gb=1.0,
        recommended_quant="int8",
        context_length=32768,
        license="apache-2.0",
        provider="Qwen",
        tags=["multilingual", "long-context"],
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        name="Qwen2.5 3B Instruct",
        description="Excellent quality-to-size ratio for instruction following",
        parameters="3B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT],
        size=ModelSize.SMALL,
        vram_fp16_gb=6.0,
        vram_int8_gb=3.5,
        vram_int4_gb=2.0,
        recommended_quant="int8",
        context_length=32768,
        license="apache-2.0",
        provider="Qwen",
        tags=["multilingual", "long-context", "recommended"],
    ),
    ModelSpec(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        name="Phi-3 Mini 4K Instruct",
        description="Microsoft's compact but powerful reasoning model",
        parameters="3.8B",
        architecture="Phi3ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.REASONING],
        size=ModelSize.SMALL,
        vram_fp16_gb=7.6,
        vram_int8_gb=4.2,
        vram_int4_gb=2.4,
        recommended_quant="int8",
        context_length=4096,
        license="mit",
        provider="Microsoft",
        tags=["reasoning", "efficient"],
    ),
    ModelSpec(
        model_id="google/gemma-2-2b-it",
        name="Gemma 2 2B Instruct",
        description="Google's compact instruction-tuned model",
        parameters="2B",
        architecture="Gemma2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT],
        size=ModelSize.SMALL,
        vram_fp16_gb=4.0,
        vram_int8_gb=2.4,
        vram_int4_gb=1.4,
        recommended_quant="int8",
        context_length=8192,
        license="gemma",
        provider="Google",
        tags=["efficient"],
    ),
    
    # =========================================================================
    # MEDIUM MODELS (3-8B, 6-16GB VRAM) - Production quality
    # =========================================================================
    ModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen2.5 7B Instruct",
        description="High-quality instruction model with excellent multilingual support",
        parameters="7B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT, ModelCategory.CODE],
        size=ModelSize.MEDIUM,
        vram_fp16_gb=14.0,
        vram_int8_gb=8.0,
        vram_int4_gb=4.5,
        recommended_quant="int8",
        context_length=32768,
        zse_optimized=True,  # Verified .zse conversion 2024-02
        license="apache-2.0",
        provider="Qwen",
        tags=["multilingual", "long-context", "recommended", "production"],
    ),
    ModelSpec(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        name="Llama 3.2 3B Instruct",
        description="Meta's latest compact Llama model",
        parameters="3B",
        architecture="LlamaForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT],
        size=ModelSize.SMALL,
        vram_fp16_gb=6.0,
        vram_int8_gb=3.5,
        vram_int4_gb=2.0,
        recommended_quant="int8",
        context_length=8192,
        license="llama3.2",
        provider="Meta",
        tags=["recommended"],
    ),
    ModelSpec(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        name="Mistral 7B Instruct v0.3",
        description="Excellent general-purpose instruction model",
        parameters="7B",
        architecture="MistralForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT],
        size=ModelSize.MEDIUM,
        vram_fp16_gb=14.0,
        vram_int8_gb=8.0,
        vram_int4_gb=4.5,
        recommended_quant="int8",
        context_length=32768,
        zse_optimized=True,  # Verified .zse conversion 2024-02
        license="apache-2.0",
        provider="Mistral AI",
        tags=["sliding-window", "recommended", "production"],
    ),
    ModelSpec(
        model_id="google/gemma-2-9b-it",
        name="Gemma 2 9B Instruct",
        description="Google's capable mid-size model with strong reasoning",
        parameters="9B",
        architecture="Gemma2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT, ModelCategory.REASONING],
        size=ModelSize.MEDIUM,
        vram_fp16_gb=18.0,
        vram_int8_gb=10.0,
        vram_int4_gb=5.5,
        recommended_quant="int8",
        context_length=8192,
        license="gemma",
        provider="Google",
        tags=["reasoning"],
    ),
    
    # =========================================================================
    # CODE MODELS - Specialized for programming
    # =========================================================================
    ModelSpec(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        name="Qwen2.5 Coder 7B",
        description="Specialized code generation and understanding model",
        parameters="7B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.CODE, ModelCategory.INSTRUCT],
        size=ModelSize.MEDIUM,
        vram_fp16_gb=14.0,
        vram_int8_gb=8.0,
        vram_int4_gb=4.5,
        recommended_quant="int8",
        context_length=32768,
        license="apache-2.0",
        provider="Qwen",
        tags=["code", "programming", "recommended"],
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        name="Qwen2.5 Coder 1.5B",
        description="Compact code model for fast completions",
        parameters="1.5B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.CODE],
        size=ModelSize.SMALL,
        vram_fp16_gb=3.0,
        vram_int8_gb=1.8,
        vram_int4_gb=1.0,
        recommended_quant="int8",
        context_length=32768,
        license="apache-2.0",
        provider="Qwen",
        tags=["code", "fast"],
    ),
    ModelSpec(
        model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        name="DeepSeek Coder 6.7B",
        description="Strong code model with excellent completion quality",
        parameters="6.7B",
        architecture="LlamaForCausalLM",
        categories=[ModelCategory.CODE],
        size=ModelSize.MEDIUM,
        vram_fp16_gb=13.4,
        vram_int8_gb=7.5,
        vram_int4_gb=4.2,
        recommended_quant="int8",
        context_length=16384,
        zse_optimized=True,  # Verified .zse conversion 2024-02
        license="deepseek",
        provider="DeepSeek",
        tags=["code", "fill-in-middle"],
    ),
    
    # =========================================================================
    # LARGE MODELS (8-13B, 16-26GB VRAM) - High quality
    # =========================================================================
    ModelSpec(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        name="Llama 3.1 8B Instruct",
        description="Meta's flagship 8B model with 128K context",
        parameters="8B",
        architecture="LlamaForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT, ModelCategory.CODE],
        size=ModelSize.MEDIUM,
        vram_fp16_gb=16.0,
        vram_int8_gb=9.0,
        vram_int4_gb=5.0,
        recommended_quant="int8",
        context_length=131072,
        license="llama3.1",
        provider="Meta",
        tags=["long-context", "recommended", "production"],
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-14B-Instruct",
        name="Qwen2.5 14B Instruct",
        description="High-capability model balancing quality and efficiency",
        parameters="14B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT, ModelCategory.CODE, ModelCategory.REASONING],
        size=ModelSize.LARGE,
        vram_fp16_gb=28.0,
        vram_int8_gb=16.0,
        vram_int4_gb=9.0,
        recommended_quant="int8",
        context_length=32768,
        zse_optimized=True,  # Verified .zse conversion 2024-02
        license="apache-2.0",
        provider="Qwen",
        tags=["multilingual", "production"],
    ),
    
    # =========================================================================
    # XLARGE MODELS (13-34B, 26-70GB VRAM) - Maximum quality
    # =========================================================================
    ModelSpec(
        model_id="Qwen/Qwen2.5-32B-Instruct",
        name="Qwen2.5 32B Instruct",
        description="Near-frontier quality with strong reasoning",
        parameters="32B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT, ModelCategory.CODE, ModelCategory.REASONING],
        size=ModelSize.XLARGE,
        vram_fp16_gb=64.0,
        vram_int8_gb=36.0,
        vram_int4_gb=20.0,
        recommended_quant="int4",
        context_length=32768,
        zse_optimized=True,  # Verified .zse conversion 2024-02
        license="apache-2.0",
        provider="Qwen",
        tags=["multilingual", "reasoning", "high-quality"],
    ),
    ModelSpec(
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        name="Mixtral 8x7B Instruct",
        description="Mixture-of-Experts model with excellent efficiency",
        parameters="47B (12B active)",
        architecture="MixtralForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT, ModelCategory.CODE],
        size=ModelSize.XLARGE,
        vram_fp16_gb=94.0,
        vram_int8_gb=52.0,
        vram_int4_gb=28.0,
        recommended_quant="int4",
        context_length=32768,
        license="apache-2.0",
        provider="Mistral AI",
        tags=["moe", "efficient"],
    ),
    ModelSpec(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        name="Llama 3.1 70B Instruct",
        description="Meta's flagship large model with exceptional quality",
        parameters="70B",
        architecture="LlamaForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT, ModelCategory.CODE, ModelCategory.REASONING],
        size=ModelSize.XXL,
        vram_fp16_gb=140.0,
        vram_int8_gb=78.0,
        vram_int4_gb=42.0,
        recommended_quant="int4",
        context_length=131072,
        license="llama3.1",
        provider="Meta",
        tags=["long-context", "high-quality", "frontier"],
    ),
    ModelSpec(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        name="Qwen2.5 72B Instruct",
        description="Frontier-class model with exceptional multilingual ability",
        parameters="72B",
        architecture="Qwen2ForCausalLM",
        categories=[ModelCategory.INSTRUCT, ModelCategory.CHAT, ModelCategory.CODE, ModelCategory.REASONING],
        size=ModelSize.XXL,
        vram_fp16_gb=144.0,
        vram_int8_gb=80.0,
        vram_int4_gb=44.0,
        recommended_quant="int4",
        context_length=32768,
        license="apache-2.0",
        provider="Qwen",
        tags=["multilingual", "high-quality", "frontier"],
    ),
]


class ModelRegistry:
    """Model registry for discovering and selecting models."""
    
    def __init__(self):
        self._models: Dict[str, ModelSpec] = {m.model_id: m for m in REGISTRY}
    
    def list_all(self) -> List[ModelSpec]:
        """List all registered models."""
        return list(self._models.values())
    
    def get(self, model_id: str) -> Optional[ModelSpec]:
        """Get a specific model by ID."""
        return self._models.get(model_id)
    
    def search(self, query: str) -> List[ModelSpec]:
        """Search models by name, description, or tags."""
        query = query.lower()
        results = []
        for model in self._models.values():
            if (query in model.model_id.lower() or
                query in model.name.lower() or
                query in model.description.lower() or
                any(query in tag for tag in model.tags)):
                results.append(model)
        return results
    
    def filter_by_category(self, category: ModelCategory) -> List[ModelSpec]:
        """Filter models by category."""
        return [m for m in self._models.values() if category in m.categories]
    
    def filter_by_size(self, size: ModelSize) -> List[ModelSpec]:
        """Filter models by size tier."""
        return [m for m in self._models.values() if m.size == size]
    
    def filter_by_vram(self, max_vram_gb: float, quantization: str = "int8") -> List[ModelSpec]:
        """Filter models that fit in given VRAM."""
        results = []
        for model in self._models.values():
            if quantization == "fp16" and model.vram_fp16_gb <= max_vram_gb:
                results.append(model)
            elif quantization == "int8" and model.vram_int8_gb <= max_vram_gb:
                results.append(model)
            elif quantization == "int4" and model.vram_int4_gb <= max_vram_gb:
                results.append(model)
        return results
    
    def get_recommended(self, max_vram_gb: Optional[float] = None) -> List[ModelSpec]:
        """Get recommended models, optionally filtered by VRAM."""
        recommended = [m for m in self._models.values() if "recommended" in m.tags]
        if max_vram_gb:
            recommended = [m for m in recommended if m.vram_int8_gb <= max_vram_gb]
        return recommended
    
    def estimate_vram(self, model_id: str, quantization: str = "int8") -> Optional[float]:
        """Estimate VRAM for a model."""
        model = self.get(model_id)
        if not model:
            return None
        if quantization == "fp16":
            return model.vram_fp16_gb
        elif quantization == "int8":
            return model.vram_int8_gb
        elif quantization == "int4":
            return model.vram_int4_gb
        return model.vram_int8_gb


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
