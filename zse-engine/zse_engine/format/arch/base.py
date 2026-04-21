"""Base architecture adapter — interface for mapping HF models to ZSE format.

Each architecture adapter knows:
1. How to read its HF config.json into a ModelConfig
2. How to map HF tensor names to canonical ZSE names
3. Which tensors should be quantized vs kept in fp16
"""

from typing import Dict, List, Optional, Tuple

from zse_engine.format.config import ModelConfig
from zse_engine.format.spec import should_quantize


class ArchAdapter:
    """Base class for architecture-specific adapters."""

    ARCH_NAME: str = "unknown"

    # Mapping from HF tensor name patterns to canonical names.
    # Subclasses override this.
    TENSOR_NAME_MAP: Dict[str, str] = {}

    def config_from_hf(self, hf_config: dict) -> ModelConfig:
        """Convert a HuggingFace config.json dict to ModelConfig.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def map_tensor_name(self, hf_name: str) -> str:
        """Map a HuggingFace tensor name to canonical ZSE name.

        Default: strip 'model.' prefix. Subclasses can override for
        more complex mappings.
        """
        name = hf_name
        if name.startswith("model."):
            name = name[6:]  # strip "model." prefix
        return name

    def should_quantize(self, tensor_name: str) -> bool:
        """Whether this tensor should be INT4 quantized."""
        return should_quantize(tensor_name)

    def tensor_load_order(self, tensor_names: List[str]) -> List[str]:
        """Return tensors in optimal load order.

        Default: sorted by name (layers grouped together).
        Override for architecture-specific ordering.
        """
        return sorted(tensor_names)

    def validate_config(self, config: ModelConfig) -> List[str]:
        """Validate a ModelConfig. Returns list of warnings (empty = OK)."""
        warnings = []
        if config.hidden_size % config.num_heads != 0:
            warnings.append(
                f"hidden_size ({config.hidden_size}) not divisible by "
                f"num_heads ({config.num_heads})"
            )
        if config.num_heads % config.num_kv_heads != 0:
            warnings.append(
                f"num_heads ({config.num_heads}) not divisible by "
                f"num_kv_heads ({config.num_kv_heads})"
            )
        return warnings


# Registry of known architectures
_REGISTRY: Dict[str, type] = {}


def register_adapter(cls: type) -> type:
    """Decorator to register an architecture adapter."""
    _REGISTRY[cls.ARCH_NAME] = cls
    return cls


def get_adapter(arch_name: str) -> ArchAdapter:
    """Get an adapter by architecture name."""
    cls = _REGISTRY.get(arch_name)
    if cls is None:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown architecture '{arch_name}'. Supported: {supported}"
        )
    return cls()


def detect_architecture(hf_config: dict) -> str:
    """Detect architecture from HF config.json.

    Looks at 'architectures' and 'model_type' fields.
    """
    # Check architectures list
    archs = hf_config.get("architectures", [])
    for arch in archs:
        arch_lower = arch.lower()
        if "llama" in arch_lower:
            return "llama"
        if "mistral" in arch_lower:
            return "mistral"
        if "qwen2" in arch_lower or "qwen" in arch_lower:
            return "qwen2"
        if "phi" in arch_lower:
            return "phi3"
        if "gemma2" in arch_lower:
            return "gemma2"
        if "gemma" in arch_lower:
            return "gemma2"

    # Fallback to model_type
    model_type = hf_config.get("model_type", "").lower()
    if model_type in _REGISTRY:
        return model_type

    raise ValueError(
        f"Cannot detect architecture from config. "
        f"architectures={archs}, model_type={hf_config.get('model_type')}"
    )
