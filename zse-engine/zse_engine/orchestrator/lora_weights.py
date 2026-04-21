"""ZSE LoRA Weights — GPU storage for LoRA adapter weight matrices.

Each LoRA adapter adds a low-rank delta to target weight matrices:
    out = W @ x + α * (B @ (A @ x))

Where:
    A: [rank, in_features] — down-projection (compresses input)
    B: [out_features, rank] — up-projection (expands to output)
    rank: typically 8-64 (much smaller than in_features/out_features)
    α: scaling factor (usually rank or rank/2)

Storage: A and B are stored as fp16 on GPU, indexed by layer name.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

from zse_compiler.types.tensor import Tensor
from zse_compiler.types.dtypes import float16


@dataclass
class LoRAWeight:
    """A single LoRA weight pair (A, B) for one target module in one layer."""
    layer_name: str         # e.g., "model.layers.0.self_attn.q_proj"
    rank: int               # r
    in_features: int        # K (e.g., 4096)
    out_features: int       # N (e.g., 4096)

    # GPU tensors (fp16)
    a_ptr: int = 0          # GPU pointer to A [rank, in_features]
    a_nbytes: int = 0
    b_ptr: int = 0          # GPU pointer to B [out_features, rank]
    b_nbytes: int = 0

    @property
    def total_gpu_bytes(self) -> int:
        return self.a_nbytes + self.b_nbytes

    @property
    def a_shape(self) -> Tuple[int, int]:
        return (self.rank, self.in_features)

    @property
    def b_shape(self) -> Tuple[int, int]:
        return (self.out_features, self.rank)

    def make_a_tensor(self) -> Tensor:
        """Create Tensor wrapper for A matrix."""
        t = Tensor(shape=self.a_shape, dtype=float16)
        t._data_ptr = self.a_ptr
        t._nbytes = self.a_nbytes
        return t

    def make_b_tensor(self) -> Tensor:
        """Create Tensor wrapper for B matrix."""
        t = Tensor(shape=self.b_shape, dtype=float16)
        t._data_ptr = self.b_ptr
        t._nbytes = self.b_nbytes
        return t


@dataclass
class LoRAAdapter:
    """A complete LoRA adapter — all weight pairs for all target modules across all layers.

    Example for rank=16 on a 32-layer model with q_proj + v_proj targets:
        64 LoRAWeight entries (32 layers × 2 targets)
        ~64 × 2 × 16 × 4096 × 2 bytes = ~16MB GPU memory (tiny!)
    """
    adapter_id: str
    rank: int                                     # r (8, 16, 32, 64)
    alpha: float                                  # Scaling factor
    target_modules: List[str] = field(default_factory=list)  # ["q_proj", "v_proj", ...]
    num_layers: int = 0                           # Model layers

    # Indexed: (layer_idx, module_name) → LoRAWeight
    weights: Dict[Tuple[int, str], LoRAWeight] = field(default_factory=dict)

    # Metadata
    base_model: str = ""                          # Which base model this adapts
    description: str = ""

    @property
    def scaling(self) -> float:
        """LoRA scaling factor: α / rank."""
        return self.alpha / self.rank if self.rank > 0 else 0.0

    @property
    def total_gpu_bytes(self) -> int:
        return sum(w.total_gpu_bytes for w in self.weights.values())

    @property
    def num_weight_pairs(self) -> int:
        return len(self.weights)

    def get_weight(self, layer_idx: int, module_name: str) -> Optional[LoRAWeight]:
        """Get LoRA weight for a specific layer and module.

        Args:
            layer_idx: Transformer layer index (0-based)
            module_name: Short module name (e.g., "q_proj", "v_proj")

        Returns:
            LoRAWeight or None if this module doesn't have a LoRA adapter
        """
        return self.weights.get((layer_idx, module_name))

    def has_weight(self, layer_idx: int, module_name: str) -> bool:
        return (layer_idx, module_name) in self.weights

    def add_weight(self, layer_idx: int, module_name: str, weight: LoRAWeight):
        self.weights[(layer_idx, module_name)] = weight

    def summary(self) -> str:
        return (
            f"LoRA Adapter '{self.adapter_id}':\n"
            f"  Rank: {self.rank}, Alpha: {self.alpha}, Scaling: {self.scaling:.4f}\n"
            f"  Targets: {self.target_modules}\n"
            f"  Layers: {self.num_layers}\n"
            f"  Weight pairs: {self.num_weight_pairs}\n"
            f"  GPU memory: {self.total_gpu_bytes / 1024**2:.1f}MB"
        )


class LoRAWeightStore:
    """Collection of loaded LoRA adapters.

    Manages GPU memory for all adapters. Supports hot-swap (load/unload
    without affecting base model or other adapters).
    """

    def __init__(self):
        self._adapters: Dict[str, LoRAAdapter] = {}

    def add(self, adapter: LoRAAdapter):
        self._adapters[adapter.adapter_id] = adapter

    def get(self, adapter_id: str) -> Optional[LoRAAdapter]:
        return self._adapters.get(adapter_id)

    def remove(self, adapter_id: str) -> Optional[LoRAAdapter]:
        return self._adapters.pop(adapter_id, None)

    def has(self, adapter_id: str) -> bool:
        return adapter_id in self._adapters

    @property
    def num_adapters(self) -> int:
        return len(self._adapters)

    @property
    def total_gpu_bytes(self) -> int:
        return sum(a.total_gpu_bytes for a in self._adapters.values())

    def list_adapters(self) -> List[str]:
        return list(self._adapters.keys())

    def summary(self) -> str:
        lines = [
            f"LoRA Store: {self.num_adapters} adapters, "
            f"{self.total_gpu_bytes / 1024**2:.1f}MB total GPU"
        ]
        for adapter in self._adapters.values():
            lines.append(f"  - {adapter.adapter_id}: rank={adapter.rank}, "
                        f"{adapter.total_gpu_bytes / 1024**2:.1f}MB")
        return '\n'.join(lines)
