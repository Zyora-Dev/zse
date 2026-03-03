"""
ZSE LoRA Implementation

Low-Rank Adaptation (LoRA) for efficient fine-tuning of large models.
Injects trainable low-rank matrices into frozen base model layers.

Key concepts:
- Original weight: W (frozen, INT4)
- LoRA: W' = W + BA where B ∈ R^(d×r), A ∈ R^(r×k)
- Only A and B are trained, W stays frozen
- r << min(d, k), so A and B are tiny compared to W

Memory savings:
- Full 7B finetune: 14GB weights + 14GB gradients + 28GB optimizer = 56GB
- QLoRA 7B: 3.5GB INT4 + 0.2GB LoRA + 0.4GB optimizer = ~4GB

Author: ZSE Team
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    
    # Core LoRA parameters
    rank: int = 64
    """Rank of the low-rank decomposition. Higher = more capacity, more VRAM."""
    
    alpha: int = 128
    """Scaling factor. Effective scaling is alpha/rank."""
    
    dropout: float = 0.05
    """Dropout probability for LoRA layers."""
    
    # Target modules
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ])
    """Which modules to apply LoRA to."""
    
    # Advanced options
    bias: str = "none"
    """Bias training: 'none', 'all', or 'lora_only'."""
    
    modules_to_save: List[str] = field(default_factory=list)
    """Additional modules to save (fully trained, not LoRA)."""
    
    # Initialization
    init_lora_weights: bool = True
    """Whether to initialize A with Kaiming and B with zeros."""
    
    # Use RSLoRA scaling (rank-stabilized)
    use_rslora: bool = False
    """Use rank-stabilized LoRA scaling (alpha/sqrt(rank) instead of alpha/rank)."""
    
    def __post_init__(self):
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {self.dropout}")
    
    @property
    def scaling(self) -> float:
        """Get the LoRA scaling factor."""
        if self.use_rslora:
            return self.alpha / math.sqrt(self.rank)
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Computes: y = Wx + (BA)x * scaling
    Where W is frozen (possibly INT4), and B, A are trainable FP16.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        use_rslora: bool = False,
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.use_rslora = use_rslora
        
        # Get dimensions from base layer
        if hasattr(base_layer, 'in_features'):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        elif hasattr(base_layer, 'weight'):
            self.out_features, self.in_features = base_layer.weight.shape
        else:
            raise ValueError("Cannot determine dimensions of base layer")
        
        # Scaling factor
        if use_rslora:
            self.scaling = alpha / math.sqrt(rank)
        else:
            self.scaling = alpha / rank
        
        # Get device and dtype from base layer
        if hasattr(base_layer, 'weight') and base_layer.weight is not None:
            device = base_layer.weight.device
            dtype = base_layer.weight.dtype
        else:
            param = next(base_layer.parameters())
            device = param.device
            dtype = param.dtype
        
        # Use float32 for training stability if base is quantized
        if dtype in (torch.int8, torch.uint8, torch.int4, torch.uint4):
            dtype = torch.float16
        
        # LoRA matrices (same dtype and device as base)
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_features, dtype=dtype, device=device)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, dtype=dtype, device=device)
        )
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self.reset_lora_parameters()
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def reset_lora_parameters(self):
        """Initialize LoRA weights."""
        # Kaiming uniform for A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Zero for B (so initial LoRA contribution is zero)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Base layer output (frozen, possibly INT4)
        base_output = self.base_layer(x)
        
        # LoRA output: (B @ A) @ x * scaling
        # Compute efficiently: B @ (A @ x) to avoid large intermediate
        x_lora = x.to(self.lora_A.dtype)
        lora_output = self.lora_dropout(x_lora)
        lora_output = F.linear(lora_output, self.lora_A)  # [batch, seq, rank]
        lora_output = F.linear(lora_output, self.lora_B)  # [batch, seq, out]
        lora_output = lora_output * self.scaling
        
        # Combine
        return base_output + lora_output.to(base_output.dtype)
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into base layer (for inference)."""
        if not hasattr(self.base_layer, 'weight'):
            raise ValueError("Cannot merge into layer without weight attribute")
        
        # Compute BA and add to base weight
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data += delta_w.to(self.base_layer.weight.dtype)
    
    def unmerge_weights(self) -> None:
        """Remove merged LoRA weights from base layer."""
        if not hasattr(self.base_layer, 'weight'):
            raise ValueError("Cannot unmerge from layer without weight attribute")
        
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data -= delta_w.to(self.base_layer.weight.dtype)


def _find_target_modules(
    model: nn.Module,
    target_names: List[str],
) -> Dict[str, nn.Module]:
    """Find all modules matching target names."""
    targets = {}
    
    for name, module in model.named_modules():
        # Check if module name ends with any target
        for target in target_names:
            if name.endswith(target):
                targets[name] = module
                break
    
    return targets


def _replace_with_lora(
    model: nn.Module,
    target_name: str,
    lora_layer: LoRALinear,
) -> None:
    """Replace a module with its LoRA version."""
    # Split name into parent path and attribute name
    parts = target_name.rsplit('.', 1)
    if len(parts) == 1:
        parent = model
        attr_name = parts[0]
    else:
        parent_name, attr_name = parts
        parent = model.get_submodule(parent_name)
    
    setattr(parent, attr_name, lora_layer)


def add_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """
    Add LoRA adapters to a model.
    
    Args:
        model: The base model (can be quantized)
        config: LoRA configuration
        
    Returns:
        Model with LoRA adapters added
    """
    # Find target modules
    targets = _find_target_modules(model, config.target_modules)
    
    if not targets:
        raise ValueError(
            f"No modules found matching targets: {config.target_modules}. "
            f"Available modules: {[n for n, _ in model.named_modules()][:20]}..."
        )
    
    # Replace with LoRA versions
    replaced = 0
    for name, module in targets.items():
        # Skip if not a linear layer
        if not isinstance(module, nn.Linear) and not hasattr(module, 'forward'):
            continue
        
        lora_layer = LoRALinear(
            base_layer=module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            use_rslora=config.use_rslora,
        )
        
        _replace_with_lora(model, name, lora_layer)
        replaced += 1
    
    print(f"Added LoRA adapters to {replaced} layers (rank={config.rank}, alpha={config.alpha})")
    
    # Store config on model for later
    model._lora_config = config
    
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only the LoRA parameters from a model.
    
    Returns a state dict containing only lora_A and lora_B weights.
    """
    lora_state = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.clone()
    
    return lora_state


def set_lora_trainable(model: nn.Module, trainable: bool = True) -> None:
    """
    Set LoRA parameters as trainable/frozen.
    
    Also ensures base model stays frozen.
    """
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = trainable
        else:
            param.requires_grad = False
    
    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into base layers.
    
    After merging, the model behaves as if LoRA was never added,
    but with the adapted weights. Good for inference.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    
    return model


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count LoRA and total parameters.
    
    Returns:
        (lora_params, total_params)
    """
    lora_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'lora_A' in name or 'lora_B' in name:
            lora_params += param.numel()
    
    return lora_params, total_params
