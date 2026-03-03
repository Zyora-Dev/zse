"""
ZSE Adapter I/O - Save and Load LoRA Adapters

Handles saving and loading LoRA adapter weights in various formats:
- SafeTensors (recommended, fast and safe)
- PyTorch state dict (fallback)

Author: ZSE Team
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import asdict

import torch
import torch.nn as nn

from .lora import LoRAConfig, LoRALinear, get_lora_state_dict


def save_lora_adapter(
    model: nn.Module,
    save_path: Union[str, Path],
    config: Optional[LoRAConfig] = None,
) -> str:
    """
    Save LoRA adapter weights to a file.
    
    Args:
        model: Model with LoRA adapters
        save_path: Path to save the adapter (will add .safetensors if no extension)
        config: LoRA config (if not attached to model)
        
    Returns:
        Path to saved adapter file
    """
    save_path = Path(save_path)
    
    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add extension if needed
    if save_path.suffix not in ['.safetensors', '.pt', '.bin']:
        save_path = save_path.with_suffix('.safetensors')
    
    # Get LoRA weights
    lora_state = get_lora_state_dict(model)
    
    if not lora_state:
        raise ValueError("No LoRA weights found in model. Did you call add_lora_to_model()?")
    
    # Save weights
    if save_path.suffix == '.safetensors':
        try:
            from safetensors.torch import save_file
            save_file(lora_state, str(save_path))
        except ImportError:
            print("Warning: safetensors not installed, falling back to PyTorch format")
            save_path = save_path.with_suffix('.pt')
            torch.save(lora_state, str(save_path))
    else:
        torch.save(lora_state, str(save_path))
    
    # Save config alongside
    lora_config = config or getattr(model, '_lora_config', None)
    if lora_config:
        config_path = save_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(lora_config), f, indent=2)
    
    # Print summary
    num_params = sum(t.numel() for t in lora_state.values())
    size_mb = sum(t.numel() * t.element_size() for t in lora_state.values()) / 1024 / 1024
    
    print(f"Saved LoRA adapter:")
    print(f"  Path: {save_path}")
    print(f"  Layers: {len(lora_state) // 2}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Size: {size_mb:.2f} MB")
    
    return str(save_path)


def load_lora_adapter(
    model: nn.Module,
    adapter_path: Union[str, Path],
    strict: bool = True,
) -> nn.Module:
    """
    Load LoRA adapter weights into a model.
    
    The model must already have LoRA layers added.
    
    Args:
        model: Model with LoRA adapters
        adapter_path: Path to adapter file
        strict: If True, raise error on missing/unexpected keys
        
    Returns:
        Model with loaded adapter weights
    """
    adapter_path = Path(adapter_path)
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    # Load weights
    if adapter_path.suffix == '.safetensors':
        try:
            from safetensors.torch import load_file
            lora_state = load_file(str(adapter_path))
        except ImportError:
            raise ImportError("safetensors required to load .safetensors files: pip install safetensors")
    else:
        lora_state = torch.load(str(adapter_path), map_location='cpu', weights_only=True)
    
    # Find LoRA modules in model
    lora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules[name] = module
    
    if not lora_modules:
        raise ValueError("No LoRA layers found in model. Did you call add_lora_to_model()?")
    
    # Load weights into modules
    loaded = 0
    missing = []
    unexpected = []
    
    for key, tensor in lora_state.items():
        # Parse key: "module.name.lora_A" -> module_name, param_name
        parts = key.rsplit('.', 1)
        if len(parts) != 2:
            unexpected.append(key)
            continue
        
        module_name, param_name = parts
        
        if module_name not in lora_modules:
            unexpected.append(key)
            continue
        
        module = lora_modules[module_name]
        
        if param_name == 'lora_A':
            module.lora_A.data.copy_(tensor.to(module.lora_A.device))
            loaded += 1
        elif param_name == 'lora_B':
            module.lora_B.data.copy_(tensor.to(module.lora_B.device))
            loaded += 1
        else:
            unexpected.append(key)
    
    # Check for missing
    for name, module in lora_modules.items():
        if f"{name}.lora_A" not in lora_state:
            missing.append(f"{name}.lora_A")
        if f"{name}.lora_B" not in lora_state:
            missing.append(f"{name}.lora_B")
    
    if strict:
        if missing:
            raise KeyError(f"Missing keys in adapter: {missing}")
        if unexpected:
            raise KeyError(f"Unexpected keys in adapter: {unexpected}")
    
    print(f"Loaded LoRA adapter:")
    print(f"  Path: {adapter_path}")
    print(f"  Loaded parameters: {loaded}")
    if missing:
        print(f"  Warning - Missing: {len(missing)} keys")
    if unexpected:
        print(f"  Warning - Unexpected: {len(unexpected)} keys")
    
    return model


def get_adapter_info(adapter_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a saved LoRA adapter.
    
    Args:
        adapter_path: Path to adapter file
        
    Returns:
        Dictionary with adapter info
    """
    adapter_path = Path(adapter_path)
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    # Load weights
    if adapter_path.suffix == '.safetensors':
        from safetensors.torch import load_file
        lora_state = load_file(str(adapter_path))
    else:
        lora_state = torch.load(str(adapter_path), map_location='cpu', weights_only=True)
    
    # Count layers and params
    num_layers = len(lora_state) // 2
    num_params = sum(t.numel() for t in lora_state.values())
    size_bytes = sum(t.numel() * t.element_size() for t in lora_state.values())
    
    # Try to determine rank from shapes
    rank = None
    for key, tensor in lora_state.items():
        if 'lora_A' in key:
            rank = tensor.shape[0]
            break
    
    # Load config if exists
    config_path = adapter_path.with_suffix('.json')
    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    return {
        'path': str(adapter_path),
        'num_layers': num_layers,
        'num_parameters': num_params,
        'size_bytes': size_bytes,
        'size_mb': size_bytes / 1024 / 1024,
        'rank': rank,
        'config': config,
    }


def merge_and_save(
    model: nn.Module,
    save_path: Union[str, Path],
    adapter_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Merge LoRA weights into base model and save as regular model.
    
    Warning: This creates a full-size model file (not just adapter).
    
    Args:
        model: Model with LoRA adapters
        save_path: Path to save merged model
        adapter_path: Optional adapter to load before merging
        
    Returns:
        Path to saved model
    """
    from .lora import merge_lora_weights
    
    # Load adapter if provided
    if adapter_path:
        load_lora_adapter(model, adapter_path)
    
    # Merge LoRA into base weights
    merge_lora_weights(model)
    
    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get state dict (excluding LoRA params which are now merged)
    state_dict = {
        k: v for k, v in model.state_dict().items()
        if 'lora_A' not in k and 'lora_B' not in k
    }
    
    if save_path.suffix == '.safetensors':
        from safetensors.torch import save_file
        save_file(state_dict, str(save_path))
    else:
        torch.save(state_dict, str(save_path))
    
    print(f"Saved merged model to: {save_path}")
    return str(save_path)
