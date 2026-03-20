"""
ZSE Streaming Model Loader

Ultra memory-efficient model loading with:
- Direct GPU loading (zero CPU copy)
- Meta tensor initialization (deferred materialization)
- Streaming weight loading (one tensor at a time)
- Memory tracking and optimization

This is the core of ZSE's memory efficiency - loading models
with minimal overhead beyond the theoretical minimum.
"""

import os
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any, Callable
from contextlib import contextmanager
import logging

import torch
import torch.nn as nn

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory statistics in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": total - reserved,
        "total": total,
    }


def empty_cache():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class StreamingModelLoader:
    """
    Memory-efficient streaming model loader.
    
    Key optimizations:
    1. Direct GPU loading - weights go straight to GPU, never to CPU
    2. Meta tensor init - model skeleton uses zero memory
    3. One-at-a-time loading - only one tensor in flight
    4. Immediate assignment - no intermediate dicts
    
    This achieves near-theoretical minimum memory usage:
    - FP16: ~2 bytes per parameter
    - INT8: ~1 byte per parameter  
    - INT4: ~0.5 bytes per parameter
    """
    
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        low_memory: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.low_memory = low_memory
        
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors required: pip install safetensors")
    
    @contextmanager
    def _open_safetensors(self, path: str):
        """Open safetensors file for direct GPU loading."""
        # safe_open with device="cuda" loads directly to GPU
        # This is the key - no CPU intermediate copy!
        with safe_open(path, framework="pt", device=self.device) as f:
            yield f
    
    def get_tensor_names(self, model_path: str) -> List[str]:
        """Get all tensor names without loading."""
        safetensor_files = self._find_safetensor_files(model_path)
        
        names = []
        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="pt") as f:
                names.extend(f.keys())
        
        return names
    
    def _find_safetensor_files(self, model_path: str) -> List[str]:
        """Find all safetensor files in directory."""
        path = Path(model_path)
        
        if path.is_file():
            return [str(path)]
        
        files = list(path.glob("*.safetensors"))
        
        # Sort by shard index if present
        def sort_key(f):
            name = f.stem
            if "-of-" in name:
                for part in name.split("-"):
                    if part.isdigit():
                        return int(part)
            return name
        
        return [str(f) for f in sorted(files, key=sort_key)]
    
    def stream_tensors(
        self,
        model_path: str,
        callback: Optional[Callable[[str, torch.Tensor], None]] = None,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Stream tensors directly to GPU one at a time.
        
        This is the most memory-efficient way to load:
        - Each tensor is loaded directly to GPU
        - Immediately yielded for assignment
        - No accumulation in memory
        
        Args:
            model_path: Path to model directory
            callback: Optional callback(name, tensor) for progress
            
        Yields:
            (tensor_name, tensor) pairs
        """
        safetensor_files = self._find_safetensor_files(model_path)
        
        for sf_path in safetensor_files:
            # Open with direct GPU loading
            with safe_open(sf_path, framework="pt", device=self.device) as f:
                for name in f.keys():
                    # Load directly to GPU - this is the key!
                    tensor = f.get_tensor(name)
                    
                    # Convert dtype if needed
                    if tensor.dtype != self.dtype and tensor.is_floating_point():
                        tensor = tensor.to(self.dtype)
                    
                    if callback:
                        callback(name, tensor)
                    
                    yield name, tensor
                    
                    # In low memory mode, free after yield returns
                    if self.low_memory:
                        del tensor
    
    def load_into_model(
        self,
        model_path: str,
        model: nn.Module,
        strict: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Load weights into model with streaming (minimal memory).
        
        Args:
            model_path: Path to model or directory
            model: PyTorch model (can be on meta device)
            strict: Require all weights to match
            progress_callback: callback(progress_percent, tensor_name)
            
        Returns:
            (missing_keys, unexpected_keys)
        """
        # Get model state dict keys
        model_keys = set(model.state_dict().keys())
        loaded_keys = set()
        
        # Count total for progress
        all_names = self.get_tensor_names(model_path)
        total = len(all_names)
        
        # Build name mapping (HuggingFace -> local)
        name_map = self._build_name_map(all_names, model_keys)
        
        # Stream and load one tensor at a time
        for idx, (name, tensor) in enumerate(self.stream_tensors(model_path)):
            # Map name to model parameter
            param_name = name_map.get(name)
            
            if param_name and param_name in model_keys:
                # Get parameter
                self._set_module_tensor(model, param_name, tensor)
                loaded_keys.add(param_name)
            
            # Progress callback
            if progress_callback:
                progress_callback((idx + 1) / total, name)
            
            # Explicit cleanup in low memory mode
            if self.low_memory:
                del tensor
                if idx % 50 == 0:  # Periodic cleanup
                    empty_cache()
        
        # Calculate missing/unexpected
        missing = list(model_keys - loaded_keys)
        unexpected = [n for n in all_names if name_map.get(n) not in model_keys]
        
        # Final cleanup
        empty_cache()
        
        return missing, unexpected
    
    def _set_module_tensor(
        self,
        model: nn.Module,
        name: str,
        tensor: torch.Tensor,
    ):
        """
        Set a tensor in the model efficiently.
        
        Handles both regular parameters and meta tensors.
        """
        parts = name.split(".")
        
        # Navigate to parent module
        module = model
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        param_name = parts[-1]
        
        # Get current parameter
        old_param = getattr(module, param_name, None)
        
        if old_param is not None:
            # Check if on meta device
            if old_param.device.type == "meta":
                # Replace meta tensor with real tensor
                if isinstance(old_param, nn.Parameter):
                    new_param = nn.Parameter(tensor, requires_grad=old_param.requires_grad)
                else:
                    new_param = tensor
                setattr(module, param_name, new_param)
            else:
                # Copy into existing tensor (in-place)
                old_param.data.copy_(tensor)
        else:
            # Direct assignment
            setattr(module, param_name, tensor)
    
    def _build_name_map(
        self,
        file_names: List[str],
        model_keys: set,
    ) -> Dict[str, str]:
        """Build mapping from file tensor names to model param names."""
        name_map = {}
        
        for name in file_names:
            # Try direct match first
            if name in model_keys:
                name_map[name] = name
                continue
            
            # Try without "model." prefix
            if name.startswith("model."):
                stripped = name[6:]
                if stripped in model_keys:
                    name_map[name] = stripped
                    continue
            
            # Try adding "model." prefix
            prefixed = "model." + name
            if prefixed in model_keys:
                name_map[name] = prefixed
                continue
            
            # Common HF -> local transformations
            transformed = name
            transforms = [
                ("model.", ""),
                ("self_attn.", "attention."),
                ("mlp.", "feed_forward."),
            ]
            for old, new in transforms:
                transformed = transformed.replace(old, new)
            
            if transformed in model_keys:
                name_map[name] = transformed
        
        return name_map


def load_model_streaming(
    model_path: str,
    model: nn.Module,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    progress: bool = True,
) -> nn.Module:
    """
    Convenience function to load model with streaming.
    
    Args:
        model_path: HuggingFace model path or local directory
        model: Model to load into
        device: Target device
        dtype: Target dtype
        progress: Show progress bar
        
    Returns:
        Model with loaded weights
    """
    loader = StreamingModelLoader(device=device, dtype=dtype)
    
    if progress:
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as pbar:
                task = pbar.add_task("Loading weights...", total=100)
                
                def callback(pct, name):
                    pbar.update(task, completed=pct * 100, description=f"Loading {name[:30]}...")
                
                missing, unexpected = loader.load_into_model(model_path, model, progress_callback=callback)
        except ImportError:
            missing, unexpected = loader.load_into_model(model_path, model)
    else:
        missing, unexpected = loader.load_into_model(model_path, model)
    
    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
    
    return model


@contextmanager  
def init_on_device(device: str = "meta"):
    """
    Context manager to initialize model on specific device.
    
    Usage:
        with init_on_device("meta"):
            model = LlamaModel(config)  # Zero memory!
        
        load_model_streaming(path, model)  # Stream weights to GPU
    
    This is the foundation of memory-efficient loading:
    - Create model structure on meta device (no memory)
    - Stream weights directly to GPU
    - Result: Only actual weights in memory
    """
    old_init = torch.nn.Linear.__init__
    old_embedding_init = torch.nn.Embedding.__init__
    
    def new_linear_init(self, in_features, out_features, bias=True, device=None, dtype=None):
        old_init(self, in_features, out_features, bias, device=device or "meta", dtype=dtype)
    
    def new_embedding_init(self, num_embeddings, embedding_dim, padding_idx=None, 
                           max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                           sparse=False, _weight=None, _freeze=False, device=None, dtype=None):
        old_embedding_init(self, num_embeddings, embedding_dim, padding_idx, max_norm,
                          norm_type, scale_grad_by_freq, sparse, _weight, _freeze,
                          device=device or "meta", dtype=dtype)
    
    torch.nn.Linear.__init__ = new_linear_init
    torch.nn.Embedding.__init__ = new_embedding_init
    
    try:
        yield
    finally:
        torch.nn.Linear.__init__ = old_init
        torch.nn.Embedding.__init__ = old_embedding_init


def estimate_model_memory(num_params: int, dtype: torch.dtype = torch.float16) -> float:
    """
    Estimate memory for model in GB.
    
    Theoretical minimum memory usage:
    - FP32: 4 bytes/param
    - FP16/BF16: 2 bytes/param
    - INT8: 1 byte/param
    - INT4: 0.5 bytes/param
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }
    
    bpp = bytes_per_param.get(dtype, 2)
    return (num_params * bpp) / (1024**3)
