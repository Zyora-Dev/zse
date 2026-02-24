"""
zStream Offload Manager

Manages tiered storage for model layers:
- GPU (Hot): Active layers for compute
- CPU (Warm): Pinned memory for fast transfer
- Disk (Cold): Memory-mapped for huge models

Key Innovation: Unlike GGUF which requires ALL weights in RAM,
zStream can memory-map weights from disk and stream on-demand.
This enables running 70B+ models on machines with limited RAM.
"""

import torch
import torch.nn as nn
import os
import mmap
import threading
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pickle
import struct
import tempfile
import shutil


class StorageTier(Enum):
    """Storage tier for layers."""
    GPU = "gpu"
    CPU = "cpu"
    CPU_PINNED = "cpu_pinned"
    DISK = "disk"
    DISK_MMAP = "disk_mmap"


@dataclass
class TierConfig:
    """Configuration for a storage tier."""
    tier: StorageTier
    max_size_bytes: int
    path: Optional[str] = None  # For disk tiers


@dataclass
class LayerData:
    """Data for a layer in a specific tier."""
    layer_idx: int
    tier: StorageTier
    
    # Tensors (when in memory)
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    
    # Disk info (when on disk)
    disk_path: Optional[str] = None
    disk_offset: int = 0
    disk_size: int = 0
    
    # Memory map (when using mmap)
    mmap_obj: Optional[mmap.mmap] = None


class OffloadManager:
    """
    Manages layer offloading across storage tiers.
    
    Tier Hierarchy:
        GPU (fastest, limited) 
            ↓ offload
        CPU Pinned (fast, larger)
            ↓ offload
        CPU Regular (medium, larger)
            ↓ offload
        Disk (slow, unlimited)
    
    Features:
    - Automatic tier management based on access patterns
    - Memory-mapped disk access for huge models
    - Pinned memory for zero-copy GPU transfers
    - Async operations for hiding latency
    """
    
    def __init__(
        self,
        gpu_budget_bytes: int,
        cpu_budget_bytes: int,
        disk_path: Optional[str] = None,
        use_pinned_memory: bool = True,
        device: int = 0,
    ):
        self.gpu_budget = gpu_budget_bytes
        self.cpu_budget = cpu_budget_bytes
        self.disk_path = disk_path or tempfile.mkdtemp(prefix="zstream_")
        self.use_pinned_memory = use_pinned_memory
        self.device = device
        
        # Layer tracking
        self._layers: Dict[int, LayerData] = {}
        self._lock = threading.RLock()
        
        # Tier usage tracking
        self._tier_usage = {
            StorageTier.GPU: 0,
            StorageTier.CPU: 0,
            StorageTier.CPU_PINNED: 0,
            StorageTier.DISK: 0,
        }
        
        # Disk management
        self._disk_file: Optional[Any] = None
        self._disk_offset = 0
        self._setup_disk()
        
        # Stats
        self.stats = {
            "gpu_offloads": 0,
            "cpu_offloads": 0,
            "disk_offloads": 0,
            "gpu_loads": 0,
            "cpu_loads": 0,
            "disk_loads": 0,
        }
    
    def _setup_disk(self):
        """Setup disk storage."""
        os.makedirs(self.disk_path, exist_ok=True)
    
    def register_layer(
        self,
        layer_idx: int,
        module: nn.Module,
        initial_tier: StorageTier = StorageTier.CPU,
    ):
        """
        Register a layer with the offload manager.
        
        Args:
            layer_idx: Layer index
            module: The layer module
            initial_tier: Where to initially store the layer
        """
        with self._lock:
            # Get state dict
            state_dict = {
                k: v.clone() for k, v in module.state_dict().items()
            }
            
            # Calculate size
            size = sum(
                t.numel() * t.element_size() 
                for t in state_dict.values()
            )
            
            layer_data = LayerData(
                layer_idx=layer_idx,
                tier=initial_tier,
                state_dict=state_dict,
            )
            
            self._layers[layer_idx] = layer_data
            self._tier_usage[initial_tier] += size
    
    def get_layer_tier(self, layer_idx: int) -> StorageTier:
        """Get the current tier of a layer."""
        with self._lock:
            if layer_idx not in self._layers:
                raise KeyError(f"Layer {layer_idx} not registered")
            return self._layers[layer_idx].tier
    
    def move_to_gpu(self, layer_idx: int, module: nn.Module) -> nn.Module:
        """
        Move a layer to GPU.
        
        Args:
            layer_idx: Layer index
            module: The module to load into
            
        Returns:
            Module with weights on GPU
        """
        with self._lock:
            layer_data = self._layers.get(layer_idx)
            if not layer_data:
                raise KeyError(f"Layer {layer_idx} not registered")
            
            current_tier = layer_data.tier
            
            if current_tier == StorageTier.GPU:
                return module  # Already on GPU
            
            # Load state dict from current tier
            if current_tier == StorageTier.DISK or current_tier == StorageTier.DISK_MMAP:
                state_dict = self._load_from_disk(layer_data)
            else:
                state_dict = layer_data.state_dict
            
            if state_dict is None:
                raise ValueError(f"No state dict for layer {layer_idx}")
            
            # Move to GPU
            gpu_state_dict = {
                k: v.to(device=f"cuda:{self.device}", non_blocking=True)
                for k, v in state_dict.items()
            }
            
            module.load_state_dict(gpu_state_dict)
            
            # Update tracking
            size = sum(t.numel() * t.element_size() for t in gpu_state_dict.values())
            self._tier_usage[current_tier] -= size
            self._tier_usage[StorageTier.GPU] += size
            
            layer_data.tier = StorageTier.GPU
            layer_data.state_dict = gpu_state_dict
            
            self.stats["gpu_loads"] += 1
            
            return module
    
    def offload_to_cpu(
        self,
        layer_idx: int,
        module: nn.Module,
        pinned: bool = True,
    ):
        """
        Offload a layer from GPU to CPU.
        
        Args:
            layer_idx: Layer index
            module: The module to offload
            pinned: Use pinned memory for fast transfer back
        """
        with self._lock:
            layer_data = self._layers.get(layer_idx)
            if not layer_data:
                return
            
            if layer_data.tier not in (StorageTier.GPU, StorageTier.CPU_PINNED):
                return  # Not on GPU
            
            # Get state dict from module
            state_dict = {k: v.clone() for k, v in module.state_dict().items()}
            
            # Move to CPU
            target_tier = StorageTier.CPU_PINNED if pinned else StorageTier.CPU
            
            if pinned and self.use_pinned_memory:
                cpu_state_dict = {
                    k: v.cpu().pin_memory() for k, v in state_dict.items()
                }
            else:
                cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            
            # Update tracking
            size = sum(t.numel() * t.element_size() for t in cpu_state_dict.values())
            self._tier_usage[layer_data.tier] -= size
            self._tier_usage[target_tier] += size
            
            layer_data.tier = target_tier
            layer_data.state_dict = cpu_state_dict
            
            # Clear GPU memory on module
            module.cpu()
            
            self.stats["cpu_offloads"] += 1
    
    def offload_to_disk(self, layer_idx: int, module: nn.Module):
        """
        Offload a layer to disk.
        
        Used for extreme memory situations or very large models.
        """
        with self._lock:
            layer_data = self._layers.get(layer_idx)
            if not layer_data:
                return
            
            # Get state dict
            if layer_data.tier == StorageTier.GPU:
                state_dict = {k: v.cpu() for k, v in module.state_dict().items()}
            else:
                state_dict = layer_data.state_dict
            
            if state_dict is None:
                return
            
            # Save to disk
            disk_path = self._save_to_disk(layer_idx, state_dict)
            
            # Update tracking
            size = sum(t.numel() * t.element_size() for t in state_dict.values())
            self._tier_usage[layer_data.tier] -= size
            self._tier_usage[StorageTier.DISK] += size
            
            layer_data.tier = StorageTier.DISK
            layer_data.state_dict = None  # Free memory
            layer_data.disk_path = disk_path
            
            # Clear module memory
            module.cpu()
            
            self.stats["disk_offloads"] += 1
    
    def _save_to_disk(
        self,
        layer_idx: int,
        state_dict: Dict[str, torch.Tensor],
    ) -> str:
        """Save layer state dict to disk."""
        path = os.path.join(self.disk_path, f"layer_{layer_idx}.pt")
        torch.save(state_dict, path)
        return path
    
    def _load_from_disk(self, layer_data: LayerData) -> Dict[str, torch.Tensor]:
        """Load layer state dict from disk."""
        if layer_data.disk_path and os.path.exists(layer_data.disk_path):
            return torch.load(layer_data.disk_path, map_location="cpu")
        raise FileNotFoundError(f"Disk file not found: {layer_data.disk_path}")
    
    def get_tier_usage(self) -> Dict[StorageTier, int]:
        """Get current usage of each tier."""
        return dict(self._tier_usage)
    
    def get_tier_usage_gb(self) -> Dict[str, float]:
        """Get tier usage in GB."""
        return {
            tier.value: usage / (1024**3)
            for tier, usage in self._tier_usage.items()
        }
    
    def should_offload_from_gpu(self) -> bool:
        """Check if we should offload from GPU."""
        return self._tier_usage[StorageTier.GPU] > self.gpu_budget
    
    def should_offload_from_cpu(self) -> bool:
        """Check if we should offload from CPU to disk."""
        cpu_usage = (
            self._tier_usage[StorageTier.CPU] +
            self._tier_usage[StorageTier.CPU_PINNED]
        )
        return cpu_usage > self.cpu_budget
    
    def cleanup(self):
        """Cleanup disk cache."""
        if os.path.exists(self.disk_path):
            shutil.rmtree(self.disk_path, ignore_errors=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get offload statistics."""
        return {
            **self.stats,
            "tier_usage_gb": self.get_tier_usage_gb(),
        }


class MemoryMappedWeights:
    """
    Memory-mapped weights for disk-streaming.
    
    This is the key to running huge models:
    - Weights stay on disk
    - Only load pages that are accessed
    - OS handles caching automatically
    
    Example:
        # 140GB model file
        weights = MemoryMappedWeights("model.safetensors")
        
        # Load only layer 0 (~2GB)
        layer_weights = weights.get_layer(0)
        
        # Total RAM used: ~2GB, not 140GB!
    """
    
    def __init__(
        self,
        path: str,
        layer_offsets: Optional[Dict[int, Tuple[int, int]]] = None,
    ):
        """
        Args:
            path: Path to weights file
            layer_offsets: Dict of layer_idx -> (offset, size)
        """
        self.path = path
        self.layer_offsets = layer_offsets or {}
        
        # Open file for memory mapping
        self._file = open(path, "rb")
        self._mmap = mmap.mmap(
            self._file.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )
        
        # Parse format if offsets not provided
        if not self.layer_offsets:
            self._detect_format()
    
    def _detect_format(self):
        """Detect file format and extract layer offsets."""
        # Check for safetensors header
        header_size = struct.unpack("<Q", self._mmap[:8])[0]
        if header_size < 10 * 1024 * 1024:  # Header should be < 10MB
            # Likely safetensors format
            self._parse_safetensors_header(header_size)
    
    def _parse_safetensors_header(self, header_size: int):
        """Parse safetensors header to find layer offsets."""
        import json
        
        header_bytes = self._mmap[8:8+header_size]
        header = json.loads(header_bytes.decode("utf-8"))
        
        # Group tensors by layer
        current_layer = -1
        layer_start = 8 + header_size
        layer_end = layer_start
        
        for name, info in sorted(header.items()):
            if name == "__metadata__":
                continue
            
            # Extract layer number from tensor name
            # e.g., "model.layers.0.self_attn.q_proj.weight"
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass
            
            if layer_idx is None:
                continue
            
            offsets = info.get("data_offsets", [0, 0])
            tensor_start = 8 + header_size + offsets[0]
            tensor_end = 8 + header_size + offsets[1]
            
            if layer_idx != current_layer:
                if current_layer >= 0:
                    self.layer_offsets[current_layer] = (layer_start, layer_end - layer_start)
                current_layer = layer_idx
                layer_start = tensor_start
            
            layer_end = tensor_end
        
        if current_layer >= 0:
            self.layer_offsets[current_layer] = (layer_start, layer_end - layer_start)
    
    def get_layer_data(self, layer_idx: int) -> bytes:
        """Get raw bytes for a layer."""
        if layer_idx not in self.layer_offsets:
            raise KeyError(f"Layer {layer_idx} not found in offsets")
        
        offset, size = self.layer_offsets[layer_idx]
        return self._mmap[offset:offset + size]
    
    def get_num_layers(self) -> int:
        """Get number of layers."""
        return len(self.layer_offsets)
    
    def close(self):
        """Close memory map."""
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
