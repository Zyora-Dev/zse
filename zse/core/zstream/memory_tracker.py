"""
Dynamic VRAM Memory Tracker

Monitors GPU memory in real-time and provides intelligent allocation decisions.

Key Features:
- Real-time VRAM monitoring
- Memory pressure detection
- Allocation recommendations
- Fragmentation tracking
"""

import torch
import threading
import time
from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class MemoryPressure(Enum):
    """Memory pressure levels."""
    LOW = "low"           # < 50% used, can load more layers
    MEDIUM = "medium"     # 50-75% used, normal operation
    HIGH = "high"         # 75-90% used, consider offloading
    CRITICAL = "critical" # > 90% used, must offload immediately


@dataclass
class GPUMemoryState:
    """Current GPU memory state."""
    total: int           # Total VRAM in bytes
    allocated: int       # Currently allocated
    reserved: int        # Reserved by PyTorch
    free: int            # Actually free
    cached: int          # Cached allocations
    
    # Computed
    used_percent: float = field(init=False)
    pressure: MemoryPressure = field(init=False)
    
    def __post_init__(self):
        self.used_percent = (self.allocated / self.total) * 100 if self.total > 0 else 0
        
        if self.used_percent < 50:
            self.pressure = MemoryPressure.LOW
        elif self.used_percent < 75:
            self.pressure = MemoryPressure.MEDIUM
        elif self.used_percent < 90:
            self.pressure = MemoryPressure.HIGH
        else:
            self.pressure = MemoryPressure.CRITICAL
    
    @property
    def free_gb(self) -> float:
        return self.free / (1024**3)
    
    @property
    def allocated_gb(self) -> float:
        return self.allocated / (1024**3)
    
    @property
    def total_gb(self) -> float:
        return self.total / (1024**3)


class MemoryTracker:
    """
    Real-time GPU memory tracker with pressure-aware decisions.
    
    Key Innovation: Dynamic layer window sizing based on actual free VRAM,
    not static configuration. This is what GGUF/llama.cpp cannot do.
    
    Usage:
        tracker = MemoryTracker(device=0)
        state = tracker.get_state()
        
        # How many layers can we fit?
        max_layers = tracker.estimate_layer_capacity(layer_size_bytes)
        
        # Should we offload?
        if tracker.should_offload():
            offload_layers(...)
    """
    
    def __init__(
        self,
        device: int = 0,
        safety_margin: float = 0.1,  # Keep 10% free for activations
        poll_interval: float = 0.1,  # 100ms polling
    ):
        self.device = device
        self.safety_margin = safety_margin
        self.poll_interval = poll_interval
        
        # State
        self._current_state: Optional[GPUMemoryState] = None
        self._state_lock = threading.Lock()
        
        # Callbacks for memory pressure events
        self._pressure_callbacks: Dict[MemoryPressure, list] = {
            level: [] for level in MemoryPressure
        }
        
        # Background monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Layer tracking
        self._loaded_layers: Dict[int, int] = {}  # layer_idx -> size_bytes
        
        # Initialize
        self._update_state()
    
    def _update_state(self) -> GPUMemoryState:
        """Update current memory state."""
        if not torch.cuda.is_available():
            # CPU fallback
            import psutil
            mem = psutil.virtual_memory()
            state = GPUMemoryState(
                total=mem.total,
                allocated=mem.used,
                reserved=0,
                free=mem.available,
                cached=0,
            )
        else:
            torch.cuda.synchronize(self.device)
            
            total = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            
            # Free = Total - Reserved (not allocated, since reserved > allocated)
            free = total - reserved
            cached = reserved - allocated
            
            state = GPUMemoryState(
                total=total,
                allocated=allocated,
                reserved=reserved,
                free=free,
                cached=cached,
            )
        
        with self._state_lock:
            old_pressure = self._current_state.pressure if self._current_state else None
            self._current_state = state
            
            # Trigger callbacks on pressure change
            if old_pressure and old_pressure != state.pressure:
                self._trigger_pressure_callbacks(state.pressure)
        
        return state
    
    def get_state(self) -> GPUMemoryState:
        """Get current memory state."""
        return self._update_state()
    
    def get_free_bytes(self) -> int:
        """Get free memory in bytes (with safety margin)."""
        state = self.get_state()
        usable = state.free * (1 - self.safety_margin)
        return int(usable)
    
    def get_free_gb(self) -> float:
        """Get free memory in GB."""
        return self.get_free_bytes() / (1024**3)
    
    def estimate_layer_capacity(
        self,
        layer_size_bytes: int,
        min_layers: int = 1,
    ) -> int:
        """
        Estimate how many layers can fit in current free memory.
        
        Args:
            layer_size_bytes: Size of one layer in bytes
            min_layers: Minimum layers to return
            
        Returns:
            Number of layers that can fit
        """
        free = self.get_free_bytes()
        capacity = free // layer_size_bytes if layer_size_bytes > 0 else 0
        return max(capacity, min_layers)
    
    def should_offload(self) -> bool:
        """Check if we should offload layers due to memory pressure."""
        state = self.get_state()
        return state.pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL)
    
    def should_load_more(self) -> bool:
        """Check if we have room to load more layers."""
        state = self.get_state()
        return state.pressure in (MemoryPressure.LOW, MemoryPressure.MEDIUM)
    
    def register_layer(self, layer_idx: int, size_bytes: int):
        """Register a loaded layer for tracking."""
        self._loaded_layers[layer_idx] = size_bytes
    
    def unregister_layer(self, layer_idx: int):
        """Unregister an offloaded layer."""
        self._loaded_layers.pop(layer_idx, None)
    
    def get_loaded_layers(self) -> Dict[int, int]:
        """Get currently loaded layers and their sizes."""
        return dict(self._loaded_layers)
    
    def get_total_loaded_size(self) -> int:
        """Get total size of loaded layers."""
        return sum(self._loaded_layers.values())
    
    def on_pressure(
        self,
        level: MemoryPressure,
        callback: Callable[[GPUMemoryState], None],
    ):
        """Register callback for memory pressure events."""
        self._pressure_callbacks[level].append(callback)
    
    def _trigger_pressure_callbacks(self, level: MemoryPressure):
        """Trigger callbacks for a pressure level."""
        state = self._current_state
        for callback in self._pressure_callbacks[level]:
            try:
                callback(state)
            except Exception as e:
                print(f"Warning: Pressure callback error: {e}")
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._update_state()
            except Exception:
                pass
            time.sleep(self.poll_interval)
    
    def clear_cache(self):
        """Clear PyTorch CUDA cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
    
    def __repr__(self) -> str:
        state = self.get_state()
        return (
            f"MemoryTracker(device={self.device}, "
            f"used={state.allocated_gb:.2f}GB/{state.total_gb:.2f}GB, "
            f"free={state.free_gb:.2f}GB, "
            f"pressure={state.pressure.value})"
        )


def get_optimal_layer_window(
    total_layers: int,
    layer_size_bytes: int,
    device: int = 0,
    min_window: int = 2,
    max_window: Optional[int] = None,
) -> Tuple[int, float]:
    """
    Calculate optimal layer window size based on available memory.
    
    This is the KEY INNOVATION over GGUF:
    - GGUF: Static allocation at model load time
    - ZSE: Dynamic window based on current free VRAM
    
    Args:
        total_layers: Total number of model layers
        layer_size_bytes: Size of each layer in bytes
        device: GPU device index
        min_window: Minimum window size
        max_window: Maximum window size (default: total_layers)
        
    Returns:
        (window_size, estimated_memory_gb)
    """
    tracker = MemoryTracker(device=device)
    max_window = max_window or total_layers
    
    # How many layers can we fit?
    capacity = tracker.estimate_layer_capacity(layer_size_bytes)
    
    # Clamp to valid range
    window_size = max(min_window, min(capacity, max_window, total_layers))
    
    # Estimated memory usage
    estimated_bytes = window_size * layer_size_bytes
    estimated_gb = estimated_bytes / (1024**3)
    
    return window_size, estimated_gb
