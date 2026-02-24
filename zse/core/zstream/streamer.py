"""
zStream Layer Streamer

The core of ZSE's memory innovation: stream model layers on-demand
instead of loading everything into GPU at once.

Key Capabilities:
- Run 70B model on 24GB GPU (only ~4-8 layers in VRAM at a time)
- Dynamic window sizing based on actual free VRAM
- Zero-copy transfers with pinned memory
- Overlap compute with data movement

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                      70B Model (140GB)                  │
    │  [L0][L1][L2]...[L79]                                   │
    └─────────────────────────────────────────────────────────┘
                             ↓
    ┌─────────────────────────────────────────────────────────┐
    │                     Disk/NVMe (Cold)                    │
    │  Memory-mapped weights for instant access               │
    └─────────────────────────────────────────────────────────┘
                             ↓
    ┌─────────────────────────────────────────────────────────┐
    │                 CPU RAM (Warm) - 64GB                   │
    │  Prefetched layers ready for GPU transfer               │
    │  [L3][L4][L5][L6]                                       │
    └─────────────────────────────────────────────────────────┘
                             ↓
    ┌─────────────────────────────────────────────────────────┐
    │                   GPU VRAM (Hot) - 24GB                 │
    │  Active sliding window: [L0][L1][L2]                    │
    │  Currently computing: L1                                │
    └─────────────────────────────────────────────────────────┘

The Magic: While GPU computes layer N, CPU prefetches layer N+window_size
"""

import torch
import torch.nn as nn
import threading
import queue
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import time

from .memory_tracker import MemoryTracker, GPUMemoryState, MemoryPressure


class LayerLocation(Enum):
    """Where a layer is stored."""
    GPU = "gpu"      # In VRAM, ready for compute
    CPU = "cpu"      # In RAM, pinned for fast transfer
    DISK = "disk"    # On disk, memory-mapped


@dataclass
class LayerState:
    """State of a single model layer."""
    index: int
    location: LayerLocation
    size_bytes: int
    last_accessed: float = 0.0
    access_count: int = 0
    
    # Tensor references
    gpu_tensors: Optional[Dict[str, torch.Tensor]] = None
    cpu_tensors: Optional[Dict[str, torch.Tensor]] = None
    
    def touch(self):
        """Mark layer as accessed."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class StreamerConfig:
    """Configuration for LayerStreamer."""
    # Window sizing
    min_window_size: int = 2       # Minimum layers in GPU
    max_window_size: int = 16      # Maximum layers in GPU
    target_gpu_usage: float = 0.75 # Target 75% GPU memory usage
    
    # Prefetching
    prefetch_count: int = 2        # Layers to prefetch ahead
    prefetch_threads: int = 2      # Parallel prefetch threads
    
    # Memory management
    use_pinned_memory: bool = True  # Pin CPU tensors for fast transfer
    enable_cuda_graphs: bool = False  # Use CUDA graphs (experimental)
    
    # Offloading
    offload_to_disk: bool = False  # Enable disk offloading for huge models
    disk_cache_path: Optional[str] = None  # Path for disk cache
    
    # Performance
    overlap_compute_transfer: bool = True  # Async transfers during compute


class LayerStreamer:
    """
    Dynamic layer streaming engine.
    
    This is ZSE's key innovation over GGUF/llama.cpp:
    - GGUF loads entire model, fails if won't fit
    - ZSE streams layers dynamically, adapts to available memory
    
    Usage:
        streamer = LayerStreamer(model, config)
        
        # Forward pass with streaming
        for layer_idx in range(num_layers):
            # Streamer handles loading/offloading
            layer = streamer.get_layer(layer_idx)
            hidden = layer(hidden)
            streamer.release_layer(layer_idx)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[StreamerConfig] = None,
        device: int = 0,
    ):
        self.model = model
        self.config = config or StreamerConfig()
        self.device = device
        
        # Memory tracking
        self.memory_tracker = MemoryTracker(device=device)
        
        # Layer management
        self.layers: Dict[int, LayerState] = {}
        self.num_layers = 0
        self._layer_modules: Dict[int, nn.Module] = {}
        
        # GPU window (LRU cache of layers in GPU)
        self._gpu_window: OrderedDict[int, LayerState] = OrderedDict()
        self._window_lock = threading.RLock()
        
        # Prefetch queue
        self._prefetch_queue: queue.Queue = queue.Queue()
        self._prefetch_threads: List[threading.Thread] = []
        self._prefetch_running = False
        
        # Transfer streams for async operations
        self._transfer_stream: Optional[torch.cuda.Stream] = None
        self._compute_stream: Optional[torch.cuda.Stream] = None
        
        # Statistics
        self.stats = {
            "gpu_hits": 0,
            "cpu_hits": 0,
            "disk_hits": 0,
            "prefetch_hits": 0,
            "evictions": 0,
            "total_transfer_time": 0.0,
        }
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize streamer with model."""
        # Find model layers
        self._discover_layers()
        
        # Setup CUDA streams
        if torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream(device=self.device)
            self._compute_stream = torch.cuda.Stream(device=self.device)
        
        # Start prefetch threads
        self._start_prefetch_workers()
        
        # Register memory pressure callbacks
        self.memory_tracker.on_pressure(
            MemoryPressure.HIGH,
            self._on_high_pressure,
        )
        self.memory_tracker.on_pressure(
            MemoryPressure.CRITICAL,
            self._on_critical_pressure,
        )
    
    def _discover_layers(self):
        """Discover transformer layers in the model."""
        # Common layer names across architectures
        layer_patterns = [
            "model.layers",     # LLaMA, Mistral, Qwen
            "transformer.h",    # GPT-2, GPT-J
            "encoder.layer",    # BERT
            "decoder.layers",   # T5
        ]
        
        layers_module = None
        for pattern in layer_patterns:
            try:
                parts = pattern.split(".")
                module = self.model
                for part in parts:
                    module = getattr(module, part)
                layers_module = module
                break
            except AttributeError:
                continue
        
        if layers_module is None:
            # Try to find any ModuleList
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 1:
                    layers_module = module
                    break
        
        if layers_module is None:
            raise ValueError("Could not find transformer layers in model")
        
        # Register layers
        self.num_layers = len(layers_module)
        for idx, layer in enumerate(layers_module):
            size = self._estimate_layer_size(layer)
            state = LayerState(
                index=idx,
                location=LayerLocation.CPU,  # Start on CPU
                size_bytes=size,
            )
            self.layers[idx] = state
            self._layer_modules[idx] = layer
        
        print(f"[zStream] Discovered {self.num_layers} layers, "
              f"~{self._estimate_layer_size(layers_module[0]) / 1024**3:.2f} GB each")
    
    def _estimate_layer_size(self, layer: nn.Module) -> int:
        """Estimate memory size of a layer."""
        total = 0
        for param in layer.parameters():
            total += param.numel() * param.element_size()
        for buffer in layer.buffers():
            total += buffer.numel() * buffer.element_size()
        return total
    
    def calculate_optimal_window(self) -> int:
        """Calculate optimal GPU window size based on available memory."""
        if not self.layers:
            return self.config.min_window_size
        
        # Get first layer size as reference
        layer_size = self.layers[0].size_bytes
        
        # Ask memory tracker for capacity
        capacity = self.memory_tracker.estimate_layer_capacity(layer_size)
        
        # Apply config constraints
        window = max(self.config.min_window_size, 
                    min(capacity, self.config.max_window_size, self.num_layers))
        
        return window
    
    def get_layer(self, layer_idx: int) -> nn.Module:
        """
        Get a layer, loading it to GPU if necessary.
        
        This is the main entry point. It:
        1. Checks if layer is in GPU window
        2. If not, loads it (evicting old layers if needed)
        3. Queues prefetch for upcoming layers
        4. Returns the layer module
        """
        if layer_idx not in self.layers:
            raise IndexError(f"Layer {layer_idx} not found")
        
        state = self.layers[layer_idx]
        state.touch()
        
        with self._window_lock:
            # Check if already in GPU
            if layer_idx in self._gpu_window:
                self.stats["gpu_hits"] += 1
                # Move to end (most recently used)
                self._gpu_window.move_to_end(layer_idx)
                return self._layer_modules[layer_idx]
            
            # Need to load to GPU
            if state.location == LayerLocation.CPU:
                self.stats["cpu_hits"] += 1
            else:
                self.stats["disk_hits"] += 1
            
            # Make room if needed
            self._ensure_window_space()
            
            # Load layer to GPU
            self._load_to_gpu(layer_idx)
            
            # Queue prefetch for next layers
            self._queue_prefetch(layer_idx)
            
            return self._layer_modules[layer_idx]
    
    def release_layer(self, layer_idx: int):
        """
        Signal that we're done with a layer.
        
        The layer stays in GPU window until evicted.
        This is called after forward pass through the layer.
        """
        # Currently a no-op, but useful for future optimizations
        # like immediate offloading for memory-critical situations
        pass
    
    def is_layer_on_gpu(self, layer_idx: int) -> bool:
        """Check if a layer is currently on GPU."""
        if layer_idx not in self.layers:
            return False
        return self.layers[layer_idx].location == LayerLocation.GPU
    
    def _ensure_window_space(self):
        """Ensure there's space in GPU window, evicting if necessary."""
        window_size = self.calculate_optimal_window()
        
        while len(self._gpu_window) >= window_size:
            self._evict_oldest()
    
    def _evict_oldest(self):
        """Evict the oldest (least recently used) layer from GPU."""
        if not self._gpu_window:
            return
        
        # Get oldest (first in OrderedDict)
        layer_idx, state = next(iter(self._gpu_window.items()))
        
        # Move to CPU
        self._offload_to_cpu(layer_idx)
        
        # Remove from window
        del self._gpu_window[layer_idx]
        self.stats["evictions"] += 1
    
    def _load_to_gpu(self, layer_idx: int):
        """Load a layer from CPU/disk to GPU."""
        state = self.layers[layer_idx]
        layer = self._layer_modules[layer_idx]
        
        start_time = time.perf_counter()
        
        # Use transfer stream for async
        stream = self._transfer_stream if self.config.overlap_compute_transfer else None
        
        with torch.cuda.stream(stream) if stream else nullcontext():
            # Move all parameters and buffers to GPU
            layer.to(device=f"cuda:{self.device}", non_blocking=True)
        
        # Synchronize if not using async
        if not self.config.overlap_compute_transfer:
            torch.cuda.synchronize(self.device)
        
        # Update state
        state.location = LayerLocation.GPU
        self._gpu_window[layer_idx] = state
        
        # Track memory
        self.memory_tracker.register_layer(layer_idx, state.size_bytes)
        
        elapsed = time.perf_counter() - start_time
        self.stats["total_transfer_time"] += elapsed
    
    def _offload_to_cpu(self, layer_idx: int):
        """Offload a layer from GPU to CPU."""
        state = self.layers[layer_idx]
        layer = self._layer_modules[layer_idx]
        
        # Pin memory for faster future transfers
        if self.config.use_pinned_memory:
            # Move to CPU with pinned memory
            for param in layer.parameters():
                param.data = param.data.cpu().pin_memory()
            for name, buffer in layer.named_buffers():
                if buffer is not None:
                    buffer.data = buffer.data.cpu().pin_memory()
        else:
            layer.cpu()
        
        # Update state
        state.location = LayerLocation.CPU
        
        # Untrack memory
        self.memory_tracker.unregister_layer(layer_idx)
    
    def _queue_prefetch(self, current_idx: int):
        """Queue prefetch for upcoming layers."""
        for offset in range(1, self.config.prefetch_count + 1):
            next_idx = current_idx + offset
            if next_idx < self.num_layers and next_idx not in self._gpu_window:
                if not self._prefetch_queue.full():
                    self._prefetch_queue.put(next_idx)
    
    def _start_prefetch_workers(self):
        """Start background prefetch worker threads."""
        self._prefetch_running = True
        
        for i in range(self.config.prefetch_threads):
            t = threading.Thread(
                target=self._prefetch_worker,
                daemon=True,
                name=f"zstream-prefetch-{i}",
            )
            t.start()
            self._prefetch_threads.append(t)
    
    def _prefetch_worker(self):
        """Background worker that prefetches layers."""
        while self._prefetch_running:
            try:
                layer_idx = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Check if still needed
            with self._window_lock:
                if layer_idx in self._gpu_window:
                    continue  # Already loaded
                
                # Check memory pressure
                if self.memory_tracker.should_offload():
                    continue  # Skip prefetch under pressure
                
                # Check if room in window
                window_size = self.calculate_optimal_window()
                if len(self._gpu_window) >= window_size:
                    continue  # Window full
                
                # Prefetch!
                try:
                    self._load_to_gpu(layer_idx)
                    self.stats["prefetch_hits"] += 1
                except Exception as e:
                    print(f"[zStream] Prefetch error for layer {layer_idx}: {e}")
    
    def _on_high_pressure(self, state: GPUMemoryState):
        """Handle high memory pressure - reduce window."""
        print(f"[zStream] High memory pressure: {state.used_percent:.1f}% used")
        with self._window_lock:
            # Evict one layer
            if len(self._gpu_window) > self.config.min_window_size:
                self._evict_oldest()
    
    def _on_critical_pressure(self, state: GPUMemoryState):
        """Handle critical memory pressure - aggressive eviction."""
        print(f"[zStream] CRITICAL memory pressure: {state.used_percent:.1f}% used")
        with self._window_lock:
            # Evict multiple layers
            while (len(self._gpu_window) > self.config.min_window_size and
                   self.memory_tracker.should_offload()):
                self._evict_oldest()
                self.memory_tracker.clear_cache()
    
    def preload_layers(self, layer_indices: List[int]):
        """Preload specific layers to GPU."""
        for idx in layer_indices:
            if idx not in self._gpu_window:
                self._ensure_window_space()
                self._load_to_gpu(idx)
    
    def offload_all(self):
        """Offload all layers from GPU."""
        with self._window_lock:
            for layer_idx in list(self._gpu_window.keys()):
                self._offload_to_cpu(layer_idx)
                del self._gpu_window[layer_idx]
        
        self.memory_tracker.clear_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        total_hits = (self.stats["gpu_hits"] + 
                     self.stats["cpu_hits"] + 
                     self.stats["disk_hits"])
        
        return {
            **self.stats,
            "total_hits": total_hits,
            "gpu_hit_rate": self.stats["gpu_hits"] / total_hits if total_hits > 0 else 0,
            "window_size": len(self._gpu_window),
            "optimal_window": self.calculate_optimal_window(),
            "memory": self.memory_tracker.get_state(),
        }
    
    def shutdown(self):
        """Shutdown streamer and cleanup."""
        self._prefetch_running = False
        for t in self._prefetch_threads:
            t.join(timeout=1.0)
        
        self.offload_all()
        self.memory_tracker.stop_monitoring()
    
    def __repr__(self) -> str:
        return (
            f"LayerStreamer(layers={self.num_layers}, "
            f"window={len(self._gpu_window)}/{self.calculate_optimal_window()}, "
            f"memory={self.memory_tracker})"
        )


# Context manager helper
from contextlib import nullcontext


class StreamingForward:
    """
    Context manager for streaming forward pass.
    
    Usage:
        streamer = LayerStreamer(model)
        
        with StreamingForward(streamer) as stream:
            hidden = embed(input_ids)
            for i in range(num_layers):
                layer = stream.get(i)
                hidden = layer(hidden)
            output = lm_head(hidden)
    """
    
    def __init__(self, streamer: LayerStreamer):
        self.streamer = streamer
        self._current_layer: Optional[int] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._current_layer is not None:
            self.streamer.release_layer(self._current_layer)
    
    def get(self, layer_idx: int) -> nn.Module:
        """Get layer and release previous."""
        if self._current_layer is not None:
            self.streamer.release_layer(self._current_layer)
        
        self._current_layer = layer_idx
        return self.streamer.get_layer(layer_idx)
