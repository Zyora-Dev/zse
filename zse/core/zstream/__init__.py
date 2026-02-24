"""
zStream - Ultra Memory-Efficient Layer Streaming

Key Innovation: Run 70B models on 24GB GPUs through dynamic layer streaming.

Unlike GGUF/llama.cpp which require ALL weights in memory (RAM + VRAM),
zStream streams layers on-demand with intelligent prefetching.

Features:
- On-demand layer loading (only active layers on GPU)
- Async prefetching (hide transfer latency)  
- GPU ↔ CPU ↔ Disk tiered storage
- Memory-mapped weights for instant access
- Sliding window of active layers
- Memory pressure-aware eviction

Example:
    from transformers import AutoModelForCausalLM
    from zse.core.zstream import StreamingModel
    
    # Load 70B model to CPU
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-70b-hf",
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    
    # Wrap for streaming - only 4 layers on GPU at once!
    streaming = StreamingModel(model, gpu_layers=4)
    
    # Generate normally
    output = streaming.generate(input_ids, max_new_tokens=100)
"""

from .memory_tracker import (
    MemoryTracker,
    MemoryPressure,
    GPUMemoryState,
)

from .streamer import (
    LayerStreamer,
    StreamerConfig,
    LayerState,
    StreamingForward,
)

from .prefetcher import (
    AsyncPrefetcher,
    PrefetchStrategy,
    BandwidthEstimator,
)

from .offload import (
    OffloadManager,
    StorageTier,
    MemoryMappedWeights,
)

from .streaming_model import (
    StreamingModel,
    StreamingConfig,
    StreamingModelOutput,
    wrap_model_for_streaming,
)


__all__ = [
    # Memory tracking
    "MemoryTracker",
    "MemoryPressure",
    "GPUMemoryState",
    
    # Layer streaming
    "LayerStreamer",
    "StreamerConfig",
    "LayerState",
    "StreamingForward",
    
    # Prefetching
    "AsyncPrefetcher",
    "PrefetchStrategy",
    "BandwidthEstimator",
    
    # Offloading
    "OffloadManager",
    "StorageTier",
    "MemoryMappedWeights",
    
    # High-level API
    "StreamingModel",
    "StreamingConfig",
    "StreamingModelOutput",
    "wrap_model_for_streaming",
]

# Version
__version__ = "0.1.0"
