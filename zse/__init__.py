"""
ZSE - Z Server Engine

Ultra memory-efficient LLM inference engine.

Key Features:
- zAttention: Custom paged, flash, and sparse attention kernels
- zQuantize: Per-tensor INT2-8 mixed precision quantization
- zKV: Quantized KV cache with sliding precision
- zStream: Layer streaming with async prefetch
- zOrchestrator: Smart memory recommendations based on FREE memory
- Efficiency modes: speed / balanced / memory / ultra

Memory Targets:
- 7B model in 3-3.5GB VRAM
- 14B model in 6GB VRAM
- 32B model in 16-20GB VRAM
- 70B model in 24-32GB VRAM

Usage:
    # CLI
    $ zse serve meta-llama/Llama-3-8B
    $ zse chat meta-llama/Llama-3-8B
    $ zse convert model.safetensors --output model.zse
    
    # Python
    from zse import Engine
    engine = Engine("meta-llama/Llama-3-8B", max_memory="8GB")
    response = engine.generate("Hello, world!")
"""

from zse.version import __version__, __version_info__

__all__ = [
    "__version__",
    "__version_info__",
]
