"""
zAttention - Custom Attention Kernels

Implements memory-efficient attention mechanisms:
- Paged attention (PagedAttention-style)
- Flash attention integration
- Sparse attention patterns
- Grouped-Query Attention (GQA) support
- Multi-Query Attention (MQA) support

Backends:
- Triton (default, easier development)
- CUDA (optimized, compiled from csrc/)
"""

from .attention import (
    zAttention,
    AttentionConfig,
    AttentionType,
    AttentionBackend,
    create_attention,
)

from .triton_kernels import (
    TRITON_AVAILABLE,
    paged_attention_v1,
    flash_attention,
    paged_attention_v1_torch,
    flash_attention_torch,
)

__all__ = [
    # Main interface
    "zAttention",
    "AttentionConfig",
    "AttentionType",
    "AttentionBackend",
    "create_attention",
    # Triton kernels
    "TRITON_AVAILABLE",
    "paged_attention_v1",
    "flash_attention",
    # Fallbacks
    "paged_attention_v1_torch",
    "flash_attention_torch",
]
