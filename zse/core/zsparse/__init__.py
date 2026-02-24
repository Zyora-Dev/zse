"""
zSparse - Sparse Attention Patterns

Implements memory-efficient attention patterns:
- Sliding window (local attention)
- Global tokens (system prompt, special markers)
- Strided patterns (every Nth token)
- Dynamic sparsity (based on attention scores)

Memory Savings:
- Full attention: O(n²) memory
- Sparse attention: O(n) memory
- 32K context: 4GB → 256MB (16x reduction)

Patterns:
- Local: attend to nearby tokens only
- Global: certain tokens attend to all
- Dilated: skip tokens at regular intervals
- Learned: dynamic based on content

Usage:
    from zse.core.zsparse import zSparseAttention, SparsePattern

    # Create sliding window attention (512 token window)
    attn = zSparseAttention(
        pattern=SparsePattern.sliding_window(window_size=512),
        num_heads=32,
        head_dim=128,
    )
    
    # Or Longformer-style (window + global tokens)
    attn = zSparseAttention(
        pattern=SparsePattern.longformer(window_size=512, num_global_start=1),
        num_heads=32,
        head_dim=128,
    )
    
    # Forward pass
    output = attn(query, key, value)

Memory Comparison (32K context, 32 heads, 128 dim):
    Full attention:     2 GB
    Sliding (512):     64 MB  (32x smaller)
    Longformer (512):  67 MB  (30x smaller)
"""

from zse.core.zsparse.patterns import (
    SparsePattern,
    PatternConfig,
    PatternType,
    create_pattern_from_name,
)

from zse.core.zsparse.mask import (
    SparseMask,
    SparseMaskGenerator,
    create_causal_sliding_window_mask,
    visualize_mask,
)

from zse.core.zsparse.sparse_attention import (
    zSparseAttention,
    SparseAttentionConfig,
    create_sparse_attention,
    replace_attention_with_sparse,
    benchmark_sparse_attention,
)

from zse.core.zsparse.triton_kernels import (
    TRITON_AVAILABLE,
    sparse_attention,
    sliding_window,
    longformer,
)

__all__ = [
    # Patterns
    "SparsePattern",
    "PatternConfig",
    "PatternType",
    "create_pattern_from_name",
    
    # Masks
    "SparseMask",
    "SparseMaskGenerator",
    "create_causal_sliding_window_mask",
    "visualize_mask",
    
    # Attention modules
    "zSparseAttention",
    "SparseAttentionConfig",
    "create_sparse_attention",
    "replace_attention_with_sparse",
    "benchmark_sparse_attention",
    
    # Kernels
    "TRITON_AVAILABLE",
    "sparse_attention",
    "sliding_window",
    "longformer",
]
