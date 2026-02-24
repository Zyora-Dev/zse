"""
zSparse Attention - Main Interface

Provides a unified interface for sparse attention computation:

1. Auto-selects best pattern based on context length
2. Integrates with existing zAttention module
3. Memory-efficient long-context processing
4. Multiple backend support (Triton, PyTorch)

Usage:
    from zse.core.zsparse import zSparseAttention, SparsePattern

    # Create sparse attention with sliding window
    sparse_attn = zSparseAttention(
        pattern=SparsePattern.sliding_window(window_size=512)
    )
    
    # Or Longformer-style (window + global)
    sparse_attn = zSparseAttention(
        pattern=SparsePattern.longformer(window_size=512, num_global_start=1)
    )
    
    # Compute attention
    output = sparse_attn(query, key, value)

Memory Savings (example 32K context):
    - Full attention: 32K × 32K × 2 bytes = 2 GB
    - Sliding window (512): 32K × 512 × 2 bytes = 32 MB (64x reduction)
    - Longformer (512 + 64 global): ~35 MB (57x reduction)

Author: ZSE Team
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn

from .patterns import SparsePattern, PatternConfig, PatternType
from .mask import SparseMask, SparseMaskGenerator
from .triton_kernels import (
    TRITON_AVAILABLE,
    sparse_attention,
    sliding_window,
    longformer,
    sparse_attention_forward_torch,
)


@dataclass
class SparseAttentionConfig:
    """Configuration for sparse attention module."""
    
    # Attention dimensions
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # For GQA
    head_dim: int = 128
    
    # Sparsity pattern
    pattern: SparsePattern = None
    
    # Backend selection
    use_triton: bool = True
    fallback_to_torch: bool = True
    
    # Memory optimization
    use_memory_efficient_inference: bool = True
    chunk_size: Optional[int] = None  # Process in chunks for very long sequences
    
    def __post_init__(self):
        if self.pattern is None:
            self.pattern = SparsePattern.sliding_window(window_size=512)
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


class zSparseAttention(nn.Module):
    """
    ZSE Sparse Attention Module
    
    Drop-in replacement for standard attention with memory savings
    for long contexts through sparse attention patterns.
    
    Supports:
    - Sliding window attention
    - Longformer-style (window + global tokens)
    - BigBird-style (window + global + random)
    - Block sparse patterns
    - Custom patterns
    """
    
    def __init__(
        self,
        config: SparseAttentionConfig = None,
        pattern: SparsePattern = None,
        num_heads: int = 32,
        head_dim: int = 128,
    ):
        super().__init__()
        
        if config is not None:
            self.config = config
        else:
            self.config = SparseAttentionConfig(
                num_heads=num_heads,
                head_dim=head_dim,
                pattern=pattern or SparsePattern.sliding_window(),
            )
        
        self.pattern = self.config.pattern
        self.scale = 1.0 / math.sqrt(self.config.head_dim)
        
        # Mask generator (with caching)
        self.mask_generator = SparseMaskGenerator(self.pattern)
        
        # Check backend availability
        self._triton_available = TRITON_AVAILABLE and self.config.use_triton
        self._using_triton = self._triton_available
        
        if not self._triton_available and not self.config.fallback_to_torch:
            raise RuntimeError("Triton not available and fallback disabled")
    
    @property
    def backend(self) -> str:
        """Current compute backend."""
        return "triton" if self._using_triton else "torch"
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sparse attention.
        
        Args:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_kv_heads, seq_len, head_dim]
            value: [batch, num_kv_heads, seq_len, head_dim]
            attention_mask: Optional additional mask (combined with sparse mask)
        
        Returns:
            Output tensor [batch, num_heads, seq_len, head_dim]
        """
        batch, num_heads, seq_len, head_dim = query.shape
        
        # Expand KV for GQA if needed
        if self.config.num_kv_heads != self.config.num_heads:
            key = self._expand_kv(key)
            value = self._expand_kv(value)
        
        # Select computation path
        if self._using_triton:
            return self._forward_triton(query, key, value)
        else:
            return self._forward_torch(query, key, value, attention_mask)
    
    def _forward_triton(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Triton sparse attention forward."""
        
        # Determine which kernel to use based on pattern
        window_size = self.pattern.window_size
        num_global = self.pattern.num_global_start + len(self.pattern.global_tokens)
        
        if num_global > 0:
            # Longformer-style
            return longformer(
                query, key, value,
                window_size=window_size,
                num_global=num_global,
            )
        elif self.pattern.stride > 0:
            # Strided + local (use general sparse)
            return sparse_attention(
                query, key, value,
                window_size=window_size,
                num_global=0,
            )
        else:
            # Pure sliding window (optimized kernel)
            return sliding_window(
                query, key, value,
                window_size=window_size,
            )
    
    def _forward_torch(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PyTorch fallback forward."""
        batch, num_heads, seq_len, head_dim = query.shape
        
        # For very long sequences, process in chunks
        if self.config.chunk_size and seq_len > self.config.chunk_size:
            return self._chunked_attention(query, key, value, attention_mask)
        
        # Generate sparse mask
        mask = self.mask_generator.generate(seq_len)
        dense_mask = mask.to_dense().to(query.device)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply sparse mask
        scores = scores.masked_fill(~dense_mask, float('-inf'))
        
        # Apply additional mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax and output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        output = torch.matmul(attn_weights, value)
        
        return output
    
    def _chunked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process attention in chunks for very long sequences.
        
        Memory efficient but may lose some long-range context.
        """
        batch, num_heads, seq_len, head_dim = query.shape
        chunk_size = self.config.chunk_size
        
        outputs = []
        
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            
            # Query chunk
            q_chunk = query[:, :, start:end, :]
            
            # Key/Value: include window before current chunk
            k_start = max(0, start - self.pattern.window_size)
            k_chunk = key[:, :, k_start:end, :]
            v_chunk = value[:, :, k_start:end, :]
            
            # Compute chunk attention
            chunk_out = sparse_attention_forward_torch(
                q_chunk, k_chunk, v_chunk,
                window_size=self.pattern.window_size,
                num_global=self.pattern.num_global_start,
            )
            
            outputs.append(chunk_out)
        
        return torch.cat(outputs, dim=2)
    
    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand key/value for GQA."""
        batch, num_kv_heads, seq_len, head_dim = kv.shape
        gqa_factor = self.config.num_heads // self.config.num_kv_heads
        
        kv = kv.unsqueeze(2).expand(
            batch, num_kv_heads, gqa_factor, seq_len, head_dim
        )
        return kv.reshape(batch, self.config.num_heads, seq_len, head_dim)
    
    def get_memory_estimate(self, seq_len: int) -> dict:
        """
        Estimate memory usage for given sequence length.
        
        Returns dict with full vs sparse memory estimates.
        """
        full_attention_bytes = seq_len * seq_len * 2  # FP16
        
        # Sparse estimate based on pattern
        sparsity = self.pattern.get_sparsity_ratio(seq_len)
        sparse_attention_bytes = full_attention_bytes * (1 - sparsity)
        
        # Additional memory for mask (if using torch backend)
        mask_bytes = 0
        if not self._using_triton:
            mask_bytes = seq_len * seq_len // 8  # boolean packed
        
        return {
            "full_attention_mb": full_attention_bytes / (1024 * 1024),
            "sparse_attention_mb": sparse_attention_bytes / (1024 * 1024),
            "mask_mb": mask_bytes / (1024 * 1024),
            "sparsity": sparsity,
            "reduction_factor": 1 / (1 - sparsity + 1e-6),
        }
    
    def extra_repr(self) -> str:
        return (
            f"heads={self.config.num_heads}, "
            f"head_dim={self.config.head_dim}, "
            f"window={self.pattern.window_size}, "
            f"global={self.pattern.num_global_start}, "
            f"backend={self.backend}"
        )


def replace_attention_with_sparse(
    model: nn.Module,
    pattern: SparsePattern = None,
    layer_indices: Optional[list] = None,
) -> nn.Module:
    """
    Replace standard attention layers with sparse attention.
    
    Args:
        model: PyTorch model with attention layers
        pattern: Sparse pattern to use (default: sliding window 512)
        layer_indices: Which layers to replace (default: all)
    
    Returns:
        Model with sparse attention layers
    
    Note: This modifies the model in-place.
    """
    if pattern is None:
        pattern = SparsePattern.sliding_window(window_size=512)
    
    replaced = 0
    
    def _replace_attention(module, name):
        nonlocal replaced
        
        for child_name, child in module.named_children():
            # Look for attention modules
            if 'attention' in child_name.lower() or 'attn' in child_name.lower():
                if hasattr(child, 'num_heads') and hasattr(child, 'head_dim'):
                    # Check if we should replace this layer
                    if layer_indices is None or replaced in layer_indices:
                        sparse_config = SparseAttentionConfig(
                            num_heads=child.num_heads,
                            head_dim=child.head_dim,
                            pattern=pattern,
                        )
                        sparse_attn = zSparseAttention(config=sparse_config)
                        setattr(module, child_name, sparse_attn)
                        replaced += 1
            else:
                _replace_attention(child, child_name)
    
    _replace_attention(model, '')
    
    return model


# Convenience functions
def create_sparse_attention(
    pattern_name: str = "sliding_window",
    num_heads: int = 32,
    head_dim: int = 128,
    **pattern_kwargs
) -> zSparseAttention:
    """
    Create sparse attention with named pattern.
    
    Args:
        pattern_name: One of "sliding_window", "longformer", "bigbird", etc.
        num_heads: Number of attention heads
        head_dim: Head dimension
        **pattern_kwargs: Pattern-specific arguments
    
    Returns:
        Configured zSparseAttention module
    """
    from .patterns import create_pattern_from_name
    
    pattern = create_pattern_from_name(pattern_name, **pattern_kwargs)
    config = SparseAttentionConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        pattern=pattern,
    )
    
    return zSparseAttention(config=config)


def benchmark_sparse_attention(
    seq_lengths: list = [1024, 4096, 8192, 16384, 32768],
    window_size: int = 512,
    num_heads: int = 32,
    head_dim: int = 128,
    device: str = "cuda"
) -> dict:
    """
    Benchmark sparse vs full attention.
    
    Returns dict with timing and memory results.
    """
    import time
    
    results = {}
    
    sparse_attn = create_sparse_attention(
        pattern_name="sliding_window",
        num_heads=num_heads,
        head_dim=head_dim,
        window_size=window_size,
    )
    
    for seq_len in seq_lengths:
        # Create inputs
        q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        # Sparse timing
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(10):
            _ = sparse_attn(q, k, v)
            
        if device == "cuda":
            torch.cuda.synchronize()
        sparse_time = (time.perf_counter() - start) / 10
        
        # Memory estimate
        mem_info = sparse_attn.get_memory_estimate(seq_len)
        
        results[seq_len] = {
            "sparse_time_ms": sparse_time * 1000,
            **mem_info,
        }
    
    return results
