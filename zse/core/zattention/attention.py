"""
ZSE Attention Module

Main interface for zAttention - ZSE's custom attention implementation.
Supports:
- Paged attention for memory-efficient decode
- Flash attention for efficient prefill
- Quantized KV cache (INT4/INT8)
- GQA (Grouped Query Attention)
- Sparse attention patterns

Author: ZSE Team
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


class AttentionType(Enum):
    """Type of attention computation."""
    PAGED = "paged"           # For decode phase with KV cache
    FLASH = "flash"           # For prefill phase
    STANDARD = "standard"     # PyTorch fallback


class AttentionBackend(Enum):
    """Attention computation backend."""
    TRITON = "triton"         # Triton kernels
    CUDA = "cuda"             # Custom CUDA kernels
    TORCH = "torch"           # PyTorch fallback


@dataclass
class AttentionConfig:
    """Configuration for attention computation."""
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int = 8192
    block_size: int = 16
    use_alibi: bool = False
    use_rope: bool = True
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = None
    backend: AttentionBackend = AttentionBackend.TRITON
    
    @property
    def gqa_factor(self) -> int:
        """GQA factor (num query heads per KV head)."""
        return self.num_heads // self.num_kv_heads
    
    @property
    def scale(self) -> float:
        """Attention scale factor."""
        return 1.0 / math.sqrt(self.head_dim)


class zAttention(nn.Module):
    """
    ZSE Attention Module
    
    Unified attention interface supporting multiple backends and optimizations.
    Automatically selects the best implementation based on:
    - Phase (prefill vs decode)
    - Available hardware (CUDA, Triton)
    - Context length
    - Memory constraints
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self._backend = config.backend
        self._triton_available = self._check_triton()
        self._cuda_available = self._check_cuda()
        
        # Select best available backend
        if self._backend == AttentionBackend.TRITON and not self._triton_available:
            self._backend = AttentionBackend.TORCH
        elif self._backend == AttentionBackend.CUDA and not self._cuda_available:
            self._backend = AttentionBackend.TRITON if self._triton_available else AttentionBackend.TORCH
    
    def _check_triton(self) -> bool:
        """Check if Triton is available."""
        try:
            from .triton_kernels import TRITON_AVAILABLE
            return TRITON_AVAILABLE
        except ImportError:
            return False
    
    def _check_cuda(self) -> bool:
        """Check if custom CUDA kernels are available."""
        try:
            import zse._C as _C
            return hasattr(_C, 'paged_attention_v1')
        except ImportError:
            return False
    
    @property
    def backend(self) -> AttentionBackend:
        """Current attention backend."""
        return self._backend
    
    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill attention computation.
        
        Args:
            query: [batch, num_heads, seq_len, head_dim]
            key: [batch, num_kv_heads, seq_len, head_dim]
            value: [batch, num_kv_heads, seq_len, head_dim]
            attention_mask: Optional attention mask
        
        Returns:
            Output tensor [batch, num_heads, seq_len, head_dim]
        """
        # Expand KV for GQA if needed
        if self.config.num_kv_heads != self.config.num_heads:
            key = self._expand_kv(key)
            value = self._expand_kv(value)
        
        if self._backend == AttentionBackend.TRITON:
            from .triton_kernels import flash_attention
            return flash_attention(query, key, value, self.config.scale)
        elif self._backend == AttentionBackend.CUDA:
            return self._cuda_flash_attention(query, key, value)
        else:
            return self._torch_attention(query, key, value, attention_mask)
    
    def decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode attention with paged KV cache.
        
        Args:
            query: [num_seqs, num_heads, head_dim]
            key_cache: [num_blocks, num_kv_heads, block_size, head_dim]
            value_cache: [num_blocks, num_kv_heads, block_size, head_dim]
            block_tables: [num_seqs, max_blocks]
            context_lens: [num_seqs]
        
        Returns:
            Output tensor [num_seqs, num_heads, head_dim]
        """
        output = torch.empty_like(query)
        
        if self._backend == AttentionBackend.TRITON:
            from .triton_kernels import paged_attention_v1
            paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                self.config.scale,
                self.config.block_size,
            )
        elif self._backend == AttentionBackend.CUDA:
            self._cuda_paged_attention(
                output, query, key_cache, value_cache, 
                block_tables, context_lens
            )
        else:
            from .triton_kernels import paged_attention_v1_torch
            paged_attention_v1_torch(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                self.config.scale,
                self.config.block_size,
            )
        
        return output
    
    def _expand_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """Expand KV for GQA."""
        batch, num_kv_heads, seq_len, head_dim = kv.shape
        gqa_factor = self.config.gqa_factor
        
        # Expand: [batch, num_kv_heads, seq_len, head_dim] 
        #      -> [batch, num_heads, seq_len, head_dim]
        kv = kv.unsqueeze(2).expand(-1, -1, gqa_factor, -1, -1)
        kv = kv.reshape(batch, num_kv_heads * gqa_factor, seq_len, head_dim)
        
        return kv
    
    def _torch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PyTorch fallback attention."""
        scale = self.config.scale
        
        # Compute attention weights
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply causal mask
        seq_len = query.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and output
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, value)
        
        return output
    
    def _cuda_flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """CUDA flash attention (when available)."""
        try:
            import zse._C as _C
            return _C.flash_attention(query, key, value, self.config.scale)
        except ImportError:
            # Fallback to Triton
            from .triton_kernels import flash_attention
            return flash_attention(query, key, value, self.config.scale)
    
    def _cuda_paged_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> None:
        """CUDA paged attention (when available)."""
        try:
            import zse._C as _C
            _C.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                self.config.scale,
                self.config.block_size,
            )
        except ImportError:
            # Fallback to Triton
            from .triton_kernels import paged_attention_v1
            paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                self.config.scale,
                self.config.block_size,
            )
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_cache: Optional[torch.Tensor] = None,
        value_cache: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with automatic phase detection.
        
        For prefill: provide query, key, value
        For decode: provide query, key_cache, value_cache, block_tables, context_lens
        """
        if is_prefill:
            assert key is not None and value is not None, \
                "key and value required for prefill"
            return self.prefill(query, key, value, attention_mask)
        else:
            assert all(x is not None for x in [key_cache, value_cache, block_tables, context_lens]), \
                "key_cache, value_cache, block_tables, context_lens required for decode"
            return self.decode(query, key_cache, value_cache, block_tables, context_lens)
    
    def get_info(self) -> Dict[str, Any]:
        """Get attention module information."""
        return {
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "head_dim": self.config.head_dim,
            "gqa_factor": self.config.gqa_factor,
            "block_size": self.config.block_size,
            "backend": self._backend.value,
            "triton_available": self._triton_available,
            "cuda_available": self._cuda_available,
        }


def create_attention(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    **kwargs,
) -> zAttention:
    """
    Factory function to create zAttention module.
    
    Args:
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value attention heads (for GQA)
        head_dim: Dimension of each attention head
        **kwargs: Additional AttentionConfig parameters
    
    Returns:
        Configured zAttention module
    """
    config = AttentionConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        **kwargs,
    )
    return zAttention(config)
