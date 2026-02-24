"""
zSparse Patterns - Sparse Attention Pattern Definitions

Implements various sparse attention patterns for efficient long-context processing:

1. **Sliding Window**: Local attention within a fixed window size
2. **Global Tokens**: Certain positions attend to/from all tokens
3. **Strided**: Attend to every Nth token for global context
4. **Block Sparse**: Attend within blocks + between selected blocks
5. **Longformer-style**: Sliding window + global tokens
6. **BigBird-style**: Local + global + random patterns

Memory Complexity:
- Full attention: O(n²)
- Sliding window (w): O(n × w)
- Strided (s): O(n × n/s) = O(n²/s)
- Combined: O(n × (w + g + n/s)) where g = global tokens

Author: ZSE Team
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple, Union
import torch


class PatternType(Enum):
    """Types of sparse attention patterns."""
    SLIDING_WINDOW = "sliding_window"
    GLOBAL_TOKENS = "global_tokens"
    STRIDED = "strided"
    BLOCK_SPARSE = "block_sparse"
    LONGFORMER = "longformer"
    BIGBIRD = "bigbird"
    CUSTOM = "custom"


@dataclass
class SparsePattern:
    """
    Base sparse attention pattern configuration.
    
    Combines multiple sparsity patterns to define which query-key pairs
    should have non-zero attention weights.
    """
    # Sliding window (local attention)
    window_size: int = 256
    window_symmetric: bool = True  # Attend both directions
    
    # Global tokens (attend to/from all positions)
    global_tokens: List[int] = field(default_factory=list)
    num_global_start: int = 0  # First N tokens are global
    num_global_end: int = 0    # Last N tokens are global
    
    # Strided/dilated attention
    stride: int = 0  # 0 = disabled, N = attend every Nth token
    stride_offset: int = 0
    
    # Block sparse
    block_size: int = 64
    block_stride: int = 0  # Blocks attend to every Nth block
    
    # Random attention (for BigBird)
    num_random: int = 0  # Number of random tokens per query
    random_seed: Optional[int] = None
    
    # Causal masking
    causal: bool = True
    
    # Maximum sequence length (for precomputation)
    max_seq_len: int = 32768
    
    @classmethod
    def sliding_window(
        cls, 
        window_size: int = 512, 
        causal: bool = True,
        max_seq_len: int = 32768
    ) -> "SparsePattern":
        """Create sliding window attention pattern."""
        return cls(
            window_size=window_size,
            causal=causal,
            max_seq_len=max_seq_len,
        )
    
    @classmethod
    def longformer(
        cls,
        window_size: int = 512,
        num_global_start: int = 1,  # [CLS] or system prompt
        global_tokens: Optional[List[int]] = None,
        causal: bool = True,
        max_seq_len: int = 32768
    ) -> "SparsePattern":
        """
        Create Longformer-style pattern: sliding window + global tokens.
        
        Good for: Document understanding, QA, summarization
        """
        return cls(
            window_size=window_size,
            num_global_start=num_global_start,
            global_tokens=global_tokens or [],
            causal=causal,
            max_seq_len=max_seq_len,
        )
    
    @classmethod
    def bigbird(
        cls,
        window_size: int = 256,
        num_global_start: int = 1,
        num_random: int = 64,
        random_seed: int = 42,
        causal: bool = True,
        max_seq_len: int = 32768
    ) -> "SparsePattern":
        """
        Create BigBird-style pattern: local + global + random.
        
        Good for: Long document understanding, reasoning
        """
        return cls(
            window_size=window_size,
            num_global_start=num_global_start,
            num_random=num_random,
            random_seed=random_seed,
            causal=causal,
            max_seq_len=max_seq_len,
        )
    
    @classmethod
    def strided_local(
        cls,
        window_size: int = 256,
        stride: int = 512,
        causal: bool = True,
        max_seq_len: int = 32768
    ) -> "SparsePattern":
        """
        Create strided + local pattern.
        
        Combines local window with dilated global attention.
        Good for: Capturing both local and long-range dependencies
        """
        return cls(
            window_size=window_size,
            stride=stride,
            causal=causal,
            max_seq_len=max_seq_len,
        )
    
    @classmethod  
    def block_sparse(
        cls,
        block_size: int = 64,
        block_stride: int = 4,  # Attend to every 4th block
        causal: bool = True,
        max_seq_len: int = 32768
    ) -> "SparsePattern":
        """
        Create block sparse pattern.
        
        Efficient for GPU tensor cores.
        """
        return cls(
            window_size=block_size,  # Local within block
            block_size=block_size,
            block_stride=block_stride,
            causal=causal,
            max_seq_len=max_seq_len,
        )
    
    def get_sparsity_ratio(self, seq_len: int) -> float:
        """
        Calculate sparsity ratio for given sequence length.
        
        Returns: Fraction of attention weights that are zero (0.0 to 1.0)
        """
        total_pairs = seq_len * seq_len
        if self.causal:
            total_pairs = seq_len * (seq_len + 1) // 2
        
        # Estimate non-zero pairs
        non_zero = 0
        
        # Sliding window
        if self.window_size > 0:
            if self.causal:
                # Each position attends to min(window_size, pos+1) previous tokens
                non_zero += sum(min(self.window_size, i + 1) for i in range(seq_len))
            else:
                non_zero += seq_len * min(self.window_size, seq_len)
        
        # Global tokens
        num_global = self.num_global_start + self.num_global_end + len(self.global_tokens)
        if num_global > 0:
            non_zero += num_global * seq_len * 2  # bidirectional
        
        # Strided
        if self.stride > 0:
            non_zero += seq_len * (seq_len // self.stride)
        
        # Random
        non_zero += seq_len * self.num_random
        
        # Clamp and avoid double counting
        non_zero = min(non_zero, total_pairs)
        
        return 1.0 - (non_zero / max(total_pairs, 1))
    
    def memory_savings(self, seq_len: int) -> str:
        """Human-readable memory savings estimate."""
        sparsity = self.get_sparsity_ratio(seq_len)
        full_mem_mb = (seq_len * seq_len * 2) / (1024 * 1024)  # FP16
        sparse_mem_mb = full_mem_mb * (1 - sparsity)
        
        return (
            f"Full: {full_mem_mb:.1f} MB → Sparse: {sparse_mem_mb:.1f} MB "
            f"({sparsity*100:.1f}% sparse, {1/(1-sparsity+1e-6):.1f}x reduction)"
        )


@dataclass
class PatternConfig:
    """
    Complete sparse pattern configuration for a model.
    
    Can specify different patterns for different layers.
    """
    # Default pattern for all layers
    default_pattern: SparsePattern = field(default_factory=SparsePattern)
    
    # Layer-specific patterns (overrides default)
    layer_patterns: dict = field(default_factory=dict)  # {layer_idx: SparsePattern}
    
    # Pattern scheduling (change pattern during generation)
    prefill_pattern: Optional[SparsePattern] = None   # Dense during prefill
    decode_pattern: Optional[SparsePattern] = None    # Sparse during decode
    
    def get_pattern(self, layer_idx: int, is_prefill: bool = True) -> SparsePattern:
        """Get pattern for specific layer and phase."""
        if is_prefill and self.prefill_pattern is not None:
            return self.prefill_pattern
        if not is_prefill and self.decode_pattern is not None:
            return self.decode_pattern
        
        return self.layer_patterns.get(layer_idx, self.default_pattern)
    
    @classmethod
    def uniform(cls, pattern: SparsePattern) -> "PatternConfig":
        """Same pattern for all layers."""
        return cls(default_pattern=pattern)
    
    @classmethod
    def alternating(
        cls, 
        pattern_a: SparsePattern, 
        pattern_b: SparsePattern,
        num_layers: int
    ) -> "PatternConfig":
        """Alternate between two patterns."""
        layer_patterns = {
            i: pattern_a if i % 2 == 0 else pattern_b
            for i in range(num_layers)
        }
        return cls(
            default_pattern=pattern_a,
            layer_patterns=layer_patterns
        )
    
    @classmethod
    def dense_sparse(
        cls,
        dense_layers: int = 4,  # First N layers use full attention
        sparse_pattern: SparsePattern = None,
        num_layers: int = 32
    ) -> "PatternConfig":
        """
        First layers use dense attention, rest use sparse.
        
        Recommended for best quality with long contexts.
        """
        if sparse_pattern is None:
            sparse_pattern = SparsePattern.longformer(window_size=512)
        
        dense = SparsePattern(window_size=10000000, causal=True)  # Effectively full
        
        layer_patterns = {
            i: dense if i < dense_layers else sparse_pattern
            for i in range(num_layers)
        }
        return cls(
            default_pattern=sparse_pattern,
            layer_patterns=layer_patterns
        )


def create_pattern_from_name(name: str, **kwargs) -> SparsePattern:
    """Create pattern from string name."""
    patterns = {
        "sliding_window": SparsePattern.sliding_window,
        "longformer": SparsePattern.longformer,
        "bigbird": SparsePattern.bigbird,
        "strided_local": SparsePattern.strided_local,
        "block_sparse": SparsePattern.block_sparse,
    }
    
    if name not in patterns:
        raise ValueError(f"Unknown pattern: {name}. Available: {list(patterns.keys())}")
    
    return patterns[name](**kwargs)
