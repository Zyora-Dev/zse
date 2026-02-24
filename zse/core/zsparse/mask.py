"""
zSparse Mask - Efficient Sparse Mask Generation

Generates memory-efficient sparse attention masks from patterns.

Key features:
- Block-sparse representation for GPU efficiency
- COO/CSR format for extremely sparse patterns
- Cached mask generation (reuse across batches)
- Dynamic mask updates (for sliding window)

Author: ZSE Team
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from .patterns import SparsePattern


@dataclass
class SparseMask:
    """
    Sparse attention mask.
    
    Supports multiple representations:
    - Dense: Full boolean mask (for debugging/small sequences)
    - Block-sparse: Blocks of attention (GPU-friendly)
    - COO: Coordinate format for very sparse patterns
    """
    
    # Pattern that generated this mask
    pattern: SparsePattern
    
    # Sequence length
    seq_len: int
    
    # Dense mask (optional, for small sequences)
    dense_mask: Optional[torch.Tensor] = None
    
    # Block-sparse representation
    block_mask: Optional[torch.Tensor] = None  # [num_blocks_q, num_blocks_k]
    block_size: int = 64
    
    # COO format (row_indices, col_indices)
    row_indices: Optional[torch.Tensor] = None
    col_indices: Optional[torch.Tensor] = None
    
    # Sliding window parameters (for efficient decode)
    window_start: Optional[torch.Tensor] = None  # [seq_len] start of window
    window_end: Optional[torch.Tensor] = None    # [seq_len] end of window
    
    # Global token indices
    global_indices: Optional[torch.Tensor] = None
    
    @property
    def is_dense(self) -> bool:
        """Check if mask is stored in dense format."""
        return self.dense_mask is not None
    
    @property
    def is_block_sparse(self) -> bool:
        """Check if mask is stored in block-sparse format."""
        return self.block_mask is not None
    
    @property
    def is_coo(self) -> bool:
        """Check if mask is stored in COO format."""
        return self.row_indices is not None and self.col_indices is not None
    
    @property 
    def sparsity(self) -> float:
        """Calculate actual sparsity of the mask."""
        if self.is_dense:
            total = self.dense_mask.numel()
            non_zero = self.dense_mask.sum().item()
            return 1.0 - (non_zero / max(total, 1))
        elif self.is_coo:
            total = self.seq_len * self.seq_len
            if self.pattern.causal:
                total = self.seq_len * (self.seq_len + 1) // 2
            non_zero = len(self.row_indices)
            return 1.0 - (non_zero / max(total, 1))
        elif self.is_block_sparse:
            total = self.block_mask.numel()
            non_zero = self.block_mask.sum().item()
            return 1.0 - (non_zero / max(total, 1))
        return 0.0
    
    def to_dense(self) -> torch.Tensor:
        """Convert to dense mask [seq_len, seq_len]."""
        if self.dense_mask is not None:
            return self.dense_mask
        
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        
        if self.is_coo:
            mask[self.row_indices, self.col_indices] = True
        elif self.is_block_sparse:
            # Expand block mask
            for i in range(self.block_mask.shape[0]):
                for j in range(self.block_mask.shape[1]):
                    if self.block_mask[i, j]:
                        q_start = i * self.block_size
                        q_end = min((i + 1) * self.block_size, self.seq_len)
                        k_start = j * self.block_size
                        k_end = min((j + 1) * self.block_size, self.seq_len)
                        mask[q_start:q_end, k_start:k_end] = True
        
        # Apply causal masking
        if self.pattern.causal:
            causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, dtype=torch.bool))
            mask = mask & causal_mask
        
        return mask
    
    def to(self, device: torch.device) -> "SparseMask":
        """Move mask to device."""
        new_mask = SparseMask(
            pattern=self.pattern,
            seq_len=self.seq_len,
            block_size=self.block_size,
        )
        
        if self.dense_mask is not None:
            new_mask.dense_mask = self.dense_mask.to(device)
        if self.block_mask is not None:
            new_mask.block_mask = self.block_mask.to(device)
        if self.row_indices is not None:
            new_mask.row_indices = self.row_indices.to(device)
            new_mask.col_indices = self.col_indices.to(device)
        if self.window_start is not None:
            new_mask.window_start = self.window_start.to(device)
            new_mask.window_end = self.window_end.to(device)
        if self.global_indices is not None:
            new_mask.global_indices = self.global_indices.to(device)
            
        return new_mask


class SparseMaskGenerator:
    """
    Efficient sparse mask generator.
    
    Features:
    - Caches masks for reuse
    - Supports incremental mask updates
    - GPU acceleration for large masks
    """
    
    def __init__(self, pattern: SparsePattern, device: str = "cpu"):
        self.pattern = pattern
        self.device = torch.device(device)
        self._cache: Dict[int, SparseMask] = {}
    
    def generate(
        self, 
        seq_len: int,
        use_cache: bool = True,
        format: str = "auto"  # "dense", "block", "coo", "auto"
    ) -> SparseMask:
        """
        Generate sparse mask for given sequence length.
        
        Args:
            seq_len: Sequence length
            use_cache: Whether to cache/reuse masks
            format: Output format (auto selects based on sparsity)
        
        Returns:
            SparseMask object
        """
        cache_key = seq_len
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Select format based on sequence length and sparsity
        if format == "auto":
            if seq_len <= 2048:
                format = "dense"
            elif self.pattern.get_sparsity_ratio(seq_len) > 0.9:
                format = "coo"
            else:
                format = "block"
        
        # Generate mask in selected format
        if format == "dense":
            mask = self._generate_dense(seq_len)
        elif format == "block":
            mask = self._generate_block_sparse(seq_len)
        else:
            mask = self._generate_coo(seq_len)
        
        if use_cache:
            self._cache[cache_key] = mask
        
        return mask
    
    def _generate_dense(self, seq_len: int) -> SparseMask:
        """Generate dense mask."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=self.device)
        
        # Sliding window
        if self.pattern.window_size > 0:
            for i in range(seq_len):
                start = max(0, i - self.pattern.window_size + 1) if self.pattern.causal else max(0, i - self.pattern.window_size // 2)
                end = i + 1 if self.pattern.causal else min(seq_len, i + self.pattern.window_size // 2 + 1)
                mask[i, start:end] = True
        
        # Global tokens at start
        if self.pattern.num_global_start > 0:
            # Global tokens attend to all (and all attend to them)
            mask[:, :self.pattern.num_global_start] = True  # All attend to global
            mask[:self.pattern.num_global_start, :] = True  # Global attends to all
        
        # Global tokens at end
        if self.pattern.num_global_end > 0:
            mask[:, -self.pattern.num_global_end:] = True
            mask[-self.pattern.num_global_end:, :] = True
        
        # Explicit global tokens
        for g in self.pattern.global_tokens:
            if g < seq_len:
                mask[:, g] = True
                mask[g, :] = True
        
        # Strided attention
        if self.pattern.stride > 0:
            for i in range(seq_len):
                # Attend to every stride-th token
                stride_indices = torch.arange(
                    self.pattern.stride_offset, 
                    seq_len if not self.pattern.causal else i + 1, 
                    self.pattern.stride, 
                    device=self.device
                )
                mask[i, stride_indices] = True
        
        # Random attention (BigBird-style)
        if self.pattern.num_random > 0:
            generator = torch.Generator(device=self.device)
            if self.pattern.random_seed is not None:
                generator.manual_seed(self.pattern.random_seed)
            
            for i in range(seq_len):
                max_k = i + 1 if self.pattern.causal else seq_len
                num_random = min(self.pattern.num_random, max_k)
                if num_random > 0:
                    random_indices = torch.randperm(max_k, generator=generator, device=self.device)[:num_random]
                    mask[i, random_indices] = True
        
        # Apply causal mask
        if self.pattern.causal:
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device))
            mask = mask & causal
        
        # Compute window bounds for efficient decode
        window_start = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        window_end = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        
        for i in range(seq_len):
            if self.pattern.causal:
                window_start[i] = max(0, i - self.pattern.window_size + 1)
                window_end[i] = i + 1
            else:
                window_start[i] = max(0, i - self.pattern.window_size // 2)
                window_end[i] = min(seq_len, i + self.pattern.window_size // 2 + 1)
        
        # Global token indices
        global_indices = []
        global_indices.extend(range(self.pattern.num_global_start))
        global_indices.extend(self.pattern.global_tokens)
        if self.pattern.num_global_end > 0:
            global_indices.extend(range(seq_len - self.pattern.num_global_end, seq_len))
        global_indices = torch.tensor(sorted(set(global_indices)), dtype=torch.long, device=self.device)
        
        return SparseMask(
            pattern=self.pattern,
            seq_len=seq_len,
            dense_mask=mask,
            window_start=window_start,
            window_end=window_end,
            global_indices=global_indices if len(global_indices) > 0 else None,
        )
    
    def _generate_block_sparse(self, seq_len: int) -> SparseMask:
        """Generate block-sparse mask."""
        block_size = self.pattern.block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        
        block_mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device=self.device)
        
        # Local blocks (diagonal + off-diagonals based on window)
        blocks_in_window = (self.pattern.window_size + block_size - 1) // block_size
        
        for i in range(num_blocks):
            if self.pattern.causal:
                # Attend to current and previous blocks within window
                start = max(0, i - blocks_in_window + 1)
                block_mask[i, start:i+1] = True
            else:
                # Symmetric window
                start = max(0, i - blocks_in_window // 2)
                end = min(num_blocks, i + blocks_in_window // 2 + 1)
                block_mask[i, start:end] = True
        
        # Global blocks
        global_blocks = set()
        
        # First N blocks
        for i in range((self.pattern.num_global_start + block_size - 1) // block_size):
            global_blocks.add(i)
        
        # Last N blocks
        for i in range(num_blocks - (self.pattern.num_global_end + block_size - 1) // block_size, num_blocks):
            global_blocks.add(i)
        
        # Explicit global tokens -> blocks
        for g in self.pattern.global_tokens:
            global_blocks.add(g // block_size)
        
        # Global blocks attend to all
        for g in global_blocks:
            if g < num_blocks:
                block_mask[:, g] = True  # All attend to global
                block_mask[g, :] = True  # Global attends to all
        
        # Apply causal at block level
        if self.pattern.causal:
            causal = torch.tril(torch.ones(num_blocks, num_blocks, dtype=torch.bool, device=self.device))
            block_mask = block_mask & causal
        
        # Block stride (attend to every N-th block)
        if self.pattern.block_stride > 0:
            for i in range(num_blocks):
                for j in range(0, num_blocks, self.pattern.block_stride):
                    if not self.pattern.causal or j <= i:
                        block_mask[i, j] = True
        
        return SparseMask(
            pattern=self.pattern,
            seq_len=seq_len,
            block_mask=block_mask,
            block_size=block_size,
        )
    
    def _generate_coo(self, seq_len: int) -> SparseMask:
        """Generate COO format mask for very sparse patterns."""
        # First generate dense, then convert (for simplicity)
        # In production, would generate COO directly
        dense_mask = self._generate_dense(seq_len)
        dense = dense_mask.dense_mask
        
        # Convert to COO
        indices = torch.nonzero(dense, as_tuple=False)
        row_indices = indices[:, 0]
        col_indices = indices[:, 1]
        
        return SparseMask(
            pattern=self.pattern,
            seq_len=seq_len,
            row_indices=row_indices,
            col_indices=col_indices,
            window_start=dense_mask.window_start,
            window_end=dense_mask.window_end,
            global_indices=dense_mask.global_indices,
        )
    
    def clear_cache(self):
        """Clear cached masks."""
        self._cache.clear()
    
    def update_for_decode(
        self, 
        mask: SparseMask, 
        new_seq_len: int
    ) -> SparseMask:
        """
        Incrementally update mask for decode phase.
        
        Instead of regenerating, extends existing mask efficiently.
        """
        if new_seq_len <= mask.seq_len:
            return mask
        
        # For now, regenerate (could be optimized)
        return self.generate(new_seq_len, use_cache=False)


def create_causal_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device = None
) -> torch.Tensor:
    """
    Fast utility to create sliding window causal mask.
    
    Returns: [seq_len, seq_len] boolean mask
    """
    if device is None:
        device = torch.device('cpu')
    
    # Create position indices
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Causal: k <= q
    causal_mask = k_pos <= q_pos
    
    # Window: q - k < window_size
    window_mask = (q_pos - k_pos) < window_size
    
    return causal_mask & window_mask


def visualize_mask(mask: Union[SparseMask, torch.Tensor], max_size: int = 64) -> str:
    """
    Create ASCII visualization of sparse mask.
    
    Returns string representation for debugging.
    """
    if isinstance(mask, SparseMask):
        dense = mask.to_dense()
    else:
        dense = mask
    
    seq_len = dense.shape[0]
    
    # Downsample if too large
    if seq_len > max_size:
        step = seq_len // max_size
        dense = dense[::step, ::step]
        seq_len = dense.shape[0]
    
    lines = []
    lines.append(f"Mask visualization ({seq_len}x{seq_len}):")
    lines.append("  " + "".join(f"{i%10}" for i in range(min(seq_len, 80))))
    
    for i in range(min(seq_len, 40)):
        row = f"{i:2d}|"
        for j in range(min(seq_len, 80)):
            row += "█" if dense[i, j] else "·"
        lines.append(row)
    
    if seq_len > 40:
        lines.append("...")
    
    return "\n".join(lines)
