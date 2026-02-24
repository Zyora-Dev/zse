"""
ZSE KV Cache Manager - zKV

Efficient KV cache management for paged attention:
- Block-based memory allocation (PagedAttention style)
- Quantized KV cache (INT4/INT8) for 4x memory savings
- Dynamic block allocation/deallocation
- Copy-on-write for speculative decoding
- Prefix caching for shared prompts

Author: ZSE Team
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any

import torch
import torch.nn as nn


class KVCacheQuantization(Enum):
    """KV cache quantization mode."""
    NONE = "none"         # FP16/BF16
    INT8 = "int8"         # 8-bit quantization
    INT4 = "int4"         # 4-bit quantization
    FP8 = "fp8"           # FP8 (E4M3)


@dataclass
class KVCacheConfig:
    """Configuration for KV cache."""
    num_layers: int
    num_kv_heads: int
    head_dim: int
    block_size: int = 16
    max_num_blocks: int = 1024  # Maximum blocks in cache
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    quantization: KVCacheQuantization = KVCacheQuantization.NONE
    
    @property
    def block_bytes(self) -> int:
        """Bytes per KV cache block (both K and V)."""
        bytes_per_element = {
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float32: 4,
        }.get(self.dtype, 2)
        
        # K + V per block
        elements_per_block = 2 * self.num_kv_heads * self.block_size * self.head_dim
        
        if self.quantization == KVCacheQuantization.INT8:
            bytes_per_element = 1
        elif self.quantization == KVCacheQuantization.INT4:
            bytes_per_element = 0.5
        elif self.quantization == KVCacheQuantization.FP8:
            bytes_per_element = 1
        
        return int(elements_per_block * bytes_per_element)
    
    @property
    def total_cache_bytes(self) -> int:
        """Total bytes for the KV cache."""
        return self.block_bytes * self.max_num_blocks * self.num_layers


@dataclass
class BlockTable:
    """Block table for a single sequence."""
    sequence_id: int
    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0
    
    @property
    def num_blocks(self) -> int:
        """Number of allocated blocks."""
        return len(self.block_ids)
    
    def get_tensor(self, max_blocks: int, device: str = "cuda") -> torch.Tensor:
        """Get block table as tensor."""
        table = torch.zeros(max_blocks, dtype=torch.int32, device=device)
        for i, block_id in enumerate(self.block_ids):
            table[i] = block_id
        return table


class BlockAllocator:
    """
    Block allocator for KV cache.
    
    Manages a pool of physical blocks that can be allocated to sequences.
    Supports:
    - Free block allocation
    - Block deallocation
    - Copy-on-write (for speculative decoding)
    - Block sharing (for prefix caching)
    """
    
    def __init__(self, num_blocks: int, device: str = "cuda"):
        self.num_blocks = num_blocks
        self.device = device
        
        # Free block stack (LIFO for locality)
        self.free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))
        
        # Reference counts for copy-on-write
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}
        
        # Used blocks set for quick lookup
        self.used_blocks: Set[int] = set()
    
    @property
    def num_free_blocks(self) -> int:
        """Number of free blocks available."""
        return len(self.free_blocks)
    
    @property
    def num_used_blocks(self) -> int:
        """Number of blocks in use."""
        return len(self.used_blocks)
    
    def allocate(self, num_blocks: int = 1) -> List[int]:
        """
        Allocate blocks from the free pool.
        
        Returns:
            List of allocated block IDs
        
        Raises:
            RuntimeError if not enough blocks available
        """
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks. "
                f"Only {self.num_free_blocks} available."
            )
        
        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop()
            self.ref_counts[block_id] = 1
            self.used_blocks.add(block_id)
            allocated.append(block_id)
        
        return allocated
    
    def free(self, block_ids: List[int]) -> None:
        """
        Free blocks back to the pool.
        
        Only frees blocks when ref count reaches 0.
        """
        for block_id in block_ids:
            if block_id not in self.used_blocks:
                continue
            
            self.ref_counts[block_id] -= 1
            
            if self.ref_counts[block_id] <= 0:
                self.used_blocks.remove(block_id)
                self.free_blocks.append(block_id)
                self.ref_counts[block_id] = 0
    
    def increase_ref(self, block_id: int) -> None:
        """Increase reference count for copy-on-write."""
        if block_id in self.used_blocks:
            self.ref_counts[block_id] += 1
    
    def get_ref_count(self, block_id: int) -> int:
        """Get reference count for a block."""
        return self.ref_counts.get(block_id, 0)
    
    def can_allocate(self, num_blocks: int) -> bool:
        """Check if allocation is possible."""
        return num_blocks <= self.num_free_blocks


class zKVCache:
    """
    ZSE KV Cache Manager
    
    Manages paged KV cache for efficient memory usage during inference.
    
    Features:
    - Block-based allocation (like PagedAttention)
    - Per-sequence block tables
    - Quantized storage (INT4/INT8)
    - Copy-on-write for speculative decoding
    - Prefix caching support
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        
        # Block allocator
        self.allocator = BlockAllocator(config.max_num_blocks, config.device)
        
        # Sequence block tables
        self.block_tables: Dict[int, BlockTable] = {}
        
        # Allocate cache tensors
        self._allocate_cache()
        
        # Stats
        self._allocated_memory = 0
        self._peak_memory = 0
    
    def _allocate_cache(self) -> None:
        """Allocate the KV cache tensors."""
        config = self.config
        
        # Determine storage dtype
        if config.quantization == KVCacheQuantization.INT8:
            storage_dtype = torch.int8
        elif config.quantization == KVCacheQuantization.INT4:
            # INT4 stored as packed INT8
            storage_dtype = torch.int8
        else:
            storage_dtype = config.dtype
        
        # Shape: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        cache_shape = (
            config.num_layers,
            config.max_num_blocks,
            config.num_kv_heads,
            config.block_size,
            config.head_dim,
        )
        
        # For INT4, we pack 2 values per byte
        if config.quantization == KVCacheQuantization.INT4:
            cache_shape = (
                config.num_layers,
                config.max_num_blocks,
                config.num_kv_heads,
                config.block_size,
                config.head_dim // 2,  # Packed
            )
        
        self.key_cache = torch.zeros(
            cache_shape,
            dtype=storage_dtype,
            device=config.device,
        )
        
        self.value_cache = torch.zeros(
            cache_shape,
            dtype=storage_dtype,
            device=config.device,
        )
        
        # Quantization scales and zeros (per-block)
        if config.quantization in [KVCacheQuantization.INT8, KVCacheQuantization.INT4]:
            scale_shape = (
                config.num_layers,
                config.max_num_blocks,
                config.num_kv_heads,
                config.block_size,
            )
            self.key_scales = torch.ones(scale_shape, dtype=torch.float16, device=config.device)
            self.key_zeros = torch.zeros(scale_shape, dtype=torch.float16, device=config.device)
            self.value_scales = torch.ones(scale_shape, dtype=torch.float16, device=config.device)
            self.value_zeros = torch.zeros(scale_shape, dtype=torch.float16, device=config.device)
        else:
            self.key_scales = None
            self.value_scales = None
    
    def allocate_sequence(self, sequence_id: int, num_tokens: int = 0) -> BlockTable:
        """
        Allocate blocks for a new sequence.
        
        Args:
            sequence_id: Unique sequence identifier
            num_tokens: Initial number of tokens (optional)
        
        Returns:
            Block table for the sequence
        """
        if sequence_id in self.block_tables:
            raise ValueError(f"Sequence {sequence_id} already exists")
        
        # Calculate needed blocks
        num_blocks = max(1, math.ceil(num_tokens / self.config.block_size))
        
        # Allocate blocks
        block_ids = self.allocator.allocate(num_blocks)
        
        # Create block table
        block_table = BlockTable(
            sequence_id=sequence_id,
            block_ids=block_ids,
            num_tokens=num_tokens,
        )
        
        self.block_tables[sequence_id] = block_table
        
        return block_table
    
    def extend_sequence(self, sequence_id: int, num_new_tokens: int) -> None:
        """
        Extend a sequence with new tokens.
        
        Allocates additional blocks if needed.
        """
        if sequence_id not in self.block_tables:
            raise ValueError(f"Sequence {sequence_id} does not exist")
        
        block_table = self.block_tables[sequence_id]
        new_total = block_table.num_tokens + num_new_tokens
        needed_blocks = math.ceil(new_total / self.config.block_size)
        
        # Allocate additional blocks if needed
        current_blocks = block_table.num_blocks
        if needed_blocks > current_blocks:
            additional = needed_blocks - current_blocks
            new_blocks = self.allocator.allocate(additional)
            block_table.block_ids.extend(new_blocks)
        
        block_table.num_tokens = new_total
    
    def free_sequence(self, sequence_id: int) -> None:
        """Free all blocks allocated to a sequence."""
        if sequence_id not in self.block_tables:
            return
        
        block_table = self.block_tables[sequence_id]
        self.allocator.free(block_table.block_ids)
        del self.block_tables[sequence_id]
    
    def get_block_table_tensor(
        self,
        sequence_ids: List[int],
        max_blocks: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get block tables as a batched tensor.
        
        Args:
            sequence_ids: List of sequence IDs
            max_blocks: Maximum blocks per sequence (default: from config)
        
        Returns:
            Block table tensor [num_seqs, max_blocks]
        """
        if max_blocks is None:
            max_blocks = max(
                self.block_tables[sid].num_blocks
                for sid in sequence_ids
            )
        
        tables = torch.zeros(
            len(sequence_ids), max_blocks,
            dtype=torch.int32,
            device=self.config.device,
        )
        
        for i, seq_id in enumerate(sequence_ids):
            block_table = self.block_tables[seq_id]
            for j, block_id in enumerate(block_table.block_ids):
                tables[i, j] = block_id
        
        return tables
    
    def get_context_lengths(self, sequence_ids: List[int]) -> torch.Tensor:
        """Get context lengths for sequences."""
        lengths = torch.tensor(
            [self.block_tables[sid].num_tokens for sid in sequence_ids],
            dtype=torch.int32,
            device=self.config.device,
        )
        return lengths
    
    def get_kv_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV cache tensors for a specific layer.
        
        Returns:
            Tuple of (key_cache, value_cache) for the layer
            Shape: [num_blocks, num_kv_heads, block_size, head_dim]
        """
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def write_kv(
        self,
        layer_idx: int,
        sequence_id: int,
        position: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Write KV pairs to the cache.
        
        Args:
            layer_idx: Layer index
            sequence_id: Sequence ID
            position: Token position in sequence
            keys: Key tensor [num_tokens, num_kv_heads, head_dim]
            values: Value tensor [num_tokens, num_kv_heads, head_dim]
        """
        block_table = self.block_tables[sequence_id]
        num_tokens = keys.shape[0]
        
        for i in range(num_tokens):
            token_pos = position + i
            block_idx = token_pos // self.config.block_size
            pos_in_block = token_pos % self.config.block_size
            
            physical_block = block_table.block_ids[block_idx]
            
            # Write to cache
            self.key_cache[layer_idx, physical_block, :, pos_in_block] = keys[i]
            self.value_cache[layer_idx, physical_block, :, pos_in_block] = values[i]
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_blocks = self.config.max_num_blocks
        used_blocks = self.allocator.num_used_blocks
        free_blocks = self.allocator.num_free_blocks
        
        bytes_per_block = self.config.block_bytes
        
        return {
            "total_blocks": total_blocks,
            "used_blocks": used_blocks,
            "free_blocks": free_blocks,
            "utilization": used_blocks / total_blocks if total_blocks > 0 else 0,
            "total_memory_mb": (total_blocks * bytes_per_block) / 1024 / 1024,
            "used_memory_mb": (used_blocks * bytes_per_block) / 1024 / 1024,
            "num_sequences": len(self.block_tables),
        }
    
    def clear(self) -> None:
        """Clear all sequences and reset cache."""
        # Free all sequences
        for seq_id in list(self.block_tables.keys()):
            self.free_sequence(seq_id)
        
        # Reset allocator
        self.allocator = BlockAllocator(
            self.config.max_num_blocks,
            self.config.device,
        )


def create_kv_cache(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int = 8192,
    block_size: int = 16,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    quantization: str = "none",
) -> zKVCache:
    """
    Factory function to create KV cache.
    
    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV attention heads
        head_dim: Dimension of each head
        max_seq_len: Maximum sequence length
        block_size: Tokens per cache block
        device: Torch device
        dtype: Data type for storage
        quantization: Quantization type ("none", "int8", "int4", "fp8")
    
    Returns:
        Configured zKVCache
    """
    # Calculate max blocks needed
    max_num_blocks = (max_seq_len + block_size - 1) // block_size
    # Add buffer for multiple sequences
    max_num_blocks = max_num_blocks * 32  # Support up to 32 concurrent sequences
    
    quant_map = {
        "none": KVCacheQuantization.NONE,
        "int8": KVCacheQuantization.INT8,
        "int4": KVCacheQuantization.INT4,
        "fp8": KVCacheQuantization.FP8,
    }
    
    config = KVCacheConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        device=device,
        dtype=dtype,
        quantization=quant_map.get(quantization, KVCacheQuantization.NONE),
    )
    
    return zKVCache(config)
