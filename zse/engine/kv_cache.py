"""
ZSE KV Cache System

Memory-efficient Key-Value cache for autoregressive generation.

Features:
- Paged KV cache (like vLLM) for memory efficiency
- Dynamic allocation/deallocation
- Multi-sequence support
- Prefix caching for prompt reuse

Memory model:
- Without KV cache: Recompute all tokens every step = O(n²) compute
- With KV cache: Only compute new token = O(n) compute per step

Author: ZSE Team
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math


@dataclass
class KVCacheConfig:
    """Configuration for KV cache."""
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int = 2048
    max_batch_size: int = 32
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    
    # Paged attention config
    page_size: int = 16  # Tokens per page
    max_pages: int = 4096  # Total pages in pool
    
    @property
    def kv_dim(self) -> int:
        """Dimension of K or V tensor per head."""
        return self.head_dim
    
    @property
    def bytes_per_token(self) -> int:
        """Memory per token in KV cache (both K and V)."""
        elem_size = 2 if self.dtype == torch.float16 else 4
        # K + V for all layers and heads
        return 2 * self.num_layers * self.num_heads * self.head_dim * elem_size
    
    def estimate_memory_gb(self, seq_len: int, batch_size: int = 1) -> float:
        """Estimate KV cache memory in GB."""
        total_bytes = self.bytes_per_token * seq_len * batch_size
        return total_bytes / (1024**3)


class KVCache:
    """
    Simple KV Cache for single sequence generation.
    
    Stores key and value tensors for all layers, enabling
    efficient autoregressive generation without recomputation.
    """
    
    def __init__(
        self,
        config: KVCacheConfig,
        batch_size: int = 1,
    ):
        self.config = config
        self.batch_size = batch_size
        self.seq_len = 0  # Current sequence length
        
        # Pre-allocate cache tensors [batch, num_heads, max_seq, head_dim]
        cache_shape = (
            batch_size,
            config.num_heads,
            config.max_seq_len,
            config.head_dim,
        )
        
        # One K and V tensor per layer
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        for _ in range(config.num_layers):
            self.key_cache.append(
                torch.zeros(cache_shape, dtype=config.dtype, device=config.device)
            )
            self.value_cache.append(
                torch.zeros(cache_shape, dtype=config.dtype, device=config.device)
            )
    
    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value and return full cached tensors.
        
        Args:
            layer_idx: Which layer's cache to update
            key: New key tensor [batch, num_heads, new_seq_len, head_dim]
            value: New value tensor [batch, num_heads, new_seq_len, head_dim]
            
        Returns:
            (cached_keys, cached_values) including new tokens
        """
        new_seq_len = key.shape[2]
        
        # Store new KV in cache
        start_pos = self.seq_len
        end_pos = start_pos + new_seq_len
        
        self.key_cache[layer_idx][:, :, start_pos:end_pos, :] = key
        self.value_cache[layer_idx][:, :, start_pos:end_pos, :] = value
        
        # Return full cached KV up to current position
        return (
            self.key_cache[layer_idx][:, :, :end_pos, :],
            self.value_cache[layer_idx][:, :, :end_pos, :],
        )
    
    def advance(self, num_tokens: int = 1):
        """Advance sequence position after generation step."""
        self.seq_len += num_tokens
    
    def reset(self):
        """Reset cache for new sequence."""
        self.seq_len = 0
        # Optionally zero out (not strictly necessary)
        for k, v in zip(self.key_cache, self.value_cache):
            k.zero_()
            v.zero_()
    
    def get_seq_len(self) -> int:
        """Get current cached sequence length."""
        return self.seq_len
    
    def memory_bytes(self) -> int:
        """Calculate current memory usage."""
        total = 0
        for k, v in zip(self.key_cache, self.value_cache):
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total
    
    def memory_gb(self) -> float:
        """Memory usage in GB."""
        return self.memory_bytes() / (1024**3)


# =============================================================================
# PAGED ATTENTION KV CACHE (vLLM-style)
# =============================================================================

@dataclass
class PageInfo:
    """Information about a memory page."""
    page_id: int
    ref_count: int = 0
    is_free: bool = True


class PagedKVCache:
    """
    Paged KV Cache for memory-efficient multi-sequence serving.
    
    Key innovations (from vLLM):
    - Memory is divided into fixed-size pages
    - Pages allocated on-demand per sequence
    - Enables memory sharing (copy-on-write for beam search)
    - Eliminates memory fragmentation
    
    Memory layout:
    - Physical pages: [num_pages, page_size, num_heads, head_dim]
    - Each sequence has a logical-to-physical page mapping
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.page_size = config.page_size
        self.num_pages = config.max_pages
        
        # Physical page pool for K and V [layers, pages, page_size, heads, head_dim]
        page_shape = (
            config.num_layers,
            self.num_pages,
            self.page_size,
            config.num_heads,
            config.head_dim,
        )
        
        self.key_pages = torch.zeros(page_shape, dtype=config.dtype, device=config.device)
        self.value_pages = torch.zeros(page_shape, dtype=config.dtype, device=config.device)
        
        # Page management
        self.free_pages: List[int] = list(range(self.num_pages))
        self.page_info: Dict[int, PageInfo] = {
            i: PageInfo(page_id=i) for i in range(self.num_pages)
        }
        
        # Sequence to page mapping: seq_id -> List[page_ids]
        self.seq_page_tables: Dict[int, List[int]] = {}
        self.seq_lengths: Dict[int, int] = {}
    
    def allocate_sequence(self, seq_id: int, initial_len: int = 0) -> bool:
        """
        Allocate pages for a new sequence.
        
        Args:
            seq_id: Unique sequence identifier
            initial_len: Initial prompt length to allocate
            
        Returns:
            True if allocation successful
        """
        if seq_id in self.seq_page_tables:
            return False  # Already exists
        
        num_pages_needed = math.ceil(initial_len / self.page_size) if initial_len > 0 else 1
        
        if len(self.free_pages) < num_pages_needed:
            return False  # Not enough memory
        
        # Allocate pages
        allocated_pages = []
        for _ in range(num_pages_needed):
            page_id = self.free_pages.pop()
            self.page_info[page_id].is_free = False
            self.page_info[page_id].ref_count = 1
            allocated_pages.append(page_id)
        
        self.seq_page_tables[seq_id] = allocated_pages
        self.seq_lengths[seq_id] = initial_len
        
        return True
    
    def free_sequence(self, seq_id: int):
        """Free all pages for a sequence."""
        if seq_id not in self.seq_page_tables:
            return
        
        for page_id in self.seq_page_tables[seq_id]:
            self.page_info[page_id].ref_count -= 1
            if self.page_info[page_id].ref_count == 0:
                self.page_info[page_id].is_free = True
                self.free_pages.append(page_id)
        
        del self.seq_page_tables[seq_id]
        del self.seq_lengths[seq_id]
    
    def append_token(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> bool:
        """
        Append a new token's KV to the sequence.
        
        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            key: Key tensor [num_heads, head_dim]
            value: Value tensor [num_heads, head_dim]
            
        Returns:
            True if successful
        """
        if seq_id not in self.seq_page_tables:
            return False
        
        seq_len = self.seq_lengths[seq_id]
        page_idx = seq_len // self.page_size
        slot_idx = seq_len % self.page_size
        
        # Check if need new page
        if page_idx >= len(self.seq_page_tables[seq_id]):
            if not self.free_pages:
                return False  # Out of memory
            
            page_id = self.free_pages.pop()
            self.page_info[page_id].is_free = False
            self.page_info[page_id].ref_count = 1
            self.seq_page_tables[seq_id].append(page_id)
        
        # Get physical page
        physical_page = self.seq_page_tables[seq_id][page_idx]
        
        # Store KV
        self.key_pages[layer_idx, physical_page, slot_idx] = key
        self.value_pages[layer_idx, physical_page, slot_idx] = value
        
        # Update length (only on layer 0 to avoid double counting)
        if layer_idx == 0:
            self.seq_lengths[seq_id] = seq_len + 1
        
        return True
    
    def get_kv(
        self,
        seq_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all cached KV for a sequence.
        
        Returns:
            (keys, values) tensors [seq_len, num_heads, head_dim]
        """
        if seq_id not in self.seq_page_tables:
            raise ValueError(f"Sequence {seq_id} not found")
        
        seq_len = self.seq_lengths[seq_id]
        pages = self.seq_page_tables[seq_id]
        
        # Gather KV from pages
        keys = []
        values = []
        
        remaining = seq_len
        for page_id in pages:
            tokens_in_page = min(remaining, self.page_size)
            keys.append(self.key_pages[layer_idx, page_id, :tokens_in_page])
            values.append(self.value_pages[layer_idx, page_id, :tokens_in_page])
            remaining -= tokens_in_page
            if remaining <= 0:
                break
        
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)
    
    def get_page_table(self, seq_id: int) -> List[int]:
        """Get physical page table for a sequence."""
        return self.seq_page_tables.get(seq_id, [])
    
    def copy_sequence(self, src_seq_id: int, dst_seq_id: int) -> bool:
        """
        Copy-on-write: Share pages between sequences.
        
        Used for beam search - initially share pages, copy on modification.
        """
        if src_seq_id not in self.seq_page_tables:
            return False
        
        # Just share the page references (copy-on-write)
        self.seq_page_tables[dst_seq_id] = self.seq_page_tables[src_seq_id].copy()
        self.seq_lengths[dst_seq_id] = self.seq_lengths[src_seq_id]
        
        # Increment ref counts
        for page_id in self.seq_page_tables[dst_seq_id]:
            self.page_info[page_id].ref_count += 1
        
        return True
    
    def num_free_pages(self) -> int:
        """Number of available pages."""
        return len(self.free_pages)
    
    def memory_usage(self) -> Dict[str, float]:
        """Memory usage statistics."""
        total_pages = self.num_pages
        used_pages = total_pages - len(self.free_pages)
        
        bytes_per_page = (
            2 *  # K and V
            self.config.num_layers *
            self.page_size *
            self.config.num_heads *
            self.config.head_dim *
            (2 if self.config.dtype == torch.float16 else 4)
        )
        
        return {
            "total_pages": total_pages,
            "used_pages": used_pages,
            "free_pages": len(self.free_pages),
            "utilization": used_pages / total_pages,
            "total_memory_gb": (total_pages * bytes_per_page) / (1024**3),
            "used_memory_gb": (used_pages * bytes_per_page) / (1024**3),
        }


# =============================================================================
# KV CACHE MANAGER
# =============================================================================

class KVCacheManager:
    """
    High-level KV cache manager for inference.
    
    Provides a simple interface for:
    - Creating/destroying caches
    - Batch operations
    - Memory monitoring
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int = 2048,
        max_batch_size: int = 32,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        use_paged: bool = True,
        page_size: int = 16,
        max_pages: int = 4096,
    ):
        self.config = KVCacheConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            dtype=dtype,
            device=device,
            page_size=page_size,
            max_pages=max_pages,
        )
        
        self.use_paged = use_paged
        
        if use_paged:
            self.paged_cache = PagedKVCache(self.config)
            self.simple_caches: Dict[int, KVCache] = {}
        else:
            self.paged_cache = None
            self.simple_caches: Dict[int, KVCache] = {}
        
        self._next_seq_id = 0
    
    def create_cache(self, batch_size: int = 1, prompt_len: int = 0) -> int:
        """
        Create a new KV cache for a sequence.
        
        Returns:
            Sequence ID for the cache
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        
        if self.use_paged:
            success = self.paged_cache.allocate_sequence(seq_id, prompt_len)
            if not success:
                raise RuntimeError("Failed to allocate paged cache - out of memory")
        else:
            self.simple_caches[seq_id] = KVCache(self.config, batch_size)
        
        return seq_id
    
    def free_cache(self, seq_id: int):
        """Free a KV cache."""
        if self.use_paged:
            self.paged_cache.free_sequence(seq_id)
        else:
            if seq_id in self.simple_caches:
                del self.simple_caches[seq_id]
    
    def update(
        self,
        seq_id: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache and get full cached KV.
        
        Args:
            seq_id: Sequence identifier
            layer_idx: Layer index
            key: New key [batch, heads, seq, dim] or [heads, dim]
            value: New value [batch, heads, seq, dim] or [heads, dim]
            
        Returns:
            (full_keys, full_values) including cached tokens
        """
        if self.use_paged:
            # For paged, handle token-by-token
            if key.ndim == 2:
                # Single token [heads, dim]
                self.paged_cache.append_token(seq_id, layer_idx, key, value)
            else:
                # Multiple tokens
                batch = key.shape[0] if key.ndim == 4 else 1
                seq_len = key.shape[-2]
                for i in range(seq_len):
                    k = key[..., i, :]
                    v = value[..., i, :]
                    if k.ndim == 3:
                        k = k.squeeze(0)
                        v = v.squeeze(0)
                    self.paged_cache.append_token(seq_id, layer_idx, k, v)
            
            # Return full cached KV
            k, v = self.paged_cache.get_kv(seq_id, layer_idx)
            return k.unsqueeze(0), v.unsqueeze(0)  # Add batch dim
        else:
            cache = self.simple_caches[seq_id]
            return cache.update(layer_idx, key, value)
    
    def advance(self, seq_id: int, num_tokens: int = 1):
        """Advance sequence position (for simple cache)."""
        if not self.use_paged and seq_id in self.simple_caches:
            self.simple_caches[seq_id].advance(num_tokens)
    
    def get_seq_len(self, seq_id: int) -> int:
        """Get cached sequence length."""
        if self.use_paged:
            return self.paged_cache.seq_lengths.get(seq_id, 0)
        else:
            return self.simple_caches[seq_id].get_seq_len() if seq_id in self.simple_caches else 0
    
    def memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if self.use_paged:
            return self.paged_cache.memory_usage()
        else:
            total_bytes = sum(c.memory_bytes() for c in self.simple_caches.values())
            return {
                "num_caches": len(self.simple_caches),
                "total_memory_gb": total_bytes / (1024**3),
            }
