"""
ZSE Prefix Caching - RadixCache

Efficient prefix caching using a radix tree (trie) structure.
Automatically detects and reuses KV cache for shared prompt prefixes.

Key benefits:
- System prompts cached once, reused across all requests
- Multi-turn chat shares conversation history KV cache
- Batch requests with common prefixes share computation

How it works:
1. Hash token sequences to create prefix keys
2. Store KV cache blocks in radix tree nodes
3. On new request, find longest matching prefix
4. Reuse cached KV blocks, only compute new tokens

Author: ZSE Team
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import OrderedDict

import torch


@dataclass
class RadixNode:
    """
    Node in the radix tree for prefix caching.
    
    Each node represents a token sequence and holds
    references to cached KV blocks.
    """
    # Token sequence for this node (edge label)
    tokens: Tuple[int, ...] = field(default_factory=tuple)
    
    # KV cache block IDs for this prefix
    block_ids: List[int] = field(default_factory=list)
    
    # Number of tokens represented by blocks
    num_cached_tokens: int = 0
    
    # Children nodes (next token -> child node)
    children: Dict[int, 'RadixNode'] = field(default_factory=dict)
    
    # Parent reference for traversal
    parent: Optional['RadixNode'] = None
    
    # Reference count (how many sequences using this prefix)
    ref_count: int = 0
    
    # Last access time for LRU eviction
    last_access: float = field(default_factory=time.time)
    
    # Is this a complete prefix (end of a cached sequence)?
    is_complete: bool = False
    
    def __hash__(self):
        return hash(self.tokens)


@dataclass  
class PrefixMatch:
    """Result of prefix matching."""
    # Matched prefix length in tokens
    matched_length: int
    
    # Block IDs for the matched prefix
    block_ids: List[int]
    
    # The matched node
    node: Optional[RadixNode]
    
    # Remaining tokens to process
    remaining_tokens: List[int]


class RadixCache:
    """
    Radix tree-based prefix cache for KV cache reuse.
    
    Implements automatic prefix detection and caching:
    - O(n) prefix matching where n = prefix length
    - LRU eviction when cache is full
    - Thread-safe for concurrent access
    
    Usage:
        cache = RadixCache(block_allocator, max_cached_tokens=100000)
        
        # Try to find cached prefix
        match = cache.match_prefix(token_ids)
        if match.matched_length > 0:
            # Reuse cached KV blocks
            reuse_blocks(match.block_ids)
        
        # After computing new KV, cache it
        cache.insert_prefix(token_ids, block_ids)
    """
    
    def __init__(
        self,
        block_allocator: Any,  # BlockAllocator from cache.py
        max_cached_tokens: int = 100000,
        block_size: int = 16,
        enable_eviction: bool = True,
    ):
        """
        Initialize RadixCache.
        
        Args:
            block_allocator: Block allocator for KV cache blocks
            max_cached_tokens: Maximum tokens to cache
            block_size: Tokens per KV cache block
            enable_eviction: Enable LRU eviction when full
        """
        self.block_allocator = block_allocator
        self.max_cached_tokens = max_cached_tokens
        self.block_size = block_size
        self.enable_eviction = enable_eviction
        
        # Root node of the radix tree
        self.root = RadixNode(tokens=())
        
        # Total cached tokens
        self.cached_tokens = 0
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "insertions": 0,
            "evictions": 0,
            "total_tokens_matched": 0,
        }
    
    def match_prefix(self, token_ids: List[int]) -> PrefixMatch:
        """
        Find the longest cached prefix for given tokens.
        
        Args:
            token_ids: Input token sequence
            
        Returns:
            PrefixMatch with matched prefix info
        """
        with self._lock:
            if not token_ids:
                return PrefixMatch(
                    matched_length=0,
                    block_ids=[],
                    node=None,
                    remaining_tokens=[],
                )
            
            # Traverse the tree to find longest match
            current = self.root
            matched_length = 0
            matched_blocks: List[int] = []
            last_matched_node = None
            
            i = 0
            while i < len(token_ids):
                token = token_ids[i]
                
                if token not in current.children:
                    # No match, stop here
                    break
                
                child = current.children[token]
                
                # Check if child's tokens match
                child_tokens = child.tokens
                match_len = 0
                
                for j, ct in enumerate(child_tokens):
                    if i + j >= len(token_ids):
                        break
                    if token_ids[i + j] != ct:
                        break
                    match_len += 1
                
                if match_len == 0:
                    break
                
                # Partial or full match
                if match_len == len(child_tokens):
                    # Full match of this node
                    matched_length += match_len
                    matched_blocks.extend(child.block_ids)
                    last_matched_node = child
                    child.last_access = time.time()
                    current = child
                    i += match_len
                else:
                    # Partial match - we matched some but not all
                    # This is still useful, but we stop here
                    # (could split node for partial caching)
                    break
            
            if matched_length > 0:
                self.stats["hits"] += 1
                self.stats["total_tokens_matched"] += matched_length
            else:
                self.stats["misses"] += 1
            
            return PrefixMatch(
                matched_length=matched_length,
                block_ids=matched_blocks.copy(),
                node=last_matched_node,
                remaining_tokens=list(token_ids[matched_length:]),
            )
    
    def insert_prefix(
        self,
        token_ids: List[int],
        block_ids: List[int],
        num_tokens: Optional[int] = None,
    ) -> bool:
        """
        Insert a prefix into the cache.
        
        Args:
            token_ids: Token sequence
            block_ids: KV cache block IDs for this prefix
            num_tokens: Number of tokens (defaults to len(token_ids))
            
        Returns:
            True if inserted successfully
        """
        with self._lock:
            if not token_ids:
                return False
            
            num_tokens = num_tokens or len(token_ids)
            
            # Check if we need to evict
            if self.enable_eviction:
                while (self.cached_tokens + num_tokens > self.max_cached_tokens 
                       and self.cached_tokens > 0):
                    if not self._evict_lru():
                        break
            
            # Navigate/create path to this prefix
            current = self.root
            i = 0
            
            while i < len(token_ids):
                token = token_ids[i]
                
                if token not in current.children:
                    # Create new node for remaining tokens
                    remaining = tuple(token_ids[i:])
                    new_node = RadixNode(
                        tokens=remaining,
                        block_ids=block_ids.copy(),
                        num_cached_tokens=num_tokens - i,
                        parent=current,
                        ref_count=1,
                        is_complete=True,
                    )
                    current.children[token] = new_node
                    self.cached_tokens += len(remaining)
                    self.stats["insertions"] += 1
                    
                    # Increment ref counts on blocks
                    for bid in block_ids:
                        self.block_allocator.inc_ref(bid)
                    
                    return True
                
                child = current.children[token]
                child_tokens = child.tokens
                
                # Find common prefix length
                common_len = 0
                for j in range(min(len(child_tokens), len(token_ids) - i)):
                    if child_tokens[j] == token_ids[i + j]:
                        common_len += 1
                    else:
                        break
                
                if common_len == len(child_tokens):
                    # Full match, continue to children
                    i += common_len
                    current = child
                else:
                    # Need to split the node
                    # Create intermediate node for common prefix
                    common_tokens = child_tokens[:common_len]
                    remaining_child = child_tokens[common_len:]
                    remaining_new = tuple(token_ids[i + common_len:])
                    
                    # Split blocks proportionally
                    blocks_for_common = common_len // self.block_size
                    
                    # Create split node
                    split_node = RadixNode(
                        tokens=common_tokens,
                        block_ids=child.block_ids[:blocks_for_common],
                        num_cached_tokens=common_len,
                        parent=current,
                        ref_count=child.ref_count + 1,
                    )
                    
                    # Update original child
                    child.tokens = remaining_child
                    child.block_ids = child.block_ids[blocks_for_common:]
                    child.parent = split_node
                    
                    # Add child to split node
                    if remaining_child:
                        split_node.children[remaining_child[0]] = child
                    
                    # Create new node for remaining new tokens
                    if remaining_new:
                        new_node = RadixNode(
                            tokens=remaining_new,
                            block_ids=block_ids.copy(),
                            num_cached_tokens=len(remaining_new),
                            parent=split_node,
                            ref_count=1,
                            is_complete=True,
                        )
                        split_node.children[remaining_new[0]] = new_node
                        
                        for bid in block_ids:
                            self.block_allocator.inc_ref(bid)
                    
                    # Replace in parent
                    current.children[token] = split_node
                    
                    self.cached_tokens += len(remaining_new) if remaining_new else 0
                    self.stats["insertions"] += 1
                    return True
            
            # Reached end of token_ids, mark current as complete
            current.is_complete = True
            current.block_ids = block_ids.copy()
            current.num_cached_tokens = num_tokens
            current.ref_count += 1
            self.stats["insertions"] += 1
            
            for bid in block_ids:
                self.block_allocator.inc_ref(bid)
            
            return True
    
    def _evict_lru(self) -> bool:
        """
        Evict least recently used prefix.
        
        Returns:
            True if eviction was successful
        """
        # Find LRU leaf node
        lru_node = None
        lru_time = float('inf')
        
        def find_lru(node: RadixNode):
            nonlocal lru_node, lru_time
            
            if node.is_complete and node.ref_count <= 1:
                if node.last_access < lru_time:
                    lru_time = node.last_access
                    lru_node = node
            
            for child in node.children.values():
                find_lru(child)
        
        find_lru(self.root)
        
        if lru_node is None:
            return False
        
        # Evict the node
        self._remove_node(lru_node)
        self.stats["evictions"] += 1
        return True
    
    def _remove_node(self, node: RadixNode) -> None:
        """Remove a node and free its blocks."""
        # Free blocks
        for bid in node.block_ids:
            self.block_allocator.free(bid)
        
        # Update token count
        self.cached_tokens -= len(node.tokens)
        
        # Remove from parent
        if node.parent:
            for token, child in list(node.parent.children.items()):
                if child is node:
                    del node.parent.children[token]
                    break
    
    def release_prefix(self, token_ids: List[int]) -> None:
        """
        Release a prefix (decrement ref count).
        
        Called when a sequence using this prefix completes.
        """
        with self._lock:
            match = self.match_prefix(token_ids)
            if match.node:
                match.node.ref_count = max(0, match.node.ref_count - 1)
    
    def clear(self) -> None:
        """Clear all cached prefixes."""
        with self._lock:
            def clear_node(node: RadixNode):
                for bid in node.block_ids:
                    self.block_allocator.free(bid)
                for child in node.children.values():
                    clear_node(child)
            
            clear_node(self.root)
            self.root = RadixNode(tokens=())
            self.cached_tokens = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = (
                self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
                if (self.stats["hits"] + self.stats["misses"]) > 0
                else 0.0
            )
            
            return {
                **self.stats,
                "hit_rate": hit_rate,
                "cached_tokens": self.cached_tokens,
                "max_cached_tokens": self.max_cached_tokens,
                "utilization": self.cached_tokens / self.max_cached_tokens if self.max_cached_tokens > 0 else 0,
            }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"RadixCache(cached_tokens={stats['cached_tokens']}, "
            f"hit_rate={stats['hit_rate']:.2%}, "
            f"utilization={stats['utilization']:.2%})"
        )


class PrefixCacheManager:
    """
    High-level manager for prefix caching.
    
    Integrates RadixCache with the KV cache system.
    
    Usage:
        manager = PrefixCacheManager(kv_cache)
        
        # Check for cached prefix before inference
        prefix_info = manager.check_prefix(token_ids)
        if prefix_info.cached:
            # Start from cached position
            start_pos = prefix_info.matched_length
        
        # After inference, cache the result
        manager.cache_sequence(seq_id, token_ids, block_ids)
    """
    
    def __init__(
        self,
        kv_cache: Any,  # zKVCache or PagedKVCache
        max_cached_tokens: int = 100000,
        enable: bool = True,
    ):
        """
        Initialize prefix cache manager.
        
        Args:
            kv_cache: KV cache instance
            max_cached_tokens: Maximum tokens to cache
            enable: Enable/disable prefix caching
        """
        self.kv_cache = kv_cache
        self.enabled = enable
        
        if enable and hasattr(kv_cache, 'allocator'):
            self.radix_cache = RadixCache(
                block_allocator=kv_cache.allocator,
                max_cached_tokens=max_cached_tokens,
                block_size=getattr(kv_cache.config, 'block_size', 16),
            )
        else:
            self.radix_cache = None
            self.enabled = False
    
    def check_prefix(self, token_ids: List[int]) -> PrefixMatch:
        """
        Check if there's a cached prefix for the given tokens.
        
        Args:
            token_ids: Input token sequence
            
        Returns:
            PrefixMatch with cache hit information
        """
        if not self.enabled or self.radix_cache is None:
            return PrefixMatch(
                matched_length=0,
                block_ids=[],
                node=None,
                remaining_tokens=list(token_ids),
            )
        
        return self.radix_cache.match_prefix(token_ids)
    
    def cache_sequence(
        self,
        token_ids: List[int],
        block_ids: List[int],
    ) -> bool:
        """
        Cache a sequence's KV blocks for prefix reuse.
        
        Args:
            token_ids: Token sequence
            block_ids: KV cache block IDs
            
        Returns:
            True if cached successfully
        """
        if not self.enabled or self.radix_cache is None:
            return False
        
        return self.radix_cache.insert_prefix(token_ids, block_ids)
    
    def release(self, token_ids: List[int]) -> None:
        """Release a cached prefix."""
        if self.enabled and self.radix_cache:
            self.radix_cache.release_prefix(token_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prefix cache statistics."""
        if not self.enabled or self.radix_cache is None:
            return {"enabled": False}
        
        stats = self.radix_cache.get_stats()
        stats["enabled"] = True
        return stats


# =============================================================================
# HASH-BASED PREFIX CACHE (Alternative simpler implementation)
# =============================================================================

class HashPrefixCache:
    """
    Simpler hash-based prefix cache.
    
    Uses token hash as key for O(1) lookup.
    Less flexible than RadixCache but faster for exact matches.
    """
    
    def __init__(
        self,
        max_entries: int = 10000,
        hash_prefix_length: int = 64,  # Hash first N tokens
    ):
        self.max_entries = max_entries
        self.hash_prefix_length = hash_prefix_length
        
        # LRU cache: hash -> (block_ids, num_tokens, last_access)
        self._cache: OrderedDict[str, Tuple[List[int], int, float]] = OrderedDict()
        self._lock = threading.RLock()
        
        self.stats = {"hits": 0, "misses": 0}
    
    def _hash_tokens(self, tokens: List[int]) -> str:
        """Create hash key from token prefix."""
        prefix = tokens[:self.hash_prefix_length]
        token_bytes = bytes(str(prefix), 'utf-8')
        return hashlib.sha256(token_bytes).hexdigest()[:16]
    
    def get(self, token_ids: List[int]) -> Optional[Tuple[List[int], int]]:
        """
        Get cached blocks for token prefix.
        
        Returns:
            (block_ids, num_cached_tokens) or None if not found
        """
        with self._lock:
            key = self._hash_tokens(token_ids)
            
            if key in self._cache:
                block_ids, num_tokens, _ = self._cache[key]
                # Update access time and move to end (LRU)
                self._cache[key] = (block_ids, num_tokens, time.time())
                self._cache.move_to_end(key)
                self.stats["hits"] += 1
                return (block_ids, num_tokens)
            
            self.stats["misses"] += 1
            return None
    
    def put(self, token_ids: List[int], block_ids: List[int]) -> None:
        """Cache blocks for token prefix."""
        with self._lock:
            key = self._hash_tokens(token_ids)
            
            # Evict if full
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)  # Remove oldest
            
            self._cache[key] = (block_ids.copy(), len(token_ids), time.time())
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
