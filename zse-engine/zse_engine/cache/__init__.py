"""ZSE KV Cache — Paged KV cache manager for LLM inference.

Our PagedAttention implementation with advantages over vLLM:
- Adaptive block sizing (not fixed 16)
- Token-level eviction (not sequence-level)
- Smart eviction (frequency + recency + recompute cost, not LRU)
- Block deduplication for shared prefixes
- GPU-only (no CPU swap)
"""

from zse_engine.cache.block_pool import BlockPool, Block
from zse_engine.cache.page_table import PageTable, SequenceEntry
from zse_engine.cache.evictor import Evictor, EvictionCandidate
from zse_engine.cache.dedup import BlockDedup
from zse_engine.cache.cache_manager import KVCacheManager, KVCacheHandle, CacheStats
from zse_engine.cache.attention_metadata import AttentionMetadata, build_attention_metadata

__all__ = [
    "BlockPool", "Block",
    "PageTable", "SequenceEntry",
    "Evictor", "EvictionCandidate",
    "BlockDedup",
    "KVCacheManager", "KVCacheHandle", "CacheStats",
    "AttentionMetadata", "build_attention_metadata",
]
