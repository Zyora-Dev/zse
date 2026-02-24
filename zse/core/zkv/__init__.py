"""
zKV - KV Cache Management

Implements memory-efficient KV cache:
- Paged allocation (PagedAttention-style blocks)
- Quantized KV cache (FP8/INT4)
- Sliding window precision (recent=FP16, old=INT4)
- Copy-on-write for beam search / parallel sampling
- Dynamic growth (no pre-allocation waste)

Memory Savings:
- Standard FP16 KV: ~2GB for 7B @ 4K context
- zKV INT4: ~0.5GB for 7B @ 4K context (4x reduction)
"""

from .cache import (
    zKVCache,
    KVCacheConfig,
    KVCacheQuantization,
    BlockAllocator,
    BlockTable,
    create_kv_cache,
)

from .radix_cache import (
    RadixCache,
    RadixNode,
    PrefixMatch,
    PrefixCacheManager,
    HashPrefixCache,
)

__all__ = [
    "zKVCache",
    "KVCacheConfig",
    "KVCacheQuantization",
    "BlockAllocator",
    "BlockTable",
    "create_kv_cache",
    # Prefix caching
    "RadixCache",
    "RadixNode",
    "PrefixMatch",
    "PrefixCacheManager",
    "HashPrefixCache",
]
