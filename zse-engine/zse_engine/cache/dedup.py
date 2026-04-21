"""ZSE Block Dedup — Shared prefix optimization via content hashing.

If two sequences share the same prompt prefix, they can share KV cache blocks.
We hash the token IDs in each block and check for matches, with collision
verification to prevent silent data corruption.

Uses FNV-1a hash for fast O(1) lookup + token equality check on match.
"""

from typing import Dict, Optional, List, Tuple

from zse_engine.cache.block_pool import BlockPool, Block


def _fnv1a_hash(token_ids: List[int]) -> int:
    """FNV-1a hash of a token ID sequence. Fast, good distribution."""
    h = 0xcbf29ce484222325  # FNV offset basis (64-bit)
    for tid in token_ids:
        h ^= tid & 0xFFFFFFFF
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF  # FNV prime
    return h


class BlockDedup:
    """Deduplicates KV cache blocks with identical token content.

    When a block is registered with the same token sequence as an existing
    block, we return the existing block_id and increment its ref_count
    instead of keeping both copies.

    Hash collisions are handled by verifying token equality on match.

    Usage:
        dedup = BlockDedup(pool)
        existing = dedup.register_block(5, [10, 20, 30, ...])
        if existing is not None:
            pool.free(block_5)  # duplicate — use existing instead
    """

    def __init__(self, pool: BlockPool):
        self._pool = pool
        # hash → (block_id, token_ids)
        self._hash_to_entry: Dict[int, Tuple[int, List[int]]] = {}
        # block_id → hash (reverse lookup for deregister)
        self._block_to_hash: Dict[int, int] = {}
        self._dedup_hits = 0
        self._dedup_misses = 0
        self._collision_count = 0

    def register_block(
        self, block_id: int, token_ids: List[int]
    ) -> Optional[int]:
        """Register a block's content. Returns existing block_id if duplicate.

        If the token content matches an existing block (hash + equality check):
        - Returns the existing block_id
        - Increments the existing block's ref_count

        If hash matches but tokens differ (collision):
        - Treats as no match, registers new block
        - Increments collision counter

        If no match:
        - Stores the hash + token mapping
        - Returns None
        """
        content_hash = _fnv1a_hash(token_ids)

        existing = self._hash_to_entry.get(content_hash)
        if existing is not None:
            existing_id, existing_tokens = existing
            if existing_id != block_id:
                existing_block = self._pool.get_block(existing_id)
                if existing_block.ref_count > 0:
                    # Verify token equality (prevents hash collision corruption)
                    if existing_tokens == token_ids:
                        self._pool.increment_ref(existing_block)
                        existing_block.content_hash = content_hash
                        self._dedup_hits += 1
                        return existing_id
                    else:
                        # Hash collision — different tokens, same hash
                        self._collision_count += 1
                        # Fall through to register as new

        # No match — register this block
        self._hash_to_entry[content_hash] = (block_id, list(token_ids))
        self._block_to_hash[block_id] = content_hash
        block = self._pool.get_block(block_id)
        block.content_hash = content_hash
        self._dedup_misses += 1
        return None

    def deregister_block(self, block_id: int):
        """Remove a block from the dedup index."""
        h = self._block_to_hash.pop(block_id, None)
        if h is not None:
            entry = self._hash_to_entry.get(h)
            if entry is not None and entry[0] == block_id:
                del self._hash_to_entry[h]

    def lookup(self, token_ids: List[int]) -> Optional[int]:
        """Check if a token sequence has a cached block (without registering)."""
        content_hash = _fnv1a_hash(token_ids)
        entry = self._hash_to_entry.get(content_hash)
        if entry is not None:
            bid, stored_tokens = entry
            block = self._pool.get_block(bid)
            if block.ref_count > 0 and stored_tokens == token_ids:
                return bid
            elif block.ref_count <= 0:
                # Stale entry
                del self._hash_to_entry[content_hash]
        return None

    @property
    def dedup_hits(self) -> int:
        return self._dedup_hits

    @property
    def dedup_misses(self) -> int:
        return self._dedup_misses

    @property
    def collision_count(self) -> int:
        return self._collision_count

    @property
    def hit_rate(self) -> float:
        total = self._dedup_hits + self._dedup_misses
        return self._dedup_hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "indexed_blocks": len(self._hash_to_entry),
            "dedup_hits": self._dedup_hits,
            "dedup_misses": self._dedup_misses,
            "collision_count": self._collision_count,
            "hit_rate": self.hit_rate,
        }
