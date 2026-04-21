"""ZSE Evictor — Smart KV cache eviction policies.

Our competitive advantage over vLLM:
- vLLM: sequence-level only, LRU only
- ZSE: token-level (block granularity), smart scoring

Policies:
- "lru": Least recently used block (simple baseline)
- "lfu": Least frequently accessed block
- "smart": Combined score = frequency × recency_weight + recompute_cost
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


@dataclass
class BlockAccessInfo:
    """Tracks access patterns for a single block."""
    seq_id: int
    block_idx: int           # Index within the sequence's block list
    block_id: int
    num_tokens: int
    access_count: int = 0    # How many times accessed in attention
    last_access_time: float = 0.0
    token_start: int = 0     # First token position in the original sequence
    is_prompt: bool = False   # Prompt tokens are more expensive to recompute


@dataclass
class EvictionCandidate:
    """A block nominated for eviction."""
    seq_id: int
    block_idx: int
    block_id: int
    score: float        # Lower score = better candidate for eviction
    num_tokens: int


class Evictor:
    """Smart eviction policy for KV cache blocks.

    Usage:
        evictor = Evictor(policy="smart")
        evictor.record_access(seq_id=0, block_idx=2, block_id=5, num_tokens=16)
        victims = evictor.select_victims(num_blocks_needed=3, candidates=block_info_list)
    """

    def __init__(self, policy: str = "smart"):
        if policy not in ("lru", "lfu", "smart"):
            raise ValueError(f"Unknown eviction policy: {policy}")
        self._policy = policy
        self._access_info: Dict[int, BlockAccessInfo] = {}  # block_id → info
        self._eviction_count = 0
        self._total_tokens_evicted = 0

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def eviction_count(self) -> int:
        return self._eviction_count

    @property
    def total_tokens_evicted(self) -> int:
        return self._total_tokens_evicted

    def record_access(
        self,
        seq_id: int,
        block_idx: int,
        block_id: int,
        num_tokens: int,
        token_start: int = 0,
        is_prompt: bool = False,
    ):
        """Record that a block was accessed during attention."""
        now = time.monotonic()

        info = self._access_info.get(block_id)
        if info is None:
            info = BlockAccessInfo(
                seq_id=seq_id,
                block_idx=block_idx,
                block_id=block_id,
                num_tokens=num_tokens,
                token_start=token_start,
                is_prompt=is_prompt,
            )
            self._access_info[block_id] = info

        info.access_count += 1
        info.last_access_time = now

    def record_batch_access(self, seq_id: int, block_ids: List[int],
                            block_size: int, prompt_len: int = 0,
                            total_seq_tokens: int = 0):
        """Record access for all blocks of a sequence (convenience).

        Args:
            total_seq_tokens: Actual total tokens (for last block accuracy)
        """
        for idx, bid in enumerate(block_ids):
            if bid == -1:
                continue  # Skip evicted holes
            token_start = idx * block_size
            # Last block may have fewer tokens
            if total_seq_tokens > 0:
                tokens_in_block = min(block_size, total_seq_tokens - token_start)
            else:
                tokens_in_block = block_size
            self.record_access(
                seq_id=seq_id,
                block_idx=idx,
                block_id=bid,
                num_tokens=tokens_in_block,
                token_start=token_start,
                is_prompt=(token_start < prompt_len),
            )

    def remove_block(self, block_id: int):
        """Remove tracking for a freed block."""
        self._access_info.pop(block_id, None)

    def _score_block(self, info: BlockAccessInfo, now: float = 0.0) -> float:
        """Score a block — LOWER score = better eviction candidate."""
        if self._policy == "lru":
            return info.last_access_time

        elif self._policy == "lfu":
            return info.access_count

        else:  # "smart"
            # Combined: frequency × recency + recompute_cost
            time_since_access = now - info.last_access_time if info.last_access_time > 0 else 1000.0
            recency = 1.0 / (1.0 + time_since_access)  # Decays with time

            frequency = info.access_count

            # Recompute cost: prompt tokens are expensive to recompute
            # (require re-running prefill), decode tokens are cheap
            recompute = 2.0 if info.is_prompt else 0.5

            # Higher score = more valuable = evict last
            return frequency * recency + recompute

    def select_victims(
        self,
        num_blocks_needed: int,
        exclude_seq_ids: Optional[set] = None,
    ) -> List[EvictionCandidate]:
        """Select blocks to evict, sorted by lowest score first.

        Args:
            num_blocks_needed: How many blocks we need to free
            exclude_seq_ids: Sequences to never evict from (e.g., currently generating)

        Returns:
            List of EvictionCandidates, length <= num_blocks_needed
        """
        if exclude_seq_ids is None:
            exclude_seq_ids = set()

        # Score all tracked blocks
        now = time.monotonic()
        scored = []
        for block_id, info in self._access_info.items():
            if info.seq_id in exclude_seq_ids:
                continue
            score = self._score_block(info, now)
            scored.append(EvictionCandidate(
                seq_id=info.seq_id,
                block_idx=info.block_idx,
                block_id=block_id,
                score=score,
                num_tokens=info.num_tokens,
            ))

        # Sort by score ascending (lowest = best eviction candidate)
        scored.sort(key=lambda c: c.score)

        # Take only what we need
        victims = scored[:num_blocks_needed]

        self._eviction_count += len(victims)
        self._total_tokens_evicted += sum(v.num_tokens for v in victims)

        return victims

    def stats(self) -> dict:
        return {
            "policy": self._policy,
            "tracked_blocks": len(self._access_info),
            "eviction_count": self._eviction_count,
            "total_tokens_evicted": self._total_tokens_evicted,
        }
