"""ZStreamer Memory Budget — Predictive VRAM admission control.

Prevents OOM bursts by tracking committed memory BEFORE admitting requests.
Unlike vLLM's fixed-budget model, this predicts future KV cache growth and
reserves headroom for decode tokens that haven't been generated yet.

Key idea:
    committed = actual_kv_used + reserved_for_active_decode_headroom
    can_admit(request) → (committed + request_cost) < budget

This means we never start a request we can't finish.
"""

from dataclasses import dataclass
from typing import Dict

from zse_engine.zstreamer.request import InferenceRequest


@dataclass
class MemoryReservation:
    """Tracks memory reserved for one active request."""
    request_id: str
    prompt_blocks: int       # Blocks used by prompt KV cache
    initial_headroom: int    # Original headroom blocks reserved
    last_decode_blocks: int  # Last known actual decode blocks used
    total_blocks: int        # prompt_blocks + actual_decode + remaining_headroom


class MemoryBudget:
    """Predictive memory budget manager for continuous batching.

    Tracks real-time block usage and reserves headroom for active requests'
    future decode tokens. This prevents admitting requests that would cause
    OOM during generation.

    Args:
        total_blocks: Total KV cache blocks available on GPU
        block_size_tokens: Tokens per block
        headroom_ratio: Fraction of max_tokens to reserve as decode headroom (0.0-1.0)
            1.0 = reserve for all max_tokens (safest, lowest throughput)
            0.5 = reserve for half of max_tokens (balanced)
            0.0 = no headroom reservation (aggressive, may need preemption)
        emergency_reserve_blocks: Always keep this many blocks free for system use
    """

    def __init__(
        self,
        total_blocks: int,
        block_size_tokens: int,
        headroom_ratio: float = 0.5,
        emergency_reserve_blocks: int = 2,
    ):
        self._total_blocks = total_blocks
        self._block_size = block_size_tokens
        self._headroom_ratio = headroom_ratio
        self._emergency_reserve = emergency_reserve_blocks

        # Active reservations by request_id
        self._reservations: Dict[str, MemoryReservation] = {}
        # Cached committed block count (O(1) instead of O(n))
        self._committed_cache: int = 0

    def _tokens_to_blocks(self, tokens: int) -> int:
        """Convert token count to block count (ceiling division)."""
        return (tokens + self._block_size - 1) // self._block_size

    def _request_cost(self, request: InferenceRequest) -> tuple:
        """Calculate (prompt_blocks, decode_headroom_blocks) for a request."""
        prompt_blocks = self._tokens_to_blocks(request.prompt_len)

        # Reserve headroom for future decode tokens
        decode_tokens = int(request.params.max_tokens * self._headroom_ratio)
        decode_blocks = self._tokens_to_blocks(decode_tokens)

        return prompt_blocks, decode_blocks

    @property
    def committed_blocks(self) -> int:
        """Total blocks committed (used + reserved headroom). O(1)."""
        return self._committed_cache

    @property
    def available_blocks(self) -> int:
        """Blocks available for new requests."""
        return max(0, self._total_blocks - self.committed_blocks - self._emergency_reserve)

    def can_admit(self, request: InferenceRequest) -> bool:
        """Check if we can admit this request without risking OOM.

        Returns True if there's enough budget for:
        - The request's prompt KV cache
        - Reserved headroom for decode tokens
        - Emergency reserve
        """
        prompt_blocks, decode_blocks = self._request_cost(request)
        needed = prompt_blocks + decode_blocks
        return needed <= self.available_blocks

    def can_admit_n(self, requests: list) -> int:
        """How many of these requests can we admit?
        Returns count (0 to len(requests)).
        """
        available = self.available_blocks
        admitted = 0
        for req in requests:
            p, d = self._request_cost(req)
            cost = p + d
            if cost <= available:
                available -= cost
                admitted += 1
            else:
                break
        return admitted

    def reserve(self, request: InferenceRequest):
        """Reserve memory for an admitted request."""
        prompt_blocks, decode_blocks = self._request_cost(request)
        total = prompt_blocks + decode_blocks
        self._reservations[request.request_id] = MemoryReservation(
            request_id=request.request_id,
            prompt_blocks=prompt_blocks,
            initial_headroom=decode_blocks,
            last_decode_blocks=0,
            total_blocks=total,
        )
        self._committed_cache += total

    def release(self, request_id: str):
        """Release memory reservation when a request finishes."""
        res = self._reservations.pop(request_id, None)
        if res is not None:
            self._committed_cache -= res.total_blocks

    def update_decode_progress(self, request_id: str, tokens_generated: int):
        """Shrink decode headroom as tokens are actually generated.

        Uses delta-based tracking to avoid double-counting:
        only adjusts for the CHANGE in actual decode blocks since last update.
        """
        res = self._reservations.get(request_id)
        if res is None:
            return

        actual_decode_blocks = self._tokens_to_blocks(tokens_generated)
        delta = actual_decode_blocks - res.last_decode_blocks

        if delta <= 0:
            return  # No new blocks used

        res.last_decode_blocks = actual_decode_blocks

        # Headroom shrinks by delta (actual blocks converted from reserved→used)
        headroom_reduction = min(delta, res.initial_headroom - (actual_decode_blocks - delta))
        headroom_reduction = max(0, min(delta, res.total_blocks - res.prompt_blocks - actual_decode_blocks))

        # Recalculate total: prompt + actual_decode + remaining_headroom
        remaining_headroom = max(0, res.initial_headroom - actual_decode_blocks)
        new_total = res.prompt_blocks + actual_decode_blocks + remaining_headroom
        old_total = res.total_blocks

        res.total_blocks = new_total
        self._committed_cache += (new_total - old_total)

    def preemption_candidates(self, blocks_needed: int) -> list:
        """Find requests to preempt to free blocks_needed blocks.

        Returns request_ids sorted by priority (lowest priority first).
        """
        candidates = sorted(
            self._reservations.values(),
            key=lambda r: r.total_blocks,  # Prefer preempting requests using most memory
            reverse=True,
        )

        to_preempt = []
        freed = 0
        for res in candidates:
            if freed >= blocks_needed:
                break
            to_preempt.append(res.request_id)
            freed += res.total_blocks

        return to_preempt

    @property
    def utilization(self) -> float:
        """Memory utilization (0.0 to 1.0)."""
        if self._total_blocks == 0:
            return 0.0
        return self.committed_blocks / self._total_blocks

    def stats(self) -> dict:
        return {
            "total_blocks": self._total_blocks,
            "committed_blocks": self.committed_blocks,
            "available_blocks": self.available_blocks,
            "num_reservations": len(self._reservations),
            "utilization": self.utilization,
            "emergency_reserve": self._emergency_reserve,
            "headroom_ratio": self._headroom_ratio,
        }
