"""ZStreamer Queue — Priority request queue with SLO-aware ordering.

Not just FCFS. Orders by:
    1. SLO urgency (requests approaching deadline go first)
    2. Explicit priority (higher = more important)
    3. Arrival time (oldest first, as tiebreaker)

Thread safety: Caller (Scheduler) holds its own lock. No internal locking needed.

Separates waiting-for-prefill vs waiting-for-resume (preempted) requests.
"""

import heapq
from typing import List, Optional, Dict

from zse_engine.zstreamer.request import InferenceRequest, RequestState


class RequestQueue:
    """Priority queue for inference requests.

    Maintains two pools:
    - waiting: new requests that need prefill (min-heap)
    - preempted: paused requests that can resume (min-heap)

    Thread safety: NO internal locks. Caller (Scheduler) must hold its own lock.

    Args:
        max_queue_size: Maximum number of waiting requests (0 = unlimited)
    """

    def __init__(self, max_queue_size: int = 0):
        self._waiting: List[tuple] = []       # Heap of (urgency, -priority, arrival, req)
        self._preempted: List[tuple] = []      # Heap of (urgency, -priority, arrival, req)
        self._all: Dict[str, InferenceRequest] = {}  # By request_id
        self._cancelled: set = set()           # Lazy deletion for cancelled requests
        self._max_size = max_queue_size
        self._counter = 0  # Tiebreaker for heap stability

    def _heap_key(self, req: InferenceRequest) -> tuple:
        """Generate heap priority key. Lower = higher priority."""
        self._counter += 1
        return (req.urgency, -req.priority, req.arrival_time, self._counter)

    def add(self, request: InferenceRequest) -> bool:
        """Add a request to the queue.

        Returns False if queue is full (backpressure signal).
        """
        if self._max_size > 0 and len(self._waiting) >= self._max_size:
            return False

        self._all[request.request_id] = request

        if request.state == RequestState.PREEMPTED:
            heapq.heappush(self._preempted, (*self._heap_key(request), request))
        else:
            request.state = RequestState.WAITING
            heapq.heappush(self._waiting, (*self._heap_key(request), request))

        return True

    def add_preempted(self, request: InferenceRequest):
        """Return a preempted request to the queue for resumption."""
        request.mark_preempted()
        heapq.heappush(self._preempted, (*self._heap_key(request), request))
        self._all[request.request_id] = request

    def pop_prefill_batch(
        self,
        max_requests: int = 1,
        max_tokens: int = 0,
    ) -> List[InferenceRequest]:
        """Pop requests for prefill scheduling.

        Preempted requests get priority (they already have KV cache, cheaper to resume).
        Then new requests, ordered by SLO urgency.

        Args:
            max_requests: Max requests to return (controls burst)
            max_tokens: Max total prompt tokens in this batch (0 = no limit)

        Returns:
            List of requests to prefill this iteration.
        """
        batch = []
        total_tokens = 0

        # Priority 1: Resume preempted requests (cheap — KV already cached)
        while self._preempted and len(batch) < max_requests:
            _, _, _, _, req = heapq.heappop(self._preempted)
            # Skip lazily-deleted
            if req.request_id in self._cancelled:
                self._cancelled.discard(req.request_id)
                continue
            tok_cost = 1  # Just the resume token
            if max_tokens > 0 and total_tokens + tok_cost > max_tokens and batch:
                # Put it back
                heapq.heappush(self._preempted, (*self._heap_key(req), req))
                break
            batch.append(req)
            total_tokens += tok_cost

        # Priority 2: New requests
        while self._waiting and len(batch) < max_requests:
            _, _, _, _, req = heapq.heappop(self._waiting)
            # Skip lazily-deleted
            if req.request_id in self._cancelled:
                self._cancelled.discard(req.request_id)
                continue
            tok_cost = req.remaining_prefill if req.remaining_prefill > 0 else req.prompt_len
            if max_tokens > 0 and total_tokens + tok_cost > max_tokens and batch:
                heapq.heappush(self._waiting, (*self._heap_key(req), req))
                break
            batch.append(req)
            total_tokens += tok_cost

        return batch

    def cancel(self, request_id: str) -> Optional[InferenceRequest]:
        """Cancel a waiting request. Returns the request if found."""
        req = self._all.pop(request_id, None)
        if req is None:
            return None

        req.mark_cancelled()
        # Lazy deletion — skip during next pop
        self._cancelled.add(request_id)
        return req

    def remove_finished(self, request_id: str):
        """Remove a finished request from tracking."""
        self._all.pop(request_id, None)

    def get(self, request_id: str) -> Optional[InferenceRequest]:
        """Look up a request by ID."""
        return self._all.get(request_id)

    @property
    def num_waiting(self) -> int:
        return len(self._waiting) - sum(1 for r in self._waiting if r[-1].request_id in self._cancelled)

    @property
    def num_preempted(self) -> int:
        return len(self._preempted) - sum(1 for r in self._preempted if r[-1].request_id in self._cancelled)

    @property
    def num_total(self) -> int:
        return len(self._all)

    @property
    def is_empty(self) -> bool:
        # Check if all remaining heap entries are cancelled
        for heap in (self._waiting, self._preempted):
            for entry in heap:
                if entry[-1].request_id not in self._cancelled:
                    return False
        return True

    def peek_waiting(self) -> List[InferenceRequest]:
        """Peek at waiting requests (read-only, for monitoring)."""
        return [entry[-1] for entry in self._waiting
                if entry[-1].request_id not in self._cancelled]

    def stats(self) -> dict:
        active_waiting = [entry[-1] for entry in self._waiting
                         if entry[-1].request_id not in self._cancelled]
        waiting_tokens = sum(r.prompt_len for r in active_waiting)
        return {
            "num_waiting": len(active_waiting),
            "num_preempted": self.num_preempted,
            "num_total": len(self._all),
            "waiting_prompt_tokens": waiting_tokens,
            "queue_full": (self._max_size > 0
                           and len(active_waiting) >= self._max_size),
        }
