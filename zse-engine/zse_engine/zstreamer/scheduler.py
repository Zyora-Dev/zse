"""ZStreamer Scheduler — Core continuous batching scheduler.

Disaggregated prefill/decode scheduling:
    - Prefill iterations: process new prompts (high compute, fills KV cache)
    - Decode iterations: process all active sequences (memory-bound)

Anti-burst design:
    - Token budget per iteration (max_batch_tokens)
    - Sequence cap per iteration (max_batch_seqs)
    - Chunked prefill for long prompts
    - Gradual admission (1-2 prefills per cycle)
    - Preemption under memory pressure

Scheduling cycle:
    1. Check for finished/cancelled requests → cleanup
    2. If decode sequences exist → schedule decode step
    3. If queue has waiting requests + memory available → schedule prefill
    4. Ratio-based: N decode steps per 1 prefill step

This is the brain of ZStreamer.
"""

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Set

from zse_engine.zstreamer.request import (
    InferenceRequest, RequestState, FinishReason,
)
from zse_engine.zstreamer.queue import RequestQueue
from zse_engine.zstreamer.memory_budget import MemoryBudget


class StepType(Enum):
    """What kind of iteration to run."""
    PREFILL = auto()   # Process new prompt(s)
    DECODE = auto()    # Generate next token for active sequences
    MIXED = auto()     # Chunked prefill + decode (interleaved)
    IDLE = auto()      # Nothing to do


@dataclass
class SchedulerOutput:
    """Output of schedule_step() — tells the batch runner what to execute."""
    step_type: StepType

    # Prefill requests (new or resuming)
    prefill_requests: List[InferenceRequest] = field(default_factory=list)
    # For chunked prefill: how many tokens to process per request this step
    prefill_chunk_sizes: List[int] = field(default_factory=list)

    # Decode requests (generate next token)
    decode_requests: List[InferenceRequest] = field(default_factory=list)

    # Requests to preempt this step (free their KV cache)
    preempt_requests: List[InferenceRequest] = field(default_factory=list)

    # Requests finished this step (for cleanup)
    finished_requests: List[InferenceRequest] = field(default_factory=list)

    @property
    def num_prefill_tokens(self) -> int:
        """Total prompt tokens to process in this step."""
        if self.prefill_chunk_sizes:
            return sum(self.prefill_chunk_sizes)
        return sum(r.remaining_prefill for r in self.prefill_requests)

    @property
    def num_decode_tokens(self) -> int:
        """Total decode tokens (1 per sequence)."""
        return len(self.decode_requests)

    @property
    def total_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_idle(self) -> bool:
        return self.step_type == StepType.IDLE


@dataclass
class SchedulerConfig:
    """Scheduler tuning parameters."""
    # Batch limits (anti-burst)
    max_batch_tokens: int = 4096    # Max total tokens per iteration
    max_batch_seqs: int = 64        # Max concurrent sequences
    max_prefill_per_step: int = 2   # Max new prefills per step (gradual admission)

    # Chunked prefill
    prefill_chunk_size: int = 512   # Max prompt tokens per chunk (0 = no chunking)

    # Scheduling ratio
    decode_per_prefill: int = 4     # N decode steps per 1 prefill opportunity

    # Preemption
    enable_preemption: bool = True  # Allow pausing low-priority requests under memory pressure

    # Speculative decoding
    speculative_k: int = 0          # Draft tokens per step (0 = disabled, 4-8 typical)

    # Queue
    max_queue_size: int = 256       # Max waiting requests (0 = unlimited)


class Scheduler:
    """Core continuous batching scheduler.

    Manages the lifecycle of all active requests and decides what to
    execute each iteration step.

    Args:
        config: Scheduler tuning parameters
        memory_budget: Predictive memory budget manager
        kv_block_size: Tokens per KV cache block (from cache manager)
    """

    def __init__(
        self,
        config: SchedulerConfig,
        memory_budget: MemoryBudget,
    ):
        self._config = config
        self._budget = memory_budget
        self._queue = RequestQueue(max_queue_size=config.max_queue_size)
        self._lock = threading.Lock()  # Protects _active, _budget, state transitions

        # Active requests (currently prefilling or decoding)
        self._active: Dict[str, InferenceRequest] = {}

        # Step counter for prefill/decode ratio
        self._step_counter = 0

        # Sequence ID counter
        self._seq_id_counter = 0

        # Stats
        self._total_steps = 0
        self._total_prefill_steps = 0
        self._total_decode_steps = 0
        self._total_preemptions = 0
        self._total_finished = 0

    def _next_seq_id(self) -> int:
        sid = self._seq_id_counter
        self._seq_id_counter += 1
        return sid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, request: InferenceRequest) -> bool:
        """Submit a new request for scheduling.

        Returns False if the queue is full (backpressure).
        Thread-safe.
        """
        return self._queue.add(request)

    def cancel_request(self, request_id: str):
        """Cancel a request (whether waiting or active). Thread-safe."""
        # Try queue first
        req = self._queue.cancel(request_id)
        if req is not None:
            return

        # Check active
        with self._lock:
            req = self._active.get(request_id)
            if req is not None:
                req.mark_cancelled()

    def schedule_step(self) -> SchedulerOutput:
        """Decide what to execute this iteration. Thread-safe."""
        with self._lock:
            return self._schedule_step_locked()

    def _schedule_step_locked(self) -> SchedulerOutput:
        self._total_steps += 1
        self._step_counter += 1

        # Phase 1: Collect finished/cancelled active requests
        finished = self._collect_finished()

        # Phase 2: Handle preemption if memory is tight
        preempted = self._handle_preemption()

        # Phase 3: Decide between prefill and decode
        output = SchedulerOutput(
            step_type=StepType.IDLE,
            finished_requests=finished,
            preempt_requests=preempted,
        )

        has_decode = self._has_decode_requests()
        has_waiting = not self._queue.is_empty
        is_prefill_turn = (self._step_counter % (self._config.decode_per_prefill + 1) == 0)

        if has_decode and has_waiting:
            # Both decode and prefill candidates — use ratio
            if is_prefill_turn:
                self._schedule_prefill(output)
                # If tokens left in budget, also schedule decode
                remaining = self._config.max_batch_tokens - output.num_prefill_tokens
                if remaining > 0:
                    self._schedule_decode(output, max_tokens=remaining)
                if output.prefill_requests or output.decode_requests:
                    output.step_type = StepType.MIXED if (output.prefill_requests and output.decode_requests) else (StepType.PREFILL if output.prefill_requests else StepType.DECODE)
            else:
                self._schedule_decode(output)
                if output.decode_requests:
                    output.step_type = StepType.DECODE
        elif has_decode:
            self._schedule_decode(output)
            if output.decode_requests:
                output.step_type = StepType.DECODE
        elif has_waiting:
            self._schedule_prefill(output)
            if output.prefill_requests:
                output.step_type = StepType.PREFILL
                self._step_counter = 0  # Reset ratio counter when only prefilling

        # Update stats
        if output.step_type == StepType.PREFILL:
            self._total_prefill_steps += 1
        elif output.step_type == StepType.DECODE:
            self._total_decode_steps += 1
        elif output.step_type == StepType.MIXED:
            self._total_prefill_steps += 1
            self._total_decode_steps += 1

        return output

    # ------------------------------------------------------------------
    # Internal scheduling
    # ------------------------------------------------------------------

    def _collect_finished(self) -> List[InferenceRequest]:
        """Remove finished/cancelled requests from active set."""
        finished = []
        to_remove = []

        for rid, req in self._active.items():
            if req.is_finished:
                finished.append(req)
                to_remove.append(rid)
                self._budget.release(rid)
                self._queue.remove_finished(rid)
                self._total_finished += 1

        for rid in to_remove:
            del self._active[rid]

        return finished

    def _has_decode_requests(self) -> bool:
        """Any active requests in DECODING state?"""
        return any(r.state == RequestState.DECODING for r in self._active.values())

    def _schedule_prefill(self, output: SchedulerOutput):
        """Schedule new requests for prefill processing."""
        # Pop candidates from queue
        candidates = self._queue.pop_prefill_batch(
            max_requests=self._config.max_prefill_per_step,
            max_tokens=self._config.max_batch_tokens,
        )

        rejected = []  # Track candidates that couldn't be admitted

        for req in candidates:
            # Respect sequence cap FIRST — before any budget checks
            if len(self._active) >= self._config.max_batch_seqs:
                rejected.append(req)
                continue

            # Admission control: check memory budget
            if not self._budget.can_admit(req):
                rejected.append(req)
                continue

            # Assign seq_id if new
            if req.seq_id is None:
                req.seq_id = self._next_seq_id()

            # Reserve memory
            self._budget.reserve(req)

            # Handle chunked prefill
            if (self._config.prefill_chunk_size > 0
                    and req.remaining_prefill > self._config.prefill_chunk_size):
                chunk = self._config.prefill_chunk_size
                output.prefill_chunk_sizes.append(chunk)
            else:
                chunk = req.remaining_prefill
                output.prefill_chunk_sizes.append(chunk)

            req.mark_prefilling()
            output.prefill_requests.append(req)
            self._active[req.request_id] = req

        # Re-add all rejected candidates back to queue (prevents request leak)
        for req in rejected:
            self._queue.add(req)

    def _schedule_decode(self, output: SchedulerOutput, max_tokens: int = 0):
        """Schedule active decode sequences for next token generation."""
        if max_tokens <= 0:
            max_tokens = self._config.max_batch_tokens

        budget = min(max_tokens, self._config.max_batch_seqs)

        # Collect all DECODING requests
        decode_reqs = [
            req for req in self._active.values()
            if req.state == RequestState.DECODING
        ]

        # Sort by urgency (most urgent first)
        decode_reqs.sort(key=lambda r: (r.urgency, -r.priority))

        # Pack into batch up to token budget (1 token per sequence for decode)
        for req in decode_reqs:
            if len(output.decode_requests) >= budget:
                break
            output.decode_requests.append(req)

    def _handle_preemption(self) -> List[InferenceRequest]:
        """Preempt low-priority requests if memory budget is critically low."""
        if not self._config.enable_preemption:
            return []

        # Only preempt if we have waiting requests and no available blocks
        if self._queue.is_empty:
            return []
        if self._budget.available_blocks > 0:
            return []

        # Find lowest-priority active decode requests to preempt
        decode_reqs = [
            req for req in self._active.values()
            if req.state == RequestState.DECODING
        ]

        if not decode_reqs:
            return []

        # Sort by priority (lowest first = best preemption candidates),
        # then by least generated tokens (preempt requests that have made least progress)
        decode_reqs.sort(key=lambda r: (r.priority, r.num_generated))

        # Calculate target: free enough blocks for smallest waiting request
        waiting = self._queue.peek_waiting()
        if waiting:
            smallest_prompt = min(r.prompt_len for r in waiting)
            target_blocks = self._budget._tokens_to_blocks(smallest_prompt) + 2
        else:
            target_blocks = 4

        preempted = []
        freed_blocks = 0

        for req in decode_reqs:
            if freed_blocks >= target_blocks:
                break

            # Calculate how many blocks this request holds BEFORE releasing
            req_blocks = self._budget._tokens_to_blocks(req.total_tokens)

            req.mark_preempted()
            self._budget.release(req.request_id)
            del self._active[req.request_id]
            self._queue.add_preempted(req)

            preempted.append(req)
            self._total_preemptions += 1
            freed_blocks += req_blocks

        return preempted

    # ------------------------------------------------------------------
    # Request state transitions (called by batch runner after execution)
    # ------------------------------------------------------------------

    def on_prefill_complete(self, request: InferenceRequest, chunk_tokens: int = 0):
        """Called when prefill finishes for a request.

        If chunked and more prompt remains, stays in PREFILLING.
        Otherwise transitions to DECODING.

        Args:
            chunk_tokens: How many tokens were processed in this chunk.
                          If 0, assumes full prefill was completed.
        """
        if chunk_tokens > 0:
            request.prefill_offset += chunk_tokens
        elif request.prefill_offset == 0:
            # Full prefill done in one shot
            request.prefill_offset = request.prompt_len

        if request.remaining_prefill > 0:
            # More chunks needed — stays in PREFILLING
            pass
        else:
            request.mark_decoding()

    def on_token_generated(self, request: InferenceRequest, token_id: int):
        """Called when a decode step produces a token."""
        request.add_token(token_id)
        self._budget.update_decode_progress(
            request.request_id, request.num_generated,
        )

        # Check stop conditions
        reason = request.should_stop()
        if reason is not None:
            request.mark_finished(reason)

    def on_request_error(self, request: InferenceRequest, error: str):
        """Called when a request encounters a runtime error."""
        request.mark_finished(FinishReason.ERROR)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def num_active(self) -> int:
        return len(self._active)

    @property
    def num_waiting(self) -> int:
        return self._queue.num_waiting

    @property
    def num_decoding(self) -> int:
        return sum(1 for r in self._active.values() if r.state == RequestState.DECODING)

    @property
    def num_prefilling(self) -> int:
        return sum(1 for r in self._active.values() if r.state == RequestState.PREFILLING)

    def stats(self) -> dict:
        queue_stats = self._queue.stats()
        budget_stats = self._budget.stats()
        return {
            "total_steps": self._total_steps,
            "prefill_steps": self._total_prefill_steps,
            "decode_steps": self._total_decode_steps,
            "total_preemptions": self._total_preemptions,
            "total_finished": self._total_finished,
            "num_active": self.num_active,
            "num_decoding": self.num_decoding,
            "num_prefilling": self.num_prefilling,
            "num_waiting": self.num_waiting,
            "queue": queue_stats,
            "memory": budget_stats,
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            f"ZStreamer Scheduler:\n"
            f"  Active: {s['num_active']} "
            f"(prefill: {s['num_prefilling']}, decode: {s['num_decoding']})\n"
            f"  Waiting: {s['num_waiting']}\n"
            f"  Steps: {s['total_steps']} "
            f"(prefill: {s['prefill_steps']}, decode: {s['decode_steps']})\n"
            f"  Preemptions: {s['total_preemptions']}\n"
            f"  Finished: {s['total_finished']}\n"
            f"  Memory: {s['memory']['utilization']:.1%} committed"
        )
