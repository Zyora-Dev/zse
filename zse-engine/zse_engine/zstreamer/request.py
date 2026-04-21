"""ZStreamer Request — Request lifecycle types for continuous batching.

States:
    WAITING → PREFILLING → DECODING → FINISHED
                                    → CANCELLED

Each request tracks timing for SLO metrics:
    - TTFT: time to first token (arrival → first decode token emitted)
    - TPOT: time per output token (avg decode latency)
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable


class RequestState(Enum):
    """Request lifecycle states."""
    WAITING = auto()      # In queue, not yet scheduled
    PREFILLING = auto()   # Prompt being processed (may be chunked)
    DECODING = auto()     # Generating tokens one at a time
    PREEMPTED = auto()    # Paused due to memory pressure (KV cached, resumable)
    FINISHED = auto()     # Generation complete (stop token, max length, or error)
    CANCELLED = auto()    # User-cancelled


class FinishReason(Enum):
    """Why generation stopped."""
    LENGTH = "length"       # Hit max_tokens
    STOP = "stop"           # Hit stop token or EOS
    CANCELLED = "cancelled" # User cancelled
    ERROR = "error"         # Runtime error
    PREEMPTED = "preempted" # Evicted under memory pressure (could retry)


@dataclass
class GenerationParams:
    """Per-request generation parameters."""
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_tokens: Optional[List[int]] = None
    seed: Optional[int] = None
    timeout_ms: Optional[float] = None  # Wall-clock timeout (None = no timeout)
    lora_id: Optional[str] = None       # LoRA adapter ID (None = base model only)


@dataclass
class InferenceRequest:
    """A single inference request in the continuous batching pipeline.

    Created when a user submits a prompt. Tracks full lifecycle from
    arrival through generation to completion.
    """
    # Identity
    request_id: str
    prompt_tokens: List[int]

    # Generation config
    params: GenerationParams = field(default_factory=GenerationParams)

    # Priority & SLO
    priority: int = 0                    # Higher = more important
    deadline_ms: Optional[float] = None  # TTFT SLO in ms (None = no SLO)

    # State
    state: RequestState = RequestState.WAITING
    output_tokens: List[int] = field(default_factory=list)
    finish_reason: Optional[FinishReason] = None

    # Chunked prefill tracking
    prefill_offset: int = 0  # How many prompt tokens have been processed
    chunk_size: int = 0      # 0 = process entire prompt at once

    # Timing (monotonic seconds)
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: Optional[float] = None  # When first output token was produced
    finish_time: Optional[float] = None

    # Sequence ID in KV cache (assigned when scheduled)
    seq_id: Optional[int] = None

    # Callback for streaming (called with each new token)
    on_token: Optional[Callable[[int], None]] = None
    # Callback when finished
    on_finish: Optional[Callable[["RequestOutput"], None]] = None

    # Past token frequencies for repetition penalty (token_id → count)
    past_tokens: Dict[int, int] = field(default_factory=dict)

    @property
    def lora_id(self) -> Optional[str]:
        """LoRA adapter ID for this request (shortcut to params.lora_id)."""
        return self.params.lora_id

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def num_generated(self) -> int:
        return len(self.output_tokens)

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + generated) — used for KV cache sizing."""
        return self.prompt_len + self.num_generated

    @property
    def remaining_prefill(self) -> int:
        """How many prompt tokens still need prefill processing."""
        return self.prompt_len - self.prefill_offset

    @property
    def is_active(self) -> bool:
        return self.state in (RequestState.PREFILLING, RequestState.DECODING)

    @property
    def is_finished(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.CANCELLED)

    @property
    def urgency(self) -> float:
        """SLO urgency score. Lower = more urgent.

        Requests with no deadline get infinity (lowest urgency).
        Requests past deadline get negative values (highest urgency).
        """
        if self.deadline_ms is None:
            return float("inf")
        elapsed_ms = (time.monotonic() - self.arrival_time) * 1000
        return self.deadline_ms - elapsed_ms

    @property
    def wait_time_ms(self) -> float:
        """How long this request has been waiting."""
        return (time.monotonic() - self.arrival_time) * 1000

    def mark_prefilling(self):
        self.state = RequestState.PREFILLING

    def mark_decoding(self):
        self.state = RequestState.DECODING

    def mark_preempted(self):
        self.state = RequestState.PREEMPTED

    def mark_finished(self, reason: FinishReason):
        self.state = RequestState.FINISHED
        self.finish_reason = reason
        self.finish_time = time.monotonic()

    def mark_cancelled(self):
        self.state = RequestState.CANCELLED
        self.finish_reason = FinishReason.CANCELLED
        self.finish_time = time.monotonic()

    def add_token(self, token_id: int):
        """Record a newly generated token."""
        self.output_tokens.append(token_id)
        self.past_tokens[token_id] = self.past_tokens.get(token_id, 0) + 1

        if self.first_token_time is None:
            self.first_token_time = time.monotonic()

        if self.on_token is not None:
            self.on_token(token_id)

    def should_stop(self) -> Optional[FinishReason]:
        """Check if generation should stop."""
        # Max tokens
        if self.num_generated >= self.params.max_tokens:
            return FinishReason.LENGTH

        # Stop tokens
        if self.output_tokens and self.params.stop_tokens:
            if self.output_tokens[-1] in self.params.stop_tokens:
                return FinishReason.STOP

        # Wall-clock timeout
        if self.params.timeout_ms is not None:
            elapsed = (time.monotonic() - self.arrival_time) * 1000
            if elapsed > self.params.timeout_ms:
                return FinishReason.LENGTH  # Treat timeout as length limit

        return None


@dataclass
class RequestOutput:
    """Final output for a completed request."""
    request_id: str
    prompt_tokens: List[int]
    output_tokens: List[int]
    finish_reason: FinishReason

    # Timing metrics (milliseconds)
    ttft_ms: float        # Time to first token
    total_time_ms: float  # Total wall time
    tpot_ms: float        # Average time per output token
    tokens_per_sec: float # Output throughput

    @staticmethod
    def from_request(req: InferenceRequest) -> "RequestOutput":
        """Build output from a finished request."""
        now = req.finish_time or time.monotonic()
        total_ms = (now - req.arrival_time) * 1000
        ttft_ms = ((req.first_token_time - req.arrival_time) * 1000
                   if req.first_token_time else total_ms)

        num_out = max(len(req.output_tokens), 1)
        decode_time = (now - req.first_token_time) * 1000 if req.first_token_time else total_ms
        tpot_ms = decode_time / num_out
        tps = num_out / (total_ms / 1000) if total_ms > 0 else 0

        return RequestOutput(
            request_id=req.request_id,
            prompt_tokens=req.prompt_tokens,
            output_tokens=req.output_tokens,
            finish_reason=req.finish_reason or FinishReason.ERROR,
            ttft_ms=ttft_ms,
            total_time_ms=total_ms,
            tpot_ms=tpot_ms,
            tokens_per_sec=tps,
        )
