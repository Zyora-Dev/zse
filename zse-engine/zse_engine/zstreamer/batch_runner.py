"""ZStreamer Batch Runner — Executes scheduled batches on GPU.

Takes SchedulerOutput (what to run) and drives the ModelRunner + Sampler
to actually execute prefill and decode steps on the GPU.

This is the bridge between the scheduler (policy) and the GPU (execution).

Fixes applied:
- No double KV allocation (model_runner.prefill handles allocation)
- No double KV free (finished_requests cleaned up only once)
- Proper vocab_size access
- Finished tracking via set to prevent duplicates
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional

from zse_engine.zstreamer.request import (
    InferenceRequest, RequestState, FinishReason, RequestOutput,
)
from zse_engine.zstreamer.scheduler import Scheduler, SchedulerOutput, StepType
from zse_engine.orchestrator.model_runner import ModelRunner
from zse_engine.orchestrator.sampler import Sampler
from zse_engine.cache.cache_manager import KVCacheManager

# Optional LoRA support
try:
    from zse_engine.orchestrator.lora_manager import LoRAManager
except ImportError:
    LoRAManager = None


@dataclass
class StepResult:
    """Result of executing one scheduler step."""
    # New tokens generated (request_id → token_id)
    new_tokens: Dict[str, int] = field(default_factory=dict)
    # Requests that finished this step
    finished: List[RequestOutput] = field(default_factory=list)
    # Requests that were preempted
    preempted_ids: List[str] = field(default_factory=list)
    # Errors (request_id → error message)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return len(self.new_tokens)


class BatchRunner:
    """Executes scheduled batches on GPU.

    Args:
        model_runner: GPU model execution engine
        sampler: Token sampler
        kv_cache: KV cache manager
        scheduler: Scheduler (for state transition callbacks)
        vocab_size: Model vocabulary size
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        sampler: Sampler,
        kv_cache: KVCacheManager,
        scheduler: Scheduler,
        vocab_size: int = 32000,
        spec_runner=None,  # Optional SpeculativeRunner
        lora_manager=None,  # Optional LoRAManager
    ):
        self._runner = model_runner
        self._sampler = sampler
        self._kv_cache = kv_cache
        self._scheduler = scheduler
        self._vocab_size = vocab_size
        self._spec_runner = spec_runner
        self._lora_manager = lora_manager

    def execute(self, output: SchedulerOutput) -> StepResult:
        """Execute a scheduled step on GPU.

        Args:
            output: SchedulerOutput from scheduler.schedule_step()

        Returns:
            StepResult with generated tokens and state changes.
        """
        result = StepResult()
        # Track which requests we've already finalized to prevent double-free
        finalized: Set[str] = set()

        # Handle preemptions first (mark idle)
        for req in output.preempt_requests:
            self._handle_preemption(req, result)

        # Execute prefill
        if output.prefill_requests:
            self._execute_prefill(output, result, finalized)

        # Execute decode
        if output.decode_requests:
            self._execute_decode(output, result, finalized)

        # Collect previously finished outputs (from scheduler._collect_finished)
        # These were finished in a PRIOR step — just emit outputs, KV already freed
        for req in output.finished_requests:
            if req.request_id not in finalized:
                self._finalize_request(req, result, finalized)

        return result

    def _execute_prefill(
        self, output: SchedulerOutput, result: StepResult, finalized: Set[str],
    ):
        """Run prefill for new requests.

        NOTE: model_runner.prefill() handles KV cache allocation internally.
        We do NOT allocate here to avoid double allocation.
        """
        for i, req in enumerate(output.prefill_requests):
            try:
                # Determine chunk range
                chunk_size = (output.prefill_chunk_sizes[i]
                              if i < len(output.prefill_chunk_sizes)
                              else req.remaining_prefill)

                start = req.prefill_offset
                end = min(start + chunk_size, req.prompt_len)
                chunk_tokens = req.prompt_tokens[start:end]

                # Resolve LoRA adapter if requested
                lora_adapter = self._resolve_lora(req)

                # Run prefill on GPU
                # model_runner.prefill() handles KV allocation + mark_active internally
                logits = self._runner.prefill(chunk_tokens, req.seq_id,
                                              lora_adapter=lora_adapter)
                req.prefill_offset = end

                # Notify scheduler of prefill progress
                self._scheduler.on_prefill_complete(req)

                # If prefill is fully done, sample first token
                if req.state == RequestState.DECODING:
                    token = self._sample_token(req, logits)
                    self._scheduler.on_token_generated(req, token)
                    result.new_tokens[req.request_id] = token

                    # Check if immediately finished (e.g., EOS token)
                    if req.is_finished:
                        self._finalize_request(req, result, finalized)

            except Exception as e:
                self._scheduler.on_request_error(req, str(e))
                result.errors[req.request_id] = str(e)
                # Clean up on error
                if req.seq_id is not None and req.request_id not in finalized:
                    try:
                        self._kv_cache.mark_idle(req.seq_id)
                        self._kv_cache.free_sequence(req.seq_id)
                    except Exception:
                        pass
                    finalized.add(req.request_id)

    def _execute_decode(
        self, output: SchedulerOutput, result: StepResult, finalized: Set[str],
    ):
        """Run decode step — uses speculative decoding if enabled, else batched.

        Speculative: draft K tokens → verify in one pass → accept 1 to K+1
        Standard: batched_decode for all M sequences
        """
        reqs = [r for r in output.decode_requests
                if not r.is_finished and r.request_id not in finalized]
        if not reqs:
            return

        # Speculative decoding path
        if self._spec_runner is not None:
            self._execute_speculative_decode(reqs, result, finalized)
            return

        # Standard batched decode path
        self._execute_batched_decode(reqs, result, finalized)

    def _execute_speculative_decode(
        self, reqs: List[InferenceRequest], result: StepResult, finalized: Set[str],
    ):
        """Speculative decode: draft→verify→accept for each sequence."""
        from zse_engine.zstreamer.spec_runner import SpeculativeRunner

        spec_results = self._spec_runner.batched_speculative_step(reqs)

        for i, (req, spec_result) in enumerate(zip(reqs, spec_results)):
            try:
                if not spec_result.accepted_tokens:
                    # Speculative step failed — treat as error
                    self._scheduler.on_request_error(req, "Speculative decode produced no tokens")
                    result.errors[req.request_id] = "Speculative decode failed"
                    continue

                # Add all accepted tokens to the request
                for token in spec_result.accepted_tokens:
                    self._scheduler.on_token_generated(req, token)
                    result.new_tokens[req.request_id] = token  # Last token for streaming

                    if req.is_finished:
                        self._finalize_request(req, result, finalized)
                        break

            except Exception as e:
                self._scheduler.on_request_error(req, str(e))
                result.errors[req.request_id] = str(e)

    def _execute_batched_decode(
        self, reqs: List[InferenceRequest], result: StepResult, finalized: Set[str],
    ):

        # Group requests by LoRA adapter for efficient batching
        # All requests with the same adapter (or None) are batched together
        adapter_groups: Dict[Optional[str], List[InferenceRequest]] = {}
        for req in reqs:
            lora_id = req.lora_id
            if lora_id not in adapter_groups:
                adapter_groups[lora_id] = []
            adapter_groups[lora_id].append(req)

        for lora_id, group_reqs in adapter_groups.items():
            lora_adapter = None
            if lora_id is not None and self._lora_manager is not None:
                lora_adapter = self._lora_manager.get_adapter(lora_id)

            # Gather batch metadata for this group
            token_ids = []
            seq_ids = []
            positions = []
            for req in group_reqs:
                last_token = (req.output_tokens[-1] if req.output_tokens
                              else req.prompt_tokens[-1])
                token_ids.append(last_token)
                seq_ids.append(req.seq_id)
                # Position for decode: prompt filled positions 0..prompt_len-1,
                # first generated token at prompt_len, second at prompt_len+1, etc.
                # num_generated counts tokens already added, so current position
                # is prompt_len + num_generated - 1 (we're decoding the LAST added token)
                positions.append(req.prompt_len + req.num_generated - 1)

            try:
                # Check if ALL requests are greedy
                all_greedy = all(
                    r.params.temperature <= 0 and r.params.repetition_penalty == 1.0
                    for r in group_reqs
                )

                # Use graph mode if available and all greedy
                use_graph = (
                    all_greedy
                    and self._runner._graph_runner is not None
                    and lora_adapter is None
                )

                if use_graph:
                    # Graph path: ONE driver call replays all kernels
                    tokens = self._runner.batched_decode_graph(
                        token_ids, seq_ids, positions,
                    )
                    for i, req in enumerate(group_reqs):
                        self._scheduler.on_token_generated(req, tokens[i])
                        result.new_tokens[req.request_id] = tokens[i]
                        if req.is_finished:
                            self._finalize_request(req, result, finalized)
                elif all_greedy:
                    # Non-graph greedy: forward + GPU argmax per row
                    self._runner.batched_decode(
                        token_ids, seq_ids, positions,
                        lora_adapter=lora_adapter,
                        skip_logits_download=True,
                    )
                    for i, req in enumerate(group_reqs):
                        token = self._runner.gpu_argmax(i)
                        self._scheduler.on_token_generated(req, token)
                        result.new_tokens[req.request_id] = token
                        if req.is_finished:
                            self._finalize_request(req, result, finalized)
                else:
                    # Standard path: download logits, CPU sampling
                    all_logits = self._runner.batched_decode(
                        token_ids, seq_ids, positions,
                        lora_adapter=lora_adapter,
                    )

                    # Sample tokens and update state
                    for i, req in enumerate(group_reqs):
                        try:
                            token = self._sample_token(req, all_logits[i])
                            self._scheduler.on_token_generated(req, token)
                            result.new_tokens[req.request_id] = token

                            if req.is_finished:
                                self._finalize_request(req, result, finalized)
                        except Exception as e:
                            self._scheduler.on_request_error(req, str(e))
                            result.errors[req.request_id] = str(e)

            except Exception as e:
                # Batch-level failure — mark all requests as errored
                for req in group_reqs:
                    self._scheduler.on_request_error(req, str(e))
                    result.errors[req.request_id] = str(e)

    def _finalize_request(
        self, req: InferenceRequest, result: StepResult, finalized: Set[str],
    ):
        """Finalize a finished request — emit output, free KV, fire callback.

        Idempotent: skips if already finalized.
        """
        if req.request_id in finalized:
            return
        finalized.add(req.request_id)

        output = RequestOutput.from_request(req)
        result.finished.append(output)

        # Free KV cache (once only)
        if req.seq_id is not None:
            try:
                self._kv_cache.mark_idle(req.seq_id)
                self._kv_cache.free_sequence(req.seq_id)
            except Exception:
                pass  # Already freed

        # Fire completion callback (once only — clear after firing)
        if req.on_finish is not None:
            cb = req.on_finish
            req.on_finish = None  # Prevent double-fire across steps
            req.on_token = None   # Also clear streaming callback
            try:
                cb(output)
            except Exception:
                pass  # Don't crash on callback errors

    def _handle_preemption(self, req: InferenceRequest, result: StepResult):
        """Handle a preempted request — mark idle but keep KV cache."""
        if req.seq_id is not None:
            self._kv_cache.mark_idle(req.seq_id)
        result.preempted_ids.append(req.request_id)

    def _sample_token(self, req: InferenceRequest, logits: bytes) -> int:
        """Sample a token using the request's generation params."""
        return self._sampler.sample(
            logits,
            self._vocab_size,
            temperature=req.params.temperature,
            top_p=req.params.top_p,
            top_k=req.params.top_k,
            repetition_penalty=req.params.repetition_penalty,
            past_tokens=req.past_tokens,
        )

    def _resolve_lora(self, req: InferenceRequest):
        """Resolve a request's lora_id to a LoRAAdapter, or None."""
        lora_id = req.lora_id
        if lora_id is None or self._lora_manager is None:
            return None
        adapter = self._lora_manager.get_adapter(lora_id)
        if adapter is None:
            # Log warning but don't fail — run without LoRA
            pass
        return adapter
