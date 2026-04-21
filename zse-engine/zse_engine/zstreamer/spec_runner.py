"""ZSE Speculative Decoding — Speculative Runner.

Orchestrates the draft→verify→accept loop for speculative decoding.
Integrates with ModelRunner (verification), DraftModel (drafting),
and SpeculativeVerifier (accept/reject).

Usage:
    spec = SpeculativeRunner(model_runner, kv_cache, draft_model, verifier, vocab_size)

    # Single sequence speculative step
    tokens = spec.speculative_step(req, K=4)
    # Returns 1 to K+1 accepted tokens

    # Batched speculative step
    all_tokens = spec.batched_speculative_step(requests, K=4)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict

from zse_engine.zstreamer.request import InferenceRequest
from zse_engine.zstreamer.draft_model import DraftModel
from zse_engine.zstreamer.verifier import SpeculativeVerifier
from zse_engine.orchestrator.model_runner import ModelRunner
from zse_engine.cache.cache_manager import KVCacheManager


@dataclass
class SpeculativeResult:
    """Result of one speculative decode step for a single sequence."""
    request_id: str
    accepted_tokens: List[int]    # 1 to K+1 tokens
    num_draft_accepted: int       # How many of K drafts were accepted
    total_tokens: int             # len(accepted_tokens)


class SpeculativeRunner:
    """Orchestrates speculative decoding for single and batched decode.

    Args:
        model_runner: Main model for verification
        kv_cache: KV cache manager (for truncation on rejection)
        draft_model: Draft token generator
        verifier: Accept/reject algorithm
        vocab_size: Model vocabulary size
        default_k: Default number of draft tokens
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        kv_cache: KVCacheManager,
        draft_model: DraftModel,
        verifier: SpeculativeVerifier,
        vocab_size: int,
        default_k: int = 4,
        lora_manager=None,
    ):
        self._runner = model_runner
        self._kv_cache = kv_cache
        self._draft = draft_model
        self._verifier = verifier
        self._vocab_size = vocab_size
        self._default_k = default_k
        self._lora_manager = lora_manager

        # Stats
        self._total_steps = 0
        self._total_draft_tokens = 0
        self._total_accepted = 0
        self._total_produced = 0

    def speculative_step(
        self,
        request: InferenceRequest,
        K: Optional[int] = None,
    ) -> SpeculativeResult:
        """One speculative decode step for a single sequence.

        1. Generate K draft tokens
        2. Verify all K+1 tokens in one main model pass
        3. Accept/reject with lossless algorithm
        4. Truncate KV cache for rejected tokens

        Args:
            request: The inference request
            K: Number of draft tokens (default: self._default_k)

        Returns:
            SpeculativeResult with 1 to K+1 accepted tokens
        """
        if K is None:
            K = self._default_k

        # Build context: all tokens so far
        context = list(request.prompt_tokens) + list(request.output_tokens)
        position = len(context)

        # Step 1: Generate K draft tokens
        draft_tokens, draft_probs = self._draft.generate_drafts(context, K)

        # Step 2: Verify — run main model on [last_token, d1, ..., dK]
        # The last accepted token is the "anchor" — we verify the K drafts after it
        last_token = context[-1] if context else 0
        verify_input = [last_token] + draft_tokens  # K+1 tokens

        # Extend KV cache and run verification pass
        # Resolve LoRA adapter for this request
        lora_adapter = None
        if request.lora_id and self._lora_manager is not None:
            lora_adapter = self._lora_manager.get_adapter(request.lora_id)

        verify_logits = self._runner.verify_tokens(
            verify_input, request.seq_id, position - 1,
            lora_adapter=lora_adapter,
        )
        # verify_logits[0] = logits after processing last_token (predicts d1)
        # verify_logits[i] = logits after processing d_i (predicts d_{i+1})
        # We need logits[0..K] to verify draft_tokens[0..K-1] + bonus

        # Step 3: Accept/reject
        accepted_tokens, num_accepted = self._verifier.verify_and_accept(
            draft_tokens,
            draft_probs,
            verify_logits,  # K+1 logit rows
            self._vocab_size,
            temperature=request.params.temperature,
        )

        # Step 4: Truncate KV cache — remove rejected draft tokens' KV entries
        # We extended by K+1 tokens in verify_tokens, but only accepted some
        # Keep: original tokens + accepted tokens
        keep_tokens = (position - 1) + 1 + len(accepted_tokens)  # anchor + accepted
        # Actually: verify_tokens extended by K+1. Total KV now = position - 1 + K + 1.
        # We want total KV = position - 1 + len(accepted_tokens)
        # But the anchor token was already in KV (at position-1), verify_tokens extended by K+1
        # So KV is now at position - 1 + K + 1 = position + K
        # We want KV at position - 1 + len(accepted_tokens)
        # Wait — verify_tokens calls extend_sequence(K+1), so:
        # KV before verify = position (prompt + generated so far)
        # KV after verify = position + K + 1
        # But the anchor token (last_token at position-1) is already in KV
        # So verify added K+1 tokens starting at position
        # We want to keep: position + len(accepted_tokens) tokens total
        target_kv_len = position + len(accepted_tokens)
        current_kv_len = position + K + 1  # What verify_tokens set it to

        if target_kv_len < current_kv_len:
            self._kv_cache.truncate_sequence(request.seq_id, target_kv_len)

        # Update stats
        self._total_steps += 1
        self._total_draft_tokens += K
        self._total_accepted += num_accepted
        self._total_produced += len(accepted_tokens)

        return SpeculativeResult(
            request_id=request.request_id,
            accepted_tokens=accepted_tokens,
            num_draft_accepted=num_accepted,
            total_tokens=len(accepted_tokens),
        )

    def batched_speculative_step(
        self,
        requests: List[InferenceRequest],
        K: Optional[int] = None,
    ) -> List[SpeculativeResult]:
        """Batched speculative decode for M sequences.

        Drafts independently per sequence, then verifies all M sequences
        in parallel (if verify supports batching), then accepts independently.

        Args:
            requests: List of M active decode requests
            K: Number of draft tokens per sequence

        Returns:
            List of M SpeculativeResults
        """
        if K is None:
            K = self._default_k

        results = []
        for req in requests:
            try:
                result = self.speculative_step(req, K)
                results.append(result)
            except Exception as e:
                # On error, produce empty result
                results.append(SpeculativeResult(
                    request_id=req.request_id,
                    accepted_tokens=[],
                    num_draft_accepted=0,
                    total_tokens=0,
                ))

        return results

    @property
    def acceptance_rate(self) -> float:
        """Average draft acceptance rate."""
        if self._total_draft_tokens == 0:
            return 0.0
        return self._total_accepted / self._total_draft_tokens

    @property
    def tokens_per_step(self) -> float:
        """Average tokens produced per speculative step."""
        if self._total_steps == 0:
            return 0.0
        return self._total_produced / self._total_steps

    @property
    def speedup_factor(self) -> float:
        """Estimated speedup vs standard decode.

        Standard decode: 1 token per main model pass.
        Speculative: tokens_per_step tokens per main model pass (+ cheap draft cost).
        """
        return self.tokens_per_step

    def stats(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "total_draft_tokens": self._total_draft_tokens,
            "total_accepted": self._total_accepted,
            "total_produced": self._total_produced,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_step": self.tokens_per_step,
            "speedup_factor": self.speedup_factor,
        }

    def summary(self) -> str:
        s = self.stats()
        return (
            f"Speculative Decoding Stats:\n"
            f"  Steps: {s['total_steps']}\n"
            f"  Drafts: {s['total_draft_tokens']}, "
            f"Accepted: {s['total_accepted']} "
            f"({s['acceptance_rate']:.1%})\n"
            f"  Tokens/step: {s['tokens_per_step']:.2f}\n"
            f"  Estimated speedup: {s['speedup_factor']:.2f}x"
        )
