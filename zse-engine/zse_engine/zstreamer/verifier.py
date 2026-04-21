"""ZSE Speculative Decoding — Verifier (accept/reject algorithm).

Implements the speculative decoding acceptance algorithm from:
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)

Key property: The output distribution is EXACTLY the same as standard
autoregressive decoding — speculative decoding is lossless.

Algorithm:
    For each draft token d_i with draft probability q(d_i) and target probability p(d_i):
        Accept with probability min(1, p(d_i) / q(d_i))
        If rejected: sample from adjusted distribution max(0, p(x) - q(x)) / Z
        Stop at first rejection.
    If all K accepted: bonus sample from p(x) at position K+1.
"""

import math
import random
import struct
from typing import List, Tuple, Optional


class SpeculativeVerifier:
    """Verifies draft tokens against target model logits.

    Implements the modified rejection sampling scheme that guarantees
    the output distribution matches standard autoregressive decoding.

    Args:
        seed: Random seed for reproducible acceptance decisions
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def verify_and_accept(
        self,
        draft_tokens: List[int],
        draft_probs: List[List[float]],
        target_logits_list: List[bytes],
        vocab_size: int,
        temperature: float = 1.0,
    ) -> Tuple[List[int], int]:
        """Verify draft tokens and return accepted tokens.

        Args:
            draft_tokens: K draft token IDs
            draft_probs: K probability distributions from draft model [K, vocab_size]
            target_logits_list: K+1 raw fp16 logits from target model verification
            vocab_size: Vocabulary size
            temperature: Sampling temperature (0 = greedy acceptance)

        Returns:
            (accepted_tokens, num_draft_accepted) where:
            - accepted_tokens: 1 to K+1 tokens (accepted drafts + bonus/resampled)
            - num_draft_accepted: how many of the K drafts were accepted (0 to K)
        """
        K = len(draft_tokens)
        assert len(draft_probs) == K
        assert len(target_logits_list) == K + 1

        # Convert target logits to probability distributions
        target_probs = []
        for logits_bytes in target_logits_list:
            logits = self._decode_fp16(logits_bytes, vocab_size)
            probs = self._logits_to_probs(logits, temperature)
            target_probs.append(probs)

        # Greedy mode: accept if draft matches argmax of target
        if temperature <= 0 or temperature < 1e-6:
            return self._verify_greedy(draft_tokens, target_probs, vocab_size)

        # Stochastic acceptance
        accepted = []
        num_draft_accepted = 0

        for i in range(K):
            draft_tok = draft_tokens[i]
            p_target = target_probs[i][draft_tok]
            p_draft = draft_probs[i][draft_tok]

            # Acceptance probability: min(1, p_target / p_draft)
            if p_draft <= 0:
                # Draft assigned zero probability — always reject
                resampled = self._sample_from_probs(target_probs[i])
                accepted.append(resampled)
                break

            accept_prob = min(1.0, p_target / p_draft)

            if self._rng.random() < accept_prob:
                # Accept this draft token
                accepted.append(draft_tok)
                num_draft_accepted += 1
            else:
                # Reject: sample from adjusted distribution
                # p_adjusted(x) = max(0, p_target(x) - p_draft(x)) / Z
                adjusted = self._adjusted_distribution(
                    target_probs[i], draft_probs[i], vocab_size,
                )
                resampled = self._sample_from_probs(adjusted)
                accepted.append(resampled)
                break
        else:
            # All K drafts accepted — bonus: sample from target at position K+1
            bonus = self._sample_from_probs(target_probs[K])
            accepted.append(bonus)

        return accepted, num_draft_accepted

    def _verify_greedy(
        self,
        draft_tokens: List[int],
        target_probs: List[List[float]],
        vocab_size: int,
    ) -> Tuple[List[int], int]:
        """Greedy verification: accept if draft matches target argmax."""
        accepted = []
        num_accepted = 0

        for i, draft_tok in enumerate(draft_tokens):
            target_tok = self._argmax(target_probs[i])
            if draft_tok == target_tok:
                accepted.append(draft_tok)
                num_accepted += 1
            else:
                # Reject — use target's argmax instead
                accepted.append(target_tok)
                break
        else:
            # All accepted — bonus from last position
            bonus = self._argmax(target_probs[len(draft_tokens)])
            accepted.append(bonus)

        return accepted, num_accepted

    def _adjusted_distribution(
        self,
        target_probs: List[float],
        draft_probs: List[float],
        vocab_size: int,
    ) -> List[float]:
        """Compute adjusted distribution: max(0, p_target - p_draft) / Z."""
        adjusted = [0.0] * vocab_size
        total = 0.0

        for i in range(vocab_size):
            val = max(0.0, target_probs[i] - draft_probs[i])
            adjusted[i] = val
            total += val

        # Normalize
        if total > 0:
            inv_total = 1.0 / total
            for i in range(vocab_size):
                adjusted[i] *= inv_total
        else:
            # Fallback: uniform over vocab (shouldn't happen in practice)
            inv_v = 1.0 / vocab_size
            for i in range(vocab_size):
                adjusted[i] = inv_v

        return adjusted

    def _logits_to_probs(self, logits: List[float], temperature: float) -> List[float]:
        """Convert logits to probability distribution with temperature."""
        if temperature != 1.0 and temperature > 0:
            logits = [l / temperature for l in logits]

        # Numerically stable softmax
        max_val = max(logits)
        exps = [math.exp(l - max_val) for l in logits]
        total = sum(exps)
        inv_total = 1.0 / total
        return [e * inv_total for e in exps]

    def _sample_from_probs(self, probs: List[float]) -> int:
        """Sample from a probability distribution."""
        r = self._rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if cumulative > r:
                return i
        return len(probs) - 1

    def _argmax(self, values: List[float]) -> int:
        max_val = values[0]
        max_idx = 0
        for i in range(1, len(values)):
            if values[i] > max_val:
                max_val = values[i]
                max_idx = i
        return max_idx

    def _decode_fp16(self, data: bytes, n: int) -> List[float]:
        """Decode fp16 bytes to float32 list."""
        return [struct.unpack_from('<e', data, i * 2)[0] for i in range(n)]


def acceptance_rate(num_draft_accepted: int, K: int) -> float:
    """Calculate acceptance rate for monitoring."""
    return num_draft_accepted / K if K > 0 else 0.0


def effective_tokens_per_step(num_tokens_produced: int) -> float:
    """Tokens produced per speculative step (1 to K+1)."""
    return float(num_tokens_produced)
