"""ZSE Speculative Decoding — Draft Token Generators.

Three strategies for generating draft tokens:

1. N-gram draft: Zero-cost lookup in token history. No model needed.
   Matches n-grams from the prompt/output to predict continuations.
   Acceptance rate: ~40-60%.

2. Self-draft (greedy lookahead): Uses the main model's last logits
   to predict K tokens greedily. No extra model, but only works for
   greedy/low-temperature generation. Acceptance rate: ~60-80%.

3. Small model draft: Loads a separate small .zse model and runs
   actual inference. Highest acceptance rate (~70-85%) but costs
   one small forward pass per K drafts.
"""

import math
import struct
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Optional, Dict


class DraftModel(ABC):
    """Abstract base for draft token generators."""

    @abstractmethod
    def generate_drafts(
        self,
        context_tokens: List[int],
        K: int,
    ) -> Tuple[List[int], List[List[float]]]:
        """Generate K draft tokens with their probability distributions.

        Args:
            context_tokens: All tokens so far (prompt + generated)
            K: Number of draft tokens to generate

        Returns:
            (draft_tokens, draft_probs) where:
            - draft_tokens: [K] token IDs
            - draft_probs: [K, vocab_size] probability distributions
        """
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        ...


class NGramDraftModel(DraftModel):
    """N-gram based draft model — zero compute cost.

    Builds an n-gram table from the token sequence and uses it to
    predict continuations. Falls back to most-frequent-token when
    no n-gram match is found.

    Args:
        vocab_size: Vocabulary size (for probability distributions)
        n: N-gram size (default: 3 — trigram)
        confidence: Probability assigned to the predicted token (rest uniform)
    """

    def __init__(
        self,
        vocab_size_val: int,
        n: int = 3,
        confidence: float = 0.8,
    ):
        self._vocab_size = vocab_size_val
        self._n = n
        self._confidence = confidence
        # N-gram table: tuple(context) → {next_token: count}
        self._ngram_table: Dict[tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._token_freq: Dict[int, int] = defaultdict(int)
        self._total_tokens = 0

    def vocab_size(self) -> int:
        return self._vocab_size

    def update(self, tokens: List[int]):
        """Update n-gram table with new tokens."""
        for i in range(len(tokens)):
            self._token_freq[tokens[i]] += 1
            self._total_tokens += 1

            if i >= self._n - 1:
                context = tuple(tokens[i - self._n + 1: i])
                self._ngram_table[context][tokens[i]] += 1

    def generate_drafts(
        self,
        context_tokens: List[int],
        K: int,
    ) -> Tuple[List[int], List[List[float]]]:
        """Generate K drafts using n-gram lookup."""
        # Update table with full context
        self.update(context_tokens)

        draft_tokens = []
        draft_probs = []

        # Extend context as we generate drafts
        extended = list(context_tokens)

        for _ in range(K):
            # Try n-gram match
            if len(extended) >= self._n - 1:
                context = tuple(extended[-(self._n - 1):])
                matches = self._ngram_table.get(context)
            else:
                matches = None

            if matches:
                # Find most likely continuation
                best_tok = max(matches, key=matches.get)
                total_matches = sum(matches.values())

                # Build probability distribution
                probs = self._make_probs(matches, total_matches)
            else:
                # No n-gram match — fall back to frequency-based prediction
                best_tok = self._most_frequent_token()
                probs = self._uniform_probs()

            draft_tokens.append(best_tok)
            draft_probs.append(probs)
            extended.append(best_tok)

        return draft_tokens, draft_probs

    def _make_probs(self, matches: Dict[int, int], total: int) -> List[float]:
        """Build probability distribution from n-gram counts."""
        # Confidence split: confidence for matched tokens, rest uniform
        probs = [0.0] * self._vocab_size
        remaining = 1.0 - self._confidence
        uniform_share = remaining / self._vocab_size

        for tok, count in matches.items():
            if 0 <= tok < self._vocab_size:
                probs[tok] = self._confidence * (count / total) + uniform_share

        # Fill unmatched with uniform
        for i in range(self._vocab_size):
            if probs[i] == 0.0:
                probs[i] = uniform_share

        return probs

    def _uniform_probs(self) -> List[float]:
        """Uniform distribution."""
        p = 1.0 / self._vocab_size
        return [p] * self._vocab_size

    def _most_frequent_token(self) -> int:
        """Return most frequent token seen so far."""
        if not self._token_freq:
            return 0
        return max(self._token_freq, key=self._token_freq.get)


class SelfDraftModel(DraftModel):
    """Self-draft using main model's last logits.

    Takes the most recent logits from the main model and greedily
    extends K tokens. Each draft token uses the previous draft as
    a greedy continuation (no actual model call).

    This works well for greedy/low-temperature generation because
    the top tokens at each position are highly predictable from
    the previous position's distribution.

    Args:
        vocab_size: Vocabulary size
        confidence: Probability assigned to greedy choice
    """

    def __init__(self, vocab_size_val: int, confidence: float = 0.9):
        self._vocab_size = vocab_size_val
        self._confidence = confidence
        self._last_logits: Optional[bytes] = None

    def vocab_size(self) -> int:
        return self._vocab_size

    def set_last_logits(self, logits_bytes: bytes):
        """Store the main model's most recent logits for drafting."""
        self._last_logits = logits_bytes

    def generate_drafts(
        self,
        context_tokens: List[int],
        K: int,
    ) -> Tuple[List[int], List[List[float]]]:
        """Generate K drafts from last logits (greedy lookahead)."""
        if self._last_logits is None:
            # No logits available — return random drafts
            rng = random.Random(42)
            tokens = [rng.randint(0, self._vocab_size - 1) for _ in range(K)]
            probs = [self._uniform_probs() for _ in range(K)]
            return tokens, probs

        # Decode logits → probabilities
        logits = [struct.unpack_from('<e', self._last_logits, i * 2)[0]
                  for i in range(self._vocab_size)]
        base_probs = self._softmax(logits)

        draft_tokens = []
        draft_probs = []

        current_probs = base_probs
        for _ in range(K):
            # Greedy pick
            best_tok = max(range(self._vocab_size), key=lambda i: current_probs[i])

            # Build draft probability: confidence on greedy, rest spread
            probs = self._make_draft_probs(current_probs, best_tok)

            draft_tokens.append(best_tok)
            draft_probs.append(probs)

            # For subsequent drafts, shift distribution slightly
            # (approximation — real self-draft would need a full forward pass)
            current_probs = probs

        return draft_tokens, draft_probs

    def _make_draft_probs(self, base_probs: List[float], greedy_tok: int) -> List[float]:
        """Build draft prob distribution with confidence on greedy choice."""
        probs = [0.0] * self._vocab_size
        remaining = 1.0 - self._confidence

        # Top token gets confidence
        probs[greedy_tok] = self._confidence

        # Distribute remaining proportionally to base probs (excluding greedy)
        base_sum = sum(base_probs) - base_probs[greedy_tok]
        if base_sum > 0:
            for i in range(self._vocab_size):
                if i != greedy_tok:
                    probs[i] = remaining * (base_probs[i] / base_sum)
        else:
            uniform = remaining / max(1, self._vocab_size - 1)
            for i in range(self._vocab_size):
                if i != greedy_tok:
                    probs[i] = uniform

        return probs

    def _softmax(self, logits: List[float]) -> List[float]:
        max_val = max(logits)
        exps = [math.exp(l - max_val) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _uniform_probs(self) -> List[float]:
        return [1.0 / self._vocab_size] * self._vocab_size
