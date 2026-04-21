"""ZSE Sampler — Token sampling from logits.

Supports:
- Greedy (argmax)
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling
- Repetition penalty

Logits are downloaded from GPU to CPU for sampling — this is tiny
(1 × vocab_size ≈ 256KB for 128k vocab) so not worth a GPU kernel.
"""

import struct
import math
import random
import array
from typing import List, Optional, Set


class Sampler:
    """Token sampler from fp16 logits.

    Usage:
        sampler = Sampler()
        token = sampler.sample(logits_bytes, vocab_size,
                               temperature=0.8, top_p=0.9)
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def sample(
        self,
        logits_bytes: bytes,
        vocab_size: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        past_tokens = None,  # Set[int] or Dict[int,int] (token→count)
    ) -> int:
        """Sample a token from fp16 logits.

        Args:
            logits_bytes: Raw fp16 bytes from GPU (vocab_size * 2 bytes)
            vocab_size: Number of tokens in vocabulary
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            top_k: Top-k filtering (0 = disabled)
            repetition_penalty: Penalize repeated tokens (1.0 = disabled)
            past_tokens: Set of token IDs already generated (for rep penalty)

        Returns:
            Sampled token ID
        """
        # Fast path: greedy with no rep penalty
        if (temperature <= 0 or temperature < 1e-6) and (repetition_penalty == 1.0 or not past_tokens):
            return self._fast_argmax_fp16(logits_bytes, vocab_size)

        # Decode fp16 logits to float32
        logits = self._decode_fp16_fast(logits_bytes, vocab_size)

        # Apply repetition penalty
        if repetition_penalty != 1.0 and past_tokens:
            for tid, count in (past_tokens.items() if isinstance(past_tokens, dict) else ((t, 1) for t in past_tokens)):
                if 0 <= tid < vocab_size:
                    # Scale penalty by frequency: penalty^(1 + log2(count))
                    import math
                    scaled = repetition_penalty ** (1.0 + math.log2(count)) if count > 1 else repetition_penalty
                    if logits[tid] > 0:
                        logits[tid] /= scaled
                    else:
                        logits[tid] *= scaled

        # Greedy
        if temperature <= 0 or temperature < 1e-6:
            return self._argmax_array(logits)

        # Temperature scaling
        if temperature != 1.0:
            inv_t = 1.0 / temperature
            for i in range(len(logits)):
                logits[i] *= inv_t

        # Top-k filtering
        if top_k > 0:
            logits = self._top_k_filter_array(logits, top_k)

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            logits = self._top_p_filter(list(logits), top_p)

        # Softmax
        probs = self._softmax(list(logits) if not isinstance(logits, list) else logits)

        # Sample from distribution
        return self._categorical_sample(probs)

    def greedy(self, logits_bytes: bytes, vocab_size: int) -> int:
        """Greedy (argmax) decoding — fast path using array module."""
        return self._fast_argmax_fp16(logits_bytes, vocab_size)

    def _fast_argmax_fp16(self, data: bytes, n: int) -> int:
        """Find argmax of fp16 values without full Python list creation."""
        if len(data) < n * 2:
            raise ValueError(f"Expected {n * 2} bytes, got {len(data)}")
        best_idx = 0
        best_val = float('-inf')
        # Process in chunks of 1024 to balance unpack overhead vs iteration
        chunk = 1024
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            count = end - start
            vals = struct.unpack_from(f'<{count}e', data, start * 2)
            for j in range(count):
                if vals[j] > best_val:
                    best_val = vals[j]
                    best_idx = start + j
        return best_idx

    def _decode_fp16(self, data: bytes, n: int) -> List[float]:
        """Decode fp16 bytes to float32 list (batch unpack)."""
        if len(data) < n * 2:
            raise ValueError(f"Expected {n * 2} bytes, got {len(data)}")
        return list(struct.unpack(f'<{n}e', data[:n * 2]))

    def _decode_fp16_fast(self, data: bytes, n: int):
        """Decode fp16 bytes to mutable array (avoids Python list overhead)."""
        if len(data) < n * 2:
            raise ValueError(f"Expected {n * 2} bytes, got {len(data)}")
        # array.array('f') is faster than list for large sequences
        arr = array.array('f', struct.unpack(f'<{n}e', data[:n * 2]))
        return arr

    def _argmax_array(self, values) -> int:
        """Return index of maximum value (works with array.array)."""
        max_val = values[0]
        max_idx = 0
        for i in range(1, len(values)):
            if values[i] > max_val:
                max_val = values[i]
                max_idx = i
        return max_idx

    def _top_k_filter_array(self, logits, k: int):
        """Top-k filter that works with array.array."""
        n = len(logits)
        if k >= n:
            return logits
        indexed = sorted(range(n), key=lambda i: logits[i], reverse=True)
        threshold = logits[indexed[k - 1]]
        for i in range(n):
            if logits[i] < threshold:
                logits[i] = float('-inf')
        return logits

    def _argmax(self, values: List[float]) -> int:
        """Return index of maximum value."""
        max_val = values[0]
        max_idx = 0
        for i in range(1, len(values)):
            if values[i] > max_val:
                max_val = values[i]
                max_idx = i
        return max_idx

    def _softmax(self, logits: List[float]) -> List[float]:
        """Numerically stable softmax."""
        max_val = max(logits)
        exps = [math.exp(l - max_val) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _top_k_filter(self, logits: List[float], k: int) -> List[float]:
        """Keep only top-k logits, set rest to -inf."""
        if k >= len(logits):
            return logits

        # Find k-th largest value
        indexed = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
        threshold = indexed[k - 1][1]

        return [l if l >= threshold else float('-inf') for l in logits]

    def _top_p_filter(self, logits: List[float], p: float) -> List[float]:
        """Nucleus sampling: keep smallest set of tokens with cumulative prob >= p."""
        # Softmax first
        probs = self._softmax(logits)

        # Sort by probability descending
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)

        cumulative = 0.0
        keep_indices = set()
        for idx, prob in indexed:
            cumulative += prob
            keep_indices.add(idx)
            if cumulative >= p:
                break

        return [l if i in keep_indices else float('-inf')
                for i, l in enumerate(logits)]

    def _categorical_sample(self, probs: List[float]) -> int:
        """Sample from a categorical distribution."""
        r = self._rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if cumulative > r:
                return i
        return len(probs) - 1  # Fallback
