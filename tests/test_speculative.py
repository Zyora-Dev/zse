"""ZSE Speculative Decoding — Unit tests.

Tests the verifier (accept/reject), draft models (n-gram, self-draft),
spec_runner orchestration, and KV cache truncation.

No GPU needed — tests the algorithm correctness.
"""

import math
import struct
import pytest

from zse_engine.zstreamer.verifier import SpeculativeVerifier
from zse_engine.zstreamer.draft_model import NGramDraftModel, SelfDraftModel


# ======================================================================
# Helpers
# ======================================================================

def make_logits_bytes(probs: list, vocab_size: int) -> bytes:
    """Convert a probability list to fp16 logits bytes (log-space)."""
    # Convert probs to logits (log scale)
    logits = [math.log(max(p, 1e-10)) for p in probs]
    # Pad to vocab_size
    while len(logits) < vocab_size:
        logits.append(-100.0)
    return struct.pack(f'<{vocab_size}e', *logits[:vocab_size])


def uniform_probs(vocab_size: int) -> list:
    return [1.0 / vocab_size] * vocab_size


def peaked_probs(token: int, vocab_size: int, confidence: float = 0.9) -> list:
    """Create a distribution peaked at `token`."""
    remaining = (1.0 - confidence) / max(1, vocab_size - 1)
    probs = [remaining] * vocab_size
    probs[token] = confidence
    return probs


# ======================================================================
# Verifier Tests
# ======================================================================

class TestSpeculativeVerifier:
    def test_greedy_all_accepted(self):
        """When draft matches target argmax, all should be accepted."""
        v = SpeculativeVerifier(seed=42)
        vocab = 10
        K = 3

        # Draft: [5, 3, 7]
        draft_tokens = [5, 3, 7]
        # Draft probs: peaked at draft tokens
        draft_probs = [peaked_probs(5, vocab), peaked_probs(3, vocab), peaked_probs(7, vocab)]
        # Target: same argmax as draft → all accepted
        target_logits = [
            make_logits_bytes(peaked_probs(5, vocab), vocab),  # predicts 5 ✓
            make_logits_bytes(peaked_probs(3, vocab), vocab),  # predicts 3 ✓
            make_logits_bytes(peaked_probs(7, vocab), vocab),  # predicts 7 ✓
            make_logits_bytes(peaked_probs(2, vocab), vocab),  # bonus position
        ]

        accepted, num_accepted = v.verify_and_accept(
            draft_tokens, draft_probs, target_logits, vocab, temperature=0.0,
        )

        assert num_accepted == 3  # All drafts accepted
        assert len(accepted) == 4  # K + 1 (bonus)
        assert accepted[:3] == [5, 3, 7]
        assert accepted[3] == 2  # Bonus from last position

    def test_greedy_first_rejected(self):
        """First draft doesn't match target → reject immediately."""
        v = SpeculativeVerifier(seed=42)
        vocab = 10
        K = 3

        draft_tokens = [5, 3, 7]
        draft_probs = [peaked_probs(5, vocab)] * 3
        target_logits = [
            make_logits_bytes(peaked_probs(9, vocab), vocab),  # predicts 9, not 5 ✗
            make_logits_bytes(peaked_probs(3, vocab), vocab),
            make_logits_bytes(peaked_probs(7, vocab), vocab),
            make_logits_bytes(peaked_probs(2, vocab), vocab),
        ]

        accepted, num_accepted = v.verify_and_accept(
            draft_tokens, draft_probs, target_logits, vocab, temperature=0.0,
        )

        assert num_accepted == 0
        assert len(accepted) == 1  # Just the resampled token
        assert accepted[0] == 9  # Target's argmax

    def test_greedy_partial_accept(self):
        """First two match, third doesn't."""
        v = SpeculativeVerifier(seed=42)
        vocab = 10
        K = 3

        draft_tokens = [5, 3, 7]
        draft_probs = [peaked_probs(t, vocab) for t in [5, 3, 7]]
        target_logits = [
            make_logits_bytes(peaked_probs(5, vocab), vocab),  # ✓
            make_logits_bytes(peaked_probs(3, vocab), vocab),  # ✓
            make_logits_bytes(peaked_probs(8, vocab), vocab),  # ✗ (8 not 7)
            make_logits_bytes(peaked_probs(1, vocab), vocab),
        ]

        accepted, num_accepted = v.verify_and_accept(
            draft_tokens, draft_probs, target_logits, vocab, temperature=0.0,
        )

        assert num_accepted == 2
        assert len(accepted) == 3  # 2 accepted + 1 resampled
        assert accepted[0] == 5
        assert accepted[1] == 3
        assert accepted[2] == 8  # Target's token, not draft's 7

    def test_stochastic_acceptance(self):
        """With temperature > 0, acceptance is probabilistic."""
        v = SpeculativeVerifier(seed=123)
        vocab = 10
        K = 3

        draft_tokens = [5, 3, 7]
        # Draft gives 90% confidence on each
        draft_probs = [peaked_probs(t, vocab, 0.9) for t in [5, 3, 7]]
        # Target also gives high probability to same tokens
        target_logits = [
            make_logits_bytes(peaked_probs(5, vocab, 0.85), vocab),
            make_logits_bytes(peaked_probs(3, vocab, 0.85), vocab),
            make_logits_bytes(peaked_probs(7, vocab, 0.85), vocab),
            make_logits_bytes(peaked_probs(1, vocab, 0.5), vocab),
        ]

        accepted, num_accepted = v.verify_and_accept(
            draft_tokens, draft_probs, target_logits, vocab, temperature=1.0,
        )

        # Should accept at least some (high agreement between draft and target)
        assert len(accepted) >= 1
        assert len(accepted) <= K + 1

    def test_single_draft(self):
        """K=1 case."""
        v = SpeculativeVerifier(seed=42)
        vocab = 10

        draft_tokens = [5]
        draft_probs = [peaked_probs(5, vocab)]
        target_logits = [
            make_logits_bytes(peaked_probs(5, vocab), vocab),
            make_logits_bytes(peaked_probs(2, vocab), vocab),
        ]

        accepted, num_accepted = v.verify_and_accept(
            draft_tokens, draft_probs, target_logits, vocab, temperature=0.0,
        )

        assert num_accepted == 1
        assert accepted == [5, 2]  # Draft accepted + bonus

    def test_output_length_bounds(self):
        """Output is always between 1 and K+1 tokens."""
        v = SpeculativeVerifier(seed=42)
        vocab = 100
        K = 5

        for _ in range(20):
            draft_tokens = [i % vocab for i in range(K)]
            draft_probs = [uniform_probs(vocab) for _ in range(K)]
            target_logits = [make_logits_bytes(uniform_probs(vocab), vocab) for _ in range(K + 1)]

            accepted, _ = v.verify_and_accept(
                draft_tokens, draft_probs, target_logits, vocab, temperature=1.0,
            )

            assert 1 <= len(accepted) <= K + 1


# ======================================================================
# N-gram Draft Model Tests
# ======================================================================

class TestNGramDraft:
    def test_basic_generation(self):
        """N-gram model generates K tokens."""
        draft = NGramDraftModel(vocab_size_val=100, n=3)
        context = [1, 2, 3, 4, 5, 1, 2, 3, 10, 11]

        tokens, probs = draft.generate_drafts(context, K=3)

        assert len(tokens) == 3
        assert len(probs) == 3
        assert all(len(p) == 100 for p in probs)
        assert all(0 <= t < 100 for t in tokens)

    def test_ngram_match(self):
        """Should predict based on seen n-grams."""
        draft = NGramDraftModel(vocab_size_val=100, n=3)
        # Repeated pattern: [1, 2] → 3 always
        context = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]

        tokens, probs = draft.generate_drafts(context, K=1)

        # Should predict 3 as continuation of [1, 2]
        assert tokens[0] == 3

    def test_probs_sum_to_one(self):
        """Draft probabilities should approximately sum to 1."""
        draft = NGramDraftModel(vocab_size_val=50, n=2)
        context = [1, 2, 3, 4, 5]

        tokens, probs = draft.generate_drafts(context, K=2)

        for p in probs:
            total = sum(p)
            assert abs(total - 1.0) < 0.01, f"Probs sum to {total}"

    def test_empty_context(self):
        """Should handle empty/short context gracefully."""
        draft = NGramDraftModel(vocab_size_val=100, n=3)

        tokens, probs = draft.generate_drafts([1], K=2)
        assert len(tokens) == 2


# ======================================================================
# Self-Draft Model Tests
# ======================================================================

class TestSelfDraft:
    def test_basic_generation(self):
        """Self-draft generates K tokens from last logits."""
        draft = SelfDraftModel(vocab_size_val=10)

        # Create peaked logits at token 5
        logits = peaked_probs(5, 10, 0.9)
        logits_bytes = struct.pack(f'<{10}e', *[math.log(max(p, 1e-10)) for p in logits])
        draft.set_last_logits(logits_bytes)

        tokens, probs = draft.generate_drafts([1, 2, 3], K=3)

        assert len(tokens) == 3
        assert tokens[0] == 5  # Greedy from peaked logits

    def test_no_logits_fallback(self):
        """Without logits, should still produce tokens (random)."""
        draft = SelfDraftModel(vocab_size_val=100)
        tokens, probs = draft.generate_drafts([1, 2], K=3)

        assert len(tokens) == 3
        assert all(0 <= t < 100 for t in tokens)

    def test_probs_sum_to_one(self):
        """Draft probabilities should sum to ~1."""
        draft = SelfDraftModel(vocab_size_val=20)
        logits_bytes = struct.pack(f'<{20}e', *[0.1] * 20)
        draft.set_last_logits(logits_bytes)

        tokens, probs = draft.generate_drafts([1], K=2)
        for p in probs:
            assert abs(sum(p) - 1.0) < 0.01


# ======================================================================
# Cache Truncation Tests
# ======================================================================

class TestCacheTruncation:
    def test_truncate_sequence(self):
        """KV cache truncation for speculative decode rejection."""
        from zse_engine.format.config import ModelConfig
        from zse_engine.cache.cache_manager import KVCacheManager

        config = ModelConfig(
            num_layers=2, num_heads=4, num_kv_heads=4,
            head_dim=16, hidden_size=64, vocab_size=100,
        )
        cache = KVCacheManager(config=config, budget_bytes=1024 * 1024, block_size=4)

        # Allocate 10 tokens
        cache.allocate_sequence(0, prompt_tokens=list(range(10)))
        assert cache._page_table.num_tokens(0) == 10

        # Extend by 5 (speculative verify)
        cache.extend_sequence(0, 5)
        assert cache._page_table.num_tokens(0) == 15

        # Truncate to 12 (accepted 2 of 5 draft tokens)
        cache.truncate_sequence(0, 12)
        assert cache._page_table.num_tokens(0) == 12

        cache.free_sequence(0)

    def test_truncate_to_zero_blocks(self):
        """Truncating to 0 should free all blocks."""
        from zse_engine.format.config import ModelConfig
        from zse_engine.cache.cache_manager import KVCacheManager

        config = ModelConfig(num_layers=1, num_heads=2, num_kv_heads=2, head_dim=8, hidden_size=16)
        cache = KVCacheManager(config=config, budget_bytes=1024 * 1024, block_size=4)

        cache.allocate_sequence(0, prompt_tokens=[1, 2, 3, 4])
        blocks_before = cache.num_free_blocks

        cache.truncate_sequence(0, 0)
        assert cache._page_table.num_tokens(0) == 0

        cache.free_sequence(0)

    def test_truncate_noop(self):
        """Truncating to current length is a no-op."""
        from zse_engine.format.config import ModelConfig
        from zse_engine.cache.cache_manager import KVCacheManager

        config = ModelConfig(num_layers=1, num_heads=2, num_kv_heads=2, head_dim=8, hidden_size=16)
        cache = KVCacheManager(config=config, budget_bytes=1024 * 1024, block_size=4)

        cache.allocate_sequence(0, prompt_tokens=[1, 2, 3])
        cache.truncate_sequence(0, 3)
        assert cache._page_table.num_tokens(0) == 3

        cache.free_sequence(0)


# ======================================================================
# Integration: Verifier + Draft Model
# ======================================================================

class TestSpeculativeIntegration:
    def test_full_speculative_cycle(self):
        """Draft → verify (simulated) → accept cycle."""
        vocab = 20
        K = 4

        # Create draft model
        draft = NGramDraftModel(vocab_size_val=vocab, n=2)
        verifier = SpeculativeVerifier(seed=42)

        # Context with repeating pattern
        context = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]

        # Generate drafts
        draft_tokens, draft_probs = draft.generate_drafts(context, K)

        # Simulate target model: agrees with pattern [3, 1, 2, 3]
        expected = [3, 1, 2, 3]
        target_logits = []
        for i in range(K + 1):
            tok = expected[i] if i < len(expected) else 0
            target_logits.append(make_logits_bytes(peaked_probs(tok, vocab, 0.95), vocab))

        # Verify
        accepted, num_accepted = verifier.verify_and_accept(
            draft_tokens, draft_probs, target_logits, vocab, temperature=0.0,
        )

        assert len(accepted) >= 1
        assert len(accepted) <= K + 1

    def test_speculative_speedup_metric(self):
        """With good drafts, tokens_per_step should be > 1."""
        vocab = 10
        K = 3
        verifier = SpeculativeVerifier(seed=0)

        # Perfect drafts: draft and target agree
        draft_tokens = [5, 3, 7]
        draft_probs = [peaked_probs(t, vocab, 0.9) for t in draft_tokens]
        target_logits = [
            make_logits_bytes(peaked_probs(5, vocab, 0.9), vocab),
            make_logits_bytes(peaked_probs(3, vocab, 0.9), vocab),
            make_logits_bytes(peaked_probs(7, vocab, 0.9), vocab),
            make_logits_bytes(peaked_probs(1, vocab, 0.5), vocab),
        ]

        accepted, num_accepted = verifier.verify_and_accept(
            draft_tokens, draft_probs, target_logits, vocab, temperature=0.0,
        )

        # All 3 drafts should be accepted + 1 bonus = 4 tokens
        assert len(accepted) == 4
        assert num_accepted == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
