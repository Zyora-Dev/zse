"""
Tests for ZSE Text Generation Module

Tests cover:
- Sampling strategies (greedy, top-k, top-p, temperature)
- Stop conditions (EOS, max tokens, stop sequences)
- Repetition penalty
- Streaming output
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
from typing import List

from zse.engine.generation import (
    SamplingParams,
    GenerationOutput,
    StreamChunk,
    Sampler,
    StopChecker,
    TextGenerator,
    CachedTextGenerator,
    StreamingCallback,
    PrintStreamCallback,
    BatchGenerator,
    BatchGenerationRequest,
    BatchGenerationOutput,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def vocab_size():
    return 100


@pytest.fixture
def mock_model(vocab_size):
    """Create a mock model that returns predictable logits."""
    model = Mock()
    
    def forward(input_ids):
        batch_size, seq_len = input_ids.shape
        # Return logits with token 42 as highest
        logits = torch.randn(batch_size, seq_len, vocab_size)
        logits[:, :, 42] = 10.0  # Make token 42 most likely
        return logits
    
    model.side_effect = forward
    model.__call__ = forward
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock()
    tokenizer.encode = Mock(return_value=[1, 2, 3])  # "Hello" -> [1, 2, 3]
    tokenizer.decode = Mock(side_effect=lambda ids, **kw: f"<{ids[0]}>")
    tokenizer.eos_token_id = 0
    tokenizer.pad_token_id = 0
    return tokenizer


# =============================================================================
# SAMPLING PARAMS TESTS
# =============================================================================

class TestSamplingParams:
    """Tests for SamplingParams."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = SamplingParams()
        
        assert params.temperature == 1.0
        assert params.top_k == 50
        assert params.top_p == 0.9
        assert params.max_new_tokens == 128
        assert params.repetition_penalty == 1.0
    
    def test_custom_params(self):
        """Test custom parameter values."""
        params = SamplingParams(
            temperature=0.7,
            top_k=40,
            top_p=0.95,
            max_new_tokens=256,
        )
        
        assert params.temperature == 0.7
        assert params.top_k == 40
        assert params.top_p == 0.95
        assert params.max_new_tokens == 256
    
    def test_invalid_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-1.0)
    
    def test_invalid_top_k(self):
        """Test that negative top_k raises error."""
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=-1)
    
    def test_invalid_top_p(self):
        """Test that invalid top_p raises error."""
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)
        
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.5)


# =============================================================================
# SAMPLER TESTS
# =============================================================================

class TestSampler:
    """Tests for token sampling."""
    
    def test_greedy_sampling(self, vocab_size):
        """Test greedy sampling (temperature=0)."""
        params = SamplingParams(temperature=0)
        sampler = Sampler(params)
        
        # Create logits with clear winner
        logits = torch.randn(vocab_size)
        logits[42] = 100.0
        
        token = sampler.sample(logits)
        assert token == 42
    
    def test_greedy_batch(self, vocab_size):
        """Test greedy on batch."""
        params = SamplingParams(temperature=0)
        sampler = Sampler(params)
        
        # Batch of 4
        logits = torch.randn(4, vocab_size)
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0
        logits[2, 30] = 100.0
        logits[3, 40] = 100.0
        
        tokens = sampler.sample_batch(logits)
        assert tokens == [10, 20, 30, 40]
    
    def test_temperature_scaling(self, vocab_size):
        """Test that higher temperature increases randomness."""
        torch.manual_seed(42)
        
        # Create clear winner with very high logit difference
        logits = torch.zeros(vocab_size)
        logits[0] = 10.0  # Much higher than others
        logits[1] = 0.0
        
        # Very low temperature - should almost always pick token 0
        low_temp = SamplingParams(temperature=0.01, top_k=0, top_p=1.0)
        sampler_low = Sampler(low_temp)
        
        low_samples = [sampler_low.sample(logits.clone()) for _ in range(100)]
        assert sum(s == 0 for s in low_samples) >= 99  # Almost always token 0
        
        # High temperature - more varied
        high_temp = SamplingParams(temperature=5.0, top_k=0, top_p=1.0)
        sampler_high = Sampler(high_temp)
        
        # With high temp, distribution becomes more uniform
        high_samples = [sampler_high.sample(logits.clone()) for _ in range(100)]
        assert sum(s == 0 for s in high_samples) < 99  # Not all token 0
    
    def test_top_k_filtering(self, vocab_size):
        """Test top-k filtering."""
        torch.manual_seed(42)
        
        params = SamplingParams(top_k=3, temperature=1.0, top_p=1.0)
        sampler = Sampler(params)
        
        # Create logits where only top 3 should be sampled
        logits = torch.zeros(vocab_size)
        logits[10] = 5.0
        logits[20] = 4.0
        logits[30] = 3.0
        logits[40] = 2.0  # Should never be sampled
        
        samples = set()
        for _ in range(100):
            samples.add(sampler.sample(logits.clone()))
        
        # Should only sample from top 3
        assert samples.issubset({10, 20, 30})
    
    def test_top_p_filtering(self, vocab_size):
        """Test top-p (nucleus) sampling filters low probability tokens."""
        torch.manual_seed(42)
        
        # The key test: with top_p filtering, low probability tokens should be excluded
        params = SamplingParams(top_p=0.9, temperature=0.5, top_k=0)
        sampler = Sampler(params)
        
        # Create very skewed distribution
        logits = torch.full((vocab_size,), -100.0)  # Very low
        logits[0] = 10.0  # Very high - will dominate
        logits[1] = 5.0   # Medium
        
        samples = set()
        for _ in range(50):
            samples.add(sampler.sample(logits.clone()))
        
        # Should only sample from top probability tokens
        # Token 0 and 1 should be the only ones with significant probability
        low_prob_tokens = [s for s in samples if s >= 2]
        assert len(low_prob_tokens) == 0  # No low probability tokens
    
    def test_repetition_penalty(self, vocab_size):
        """Test repetition penalty reduces repeated token probability."""
        params = SamplingParams(
            temperature=0,
            repetition_penalty=2.0,
        )
        sampler = Sampler(params)
        
        # Token 42 is best, but penalized
        logits = torch.zeros(vocab_size)
        logits[42] = 5.0
        logits[43] = 4.0
        
        # Without penalty, would pick 42
        no_penalty = Sampler(SamplingParams(temperature=0))
        assert no_penalty.sample(logits.clone()) == 42
        
        # With penalty and 42 in history, should pick 43
        generated = [42, 42, 42]
        token = sampler.sample(logits.clone(), generated)
        assert token == 43
    
    def test_frequency_penalty(self, vocab_size):
        """Test frequency penalty."""
        params = SamplingParams(
            temperature=0,
            frequency_penalty=1.0,
            repetition_penalty=1.0,
        )
        sampler = Sampler(params)
        
        logits = torch.zeros(vocab_size)
        logits[10] = 5.0
        logits[20] = 4.8
        
        # Token 10 appears 3 times, gets penalty of 3*1.0 = 3.0
        generated = [10, 10, 10]
        token = sampler.sample(logits.clone(), generated)
        
        # 10's logit: 5.0 - 3.0 = 2.0, 20's logit: 4.8 - 0 = 4.8
        assert token == 20
    
    def test_presence_penalty(self, vocab_size):
        """Test presence penalty."""
        params = SamplingParams(
            temperature=0,
            presence_penalty=2.0,
            repetition_penalty=1.0,
        )
        sampler = Sampler(params)
        
        logits = torch.zeros(vocab_size)
        logits[10] = 5.0
        logits[20] = 4.0
        
        # Token 10 present once, gets penalty of 2.0
        generated = [10]
        token = sampler.sample(logits.clone(), generated)
        
        # 10's logit: 5.0 - 2.0 = 3.0, 20's logit: 4.0 - 0 = 4.0
        assert token == 20
    
    def test_callable_sampler(self, vocab_size):
        """Test sampler can be called directly."""
        params = SamplingParams(temperature=0)
        sampler = Sampler(params)
        
        logits = torch.zeros(vocab_size)
        logits[99] = 100.0
        
        # Call directly
        token = sampler(logits)
        assert token == 99


# =============================================================================
# STOP CHECKER TESTS
# =============================================================================

class TestStopChecker:
    """Tests for stop condition checking."""
    
    def test_max_tokens_stop(self):
        """Test stopping at max tokens."""
        checker = StopChecker(max_new_tokens=5)
        
        # Not stopped yet
        should_stop, reason = checker.should_stop(42, [1, 2, 3, 4])
        assert not should_stop
        
        # Now at limit
        should_stop, reason = checker.should_stop(42, [1, 2, 3, 4, 5])
        assert should_stop
        assert reason == "length"
    
    def test_eos_token_stop(self):
        """Test stopping at EOS token."""
        checker = StopChecker(eos_token_id=0, max_new_tokens=100)
        
        # Regular token
        should_stop, reason = checker.should_stop(42, [1, 2, 3])
        assert not should_stop
        
        # EOS token
        should_stop, reason = checker.should_stop(0, [1, 2, 3])
        assert should_stop
        assert reason == "eos"
    
    def test_stop_token_ids(self):
        """Test stopping at custom stop tokens."""
        checker = StopChecker(
            stop_token_ids=[10, 20, 30],
            max_new_tokens=100,
        )
        
        # Regular token
        should_stop, _ = checker.should_stop(42, [1, 2, 3])
        assert not should_stop
        
        # Stop token
        should_stop, reason = checker.should_stop(20, [1, 2, 3])
        assert should_stop
        assert reason == "stop_token"
    
    def test_stop_sequences(self):
        """Test stopping at text sequences."""
        checker = StopChecker(
            stop_sequences=["</s>", "\n\n"],
            max_new_tokens=100,
        )
        
        # No match
        should_stop, _ = checker.should_stop(42, [1], "Hello world")
        assert not should_stop
        
        # Match sequence
        should_stop, reason = checker.should_stop(42, [1], "Hello</s>")
        assert should_stop
        assert reason == "stop_sequence"
    
    def test_priority_order(self):
        """Test that length is checked first."""
        checker = StopChecker(
            eos_token_id=0,
            max_new_tokens=3,
        )
        
        # At length limit, even with regular token
        should_stop, reason = checker.should_stop(42, [1, 2, 3])
        assert should_stop
        assert reason == "length"


# =============================================================================
# TEXT GENERATOR TESTS
# =============================================================================

class TestTextGenerator:
    """Tests for text generation."""
    
    def test_generate_basic(self, mock_model, mock_tokenizer):
        """Test basic generation."""
        generator = TextGenerator(mock_model, mock_tokenizer, device="cpu")
        
        params = SamplingParams(
            temperature=0,
            max_new_tokens=5,
        )
        
        output = generator.generate("Hello", params)
        
        assert isinstance(output, GenerationOutput)
        assert len(output.tokens) == 5
        assert output.num_tokens == 5
        assert output.is_finished
    
    def test_generate_stream(self, mock_model, mock_tokenizer):
        """Test streaming generation."""
        generator = TextGenerator(mock_model, mock_tokenizer, device="cpu")
        
        params = SamplingParams(
            temperature=0,
            max_new_tokens=5,
        )
        
        chunks = list(generator.generate_stream("Hello", params))
        
        assert len(chunks) == 5
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, StreamChunk)
            assert chunk.token_id == 42  # Mock returns 42 as highest
        
        # Last chunk should be finished
        assert chunks[-1].is_finished
        assert chunks[-1].finish_reason == "length"
    
    def test_generate_with_eos_stop(self, mock_tokenizer):
        """Test that generation stops at EOS."""
        # Create model that returns EOS after 3 tokens
        call_count = [0]
        
        class EosModel(nn.Module):
            def forward(self, input_ids):
                call_count[0] += 1
                batch_size, seq_len = input_ids.shape
                logits = torch.randn(batch_size, seq_len, 100)
                if call_count[0] >= 4:  # After 3 tokens
                    logits[:, :, 0] = 100.0  # EOS
                else:
                    logits[:, :, 42] = 100.0
                return logits
        
        model = EosModel()
        
        generator = TextGenerator(model, mock_tokenizer, device="cpu")
        
        params = SamplingParams(
            temperature=0,
            max_new_tokens=100,
            eos_token_id=0,
        )
        
        output = generator.generate("Hello", params)
        
        # Should stop at EOS, not max tokens
        assert output.finish_reason == "eos"
        assert output.num_tokens < 100
    
    def test_streaming_latency_tracking(self, mock_model, mock_tokenizer):
        """Test that streaming tracks latency."""
        generator = TextGenerator(mock_model, mock_tokenizer, device="cpu")
        
        params = SamplingParams(temperature=0, max_new_tokens=3)
        
        chunks = list(generator.generate_stream("Hello", params))
        
        for chunk in chunks:
            assert chunk.latency_ms >= 0


# =============================================================================
# CALLBACK TESTS  
# =============================================================================

class TestStreamingCallback:
    """Tests for streaming callbacks."""
    
    def test_print_stream_callback(self, capsys, mock_model, mock_tokenizer):
        """Test print callback captures output."""
        generator = TextGenerator(mock_model, mock_tokenizer, device="cpu")
        callback = PrintStreamCallback()
        
        params = SamplingParams(temperature=0, max_new_tokens=3)
        
        for chunk in generator.generate_stream("Hello", params):
            callback.on_token(chunk)
        
        # Check accumulated text
        assert len(callback.all_text) > 0


# =============================================================================
# BATCH GENERATION TESTS
# =============================================================================

class TestBatchGenerator:
    """Tests for batch generation."""
    
    def test_batch_generate(self, mock_model, mock_tokenizer):
        """Test batch generation."""
        generator = BatchGenerator(
            mock_model, 
            mock_tokenizer, 
            device="cpu",
            max_batch_size=2,
        )
        
        requests = [
            BatchGenerationRequest(
                request_id="req_1",
                prompt="Hello",
                params=SamplingParams(temperature=0, max_new_tokens=3),
            ),
            BatchGenerationRequest(
                request_id="req_2",
                prompt="World",
                params=SamplingParams(temperature=0, max_new_tokens=3),
            ),
        ]
        
        outputs = generator.generate_batch(requests)
        
        assert len(outputs) == 2
        assert outputs[0].request_id == "req_1"
        assert outputs[1].request_id == "req_2"
        
        for out in outputs:
            assert isinstance(out.output, GenerationOutput)
            assert out.output.num_tokens == 3


# =============================================================================
# CACHED GENERATOR TESTS
# =============================================================================

class TestCachedTextGenerator:
    """Tests for KV-cached generation."""
    
    def test_cached_generation_basic(self, mock_model, mock_tokenizer):
        """Test basic cached generation."""
        # Mock KV cache manager
        kv_manager = Mock()
        kv_manager.create_cache = Mock(return_value=0)
        kv_manager.free_cache = Mock()
        
        generator = CachedTextGenerator(
            mock_model, 
            mock_tokenizer, 
            kv_manager,
            device="cpu",
        )
        
        params = SamplingParams(temperature=0, max_new_tokens=3)
        
        chunks = list(generator.generate_stream("Hello", params))
        
        assert len(chunks) == 3
        assert kv_manager.create_cache.called
        assert kv_manager.free_cache.called
    
    def test_cache_freed_on_error(self, mock_tokenizer):
        """Test that cache is freed even on error."""
        kv_manager = Mock()
        kv_manager.create_cache = Mock(return_value=0)
        kv_manager.free_cache = Mock()
        
        # Model that raises error
        class FailingModel(nn.Module):
            def forward(self, input_ids):
                raise RuntimeError("Test error")
        
        model = FailingModel()
        
        generator = CachedTextGenerator(
            model,
            mock_tokenizer,
            kv_manager,
            device="cpu",
        )
        
        params = SamplingParams(temperature=0, max_new_tokens=3)
        
        with pytest.raises(RuntimeError):
            list(generator.generate_stream("Hello", params))
        
        # Cache should still be freed
        assert kv_manager.free_cache.called


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestGenerationIntegration:
    """Integration tests for generation."""
    
    def test_full_generation_pipeline(self):
        """Test complete generation pipeline with simple model."""
        # Simple language model for testing
        class SimpleModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, 64)
                self.proj = nn.Linear(64, vocab_size)
            
            def forward(self, input_ids):
                x = self.embed(input_ids)
                return self.proj(x)
        
        vocab_size = 100
        model = SimpleModel(vocab_size)
        
        # Simple tokenizer
        class SimpleTokenizer:
            def __init__(self):
                self.eos_token_id = 0
                self.pad_token_id = 0
            
            def encode(self, text):
                return [ord(c) % vocab_size for c in text[:10]]
            
            def decode(self, ids, **kwargs):
                return "".join(chr(i + 65) for i in ids)
        
        tokenizer = SimpleTokenizer()
        generator = TextGenerator(model, tokenizer, device="cpu")
        
        params = SamplingParams(
            temperature=0.8,
            top_k=10,
            max_new_tokens=20,
        )
        
        output = generator.generate("Test", params)
        
        assert output.num_tokens > 0
        assert output.num_tokens <= 20
        assert len(output.text) > 0
    
    def test_deterministic_greedy(self):
        """Test that greedy is deterministic."""
        class DeterministicModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0
            
            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                logits = torch.zeros(batch_size, seq_len, 100)
                # Predictable sequence: 10, 20, 30, 40, 50
                token = ((self.counter % 5) + 1) * 10
                logits[:, :, token] = 100.0
                self.counter += 1
                return logits
        
        class SimpleTokenizer:
            eos_token_id = 0
            pad_token_id = 0
            
            def encode(self, text):
                return [1, 2, 3]
            
            def decode(self, ids, **kwargs):
                return str(ids[0])
        
        model = DeterministicModel()
        tokenizer = SimpleTokenizer()
        
        generator = TextGenerator(model, tokenizer, device="cpu")
        params = SamplingParams(temperature=0, max_new_tokens=5)
        
        output1 = generator.generate("Hello", params)
        
        model.counter = 0
        output2 = generator.generate("Hello", params)
        
        assert output1.tokens == output2.tokens
