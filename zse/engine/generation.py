"""
ZSE Text Generation Module

Token-by-token text generation with streaming output.

Features:
- Multiple sampling strategies (greedy, top-k, top-p, temperature)
- Streaming output via generators
- Stop conditions (EOS, max tokens, stop sequences)
- Repetition penalty
- KV cache integration

Author: ZSE Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    Optional, Dict, List, Tuple, Any, 
    Iterator, AsyncIterator, Callable, Union
)
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time


@dataclass
class SamplingParams:
    """Parameters for token sampling."""
    
    # Basic sampling
    temperature: float = 1.0          # 0 = greedy, >1 = more random
    top_k: int = 50                   # 0 = disabled
    top_p: float = 0.9                # 1.0 = disabled (nucleus sampling)
    
    # Generation limits
    max_new_tokens: int = 128
    min_new_tokens: int = 0
    
    # Stopping
    stop_token_ids: List[int] = field(default_factory=list)
    stop_sequences: List[str] = field(default_factory=list)
    eos_token_id: Optional[int] = None
    
    # Repetition control
    repetition_penalty: float = 1.0   # >1 penalizes repetition
    presence_penalty: float = 0.0     # Penalize tokens that appear
    frequency_penalty: float = 0.0    # Penalize frequent tokens
    
    # Output control
    skip_special_tokens: bool = True
    
    def __post_init__(self):
        # Validation
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


@dataclass
class GenerationOutput:
    """Output from text generation."""
    text: str
    tokens: List[int]
    finish_reason: str  # "stop", "length", "eos"
    
    # Stats
    num_tokens: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    
    # For streaming
    is_finished: bool = False
    

@dataclass 
class StreamChunk:
    """A chunk of streamed output."""
    text: str
    token_id: int
    is_finished: bool = False
    finish_reason: Optional[str] = None
    
    # Timing
    latency_ms: float = 0.0


class Sampler:
    """
    Token sampler with various strategies.
    
    Supports:
    - Greedy (temperature=0)
    - Temperature scaling
    - Top-k filtering
    - Top-p (nucleus) sampling
    - Repetition penalty
    """
    
    def __init__(self, params: Optional[SamplingParams] = None):
        self.params = params or SamplingParams()
    
    def __call__(
        self,
        logits: torch.Tensor,
        generated_tokens: Optional[List[int]] = None,
    ) -> int:
        """
        Sample next token from logits.
        
        Args:
            logits: [vocab_size] logits from model
            generated_tokens: Previously generated tokens for repetition penalty
            
        Returns:
            Sampled token ID
        """
        return self.sample(logits, generated_tokens)
    
    def sample(
        self,
        logits: torch.Tensor,
        generated_tokens: Optional[List[int]] = None,
    ) -> int:
        """Sample a single token."""
        # Ensure 1D
        if logits.ndim > 1:
            logits = logits[-1]  # Take last position
        
        logits = logits.float()
        
        # Apply repetition penalty
        if generated_tokens and self.params.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, generated_tokens)
        
        # Apply frequency/presence penalties
        if generated_tokens and (self.params.frequency_penalty != 0 or self.params.presence_penalty != 0):
            logits = self._apply_frequency_penalty(logits, generated_tokens)
        
        # Temperature = 0 means greedy
        if self.params.temperature == 0:
            return logits.argmax().item()
        
        # Apply temperature
        logits = logits / self.params.temperature
        
        # Apply top-k
        if self.params.top_k > 0:
            logits = self._top_k_filter(logits, self.params.top_k)
        
        # Apply top-p (nucleus)
        if self.params.top_p < 1.0:
            logits = self._top_p_filter(logits, self.params.top_p)
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        
        return token_id
    
    def sample_batch(
        self,
        logits: torch.Tensor,
        generated_tokens: Optional[List[List[int]]] = None,
    ) -> List[int]:
        """Sample tokens for a batch."""
        batch_size = logits.shape[0]
        tokens = []
        
        for i in range(batch_size):
            prev_tokens = generated_tokens[i] if generated_tokens else None
            token = self.sample(logits[i], prev_tokens)
            tokens.append(token)
        
        return tokens
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        penalty = self.params.repetition_penalty
        
        for token_id in set(generated_tokens):
            if logits[token_id] < 0:
                logits[token_id] *= penalty
            else:
                logits[token_id] /= penalty
        
        return logits
    
    def _apply_frequency_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
    ) -> torch.Tensor:
        """Apply frequency and presence penalties."""
        from collections import Counter
        
        token_counts = Counter(generated_tokens)
        
        for token_id, count in token_counts.items():
            # Frequency penalty scales with count
            logits[token_id] -= self.params.frequency_penalty * count
            # Presence penalty is binary
            logits[token_id] -= self.params.presence_penalty
        
        return logits
    
    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k logits."""
        if k >= logits.shape[-1]:
            return logits
        
        values, _ = torch.topk(logits, k)
        min_value = values[-1]
        
        return torch.where(
            logits < min_value,
            torch.full_like(logits, float('-inf')),
            logits,
        )
    
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Keep tokens with cumulative probability <= p (nucleus sampling)."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find cutoff
        sorted_indices_to_remove = cumulative_probs > p
        # Keep at least one token
        sorted_indices_to_remove[0] = False
        
        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        
        logits[indices_to_remove] = float('-inf')
        return logits


class StopChecker:
    """Check stopping conditions for generation."""
    
    def __init__(
        self,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_sequences: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        tokenizer: Any = None,
    ):
        self.eos_token_id = eos_token_id
        self.stop_token_ids = set(stop_token_ids or [])
        self.stop_sequences = stop_sequences or []
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
    
    def should_stop(
        self,
        token_id: int,
        generated_tokens: List[int],
        generated_text: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check if generation should stop.
        
        Returns:
            (should_stop, reason)
        """
        # Check max tokens
        if len(generated_tokens) >= self.max_new_tokens:
            return True, "length"
        
        # Check EOS
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            return True, "eos"
        
        # Check stop token IDs
        if token_id in self.stop_token_ids:
            return True, "stop_token"
        
        # Check stop sequences (requires text)
        if generated_text and self.stop_sequences:
            for seq in self.stop_sequences:
                if generated_text.endswith(seq):
                    return True, "stop_sequence"
        
        return False, ""


class TextGenerator:
    """
    Text generator with streaming support.
    
    Usage:
        generator = TextGenerator(model, tokenizer)
        
        # Non-streaming
        output = generator.generate("Hello", max_new_tokens=50)
        
        # Streaming
        for chunk in generator.generate_stream("Hello"):
            print(chunk.text, end="", flush=True)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Get special tokens
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', self.eos_token_id)
    
    def generate(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
    ) -> GenerationOutput:
        """
        Generate text (non-streaming).
        
        Args:
            prompt: Input text
            params: Sampling parameters
            
        Returns:
            Complete generation output
        """
        params = params or SamplingParams()
        
        # Collect all chunks
        chunks = list(self.generate_stream(prompt, params))
        
        # Combine
        text = "".join(c.text for c in chunks)
        tokens = [c.token_id for c in chunks]
        
        finish_reason = chunks[-1].finish_reason if chunks else "length"
        total_time = sum(c.latency_ms for c in chunks) / 1000
        
        return GenerationOutput(
            text=text,
            tokens=tokens,
            finish_reason=finish_reason,
            num_tokens=len(tokens),
            generation_time=total_time,
            tokens_per_second=len(tokens) / total_time if total_time > 0 else 0,
            is_finished=True,
        )
    
    def generate_stream(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
    ) -> Iterator[StreamChunk]:
        """
        Generate text with streaming output.
        
        Yields:
            StreamChunk for each generated token
        """
        params = params or SamplingParams()
        
        # Encode prompt
        input_ids = self._encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Setup
        sampler = Sampler(params)
        stop_checker = StopChecker(
            eos_token_id=params.eos_token_id or self.eos_token_id,
            stop_token_ids=params.stop_token_ids,
            stop_sequences=params.stop_sequences,
            max_new_tokens=params.max_new_tokens,
            tokenizer=self.tokenizer,
        )
        
        generated_tokens: List[int] = []
        generated_text = ""
        
        # Generation loop with KV cache for 2x+ speedup
        with torch.no_grad():
            current_ids = input_ids
            past_key_values = None  # KV cache
            
            for _ in range(params.max_new_tokens):
                start_time = time.perf_counter()
                
                # Forward pass with KV cache
                output = self.model(
                    current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                # Handle different output formats (raw tensor vs HuggingFace output)
                if hasattr(output, 'logits'):
                    logits = output.logits
                    past_key_values = output.past_key_values
                else:
                    logits = output
                
                # Get logits for last position
                next_token_logits = logits[0, -1, :]
                
                # Sample next token
                next_token = sampler.sample(next_token_logits, generated_tokens)
                generated_tokens.append(next_token)
                
                # Decode token
                token_text = self._decode([next_token])
                generated_text += token_text
                
                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Check stopping
                should_stop, reason = stop_checker.should_stop(
                    next_token, generated_tokens, generated_text
                )
                
                # Yield chunk
                yield StreamChunk(
                    text=token_text,
                    token_id=next_token,
                    is_finished=should_stop,
                    finish_reason=reason if should_stop else None,
                    latency_ms=latency_ms,
                )
                
                if should_stop:
                    break
                
                # Only pass the new token (KV cache has the rest)
                current_ids = torch.tensor([[next_token]], device=self.device)
    
    async def generate_stream_async(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Async streaming generation.
        
        Yields:
            StreamChunk for each generated token
        """
        for chunk in self.generate_stream(prompt, params):
            yield chunk
            # Allow other tasks to run
            await asyncio.sleep(0)
    
    def _encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text)
        elif hasattr(self.tokenizer, '__call__'):
            return self.tokenizer(text)['input_ids']
        else:
            raise ValueError("Tokenizer must have encode() or __call__() method")
    
    def _decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        else:
            raise ValueError("Tokenizer must have decode() method")


class StreamingCallback:
    """
    Callback interface for streaming generation.
    
    Subclass this to customize streaming behavior:
    - on_token: Called for each generated token
    - on_finish: Called when generation completes
    """
    
    def on_token(self, chunk: StreamChunk):
        """Called for each generated token."""
        pass
    
    def on_finish(self, output: GenerationOutput):
        """Called when generation finishes."""
        pass


class PrintStreamCallback(StreamingCallback):
    """Print tokens as they're generated."""
    
    def __init__(self, end: str = "", flush: bool = True):
        self.end = end
        self.flush = flush
        self.all_text = ""
    
    def on_token(self, chunk: StreamChunk):
        print(chunk.text, end=self.end, flush=self.flush)
        self.all_text += chunk.text
    
    def on_finish(self, output: GenerationOutput):
        print()  # Newline at end


# =============================================================================
# GENERATION WITH KV CACHE
# =============================================================================

class CachedTextGenerator(TextGenerator):
    """
    Text generator with KV cache for efficient generation.
    
    Uses KV cache to avoid recomputing attention for previous tokens.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        kv_cache_manager: Any,  # KVCacheManager
        device: str = "cuda",
    ):
        super().__init__(model, tokenizer, device)
        self.kv_manager = kv_cache_manager
        self._cache_seq_id: Optional[int] = None
    
    def generate_stream(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
    ) -> Iterator[StreamChunk]:
        """Generate with KV cache."""
        params = params or SamplingParams()
        
        # Encode prompt
        prompt_tokens = self._encode(prompt)
        
        # Allocate KV cache
        self._cache_seq_id = self.kv_manager.create_cache(
            batch_size=1,
            prompt_len=len(prompt_tokens),
        )
        
        try:
            # Setup
            sampler = Sampler(params)
            stop_checker = StopChecker(
                eos_token_id=params.eos_token_id or self.eos_token_id,
                stop_token_ids=params.stop_token_ids,
                stop_sequences=params.stop_sequences,
                max_new_tokens=params.max_new_tokens,
            )
            
            generated_tokens: List[int] = []
            generated_text = ""
            
            # Prefill: process entire prompt
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                # Prefill pass
                logits = self._forward_with_cache(input_ids, is_prefill=True)
                next_token_logits = logits[0, -1, :]
                
                # Sample first token
                next_token = sampler.sample(next_token_logits, generated_tokens)
                generated_tokens.append(next_token)
                token_text = self._decode([next_token])
                generated_text += token_text
                
                yield StreamChunk(
                    text=token_text,
                    token_id=next_token,
                    is_finished=False,
                )
                
                # Decode: generate one token at a time
                for _ in range(params.max_new_tokens - 1):
                    start_time = time.perf_counter()
                    
                    # Single token input
                    input_ids = torch.tensor([[next_token]], device=self.device)
                    
                    # Forward with cache
                    logits = self._forward_with_cache(input_ids, is_prefill=False)
                    next_token_logits = logits[0, -1, :]
                    
                    # Sample
                    next_token = sampler.sample(next_token_logits, generated_tokens)
                    generated_tokens.append(next_token)
                    token_text = self._decode([next_token])
                    generated_text += token_text
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Check stopping
                    should_stop, reason = stop_checker.should_stop(
                        next_token, generated_tokens, generated_text
                    )
                    
                    yield StreamChunk(
                        text=token_text,
                        token_id=next_token,
                        is_finished=should_stop,
                        finish_reason=reason if should_stop else None,
                        latency_ms=latency_ms,
                    )
                    
                    if should_stop:
                        break
        
        finally:
            # Free KV cache
            if self._cache_seq_id is not None:
                self.kv_manager.free_cache(self._cache_seq_id)
                self._cache_seq_id = None
    
    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        """
        Forward pass using KV cache.
        
        Override this for models with actual KV cache support.
        Default implementation just does regular forward.
        
        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        # For models without built-in cache, just do regular forward
        output = self.model(input_ids)
        
        # Handle different output formats (raw tensor vs HuggingFace output)
        if hasattr(output, 'logits'):
            return output.logits
        return output


# =============================================================================
# BATCH GENERATION
# =============================================================================

@dataclass
class BatchGenerationRequest:
    """A single request in a batch."""
    request_id: str
    prompt: str
    params: SamplingParams = field(default_factory=SamplingParams)


@dataclass
class BatchGenerationOutput:
    """Output for batch generation."""
    request_id: str
    output: GenerationOutput


class BatchGenerator:
    """
    Generate text for multiple prompts efficiently.
    
    Batches prompts together for better GPU utilization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda",
        max_batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
    
    def generate_batch(
        self,
        requests: List[BatchGenerationRequest],
    ) -> List[BatchGenerationOutput]:
        """
        Generate for multiple requests.
        
        Args:
            requests: List of generation requests
            
        Returns:
            List of outputs (same order as requests)
        """
        outputs = []
        
        # Process in batches
        for i in range(0, len(requests), self.max_batch_size):
            batch = requests[i:i + self.max_batch_size]
            batch_outputs = self._generate_batch(batch)
            outputs.extend(batch_outputs)
        
        return outputs
    
    def _generate_batch(
        self,
        requests: List[BatchGenerationRequest],
    ) -> List[BatchGenerationOutput]:
        """Generate for a single batch."""
        # For simplicity, process one at a time
        # In production, use proper batched attention
        outputs = []
        
        generator = TextGenerator(self.model, self.tokenizer, self.device)
        
        for req in requests:
            output = generator.generate(req.prompt, req.params)
            outputs.append(BatchGenerationOutput(
                request_id=req.request_id,
                output=output,
            ))
        
        return outputs
