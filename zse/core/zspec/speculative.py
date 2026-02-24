"""
ZSE Speculative Decoding - zSpec

Accelerates autoregressive generation by using a fast draft model
to generate multiple tokens, then verifying them in parallel with
the target model.

Key benefits:
- 2-3x faster generation for suitable model pairs
- No quality degradation (mathematically equivalent output)
- Works with any draft/target model pair

How it works:
1. Draft model generates K candidate tokens autoregressively
2. Target model verifies all K tokens in single forward pass
3. Accept tokens until first mismatch
4. Resample from combined distribution at rejection point

Model pairing:
- Draft: Small/quantized version (e.g., 1B params)
- Target: Full model (e.g., 7B params)
- Same tokenizer required

Author: ZSE Team
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    # Number of tokens to speculate per step
    num_speculative_tokens: int = 5
    
    # Acceptance threshold (0-1, higher = stricter)
    acceptance_threshold: float = 0.0
    
    # Use tree-based speculation (multiple branches)
    use_tree_attention: bool = False
    
    # Max tree width for tree speculation  
    tree_width: int = 2
    
    # Fallback to normal decoding if acceptance rate < threshold
    min_acceptance_rate: float = 0.3
    
    # Number of steps before adapting speculation length
    adaptation_window: int = 100
    
    # Whether to use target model KV cache for draft
    share_kv_cache: bool = False


@dataclass
class SpeculativeOutput:
    """Output from speculative decoding step."""
    # Accepted token IDs
    token_ids: torch.Tensor
    
    # Number of tokens accepted (including resampled)
    num_accepted: int
    
    # Acceptance rate for this step
    acceptance_rate: float
    
    # Time statistics
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    total_time_ms: float = 0.0


class SpeculativeDecoder:
    """
    Speculative decoding engine.
    
    Uses a fast draft model to generate multiple candidate tokens,
    then verifies them efficiently with the target model.
    
    Usage:
        decoder = SpeculativeDecoder(
            target_model=llama_7b,
            draft_model=llama_1b,  # or quantized version
        )
        
        # Generate with speculation
        for output in decoder.generate(prompt_ids, max_tokens=100):
            print(tokenizer.decode(output.token_ids))
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        config: Optional[SpeculativeConfig] = None,
        target_device: str = "cuda:0",
        draft_device: Optional[str] = None,  # None = same as target
    ):
        """
        Initialize speculative decoder.
        
        Args:
            target_model: Large target model (verifier)
            draft_model: Small draft model (proposer)
            config: Speculation configuration
            target_device: Device for target model
            draft_device: Device for draft model
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config or SpeculativeConfig()
        self.target_device = target_device
        self.draft_device = draft_device or target_device
        
        # Move models to devices
        self.target_model = self.target_model.to(target_device)
        self.draft_model = self.draft_model.to(self.draft_device)
        
        # Set to eval mode
        self.target_model.eval()
        self.draft_model.eval()
        
        # Statistics for adaptive speculation
        self._acceptance_history: List[float] = []
        self._adaptive_k = self.config.num_speculative_tokens
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Stats
        self.stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "speculation_steps": 0,
            "avg_acceptance_rate": 0.0,
            "avg_tokens_per_step": 0.0,
            "draft_time_ms": 0.0,
            "verify_time_ms": 0.0,
        }
    
    def _draft_generate(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
        temperature: float = 1.0,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Generate draft tokens using the small model.
        
        Args:
            input_ids: Current token sequence [batch, seq_len]
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            past_key_values: KV cache from previous steps
            
        Returns:
            (draft_tokens, draft_probs, new_past_key_values)
        """
        draft_tokens = []
        draft_probs = []
        
        current_input = input_ids.to(self.draft_device)
        current_past = past_key_values
        
        with torch.no_grad():
            for _ in range(num_tokens):
                # Forward pass
                outputs = self.draft_model(
                    input_ids=current_input if current_past is None else current_input[:, -1:],
                    past_key_values=current_past,
                    use_cache=True,
                )
                
                logits = outputs.logits[:, -1, :]  # [batch, vocab]
                current_past = outputs.past_key_values
                
                # Sample
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                    probs = F.softmax(logits, dim=-1)
                
                draft_tokens.append(next_token)
                # Store probability of sampled token
                draft_probs.append(probs.gather(1, next_token))
                
                current_input = torch.cat([current_input, next_token], dim=-1)
        
        # Stack results
        draft_tokens = torch.cat(draft_tokens, dim=1)  # [batch, num_tokens]
        draft_probs = torch.cat(draft_probs, dim=1)    # [batch, num_tokens]
        
        return draft_tokens, draft_probs, current_past
    
    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        temperature: float = 1.0,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, int, Any]:
        """
        Verify draft tokens using target model.
        
        Uses speculative sampling algorithm to accept/reject tokens
        while maintaining correct output distribution.
        
        Args:
            input_ids: Original input sequence
            draft_tokens: Draft tokens to verify [batch, k]
            draft_probs: Draft model probabilities [batch, k]
            temperature: Sampling temperature
            past_key_values: Target model KV cache
            
        Returns:
            (accepted_tokens, num_accepted, new_past_key_values)
        """
        batch_size = input_ids.shape[0]
        num_draft = draft_tokens.shape[1]
        
        # Concatenate input with draft tokens for parallel verification
        verify_ids = torch.cat([
            input_ids.to(self.target_device),
            draft_tokens.to(self.target_device),
        ], dim=1)
        
        with torch.no_grad():
            # Single forward pass to verify all tokens
            outputs = self.target_model(
                input_ids=verify_ids if past_key_values is None else verify_ids[:, -(num_draft + 1):],
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            # Get target probabilities for draft positions
            # logits shape: [batch, seq_len, vocab]
            # We need probs at positions where draft tokens were placed
            target_logits = outputs.logits[:, -(num_draft + 1):, :]
            
            if temperature > 0:
                target_probs = F.softmax(target_logits / temperature, dim=-1)
            else:
                target_probs = F.softmax(target_logits, dim=-1)
            
            new_past = outputs.past_key_values
        
        # Speculative sampling: accept/reject each draft token
        accepted_tokens = []
        num_accepted = 0
        
        for i in range(num_draft):
            # Get probabilities for this position
            draft_token = draft_tokens[:, i:i+1]  # [batch, 1]
            
            # Target prob of draft token at this position
            target_p = target_probs[:, i, :].gather(1, draft_token.to(self.target_device))
            draft_p = draft_probs[:, i:i+1].to(self.target_device)
            
            # Acceptance probability: min(1, target_p / draft_p)
            acceptance_prob = torch.clamp(target_p / (draft_p + 1e-10), max=1.0)
            
            # Sample uniform for acceptance test
            uniform = torch.rand_like(acceptance_prob)
            accept = uniform < acceptance_prob
            
            if accept.all():
                accepted_tokens.append(draft_token.to(self.target_device))
                num_accepted += 1
            else:
                # Rejection - resample from adjusted distribution
                # p_adjusted = max(0, target_p - draft_p) / Z
                adjusted = F.relu(target_probs[:, i, :] - F.softmax(
                    self.draft_model(
                        input_ids=torch.cat([input_ids.to(self.draft_device)] + 
                                           [t.to(self.draft_device) for t in accepted_tokens], dim=1)
                        if accepted_tokens else input_ids.to(self.draft_device),
                        use_cache=False,
                    ).logits[:, -1, :].to(self.target_device) / temperature, dim=-1))
                
                # Normalize
                adjusted = adjusted / (adjusted.sum(dim=-1, keepdim=True) + 1e-10)
                
                # Resample
                if adjusted.sum() > 0:
                    resampled = torch.multinomial(adjusted, num_samples=1)
                else:
                    # Fallback to target distribution
                    resampled = torch.multinomial(target_probs[:, i, :], num_samples=1)
                
                accepted_tokens.append(resampled)
                num_accepted += 1
                break  # Stop at first rejection
        
        # If all accepted, sample one more from target
        if num_accepted == num_draft:
            bonus_token = torch.multinomial(target_probs[:, -1, :], num_samples=1)
            accepted_tokens.append(bonus_token)
            num_accepted += 1
        
        # Concatenate accepted tokens
        if accepted_tokens:
            accepted = torch.cat(accepted_tokens, dim=1)
        else:
            accepted = torch.empty((batch_size, 0), dtype=torch.long, device=self.target_device)
        
        return accepted, num_accepted, new_past
    
    def _adapt_speculation_length(self) -> None:
        """Adapt number of speculative tokens based on acceptance rate."""
        if len(self._acceptance_history) < self.config.adaptation_window:
            return
        
        recent_rate = sum(self._acceptance_history[-self.config.adaptation_window:]) / self.config.adaptation_window
        
        if recent_rate > 0.8 and self._adaptive_k < 10:
            # High acceptance - try more tokens
            self._adaptive_k += 1
        elif recent_rate < 0.4 and self._adaptive_k > 2:
            # Low acceptance - try fewer tokens
            self._adaptive_k -= 1
    
    def speculative_step(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        draft_past: Optional[Any] = None,
        target_past: Optional[Any] = None,
    ) -> Tuple[SpeculativeOutput, Any, Any]:
        """
        Single speculative decoding step.
        
        Args:
            input_ids: Current sequence [batch, seq_len]
            temperature: Sampling temperature
            draft_past: Draft model KV cache
            target_past: Target model KV cache
            
        Returns:
            (output, new_draft_past, new_target_past)
        """
        start_time = time.perf_counter()
        
        # Draft phase
        draft_start = time.perf_counter()
        draft_tokens, draft_probs, new_draft_past = self._draft_generate(
            input_ids,
            num_tokens=self._adaptive_k,
            temperature=temperature,
            past_key_values=draft_past,
        )
        draft_time = (time.perf_counter() - draft_start) * 1000
        
        # Verify phase
        verify_start = time.perf_counter()
        accepted_tokens, num_accepted, new_target_past = self._verify_tokens(
            input_ids,
            draft_tokens,
            draft_probs,
            temperature=temperature,
            past_key_values=target_past,
        )
        verify_time = (time.perf_counter() - verify_start) * 1000
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate acceptance rate
        acceptance_rate = num_accepted / (self._adaptive_k + 1)
        self._acceptance_history.append(acceptance_rate)
        
        # Update stats
        with self._lock:
            self.stats["total_tokens"] += num_accepted
            self.stats["accepted_tokens"] += min(num_accepted, self._adaptive_k)
            self.stats["speculation_steps"] += 1
            self.stats["draft_time_ms"] += draft_time
            self.stats["verify_time_ms"] += verify_time
            
            if self.stats["speculation_steps"] > 0:
                self.stats["avg_acceptance_rate"] = (
                    self.stats["accepted_tokens"] / 
                    (self.stats["speculation_steps"] * self._adaptive_k)
                )
                self.stats["avg_tokens_per_step"] = (
                    self.stats["total_tokens"] / self.stats["speculation_steps"]
                )
        
        # Adapt speculation length
        self._adapt_speculation_length()
        
        output = SpeculativeOutput(
            token_ids=accepted_tokens,
            num_accepted=num_accepted,
            acceptance_rate=acceptance_rate,
            draft_time_ms=draft_time,
            verify_time_ms=verify_time,
            total_time_ms=total_time,
        )
        
        return output, new_draft_past, new_target_past
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        stop_token_id: Optional[int] = None,
        **kwargs,
    ):
        """
        Generate tokens using speculative decoding.
        
        Yields SpeculativeOutput for each step.
        
        Args:
            input_ids: Initial token sequence [1, seq_len]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_token_id: EOS token ID
            
        Yields:
            SpeculativeOutput for each generation step
        """
        current_ids = input_ids.clone()
        total_generated = 0
        
        draft_past = None
        target_past = None
        
        while total_generated < max_tokens:
            output, draft_past, target_past = self.speculative_step(
                current_ids,
                temperature=temperature,
                draft_past=draft_past,
                target_past=target_past,
            )
            
            # Update sequence
            current_ids = torch.cat([current_ids, output.token_ids], dim=1)
            total_generated += output.num_accepted
            
            yield output
            
            # Check for stop token
            if stop_token_id is not None:
                if (output.token_ids == stop_token_id).any():
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get speculation statistics."""
        return {
            **self.stats,
            "current_k": self._adaptive_k,
            "acceptance_history_len": len(self._acceptance_history),
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self.stats = {
                "total_tokens": 0,
                "accepted_tokens": 0,
                "speculation_steps": 0,
                "avg_acceptance_rate": 0.0,
                "avg_tokens_per_step": 0.0,
                "draft_time_ms": 0.0,
                "verify_time_ms": 0.0,
            }
            self._acceptance_history.clear()
            self._adaptive_k = self.config.num_speculative_tokens


# =============================================================================
# SELF-SPECULATIVE DECODING (No separate draft model)
# =============================================================================

class SelfSpeculativeDecoder:
    """
    Self-speculative decoding using early exit.
    
    Uses early layers of the target model as the draft model,
    avoiding the need for a separate smaller model.
    
    Benefits:
    - No additional model needed
    - Better alignment between draft and target
    - Easier deployment
    """
    
    def __init__(
        self,
        model: nn.Module,
        draft_exit_layer: int = 4,  # Exit early at this layer for draft
        config: Optional[SpeculativeConfig] = None,
        device: str = "cuda:0",
    ):
        """
        Initialize self-speculative decoder.
        
        Args:
            model: Target model with accessible layers
            draft_exit_layer: Layer to use for draft generation
            config: Speculation configuration
            device: Device for inference
        """
        self.model = model.to(device)
        self.draft_exit_layer = draft_exit_layer
        self.config = config or SpeculativeConfig()
        self.device = device
        
        self.model.eval()
        
        # Stats
        self.stats = {
            "total_tokens": 0,
            "speculation_steps": 0,
        }
    
    def _early_exit_forward(
        self,
        input_ids: torch.Tensor,
        exit_layer: int,
    ) -> torch.Tensor:
        """
        Forward pass with early exit for draft generation.
        
        Args:
            input_ids: Input token IDs
            exit_layer: Layer to exit at
            
        Returns:
            Logits from early exit
        """
        # This requires model modification to support early exit
        # For now, return full model output (fallback)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=False)
            return outputs.logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Generate with self-speculation.
        
        Falls back to normal generation for now.
        Full implementation requires model-specific early exit support.
        """
        current_ids = input_ids.clone().to(self.device)
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids=current_ids, use_cache=False)
                logits = outputs.logits[:, -1, :]
                
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                yield SpeculativeOutput(
                    token_ids=next_token,
                    num_accepted=1,
                    acceptance_rate=1.0,
                )


# =============================================================================
# MEDUSA-STYLE SPECULATIVE DECODING
# =============================================================================

class MedusaHead(nn.Module):
    """
    Medusa head for parallel token prediction.
    
    Predicts multiple future tokens in parallel from a single
    hidden state, enabling efficient tree-based speculation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,  # Number of speculative heads
    ):
        super().__init__()
        self.num_heads = num_heads
        
        # Each head predicts a different future token
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_heads)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Predict multiple future tokens.
        
        Args:
            hidden_states: [batch, seq, hidden]
            
        Returns:
            List of logits for each head [batch, seq, vocab]
        """
        return [head(hidden_states) for head in self.heads]


class MedusaDecoder:
    """
    Medusa-style speculative decoding.
    
    Uses additional prediction heads to generate multiple
    candidate tokens in parallel, verified with tree attention.
    
    Benefits over standard speculation:
    - No separate draft model needed
    - Parallel candidate generation
    - Tree-structured verification
    """
    
    def __init__(
        self,
        model: nn.Module,
        medusa_heads: Optional[MedusaHead] = None,
        num_candidates: int = 4,
        device: str = "cuda:0",
    ):
        """
        Initialize Medusa decoder.
        
        Args:
            model: Base model
            medusa_heads: Pre-trained Medusa heads (optional)
            num_candidates: Number of candidate tokens per position
            device: Device for inference
        """
        self.model = model.to(device)
        self.device = device
        self.num_candidates = num_candidates
        
        # Initialize or use provided Medusa heads
        if medusa_heads is not None:
            self.medusa_heads = medusa_heads.to(device)
        else:
            # Create untrained heads (would need fine-tuning)
            hidden_size = getattr(model.config, 'hidden_size', 4096)
            vocab_size = getattr(model.config, 'vocab_size', 32000)
            self.medusa_heads = MedusaHead(
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_heads=num_candidates,
            ).to(device)
        
        self.model.eval()
        self.medusa_heads.eval()
        
        self.stats = {
            "total_tokens": 0,
            "steps": 0,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 100,
        temperature: float = 1.0,
        **kwargs,
    ):
        """
        Generate with Medusa-style speculation.
        
        Note: Full tree verification requires attention mask modification.
        This is a simplified implementation.
        """
        current_ids = input_ids.clone().to(self.device)
        
        for step in range(max_tokens):
            with torch.no_grad():
                # Get hidden states and main logits
                outputs = self.model(
                    input_ids=current_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )
                
                hidden = outputs.hidden_states[-1][:, -1:, :]  # Last layer, last position
                main_logits = outputs.logits[:, -1, :]
                
                # Get Medusa head predictions
                medusa_logits = self.medusa_heads(hidden)
                
                # For now, just use main prediction (full impl needs tree verification)
                if temperature > 0:
                    probs = F.softmax(main_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = main_logits.argmax(dim=-1, keepdim=True)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                yield SpeculativeOutput(
                    token_ids=next_token,
                    num_accepted=1,
                    acceptance_rate=1.0,
                )
            
            self.stats["total_tokens"] += 1
            self.stats["steps"] += 1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_speculation_speedup(
    target_time_per_token: float,
    draft_time_per_token: float,
    acceptance_rate: float,
    num_speculative_tokens: int,
) -> float:
    """
    Estimate speedup from speculative decoding.
    
    Args:
        target_time_per_token: Time for target model forward (ms)
        draft_time_per_token: Time for draft model forward (ms)
        acceptance_rate: Average acceptance rate (0-1)
        num_speculative_tokens: Number of tokens to speculate
        
    Returns:
        Expected speedup factor
    """
    # Time for standard decoding
    standard_time = target_time_per_token
    
    # Time for speculative decoding
    draft_time = draft_time_per_token * num_speculative_tokens
    verify_time = target_time_per_token  # Single forward for all tokens
    spec_time = draft_time + verify_time
    
    # Expected tokens per speculation step
    expected_tokens = 1 + acceptance_rate * num_speculative_tokens
    
    # Time per token with speculation
    spec_time_per_token = spec_time / expected_tokens
    
    return standard_time / spec_time_per_token


def is_compatible_models(
    target_model: nn.Module,
    draft_model: nn.Module,
) -> bool:
    """
    Check if models are compatible for speculative decoding.
    
    Requirements:
    - Same vocabulary size
    - Both are autoregressive LMs
    """
    target_vocab = getattr(target_model.config, 'vocab_size', None)
    draft_vocab = getattr(draft_model.config, 'vocab_size', None)
    
    if target_vocab is None or draft_vocab is None:
        return False
    
    return target_vocab == draft_vocab
