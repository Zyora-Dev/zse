"""
Base Model Architecture

Foundation for all ZSE model implementations.
Provides common infrastructure for:
- Weight initialization
- Forward pass hooks
- KV cache management
- Attention integration
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    """Configuration for model architecture."""
    
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        vocab_size: int = 32000,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        hidden_act: str = "silu",
        tie_word_embeddings: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        **kwargs,  # Allow subclass-specific args
    ):
        # Core dimensions
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        # Vocabulary
        self.vocab_size = vocab_size
        
        # Position encoding
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        
        # Normalization
        self.rms_norm_eps = rms_norm_eps
        
        # Attention
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        
        # MLP
        self.mlp_bias = mlp_bias
        self.hidden_act = hidden_act
        
        # Embedding
        self.tie_word_embeddings = tie_word_embeddings
        
        # Compute settings
        self.dtype = dtype
        self.device = device
        
        # Store extra kwargs for subclasses
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads
    
    @property
    def kv_head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary (e.g., HuggingFace config)."""
        # Map common HF config keys
        mapping = {
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size", 
            "num_hidden_layers": "num_hidden_layers",
            "num_attention_heads": "num_attention_heads",
            "num_key_value_heads": "num_key_value_heads",
            "vocab_size": "vocab_size",
            "max_position_embeddings": "max_position_embeddings",
            "rope_theta": "rope_theta",
            "rope_scaling": "rope_scaling",
            "rms_norm_eps": "rms_norm_eps",
            "attention_dropout": "attention_dropout",
            "attention_bias": "attention_bias",
            "mlp_bias": "mlp_bias",
            "hidden_act": "hidden_act",
            "tie_word_embeddings": "tie_word_embeddings",
        }
        
        kwargs = {}
        for hf_key, our_key in mapping.items():
            if hf_key in config_dict:
                kwargs[our_key] = config_dict[hf_key]
        
        # Handle dtype
        if "torch_dtype" in config_dict:
            dtype_str = config_dict["torch_dtype"]
            if dtype_str == "float16":
                kwargs["dtype"] = torch.float16
            elif dtype_str == "bfloat16":
                kwargs["dtype"] = torch.bfloat16
            elif dtype_str == "float32":
                kwargs["dtype"] = torch.float32
        
        return cls(**kwargs)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stability
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        device: Optional[str] = None,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cache
        self._build_cache(max_position_embeddings, device)
    
    def _build_cache(self, seq_len: int, device: Optional[str] = None):
        """Build sin/cos cache for positions."""
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        positions = positions / self.scaling_factor
        
        # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.outer(positions, self.inv_freq.to(device))
        
        # Expand to full dimension: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get rotary embeddings for positions.
        
        Args:
            x: Input tensor [batch, heads, seq_len, head_dim]
            position_ids: Position indices [batch, seq_len]
        
        Returns:
            (cos, sin) embeddings
        """
        seq_len = x.shape[2]
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len, x.device)
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)
        else:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
        
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, kv_heads, seq_len, head_dim]
        cos: Cosine embeddings [seq, dim] or [batch, seq, dim]
        sin: Sine embeddings [seq, dim] or [batch, seq, dim]
    
    Returns:
        Rotated (query, key) tensors
    """
    def rotate_half(x):
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Handle different cos/sin shapes
    # cos/sin can be [seq, dim] or [batch, seq, dim]
    if cos.dim() == 2:
        # [seq, dim] -> [1, 1, seq, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # [batch, seq, dim] -> [batch, 1, seq, dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads to match query heads (for GQA).
    
    Args:
        hidden_states: [batch, kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each KV head
    
    Returns:
        Expanded tensor [batch, num_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for ZSE model implementations.
    
    Provides:
    - Common initialization
    - Forward pass structure
    - KV cache management hooks
    - zAttention integration points
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._kv_cache = None
        self._use_zattention = True
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs [batch, seq_len]
            past_key_values: Cached KV pairs for each layer
            use_cache: Whether to return updated KV cache
        
        Returns:
            (logits, past_key_values)
        """
        pass
    
    @abstractmethod
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare inputs for autoregressive generation."""
        pass
    
    def enable_zattention(self, enable: bool = True):
        """Enable or disable zAttention kernels."""
        self._use_zattention = enable
    
    def init_kv_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: Optional[str] = None,
    ):
        """Initialize KV cache for generation."""
        device = device or self.config.device
        
        # Initialize cache for each layer
        self._kv_cache = []
        for _ in range(self.config.num_hidden_layers):
            k_cache = torch.zeros(
                batch_size,
                self.config.num_key_value_heads,
                max_seq_len,
                self.config.head_dim,
                dtype=self.config.dtype,
                device=device,
            )
            v_cache = torch.zeros(
                batch_size,
                self.config.num_key_value_heads,
                max_seq_len,
                self.config.head_dim,
                dtype=self.config.dtype,
                device=device,
            )
            self._kv_cache.append((k_cache, v_cache))
    
    def clear_kv_cache(self):
        """Clear KV cache."""
        self._kv_cache = None
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding and hasattr(self, "embed_tokens"):
            n_params -= self.embed_tokens.weight.numel()
            if not self.config.tie_word_embeddings and hasattr(self, "lm_head"):
                n_params -= self.lm_head.weight.numel()
        
        return n_params
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
