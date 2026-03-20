"""
Mistral Model Architecture

ZSE implementation of Mistral models.

Mistral is architecturally similar to Llama with:
- Sliding Window Attention (SWA)
- GQA with 8 KV heads
- Slightly different intermediate size
"""

from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn

from .llama import (
    LlamaConfig,
    LlamaAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaModel,
)
from .base import RMSNorm


class MistralConfig(LlamaConfig):
    """Mistral-specific configuration."""
    
    model_type: str = "mistral"
    
    # Mistral 7B defaults
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 32000
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    
    # Sliding Window Attention
    sliding_window: int = 4096
    
    @classmethod
    def mistral_7b(cls) -> "MistralConfig":
        """Mistral 7B configuration."""
        return cls(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            max_position_embeddings=32768,
            sliding_window=4096,
        )
    
    @classmethod
    def mistral_7b_instruct(cls) -> "MistralConfig":
        """Mistral 7B Instruct configuration."""
        return cls.mistral_7b()
    
    @classmethod
    def mixtral_8x7b(cls) -> "MistralConfig":
        """Mixtral 8x7B configuration (MoE not implemented yet)."""
        return cls(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=32000,
            max_position_embeddings=32768,
            sliding_window=4096,
        )


class MistralAttention(LlamaAttention):
    """
    Mistral attention with sliding window support.
    
    Sliding Window Attention limits attention to a local window,
    reducing memory for very long sequences.
    """
    
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = getattr(config, "sliding_window", None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward with optional sliding window mask."""
        # For long sequences, apply sliding window to KV cache
        if past_key_value is not None and self.sliding_window is not None:
            past_key, past_value = past_key_value
            kv_len = past_key.shape[2]
            
            if kv_len > self.sliding_window:
                # Truncate KV cache to sliding window
                past_key = past_key[:, :, -self.sliding_window:, :]
                past_value = past_value[:, :, -self.sliding_window:, :]
                past_key_value = (past_key, past_value)
        
        return super().forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )


class MistralDecoderLayer(nn.Module):
    """Mistral decoder layer with sliding window attention."""
    
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = MistralAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through decoder layer."""
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class MistralModel(LlamaModel):
    """
    Mistral language model.
    
    Inherits from LlamaModel with:
    - Sliding window attention
    - Different default hyperparameters
    """
    
    def __init__(self, config: Union[MistralConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = MistralConfig.from_dict(config)
        
        # Initialize base (skip LlamaModel.__init__)
        nn.Module.__init__(self)
        self.config = config
        self.vocab_size = config.vocab_size
        self._kv_cache = None
        self._use_zattention = True
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Mistral transformer layers
        self.layers = nn.ModuleList([
            MistralDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        self.gradient_checkpointing = False
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[MistralConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **loader_kwargs,
    ) -> "MistralModel":
        """Load a pretrained Mistral model."""
        from ..loader.huggingface_loader import ModelHub
        from ..loader.base import LoadConfig
        
        load_config = LoadConfig(device=device, dtype=dtype, **loader_kwargs)
        hub = ModelHub(load_config)
        
        info = hub.load_info(model_path)
        
        if config is None:
            import json
            with open(info.config_file) as f:
                config_dict = json.load(f)
            config = MistralConfig.from_dict(config_dict)
            config.dtype = dtype
            config.device = device
        
        model = cls(config)
        model = model.to(dtype=dtype)
        model = hub.load(model_path, model)
        model = model.to(device)
        
        return model
