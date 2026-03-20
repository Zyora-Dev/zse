"""
Llama Model Architecture

ZSE implementation of Llama models (1, 2, 3, 3.1, 3.2, 3.3).

Features:
- Full zAttention kernel integration
- GQA (Grouped Query Attention) support
- RoPE with scaling for long contexts
- Efficient KV cache management
- INT4/INT8 quantization compatible
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    ModelConfig,
    BaseModel,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

# Try to import zAttention
try:
    from zse.core.zattention import zAttention
    ZATTENTION_AVAILABLE = True
except ImportError:
    ZATTENTION_AVAILABLE = False


class LlamaConfig(ModelConfig):
    """Llama-specific configuration."""
    
    model_type: str = "llama"
    
    # Llama 3 defaults
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA with 8 KV heads
    vocab_size: int = 128256
    max_position_embeddings: int = 8192
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    
    @classmethod
    def llama_7b(cls) -> "LlamaConfig":
        """Llama 2 7B configuration."""
        return cls(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            vocab_size=32000,
            max_position_embeddings=4096,
            rope_theta=10000.0,
        )
    
    @classmethod
    def llama_13b(cls) -> "LlamaConfig":
        """Llama 2 13B configuration."""
        return cls(
            hidden_size=5120,
            intermediate_size=13824,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=40,
            vocab_size=32000,
            max_position_embeddings=4096,
        )
    
    @classmethod
    def llama_70b(cls) -> "LlamaConfig":
        """Llama 2 70B configuration."""
        return cls(
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            vocab_size=32000,
            max_position_embeddings=4096,
        )
    
    @classmethod
    def llama3_8b(cls) -> "LlamaConfig":
        """Llama 3 8B configuration."""
        return cls(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=8192,
            rope_theta=500000.0,
        )
    
    @classmethod
    def llama3_70b(cls) -> "LlamaConfig":
        """Llama 3 70B configuration."""
        return cls(
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=8192,
            rope_theta=500000.0,
        )


class LlamaAttention(nn.Module):
    """
    Llama attention with zAttention kernel integration.
    
    Supports:
    - Multi-head attention (MHA)
    - Grouped Query Attention (GQA)
    - Multi-Query Attention (MQA)
    - Flash Attention via zAttention kernels
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # Projection layers
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        
        # zAttention
        self._use_zattention = ZATTENTION_AVAILABLE
        if self._use_zattention:
            from zse.core.zattention import AttentionConfig
            attn_config = AttentionConfig(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
            )
            self.zattention = zAttention(attn_config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV cache.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len] or None for causal
            position_ids: [batch, seq_len]
            past_key_value: (key, value) cache from previous forward
            use_cache: Whether to return updated cache
        
        Returns:
            (output, (key, value)) or (output, None)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch, heads, seq, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Attention
        if self._use_zattention and not self.training:
            # Use zAttention kernels for inference
            # zAttention handles GQA internally, so pass unrepeated KV
            attn_output = self.zattention(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                is_prefill=True,
            )
        else:
            # Standard PyTorch attention - need to repeat KV for GQA
            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = self._pytorch_attention(
                query_states,
                key_states_expanded,
                value_states_expanded,
                attention_mask,
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value
    
    def _pytorch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        # Try to use SDPA if available
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=mask,
                is_causal=mask is None,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
            )
        
        # Manual attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn_weights = attn_weights + mask
        else:
            # Causal mask
            seq_len = query.shape[2]
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=query.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)
        
        return torch.matmul(attn_weights, value)


class LlamaMLP(nn.Module):
    """
    Llama MLP with SwiGLU activation.
    
    FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Single Llama transformer decoder layer."""
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attn = LlamaAttention(config, layer_idx)
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
        """
        Forward pass through decoder layer.
        
        Returns:
            (hidden_states, past_key_value)
        """
        residual = hidden_states
        
        # Pre-norm attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class LlamaModel(BaseModel):
    """
    Llama language model for ZSE inference.
    
    Supports:
    - Llama 1, 2, 3 (all sizes)
    - GQA for efficient KV cache
    - RoPE with dynamic scaling
    - zAttention kernel acceleration
    """
    
    def __init__(self, config: Union[LlamaConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = LlamaConfig.from_dict(config)
        
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass through Llama model.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (1 = attend, 0 = mask)
            position_ids: [batch, seq_len]
            past_key_values: List of (key, value) for each layer
            use_cache: Return KV cache for generation
        
        Returns:
            (logits, past_key_values)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare position IDs
        if position_ids is None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_len, past_len + seq_len,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask (convert to 4D if needed)
        if attention_mask is not None and attention_mask.dim() == 2:
            # Create causal mask
            attention_mask = self._prepare_attention_mask(
                attention_mask, hidden_states.dtype
            )
        
        # Initialize past_key_values list if needed
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Forward through layers
        presents = []
        for idx, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            if self.gradient_checkpointing and self.training:
                hidden_states, present_kv = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_cache,
                )
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )
            
            if use_cache:
                presents.append(present_kv)
        
        # Final norm and LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, presents if use_cache else None
    
    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert 2D mask to 4D causal mask."""
        batch_size, seq_len = attention_mask.shape
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=attention_mask.device),
            diagonal=1,
        )
        
        # Combine with attention mask
        # attention_mask: 1 = attend, 0 = mask
        mask = attention_mask[:, None, None, :].to(dtype)
        mask = (1.0 - mask) * float("-inf")
        
        # Add causal mask
        mask = mask + causal_mask[None, None, :, :]
        
        return mask
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare inputs for autoregressive generation."""
        # Only use last token if we have cache
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # Calculate position IDs
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2]
            position_ids = torch.arange(
                past_len, past_len + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
        else:
            position_ids = None
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
    
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Prompt tokens [batch, prompt_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            do_sample: Use sampling vs greedy
            eos_token_id: Stop token
            pad_token_id: Padding token
        
        Returns:
            Generated tokens [batch, prompt_len + generated_len]
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Prepare inputs
            model_inputs = self.prepare_inputs_for_generation(
                generated,
                past_key_values=past_key_values,
            )
            
            # Forward pass
            outputs = self(**model_inputs)
            logits = outputs[0][:, -1, :]  # Last token logits
            past_key_values = outputs[1]
            
            # Sample next token
            if do_sample:
                next_token = self._sample(
                    logits, temperature, top_p, top_k
                )
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Sample from logits with temperature and top-p/top-k."""
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        # Apply top-p (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[LlamaConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        **loader_kwargs,
    ) -> "LlamaModel":
        """
        Load a pretrained Llama model.
        
        Args:
            model_path: Path to model or HuggingFace repo ID
            config: Override config (loads from model if None)
            device: Target device
            dtype: Model dtype
            **loader_kwargs: Additional loader arguments
        
        Returns:
            Loaded LlamaModel
        """
        from ..loader.huggingface_loader import ModelHub
        from ..loader.base import LoadConfig
        
        # Setup loader
        load_config = LoadConfig(device=device, dtype=dtype, **loader_kwargs)
        hub = ModelHub(load_config)
        
        # Get model info
        info = hub.load_info(model_path)
        
        # Create config from model info if not provided
        if config is None:
            import json
            with open(info.config_file) as f:
                config_dict = json.load(f)
            config = LlamaConfig.from_dict(config_dict)
            config.dtype = dtype
            config.device = device
        
        # Create model
        model = cls(config)
        model = model.to(dtype=dtype)
        
        # Load weights
        model = hub.load(model_path, model)
        model = model.to(device)
        
        return model
