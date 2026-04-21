"""Llama architecture adapter — LlamaForCausalLM, CodeLlama, Mistral, etc.

Covers the Llama family which shares the same tensor layout:
- RMSNorm (not LayerNorm)
- SiLU activation in MLP (gate * up, then down)
- RoPE positional encoding
- Optional GQA (num_kv_heads < num_heads)
- Optional sliding window (Mistral)
"""

from typing import Dict, List

from zse_engine.format.config import ModelConfig
from zse_engine.format.arch.base import ArchAdapter, register_adapter


@register_adapter
class LlamaAdapter(ArchAdapter):
    """Adapter for Llama-family models (Llama 2/3, CodeLlama, Mistral, etc.)."""

    ARCH_NAME = "llama"

    def config_from_hf(self, hf_config: dict) -> ModelConfig:
        """Convert HuggingFace LlamaConfig to ZSE ModelConfig."""
        hidden_size = hf_config["hidden_size"]
        num_heads = hf_config["num_attention_heads"]
        num_kv_heads = hf_config.get("num_key_value_heads", num_heads)
        head_dim = hf_config.get("head_dim", hidden_size // num_heads)

        config = ModelConfig(
            arch="llama",
            num_layers=hf_config["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=hf_config["intermediate_size"],
            vocab_size=hf_config["vocab_size"],
            max_seq_len=hf_config.get("max_position_embeddings", 4096),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-5),
            rope_scaling=hf_config.get("rope_scaling"),
            sliding_window=hf_config.get("sliding_window"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
        )
        return config

    def map_tensor_name(self, hf_name: str) -> str:
        """Map HF Llama tensor names to canonical ZSE names.

        HF format:  model.layers.0.self_attn.q_proj.weight
        ZSE format: layers.0.self_attn.q_proj.weight

        HF format:  model.embed_tokens.weight
        ZSE format: embed_tokens.weight

        HF format:  model.norm.weight
        ZSE format: norm.weight

        HF format:  lm_head.weight
        ZSE format: lm_head.weight
        """
        name = hf_name
        if name.startswith("model."):
            name = name[6:]
        return name

    def expected_tensors(self, config: ModelConfig) -> List[str]:
        """List all expected tensor names for a Llama model."""
        tensors = ["embed_tokens.weight"]

        for i in range(config.num_layers):
            prefix = f"layers.{i}"
            tensors.extend([
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ])

        tensors.append("norm.weight")
        if not config.tie_word_embeddings:
            tensors.append("lm_head.weight")

        return tensors


@register_adapter
class MistralAdapter(LlamaAdapter):
    """Mistral — same layout as Llama with sliding window."""

    ARCH_NAME = "mistral"

    def config_from_hf(self, hf_config: dict) -> ModelConfig:
        config = super().config_from_hf(hf_config)
        config.arch = "mistral"
        # Mistral always uses sliding window
        if config.sliding_window is None:
            config.sliding_window = hf_config.get("sliding_window", 4096)
        return config
