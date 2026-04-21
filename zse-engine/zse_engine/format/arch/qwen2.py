"""Qwen2/2.5 architecture adapter — Qwen2ForCausalLM.

Key differences from Llama:
- QKV projections have biases (q_proj.bias, k_proj.bias, v_proj.bias)
- O projection has NO bias
- tie_word_embeddings=true (no lm_head.weight)
- head_dim = hidden_size // num_attention_heads (not explicit in config)
- max_window_layers field (layers above this use full attention)
- Same MLP structure as Llama (gate/up/down, SiLU)
"""

from typing import List

from zse_engine.format.config import ModelConfig
from zse_engine.format.arch.base import ArchAdapter, register_adapter


@register_adapter
class Qwen2Adapter(ArchAdapter):
    """Adapter for Qwen2 / Qwen2.5 models."""

    ARCH_NAME = "qwen2"

    def config_from_hf(self, hf_config: dict) -> ModelConfig:
        """Convert HuggingFace Qwen2Config to ZSE ModelConfig."""
        hidden_size = hf_config["hidden_size"]
        num_heads = hf_config["num_attention_heads"]
        num_kv_heads = hf_config.get("num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads

        config = ModelConfig(
            arch="qwen2",
            num_layers=hf_config["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=hf_config["intermediate_size"],
            vocab_size=hf_config["vocab_size"],
            max_seq_len=hf_config.get("max_position_embeddings", 32768),
            rope_theta=hf_config.get("rope_theta", 1000000.0),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_scaling=hf_config.get("rope_scaling"),
            sliding_window=hf_config.get("sliding_window"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
        )
        return config

    def map_tensor_name(self, hf_name: str) -> str:
        """Map HF Qwen2 tensor names to canonical ZSE names.

        Same as Llama: strip 'model.' prefix.
        Qwen2 tensor layout is identical to Llama except for biases.
        """
        name = hf_name
        if name.startswith("model."):
            name = name[6:]
        return name

    def expected_tensors(self, config: ModelConfig) -> List[str]:
        """List all expected tensor names for a Qwen2 model."""
        tensors = ["embed_tokens.weight"]

        for i in range(config.num_layers):
            prefix = f"layers.{i}"
            tensors.extend([
                # Attention — note QKV have biases, O does not
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.o_proj.weight",
                # MLP
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                # Norms
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ])

        tensors.append("norm.weight")

        # Qwen2 typically ties embeddings, but check
        if not config.tie_word_embeddings:
            tensors.append("lm_head.weight")

        return tensors
