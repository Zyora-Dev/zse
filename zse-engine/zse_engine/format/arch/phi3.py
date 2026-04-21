"""Phi-3 architecture adapter — Phi3ForCausalLM.

Key differences from Llama:
- Uses SuRoPE (su/longrope) rope_scaling with short_factor/long_factor
- QKV are fused into a single qkv_proj tensor (must be split)
- Gate and up projections fused into gate_up_proj (must be split)
- No MLP bias, no attention bias
- original_max_position_embeddings field
- Uses RMSNorm like Llama
"""

from typing import List

from zse_engine.format.config import ModelConfig
from zse_engine.format.arch.base import ArchAdapter, register_adapter


@register_adapter
class Phi3Adapter(ArchAdapter):
    """Adapter for Phi-3 / Phi-3.5 models."""

    ARCH_NAME = "phi3"

    def config_from_hf(self, hf_config: dict) -> ModelConfig:
        hidden_size = hf_config["hidden_size"]
        num_heads = hf_config["num_attention_heads"]
        num_kv_heads = hf_config.get("num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads

        config = ModelConfig(
            arch="phi3",
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
            tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
        )
        return config

    def map_tensor_name(self, hf_name: str) -> str:
        """Map HF Phi-3 tensor names to canonical ZSE names.

        Phi-3 uses fused QKV and gate_up projections:
        - model.layers.N.self_attn.qkv_proj.weight → must split to q/k/v
        - model.layers.N.mlp.gate_up_proj.weight → must split to gate/up

        We keep the fused names and split at load time in the engine.
        """
        name = hf_name
        if name.startswith("model."):
            name = name[6:]
        return name

    def should_quantize(self, tensor_name: str) -> bool:
        """Phi-3 specific quantization decisions."""
        from zse_engine.format.spec import should_quantize
        return should_quantize(tensor_name)

    def expected_tensors(self, config: ModelConfig) -> List[str]:
        """List all expected tensor names for Phi-3."""
        tensors = ["embed_tokens.weight"]

        for i in range(config.num_layers):
            prefix = f"layers.{i}"
            tensors.extend([
                # Attention — fused QKV
                f"{prefix}.self_attn.qkv_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                # MLP — fused gate_up
                f"{prefix}.mlp.gate_up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                # Norms
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ])

        tensors.append("norm.weight")

        if not config.tie_word_embeddings:
            tensors.append("lm_head.weight")

        return tensors
