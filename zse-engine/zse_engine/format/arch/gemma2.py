"""Gemma-2 architecture adapter — Gemma2ForCausalLM.

Key differences from Llama:
- Uses separate head_dim (256 for 9B/27B, different from hidden_size/num_heads)
- Pre and post feedforward layernorms (4 norms per layer instead of 2)
- Uses GeGLU activation (gegelu) not SiLU
- Logit soft-capping (attn_logit_softcapping, final_logit_softcapping)
- Alternating sliding window attention (even layers global, odd layers local)
- query_pre_attn_scalar for scaling queries
- No QKV bias, no attention output bias
- tie_word_embeddings is commonly true
"""

from typing import List

from zse_engine.format.config import ModelConfig
from zse_engine.format.arch.base import ArchAdapter, register_adapter


@register_adapter
class Gemma2Adapter(ArchAdapter):
    """Adapter for Gemma-2 models (2B, 9B, 27B)."""

    ARCH_NAME = "gemma2"

    def config_from_hf(self, hf_config: dict) -> ModelConfig:
        hidden_size = hf_config["hidden_size"]
        num_heads = hf_config["num_attention_heads"]
        num_kv_heads = hf_config.get("num_key_value_heads", num_heads)
        # Gemma-2 has explicit head_dim that may differ from hidden_size//num_heads
        head_dim = hf_config.get("head_dim", hidden_size // num_heads)

        config = ModelConfig(
            arch="gemma2",
            num_layers=hf_config["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=hf_config["intermediate_size"],
            vocab_size=hf_config["vocab_size"],
            max_seq_len=hf_config.get("max_position_embeddings", 8192),
            rope_theta=hf_config.get("rope_theta", 10000.0),
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            sliding_window=hf_config.get("sliding_window"),
            tie_word_embeddings=hf_config.get("tie_word_embeddings", True),
        )
        return config

    def map_tensor_name(self, hf_name: str) -> str:
        """Map HF Gemma-2 tensor names to canonical ZSE names."""
        name = hf_name
        if name.startswith("model."):
            name = name[6:]
        return name

    def expected_tensors(self, config: ModelConfig) -> List[str]:
        """List all expected tensor names for Gemma-2."""
        tensors = ["embed_tokens.weight"]

        for i in range(config.num_layers):
            prefix = f"layers.{i}"
            tensors.extend([
                # Attention
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                # MLP (GeGLU: gate + up + down)
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                # 4 norms per layer (pre/post attention + pre/post feedforward)
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.pre_feedforward_layernorm.weight",
                f"{prefix}.post_feedforward_layernorm.weight",
            ])

        tensors.append("norm.weight")

        if not config.tie_word_embeddings:
            tensors.append("lm_head.weight")

        return tensors
