"""Gemma 4 (unified, encoder-free multimodal) architecture adapter.

Model: Gemma4UnifiedForConditionalGeneration  (google/gemma-4-12B-it)

Gemma 4 is a UNIFIED multimodal model — no separate vision/audio encoders.
Image patches and raw-audio frames are projected directly into the text
backbone's embedding space and spliced into the token stream at special token
positions (image_token_id, audio_token_id).

Text backbone (gemma4_unified_text) — key facts from the real config.json:
  - 48 layers, hidden_size 3840, intermediate 15360, vocab 262144
  - num_attention_heads 16, num_key_value_heads 8 (sliding layers)
  - head_dim 256 (sliding) but global_head_dim 512 on full-attention layers
  - num_global_key_value_heads 1 on full-attention layers
  - layer_types: repeating 5 "sliding_attention" : 1 "full_attention" (×8)
  - dual RoPE: sliding theta=10000 (default); full theta=1e6, "proportional",
    partial_rotary_factor 0.25
  - attention_k_eq_v: true (K and V share a projection)
  - hidden_activation gelu_pytorch_tanh (GeGLU)
  - final_logit_softcapping 30.0
  - sliding_window 1024, tie_word_embeddings true, rms_norm_eps 1e-6

Confirmed from the real safetensors header (677 tensors, Stage-0 inventory):
  - Per-layer QK-norm: self_attn.q_norm.weight / k_norm.weight ([head_dim]).
  - Per-layer learnable scalar: layers.N.layer_scalar ([1]).
  - q_proj [4096,3840], k/v_proj [2048,3840], o_proj [3840,4096],
    gate/up [15360,3840], down [3840,15360], 4 norms/layer.

Multimodal surface is just 11 tensors (truly encoder-free):
  Vision (9): vision_embedder.patch_dense (w/b) [3840,6912], patch_ln1/ln2,
    pos_embedding [1120,2,3840], pos_norm; embed_vision.embedding_projection
    [3840,3840].
  Audio (1): embed_audio.embedding_projection [3840,640].
  (image_token_id 258880, audio_token_id 258881 mark splice positions.)
"""

from typing import List

from zse_engine.format.config import ModelConfig
from zse_engine.format.arch.base import ArchAdapter, register_adapter


@register_adapter
class Gemma4Adapter(ArchAdapter):
    """Adapter for Gemma 4 unified multimodal models (12B and friends)."""

    ARCH_NAME = "gemma4"

    def config_from_hf(self, hf_config: dict) -> ModelConfig:
        # Unified config nests the language model under "text_config".
        tc = hf_config.get("text_config", hf_config)

        hidden_size = tc["hidden_size"]
        num_heads = tc["num_attention_heads"]
        num_kv_heads = tc.get("num_key_value_heads", num_heads)
        head_dim = tc.get("head_dim", hidden_size // num_heads)

        # Dual RoPE: sliding (default) vs full ("proportional", partial rotary).
        rope_params = tc.get("rope_parameters", {})
        sliding_rope = rope_params.get("sliding_attention", {})
        full_rope = rope_params.get("full_attention", {})
        rope_theta = sliding_rope.get("rope_theta", tc.get("rope_theta", 10000.0))
        global_rope_theta = full_rope.get("rope_theta")
        partial_rotary = full_rope.get("partial_rotary_factor")

        # Multimodal sub-configs (encoder-free projections).
        mm = None
        vc = hf_config.get("vision_config")
        ac = hf_config.get("audio_config")
        if vc or ac:
            mm = {}
            if vc:
                mm["vision"] = {
                    "patch_size": vc.get("patch_size"),
                    "model_patch_size": vc.get("model_patch_size"),
                    "mm_embed_dim": vc.get("mm_embed_dim"),
                    "mm_posemb_size": vc.get("mm_posemb_size"),
                    "num_soft_tokens": vc.get("num_soft_tokens"),
                    "output_proj_dims": vc.get("output_proj_dims"),
                    "pooling_kernel_size": vc.get("pooling_kernel_size"),
                    "rms_norm_eps": vc.get("rms_norm_eps"),
                    "image_token_id": hf_config.get("image_token_id"),
                }
            if ac:
                mm["audio"] = {
                    "audio_embed_dim": ac.get("audio_embed_dim"),
                    "audio_samples_per_token": ac.get("audio_samples_per_token"),
                    "output_proj_dims": ac.get("output_proj_dims"),
                    "rms_norm_eps": ac.get("rms_norm_eps"),
                    "audio_token_id": hf_config.get("audio_token_id"),
                }

        config = ModelConfig(
            arch="gemma4",
            num_layers=tc["num_hidden_layers"],
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"],
            max_seq_len=tc.get("max_position_embeddings", 262144),
            rope_theta=rope_theta,
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            sliding_window=tc.get("sliding_window"),
            tie_word_embeddings=tc.get("tie_word_embeddings",
                                       hf_config.get("tie_word_embeddings", True)),
            # Gemma-family extensions
            layer_types=tc.get("layer_types"),
            global_head_dim=tc.get("global_head_dim"),
            global_num_kv_heads=tc.get("num_global_key_value_heads"),
            global_rope_theta=global_rope_theta,
            partial_rotary_factor=partial_rotary,
            hidden_activation=tc.get("hidden_activation"),
            final_logit_softcapping=tc.get("final_logit_softcapping"),
            attn_logit_softcapping=tc.get("attn_logit_softcapping"),
            query_pre_attn_scalar=tc.get("query_pre_attn_scalar"),
            attention_k_eq_v=tc.get("attention_k_eq_v", False),
            embed_scale=True,  # Gemma scales embeddings by sqrt(hidden_size)
            multimodal=mm,
        )
        return config

    def map_tensor_name(self, hf_name: str) -> str:
        """Map HF Gemma 4 unified tensor names to canonical ZSE names.

        HF unified models nest the text decoder under a language-model prefix
        and the modality projections under their own prefixes. We canonicalize
        the text backbone to the same flat layout the engine already uses
        (embed_tokens.weight, layers.N..., norm.weight) and keep modality
        tensors under explicit vision./audio. prefixes.

        Confirmed against the real safetensors index by the inventory script
        before Stage-1 inference work.
        """
        name = hf_name

        # Strip common outer wrappers.
        for prefix in ("model.",):
            if name.startswith(prefix):
                name = name[len(prefix):]

        # Text backbone: language_model.* -> flat.
        if name.startswith("language_model."):
            name = name[len("language_model."):]
            # transformers sometimes double-nests: language_model.model.layers...
            if name.startswith("model."):
                name = name[len("model."):]
            return name

        # --- Multimodal projections (encoder-free) — real Gemma 4 prefixes ---
        # Vision patch embedder: single dense matmul + layernorms + pos embedding.
        if name.startswith("vision_embedder."):
            return "vision." + name[len("vision_embedder."):]
        # Vision/audio projection into the token embedding space.
        if name.startswith("embed_vision."):
            return "vision.proj." + name[len("embed_vision."):]
        if name.startswith("embed_audio."):
            return "audio.proj." + name[len("embed_audio."):]
        # Defensive fallbacks for alternate naming.
        if name.startswith("audio_embedder."):
            return "audio." + name[len("audio_embedder."):]

        return name

    def should_quantize(self, tensor_name: str) -> bool:
        """Quantize the big text-backbone matmuls; keep modality projections,
        norms, and embeddings in fp16 (small, precision-sensitive)."""
        # Never quantize modality projection / embedding / norm tensors.
        if tensor_name.startswith("vision.") or tensor_name.startswith("audio."):
            return False
        return super().should_quantize(tensor_name)

    def expected_tensors(self, config: ModelConfig) -> List[str]:
        """Text-backbone tensor inventory. Modality tensors are validated
        separately (their exact names come from the inventory script).

        Layer-type aware: full-attention layers use attention_k_eq_v, so their
        value projection is shared with the key (value = key) and NO v_proj
        weight is stored. Verified against the real 677-tensor header:
        40 sliding × 14 + 8 full × 13 (no v_proj) + embed + norm + 11 mm = 677.
        """
        layer_types = config.layer_types or []

        def is_full(i: int) -> bool:
            if config.attention_k_eq_v and i < len(layer_types):
                return layer_types[i] == "full_attention"
            return False

        tensors = ["embed_tokens.weight"]
        for i in range(config.num_layers):
            prefix = f"layers.{i}"
            tensors.extend([
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.pre_feedforward_layernorm.weight",
                f"{prefix}.post_feedforward_layernorm.weight",
                f"{prefix}.layer_scalar",
            ])
            # v_proj only on layers that DON'T share value with key.
            if not is_full(i):
                tensors.append(f"{prefix}.self_attn.v_proj.weight")
        tensors.append("norm.weight")
        if not config.tie_word_embeddings:
            tensors.append("lm_head.weight")
        return tensors

    def validate_config(self, config: ModelConfig) -> List[str]:
        warnings = super().validate_config(config)
        if config.global_head_dim and config.global_head_dim != config.head_dim:
            warnings.append(
                f"Gemma 4 mixed head_dim: sliding={config.head_dim} "
                f"global={config.global_head_dim} — engine must handle per-layer "
                f"head_dim before inference is correct (Stage 1)."
            )
        if config.partial_rotary_factor:
            warnings.append(
                f"Partial rotary factor {config.partial_rotary_factor} on full-"
                f"attention layers — RoPE kernel must rotate only the first "
                f"{config.partial_rotary_factor:.0%} of head dims (Stage 1)."
            )
        return warnings
