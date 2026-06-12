"""ZSE Model Config — Architecture metadata for the model."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

from zse_engine.format.spec import QuantMethod, DEFAULT_GROUP_SIZE, DEFAULT_QUANT_BITS
from zse_engine.format import serializer


@dataclass
class QuantConfig:
    """Quantization configuration."""
    method: int = QuantMethod.INT4_ASYM
    bits: int = DEFAULT_QUANT_BITS
    group_size: int = DEFAULT_GROUP_SIZE
    scale_dtype: str = "float16"  # dtype for scales and zeros
    tiled_weights: bool = True    # INT4 weights stored in WMMA-tiled format

    def to_dict(self) -> dict:
        return {"method": self.method, "bits": self.bits,
                "group_size": self.group_size, "scale_dtype": self.scale_dtype,
                "tiled_weights": self.tiled_weights}

    @classmethod
    def from_dict(cls, d: dict) -> 'QuantConfig':
        # Backward compat: old .zse files don't have tiled_weights
        known = {"method", "bits", "group_size", "scale_dtype", "tiled_weights"}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class ModelConfig:
    """Full model architecture configuration."""
    arch: str = "llama"                 # "llama", "mistral", "qwen"
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32              # For GQA (grouped query attention)
    head_dim: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 11008
    vocab_size: int = 32000
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    rope_scaling: Optional[Dict] = None  # For extended context
    sliding_window: Optional[int] = None  # Mistral
    tie_word_embeddings: bool = False
    # --- Gemma-family extensions (optional; None/empty = not applicable) ---
    # Per-layer attention type pattern, e.g. ["sliding","sliding",...,"full",...].
    # When set, layers may have different head_dim / kv_heads / rope theta.
    layer_types: Optional[list] = None
    # Full-attention (global) layer overrides — Gemma 4 uses larger head_dim and
    # fewer KV heads on its 1-in-6 "full_attention" layers.
    global_head_dim: Optional[int] = None
    global_num_kv_heads: Optional[int] = None
    global_rope_theta: Optional[float] = None
    # Partial rotary factor for full-attention layers (Gemma 4 = 0.25).
    partial_rotary_factor: Optional[float] = None
    # GeGLU / softcap / embed-scaling knobs (Gemma family).
    hidden_activation: Optional[str] = None     # "gelu_pytorch_tanh" => GeGLU
    final_logit_softcapping: Optional[float] = None
    attn_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: Optional[float] = None
    attention_k_eq_v: bool = False              # K and V projections shared
    embed_scale: bool = False                   # multiply embeddings by sqrt(hidden)
    # Multimodal (Gemma 4 unified, encoder-free). None = text-only.
    multimodal: Optional[Dict] = None           # {"vision": {...}, "audio": {...}}
    quant: QuantConfig = field(default_factory=QuantConfig)

    def to_dict(self) -> dict:
        d = {}
        for k, v in asdict(self).items():
            if k == "quant":
                d[k] = self.quant.to_dict()
            elif v is not None:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelConfig':
        quant_data = d.pop("quant", {})
        config = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if quant_data:
            config.quant = QuantConfig.from_dict(quant_data)
        return config

    def serialize(self) -> bytes:
        return serializer.encode(self.to_dict())

    @classmethod
    def deserialize(cls, data: bytes) -> 'ModelConfig':
        d = serializer.decode(data)
        return cls.from_dict(d)

    @property
    def kv_head_dim(self) -> int:
        return self.head_dim

    @property
    def kv_cache_bytes_per_token(self) -> int:
        """Bytes of KV cache per token per layer (both K and V)."""
        # K: num_kv_heads * head_dim * 2 (float16) + V: same
        return self.num_kv_heads * self.head_dim * 2 * 2

    @property
    def total_kv_cache_bytes_per_token(self) -> int:
        """Total KV cache bytes per token across ALL layers."""
        return self.kv_cache_bytes_per_token * self.num_layers

    def estimate_model_size_bytes(self) -> int:
        """Rough estimate of quantized model weight size."""
        params = 0
        # Attention: Q, K, V, O per layer
        params += self.num_layers * (
            self.hidden_size * self.hidden_size +          # Q
            self.hidden_size * self.num_kv_heads * self.head_dim +  # K
            self.hidden_size * self.num_kv_heads * self.head_dim +  # V
            self.hidden_size * self.hidden_size            # O
        )
        # MLP: gate, up, down per layer
        params += self.num_layers * (
            self.hidden_size * self.intermediate_size * 2 +  # gate + up
            self.intermediate_size * self.hidden_size        # down
        )
        # Embed + LM head (fp16)
        fp16_params = self.vocab_size * self.hidden_size * 2  # embed + lm_head in fp16 bytes
        # Norms (fp16, tiny)
        fp16_params += self.num_layers * self.hidden_size * 2 * 2  # 2 norms per layer

        # Quantized params: 4 bits per weight + scales
        quant_bytes = (params * self.quant.bits) // 8
        scale_overhead = (params // self.quant.group_size) * 4  # 2 bytes scale + 2 bytes zero

        return quant_bytes + scale_overhead + fp16_params
