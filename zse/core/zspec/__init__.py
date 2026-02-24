"""
zSpec - Speculative Decoding

Accelerates autoregressive generation using speculation:
- 2-3x faster generation with draft model verification
- Self-speculative with early exit
- Medusa-style parallel heads

Usage:
    from zse.core.zspec import SpeculativeDecoder
    
    decoder = SpeculativeDecoder(
        target_model=llama_7b,
        draft_model=llama_1b,
    )
    
    for output in decoder.generate(input_ids, max_tokens=100):
        print(f"Generated {output.num_accepted} tokens")
"""

from .speculative import (
    SpeculativeDecoder,
    SpeculativeConfig,
    SpeculativeOutput,
    SelfSpeculativeDecoder,
    MedusaDecoder,
    MedusaHead,
    estimate_speculation_speedup,
    is_compatible_models,
)

__all__ = [
    # Main decoder
    "SpeculativeDecoder",
    "SpeculativeConfig",
    "SpeculativeOutput",
    # Self-speculative
    "SelfSpeculativeDecoder",
    # Medusa
    "MedusaDecoder",
    "MedusaHead",
    # Utilities
    "estimate_speculation_speedup",
    "is_compatible_models",
]
