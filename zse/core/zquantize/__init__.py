"""
zQuantize - Quantization Engine

Implements ultra memory-efficient quantization:
- GPTQ: Post-training quantization (INT4/INT8)
- HQQ: Half-Quadratic Quantization (no calibration needed)
- FP8: 8-bit floating point (H100+)
- Dynamic INT8: Runtime quantization
- Mixed precision: Per-tensor INT2-8 configuration

Key Innovation:
- Per-layer sensitivity analysis
- Per-tensor precision assignment
- INT2/INT3 for compressible layers
- INT8 for critical layers (embeddings, lm_head)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zse.core.zquantize.quantizer import Quantizer
    from zse.core.zquantize.config import QuantizationConfig

__all__ = [
    "Quantizer",
    "QuantizationConfig",
]
