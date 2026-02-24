"""
ZSE Efficiency Module

Memory efficiency modes and optimizations.

Modes:
- speed: Maximum throughput, uses all available memory
- balanced: Good throughput with moderate memory usage (default)
- memory: Low memory footprint, reduced throughput
- ultra: Extreme memory savings for consumer GPUs

Features:
- Target VRAM mode: "Fit model in X GB"
- Activation checkpointing for inference
- Memory planning and estimation
- Auto-configuration based on hardware
- INT8/INT4 quantization for 50-75% memory reduction
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zse.efficiency.modes import EfficiencyMode
    from zse.efficiency.target_memory import TargetMemoryConfig
    from zse.efficiency.memory_planner import MemoryPlanner

# Import quantization
from zse.efficiency.quantization import (
    QuantType,
    QuantConfig,
    QuantizedLinear,
    quantize_model,
    quantize_tensor_int8,
    quantize_tensor_int4,
    dequantize_tensor_int8,
    dequantize_tensor_int4,
    get_model_memory,
    estimate_quantized_memory,
    compare_quantization_memory,
)

# Import fused kernels (optional - requires Triton)
try:
    from zse.efficiency.triton_quant_kernels import (
        int8_fused_matmul,
        int4_fused_matmul,
        FusedQuantizedLinear,
        replace_linear_with_fused,
        benchmark_fused_vs_unfused,
    )
    FUSED_KERNELS_AVAILABLE = True
except ImportError:
    FUSED_KERNELS_AVAILABLE = False

__all__ = [
    "EfficiencyMode",
    "TargetMemoryConfig",
    "MemoryPlanner",
    # Quantization
    "QuantType",
    "QuantConfig",
    "QuantizedLinear",
    "quantize_model",
    "quantize_tensor_int8",
    "quantize_tensor_int4",
    "dequantize_tensor_int8",
    "dequantize_tensor_int4",
    "get_model_memory",
    "estimate_quantized_memory",
    "compare_quantization_memory",
    # Fused kernels (when available)
    "FUSED_KERNELS_AVAILABLE",
    "int8_fused_matmul",
    "int4_fused_matmul",
    "FusedQuantizedLinear",
    "replace_linear_with_fused",
    "benchmark_fused_vs_unfused",
]
