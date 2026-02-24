"""
ZSE Quantization Module

Ultra memory-efficient INT8/INT4 quantization for LLM inference.

Memory savings:
- FP16: 2 bytes/param (baseline)
- INT8: 1 byte/param (50% reduction)
- INT4: 0.5 bytes/param (75% reduction)

Key features:
- Per-channel quantization for accuracy
- Triton kernels for fast dequant
- Fused dequant-matmul for zero overhead
- Compatible with streaming loader

Author: ZSE Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# Import fused kernels for faster inference
FUSED_KERNELS_AVAILABLE = False
try:
    from .triton_quant_kernels import int8_fused_matmul, int4_fused_matmul
    FUSED_KERNELS_AVAILABLE = True
except ImportError:
    pass


class QuantType(Enum):
    """Quantization precision types."""
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # NormalFloat4 (QLoRA style)


@dataclass
class QuantConfig:
    """Quantization configuration."""
    quant_type: QuantType = QuantType.INT8
    group_size: int = 128  # For group-wise quantization
    symmetric: bool = True  # Symmetric vs asymmetric
    per_channel: bool = True  # Per-channel vs per-tensor
    compute_dtype: torch.dtype = torch.float16  # Compute precision


def estimate_quantized_memory(num_params: int, quant_type: QuantType) -> float:
    """
    Estimate memory for quantized model in GB.
    
    Args:
        num_params: Number of parameters
        quant_type: Quantization type
        
    Returns:
        Estimated memory in GB
    """
    bytes_per_param = {
        QuantType.FP16: 2.0,
        QuantType.INT8: 1.0 + 0.03,  # weights + scales
        QuantType.INT4: 0.5 + 0.03,  # weights + scales
        QuantType.NF4: 0.5 + 0.03,
    }
    
    bpp = bytes_per_param.get(quant_type, 2.0)
    return (num_params * bpp) / (1024**3)


# =============================================================================
# INT8 QUANTIZATION
# =============================================================================

def quantize_tensor_int8(
    tensor: torch.Tensor,
    per_channel: bool = True,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize FP16/FP32 tensor to INT8.
    
    Args:
        tensor: Input tensor [out_features, in_features] or [features]
        per_channel: Quantize per output channel
        symmetric: Use symmetric quantization (no zero point)
        
    Returns:
        (quantized_tensor, scales, zero_points)
    """
    if per_channel and tensor.ndim >= 2:
        # Per-channel: compute scale per output channel (row)
        axis = 1 if tensor.ndim == 2 else tuple(range(1, tensor.ndim))
        tensor_max = tensor.abs().amax(dim=axis, keepdim=True)
    else:
        # Per-tensor: single scale
        tensor_max = tensor.abs().max()
    
    # Prevent division by zero
    tensor_max = torch.clamp(tensor_max, min=1e-8)
    
    if symmetric:
        # Symmetric: range [-127, 127], zero point = 0
        scale = tensor_max / 127.0
        quantized = torch.round(tensor / scale).to(torch.int8)
        zero_point = None
    else:
        # Asymmetric: range [0, 255], with zero point
        tensor_min = tensor.min() if not per_channel else tensor.amin(dim=axis, keepdim=True)
        scale = (tensor_max - tensor_min) / 255.0
        zero_point = torch.round(-tensor_min / scale).to(torch.int8)
        quantized = torch.round(tensor / scale + zero_point).to(torch.int8)
    
    return quantized, scale.squeeze(), zero_point


def dequantize_tensor_int8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize INT8 tensor back to FP16/FP32."""
    # Handle per-channel scale broadcasting for 2D tensors
    if quantized.ndim == 2 and scale.ndim == 1:
        scale = scale.unsqueeze(1)  # [out_features] -> [out_features, 1]
    
    if zero_point is not None:
        if quantized.ndim == 2 and zero_point.ndim == 1:
            zero_point = zero_point.unsqueeze(1)
        return ((quantized.float() - zero_point.float()) * scale).to(dtype)
    else:
        return (quantized.float() * scale).to(dtype)


# =============================================================================
# INT4 QUANTIZATION (Packed)
# =============================================================================

def quantize_tensor_int4(
    tensor: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize FP16/FP32 tensor to INT4 (packed into INT8).
    
    Two INT4 values packed into one INT8:
    - Lower 4 bits: first value
    - Upper 4 bits: second value
    
    Args:
        tensor: Input tensor [out_features, in_features]
        group_size: Number of values per quantization group
        
    Returns:
        (packed_tensor, scales) where packed is half the original size
    """
    out_features, in_features = tensor.shape
    
    # Ensure in_features is divisible by group_size
    assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    
    # Reshape for group-wise quantization
    num_groups = in_features // group_size
    tensor_grouped = tensor.view(out_features, num_groups, group_size)
    
    # Compute per-group scales
    group_max = tensor_grouped.abs().amax(dim=2, keepdim=True)
    group_max = torch.clamp(group_max, min=1e-8)
    scales = group_max / 7.0  # INT4 symmetric range: [-7, 7]
    
    # Quantize to [-7, 7]
    quantized = torch.round(tensor_grouped / scales).clamp(-7, 7).to(torch.int8)
    quantized = quantized.view(out_features, in_features)
    
    # Pack two INT4 values into one INT8
    # Shift values to [0, 15] range for packing
    quantized_shifted = quantized + 8  # Now [1, 15]
    
    # Pack pairs
    assert in_features % 2 == 0, "in_features must be even for INT4 packing"
    packed = (quantized_shifted[:, 0::2] & 0x0F) | ((quantized_shifted[:, 1::2] & 0x0F) << 4)
    packed = packed.to(torch.uint8)
    
    scales = scales.squeeze(-1)  # [out_features, num_groups]
    
    return packed, scales


def dequantize_tensor_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Dequantize INT4 (packed) tensor back to FP16.
    
    Args:
        packed: Packed INT4 tensor [out_features, in_features//2]
        scales: Quantization scales [out_features, num_groups]
        group_size: Group size used during quantization
        dtype: Output dtype
        
    Returns:
        Dequantized tensor [out_features, in_features]
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2
    
    # Unpack INT4 values
    low = (packed & 0x0F).to(torch.int8) - 8  # Back to [-7, 7]
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    
    # Interleave to original order
    unpacked = torch.zeros(out_features, in_features, dtype=torch.int8, device=packed.device)
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high
    
    # Reshape for group-wise dequantization
    num_groups = in_features // group_size
    unpacked_grouped = unpacked.view(out_features, num_groups, group_size)
    scales_expanded = scales.unsqueeze(-1)  # [out_features, num_groups, 1]
    
    # Dequantize
    dequantized = (unpacked_grouped.float() * scales_expanded).view(out_features, in_features)
    
    return dequantized.to(dtype)


# =============================================================================
# TRITON KERNELS FOR FAST DEQUANTIZATION
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _dequant_int8_kernel(
        output_ptr,
        input_ptr,
        scale_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for fast INT8 dequantization."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load quantized values and scales
        x = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.load(scale_ptr + offsets // tl.load(scale_ptr), mask=mask)  # Simplified
        
        # Dequantize
        result = x * scale
        
        tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)
    
    @triton.jit
    def _int8_matmul_dequant_kernel(
        output_ptr,
        input_ptr,
        weight_ptr,
        scale_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused INT8 dequantization + matmul kernel.
        
        Dequantizes INT8 weights on-the-fly during matmul,
        avoiding memory overhead of full dequantization.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Compute output block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        
        # Pointers for A (input) and W (weight)
        a_ptrs = input_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        w_ptrs = weight_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        
        # Load scales for this output block
        scale = tl.load(scale_ptr + offs_n)
        
        # Accumulator
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            # Load input block
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
            
            # Load INT8 weight and dequantize on-the-fly
            w_int8 = tl.load(w_ptrs, mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0)
            w = w_int8.to(tl.float32) * scale[None, :]
            
            # Accumulate
            acc += tl.dot(a, w.to(a.dtype))
            
            # Advance pointers
            a_ptrs += BLOCK_K * stride_ak
            w_ptrs += BLOCK_K * stride_wk
        
        # Store result
        o_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# =============================================================================
# QUANTIZED LINEAR LAYER
# =============================================================================

class QuantizedLinear(nn.Module):
    """
    Memory-efficient quantized linear layer.
    
    Stores weights in INT8 or INT4, dequantizes on-the-fly during forward.
    Memory usage:
    - INT8: ~50% of FP16
    - INT4: ~25% of FP16
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        quant_type: QuantType = QuantType.INT8,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.group_size = group_size
        
        # Register quantized weight and scales as buffers (not parameters)
        if quant_type == QuantType.INT8:
            self.register_buffer("weight_quantized", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("weight_scale", torch.zeros(out_features, dtype=torch.float16))
        elif quant_type == QuantType.INT4:
            packed_size = in_features // 2
            num_groups = in_features // group_size
            self.register_buffer("weight_quantized", torch.zeros(out_features, packed_size, dtype=torch.uint8))
            self.register_buffer("weight_scale", torch.zeros(out_features, num_groups, dtype=torch.float16))
        else:
            raise ValueError(f"Unsupported quant_type: {quant_type}")
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
    
    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        quant_type: QuantType = QuantType.INT8,
        group_size: int = 128,
    ) -> "QuantizedLinear":
        """
        Convert FP16 Linear to quantized linear.
        
        Args:
            linear: Source linear layer
            quant_type: Target quantization type
            group_size: Group size for INT4
            
        Returns:
            Quantized linear layer
        """
        device = linear.weight.device
        
        q_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            quant_type=quant_type,
            group_size=group_size,
        )
        
        # Quantize weights
        weight = linear.weight.data.float()
        
        if quant_type == QuantType.INT8:
            q_weight, scale, _ = quantize_tensor_int8(weight, per_channel=True, symmetric=True)
            q_linear.weight_quantized.copy_(q_weight)
            q_linear.weight_scale.copy_(scale.half())
        elif quant_type == QuantType.INT4:
            q_weight, scale = quantize_tensor_int4(weight, group_size=group_size)
            q_linear.weight_quantized.copy_(q_weight)
            q_linear.weight_scale.copy_(scale.half())
        
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data.half())
        
        # Move to the same device as original layer
        return q_linear.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with on-the-fly dequantization.
        
        Uses fused Triton kernels when available for ~3-4x speedup.
        Falls back to dequantize-then-matmul otherwise.
        """
        # Try to use fused kernels for maximum performance
        if FUSED_KERNELS_AVAILABLE and x.is_cuda:
            try:
                if self.quant_type == QuantType.INT8:
                    return int8_fused_matmul(
                        x.half(),
                        self.weight_quantized,
                        self.weight_scale,
                        self.bias,
                    )
                elif self.quant_type == QuantType.INT4:
                    return int4_fused_matmul(
                        x.half(),
                        self.weight_quantized,
                        self.weight_scale,
                        self.bias,
                        self.group_size,
                    )
            except Exception:
                # Fall back to unfused path on any error
                pass
        
        # Fallback: Dequantize weights then matmul
        if self.quant_type == QuantType.INT8:
            weight = dequantize_tensor_int8(
                self.weight_quantized,
                self.weight_scale,
                dtype=x.dtype,
            )
        elif self.quant_type == QuantType.INT4:
            weight = dequantize_tensor_int4(
                self.weight_quantized,
                self.weight_scale,
                group_size=self.group_size,
                dtype=x.dtype,
            )
        else:
            raise ValueError(f"Unsupported quant_type: {self.quant_type}")
        
        # Ensure weight is on the same device as input
        weight = weight.to(x.device)
        
        # Convert bias to match input dtype and device
        bias = self.bias.to(x.dtype).to(x.device) if self.bias is not None else None
        
        # Standard linear operation
        output = F.linear(x, weight, bias)
        
        return output
    
    def memory_bytes(self) -> int:
        """Calculate actual memory usage of this layer."""
        weight_bytes = self.weight_quantized.numel() * self.weight_quantized.element_size()
        scale_bytes = self.weight_scale.numel() * self.weight_scale.element_size()
        bias_bytes = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        return weight_bytes + scale_bytes + bias_bytes


# =============================================================================
# MODEL QUANTIZATION UTILITIES
# =============================================================================

def quantize_model(
    model: nn.Module,
    quant_type: QuantType = QuantType.INT8,
    group_size: int = 128,
    skip_layers: Optional[list] = None,
) -> nn.Module:
    """
    Quantize all Linear layers in a model.
    
    Args:
        model: PyTorch model to quantize
        quant_type: Target quantization type
        group_size: Group size for INT4
        skip_layers: Layer names to skip (e.g., ["lm_head"])
        
    Returns:
        Quantized model (in-place modification)
    """
    skip_layers = skip_layers or []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if should skip
            if any(skip in name for skip in skip_layers):
                continue
            
            # Check if already quantized
            if isinstance(module, QuantizedLinear):
                continue
            
            # Quantize
            q_module = QuantizedLinear.from_float(module, quant_type, group_size)
            
            # Replace in parent
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            
            setattr(parent, parts[-1], q_module)
    
    return model


def get_model_memory(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model memory usage.
    
    Returns:
        Dict with memory stats in GB
    """
    total_params = 0
    total_bytes = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()
    
    for name, buffer in model.named_buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    
    return {
        "params": total_params,
        "bytes": total_bytes,
        "gb": total_bytes / (1024**3),
        "mb": total_bytes / (1024**2),
    }


def compare_quantization_memory(num_params: int) -> Dict[str, float]:
    """
    Compare memory usage across quantization types.
    
    Args:
        num_params: Number of model parameters
        
    Returns:
        Dict mapping quant_type to memory in GB
    """
    return {
        "FP32": num_params * 4 / (1024**3),
        "FP16": num_params * 2 / (1024**3),
        "INT8": num_params * 1.03 / (1024**3),  # +3% for scales
        "INT4": num_params * 0.53 / (1024**3),  # +6% for scales
    }
