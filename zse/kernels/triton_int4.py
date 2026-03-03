"""
ZSE INT4 Matrix Multiplication - Triton Implementation

Uses Triton for reliable, cross-platform CUDA kernel generation.
Fused INT4 dequantization + matrix multiplication.
"""

import torch
from typing import Optional

# Check if Triton is available
_TRITON_AVAILABLE = False
_TRITON_ERROR = None

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError as e:
    _TRITON_ERROR = f"Triton import failed: {e}"
except Exception as e:
    _TRITON_ERROR = f"Triton error: {e}"


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return _TRITON_AVAILABLE


def get_triton_error() -> Optional[str]:
    """Get error message if Triton isn't available."""
    return _TRITON_ERROR


# Only define Triton kernel if available
if _TRITON_AVAILABLE:
    @triton.jit
    def int4_matmul_kernel(
        # Pointers
        A_ptr, B_ptr, scales_ptr, C_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_scales_n, stride_scales_g,
        # Quantization params
        group_size,
        # Meta-parameters
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """
        INT4 fused dequantize + matmul kernel.
        
        A: [M, K] float16 input
        B: [N, K//2] uint8 packed INT4 weights (transposed storage)
        scales: [N, num_groups] float16 scales
        C: [M, N] float16 output
        """
        # Block indices
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Offsets for this block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Iterate over K dimension in blocks
        for k_start in range(0, K, BLOCK_K):
            # Load A tile [BLOCK_M, BLOCK_K]
            offs_k = k_start + tl.arange(0, BLOCK_K)
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
            
            # Load B tile (packed INT4) and dequantize
            # B is [N, K//2] uint8, we need [BLOCK_N, BLOCK_K]
            offs_k_packed = (k_start + tl.arange(0, BLOCK_K)) // 2
            b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k_packed[None, :] * stride_bk
            b_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K // 2)
            b_packed = tl.load(b_ptrs, mask=b_mask, other=0)
            
            # Unpack INT4: even indices get low nibble, odd get high nibble
            k_is_odd = (tl.arange(0, BLOCK_K) % 2) == 1
            b_low = (b_packed & 0x0F).to(tl.int8) - 8  # shift to signed
            b_high = (b_packed >> 4).to(tl.int8) - 8
            b_unpacked = tl.where(k_is_odd[None, :], b_high, b_low).to(tl.float32)
            
            # Load scales for this K block
            group_idx = k_start // group_size
            scale_ptrs = scales_ptr + offs_n * stride_scales_n + group_idx * stride_scales_g
            scale_mask = offs_n < N
            scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0).to(tl.float32)
            
            # Dequantize: weight = int_val * scale
            b_dequant = b_unpacked * scales[:, None]  # [BLOCK_N, BLOCK_K]
            
            # Matrix multiply: A @ B^T
            # A: [BLOCK_M, BLOCK_K], B: [BLOCK_N, BLOCK_K]
            # Result: [BLOCK_M, BLOCK_N]
            acc += tl.dot(a, tl.trans(b_dequant))
        
        # Store result
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def int4_matmul_triton(
    x: torch.Tensor,           # [*, K] input
    weight_packed: torch.Tensor,  # [N, K//2] packed INT4
    scales: torch.Tensor,      # [N, num_groups]
    group_size: int = 128
) -> torch.Tensor:
    """
    INT4 matrix multiplication using Triton.
    
    Args:
        x: Input tensor [..., K] in float16
        weight_packed: Packed INT4 weights [N, K//2] as uint8
        scales: Per-group scales [N, num_groups] in float16
        group_size: Number of elements per quantization group
    
    Returns:
        Output tensor [..., N] in float16
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError(f"Triton not available: {_TRITON_ERROR}")
    
    # Flatten input to 2D
    orig_shape = x.shape
    x_flat = x.view(-1, x.shape[-1]).contiguous()
    M, K = x_flat.shape
    N = weight_packed.shape[0]
    
    # Ensure float16
    if x_flat.dtype != torch.float16:
        x_flat = x_flat.half()
    
    # Allocate output
    output = torch.empty((M, N), dtype=torch.float16, device=x.device)
    
    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # Must be even for INT4 unpacking
    
    # Grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    int4_matmul_kernel[grid](
        x_flat, weight_packed, scales, output,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        weight_packed.stride(0), weight_packed.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        group_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    # Reshape to match input batch dims
    output_shape = orig_shape[:-1] + (N,)
    return output.view(output_shape)


class TritonInt4Linear(torch.nn.Module):
    """
    INT4 Linear layer using Triton kernels.
    Drop-in replacement for QuantizedLinearZSE.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_packed: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = int4_matmul_triton(x, self.weight_packed, self.scales, self.group_size)
        if self.bias is not None:
            output = output + self.bias
        return output
