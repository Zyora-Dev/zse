"""
ZSE INT4 Matrix Multiplication - Python Bindings

This module provides Python bindings for the custom CUDA INT4 matmul kernel.
Uses PyTorch's JIT compilation to compile CUDA code at runtime.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple

# Global cache for compiled kernels
_COMPILED_KERNEL = None
_COMPILE_ERROR = None


def get_kernel_path() -> Path:
    """Get path to the CUDA kernel source file."""
    return Path(__file__).parent / "int4_matmul.cu"


def _compile_kernels():
    """Compile CUDA kernels using PyTorch's JIT compilation."""
    global _COMPILED_KERNEL, _COMPILE_ERROR
    
    if _COMPILED_KERNEL is not None:
        return _COMPILED_KERNEL
    
    if _COMPILE_ERROR is not None:
        raise _COMPILE_ERROR
    
    try:
        from torch.utils.cpp_extension import load_inline
        
        # Read CUDA source
        kernel_path = get_kernel_path()
        with open(kernel_path, 'r') as f:
            cuda_src = f.read()
        
        # C++ wrapper code
        cpp_src = """
#include <torch/extension.h>
#include <cuda_fp16.h>

// Declare external CUDA kernels
extern "C" void int4_gemm_kernel(
    const __half* A, const uint8_t* B, const __half* scales, __half* C,
    int M, int N, int K, int group_size
);

extern "C" void int4_gemv_kernel(
    const __half* x, const uint8_t* W, const __half* scales, __half* y,
    int N, int K, int group_size
);

// Python-callable wrapper for INT4 GEMM
torch::Tensor int4_gemm(
    torch::Tensor input,       // [batch, seq_len, K] or [M, K]
    torch::Tensor weight,      // [N, K/2] packed INT4
    torch::Tensor scales,      // [N, num_groups]
    int group_size
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(weight.dtype() == torch::kUInt8, "weight must be uint8");
    TORCH_CHECK(scales.dtype() == torch::kHalf, "scales must be float16");
    
    // Flatten input to 2D
    auto input_shape = input.sizes();
    int64_t M = 1;
    for (int i = 0; i < input_shape.size() - 1; i++) {
        M *= input_shape[i];
    }
    int64_t K = input_shape[input_shape.size() - 1];
    int64_t N = weight.size(0);
    
    // Ensure contiguous
    auto A = input.contiguous().view({M, K});
    auto B = weight.contiguous();
    auto S = scales.contiguous();
    
    // Allocate output
    auto output = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kHalf).device(input.device()));
    
    // Calculate grid dimensions
    dim3 block(256);
    dim3 grid((N + 63) / 64, (M + 63) / 64);
    
    // Launch kernel
    int4_gemm_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        B.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(S.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        M, N, K, group_size
    );
    
    // Reshape to match input batch dimensions
    std::vector<int64_t> output_shape;
    for (int i = 0; i < input_shape.size() - 1; i++) {
        output_shape.push_back(input_shape[i]);
    }
    output_shape.push_back(N);
    
    return output.view(output_shape);
}

// Python-callable wrapper for INT4 GEMV (optimized for single vector)
torch::Tensor int4_gemv(
    torch::Tensor input,       // [K]
    torch::Tensor weight,      // [N, K/2] packed INT4
    torch::Tensor scales,      // [N, num_groups]
    int group_size
) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    
    int K = input.size(-1);
    int N = weight.size(0);
    
    auto x = input.contiguous().view({K});
    auto W = weight.contiguous();
    auto S = scales.contiguous();
    
    auto output = torch::empty({N}, torch::TensorOptions().dtype(torch::kHalf).device(input.device()));
    
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    
    int4_gemv_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        W.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(S.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        N, K, group_size
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int4_gemm", &int4_gemm, "INT4 GEMM (CUDA)");
    m.def("int4_gemv", &int4_gemv, "INT4 GEMV (CUDA)");
}
"""
        
        # Compile
        _COMPILED_KERNEL = load_inline(
            name="zse_int4_matmul",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["int4_gemm", "int4_gemv"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-arch=sm_80",  # Ampere+
            ],
            verbose=False,
        )
        
        return _COMPILED_KERNEL
    
    except Exception as e:
        _COMPILE_ERROR = RuntimeError(f"Failed to compile INT4 kernels: {e}")
        raise _COMPILE_ERROR


def is_kernel_available() -> bool:
    """Check if CUDA kernels can be compiled and loaded."""
    try:
        _compile_kernels()
        return True
    except:
        return False


class Int4MatmulFunction(torch.autograd.Function):
    """Autograd function for INT4 matmul (forward only, no backward for inference)."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, 
                scales: torch.Tensor, group_size: int) -> torch.Tensor:
        kernel = _compile_kernels()
        
        # Use GEMM for batched input, GEMV for single vector
        input_flat = input.view(-1, input.size(-1))
        if input_flat.size(0) == 1:
            result = kernel.int4_gemv(input_flat.squeeze(0), weight, scales, group_size)
            return result.unsqueeze(0).view(input.shape[:-1] + (weight.size(0),))
        else:
            return kernel.int4_gemm(input, weight, scales, group_size)


def int4_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    """
    Perform matrix multiplication with INT4 quantized weights.
    
    Args:
        input: Input tensor [*, K] in float16
        weight: Packed INT4 weights [N, K/2] as uint8
        scales: Per-group scales [N, num_groups] in float16
        group_size: Number of elements per quantization group
    
    Returns:
        Output tensor [*, N] in float16
    """
    return Int4MatmulFunction.apply(input, weight, scales, group_size)


class Int4Linear(torch.nn.Module):
    """
    Linear layer with INT4 quantized weights using custom CUDA kernel.
    
    This is a drop-in replacement for QuantizedLinearZSE that uses
    efficient CUDA kernels instead of Python dequantization.
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
        
        # Register buffers (not parameters, since we're not training)
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is float16
        if x.dtype != torch.float16:
            x = x.half()
        
        # Use custom CUDA kernel
        output = int4_matmul(x, self.weight_packed, self.scales, self.group_size)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, group_size={self.group_size}"


# Fallback implementation using PyTorch (for debugging or CPU)
def int4_matmul_pytorch(
    input: torch.Tensor,
    weight_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128
) -> torch.Tensor:
    """
    Pure PyTorch fallback for INT4 matmul (slow but works everywhere).
    """
    N, half_K = weight_packed.shape
    K = half_K * 2
    
    # Unpack INT4
    low = (weight_packed & 0x0F).to(torch.int8) - 8
    high = (weight_packed >> 4).to(torch.int8) - 8
    weight_unpacked = torch.stack([low, high], dim=-1).view(N, K)
    
    # Dequantize
    num_groups = (K + group_size - 1) // group_size
    weight_float = weight_unpacked.float()
    
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, K)
        weight_float[:, start:end] *= scales[:, g:g+1].float()
    
    # Matmul
    return torch.matmul(input.float(), weight_float.T).to(input.dtype)
