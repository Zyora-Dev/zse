"""
ZSE INT4 Matrix Multiplication - Optimized Triton Implementation (v2)

Key features:
1. @triton.autotune - finds best tile config per GPU automatically
2. Software pipelining - hides memory latency
3. Vectorized INT4 unpacking (no scalar tl.where)
4. Weight layout [K//2, N] for coalesced memory access
5. Dedicated GEMV kernel for decode phase (batch_size=1)
6. num_warps/num_stages tuning for Ampere/Hopper

This kernel works directly with ZSE INT4 format after repacking.
No conversion to bitsandbytes format needed.
"""

import torch
from typing import Optional

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


def is_triton_v2_available() -> bool:
    """Check if Triton v2 kernels are available."""
    return _TRITON_AVAILABLE


def get_triton_v2_error() -> Optional[str]:
    """Get error message if Triton v2 isn't available."""
    return _TRITON_ERROR


# =============================================================================
# Repacking functions - convert v1 layout to v2 layout (one-time at load)
# =============================================================================

def repack_weights_for_v2(weight_packed_nk: torch.Tensor) -> torch.Tensor:
    """
    Convert weight layout from [N, K//2] (v1/ZSE) to [K//2, N] (v2).
    
    This is a ONE-TIME operation done at model load time.
    After repacking, all forward passes are faster due to coalesced memory access.
    
    Args:
        weight_packed_nk: Packed INT4 weights [N, K//2] as uint8
        
    Returns:
        Repacked weights [K//2, N] as uint8
    """
    return weight_packed_nk.t().contiguous()


def repack_scales_for_v2(scales_ng: torch.Tensor) -> torch.Tensor:
    """
    Convert scales from [N, num_groups] (v1/ZSE) to [num_groups, N] (v2).
    
    Args:
        scales_ng: Per-group scales [N, num_groups] as float16
        
    Returns:
        Repacked scales [num_groups, N] as float16
    """
    return scales_ng.t().contiguous()


# =============================================================================
# Triton kernels (only defined if Triton is available)
# =============================================================================

if _TRITON_AVAILABLE:
    
    # Autotuning configs for GEMM
    # With per-k scale loading, BLOCK_K can be any size
    _gemm_autotune_configs = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=4),
    ]

    @triton.autotune(configs=_gemm_autotune_configs, key=["M", "N", "K"])
    @triton.jit
    def _int4_gemm_kernel_v2(
        # Pointers
        A_ptr, B_ptr, scales_ptr, C_ptr,
        # Dimensions
        M, N, K,
        # Strides for A [M, K]
        stride_am, stride_ak,
        # Strides for B [K//2, N] (v2 layout)
        stride_bk, stride_bn,
        # Strides for C [M, N]
        stride_cm, stride_cn,
        # Strides for scales [num_groups, N] (v2 layout)
        stride_sg, stride_sn,
        # Quantization
        group_size,
        # Tile sizes (set by autotune)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        INT4 GEMM kernel with correct per-k scale loading (Option 3).
        
        A: [M, K] float16 input activations
        B: [K//2, N] uint8 packed INT4 weights (v2 layout)
        scales: [num_groups, N] float16 scales (v2 layout)
        C: [M, N] float16 output
        
        Packing format:
        - packed[k//2, n] low nibble  = weight[2*(k//2), n]   (even K)
        - packed[k//2, n] high nibble = weight[2*(k//2)+1, n] (odd K)
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_start = k * BLOCK_K

            # Load packed B tile [BLOCK_K//2, BLOCK_N]
            offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K // 2)
            b_packed = tl.load(
                B_ptr + offs_k_packed[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(offs_k_packed[:, None] < K // 2) & (offs_n[None, :] < N),
                other=0
            )

            # Unpack INT4 to signed [-8, 7] - use FP16 for Tensor Cores
            b_even = ((b_packed & 0x0F).to(tl.int8) - 8).to(tl.float16)  # [BLOCK_K//2, BLOCK_N]
            b_odd = ((b_packed >> 4).to(tl.int8) - 8).to(tl.float16)     # [BLOCK_K//2, BLOCK_N]

            # === Option 3: Per-k scale loading ===
            # Even k indices: k_start, k_start+2, k_start+4, ...
            offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2
            group_idxs_even = offs_k_even // group_size  # [BLOCK_K//2]
            scales_even = tl.load(
                scales_ptr + group_idxs_even[:, None] * stride_sg + offs_n[None, :] * stride_sn,
                mask=(offs_k_even[:, None] < K) & (offs_n[None, :] < N),
                other=1.0
            )  # [BLOCK_K//2, BLOCK_N] - already FP16

            # Odd k indices: k_start+1, k_start+3, k_start+5, ...
            offs_k_odd = k_start + tl.arange(0, BLOCK_K // 2) * 2 + 1
            group_idxs_odd = offs_k_odd // group_size  # [BLOCK_K//2]
            scales_odd = tl.load(
                scales_ptr + group_idxs_odd[:, None] * stride_sg + offs_n[None, :] * stride_sn,
                mask=(offs_k_odd[:, None] < K) & (offs_n[None, :] < N),
                other=1.0
            )  # [BLOCK_K//2, BLOCK_N] - already FP16

            # Apply per-k scales - stay in FP16
            b_even = b_even * scales_even  # [BLOCK_K//2, BLOCK_N]
            b_odd = b_odd * scales_odd     # [BLOCK_K//2, BLOCK_N]

            # Load A for even K indices - keep as FP16
            a_even = tl.load(
                A_ptr + offs_m[:, None] * stride_am + offs_k_even[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k_even[None, :] < K),
                other=0.0
            )  # already FP16
            
            # Load A for odd K indices - keep as FP16
            a_odd = tl.load(
                A_ptr + offs_m[:, None] * stride_am + offs_k_odd[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k_odd[None, :] < K),
                other=0.0
            )  # already FP16

            # Accumulate: FP16 inputs, FP32 accumulator - enables Tensor Cores
            acc = tl.dot(a_even, b_even, acc, out_dtype=tl.float32)
            acc = tl.dot(a_odd, b_odd, acc, out_dtype=tl.float32)

        # Store output
        offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C_ptr + offs_m_out[:, None] * stride_cm + offs_n_out[None, :] * stride_cn
        c_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

    # Autotuning configs for GEMV (decode phase)
    # With per-k scale loading, BLOCK_K can be any size
    _gemv_autotune_configs = [
        triton.Config({"BLOCK_K": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_K": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_K": 256, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_K": 256, "BLOCK_N": 128}, num_warps=8),
    ]

    @triton.autotune(configs=_gemv_autotune_configs, key=["N", "K"])
    @triton.jit
    def _int4_gemv_kernel_v2(
        x_ptr, W_ptr, scales_ptr, y_ptr,
        N, K,
        stride_wk, stride_wn,
        stride_sg, stride_sn,
        group_size,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        GEMV kernel with correct per-k scale loading (Option 3).
        
        x: [K] float16 input
        W: [K//2, N] uint8 packed INT4 weights (v2 layout)
        scales: [num_groups, N] float16 scales (v2 layout)
        y: [N] float16 output
        """
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_start = k * BLOCK_K
            
            # Load packed weights [BLOCK_K//2, BLOCK_N]
            offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K // 2)
            w_packed = tl.load(
                W_ptr + offs_k_packed[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=(offs_k_packed[:, None] < K // 2) & n_mask[None, :],
                other=0,
            )

            # Unpack INT4 to signed [-8, 7] - use FP16
            w_even = ((w_packed & 0x0F).to(tl.int8) - 8).to(tl.float16)  # [BLOCK_K//2, BLOCK_N]
            w_odd = ((w_packed >> 4).to(tl.int8) - 8).to(tl.float16)     # [BLOCK_K//2, BLOCK_N]

            # === Option 3: Per-k scale loading ===
            # Even k indices
            offs_k_even = k_start + tl.arange(0, BLOCK_K // 2) * 2
            group_idxs_even = offs_k_even // group_size  # [BLOCK_K//2]
            scales_even = tl.load(
                scales_ptr + group_idxs_even[:, None] * stride_sg + offs_n[None, :] * stride_sn,
                mask=(offs_k_even[:, None] < K) & n_mask[None, :],
                other=1.0
            )  # [BLOCK_K//2, BLOCK_N] - already FP16

            # Odd k indices  
            offs_k_odd = k_start + tl.arange(0, BLOCK_K // 2) * 2 + 1
            group_idxs_odd = offs_k_odd // group_size  # [BLOCK_K//2]
            scales_odd = tl.load(
                scales_ptr + group_idxs_odd[:, None] * stride_sg + offs_n[None, :] * stride_sn,
                mask=(offs_k_odd[:, None] < K) & n_mask[None, :],
                other=1.0
            )  # [BLOCK_K//2, BLOCK_N] - already FP16

            # Apply per-k scales - stay in FP16
            w_even = w_even * scales_even  # [BLOCK_K//2, BLOCK_N]
            w_odd = w_odd * scales_odd     # [BLOCK_K//2, BLOCK_N]

            # Load x values - keep as FP16, compute in FP32 for accumulation
            x_even = tl.load(x_ptr + offs_k_even, mask=offs_k_even < K, other=0.0)
            x_odd = tl.load(x_ptr + offs_k_odd, mask=offs_k_odd < K, other=0.0)

            # Dot product - FP16 multiply, FP32 accumulate
            acc += tl.sum((x_even[:, None] * w_even).to(tl.float32), axis=0)
            acc += tl.sum((x_odd[:, None] * w_odd).to(tl.float32), axis=0)

        tl.store(y_ptr + offs_n, acc.to(tl.float16), mask=n_mask)


def int4_matmul_triton_v2(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Optimized INT4 matrix multiplication using Triton v2 kernels.
    
    Args:
        x: Input tensor [*, K] as float16
        weight_packed: Packed INT4 weights [K//2, N] as uint8 (v2 layout)
        scales: Per-group scales [num_groups, N] as float16 (v2 layout)
        group_size: Quantization group size
        
    Returns:
        Output tensor [*, N] as float16
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError(f"Triton not available: {_TRITON_ERROR}")

    orig_shape = x.shape
    x_flat = x.reshape(-1, x.shape[-1]).contiguous()
    if x_flat.dtype != torch.float16:
        x_flat = x_flat.half()

    M, K = x_flat.shape
    half_K, N = weight_packed.shape

    assert half_K == K // 2, f"Weight shape mismatch: got [K//2={half_K}, N={N}], expected K//2={K // 2}"

    output = torch.empty((M, N), dtype=torch.float16, device=x.device)

    # Route to GEMV for single-token decode (the hot path)
    if M == 1:
        grid = (triton.cdiv(N, 32),)
        _int4_gemv_kernel_v2[grid](
            x_flat.squeeze(0), weight_packed, scales, output.squeeze(0),
            N, K,
            weight_packed.stride(0), weight_packed.stride(1),
            scales.stride(0), scales.stride(1),
            group_size,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _int4_gemm_kernel_v2[grid](
            x_flat, weight_packed, scales, output,
            M, N, K,
            x_flat.stride(0), x_flat.stride(1),
            weight_packed.stride(0), weight_packed.stride(1),
            output.stride(0), output.stride(1),
            scales.stride(0), scales.stride(1),
            group_size,
        )

    return output.view(orig_shape[:-1] + (N,))


# =============================================================================
# PyTorch Module wrapper
# =============================================================================

class TritonInt4LinearV2(torch.nn.Module):
    """
    INT4 Linear layer using optimized Triton v2 kernels.
    
    Expects weights in v2 layout:
    - weight_packed: [K//2, N] (transposed from ZSE's [N, K//2])
    - scales: [num_groups, N] (transposed from ZSE's [N, num_groups])
    
    Use from_zse_layout() to convert from standard ZSE layout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_packed: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)

    @classmethod
    def from_zse_layout(
        cls,
        in_features: int,
        out_features: int,
        weight_packed_nk: torch.Tensor,
        scales_ng: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ) -> "TritonInt4LinearV2":
        """
        Create layer from ZSE's standard v1 layout.
        
        Args:
            weight_packed_nk: [N, K//2] packed INT4 (ZSE layout)
            scales_ng: [N, num_groups] scales (ZSE layout)
        """
        return cls(
            in_features, out_features,
            repack_weights_for_v2(weight_packed_nk),
            repack_scales_for_v2(scales_ng),
            bias, group_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = int4_matmul_triton_v2(x, self.weight_packed, self.scales, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"group_size={self.group_size}"
        )
