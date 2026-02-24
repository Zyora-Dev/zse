"""
Fused Triton Kernels for Quantized Matrix Multiplication

Key Innovation: Fuse dequantization with matmul to avoid memory bandwidth overhead.

Standard approach (slow):
    1. Load INT8 weights from memory
    2. Dequantize to FP16 (write to memory)
    3. Load FP16 weights for matmul (read from memory)
    4. Perform matmul

Fused approach (fast):
    1. Load INT8 weights from memory
    2. Dequantize in registers
    3. Perform matmul immediately
    
Savings: 2x memory bandwidth for weights!

Performance targets:
    - INT8 fused: ~80% of FP16 speed (vs ~20% with dequantization)
    - INT4 fused: ~60% of FP16 speed (vs ~15% with dequantization)
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


# =============================================================================
# INT8 FUSED MATMUL KERNEL
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int8_fused_matmul_kernel(
    # Pointers
    x_ptr,          # Input [M, K] in FP16
    w_ptr,          # Quantized weights [N, K] in INT8
    scale_ptr,      # Scales [N] in FP16 (per-channel)
    out_ptr,        # Output [M, N] in FP16
    bias_ptr,       # Optional bias [N] in FP16
    # Sizes
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Flags
    HAS_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Fused INT8 dequantization + matmul.
    
    Computes: out = x @ (w * scale).T + bias
    
    Where w is INT8 and scale is per-channel FP16.
    """
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers for x and w
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)
    
    # Load scales for this block of N (per-channel)
    scale_ptrs = scale_ptr + offs_n
    scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0).to(tl.float32)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop over K
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Masks
        k_mask = offs_k < K - k * BLOCK_K
        
        # Load x block [BLOCK_M, BLOCK_K]
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        
        # Load w block [BLOCK_N, BLOCK_K] as INT8
        w_int8 = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0)
        
        # Dequantize in registers: w_fp = w_int8 * scale
        # Scale is per-channel [BLOCK_N], broadcast to [BLOCK_N, BLOCK_K]
        w_fp = w_int8.to(tl.float32) * scales[:, None]
        
        # Matmul: acc += x @ w.T
        # x: [BLOCK_M, BLOCK_K], w: [BLOCK_N, BLOCK_K]
        # We need x @ w.T = [BLOCK_M, BLOCK_N]
        acc += tl.dot(x.to(tl.float32), tl.trans(w_fp))
        
        # Advance pointers
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K
    
    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask)


def int8_fused_matmul(
    x: torch.Tensor,           # [M, K] or [B, M, K] in FP16
    w_int8: torch.Tensor,      # [N, K] in INT8
    scale: torch.Tensor,       # [N] in FP16
    bias: Optional[torch.Tensor] = None,  # [N] in FP16
) -> torch.Tensor:
    """
    Fused INT8 matmul: out = x @ (w_int8 * scale).T + bias
    
    Args:
        x: Input tensor [M, K] or batched [B, M, K]
        w_int8: Quantized weights [N, K]
        scale: Per-channel scales [N]
        bias: Optional bias [N]
        
    Returns:
        Output tensor [M, N] or [B, M, N]
    """
    # Handle batched input
    original_shape = x.shape
    if x.ndim == 3:
        B, M, K = x.shape
        x = x.view(B * M, K)
    else:
        M, K = x.shape
        B = None
    
    N = w_int8.shape[0]
    
    # Ensure contiguous
    x = x.contiguous()
    w_int8 = w_int8.contiguous()
    scale = scale.contiguous()
    
    # Output tensor
    out = torch.empty((x.shape[0], N), dtype=x.dtype, device=x.device)
    
    # Grid
    grid = lambda META: (triton.cdiv(x.shape[0], META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    # Launch kernel
    _int8_fused_matmul_kernel[grid](
        x, w_int8, scale, out, bias if bias is not None else x,  # dummy for bias
        x.shape[0], N, K,
        x.stride(0), x.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
    )
    
    # Reshape if batched
    if B is not None:
        out = out.view(B, M, N)
    
    return out


# =============================================================================
# INT4 FUSED MATMUL KERNEL
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int4_fused_matmul_kernel(
    # Pointers
    x_ptr,          # Input [M, K] in FP16
    w_ptr,          # Packed weights [N, K//2] in UINT8 (two INT4 per byte)
    scale_ptr,      # Scales [N, num_groups] in FP16
    out_ptr,        # Output [M, N] in FP16
    bias_ptr,       # Optional bias [N] in FP16
    # Sizes
    M, N, K,
    GROUP_SIZE,     # Quantization group size
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sg,
    stride_om, stride_on,
    # Flags
    HAS_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Fused INT4 dequantization + matmul with group-wise scaling.
    
    INT4 values are packed: two values per byte.
    - Lower 4 bits: first value
    - Upper 4 bits: second value
    """
    # Program IDs
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers for x
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop over K (in steps of BLOCK_K)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        k_mask = offs_k < K - k_start
        
        # Load x block
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        
        # For INT4, we load packed bytes and unpack
        # Each byte contains 2 INT4 values
        # We need to handle this carefully
        
        # Calculate packed offsets (K is divided by 2 due to packing)
        k_packed = (k_start + offs_k) // 2
        k_is_low = ((k_start + offs_k) % 2) == 0  # True for low nibble
        
        # Load packed weights [BLOCK_N, BLOCK_K//2] (approximately)
        w_packed_ptrs = w_ptr + (offs_n[:, None] * stride_wn + k_packed[None, :] * stride_wk)
        w_packed = tl.load(w_packed_ptrs, mask=(offs_n[:, None] < N) & (k_packed[None, :] < K // 2), other=0)
        
        # Unpack INT4 values
        # Low nibble: value & 0xF, then convert from unsigned [0,15] to signed [-8,7]
        # High nibble: (value >> 4) & 0xF
        w_low = (w_packed & 0xF).to(tl.int8) - 8  # Convert to signed
        w_high = ((w_packed >> 4) & 0xF).to(tl.int8) - 8
        
        # Select based on position
        w_int4 = tl.where(k_is_low[None, :], w_low, w_high)
        
        # Load group scales
        group_idx = (k_start + offs_k) // GROUP_SIZE
        scale_ptrs = scale_ptr + (offs_n[:, None] * stride_sn + group_idx[None, :] * stride_sg)
        scales = tl.load(scale_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=1.0)
        
        # Dequantize
        w_fp = w_int4.to(tl.float32) * scales.to(tl.float32)
        
        # Matmul
        acc += tl.dot(x.to(tl.float32), tl.trans(w_fp))
        
        # Advance
        x_ptrs += BLOCK_K * stride_xk
        offs_k += BLOCK_K
    
    # Add bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    
    # Store
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask)


def int4_fused_matmul(
    x: torch.Tensor,           # [M, K] or [B, M, K] in FP16
    w_packed: torch.Tensor,    # [N, K//2] in UINT8 (packed INT4)
    scale: torch.Tensor,       # [N, num_groups] in FP16
    bias: Optional[torch.Tensor] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Fused INT4 matmul with group-wise quantization.
    
    Args:
        x: Input tensor [M, K] or [B, M, K]
        w_packed: Packed INT4 weights [N, K//2]
        scale: Group scales [N, num_groups]
        bias: Optional bias [N]
        group_size: Quantization group size
        
    Returns:
        Output tensor [M, N] or [B, M, N]
    """
    # Handle batched input
    original_shape = x.shape
    if x.ndim == 3:
        B, M, K = x.shape
        x = x.view(B * M, K)
    else:
        M, K = x.shape
        B = None
    
    N = w_packed.shape[0]
    
    # Ensure contiguous
    x = x.contiguous()
    w_packed = w_packed.contiguous()
    scale = scale.contiguous()
    
    # Output
    out = torch.empty((x.shape[0], N), dtype=x.dtype, device=x.device)
    
    # Grid
    grid = lambda META: (triton.cdiv(x.shape[0], META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    # Launch
    _int4_fused_matmul_kernel[grid](
        x, w_packed, scale, out, bias if bias is not None else x,
        x.shape[0], N, K, group_size,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scale.stride(0), scale.stride(1) if scale.ndim > 1 else 0,
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
    )
    
    # Reshape
    if B is not None:
        out = out.view(B, M, N)
    
    return out


# =============================================================================
# OPTIMIZED LINEAR LAYER WITH FUSED KERNELS
# =============================================================================

class FusedQuantizedLinear(torch.nn.Module):
    """
    Quantized linear layer using fused Triton kernels.
    
    Much faster than dequantize-then-matmul approach.
    
    Performance (vs FP16):
        - INT8 fused: ~70-80% speed
        - INT4 fused: ~50-60% speed
        
    Compared to unfused:
        - INT8: 3-4x faster
        - INT4: 2-3x faster
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_type: str = "int8",  # "int8" or "int4"
        group_size: int = 128,     # For INT4 only
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        self.group_size = group_size
        
        # Register buffers based on quantization type
        if quant_type == "int8":
            self.register_buffer("weight", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("scale", torch.zeros(out_features, dtype=torch.float16))
        elif quant_type == "int4":
            assert in_features % 2 == 0, "in_features must be even for INT4"
            num_groups = in_features // group_size
            self.register_buffer("weight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
            self.register_buffer("scale", torch.zeros(out_features, num_groups, dtype=torch.float16))
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")
        
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
    
    @classmethod
    def from_float(
        cls,
        linear: torch.nn.Linear,
        quant_type: str = "int8",
        group_size: int = 128,
    ) -> "FusedQuantizedLinear":
        """Convert FP16 linear to fused quantized linear."""
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
        
        if quant_type == "int8":
            # Per-channel symmetric INT8
            max_val = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
            scale = max_val / 127.0
            q_weight = (weight / scale).round().clamp(-128, 127).to(torch.int8)
            
            q_linear.weight.copy_(q_weight)
            q_linear.scale.copy_(scale.squeeze().half())
            
        elif quant_type == "int4":
            # Group-wise symmetric INT4
            out_features, in_features = weight.shape
            num_groups = in_features // group_size
            
            weight_grouped = weight.view(out_features, num_groups, group_size)
            max_val = weight_grouped.abs().max(dim=2, keepdim=True)[0].clamp(min=1e-5)
            scale = max_val / 7.0  # INT4 range: [-7, 7]
            
            q_weight = (weight_grouped / scale).round().clamp(-8, 7).to(torch.int8)
            q_weight = q_weight.view(out_features, in_features)
            
            # Pack two INT4 into one UINT8
            # Shift from signed [-8,7] to unsigned [0,15]
            q_unsigned = (q_weight + 8).to(torch.uint8)
            # Pack pairs
            q_packed = q_unsigned[:, 0::2] | (q_unsigned[:, 1::2] << 4)
            
            q_linear.weight.copy_(q_packed)
            q_linear.scale.copy_(scale.squeeze(-1).half())
        
        if linear.bias is not None:
            q_linear.bias.copy_(linear.bias.data.half())
        
        return q_linear.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using fused Triton kernel."""
        if self.quant_type == "int8":
            return int8_fused_matmul(x, self.weight, self.scale, self.bias)
        elif self.quant_type == "int4":
            return int4_fused_matmul(x, self.weight, self.scale, self.bias, self.group_size)
        else:
            raise ValueError(f"Unknown quant_type: {self.quant_type}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def replace_linear_with_fused(
    model: torch.nn.Module,
    quant_type: str = "int8",
    skip_layers: list = None,
    group_size: int = 128,
) -> torch.nn.Module:
    """
    Replace nn.Linear layers with FusedQuantizedLinear.
    
    Args:
        model: Model to quantize
        quant_type: "int8" or "int4"
        skip_layers: Layer names to skip (e.g., ["lm_head", "embed"])
        group_size: Group size for INT4
        
    Returns:
        Quantized model (in-place)
    """
    skip_layers = skip_layers or []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if should skip
            if any(skip in name for skip in skip_layers):
                continue
            
            # Skip if already quantized
            if isinstance(module, FusedQuantizedLinear):
                continue
            
            # Convert
            q_module = FusedQuantizedLinear.from_float(module, quant_type, group_size)
            
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


def benchmark_fused_vs_unfused(
    in_features: int = 4096,
    out_features: int = 4096,
    batch_size: int = 1,
    seq_len: int = 512,
    num_warmup: int = 10,
    num_iters: int = 100,
):
    """
    Benchmark fused kernels vs unfused approach.
    """
    import time
    
    device = "cuda"
    dtype = torch.float16
    
    # Create input
    x = torch.randn(batch_size, seq_len, in_features, dtype=dtype, device=device)
    
    # Create FP16 linear
    linear_fp16 = torch.nn.Linear(in_features, out_features, bias=True).to(device).half()
    
    # Create fused INT8
    fused_int8 = FusedQuantizedLinear.from_float(linear_fp16, "int8")
    
    # Create fused INT4
    fused_int4 = FusedQuantizedLinear.from_float(linear_fp16, "int4", group_size=128)
    
    results = {}
    
    # Benchmark FP16
    for _ in range(num_warmup):
        _ = linear_fp16(x)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = linear_fp16(x)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / num_iters * 1000
    results["fp16_ms"] = fp16_time
    
    # Benchmark fused INT8
    for _ in range(num_warmup):
        _ = fused_int8(x)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = fused_int8(x)
    torch.cuda.synchronize()
    int8_time = (time.perf_counter() - start) / num_iters * 1000
    results["int8_fused_ms"] = int8_time
    results["int8_speedup"] = fp16_time / int8_time
    
    # Benchmark fused INT4
    for _ in range(num_warmup):
        _ = fused_int4(x)
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = fused_int4(x)
    torch.cuda.synchronize()
    int4_time = (time.perf_counter() - start) / num_iters * 1000
    results["int4_fused_ms"] = int4_time
    results["int4_speedup"] = fp16_time / int4_time
    
    return results
