"""ZSE Inference Kernels — All LLM GPU kernels for the forward pass.

Complex kernels (dequant_matmul, paged_attention) are written as raw CUDA C
strings and compiled via RuntimeCompiler. Simpler kernels use @zse.kernel.

This is pragmatic: the AST parser handles simple element-wise ops well, but
complex tiling + shared memory + INT4 unpacking patterns exceed its current
capabilities. We can migrate to @zse.kernel as the parser improves.

All kernels target fp16 compute (matching .zse weight format).
"""

from dataclasses import dataclass
from typing import Dict, Optional

from zse_compiler.runtime.compiler import RuntimeCompiler, CompiledKernel
from zse_compiler.runtime.launcher import KernelLauncher, LaunchConfig
from zse_compiler.types.tensor import Tensor

# Portable @zse.kernel MFMA INT4 dequant matmul (Phase 3, ROCm/CDNA only).
# Pre-generate HIP source at module import (pure Python AST->IR->HIP C; no GPU).
from zse_engine.orchestrator.portable_kernels import mfma_dequant_matmul_int4_v3 as _mfma_v3_kernel
try:
    MFMA_DEQUANT_MATMUL_INT4_V3_HIP = _mfma_v3_kernel.source("rocm")
except Exception:
    MFMA_DEQUANT_MATMUL_INT4_V3_HIP = None

# Portable @zse.kernel wave-64 batched INT4 GEMV (small-M decode, ROCm-tuned).
# 2.13-2.26x faster than the hand-written batched_dequant_gemv_int4 at M=4 on MI300X.
from zse_engine.orchestrator.portable_kernels import bgemv_int4_wave64 as _bgemv_wave64_kernel
try:
    BGEMV_INT4_WAVE64_HIP = _bgemv_wave64_kernel.source("rocm")
except Exception:
    BGEMV_INT4_WAVE64_HIP = None

# Portable @zse.kernel fused RoPE + KV-cache-write (decode path).
# Collapses batched_rotary_embedding + batched_kv_cache_write into one launch
# and eliminates the K-buffer round-trip (rotated K goes straight to KV cache).
from zse_engine.orchestrator.portable_kernels import fused_rope_kv_write as _fused_rope_kv_kernel
try:
    FUSED_ROPE_KV_WRITE_HIP = _fused_rope_kv_kernel.source("rocm")
except Exception:
    FUSED_ROPE_KV_WRITE_HIP = None


# ============================================================================
# Hand-tuned wave-64 INT4 GEMV with 128-bit (uint4) weight loads — ROCm only
# ============================================================================
# Profiler showed bgemv_int4_wave64 = 64% of decode GPU time at ~13% of MI300X
# HBM3 peak. Root cause: 32-bit weight loads under-utilize the memory pipeline.
# This v2 reads 128 bits (4x uint32 = 32 nibbles) per memory transaction.
#
# Constraints: K % 32 == 0, group_size % 32 == 0 (both true for Qwen2.5-32B:
# K in {5120, 27648}, group_size = 128).
#
# Layout identical to v1: grid = (ceil(N/8),), block = (512,) = 8 wavefronts,
# 1 wavefront per N-row, fully coalesced loads across lanes in a wavefront.
BGEMV_INT4_WAVE64_V2_HIP = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void bgemv_int4_wave64_v2(
    __half* __restrict__ out,
    const unsigned char* __restrict__ weight,
    const __half* __restrict__ scales,
    const __half* __restrict__ zeros,
    const __half* __restrict__ inp,
    int M, int N, int K, int group_size)
{
    int tid    = threadIdx.x;
    int wf_id  = tid >> 6;
    int lane   = tid & 63;
    int row    = blockIdx.x * 8 + wf_id;
    if (row >= N) return;

    int num_groups = (K + group_size - 1) / group_size;
    int half_K     = K >> 1;
    int num_u4     = half_K >> 4;   // K / 32 (each uint4 = 32 nibbles)

    const uint4* wq4 = reinterpret_cast<const uint4*>(weight + (size_t)row * (size_t)half_K);

    float acc[8];
    #pragma unroll
    for (int m = 0; m < 8; m++) acc[m] = 0.0f;

    int prev_g = -1;
    float s_val = 0.0f;
    float z_val = 0.0f;
    int scale_row_off = row * num_groups;

    for (int i = lane; i < num_u4; i += 64) {
        uint4 packed = wq4[i];

        int nibbles[32];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            nibbles[j     ] = (int)((packed.x >> (4*j)) & 0xF);
            nibbles[j +  8] = (int)((packed.y >> (4*j)) & 0xF);
            nibbles[j + 16] = (int)((packed.z >> (4*j)) & 0xF);
            nibbles[j + 24] = (int)((packed.w >> (4*j)) & 0xF);
        }

        int k_base = i * 32;
        int g = k_base / group_size;
        if (g != prev_g) {
            s_val = __half2float(scales[scale_row_off + g]);
            z_val = __half2float(zeros [scale_row_off + g]);
            prev_g = g;
        }

        float w_dq[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            w_dq[j] = (float)nibbles[j] * s_val + z_val;
        }

        #pragma unroll
        for (int m = 0; m < 8; m++) {
            if (m < M) {
                int inp_row_off = m * K + k_base;
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    acc[m] += w_dq[j] * __half2float(inp[inp_row_off + j]);
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < 8; m++) {
        if (m < M) {
            float v = acc[m];
            v += __shfl_xor(v, 32);
            v += __shfl_xor(v, 16);
            v += __shfl_xor(v,  8);
            v += __shfl_xor(v,  4);
            v += __shfl_xor(v,  2);
            v += __shfl_xor(v,  1);
            if (lane == 0) {
                if (v >  65504.0f) v =  65504.0f;
                if (v < -65504.0f) v = -65504.0f;
                out[m * N + row] = __float2half(v);
            }
        }
    }
}
"""


# ============================================================================
# CUDA C Kernel Sources
# ============================================================================

# Common header for all CUDA kernels — required for half/fp16 support
# NVRTC doesn't have default include paths, so we use --include-path
# at compile time. The header is just the include directive.
CUDA_HEADER = r"""
#include <cuda_fp16.h>
#define WARP_SIZE 32
"""

RMSNORM_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void rmsnorm(
    half* __restrict__ out,
    const half* __restrict__ input,
    const half* __restrict__ weight,
    int hidden_size,
    float eps
) {
    // One block per token (row)
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const half* x = input + row * hidden_size;
    half* o = out + row * hidden_size;

    // Compute sum of squares (use float accumulator)
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    // Cross-warp via shared memory
    __shared__ float warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    // Broadcast rms to all threads
    __shared__ float rms_shared;
    if (tid == 0) {
        rms_shared = rsqrtf(sum_sq / (float)hidden_size + eps);
    }
    __syncthreads();
    float rms = rms_shared;

    // Apply normalization + weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]);
        float w = __half2float(weight[i]);
        o[i] = __float2half(val * rms * w);
    }
}
"""

SILU_MUL_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void silu_mul(
    half* __restrict__ out,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        // SiLU(x) = x * sigmoid(x)
        float silu_g = g / (1.0f + expf(-g));
        out[idx] = __float2half(silu_g * u);
    }
}
"""

ROTARY_EMBEDDING_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void rotary_embedding(
    half* __restrict__ q,       // [seq_len, num_heads, head_dim]
    half* __restrict__ k,       // [seq_len, num_kv_heads, head_dim]
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    const int* __restrict__ pos_buf,  // GPU buffer: pos_buf[0] = position_offset
    float theta_base
) {
    int position_offset = pos_buf[0];
    // One thread per (position, head, pair)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_q = seq_len * num_heads * half_dim;
    int total_k = seq_len * num_kv_heads * half_dim;

    if (idx < total_q) {
        // Decode position, head, pair index
        int pair = idx % half_dim;
        int head = (idx / half_dim) % num_heads;
        int pos = idx / (half_dim * num_heads);

        float freq = 1.0f / powf(theta_base, (float)(2 * pair) / (float)head_dim);
        float angle = (float)(pos + position_offset) * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int base = pos * num_heads * head_dim + head * head_dim;
        float q0 = __half2float(q[base + pair]);
        float q1 = __half2float(q[base + pair + half_dim]);
        q[base + pair]            = __float2half(q0 * cos_val - q1 * sin_val);
        q[base + pair + half_dim] = __float2half(q1 * cos_val + q0 * sin_val);
    }

    if (idx < total_k) {
        int pair = idx % half_dim;
        int head = (idx / half_dim) % num_kv_heads;
        int pos = idx / (half_dim * num_kv_heads);

        float freq = 1.0f / powf(theta_base, (float)(2 * pair) / (float)head_dim);
        float angle = (float)(pos + position_offset) * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int base = pos * num_kv_heads * head_dim + head * head_dim;
        float k0 = __half2float(k[base + pair]);
        float k1 = __half2float(k[base + pair + half_dim]);
        k[base + pair]            = __float2half(k0 * cos_val - k1 * sin_val);
        k[base + pair + half_dim] = __float2half(k1 * cos_val + k0 * sin_val);
    }
}
"""

RESIDUAL_ADD_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void residual_add(
    half* __restrict__ out,
    const half* __restrict__ a,
    const half* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(a[idx]) + __half2float(b[idx]);
        val = fmaxf(-65504.0f, fminf(65504.0f, val));
        out[idx] = __float2half(val);
    }
}
"""

BIAS_ADD_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void bias_add(
    half* __restrict__ out,       // [M, N] — modified in-place
    const half* __restrict__ bias, // [N]
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int col = idx % N;
        out[idx] = __float2half(__half2float(out[idx]) + __half2float(bias[col]));
    }
}
"""

EMBEDDING_LOOKUP_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void embedding_lookup(
    half* __restrict__ out,       // [seq_len, hidden_size]
    const half* __restrict__ table, // [vocab_size, hidden_size]
    const int* __restrict__ tokens, // [seq_len]
    int hidden_size,
    int seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len * hidden_size) {
        int tok_idx = idx / hidden_size;
        int dim_idx = idx % hidden_size;
        int token_id = tokens[tok_idx];
        out[idx] = table[token_id * hidden_size + dim_idx];
    }
}
"""

# Dequantize INT4 and multiply: out = dequant(W) @ x
# W is stored as packed uint8 (2 values per byte), with per-group scales and zeros
DEQUANT_MATMUL_INT4_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void dequant_matmul_int4(
    half* __restrict__ out,         // [M, N] output
    const unsigned char* __restrict__ weight, // [N, K/2] packed INT4
    const half* __restrict__ scales,  // [N, K/group_size] scales
    const half* __restrict__ zeros,   // [N, K/group_size] zero-points
    const half* __restrict__ input,   // [M, K] input
    int M, int N, int K,
    int group_size
) {
    // Each thread computes one output element out[m, n]
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    int num_groups = (K + group_size - 1) / group_size;

    for (int k = 0; k < K; k++) {
        // Dequantize weight[n, k]
        int group_idx = k / group_size;
        float scale = __half2float(scales[n * num_groups + group_idx]);
        float zero = __half2float(zeros[n * num_groups + group_idx]);

        // Unpack INT4 from packed byte
        int byte_idx = (n * K + k) / 2;
        unsigned char packed = weight[byte_idx];
        int nibble;
        if (k % 2 == 0) {
            nibble = packed & 0x0F;
        } else {
            nibble = (packed >> 4) & 0x0F;
        }

        float w = (float)nibble * scale + zero;
        float x = __half2float(input[m * K + k]);
        acc += w * x;
    }

    out[m * N + n] = __float2half(acc);
}
"""

DEQUANT_MATMUL_INT8_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void dequant_matmul_int8(
    half* __restrict__ out,         // [M, N] output
    const signed char* __restrict__ weight, // [N, K] INT8
    const half* __restrict__ scales,  // [N, K/group_size] scales
    const half* __restrict__ input,   // [M, K] input
    int M, int N, int K,
    int group_size
) {
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    int num_groups = (K + group_size - 1) / group_size;

    for (int k = 0; k < K; k++) {
        int group_idx = k / group_size;
        float scale = __half2float(scales[n * num_groups + group_idx]);

        float w = (float)weight[n * K + k] * scale;
        float x = __half2float(input[m * K + k]);
        acc += w * x;
    }

    out[m * N + n] = __float2half(acc);
}
"""

# FP16 matmul for embedding/lm_head (no quantization)
FP16_MATMUL_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void fp16_matmul(
    half* __restrict__ out,       // [M, N]
    const half* __restrict__ A,   // [M, K]
    const half* __restrict__ B,   // [K, N] (column-major for B, or [N, K] row-major transposed)
    int M, int N, int K
) {
    int m = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = __half2float(A[m * K + k]);
        float b = __half2float(B[n * K + k]);  // B is [N, K] row-major (transposed)
        acc += a * b;
    }
    out[m * N + n] = __float2half(acc);
}
"""

SOFTMAX_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void softmax(
    half* __restrict__ out,
    const half* __restrict__ input,
    int rows,
    int cols
) {
    // One block per row — online softmax (numerically stable)
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const half* x = input + row * cols;
    half* o = out + row * cols;

    // Phase 1: find max
    float max_val = -1e30f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float v = __half2float(x[i]);
        if (v > max_val) max_val = v;
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));

    __shared__ float warp_max[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_max[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0) {
        max_val = (lane_id < (blockDim.x + 31) / 32) ? warp_max[lane_id] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    __shared__ float max_shared;
    if (tid == 0) max_shared = max_val;
    __syncthreads();
    max_val = max_shared;

    // Phase 2: compute exp sum
    float sum_exp = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum_exp += expf(__half2float(x[i]) - max_val);
    }

    // Reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);

    __shared__ float warp_sum[32];
    if (lane_id == 0) warp_sum[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0) {
        sum_exp = (lane_id < (blockDim.x + 31) / 32) ? warp_sum[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float sum_shared;
    if (tid == 0) sum_shared = sum_exp;
    __syncthreads();
    sum_exp = sum_shared;

    // Phase 3: normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < cols; i += blockDim.x) {
        o[i] = __float2half(expf(__half2float(x[i]) - max_val) * inv_sum);
    }
}
"""

# Paged attention: reads K/V from paged block tables
# Simplified single-head-per-block version
PAGED_ATTENTION_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void paged_attention(
    half* __restrict__ out,             // [num_seqs, num_heads, head_dim]
    const half* __restrict__ q,         // [num_seqs, num_heads, head_dim]
    const half* __restrict__ kv_cache,  // Contiguous GPU slab
    const int* __restrict__ block_table, // [num_seqs, max_blocks]
    const int* __restrict__ seq_lens,   // [num_seqs]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,                     // Tokens per block
    int max_blocks_per_seq,
    int num_layers,
    int layer_idx,                      // Which layer's KV to read
    float scale                         // 1/sqrt(head_dim)
) {
    // Grid: (num_seqs, num_heads)
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);  // GQA mapping

    // Block layout: block holds all layers' KV
    // Per block: [num_layers, 2(K+V), num_kv_heads, block_size, head_dim] in fp16
    int kv_head_stride = block_size * head_dim;
    int kv_type_stride = num_kv_heads * kv_head_stride;
    int layer_stride = 2 * kv_type_stride;
    int block_bytes_in_halfs = num_layers * layer_stride;

    // Load Q for this head
    const half* q_ptr = q + seq_idx * num_heads * head_dim + head_idx * head_dim;

    // Compute attention scores over all KV tokens
    // Using shared memory for scores
    extern __shared__ float smem[];
    float* scores = smem;  // [seq_len]

    float max_score = -1e30f;

    // Iterate over blocks
    int num_blocks = (seq_len + block_size - 1) / block_size;
    for (int b = 0; b < num_blocks; b++) {
        int block_id = block_table[seq_idx * max_blocks_per_seq + b];
        if (block_id < 0) continue;  // Evicted block

        int tokens_in_block = min(block_size, seq_len - b * block_size);

        // K pointer for this block, layer, kv_head
        const half* k_base = kv_cache + block_id * block_bytes_in_halfs
                            + layer_idx * layer_stride
                            + kv_head_idx * kv_head_stride;

        for (int t = tid; t < tokens_in_block; t += blockDim.x) {
            // Dot product: Q · K[t]
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += __half2float(q_ptr[d]) * __half2float(k_base[t * head_dim + d]);
            }
            dot *= scale;

            int global_t = b * block_size + t;
            scores[global_t] = dot;
            if (dot > max_score) max_score = dot;
        }
    }
    __syncthreads();

    // Reduce max across ALL warps (not just warp 0)
    for (int offset = 16; offset > 0; offset >>= 1)
        max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffff, max_score, offset));

    __shared__ float warp_maxes[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_maxes[warp_id] = max_score;
    __syncthreads();

    if (warp_id == 0) {
        max_score = (lane_id < (blockDim.x + 31) / 32) ? warp_maxes[lane_id] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffff, max_score, offset));
    }
    __shared__ float smax;
    if (tid == 0) smax = max_score;
    __syncthreads();
    max_score = smax;

    // Softmax: exp and sum
    float sum_exp = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }

    // Reduce sum across ALL warps
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);

    __shared__ float warp_sums[32];
    if (lane_id == 0) warp_sums[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0) {
        sum_exp = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float ssum;
    if (tid == 0) ssum = sum_exp;
    __syncthreads();
    sum_exp = ssum;

    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of V
    half* out_ptr = out + seq_idx * num_heads * head_dim + head_idx * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            int block_id = block_table[seq_idx * max_blocks_per_seq + b];
            if (block_id < 0) continue;

            int tokens_in_block = min(block_size, seq_len - b * block_size);

            // V pointer: offset by kv_type_stride from K
            const half* v_base = kv_cache + block_id * block_bytes_in_halfs
                                + layer_idx * layer_stride
                                + kv_type_stride  // V comes after K
                                + kv_head_idx * kv_head_stride;

            for (int t = 0; t < tokens_in_block; t++) {
                int global_t = b * block_size + t;
                acc += scores[global_t] * __half2float(v_base[t * head_dim + d]);
            }
        }
        out_ptr[d] = __float2half(acc);
    }
}
"""

# ============================================================================
# Paged Attention v2 — ROCm-only, GQA-aware + online softmax (FA-1 style)
# ============================================================================
# Grid:  (num_seqs, num_kv_heads)         — 5× fewer blocks than v1, 5× less KV HBM traffic
# Block: (head_dim,)  e.g. 128             — one thread per output dim, full wavefront util
# Each block loads K/V for ONE kv_head ONCE and computes the full attention output
# for all `qpkv = num_heads / num_kv_heads` query heads that share it.
# Online softmax means K and V are each read exactly ONCE (no scores[seq_len] scratch).
#
# Shared memory layout (bytes):
#   k_tile  : block_size * head_dim * 2
#   v_tile  : block_size * head_dim * 2
#   xreduce : num_warps  * block_size * 4   (cross-warp dot-product reduction)
# Total at Qwen2.5-32B (block_size=8, head_dim=128, num_warps=2): 4096 + 4096 + 64 = 8256 B
PAGED_ATTENTION_V2_HIP = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define QPKV_MAX 16
#define BLOCK_SIZE_MAX 32

extern "C" __global__ void paged_attention_v2(
    half* __restrict__ out,             // [num_seqs, num_heads, head_dim]
    const half* __restrict__ q,         // [num_seqs, num_heads, head_dim]
    const half* __restrict__ kv_cache,
    const int* __restrict__ block_table, // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,    // [num_seqs]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_layers,
    int layer_idx,
    float scale
) {
    int seq_idx = blockIdx.x;
    int kv_head_idx = blockIdx.y;
    int tid = threadIdx.x;          // 0 .. head_dim-1
    int qpkv = num_heads / num_kv_heads;

    int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) {
        // Zero-fill output so downstream kernels see deterministic memory.
        for (int h = 0; h < qpkv; h++) {
            int head_idx = kv_head_idx * qpkv + h;
            out[seq_idx * num_heads * head_dim + head_idx * head_dim + tid] = __float2half(0.0f);
        }
        return;
    }

    // KV cache strides (matches v1 layout)
    int kv_head_stride = block_size * head_dim;
    int kv_type_stride = num_kv_heads * kv_head_stride;
    int layer_stride   = 2 * kv_type_stride;
    int block_bytes_in_halfs = num_layers * layer_stride;

    // Load Q for all qpkv heads that share this kv_head — one dim per thread, into registers
    float q_reg[QPKV_MAX];
    #pragma unroll
    for (int h = 0; h < QPKV_MAX; h++) q_reg[h] = 0.0f;
    for (int h = 0; h < qpkv; h++) {
        int head_idx = kv_head_idx * qpkv + h;
        q_reg[h] = __half2float(q[seq_idx * num_heads * head_dim + head_idx * head_dim + tid]);
    }

    // Per-thread online-softmax state: this thread's dim of acc, per query head
    float acc[QPKV_MAX];
    float m_state[QPKV_MAX];
    float s_state[QPKV_MAX];
    #pragma unroll
    for (int h = 0; h < QPKV_MAX; h++) {
        acc[h]     = 0.0f;
        m_state[h] = -1e30f;
        s_state[h] = 0.0f;
    }

    // Shared memory: k_tile | v_tile | xreduce
    extern __shared__ char smem_raw[];
    half*  k_tile   = reinterpret_cast<half*>(smem_raw);
    half*  v_tile   = k_tile + block_size * head_dim;
    float* xreduce = reinterpret_cast<float*>(v_tile + block_size * head_dim);

    int warp_id   = tid / warpSize;
    int lane_id   = tid % warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;

    int num_blocks_seq = (seq_len + block_size - 1) / block_size;
    for (int b = 0; b < num_blocks_seq; b++) {
        int block_id = block_table[seq_idx * max_blocks_per_seq + b];
        if (block_id < 0) continue;

        int tokens_in_block = block_size;
        int tail = seq_len - b * block_size;
        if (tail < tokens_in_block) tokens_in_block = tail;

        // Cooperative load of K and V tiles into shared memory (coalesced)
        const half* k_base = kv_cache + block_id * block_bytes_in_halfs
                             + layer_idx * layer_stride
                             + kv_head_idx * kv_head_stride;
        const half* v_base = k_base + kv_type_stride;

        int load_total = block_size * head_dim;
        for (int i = tid; i < load_total; i += blockDim.x) {
            k_tile[i] = k_base[i];
            v_tile[i] = v_base[i];
        }
        __syncthreads();

        // For each query head sharing this kv_head:
        for (int h = 0; h < qpkv; h++) {
            // ---- QK^T: produce scores_local[t] for t in [0, tokens_in_block) ----
            // Each thread tid contributes q_reg[h] * k_tile[t][tid] to dot product.
            // Reduce across head_dim threads: wavefront-internal shfl, then cross-warp via xreduce.
            float scores_local[BLOCK_SIZE_MAX];
            #pragma unroll
            for (int t = 0; t < BLOCK_SIZE_MAX; t++) scores_local[t] = 0.0f;

            for (int t = 0; t < tokens_in_block; t++) {
                float partial = q_reg[h] * __half2float(k_tile[t * head_dim + tid]);
                // Wavefront reduce (works for wavefront=64 on AMD, warp=32 on NVIDIA)
                for (int off = warpSize / 2; off > 0; off >>= 1) {
                    partial += __shfl_xor(partial, off);
                }
                if (lane_id == 0) {
                    xreduce[warp_id * block_size + t] = partial;
                }
            }
            __syncthreads();

            // Cross-warp reduce in warp 0 — produces final scaled dot per token
            if (warp_id == 0 && lane_id < tokens_in_block) {
                float s = 0.0f;
                for (int w = 0; w < num_warps; w++) {
                    s += xreduce[w * block_size + lane_id];
                }
                xreduce[lane_id] = s * scale;
            }
            __syncthreads();

            // All threads read final scores
            for (int t = 0; t < tokens_in_block; t++) {
                scores_local[t] = xreduce[t];
            }

            // ---- Online softmax update ----
            float m_tile = scores_local[0];
            for (int t = 1; t < tokens_in_block; t++) {
                m_tile = fmaxf(m_tile, scores_local[t]);
            }
            float m_new = fmaxf(m_state[h], m_tile);
            float alpha = expf(m_state[h] - m_new);

            float p_t[BLOCK_SIZE_MAX];
            #pragma unroll
            for (int t = 0; t < BLOCK_SIZE_MAX; t++) p_t[t] = 0.0f;
            float sum_tile = 0.0f;
            for (int t = 0; t < tokens_in_block; t++) {
                p_t[t]    = expf(scores_local[t] - m_new);
                sum_tile += p_t[t];
            }

            // ---- Update accumulator (this thread's dim) ----
            float new_acc = acc[h] * alpha;
            for (int t = 0; t < tokens_in_block; t++) {
                new_acc += p_t[t] * __half2float(v_tile[t * head_dim + tid]);
            }
            acc[h]     = new_acc;
            s_state[h] = s_state[h] * alpha + sum_tile;
            m_state[h] = m_new;
            __syncthreads();  // safe to overwrite xreduce on next h/b iteration
        }
    }

    // Final normalize + writeback
    for (int h = 0; h < qpkv; h++) {
        int head_idx = kv_head_idx * qpkv + h;
        float result = (s_state[h] > 0.0f) ? (acc[h] / s_state[h]) : 0.0f;
        out[seq_idx * num_heads * head_dim + head_idx * head_dim + tid] = __float2half(result);
    }
}
"""

# Prefill attention: causal self-attention for multiple query positions
# Grid: (seq_len, num_heads), Block: (head_dim_threads)
# Each block computes attention for one (query_pos, head) pair
PREFILL_ATTENTION_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void prefill_attention(
    half* __restrict__ out,             // [seq_len, num_heads, head_dim]
    const half* __restrict__ q,         // [seq_len, num_heads, head_dim]
    const half* __restrict__ k,         // [seq_len, num_kv_heads, head_dim]
    const half* __restrict__ v,         // [seq_len, num_kv_heads, head_dim]
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int q_pos = blockIdx.x;     // Which query position (0..seq_len-1)
    int head_idx = blockIdx.y;  // Which attention head
    int tid = threadIdx.x;

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);  // GQA

    const half* q_ptr = q + q_pos * num_heads * head_dim + head_idx * head_dim;

    // Shared memory for attention scores (causal: only positions 0..q_pos)
    extern __shared__ float smem[];
    float* scores = smem;

    int attend_len = q_pos + 1;  // Causal: attend to positions 0..q_pos

    // Compute Q·K scores
    float max_score = -1e30f;
    for (int kv_pos = tid; kv_pos < attend_len; kv_pos += blockDim.x) {
        const half* k_ptr = k + kv_pos * num_kv_heads * head_dim + kv_head_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
        dot *= scale;
        scores[kv_pos] = dot;
        if (dot > max_score) max_score = dot;
    }
    __syncthreads();

    // Cross-warp max reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffff, max_score, offset));

    __shared__ float warp_maxes[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_maxes[warp_id] = max_score;
    __syncthreads();

    if (warp_id == 0) {
        max_score = (lane_id < (blockDim.x + 31) / 32) ? warp_maxes[lane_id] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1)
            max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffff, max_score, offset));
    }
    __shared__ float smax;
    if (tid == 0) smax = max_score;
    __syncthreads();
    max_score = smax;

    // Softmax
    float sum_exp = 0.0f;
    for (int i = tid; i < attend_len; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_score);
        sum_exp += scores[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
    __shared__ float warp_sums[32];
    if (lane_id == 0) warp_sums[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        sum_exp = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
    }
    __shared__ float ssum;
    if (tid == 0) ssum = sum_exp;
    __syncthreads();
    sum_exp = ssum;

    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
    for (int i = tid; i < attend_len; i += blockDim.x) {
        scores[i] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of V
    half* out_ptr = out + q_pos * num_heads * head_dim + head_idx * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int kv_pos = 0; kv_pos < attend_len; kv_pos++) {
            const half* v_ptr = v + kv_pos * num_kv_heads * head_dim + kv_head_idx * head_dim;
            acc += scores[kv_pos] * __half2float(v_ptr[d]);
        }
        out_ptr[d] = __float2half(acc);
    }
}
"""

# Write KV to cache: stores Q/K projections into paged blocks
KV_CACHE_WRITE_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void kv_cache_write(
    half* __restrict__ kv_cache,       // Contiguous GPU slab
    const half* __restrict__ k_proj,   // [seq_len, num_kv_heads, head_dim]
    const half* __restrict__ v_proj,   // [seq_len, num_kv_heads, head_dim]
    const int* __restrict__ block_table, // [max_blocks]
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_layers,
    int layer_idx,
    const int* __restrict__ pos_buf    // GPU buffer: pos_buf[0] = position_offset
) {
    int position_offset = pos_buf[0];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_kv_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int h = (idx / head_dim) % num_kv_heads;
    int t = idx / (head_dim * num_kv_heads);

    int global_pos = position_offset + t;
    int block_idx = global_pos / block_size;
    int token_in_block = global_pos % block_size;

    int block_id = block_table[block_idx];
    if (block_id < 0) return;

    // Block layout: [num_layers, 2(K+V), num_kv_heads, block_size, head_dim]
    int kv_head_stride = block_size * head_dim;
    int kv_type_stride = num_kv_heads * kv_head_stride;
    int layer_stride = 2 * kv_type_stride;
    int block_total = num_layers * layer_stride;

    int k_offset = block_id * block_total
                 + layer_idx * layer_stride
                 + h * kv_head_stride
                 + token_in_block * head_dim + d;

    int v_offset = block_id * block_total
                 + layer_idx * layer_stride
                 + kv_type_stride  // V after K
                 + h * kv_head_stride
                 + token_in_block * head_dim + d;

    kv_cache[k_offset] = k_proj[t * num_kv_heads * head_dim + h * head_dim + d];
    kv_cache[v_offset] = v_proj[t * num_kv_heads * head_dim + h * head_dim + d];
}
"""

# Batched KV cache write — M sequences, 1 token each, single kernel launch.
# Replaces M separate kv_cache_write launches per layer.
BATCHED_KV_CACHE_WRITE_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void batched_kv_cache_write(
    half* __restrict__ kv_cache,       // Contiguous GPU slab
    const half* __restrict__ k_proj,   // [M, num_kv_heads * head_dim]
    const half* __restrict__ v_proj,   // [M, num_kv_heads * head_dim]
    const int* __restrict__ block_table, // [M, max_blocks_per_seq] padded
    const int* __restrict__ pos_buf,   // [M] positions
    int M,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq,
    int num_layers,
    int layer_idx
) {
    // Grid: ceil(num_kv_heads * head_dim / 256), M
    // Each block handles one sequence's KV write
    int seq_idx = blockIdx.y;
    if (seq_idx >= M) return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kv_dim = num_kv_heads * head_dim;
    if (idx >= kv_dim) return;

    int d = idx % head_dim;
    int h = idx / head_dim;

    int position = pos_buf[seq_idx];
    int block_idx = position / block_size;
    int token_in_block = position % block_size;

    int block_id = block_table[seq_idx * max_blocks_per_seq + block_idx];
    if (block_id < 0) return;

    // Block layout: [num_layers, 2(K+V), num_kv_heads, block_size, head_dim]
    int kv_head_stride = block_size * head_dim;
    int kv_type_stride = num_kv_heads * kv_head_stride;
    int layer_stride = 2 * kv_type_stride;
    int block_total = num_layers * layer_stride;

    int k_offset = block_id * block_total
                 + layer_idx * layer_stride
                 + h * kv_head_stride
                 + token_in_block * head_dim + d;

    int v_offset = block_id * block_total
                 + layer_idx * layer_stride
                 + kv_type_stride
                 + h * kv_head_stride
                 + token_in_block * head_dim + d;

    int inp_idx = seq_idx * kv_dim + idx;
    kv_cache[k_offset] = k_proj[inp_idx];
    kv_cache[v_offset] = v_proj[inp_idx];
}
"""

# Tiled FP16 matmul: shared memory tiling for ~10-20x speedup
# Tile size 32x32, cooperative loading of A and B tiles
TILED_FP16_MATMUL_CUDA = CUDA_HEADER + r"""
#define TILE 32
extern "C" __global__ void tiled_fp16_matmul(
    half* __restrict__ out,       // [M, N]
    const half* __restrict__ A,   // [M, K]
    const half* __restrict__ B,   // [N, K] row-major (transposed)
    int M, int N, int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x;  // N dimension
    int by = blockIdx.y;  // M dimension
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float acc = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        // Load A tile: A[row, t*TILE + tx]
        int a_col = t * TILE + tx;
        if (row < M && a_col < K)
            As[ty][tx] = __half2float(A[row * K + a_col]);
        else
            As[ty][tx] = 0.0f;

        // Load B tile: B[col, t*TILE + ty] (B is [N, K])
        int b_col = t * TILE + ty;
        if (col < N && b_col < K)
            Bs[ty][tx] = __half2float(B[col * K + b_col]);
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        acc = fmaxf(-65504.0f, fminf(65504.0f, acc));
        out[row * N + col] = __float2half(acc);
    }
}
"""

# Tiled dequant INT4 matmul: shared memory tiling + dequantization
TILED_DEQUANT_MATMUL_INT4_CUDA = CUDA_HEADER + r"""
#define TILE 32
extern "C" __global__ void tiled_dequant_matmul_int4(
    half* __restrict__ out,         // [M, N]
    const unsigned char* __restrict__ weight, // [N, K/2] packed INT4
    const half* __restrict__ scales,  // [N, K/group_size]
    const half* __restrict__ zeros,   // [N, K/group_size]
    const half* __restrict__ input,   // [M, K]
    int M, int N, int K,
    int group_size
) {
    __shared__ float As[TILE][TILE];   // Input tile
    __shared__ float Bs[TILE][TILE];   // Dequantized weight tile

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;  // M dim
    int col = bx * TILE + tx;  // N dim

    float acc = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    int num_groups = (K + group_size - 1) / group_size;

    for (int t = 0; t < numTiles; t++) {
        // Load input tile A[row, t*TILE + tx]
        int a_col = t * TILE + tx;
        if (row < M && a_col < K)
            As[ty][tx] = __half2float(input[row * K + a_col]);
        else
            As[ty][tx] = 0.0f;

        // Load + dequant weight tile: weight[col, t*TILE + ty]
        int w_k = t * TILE + ty;
        if (col < N && w_k < K) {
            int group_idx = w_k / group_size;
            float scale = __half2float(scales[col * num_groups + group_idx]);
            float zero = __half2float(zeros[col * num_groups + group_idx]);

            int byte_idx = (col * K + w_k) / 2;
            unsigned char packed = weight[byte_idx];
            int nibble = (w_k % 2 == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
            Bs[ty][tx] = (float)nibble * scale + zero;
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        // Clamp to fp16 range to prevent Inf → NaN propagation in deeper layers
        acc = fmaxf(-65504.0f, fminf(65504.0f, acc));
        out[row * N + col] = __float2half(acc);
    }
}
"""

# Tiled dequant INT8 matmul with shared memory tiling
TILED_DEQUANT_MATMUL_INT8_CUDA = CUDA_HEADER + r"""
#define TILE 32
extern "C" __global__ void tiled_dequant_matmul_int8(
    half* __restrict__ out,         // [M, N]
    const signed char* __restrict__ weight, // [N, K] INT8
    const half* __restrict__ scales,  // [N, K/group_size]
    const half* __restrict__ input,   // [M, K]
    int M, int N, int K,
    int group_size
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float acc = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;
    int num_groups = (K + group_size - 1) / group_size;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE + tx;
        if (row < M && a_col < K)
            As[ty][tx] = __half2float(input[row * K + a_col]);
        else
            As[ty][tx] = 0.0f;

        int w_k = t * TILE + ty;
        if (col < N && w_k < K) {
            int group_idx = w_k / group_size;
            float scale = __half2float(scales[col * num_groups + group_idx]);
            Bs[ty][tx] = (float)weight[col * K + w_k] * scale;
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        acc = fmaxf(-65504.0f, fminf(65504.0f, acc));
        out[row * N + col] = __float2half(acc);
    }
}
"""

# Fused residual add + RMSNorm: saves one global memory round-trip
FUSED_RESIDUAL_RMSNORM_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void fused_residual_rmsnorm(
    half* __restrict__ out,           // [rows, hidden_size] — normalized output
    half* __restrict__ residual_out,  // [rows, hidden_size] — updated residual (a + b)
    const half* __restrict__ a,       // [rows, hidden_size]
    const half* __restrict__ b,       // [rows, hidden_size]
    const half* __restrict__ weight,  // [hidden_size]
    int hidden_size,
    float eps
) {
    // One block per row: compute residual = a + b, then RMSNorm(residual)
    int row = blockIdx.x;
    int tid = threadIdx.x;

    half* res_row = residual_out + row * hidden_size;
    const half* a_row = a + row * hidden_size;
    const half* b_row = b + row * hidden_size;
    half* o = out + row * hidden_size;

    // Phase 1: Compute residual and sum of squares in one pass
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(a_row[i]) + __half2float(b_row[i]);
        val = fmaxf(-65504.0f, fminf(65504.0f, val));
        res_row[i] = __float2half(val);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    __shared__ float warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float rms_shared;
    if (tid == 0) {
        rms_shared = rsqrtf(sum_sq / (float)hidden_size + eps);
    }
    __syncthreads();
    float rms = rms_shared;

    // Phase 2: Apply normalization + weight
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(res_row[i]);
        float w = __half2float(weight[i]);
        o[i] = __float2half(val * rms * w);
    }
}
"""

# Vectorized residual add using half2 (2x throughput)
RESIDUAL_ADD_VEC_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void residual_add_vec(
    half* __restrict__ out,
    const half* __restrict__ a,
    const half* __restrict__ b,
    int n
) {
    // 2 elements per thread for throughput
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 2;
    if (base < n) {
        float va = __half2float(a[base]) + __half2float(b[base]);
        out[base] = __float2half(va);
    }
    if (base + 1 < n) {
        float vb = __half2float(a[base + 1]) + __half2float(b[base + 1]);
        out[base + 1] = __float2half(vb);
    }
}
"""

# Vectorized SiLU * up using half2
SILU_MUL_VEC_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void silu_mul_vec(
    half* __restrict__ out,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Process 2 elements per thread
    int base = idx * 2;
    if (base < n) {
        float g0 = __half2float(gate[base]);
        float u0 = __half2float(up[base]);
        float silu0 = g0 / (1.0f + expf(-g0));
        out[base] = __float2half(silu0 * u0);
    }
    if (base + 1 < n) {
        float g1 = __half2float(gate[base + 1]);
        float u1 = __half2float(up[base + 1]);
        float silu1 = g1 / (1.0f + expf(-g1));
        out[base + 1] = __float2half(silu1 * u1);
    }
}
"""

# Batched RoPE: handles M sequences with different positions in one launch
BATCHED_ROTARY_EMBEDDING_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void batched_rotary_embedding(
    half* __restrict__ q,           // [M, num_heads, head_dim]
    half* __restrict__ k,           // [M, num_kv_heads, head_dim]
    const int* __restrict__ positions, // [M] — position per sequence
    int M,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta_base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_q = M * num_heads * half_dim;
    int total_k = M * num_kv_heads * half_dim;

    if (idx < total_q) {
        int pair = idx % half_dim;
        int head = (idx / half_dim) % num_heads;
        int seq = idx / (half_dim * num_heads);

        int pos = positions[seq];
        float freq = 1.0f / powf(theta_base, (float)(2 * pair) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int base = seq * num_heads * head_dim + head * head_dim;
        float q0 = __half2float(q[base + pair]);
        float q1 = __half2float(q[base + pair + half_dim]);
        q[base + pair]            = __float2half(q0 * cos_val - q1 * sin_val);
        q[base + pair + half_dim] = __float2half(q1 * cos_val + q0 * sin_val);
    }

    if (idx < total_k) {
        int pair = idx % half_dim;
        int head = (idx / half_dim) % num_kv_heads;
        int seq = idx / (half_dim * num_kv_heads);

        int pos = positions[seq];
        float freq = 1.0f / powf(theta_base, (float)(2 * pair) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int base = seq * num_kv_heads * head_dim + head * head_dim;
        float k0 = __half2float(k[base + pair]);
        float k1 = __half2float(k[base + pair + half_dim]);
        k[base + pair]            = __float2half(k0 * cos_val - k1 * sin_val);
        k[base + pair + half_dim] = __float2half(k1 * cos_val + k0 * sin_val);
    }
}
"""


# ============================================================================
# FP32 Residual Stream Kernels
# ============================================================================

# RMSNorm with fp32 input (residual stream) → fp16 output (for matmul)
RMSNORM_F32IN_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void rmsnorm_f32in(
    half* __restrict__ out,
    const float* __restrict__ input,
    const half* __restrict__ weight,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* x = input + row * hidden_size;
    half* o = out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    __shared__ float warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float rms_shared;
    if (tid == 0) rms_shared = rsqrtf(sum_sq / (float)hidden_size + eps);
    __syncthreads();
    float rms = rms_shared;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[i];
        float w = __half2float(weight[i]);
        o[i] = __float2half(val * rms * w);
    }
}
"""

# Embedding lookup → fp32 output
EMBEDDING_LOOKUP_F32OUT_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void embedding_lookup_f32out(
    float* __restrict__ out,
    const half* __restrict__ table,
    const int* __restrict__ token_ids,
    int hidden_size, int num_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_size;
    if (idx >= total) return;

    int token_pos = idx / hidden_size;
    int dim = idx % hidden_size;
    int token_id = token_ids[token_pos];

    out[token_pos * hidden_size + dim] = __half2float(table[token_id * hidden_size + dim]);
}
"""

# Residual add: fp32 out = fp32 a + fp16 b
RESIDUAL_ADD_F32_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void residual_add_f32(
    float* __restrict__ out,
    const float* __restrict__ a,
    const half* __restrict__ b,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + __half2float(b[idx]);
    }
}
"""

# Fused: fp32 residual = fp16 a + fp32 b; fp16 norm_out = RMSNorm(residual)
FUSED_RESIDUAL_RMSNORM_F32_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void fused_residual_rmsnorm_f32(
    half* __restrict__ norm_out,
    float* __restrict__ residual_out,
    const half* __restrict__ a,
    const float* __restrict__ b,
    const half* __restrict__ weight,
    int hidden_size,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float* res_row = residual_out + row * hidden_size;
    const half* a_row = a + row * hidden_size;
    const float* b_row = b + row * hidden_size;
    half* o = norm_out + row * hidden_size;

    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(a_row[i]) + b_row[i];
        res_row[i] = val;
        sum_sq += val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    __shared__ float warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float rms_shared;
    if (tid == 0) rms_shared = rsqrtf(sum_sq / (float)hidden_size + eps);
    __syncthreads();
    float rms = rms_shared;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = res_row[i];
        float w = __half2float(weight[i]);
        o[i] = __float2half(val * rms * w);
    }
}
"""


# LoRA scaled add: out += scaling * delta
LORA_SCALED_ADD_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void lora_scaled_add(
    half* __restrict__ out,
    const half* __restrict__ delta,
    int n,
    float scaling
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(out[idx]) + scaling * __half2float(delta[idx]);
        out[idx] = __float2half(val);
    }
}
"""


# ============================================================================
# GEMV Kernels — Optimized for M=1 decode (memory-bandwidth bound)
# ============================================================================

# INT4 dequant GEMV: y[N] = dequant(W[N,K/2]) @ x[K]
#
# Optimized for A100 memory bandwidth saturation:
# - 8 rows per block (1 warp per row), no cross-warp sync needed
# - Vectorized uint32 loads (8 INT4 values per load)
# - Scale/zero cached in registers per group (avoid redundant global reads)
# - Input vector cached in L1/L2 (~7KB for hidden_size=3584)
DEQUANT_GEMV_INT4_CUDA = CUDA_HEADER + r"""
#define GEMV_RPB 8
extern "C" __global__ void dequant_gemv_int4(
    half* __restrict__ out,             // [N]
    const unsigned char* __restrict__ weight, // [N, K/2] packed INT4
    const half* __restrict__ scales,    // [N, K/group_size]
    const half* __restrict__ zeros,     // [N, K/group_size]
    const half* __restrict__ input,     // [K]
    int N, int K,
    int group_size
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * GEMV_RPB + warp_id;

    if (row >= N) return;

    int num_groups = (K + group_size - 1) / group_size;
    int half_K = K / 2;
    const unsigned char* w_row = weight + (long long)row * half_K;
    const unsigned int* w_row4 = (const unsigned int*)w_row;
    int num_uint32 = half_K / 4;

    // Precompute scale/zero row offset
    int scale_row_off = row * num_groups;

    float acc = 0.0f;
    int prev_g = -1;
    float s = 0.0f, z = 0.0f;

    for (int i = lane_id; i < num_uint32; i += 32) {
        unsigned int packed4 = w_row4[i];
        int k_base = i * 8;

        // Cache scale/zero per group — changes every group_size/8 iterations
        int g = k_base / group_size;
        if (g != prev_g) {
            s = __half2float(scales[scale_row_off + g]);
            z = __half2float(zeros[scale_row_off + g]);
            prev_g = g;
        }

        // Unpack 8 nibbles
        float w0 = (float)( packed4        & 0xF) * s + z;
        float w1 = (float)((packed4 >>  4) & 0xF) * s + z;
        float w2 = (float)((packed4 >>  8) & 0xF) * s + z;
        float w3 = (float)((packed4 >> 12) & 0xF) * s + z;
        float w4 = (float)((packed4 >> 16) & 0xF) * s + z;
        float w5 = (float)((packed4 >> 20) & 0xF) * s + z;
        float w6 = (float)((packed4 >> 24) & 0xF) * s + z;
        float w7 = (float)((packed4 >> 28) & 0xF) * s + z;

        acc += w0 * __half2float(input[k_base    ]);
        acc += w1 * __half2float(input[k_base + 1]);
        acc += w2 * __half2float(input[k_base + 2]);
        acc += w3 * __half2float(input[k_base + 3]);
        acc += w4 * __half2float(input[k_base + 4]);
        acc += w5 * __half2float(input[k_base + 5]);
        acc += w6 * __half2float(input[k_base + 6]);
        acc += w7 * __half2float(input[k_base + 7]);
    }

    // Handle remaining bytes
    int remainder_start = num_uint32 * 4;
    for (int byte_idx = remainder_start + lane_id; byte_idx < half_K; byte_idx += 32) {
        unsigned char packed = w_row[byte_idx];
        int k0 = byte_idx * 2;
        int g0 = k0 / group_size;
        float s0 = __half2float(scales[scale_row_off + g0]);
        float z0 = __half2float(zeros[scale_row_off + g0]);
        acc += ((float)(packed & 0xF) * s0 + z0) * __half2float(input[k0]);
        acc += ((float)((packed >> 4) & 0xF) * s0 + z0) * __half2float(input[k0 + 1]);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, offset);

    if (lane_id == 0) {
        acc = fmaxf(-65504.0f, fminf(65504.0f, acc));
        out[row] = __float2half(acc);
    }
}
"""

# Batched INT4 GEMV: out[M,N] = W[N,K] @ inp[M,K]^T for small M (2-8)
# Each block handles 8 output rows across ALL M input vectors.
# Reads weight row ONCE, computes M dot products simultaneously.
# Key optimization: weight bandwidth is amortized across M sequences.
BATCHED_DEQUANT_GEMV_INT4_CUDA = CUDA_HEADER + r"""
#define BGEMV_RPB 8
#define BGEMV_MAX_M 8
extern "C" __global__ void batched_dequant_gemv_int4(
    half* __restrict__ out,             // [M, N] row-major
    const unsigned char* __restrict__ weight, // [N, K/2] packed INT4
    const half* __restrict__ scales,    // [N, K/group_size]
    const half* __restrict__ zeros,     // [N, K/group_size]
    const half* __restrict__ input,     // [M, K] row-major
    int M, int N, int K,
    int group_size
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * BGEMV_RPB + warp_id;

    if (row >= N) return;

    int num_groups = (K + group_size - 1) / group_size;
    int half_K = K / 2;
    const unsigned char* w_row = weight + (long long)row * half_K;
    const unsigned int* w_row4 = (const unsigned int*)w_row;
    int num_uint32 = half_K / 4;
    int scale_row_off = row * num_groups;

    // Accumulators for each input row (M <= 8)
    float acc[BGEMV_MAX_M];
    for (int m = 0; m < M && m < BGEMV_MAX_M; m++) acc[m] = 0.0f;

    int prev_g = -1;
    float s = 0.0f, z = 0.0f;

    for (int i = lane_id; i < num_uint32; i += 32) {
        unsigned int packed4 = w_row4[i];
        int k_base = i * 8;

        int g = k_base / group_size;
        if (g != prev_g) {
            s = __half2float(scales[scale_row_off + g]);
            z = __half2float(zeros[scale_row_off + g]);
            prev_g = g;
        }

        // Dequantize 8 weights
        float w0 = (float)( packed4        & 0xF) * s + z;
        float w1 = (float)((packed4 >>  4) & 0xF) * s + z;
        float w2 = (float)((packed4 >>  8) & 0xF) * s + z;
        float w3 = (float)((packed4 >> 12) & 0xF) * s + z;
        float w4 = (float)((packed4 >> 16) & 0xF) * s + z;
        float w5 = (float)((packed4 >> 20) & 0xF) * s + z;
        float w6 = (float)((packed4 >> 24) & 0xF) * s + z;
        float w7 = (float)((packed4 >> 28) & 0xF) * s + z;

        // Compute dot product against each of M input vectors
        for (int m = 0; m < M && m < BGEMV_MAX_M; m++) {
            const half* inp_m = input + m * K;
            acc[m] += w0 * __half2float(inp_m[k_base    ]);
            acc[m] += w1 * __half2float(inp_m[k_base + 1]);
            acc[m] += w2 * __half2float(inp_m[k_base + 2]);
            acc[m] += w3 * __half2float(inp_m[k_base + 3]);
            acc[m] += w4 * __half2float(inp_m[k_base + 4]);
            acc[m] += w5 * __half2float(inp_m[k_base + 5]);
            acc[m] += w6 * __half2float(inp_m[k_base + 6]);
            acc[m] += w7 * __half2float(inp_m[k_base + 7]);
        }
    }

    // Warp reduction for each M accumulator
    for (int m = 0; m < M && m < BGEMV_MAX_M; m++) {
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);

        if (lane_id == 0) {
            float val = fmaxf(-65504.0f, fminf(65504.0f, acc[m]));
            out[m * N + row] = __float2half(val);
        }
    }
}
"""

# FP16 GEMV: y[N] = W[N,K] @ x[K]  (W is [N,K] row-major)
# Same warp-per-row design as INT4 GEMV: 8 rows per block, 32 threads per row.
FP16_GEMV_CUDA = CUDA_HEADER + r"""
#define FP16_GEMV_RPB 8
extern "C" __global__ void fp16_gemv(
    half* __restrict__ out,         // [N]
    const half* __restrict__ weight, // [N, K] row-major
    const half* __restrict__ input,  // [K]
    int N, int K
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x * FP16_GEMV_RPB + warp_id;

    if (row >= N) return;

    const half* w_row = weight + (long long)row * K;

    float acc = 0.0f;
    for (int k = lane_id; k < K; k += 32) {
        acc += __half2float(w_row[k]) * __half2float(input[k]);
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, offset);

    if (lane_id == 0) {
        acc = fmaxf(-65504.0f, fminf(65504.0f, acc));
        out[row] = __float2half(acc);
    }
}
"""


# Fused Gate+Up GEMV: computes both gate[N] and up[N] in one kernel.
# Gate and Up projections read the same input and have the same N=intermediate_size.
# Fusing saves one full pass reading the input vector + kernel launch overhead.
# Each warp computes one output row from EITHER gate or up (decided by warp assignment).
FUSED_GATE_UP_GEMV_INT4_CUDA = CUDA_HEADER + r"""
#define FGU_RPB 8
extern "C" __global__ void fused_gate_up_gemv_int4(
    half* __restrict__ gate_out,        // [N]
    half* __restrict__ up_out,          // [N]
    const unsigned char* __restrict__ gate_weight, // [N, K/2]
    const half* __restrict__ gate_scales,
    const half* __restrict__ gate_zeros,
    const unsigned char* __restrict__ up_weight,   // [N, K/2]
    const half* __restrict__ up_scales,
    const half* __restrict__ up_zeros,
    const half* __restrict__ input,     // [K]
    int N, int K,
    int group_size
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    // First 4 warps do gate rows, last 4 warps do up rows
    int is_up = warp_id >= 4;
    int local_warp = warp_id & 3;  // 0-3 within gate/up
    int row = blockIdx.x * 4 + local_warp;

    if (row >= N) return;

    int num_groups = (K + group_size - 1) / group_size;
    int half_K = K / 2;

    const unsigned char* w_row;
    const half* sc;
    const half* zr;
    if (is_up) {
        w_row = up_weight + (long long)row * half_K;
        sc = up_scales;
        zr = up_zeros;
    } else {
        w_row = gate_weight + (long long)row * half_K;
        sc = gate_scales;
        zr = gate_zeros;
    }

    const unsigned int* w_row4 = (const unsigned int*)w_row;
    int num_uint32 = half_K / 4;
    int scale_row_off = row * num_groups;

    float acc = 0.0f;
    int prev_g = -1;
    float s = 0.0f, z = 0.0f;

    for (int i = lane_id; i < num_uint32; i += 32) {
        unsigned int packed4 = w_row4[i];
        int k_base = i * 8;

        int g = k_base / group_size;
        if (g != prev_g) {
            s = __half2float(sc[scale_row_off + g]);
            z = __half2float(zr[scale_row_off + g]);
            prev_g = g;
        }

        float w0 = (float)( packed4        & 0xF) * s + z;
        float w1 = (float)((packed4 >>  4) & 0xF) * s + z;
        float w2 = (float)((packed4 >>  8) & 0xF) * s + z;
        float w3 = (float)((packed4 >> 12) & 0xF) * s + z;
        float w4 = (float)((packed4 >> 16) & 0xF) * s + z;
        float w5 = (float)((packed4 >> 20) & 0xF) * s + z;
        float w6 = (float)((packed4 >> 24) & 0xF) * s + z;
        float w7 = (float)((packed4 >> 28) & 0xF) * s + z;

        acc += w0 * __half2float(input[k_base    ]);
        acc += w1 * __half2float(input[k_base + 1]);
        acc += w2 * __half2float(input[k_base + 2]);
        acc += w3 * __half2float(input[k_base + 3]);
        acc += w4 * __half2float(input[k_base + 4]);
        acc += w5 * __half2float(input[k_base + 5]);
        acc += w6 * __half2float(input[k_base + 6]);
        acc += w7 * __half2float(input[k_base + 7]);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, offset);

    if (lane_id == 0) {
        acc = fmaxf(-65504.0f, fminf(65504.0f, acc));
        if (is_up)
            up_out[row] = __float2half(acc);
        else
            gate_out[row] = __float2half(acc);
    }
}
"""

# Batched fused gate+up GEMV for MLP: reads input ONCE, computes both gate and up
# For M=2-8 sequences. Reads each weight row once across M input vectors.
# Grid: ceil(N/4), M; Block: 256 (8 warps, 4 gate + 4 up per block)
BATCHED_FUSED_GATE_UP_GEMV_INT4_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void batched_fused_gate_up_gemv_int4(
    half* __restrict__ gate_out,        // [M, N]
    half* __restrict__ up_out,          // [M, N]
    const unsigned char* __restrict__ gate_weight, // [N, K/2]
    const half* __restrict__ gate_scales,
    const half* __restrict__ gate_zeros,
    const unsigned char* __restrict__ up_weight,   // [N, K/2]
    const half* __restrict__ up_scales,
    const half* __restrict__ up_zeros,
    const half* __restrict__ input,     // [M, K]
    int M, int N, int K,
    int group_size
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int seq_idx = blockIdx.y;
    if (seq_idx >= M) return;

    // First 4 warps do gate rows, last 4 do up rows
    int is_up = warp_id >= 4;
    int local_warp = warp_id & 3;
    int row = blockIdx.x * 4 + local_warp;

    if (row >= N) return;

    int num_groups = (K + group_size - 1) / group_size;
    int half_K = K / 2;

    const unsigned char* w_row;
    const half* sc;
    const half* zr;
    if (is_up) {
        w_row = up_weight + (long long)row * half_K;
        sc = up_scales;
        zr = up_zeros;
    } else {
        w_row = gate_weight + (long long)row * half_K;
        sc = gate_scales;
        zr = gate_zeros;
    }

    const unsigned int* w_row4 = (const unsigned int*)w_row;
    int num_uint32 = half_K / 4;
    int scale_row_off = row * num_groups;
    const half* inp_m = input + seq_idx * K;

    float acc = 0.0f;
    int prev_g = -1;
    float s = 0.0f, z = 0.0f;

    for (int i = lane_id; i < num_uint32; i += 32) {
        unsigned int packed4 = w_row4[i];
        int k_base = i * 8;

        int g = k_base / group_size;
        if (g != prev_g) {
            s = __half2float(sc[scale_row_off + g]);
            z = __half2float(zr[scale_row_off + g]);
            prev_g = g;
        }

        float w0 = (float)( packed4        & 0xF) * s + z;
        float w1 = (float)((packed4 >>  4) & 0xF) * s + z;
        float w2 = (float)((packed4 >>  8) & 0xF) * s + z;
        float w3 = (float)((packed4 >> 12) & 0xF) * s + z;
        float w4 = (float)((packed4 >> 16) & 0xF) * s + z;
        float w5 = (float)((packed4 >> 20) & 0xF) * s + z;
        float w6 = (float)((packed4 >> 24) & 0xF) * s + z;
        float w7 = (float)((packed4 >> 28) & 0xF) * s + z;

        acc += w0 * __half2float(inp_m[k_base    ]);
        acc += w1 * __half2float(inp_m[k_base + 1]);
        acc += w2 * __half2float(inp_m[k_base + 2]);
        acc += w3 * __half2float(inp_m[k_base + 3]);
        acc += w4 * __half2float(inp_m[k_base + 4]);
        acc += w5 * __half2float(inp_m[k_base + 5]);
        acc += w6 * __half2float(inp_m[k_base + 6]);
        acc += w7 * __half2float(inp_m[k_base + 7]);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, offset);

    if (lane_id == 0) {
        acc = fmaxf(-65504.0f, fminf(65504.0f, acc));
        if (is_up)
            up_out[seq_idx * N + row] = __float2half(acc);
        else
            gate_out[seq_idx * N + row] = __float2half(acc);
    }
}
"""


# GPU Argmax kernel: finds argmax of fp16 array on GPU.
# Avoids downloading 300KB of logits for greedy decoding.
# Single block, 256 threads. Each thread scans vocab/256 elements,
# then tree reduction in shared memory.
ARGMAX_FP16_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void argmax_fp16(
    int* __restrict__ out_idx,      // [1] output: argmax index
    const half* __restrict__ data,  // [N] fp16 values
    int N
) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Each thread finds local max
    float best_val = -1e30f;
    int best_idx = 0;
    for (int i = tid; i < N; i += nthreads) {
        float v = __half2float(data[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // Shared memory reduction
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    // Tree reduction
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[0] = s_idx[0];
    }
}
"""

# Batched argmax: Grid=(M,), Block=(256). Each block handles one row.
BATCHED_ARGMAX_FP16_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void batched_argmax_fp16(
    int* __restrict__ out_idx,      // [M] output: argmax indices
    const half* __restrict__ data,  // [M, N] fp16 values
    int M, int N
) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const half* row_data = data + row * N;

    // Each thread finds local max
    float best_val = -1e30f;
    int best_idx = 0;
    for (int i = tid; i < N; i += nthreads) {
        float v = __half2float(row_data[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // Shared memory reduction
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_idx[row] = s_idx[0];
    }
}
"""
# Computes: out[M, N] = input[M, K] × weight[N, K]^T with INT4 dequantization
#
# Uses NVIDIA tensor cores via WMMA API:
# - Fragment size: 16×16×16 (FP16 inputs, FP32 accumulator)
# - Dequantizes INT4 weights to FP16 in shared memory
# - Input padded to M=16 for small batch sizes (M=2-8)
#
# Block tile: [16, 64] — 4 warps, each handles one 16×16 output column chunk
# Each warp accumulates over K in steps of 16 using wmma::mma_sync
#
# Weight layout: [N, K/2] packed INT4 (low nibble = even k, high nibble = odd k)
# Scales: [N, K/group_size] FP16
# Zeros: [N, K/group_size] FP16

WMMA_HEADER = r"""
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
"""

# Weight repack: row-major [N, K/2] → tiled [num_n_tiles, num_k_tiles, 64, 8]
# Each tile is 64 weight rows × 8 bytes (16 INT4 values per row) = 512 bytes
# This makes WMMA weight loads coalesced (128 threads × 4 bytes = 512 bytes)
REPACK_INT4_TILED_CUDA = CUDA_HEADER + r"""
extern "C" __global__ void repack_int4_tiled(
    unsigned char* __restrict__ dst,        // tiled: [num_n_tiles * num_k_tiles * 512]
    const unsigned char* __restrict__ src,  // row-major: [N, K/2]
    int N, int half_K,                      // half_K = K/2
    int num_k_tiles                         // = ceil(K / 16)
) {
    // Grid: (num_k_tiles, num_n_tiles), Block: (256)
    // Each block repacks one [64, 8] tile (512 bytes)
    int k_tile = blockIdx.x;
    int n_tile = blockIdx.y;

    int tile_offset = ((long long)n_tile * num_k_tiles + k_tile) * 512;

    // 256 threads, 512 bytes → 2 bytes per thread
    for (int i = threadIdx.x; i < 512; i += 256) {
        int n_local = i / 8;      // 0-63 within tile
        int byte_idx = i % 8;     // 0-7 within row's K-chunk

        int n_global = n_tile * 64 + n_local;
        int k_byte = k_tile * 8 + byte_idx;  // byte offset in K/2 dim

        unsigned char val = 0;
        if (n_global < N && k_byte < half_K) {
            val = src[(long long)n_global * half_K + k_byte];
        }
        dst[tile_offset + i] = val;
    }
}
"""

WMMA_DEQUANT_GEMM_INT4_CUDA = WMMA_HEADER + r"""
// WMMA INT4 GEMM with TILED weight layout for coalesced global reads.
//
// Weight tiles: [num_n_tiles, num_k_tiles, 64, 8] packed INT4
// Each [64, 8] tile = 64 rows × 16 INT4 values = 512 bytes, contiguous.
//
// Block tile: [16, 64], 4 warps each handling [16, 16] output
// smem_A: [16][16] input, row-major (WMMA row_major load)
// smem_B: [16][64] weight, column-major layout: smem_B[k][n_local]
//         WMMA col_major load reads B[k][j] for warp's 16 columns

#define WMMA_M 16
#define WMMA_N 64
#define WMMA_K 16

extern "C" __global__ void wmma_dequant_gemm_int4(
    half* __restrict__ out,
    const unsigned char* __restrict__ weight_tiled,
    const half* __restrict__ scales,
    const half* __restrict__ zeros,
    const half* __restrict__ input,
    int M, int N, int K,
    int group_size,
    int num_k_tiles
) {
    int n_start = blockIdx.x * WMMA_N;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // smem_A: [16][16] row-major, padded to avoid bank conflicts
    // smem_B: [16][72] column-major: smem_B[k][n] with stride 72 (64+8 pad)
    __shared__ half smem_A[WMMA_M * (WMMA_K + 8)];  // 16 * 24 = 384 half
    __shared__ half smem_B[WMMA_K * (WMMA_N + 8)];   // 16 * 72 = 1152 half
    __shared__ float smem_store[16 * 16];

    const int A_stride = WMMA_K + 8;   // 24
    const int B_stride = WMMA_N + 8;   // 72

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    int num_groups = (K + group_size - 1) / group_size;
    int n_tile_idx = n_start / 64;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k_start = kt * WMMA_K;

        // ===== Load input → smem_A[row][col], row-major =====
        // 128 threads, 16*16=256 elements, 2 per thread
        {
            int idx = threadIdx.x;
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;
            if (r < 16) {
                int gk = k_start + c;
                smem_A[r * A_stride + c] = (r < M && gk < K) ? input[r * K + gk] : __float2half(0.0f);
            }
            idx += 128;
            r = idx / WMMA_K;
            c = idx % WMMA_K;
            if (r < 16) {
                int gk = k_start + c;
                smem_A[r * A_stride + c] = (r < M && gk < K) ? input[r * K + gk] : __float2half(0.0f);
            }
        }

        // ===== Load + dequant weight → smem_B[k][n], column-major =====
        // Tiled weight: 512 contiguous bytes for this (n_tile, k_tile)
        // Layout in tile: [64_n][8_bytes] = [64_n][16_INT4]
        // We need to dequant and store as smem_B[k][n] (column-major)
        //
        // 128 threads, 64*16=1024 FP16 values to write, 8 per thread
        {
            long long tile_base = ((long long)n_tile_idx * num_k_tiles + kt) * 512;
            // Each thread: my_idx in [0,127]
            // n_local = my_idx / 2 (0-63), chunk = my_idx % 2 (0 or 1)
            // chunk 0: K positions [0:8], chunk 1: K positions [8:16]
            int n_local = threadIdx.x / 2;
            int chunk = threadIdx.x % 2;
            int n_global = n_start + n_local;

            // Read 4 packed bytes = 8 INT4 values (COALESCED from tiled layout!)
            const unsigned int* tile_ptr = (const unsigned int*)(weight_tiled + tile_base);
            unsigned int packed = tile_ptr[threadIdx.x];

            // Dequant using scales/zeros (row-major access)
            int k_col = k_start + chunk * 8;
            int g = k_col / group_size;
            if (g >= num_groups) g = num_groups - 1;
            float s = 0.0f, z = 0.0f;
            if (n_global < N) {
                s = __half2float(scales[n_global * num_groups + g]);
                z = __half2float(zeros[n_global * num_groups + g]);
            }

            // Dequant 8 INT4 → 8 FP16, store COLUMN-MAJOR: smem_B[k][n_local]
            int k_base = chunk * 8;
            smem_B[(k_base    ) * B_stride + n_local] = __float2half((float)( packed        & 0xF) * s + z);
            smem_B[(k_base + 1) * B_stride + n_local] = __float2half((float)((packed >>  4) & 0xF) * s + z);
            smem_B[(k_base + 2) * B_stride + n_local] = __float2half((float)((packed >>  8) & 0xF) * s + z);
            smem_B[(k_base + 3) * B_stride + n_local] = __float2half((float)((packed >> 12) & 0xF) * s + z);
            smem_B[(k_base + 4) * B_stride + n_local] = __float2half((float)((packed >> 16) & 0xF) * s + z);
            smem_B[(k_base + 5) * B_stride + n_local] = __float2half((float)((packed >> 20) & 0xF) * s + z);
            smem_B[(k_base + 6) * B_stride + n_local] = __float2half((float)((packed >> 24) & 0xF) * s + z);
            smem_B[(k_base + 7) * B_stride + n_local] = __float2half((float)((packed >> 28) & 0xF) * s + z);
        }

        __syncthreads();

        // ===== WMMA tensor core compute =====
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;

            // A: smem_A[0:16, 0:16] row-major, stride=A_stride
            wmma::load_matrix_sync(frag_a, smem_A, A_stride);

            // B: smem_B[0:16, warp_id*16 : warp_id*16+16] col-major, stride=B_stride
            // smem_B is [k][n] with stride B_stride, col_major load reads columns
            wmma::load_matrix_sync(frag_b, smem_B + warp_id * 16, B_stride);

            wmma::mma_sync(acc, frag_a, frag_b, acc);
        }

        __syncthreads();
    }

    // ===== Store results =====
    int n_warp = n_start + warp_id * 16;
    for (int w = 0; w < 4; w++) {
        if (warp_id == w && n_warp < N) {
            wmma::store_matrix_sync(smem_store, acc, 16, wmma::mem_row_major);
        }
        __syncthreads();
        if (warp_id == w) {
            for (int i = lane_id; i < 16 * 16; i += 32) {
                int r = i / 16;
                int c = i % 16;
                if (r < M && n_warp + c < N) {
                    float val = smem_store[i];
                    val = fmaxf(-65504.0f, fminf(65504.0f, val));
                    out[r * N + n_warp + c] = __float2half(val);
                }
            }
        }
        __syncthreads();
    }
}
"""


# ============================================================================
# Kernel Manager — Compiles and caches all kernels
# ============================================================================

class InferenceKernels:
    """Compiles and manages all inference kernels.

    Usage:
        kernels = InferenceKernels(backend="cuda")
        kernels.compile_all()
        kernels.launch_rmsnorm(out, input, weight, hidden_size, eps)
    """

    KERNEL_SOURCES = {
        # Core kernels (used in current forward pass)
        "rmsnorm": RMSNORM_CUDA,
        "silu_mul": SILU_MUL_CUDA,
        "rotary_embedding": ROTARY_EMBEDDING_CUDA,
        "bias_add": BIAS_ADD_CUDA,
        "paged_attention": PAGED_ATTENTION_CUDA,
        "prefill_attention": PREFILL_ATTENTION_CUDA,
        "kv_cache_write": KV_CACHE_WRITE_CUDA,
        "batched_kv_cache_write": BATCHED_KV_CACHE_WRITE_CUDA,
        # Tiled matmuls (prefill + large M)
        "tiled_fp16_matmul": TILED_FP16_MATMUL_CUDA,
        "tiled_dequant_matmul_int4": TILED_DEQUANT_MATMUL_INT4_CUDA,
        "tiled_dequant_matmul_int8": TILED_DEQUANT_MATMUL_INT8_CUDA,
        # Optimized fused kernels
        "fused_residual_rmsnorm": FUSED_RESIDUAL_RMSNORM_CUDA,
        "batched_rotary_embedding": BATCHED_ROTARY_EMBEDDING_CUDA,
        "lora_scaled_add": LORA_SCALED_ADD_CUDA,
        # FP32 residual stream kernels
        "rmsnorm_f32in": RMSNORM_F32IN_CUDA,
        "embedding_lookup_f32out": EMBEDDING_LOOKUP_F32OUT_CUDA,
        "residual_add_f32": RESIDUAL_ADD_F32_CUDA,
        "fused_residual_rmsnorm_f32": FUSED_RESIDUAL_RMSNORM_F32_CUDA,
        # GEMV kernels (M=1 decode optimization)
        "dequant_gemv_int4": DEQUANT_GEMV_INT4_CUDA,
        "batched_dequant_gemv_int4": BATCHED_DEQUANT_GEMV_INT4_CUDA,
        "fp16_gemv": FP16_GEMV_CUDA,
        "fused_gate_up_gemv_int4": FUSED_GATE_UP_GEMV_INT4_CUDA,
        "argmax_fp16": ARGMAX_FP16_CUDA,
        "batched_argmax_fp16": BATCHED_ARGMAX_FP16_CUDA,
    }

    def __init__(self, backend: str = "cuda"):
        self._backend = backend
        self._compiler = RuntimeCompiler()
        self._launcher = KernelLauncher()
        self._compiled: Dict[str, CompiledKernel] = {}
        self._fast_refs = []  # Keep ctypes refs alive for fast_launch

        # Per-instance kernel source map: starts from class-level CUDA sources,
        # then overlays backend-specific kernels (e.g. ROCm-only MFMA paths).
        self._kernel_sources: Dict[str, str] = dict(self.KERNEL_SOURCES)
        if backend == "rocm" and MFMA_DEQUANT_MATMUL_INT4_V3_HIP is not None:
            # Lazy-compiled (not in essential set) — first launch in model_runner triggers compile.
            self._kernel_sources["mfma_dequant_matmul_int4_v3"] = MFMA_DEQUANT_MATMUL_INT4_V3_HIP
        if backend == "rocm" and BGEMV_INT4_WAVE64_HIP is not None:
            # Native wavefront-64 batched INT4 GEMV — 2.13x over bgemv on MI300X at M=4.
            self._kernel_sources["bgemv_int4_wave64"] = BGEMV_INT4_WAVE64_HIP
        if backend == "rocm":
            # 128-bit (uint4) weight-load variant — targets HBM bandwidth ceiling.
            # Requires K%32==0 AND group_size%32==0.
            self._kernel_sources["bgemv_int4_wave64_v2"] = BGEMV_INT4_WAVE64_V2_HIP
        if backend == "rocm" and FUSED_ROPE_KV_WRITE_HIP is not None:
            # Fused RoPE + KV cache write — collapses 2 launches per layer to 1.
            self._kernel_sources["fused_rope_kv_write"] = FUSED_ROPE_KV_WRITE_HIP
        if backend == "rocm":
            # GQA-aware + online-softmax paged attention (Fix #1 + #2).
            # 5× less KV HBM traffic vs v1 on Qwen2.5-32B (qpkv=5).
            self._kernel_sources["paged_attention_v2"] = PAGED_ATTENTION_V2_HIP

    def compile_all(self, quant_type: str = "int4"):
        """Compile essential inference kernels for fast cold start.

        Only compiles the kernels needed for the model's quant type.
        Non-essential kernels are lazy-compiled on first use via get().
        """
        items = self._get_essential_items(quant_type)
        backend = self._backend

        if backend == "cuda":
            self._compile_all_cuda_parallel(items)
        elif backend == "rocm":
            self._compile_all_rocm_parallel(items)
        else:
            for name, source in items:
                self._compiled[name] = self._compiler.compile(source, name, backend)

    def _get_essential_items(self, quant_type: str = "int4"):
        """Get the essential kernel (name, source) pairs for a quant type."""
        essential = {
            "silu_mul", "rotary_embedding", "batched_rotary_embedding",
            "paged_attention", "prefill_attention",
            "kv_cache_write", "batched_kv_cache_write",
            "argmax_fp16",
            "rmsnorm_f32in", "embedding_lookup_f32out",
            "residual_add_f32", "fused_residual_rmsnorm_f32",
        }
        if quant_type == "int4":
            essential |= {
                "dequant_gemv_int4", "batched_dequant_gemv_int4",
                "fused_gate_up_gemv_int4", "fp16_gemv",
            }
        elif quant_type == "int8":
            essential |= {
                "dequant_matmul_int8", "tiled_dequant_matmul_int8",
                "fp16_matmul", "fp16_gemv",
            }
        else:
            essential |= {"fp16_matmul", "tiled_fp16_matmul", "fp16_gemv"}
        return [(n, s) for n, s in self.KERNEL_SOURCES.items() if n in essential]

    def _detect_arch_string(self) -> str:
        """Detect the current CUDA device's compute capability and return NVRTC arch string.

        Falls back to compute_80 if detection fails (matches the historical default).
        Cached on the instance to avoid repeated driver calls.
        """
        cached = getattr(self, "_cached_arch_string", None)
        if cached is not None:
            return cached
        try:
            import ctypes
            driver = self._compiler._load_cuda_driver()
            if driver is None:
                self._cached_arch_string = "compute_80"
                return self._cached_arch_string
            # Get current device from active context (engine has already initialized it).
            device = ctypes.c_int(0)
            rc = driver.cuCtxGetDevice(ctypes.byref(device))
            if rc != 0:
                # Fall back to device 0.
                device = ctypes.c_int(0)
                driver.cuDeviceGet(ctypes.byref(device), 0)
            from zse_compiler.runtime.compiler import RuntimeCompiler
            arch = RuntimeCompiler._detect_cuda_arch(driver, device)
        except Exception:
            arch = "compute_80"
        self._cached_arch_string = arch
        return arch

    def _compile_ptx_only(self, quant_type: str = "int4"):
        """Compile kernel sources to PTX in parallel (CPU-only, no GPU context needed).

        Returns list of (name, ptx_buf) tuples for loading on main thread.
        """
        import concurrent.futures
        import ctypes

        items = self._get_essential_items(quant_type)
        arch = self._detect_arch_string()
        arch_opt = f"--gpu-architecture={arch}".encode()

        def _compile_to_ptx(name_source):
            name, source = name_source
            from zse_compiler.runtime.compiler import RuntimeCompiler
            compiler = RuntimeCompiler()
            nvrtc = compiler._load_nvrtc()
            if nvrtc is None:
                raise RuntimeError(f"NVRTC not available for kernel '{name}'")
            source_bytes = source.encode("utf-8")
            prog = ctypes.c_void_p()
            status = nvrtc.nvrtcCreateProgram(
                ctypes.byref(prog), source_bytes, b"zse_kernel.cu", 0, None, None,
            )
            if status != 0:
                raise RuntimeError(f"nvrtcCreateProgram failed: {status}")
            options = [arch_opt, b"-default-device"]
            for inc_path in compiler._find_cuda_include_paths():
                options.append(f"--include-path={inc_path}".encode())
            opts_array = (ctypes.c_char_p * len(options))(*options)
            status = nvrtc.nvrtcCompileProgram(prog, len(options), opts_array)
            if status != 0:
                log_size = ctypes.c_size_t(0)
                nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
                log_buf = ctypes.create_string_buffer(log_size.value)
                nvrtc.nvrtcGetProgramLog(prog, log_buf)
                raise RuntimeError(f"NVRTC compile failed for '{name}':\n{log_buf.value.decode()}")
            ptx_size = ctypes.c_size_t(0)
            nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
            ptx_buf = ctypes.create_string_buffer(ptx_size.value)
            nvrtc.nvrtcGetPTX(prog, ptx_buf)
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            return name, ptx_buf

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            return list(pool.map(_compile_to_ptx, items))

    def _load_ptx_modules(self, ptx_results):
        """Load pre-compiled PTX into CUDA modules on main thread (needs GPU context)."""
        import ctypes
        from zse_compiler.runtime.compiler import CompiledKernel

        driver = self._compiler._load_cuda_driver()
        if driver is None:
            raise RuntimeError("CUDA driver not available")

        for name, ptx_buf in ptx_results:
            module = ctypes.c_void_p()
            status = driver.cuModuleLoadData(ctypes.byref(module), ptx_buf)
            if status != 0:
                raise RuntimeError(f"cuModuleLoadData failed for '{name}': {status}")
            func = ctypes.c_void_p()
            status = driver.cuModuleGetFunction(
                ctypes.byref(func), module, name.encode("utf-8"),
            )
            if status != 0:
                raise RuntimeError(f"cuModuleGetFunction failed for '{name}': {status}")
            self._compiled[name] = CompiledKernel(
                name=name, backend="cuda", function=func,
                module=module, source="[parallel-compiled]",
                _driver=driver,
            )

    def _get_cache_dir(self):
        """Get kernel cache directory."""
        import os, hashlib
        cache_dir = os.path.expanduser(f"~/.cache/zse/kernels/{self._backend}")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _cache_key(self, name: str, source: str) -> str:
        """Generate cache key from kernel name + source hash."""
        import hashlib
        h = hashlib.md5(source.encode()).hexdigest()[:12]
        return f"{name}_{h}"

    def _compile_all_cuda_parallel(self, items):
        """Parallel CUDA compilation: NVRTC in threads, module load on main thread."""
        import concurrent.futures
        import ctypes

        # Get driver from the existing compiler (already proven to work)
        driver = self._compiler._load_cuda_driver()
        if driver is None:
            raise RuntimeError("CUDA driver not available for parallel compilation")

        arch = self._detect_arch_string()
        arch_opt = f"--gpu-architecture={arch}".encode()

        # Step 1: Compile source → PTX in parallel (context-independent)
        def _compile_to_ptx(name_source):
            name, source = name_source
            from zse_compiler.runtime.compiler import RuntimeCompiler
            compiler = RuntimeCompiler()
            nvrtc = compiler._load_nvrtc()
            if nvrtc is None:
                raise RuntimeError(f"NVRTC not available for kernel '{name}'")
            source_bytes = source.encode("utf-8")
            prog = ctypes.c_void_p()
            status = nvrtc.nvrtcCreateProgram(
                ctypes.byref(prog), source_bytes, b"zse_kernel.cu", 0, None, None,
            )
            if status != 0:
                raise RuntimeError(f"nvrtcCreateProgram failed: {status}")
            options = [arch_opt, b"-default-device"]
            for inc_path in compiler._find_cuda_include_paths():
                options.append(f"--include-path={inc_path}".encode())
            opts_array = (ctypes.c_char_p * len(options))(*options)
            status = nvrtc.nvrtcCompileProgram(prog, len(options), opts_array)
            if status != 0:
                log_size = ctypes.c_size_t(0)
                nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
                log_buf = ctypes.create_string_buffer(log_size.value)
                nvrtc.nvrtcGetProgramLog(prog, log_buf)
                raise RuntimeError(f"NVRTC compile failed for '{name}':\n{log_buf.value.decode()}")
            ptx_size = ctypes.c_size_t(0)
            nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
            ptx_buf = ctypes.create_string_buffer(ptx_size.value)
            nvrtc.nvrtcGetPTX(prog, ptx_buf)
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            return name, ptx_buf

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            ptx_results = list(pool.map(_compile_to_ptx, items))

        # Step 2: Load modules + get functions on main thread (context-dependent)
        from zse_compiler.runtime.compiler import CompiledKernel

        for name, ptx_buf in ptx_results:
            module = ctypes.c_void_p()
            status = driver.cuModuleLoadData(ctypes.byref(module), ptx_buf)
            if status != 0:
                raise RuntimeError(f"cuModuleLoadData failed for '{name}': {status}")
            func = ctypes.c_void_p()
            status = driver.cuModuleGetFunction(
                ctypes.byref(func), module, name.encode("utf-8"),
            )
            if status != 0:
                raise RuntimeError(f"cuModuleGetFunction failed for '{name}': {status}")
            self._compiled[name] = CompiledKernel(
                name=name, backend="cuda", function=func,
                module=module, source="[parallel-compiled]",
                _driver=driver,
            )

    def _compile_all_rocm_parallel(self, items):
        """Parallel ROCm compilation with disk caching."""
        import concurrent.futures
        import ctypes
        import os

        hip = self._compiler._load_hip_runtime()
        hiprtc = self._compiler._load_hiprtc()
        if hip is None or hiprtc is None:
            raise RuntimeError("HIP runtime or HIPRTC not available")

        # Transform all sources to HIP C
        hip_items = [(name, self._cuda_to_hip(source)) for name, source in items]

        # Check cache
        cache_dir = self._get_cache_dir()
        cached = {}  # name -> code_bytes
        to_compile = []  # (name, source) that need compilation

        for name, source in hip_items:
            cache_file = os.path.join(cache_dir, self._cache_key(name, source) + ".hipfb")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    cached[name] = f.read()
            else:
                to_compile.append((name, source))

        # Compile only uncached kernels
        code_results = []
        if to_compile:
            inc_paths = self._compiler._find_rocm_include_paths()

            def _compile_to_code(name_source):
                name, source = name_source
                from zse_compiler.runtime.compiler import RuntimeCompiler
                local_hiprtc = RuntimeCompiler._load_hiprtc()
                if local_hiprtc is None:
                    raise RuntimeError(f"HIPRTC not available for kernel '{name}'")

                source_bytes = source.encode("utf-8")
                prog = ctypes.c_void_p()
                local_hiprtc.hiprtcCreateProgram(
                    ctypes.byref(prog), source_bytes, b"zse_kernel.hip", 0, None, None,
                )

                options = [f"-I{p}".encode() for p in inc_paths]
                if options:
                    opts_array = (ctypes.c_char_p * len(options))(*options)
                    status = local_hiprtc.hiprtcCompileProgram(prog, len(options), opts_array)
                else:
                    status = local_hiprtc.hiprtcCompileProgram(prog, 0, None)

                if status != 0:
                    log_size = ctypes.c_size_t(0)
                    local_hiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(log_size))
                    log_buf = ctypes.create_string_buffer(log_size.value)
                    local_hiprtc.hiprtcGetProgramLog(prog, log_buf)
                    raise RuntimeError(f"HIPRTC compile failed for '{name}':\n{log_buf.value.decode()}")

                code_size = ctypes.c_size_t(0)
                local_hiprtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size))
                code_buf = ctypes.create_string_buffer(code_size.value)
                local_hiprtc.hiprtcGetCode(prog, code_buf)
                local_hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
                return name, code_buf.raw

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                code_results = list(pool.map(_compile_to_code, to_compile))

            # Save newly compiled to cache
            for name, code_bytes in code_results:
                source = dict(hip_items)[name]
                cache_file = os.path.join(cache_dir, self._cache_key(name, source) + ".hipfb")
                try:
                    with open(cache_file, "wb") as f:
                        f.write(code_bytes)
                except Exception:
                    pass  # Non-fatal: cache write failure

        # Step 2: Load all modules on main thread (cached + freshly compiled)
        from zse_compiler.runtime.compiler import CompiledKernel

        all_code = [(name, cached[name]) for name in cached]
        all_code += code_results

        for name, code_bytes in all_code:
            code_buf = ctypes.create_string_buffer(code_bytes)
            module = ctypes.c_void_p()
            hip.hipModuleLoadData(ctypes.byref(module), code_buf)
            func = ctypes.c_void_p()
            hip.hipModuleGetFunction(ctypes.byref(func), module, name.encode("utf-8"))
            self._compiled[name] = CompiledKernel(
                name=name, backend="rocm", function=func,
                module=module, source="[cached]",
                _driver=hip,
            )

    def compile_kernel(self, name: str):
        """Compile a single kernel by name."""
        if name not in self._kernel_sources:
            raise ValueError(f"Unknown kernel: {name}")
        source = self._kernel_sources[name]
        # Skip CUDA->HIP transform if source already targets HIP natively
        # (e.g. portable @zse.kernel sources generated with backend='rocm').
        if self._backend == "rocm" and "hip/hip_runtime.h" not in source:
            source = self._cuda_to_hip(source)
        self._compiled[name] = self._compiler.compile(
            source, name, self._backend
        )

    @staticmethod
    def _cuda_to_hip(source: str) -> str:
        """Transform CUDA C kernel source to HIP C for ROCm compilation.

        Key differences for AMD wavefront-64:
        - cuda_fp16.h → hip/hip_fp16.h
        - WARP_SIZE 32 → 64 (single define change, no fragile string replaces)
        - Warp shuffle masks: 32-bit → 64-bit
        - Shared memory for warp results scaled to max wavefronts
        - GEMV kernel adaptations
        """
        import re

        # Headers
        source = source.replace('#include <cuda_fp16.h>', '#include <hip/hip_fp16.h>')
        source = source.replace('#include <cuda_runtime.h>', '#include <hip/hip_runtime.h>')

        # WARP_SIZE: single define change (all kernels use WARP_SIZE via CUDA_HEADER)
        source = source.replace('#define WARP_SIZE 32', '#define WARP_SIZE 64')

        # For any remaining hardcoded warp-size patterns (legacy):
        # tid / 32 and tid % 32 — only for warp_id/lane_id computation
        source = source.replace('tid / 32', 'tid / WARP_SIZE')
        source = source.replace('threadIdx.x / 32', 'threadIdx.x / WARP_SIZE')
        source = source.replace('tid % 32', 'tid % WARP_SIZE')
        source = source.replace('threadIdx.x % 32', 'threadIdx.x % WARP_SIZE')

        # Warp shuffle masks: 32-bit → 64-bit for wavefront64
        source = source.replace('0xffffffffffffffff', '__ZSE_MASK64__')
        source = source.replace('0xFFFFFFFFFFFFFFFF', '__ZSE_MASK64__')
        source = source.replace('0xFFFFFFFF,', '0xFFFFFFFFFFFFFFFFULL,')
        source = source.replace('0xffffffff,', '0xFFFFFFFFFFFFFFFFULL,')
        source = source.replace('__ZSE_MASK64__', '0xFFFFFFFFFFFFFFFFULL')

        # Shared memory for warp results: 32 → 16 (1024/64=16 wavefronts max)
        source = source.replace('warp_sums[32]', 'warp_sums[16]')
        source = source.replace('warp_max[32]', 'warp_max[16]')
        source = source.replace('warp_maxes[32]', 'warp_maxes[16]')
        source = source.replace('warp_sum[32]', 'warp_sum[16]')

        # Warp count: (blockDim.x + 31) / 32 → (blockDim.x + 63) / 64
        source = source.replace('(blockDim.x + 31) / 32', '(blockDim.x + 63) / 64')

        # Warp reduction offsets: start at 32 for wavefront64 (was 16 for warp32)
        source = re.sub(
            r'for \(int offset = 16;',
            'for (int offset = 32;',
            source
        )

        # Loop strides that are warp-sized
        source = re.sub(r'(\w+) \+= 32\)', r'\1 += WARP_SIZE)', source)

        # --- GEMV kernel adaptations (256 threads = 4 wavefronts of 64) ---
        source = source.replace('#define GEMV_RPB 8', '#define GEMV_RPB 8')  # Keep 8
        source = source.replace('#define FP16_GEMV_RPB 8', '#define FP16_GEMV_RPB 8')
        # Fused gate+up: 8 warps (4 gate + 4 up) → 4 wavefronts (2 gate + 2 up)
        source = source.replace('#define FGU_RPB 8', '#define FGU_RPB 4')
        source = source.replace('int is_up = warp_id >= 4', 'int is_up = warp_id >= 2')
        source = source.replace('int local_warp = warp_id & 3', 'int local_warp = warp_id & 1')
        source = source.replace('blockIdx.x * 4 + local_warp', 'blockIdx.x * 2 + local_warp')

        return source

    def get(self, name: str) -> CompiledKernel:
        """Get a compiled kernel."""
        if name not in self._compiled:
            self.compile_kernel(name)
        return self._compiled[name]

    def launch(self, name: str, grid: tuple, block: tuple, *args,
               shared_mem_bytes: int = 0, stream=None):
        """Launch a kernel by name (async — no GPU sync)."""
        kernel = self.get(name)
        config = LaunchConfig(grid=grid, block=block,
                              shared_mem_bytes=shared_mem_bytes,
                              stream=stream)
        self._launcher.launch(kernel, config, *args)

    def launch_prepacked(self, name: str, grid: tuple, block: tuple,
                         prepacked, shared_mem_bytes: int = 0, stream=None):
        """Launch with pre-packed args — zero allocation per call."""
        kernel = self.get(name)
        config = LaunchConfig(grid=grid, block=block,
                              shared_mem_bytes=shared_mem_bytes,
                              stream=stream)
        self._launcher.launch_prepacked(kernel, config, prepacked)

    def sync(self):
        """Explicit GPU sync — call only when host needs GPU results."""
        if self._compiled:
            kernel = next(iter(self._compiled.values()))
            self._launcher.sync(kernel)

    @property
    def num_compiled(self) -> int:
        return len(self._compiled)

    @property
    def kernel_names(self) -> list:
        return list(self.KERNEL_SOURCES.keys())

    def is_compiled(self, name: str) -> bool:
        return name in self._compiled
