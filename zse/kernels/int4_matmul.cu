/*
ZSE INT4 Matrix Multiplication CUDA Kernel

This kernel performs matrix multiplication with INT4 weights:
- Input: FP16 activations [M, K]
- Weights: Packed INT4 [N, K/2] as uint8
- Scales: FP16 [N, K/group_size]
- Output: FP16 [M, N]

Optimizations:
- Shared memory tiling
- Vectorized loads
- Register blocking
- Fused dequantization
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Tile sizes for shared memory
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define THREADS_PER_BLOCK 256

// Unpack INT4 from uint8 (2 values per byte)
__device__ __forceinline__ void unpack_int4(
    uint8_t packed,
    int8_t* val0,
    int8_t* val1
) {
    // Low 4 bits (shifted from [0,15] to [-8,7])
    *val0 = (int8_t)(packed & 0x0F) - 8;
    // High 4 bits
    *val1 = (int8_t)(packed >> 4) - 8;
}

// Dequantize INT4 value to FP16
__device__ __forceinline__ __half dequantize_int4_val(
    int8_t val,
    __half scale
) {
    return __hmul(__int2half_rn((int)val), scale);
}

/*
 * INT4 GEMM Kernel
 * C[M,N] = A[M,K] @ B[K,N]
 * Where B is stored as packed INT4 with scales
 */
extern "C" __global__ void int4_gemm_kernel(
    const __half* __restrict__ A,      // [M, K] input activations
    const uint8_t* __restrict__ B,     // [N, K/2] packed INT4 weights (row-major, transposed)
    const __half* __restrict__ scales, // [N, num_groups] scales
    __half* __restrict__ C,            // [M, N] output
    int M, int N, int K,
    int group_size
) {
    // Block indices
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    
    // Thread indices within block
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Shared memory for tiles
    __shared__ __half As[TILE_M][TILE_K + 1];  // +1 to avoid bank conflicts
    __shared__ __half Bs[TILE_K][TILE_N + 1];
    
    // Accumulator registers
    float acc[4][4] = {0.0f};
    
    // Starting positions
    int m_start = block_m * TILE_M;
    int n_start = block_n * TILE_N;
    
    // Number of K tiles
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k_start = k_tile * TILE_K;
        
        // Load A tile to shared memory
        // Each thread loads multiple elements
        for (int i = tid; i < TILE_M * TILE_K; i += THREADS_PER_BLOCK) {
            int tile_m = i / TILE_K;
            int tile_k = i % TILE_K;
            int global_m = m_start + tile_m;
            int global_k = k_start + tile_k;
            
            if (global_m < M && global_k < K) {
                As[tile_m][tile_k] = A[global_m * K + global_k];
            } else {
                As[tile_m][tile_k] = __float2half(0.0f);
            }
        }
        
        // Load B tile and dequantize to shared memory
        for (int i = tid; i < TILE_N * TILE_K / 2; i += THREADS_PER_BLOCK) {
            int tile_n = i / (TILE_K / 2);
            int tile_k_pair = i % (TILE_K / 2);
            int tile_k = tile_k_pair * 2;
            
            int global_n = n_start + tile_n;
            int global_k = k_start + tile_k;
            
            if (global_n < N && global_k < K) {
                // Load packed INT4
                uint8_t packed = B[global_n * (K / 2) + global_k / 2];
                
                // Get scale for this group
                int group_idx = global_k / group_size;
                __half scale = scales[global_n * ((K + group_size - 1) / group_size) + group_idx];
                
                // Unpack and dequantize
                int8_t val0, val1;
                unpack_int4(packed, &val0, &val1);
                
                Bs[tile_k][tile_n] = dequantize_int4_val(val0, scale);
                if (tile_k + 1 < TILE_K) {
                    Bs[tile_k + 1][tile_n] = dequantize_int4_val(val1, scale);
                }
            } else {
                Bs[tile_k][tile_n] = __float2half(0.0f);
                if (tile_k + 1 < TILE_K) {
                    Bs[tile_k + 1][tile_n] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        // Each thread computes a 4x4 block of the output tile
        int thread_m = (tid / 16) * 4;
        int thread_n = (tid % 16) * 4;
        
        if (thread_m < TILE_M && thread_n < TILE_N) {
            for (int k = 0; k < TILE_K; k++) {
                // Load values from shared memory
                float a0 = __half2float(As[thread_m + 0][k]);
                float a1 = __half2float(As[thread_m + 1][k]);
                float a2 = __half2float(As[thread_m + 2][k]);
                float a3 = __half2float(As[thread_m + 3][k]);
                
                float b0 = __half2float(Bs[k][thread_n + 0]);
                float b1 = __half2float(Bs[k][thread_n + 1]);
                float b2 = __half2float(Bs[k][thread_n + 2]);
                float b3 = __half2float(Bs[k][thread_n + 3]);
                
                // Accumulate
                acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
                acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
                acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
                acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    int thread_m = (tid / 16) * 4;
    int thread_n = (tid % 16) * 4;
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int global_m = m_start + thread_m + i;
            int global_n = n_start + thread_n + j;
            
            if (global_m < M && global_n < N) {
                C[global_m * N + global_n] = __float2half(acc[i][j]);
            }
        }
    }
}

/*
 * Optimized INT4 GEMV Kernel (for batch size 1 / decode step)
 * y[N] = x[K] @ W[K,N] (where W is INT4 packed)
 */
extern "C" __global__ void int4_gemv_kernel(
    const __half* __restrict__ x,      // [K] input vector
    const uint8_t* __restrict__ W,     // [N, K/2] packed INT4 weights
    const __half* __restrict__ scales, // [N, num_groups] scales
    __half* __restrict__ y,            // [N] output
    int N, int K,
    int group_size
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (n >= N) return;
    
    float acc = 0.0f;
    int num_groups = (K + group_size - 1) / group_size;
    
    // Process 2 elements at a time (one packed byte)
    for (int k = 0; k < K; k += 2) {
        uint8_t packed = W[n * (K / 2) + k / 2];
        
        int group_idx = k / group_size;
        float scale = __half2float(scales[n * num_groups + group_idx]);
        
        int8_t val0, val1;
        unpack_int4(packed, &val0, &val1);
        
        float w0 = (float)val0 * scale;
        float w1 = (float)val1 * scale;
        
        acc += __half2float(x[k]) * w0;
        if (k + 1 < K) {
            acc += __half2float(x[k + 1]) * w1;
        }
    }
    
    y[n] = __float2half(acc);
}
