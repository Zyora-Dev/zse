"""ZSE GPU primitives — thread indexing, shared memory, synchronization.

These are marker functions. At runtime in Python they return placeholder values.
The AST parser recognizes calls to these and emits corresponding IR nodes,
which codegen translates to backend-specific intrinsics:

    zse.thread_id(0) → threadIdx.x (CUDA) / thread_position_in_threadgroup (Metal)
    zse.block_id(0)  → blockIdx.x (CUDA) / threadgroup_position_in_grid (Metal)
"""


# --- Thread / Block indexing ---

def thread_id(axis: int = 0) -> int:
    """Get thread index within a block.
    axis: 0=x, 1=y, 2=z

    CUDA:  threadIdx.x/y/z
    HIP:   threadIdx.x/y/z (same as CUDA)
    Metal: thread_position_in_threadgroup[axis]
    """
    return 0  # Placeholder — replaced by IR node during compilation


def block_id(axis: int = 0) -> int:
    """Get block index within the grid.
    axis: 0=x, 1=y, 2=z

    CUDA:  blockIdx.x/y/z
    HIP:   blockIdx.x/y/z
    Metal: threadgroup_position_in_grid[axis]
    """
    return 0


def block_dim(axis: int = 0) -> int:
    """Get block dimension (number of threads per block).
    axis: 0=x, 1=y, 2=z

    CUDA:  blockDim.x/y/z
    HIP:   blockDim.x/y/z
    Metal: threads_per_threadgroup[axis]
    """
    return 0


def grid_dim(axis: int = 0) -> int:
    """Get grid dimension (number of blocks).
    axis: 0=x, 1=y, 2=z

    CUDA:  gridDim.x/y/z
    HIP:   gridDim.x/y/z
    Metal: threadgroups_per_grid[axis]
    """
    return 0


# --- Global thread index (convenience) ---

def global_id(axis: int = 0) -> int:
    """Get global thread index = block_id * block_dim + thread_id.

    Convenience function — AST parser expands this inline.
    """
    return 0


# --- Shared memory ---

def shared_memory(shape: tuple, dtype=None) -> None:
    """Declare shared memory (block-local fast memory).

    CUDA:  __shared__ T smem[shape]
    HIP:   __shared__ T smem[shape]
    Metal: threadgroup T smem[shape]
    """
    return None


# --- Synchronization ---

def syncthreads() -> None:
    """Synchronize all threads in a block.

    CUDA:  __syncthreads()
    HIP:   __syncthreads()
    Metal: threadgroup_barrier(mem_flags::mem_threadgroup)
    """
    pass


# --- Atomics ---

def atomic_add(ptr, val) -> None:
    """Atomic addition.

    CUDA:  atomicAdd(ptr, val)
    HIP:   atomicAdd(ptr, val)
    Metal: atomic_fetch_add_explicit(ptr, val, memory_order_relaxed)
    """
    pass


# --- Math intrinsics ---

def exp(x):
    """Exponential — maps to __expf / exp / metal::exp"""
    return 0.0


def log(x):
    """Natural log — maps to __logf / log / metal::log"""
    return 0.0


def sqrt(x):
    """Square root — maps to sqrtf / sqrt / metal::sqrt"""
    return 0.0


def rsqrt(x):
    """Reciprocal square root — maps to rsqrtf / rsqrt / metal::rsqrt"""
    return 0.0


def max_val(a, b):
    """Max — maps to fmaxf / fmax / metal::max"""
    return 0.0


def min_val(a, b):
    """Min — maps to fminf / fmin / metal::min"""
    return 0.0


def fma(a, b, c):
    """Fused multiply-add — maps to __fmaf_rn / fma / metal::fma"""
    return 0.0


# --- Warp-level primitives ---

def warp_shuffle_down(val, offset: int, width: int = 32):
    """Shuffle value down within a warp.

    Each thread receives the value from the thread `offset` lanes above it.
    Used for fast warp-level reductions without shared memory.

    CUDA:  __shfl_down_sync(0xffffffff, val, offset, width)
    HIP:   __shfl_down(val, offset, width)
    Metal: simd_shuffle_down(val, offset)
    """
    return 0.0


def warp_shuffle_up(val, offset: int, width: int = 32):
    """Shuffle value up within a warp.

    CUDA:  __shfl_up_sync(0xffffffff, val, offset, width)
    HIP:   __shfl_up(val, offset, width)
    Metal: simd_shuffle_up(val, offset)
    """
    return 0.0


def warp_shuffle_xor(val, lane_mask: int, width: int = 32):
    """Shuffle value with XOR of lane index.

    Used for butterfly-pattern reductions (very fast).

    CUDA:  __shfl_xor_sync(0xffffffff, val, lane_mask, width)
    HIP:   __shfl_xor(val, lane_mask, width)
    Metal: simd_shuffle_xor(val, lane_mask)
    """
    return 0.0


def warp_shuffle(val, src_lane: int, width: int = 32):
    """Read value from a specific lane in the warp.

    CUDA:  __shfl_sync(0xffffffff, val, src_lane, width)
    HIP:   __shfl(val, src_lane, width)
    Metal: simd_shuffle(val, src_lane)
    """
    return 0.0


def warp_ballot(predicate) -> int:
    """Ballot — returns bitmask of which lanes have predicate=true.

    CUDA:  __ballot_sync(0xffffffff, predicate)
    HIP:   __ballot(predicate)
    Metal: simd_ballot(predicate)
    """
    return 0


def warp_all(predicate) -> bool:
    """Returns true if predicate is true for ALL lanes in warp.

    CUDA:  __all_sync(0xffffffff, predicate)
    HIP:   __all(predicate)
    Metal: simd_all(predicate)
    """
    return True


def warp_any(predicate) -> bool:
    """Returns true if predicate is true for ANY lane in warp.

    CUDA:  __any_sync(0xffffffff, predicate)
    HIP:   __any(predicate)
    Metal: simd_any(predicate)
    """
    return True


def lane_id() -> int:
    """Get lane index within current warp (0-31).

    CUDA:  threadIdx.x % 32 (or via __lane_id())
    HIP:   __lane_id()
    Metal: thread_index_in_simdgroup
    """
    return 0


def warp_id() -> int:
    """Get warp index within current block.

    CUDA:  threadIdx.x / 32
    HIP:   threadIdx.x / 64 (AMD wavefront=64)
    Metal: thread_index_in_threadgroup / threads_per_simdgroup
    """
    return 0


# --- Warp-level reductions (high-level built-ins) ---

def warp_reduce_sum(val):
    """Reduce (sum) across all lanes in a warp. Returns result in all lanes.

    Implemented as butterfly shuffle-xor pattern:
        for offset in [16, 8, 4, 2, 1]:
            val += warp_shuffle_xor(val, offset)

    This is a built-in — codegen expands it to the shuffle loop.
    """
    return 0.0


def warp_reduce_max(val):
    """Reduce (max) across all lanes in a warp. Returns result in all lanes.

    Implemented as butterfly shuffle-xor pattern with fmaxf.
    """
    return 0.0


def warp_reduce_min(val):
    """Reduce (min) across all lanes in a warp. Returns result in all lanes.

    Implemented as butterfly shuffle-xor pattern with fminf.
    """
    return 0.0


# --- Vectorized memory types ---

def load_float4(tensor, idx: int):
    """Load 4 consecutive floats as a single 128-bit transaction.

    CUDA:  float4 v = reinterpret_cast<float4*>(ptr)[idx]
    HIP:   float4 v = reinterpret_cast<float4*>(ptr)[idx]
    Metal: float4 v = reinterpret_cast<device float4*>(ptr)[idx]
    """
    return (0.0, 0.0, 0.0, 0.0)


def store_float4(tensor, idx: int, v0, v1, v2, v3):
    """Store 4 consecutive floats as a single 128-bit transaction.

    CUDA:  reinterpret_cast<float4*>(ptr)[idx] = make_float4(v0, v1, v2, v3)
    HIP:   same
    Metal: reinterpret_cast<device float4*>(ptr)[idx] = float4(v0, v1, v2, v3)
    """
    pass


def load_half2(tensor, idx: int):
    """Load 2 consecutive half-precision values as a single 32-bit transaction.

    CUDA:  half2 v = reinterpret_cast<half2*>(ptr)[idx]
    HIP:   half2 v = reinterpret_cast<half2*>(ptr)[idx]
    Metal: half2 v = reinterpret_cast<device half2*>(ptr)[idx]
    """
    return (0.0, 0.0)


def store_half2(tensor, idx: int, v0, v1):
    """Store 2 consecutive half-precision values as a single 32-bit transaction."""
    pass


# --- Block-level reduction built-ins ---

def block_reduce_sum(val):
    """Reduce (sum) across all threads in a block.

    Two-stage: warp-level reduction → shared memory → cross-warp reduction.
    Returns final result in thread 0 (other threads get undefined value).

    Codegen expands this to:
        1. warp_reduce_sum(val) per warp
        2. Store warp results to shared memory
        3. syncthreads
        4. First warp reduces the warp results
    """
    return 0.0


def block_reduce_max(val):
    """Reduce (max) across all threads in a block. Result in thread 0."""
    return 0.0


def block_reduce_min(val):
    """Reduce (min) across all threads in a block. Result in thread 0."""
    return 0.0


# --- Tiling helper ---

def tile_load(tensor, tile_row: int, tile_col: int, tile_size: int, shared_buf=None):
    """Load a tile from global memory into shared memory.

    Cooperative load — each thread in the block loads one or more elements.
    syncthreads is called after load completes.
    """
    return None


def tile_store(shared_buf, tensor, tile_row: int, tile_col: int, tile_size: int):
    """Store a tile from shared memory back to global memory."""
    pass


# --- Additional atomics ---

def atomic_max(ptr, val):
    """Atomic max.
    CUDA:  atomicMax(ptr, val)
    HIP:   atomicMax(ptr, val)
    Metal: atomic_fetch_max_explicit(ptr, val, memory_order_relaxed)
    """
    pass


def atomic_min(ptr, val):
    """Atomic min."""
    pass


def atomic_cas(ptr, compare, val):
    """Atomic compare-and-swap.
    CUDA:  atomicCAS(ptr, compare, val)
    HIP:   atomicCAS(ptr, compare, val)
    Metal: atomic_compare_exchange_weak_explicit(...)
    """
    return 0


# --- FP16 Conversion ---

def half_to_float(x):
    """Convert half-precision to float.
    CUDA:  __half2float(x)
    HIP:   __half2float(x)
    Metal: float(x)
    """
    return 0.0


def float_to_half(x):
    """Convert float to half-precision.
    CUDA:  __float2half(x)
    HIP:   __float2half(x)
    Metal: half(x)
    """
    return 0.0


# --- Additional Math ---

def pow(base, exponent):
    """Power function — maps to powf / pow / metal::pow"""
    return 0.0


def cos(x):
    """Cosine — maps to cosf / cos / metal::cos"""
    return 0.0


def sin(x):
    """Sine — maps to sinf / sin / metal::sin"""
    return 0.0


# --- Dynamic Shared Memory ---

def dynamic_shared_memory(dtype=None):
    """Declare dynamic shared memory (size set at kernel launch).

    CUDA:  extern __shared__ float smem[];
    HIP:   extern __shared__ float smem[];
    Metal: threadgroup float* smem [[threadgroup(0)]]
    """
    return None


# --- Tensor Core / WMMA intrinsics ---

def wmma_load_a(tensor, row: int, col: int, stride: int):
    """Load a 16x16 matrix tile into WMMA fragment A.

    CUDA:  nvcuda::wmma::load_matrix_sync(frag_a, ptr + row*stride + col, stride)
    HIP:   rocwmma::load_matrix_sync(frag_a, ptr + row*stride + col, stride)
    Metal: simdgroup_matrix<half, 8, 8> — Apple uses 8x8 simdgroup matrices
    """
    return None


def wmma_load_b(tensor, row: int, col: int, stride: int):
    """Load a 16x16 matrix tile into WMMA fragment B."""
    return None


def wmma_fill(val=0.0):
    """Initialize a WMMA accumulator fragment to a value.

    CUDA:  nvcuda::wmma::fill_fragment(frag_c, val)
    """
    return None


def wmma_mma(a_frag, b_frag, c_frag):
    """Execute WMMA matrix multiply-accumulate: C = A * B + C

    CUDA:  nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c)
    HIP:   rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c)
    Metal: simdgroup_multiply_accumulate(c, a, b, c)
    """
    return None


def wmma_store(tensor, row: int, col: int, stride: int, frag=None):
    """Store WMMA accumulator fragment back to global memory.

    CUDA:  nvcuda::wmma::store_matrix_sync(ptr + row*stride + col, frag_c, stride, nvcuda::wmma::mem_row_major)
    """
    return None


# Mapping for AST parser to recognize these calls
PRIMITIVE_FUNCTIONS = {
    # Thread indexing
    "thread_id", "block_id", "block_dim", "grid_dim", "global_id",
    "lane_id", "warp_id",
    # Memory
    "shared_memory", "dynamic_shared_memory", "syncthreads",
    # Atomics
    "atomic_add", "atomic_max", "atomic_min", "atomic_cas",
    # Math
    "exp", "log", "sqrt", "rsqrt", "max_val", "min_val", "fma",
    "pow", "cos", "sin",
    # FP16 conversion
    "half_to_float", "float_to_half",
    # Warp primitives
    "warp_shuffle_down", "warp_shuffle_up", "warp_shuffle_xor", "warp_shuffle",
    "warp_ballot", "warp_all", "warp_any",
    # Warp reductions
    "warp_reduce_sum", "warp_reduce_max", "warp_reduce_min",
    # Block reductions
    "block_reduce_sum", "block_reduce_max", "block_reduce_min",
    # Vectorized memory
    "load_float4", "store_float4", "load_half2", "store_half2",
    # Tiling
    "tile_load", "tile_store",
    # WMMA / Tensor Core
    "wmma_load_a", "wmma_load_b", "wmma_fill", "wmma_mma", "wmma_store",
}
