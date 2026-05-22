"""ZSE Portable Kernels — LLM inference kernels written as @zse.kernel Python.

These kernels auto-generate correct code for all three backends:
- CUDA C (NVIDIA)
- HIP C (AMD ROCm)
- Metal Shading Language (Apple)

Unlike the raw CUDA C strings in kernels.py, these are backend-portable
and use the ZSE kernel compiler's full pipeline.

Note: We annotate params with string type hints that the AST parser resolves:
- "half_tensor" → half* (CUDA) / half* (HIP) / device half* (Metal)
- "uint8_tensor" → unsigned char* / uchar*
- "int8_tensor" → signed char* / char*
- "int32_tensor" → int*
"""

import zse_compiler as zse


# ============================================================================
# INT4 Dequantize + Matrix Multiply
# ============================================================================
# out[m,n] = sum_k( dequant(weight[n,k]) * input[m,k] )
# Weight is packed uint8: 2 INT4 values per byte
# Scales/zeros per group: scale[n, k/group_size], zeros[n, k/group_size]

@zse.kernel
def dequant_matmul_int4(
    out: "half_tensor",         # [M, N]
    weight: "uint8_tensor",     # [N, K/2] packed INT4
    scales: "half_tensor",      # [N, num_groups]
    zeros: "half_tensor",       # [N, num_groups]
    inp: "half_tensor",         # [M, K]
    M: int, N: int, K: int,
    group_size: int,
):
    m = zse.block_id(1)
    n = zse.block_id(0) * zse.block_dim(0) + zse.thread_id(0)

    if n >= N:
        return
    if m >= M:
        return

    acc: float = 0.0
    num_groups = (K + group_size - 1) / group_size

    for k in range(K):
        group_idx = k / group_size
        scale_val = zse.half_to_float(scales[n * int(num_groups) + group_idx])
        zero_val = zse.half_to_float(zeros[n * int(num_groups) + group_idx])

        # Unpack INT4 from packed byte
        byte_idx = (n * K + k) / 2
        packed = int(weight[byte_idx])
        # Even k: low nibble, Odd k: high nibble
        nibble = packed & 0x0F if k % 2 == 0 else (packed >> 4) & 0x0F

        w = (float(nibble) - zero_val) * scale_val
        x = zse.half_to_float(inp[m * K + k])
        acc = acc + w * x

    out[m * N + n] = zse.float_to_half(acc)


# ============================================================================
# INT8 Dequantize + Matrix Multiply (symmetric quantization)
# ============================================================================

@zse.kernel
def dequant_matmul_int8(
    out: "half_tensor",         # [M, N]
    weight: "int8_tensor",      # [N, K]
    scales: "half_tensor",      # [N, num_groups]
    inp: "half_tensor",         # [M, K]
    M: int, N: int, K: int,
    group_size: int,
):
    m = zse.block_id(1)
    n = zse.block_id(0) * zse.block_dim(0) + zse.thread_id(0)

    if n >= N:
        return
    if m >= M:
        return

    acc: float = 0.0
    num_groups = (K + group_size - 1) / group_size

    for k in range(K):
        group_idx = k / group_size
        scale_val = zse.half_to_float(scales[n * int(num_groups) + group_idx])

        w = float(weight[n * K + k]) * scale_val
        x = zse.half_to_float(inp[m * K + k])
        acc = acc + w * x

    out[m * N + n] = zse.float_to_half(acc)


# ============================================================================
# Paged Attention — reads K/V from paged block tables
# ============================================================================
# Grid: (num_seqs, num_heads), Block: (head_dim or 256)
# Each block computes attention for one (sequence, head) pair

@zse.kernel
def paged_attention(
    out: "half_tensor",             # [num_seqs, num_heads, head_dim]
    q: "half_tensor",               # [num_seqs, num_heads, head_dim]
    kv_cache: "half_tensor",        # Contiguous GPU slab
    block_table: "int32_tensor",    # [num_seqs, max_blocks]
    seq_lens: "int32_tensor",       # [num_seqs]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    num_layers: int,
    layer_idx: int,
    scale: float,
):
    seq_idx = zse.block_id(0)
    head_idx = zse.block_id(1)
    tid = zse.thread_id(0)

    seq_len = int(seq_lens[seq_idx])
    if seq_len == 0:
        return

    # GQA mapping: multiple Q heads share one KV head
    kv_head_idx = head_idx / (num_heads / num_kv_heads)

    # Block layout sizes (in half elements)
    kv_head_stride = block_size * head_dim
    kv_type_stride = num_kv_heads * kv_head_stride
    layer_stride = 2 * kv_type_stride
    block_total = num_layers * layer_stride

    # Dynamic shared memory for attention scores
    scores = zse.dynamic_shared_memory()

    # Q pointer for this (seq, head)
    q_base = seq_idx * num_heads * head_dim + head_idx * head_dim

    # Phase 1: compute Q·K scores
    max_score: float = -1000000.0
    num_blocks = (seq_len + block_size - 1) / block_size

    for b in range(int(num_blocks)):
        block_id_val = int(block_table[seq_idx * max_blocks_per_seq + b])
        if block_id_val < 0:
            pass  # skip evicted
        else:
            tokens_in_block = min(block_size, seq_len - b * block_size)
            # K base for this block
            k_base = block_id_val * block_total + layer_idx * layer_stride + kv_head_idx * kv_head_stride

            for t in range(tokens_in_block):
                if tid == 0:
                    dot: float = 0.0
                    for d in range(head_dim):
                        dot = dot + zse.half_to_float(q[q_base + d]) * zse.half_to_float(kv_cache[k_base + t * head_dim + d])
                    dot = dot * scale
                    global_t = b * block_size + t
                    scores[global_t] = dot
                    if dot > max_score:
                        max_score = dot

    zse.syncthreads()

    # Phase 2: softmax
    if tid == 0:
        sum_exp: float = 0.0
        for i in range(seq_len):
            scores[i] = zse.exp(scores[i] - max_score)
            sum_exp = sum_exp + scores[i]

        inv_sum: float = 1.0 / sum_exp
        for i in range(seq_len):
            scores[i] = scores[i] * inv_sum

    zse.syncthreads()

    # Phase 3: weighted sum of V
    out_base = seq_idx * num_heads * head_dim + head_idx * head_dim

    for d in range(head_dim):
        if tid == 0:
            acc: float = 0.0
            for b in range(int(num_blocks)):
                block_id_val2 = int(block_table[seq_idx * max_blocks_per_seq + b])
                if block_id_val2 >= 0:
                    tokens_in_block2 = min(block_size, seq_len - b * block_size)
                    v_base = block_id_val2 * block_total + layer_idx * layer_stride + kv_type_stride + kv_head_idx * kv_head_stride
                    for t2 in range(tokens_in_block2):
                        global_t2 = b * block_size + t2
                        acc = acc + scores[global_t2] * zse.half_to_float(kv_cache[v_base + t2 * head_dim + d])
            out[out_base + d] = zse.float_to_half(acc)


# ============================================================================
# Tiled INT4 Dequantize + Matrix Multiply (production-equivalent)
# ============================================================================
# Matches semantics of the hand-written TILED_DEQUANT_MATMUL_INT4_CUDA blob:
#   out[m, n] = sum_k( (nibble(weight[n, k]) * scale[n, g] + zero[n, g]) * input[m, k] )
# where g = k / group_size.
#
# This is the asymmetric dequant convention used by the .zse format (matches
# `low * scale + zero` in format/quantize.py). DO NOT change to (q - zero) * scale.
#
# Grid: (ceil(N/32), ceil(M/32)), Block: (32, 32) — one thread per output element.
# Tile loaded cooperatively into shared memory; inner k-loop is a scalar dot
# product across the tile. Phase-1 portable port: no tensor-core / MFMA yet.

@zse.kernel
def tiled_dequant_matmul_int4(
    out: "half_tensor",          # [M, N]
    weight: "uint8_tensor",      # [N, K/2] packed INT4 (low nibble first)
    scales: "half_tensor",       # [N, num_groups]
    zeros: "half_tensor",        # [N, num_groups]
    inp: "half_tensor",          # [M, K]
    M: int, N: int, K: int,
    group_size: int,
):
    bx = zse.block_id(0)
    by = zse.block_id(1)
    tx = zse.thread_id(0)
    ty = zse.thread_id(1)

    row = by * 32 + ty   # M dim
    col = bx * 32 + tx   # N dim

    As = zse.shared_memory((32, 32), zse.float32)
    Bs = zse.shared_memory((32, 32), zse.float32)

    acc: float = 0.0
    num_tiles = (K + 31) / 32
    num_groups = (K + group_size - 1) / group_size

    for t in range(int(num_tiles)):
        # ---- Load input tile As[ty, tx] = input[row, t*32 + tx] ----
        a_col = t * 32 + tx
        if row < M:
            if a_col < K:
                As[ty * 32 + tx] = zse.half_to_float(inp[row * K + a_col])
            else:
                As[ty * 32 + tx] = 0.0
        else:
            As[ty * 32 + tx] = 0.0

        # ---- Load + dequant weight tile Bs[ty, tx] = dequant(weight[col, t*32 + ty]) ----
        w_k = t * 32 + ty
        if col < N:
            if w_k < K:
                group_idx = w_k / group_size
                scale_val = zse.half_to_float(scales[col * int(num_groups) + group_idx])
                zero_val = zse.half_to_float(zeros[col * int(num_groups) + group_idx])

                byte_idx = (col * K + w_k) / 2
                packed = int(weight[byte_idx])
                nibble = packed & 0x0F if w_k % 2 == 0 else (packed >> 4) & 0x0F
                Bs[ty * 32 + tx] = float(nibble) * scale_val + zero_val
            else:
                Bs[ty * 32 + tx] = 0.0
        else:
            Bs[ty * 32 + tx] = 0.0

        zse.syncthreads()

        # ---- Scalar inner k-loop across the 32-wide tile ----
        for k in range(32):
            acc = acc + As[ty * 32 + k] * Bs[k * 32 + tx]

        zse.syncthreads()

    # ---- Store with fp16-range clamp (matches production kernel) ----
    if row < M:
        if col < N:
            if acc > 65504.0:
                acc = 65504.0
            if acc < -65504.0:
                acc = -65504.0
            out[row * N + col] = zse.float_to_half(acc)


# ============================================================================
# MFMA-accelerated INT4 Dequantize + Matrix Multiply (Phase 2, ROCm/CDNA only)
# ============================================================================
# Same numerical semantics as `tiled_dequant_matmul_int4` above, but the inner
# scalar k-loop is replaced by `mfma_f32_16x16x16_f16`.
#
# Layout (per CDNA3 MFMA 16x16x16 fp16 fragment spec):
#   1 wavefront (64 lanes) -> one 16x16 output tile.
#   Per-lane fragments:
#     a_frag[i] = A[lane % 16, (lane / 16) * 4 + i]      for i in 0..3
#     b_frag[i] = B[(lane / 16) * 4 + i, lane % 16]      for i in 0..3
#     c_frag[i] = D[(lane / 16) * 4 + i, lane % 16]      for i in 0..3  (fp32)
#
# Grid: (ceil(N/16), ceil(M/16)), Block: (64, 1, 1).
# Tail M/N elements are guarded; K must be a multiple of 16 (LLM-tile aligned).

@zse.kernel
def mfma_dequant_matmul_int4(
    out: "half_tensor",          # [M, N]
    weight: "uint8_tensor",      # [N, K/2] packed INT4 (low nibble first)
    scales: "half_tensor",       # [N, num_groups]
    zeros: "half_tensor",        # [N, num_groups]
    inp: "half_tensor",          # [M, K]
    M: int, N: int, K: int,
    group_size: int,
):
    m_block = zse.block_id(1)
    n_block = zse.block_id(0)
    lane = zse.thread_id(0)         # 0..63

    row_in_a = lane % 16            # M row this lane reads from A
    k_quad = int(lane / 16)         # which group of 4 K-elements (0..3)
    col_in_b = lane % 16            # N col this lane reads from B (and writes to D)

    m_a = m_block * 16 + row_in_a   # global M row for A loads
    n_b = n_block * 16 + col_in_b   # global N col for B loads / D stores

    a_frag = zse.local_array(4, zse.float16)
    b_frag = zse.local_array(4, zse.float16)
    c_frag = zse.local_array(4, zse.float32)

    for i in range(4):
        c_frag[i] = 0.0

    num_groups = (K + group_size - 1) / group_size

    for k_tile in range(0, K, 16):
        # ---- Load A fragment: a_frag[i] = inp[m_a, k_tile + k_quad*4 + i] ----
        for ia in range(4):
            ka = k_tile + k_quad * 4 + ia
            if m_a < M:
                if ka < K:
                    a_frag[ia] = inp[m_a * K + ka]
                else:
                    a_frag[ia] = zse.float_to_half(0.0)
            else:
                a_frag[ia] = zse.float_to_half(0.0)

        # ---- Load + dequant B fragment: b_frag[i] = dequant(weight[n_b, k_tile + k_quad*4 + i]) ----
        for ib in range(4):
            kb = k_tile + k_quad * 4 + ib
            if n_b < N:
                if kb < K:
                    group_idx = kb / group_size
                    scale_val = zse.half_to_float(scales[n_b * int(num_groups) + group_idx])
                    zero_val = zse.half_to_float(zeros[n_b * int(num_groups) + group_idx])

                    byte_idx = (n_b * K + kb) / 2
                    packed = int(weight[byte_idx])
                    nibble = packed & 0x0F if kb % 2 == 0 else (packed >> 4) & 0x0F
                    b_frag[ib] = zse.float_to_half(float(nibble) * scale_val + zero_val)
                else:
                    b_frag[ib] = zse.float_to_half(0.0)
            else:
                b_frag[ib] = zse.float_to_half(0.0)

        # ---- One CDNA3 matrix-core op per 16-wide K tile ----
        zse.mfma_f32_16x16x16_f16(a_frag, b_frag, c_frag)

    # ---- Store D fragment back to global: out[m_block*16 + k_quad*4 + i, n_b] = c_frag[i] ----
    for i in range(4):
        out_row = m_block * 16 + k_quad * 4 + i
        if out_row < M:
            if n_b < N:
                val = c_frag[i]
                if val > 65504.0:
                    val = 65504.0
                if val < -65504.0:
                    val = -65504.0
                out[out_row * N + n_b] = zse.float_to_half(val)


# ============================================================================
# Phase-3 MFMA INT4 Dequant + Matmul (ROCm/CDNA only)
# ============================================================================
# Builds on Phase 2 with three optimizations layered on:
#   (1) Cooperative dequant of B-tile into shared memory once per outer chunk,
#       reused across 4 MFMA passes — turns scalar per-K dequant into per-chunk.
#   (2) Vectorized weight load via zse.reinterpret(weight, uint32_t*) + zse.unpack_uint4
#       — fetches 8 packed nibbles per 32-bit load instead of 16 separate byte loads.
#   (3) Scale/zero lookup hoisted outside the inner k-loop — one lookup per lane
#       per chunk instead of one per K-step.
#
# Assumptions (matches all real .zse models):
#   - K divisible by CHUNK_K=64
#   - group_size divisible by 16 (lane window of 16 K-values lies within one group)
#   - K is even (packed INT4 storage)
#
# Layout:
#   Block = (64, 1, 1) = 1 wavefront -> one 16x16 output tile
#   CHUNK_K = 64; 4 inner MFMA passes per chunk

@zse.kernel
def mfma_dequant_matmul_int4_v3(
    out: "half_tensor",          # [M, N]
    weight: "uint8_tensor",      # [N, K/2] packed INT4 (low nibble first)
    scales: "half_tensor",       # [N, num_groups]
    zeros: "half_tensor",        # [N, num_groups]
    inp: "half_tensor",          # [M, K]
    M: int, N: int, K: int,
    group_size: int,
):
    m_block = zse.block_id(1)
    n_block = zse.block_id(0)
    lane = zse.thread_id(0)         # 0..63

    # Cooperative-load mapping: each lane covers 1 row x 16 K-cols within chunk
    row_load = int(lane / 4)        # 0..15 (M-row for A; N-col for B-tile)
    kload_off = (lane % 4) * 16     # 0..48

    # MFMA fragment mapping (16x16x16 fp16)
    row_frag = lane % 16            # 0..15
    kquad = int(lane / 16)          # 0..3

    As = zse.shared_memory((16, 64), zse.float16)
    Bs = zse.shared_memory((16, 64), zse.float16)

    # Vectorized weight pointer (8 nibbles per uint32 load)
    wq = zse.reinterpret(weight, zse.uint32)

    a_frag = zse.local_array(4, zse.float16)
    b_frag = zse.local_array(4, zse.float16)
    c_frag = zse.local_array(4, zse.float32)
    nibbles = zse.local_array(16, zse.int32)

    for i in range(4):
        c_frag[i] = 0.0

    num_groups = (K + group_size - 1) / group_size

    # ===== Outer K-chunk loop =====
    for k_chunk in range(0, K, 64):
        # ---- Load A tile [16 x 64]: As[row_load, kload_off+j] = inp[m_block*16 + row_load, k_chunk + kload_off + j] ----
        m_a_load = m_block * 16 + row_load
        for ja in range(16):
            kga = k_chunk + kload_off + ja
            if m_a_load < M:
                if kga < K:
                    As[row_load * 64 + kload_off + ja] = inp[m_a_load * K + kga]
                else:
                    As[row_load * 64 + kload_off + ja] = zse.float_to_half(0.0)
            else:
                As[row_load * 64 + kload_off + ja] = zse.float_to_half(0.0)

        # ---- Load + dequant B tile [16 x 64]: Bs[row_load, kload_off+j] = dequant(weight[n_block*16 + row_load, k_chunk + kload_off + j]) ----
        n_b_load = n_block * 16 + row_load
        if n_b_load < N:
            # Hoist scale/zero (one group per lane window under our assumptions)
            group_idx = (k_chunk + kload_off) / group_size
            scale_v = zse.half_to_float(scales[n_b_load * int(num_groups) + group_idx])
            zero_v = zse.half_to_float(zeros[n_b_load * int(num_groups) + group_idx])

            # Two u32 loads from packed weight -> 16 nibbles
            byte_base = n_b_load * (K / 2) + (k_chunk + kload_off) / 2
            u32_idx = byte_base / 4
            packed0 = wq[u32_idx]
            zse.unpack_uint4(packed0, nibbles, 0)
            packed1 = wq[u32_idx + 1]
            zse.unpack_uint4(packed1, nibbles, 8)

            for jb in range(16):
                kgb = k_chunk + kload_off + jb
                if kgb < K:
                    Bs[row_load * 64 + kload_off + jb] = zse.float_to_half(float(nibbles[jb]) * scale_v + zero_v)
                else:
                    Bs[row_load * 64 + kload_off + jb] = zse.float_to_half(0.0)
        else:
            for jc in range(16):
                Bs[row_load * 64 + kload_off + jc] = zse.float_to_half(0.0)

        zse.syncthreads()

        # ---- 4 MFMA k-passes within chunk ----
        for k_inner in range(0, 64, 16):
            for ia in range(4):
                a_frag[ia] = As[row_frag * 64 + k_inner + kquad * 4 + ia]
            for ib in range(4):
                b_frag[ib] = Bs[row_frag * 64 + k_inner + kquad * 4 + ib]
            zse.mfma_f32_16x16x16_f16(a_frag, b_frag, c_frag)

        zse.syncthreads()

    # ===== Store D fragment =====
    n_b_store = n_block * 16 + (lane % 16)
    for i in range(4):
        out_row = m_block * 16 + kquad * 4 + i
        if out_row < M:
            if n_b_store < N:
                val = c_frag[i]
                if val > 65504.0:
                    val = 65504.0
                if val < -65504.0:
                    val = -65504.0
                out[out_row * N + n_b_store] = zse.float_to_half(val)


# ============================================================================
# Small-M INT4 Dequant GEMV — native wavefront-64 layout (ROCm-tuned)
# ============================================================================
# Replaces the hand-written `batched_dequant_gemv_int4` C-string for M=2..8.
#
# Why a new kernel:
#   The existing bgemv is NVIDIA-style (32-lane warps, warp_id = tid/32).
#   On AMD wavefront-64 hardware, two "logical 32-warps" share one wavefront
#   and read different N-rows → divergent memory accesses serialize within
#   the wavefront. Effective BW is ~310 GB/s on Gate/Up (8.8% of MI300X peak).
#
# Design:
#   - 8 wavefronts (64 lanes each) per block = 512 threads/block
#   - 1 wavefront per output N-row, 8 N-rows per block
#   - Each lane processes one u32 (8 packed INT4 nibbles) per iteration,
#     striding K with stride 64 u32s = 512 K-elements per wavefront iter.
#   - All lanes in a wavefront read the SAME row → fully coalesced loads.
#   - Dequant once per lane per iter, then 8 FMAs per M-row (M reuse).
#   - Wavefront-64 reduce at end via __shfl_xor(width=64).
#
# Constraints:
#   - M <= 8 (uses fixed-size local_array[8])
#   - K % 8 == 0 (u32-aligned weight loads)
#   - group_size >= 8 (so 8-K window stays in one group per lane)

@zse.kernel
def bgemv_int4_wave64(
    out: "half_tensor",          # [M, N]
    weight: "uint8_tensor",      # [N, K/2] packed INT4 (low nibble first)
    scales: "half_tensor",       # [N, num_groups]
    zeros: "half_tensor",        # [N, num_groups]
    inp: "half_tensor",          # [M, K]
    M: int, N: int, K: int,
    group_size: int,
):
    tid = zse.thread_id(0)
    wf_id = tid / 64                  # 0..7
    lane = tid % 64                   # 0..63
    row = zse.block_id(0) * 8 + wf_id

    if row < N:
        num_groups = (K + group_size - 1) / group_size
        half_K = K / 2
        num_u32 = half_K / 4          # = K / 8

        wq = zse.reinterpret(weight, zse.uint32)

        nibbles = zse.local_array(8, zse.int32)
        w_dq = zse.local_array(8, zse.float32)
        acc = zse.local_array(8, zse.float32)

        for m_init in range(8):
            acc[m_init] = 0.0

        # Per-lane scale/zero cache (group changes are rare since one lane's
        # 8-K window fits inside one group when group_size >= 8).
        prev_g: int = -1
        s_val: float = 0.0
        z_val: float = 0.0

        scale_row_off = row * int(num_groups)
        w_row_u32_base = row * int(num_u32)

        # Each lane strides through num_u32 with stride 64.
        for i in range(int(lane), int(num_u32), 64):
            packed = wq[w_row_u32_base + i]
            zse.unpack_uint4(packed, nibbles, 0)

            k_base = i * 8
            g = k_base / group_size
            if g != prev_g:
                s_val = zse.half_to_float(scales[scale_row_off + g])
                z_val = zse.half_to_float(zeros[scale_row_off + g])
                prev_g = g

            # Dequant once per K-window
            for j in range(8):
                w_dq[j] = float(nibbles[j]) * s_val + z_val

            # Accumulate against all M input rows
            for m in range(8):
                if m < M:
                    inp_row_off = m * K + k_base
                    for j2 in range(8):
                        acc[m] = acc[m] + w_dq[j2] * zse.half_to_float(inp[inp_row_off + j2])

        # ===== Wavefront-64 reduction (manual butterfly) =====
        for m_red in range(8):
            if m_red < M:
                v = acc[m_red]
                v = v + zse.warp_shuffle_xor(v, 32, 64)
                v = v + zse.warp_shuffle_xor(v, 16, 64)
                v = v + zse.warp_shuffle_xor(v, 8, 64)
                v = v + zse.warp_shuffle_xor(v, 4, 64)
                v = v + zse.warp_shuffle_xor(v, 2, 64)
                v = v + zse.warp_shuffle_xor(v, 1, 64)
                if lane == 0:
                    if v > 65504.0:
                        v = 65504.0
                    if v < -65504.0:
                        v = -65504.0
                    out[m_red * N + row] = zse.float_to_half(v)


# ============================================================================
# Fused RoPE + KV cache write — single kernel, no K-buffer round-trip
# ============================================================================
# Replaces:
#   1. batched_rotary_embedding  (RoPE in-place on Q and K)
#   2. batched_kv_cache_write    (write K, V to paged KV cache)
#
# Wins:
#   - Saves 1 launch per layer (64 layers × 1 = 64 fewer launches per token)
#   - K never round-trips through global memory: RoPE-rotated K is written
#     directly to the KV cache slot. The intermediate `k_buf` write+read
#     is eliminated entirely.
#   - V write is along for the ride (same per-sequence offset math).
#
# Grid layout:
#   grid = (ceil(max_threads / 256), M),  block = (256, 1, 1)
#   max_threads = max(num_heads * head_dim/2, num_kv_heads * head_dim)
#
# Each block.y handles one sequence; threads in block.x cover element work.
# Q heads typically dominate (e.g. Qwen2.5-32B: Q=40*64=2560 vs V=8*128=1024).

@zse.kernel
def fused_rope_kv_write(
    q: "half_tensor",              # [M, num_heads, head_dim] in/out
    k: "half_tensor",              # [M, num_kv_heads, head_dim] in (no writeback)
    v: "half_tensor",              # [M, num_kv_heads, head_dim] in
    kv_cache: "half_tensor",       # contiguous KV slab
    block_table: "int32_tensor",   # [M, max_blocks_per_seq]
    positions: "int32_tensor",     # [M]
    M: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    num_layers: int,
    layer_idx: int,
    theta_base: float,
):
    seq = zse.block_id(1)
    if seq >= M:
        return

    idx = zse.block_id(0) * zse.block_dim(0) + zse.thread_id(0)

    half_dim = head_dim / 2
    pos = int(positions[seq])

    # ===== Q-side RoPE (in place) =====
    total_q = num_heads * half_dim
    if idx < total_q:
        pair_q = idx % half_dim
        head_q = idx / half_dim
        freq_q = 1.0 / zse.pow(theta_base, float(2 * pair_q) / float(head_dim))
        angle_q = float(pos) * freq_q
        cq = zse.cos(angle_q)
        sq = zse.sin(angle_q)
        base_q = seq * num_heads * head_dim + head_q * head_dim
        q0 = zse.half_to_float(q[base_q + pair_q])
        q1 = zse.half_to_float(q[base_q + pair_q + half_dim])
        q[base_q + pair_q] = zse.float_to_half(q0 * cq - q1 * sq)
        q[base_q + pair_q + half_dim] = zse.float_to_half(q1 * cq + q0 * sq)

    # ===== Shared KV cache offset math =====
    block_idx_token = pos / block_size
    token_in_block = pos % block_size
    bid = int(block_table[seq * max_blocks_per_seq + block_idx_token])

    if bid >= 0:
        kv_head_stride = block_size * head_dim
        kv_type_stride = num_kv_heads * kv_head_stride
        layer_stride = 2 * kv_type_stride
        block_total = num_layers * layer_stride
        kv_base = bid * block_total + layer_idx * layer_stride

        # ===== K-side RoPE → write directly to KV cache (no k_buf round-trip) =====
        total_k_rope = num_kv_heads * half_dim
        if idx < total_k_rope:
            pair_k = idx % half_dim
            head_k = idx / half_dim
            freq_k = 1.0 / zse.pow(theta_base, float(2 * pair_k) / float(head_dim))
            angle_k = float(pos) * freq_k
            ck = zse.cos(angle_k)
            sk = zse.sin(angle_k)
            base_k = seq * num_kv_heads * head_dim + head_k * head_dim
            k0 = zse.half_to_float(k[base_k + pair_k])
            k1 = zse.half_to_float(k[base_k + pair_k + half_dim])
            k_off = kv_base + head_k * kv_head_stride + token_in_block * head_dim
            kv_cache[k_off + pair_k] = zse.float_to_half(k0 * ck - k1 * sk)
            kv_cache[k_off + pair_k + half_dim] = zse.float_to_half(k1 * ck + k0 * sk)

        # ===== V write (no rotation) =====
        total_v = num_kv_heads * head_dim
        if idx < total_v:
            d = idx % head_dim
            head_v = idx / head_dim
            v_off = kv_base + kv_type_stride + head_v * kv_head_stride + token_in_block * head_dim + d
            kv_cache[v_off] = v[seq * num_kv_heads * head_dim + head_v * head_dim + d]
