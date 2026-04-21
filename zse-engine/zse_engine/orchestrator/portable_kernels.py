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
