"""
ZSE Triton Attention Kernels

Paged attention implementation using Triton for memory-efficient inference.
Based on PagedAttention from vLLM but with ZSE optimizations:
- Supports quantized KV cache (INT4/INT8)
- Dynamic block sizing based on available memory
- GQA (Grouped Query Attention) support

Author: ZSE Team
"""

import math
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def check_triton_available():
    """Check if Triton is available."""
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available. Install with: pip install triton>=2.1.0"
        )


if TRITON_AVAILABLE:
    @triton.jit
    def _paged_attention_v1_kernel(
        # Pointers
        output_ptr,           # [num_seqs, num_heads, head_dim]
        query_ptr,            # [num_seqs, num_heads, head_dim]
        key_cache_ptr,        # [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache_ptr,      # [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables_ptr,     # [num_seqs, max_num_blocks_per_seq]
        context_lens_ptr,     # [num_seqs]
        # Sizes
        num_seqs,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_num_blocks_per_seq,
        # Strides for query
        stride_qn,
        stride_qh,
        # Strides for key/value cache
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        # Strides for block table
        stride_btn,
        # Scale
        scale,
        # Softmax params
        BLOCK_M: tl.constexpr,  # Block size for sequence dimension
        BLOCK_DMODEL: tl.constexpr,  # Must equal head_dim
    ):
        """
        Paged Attention V1 Triton Kernel
        
        This kernel computes attention over paged KV cache blocks.
        Each program instance handles one (sequence, head) pair.
        
        Memory layout:
        - Query: [num_seqs, num_heads, head_dim]
        - Key cache: [num_blocks, num_kv_heads, block_size, head_dim]
        - Value cache: [num_blocks, num_kv_heads, block_size, head_dim]
        - Block table: [num_seqs, max_blocks] - maps seq position to block index
        """
        # Program ID: one program per (seq, head) pair
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        
        # For GQA: map query head to KV head
        kv_head_idx = head_idx // (num_heads // num_kv_heads)
        
        # Get context length for this sequence
        context_len = tl.load(context_lens_ptr + seq_idx)
        
        # Calculate number of blocks for this sequence
        num_blocks = (context_len + block_size - 1) // block_size
        
        # Load query vector for this (seq, head)
        q_offset = seq_idx * stride_qn + head_idx * stride_qh
        q = tl.load(
            query_ptr + q_offset + tl.arange(0, BLOCK_DMODEL),
            mask=tl.arange(0, BLOCK_DMODEL) < head_dim,
            other=0.0
        )
        
        # Initialize accumulators
        m_i = float("-inf")  # Max value for stable softmax
        l_i = 0.0           # Sum of exp values
        acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
        
        # Iterate over all blocks in the sequence
        for block_idx in range(num_blocks):
            # Get physical block number from block table
            block_table_offset = seq_idx * stride_btn + block_idx
            physical_block_num = tl.load(block_tables_ptr + block_table_offset)
            
            # Calculate start position in this block
            start_pos = block_idx * block_size
            
            # Load keys from this block
            # Key shape in cache: [num_blocks, num_kv_heads, block_size, head_dim]
            k_base = (
                physical_block_num * stride_kb + 
                kv_head_idx * stride_kh
            )
            
            # Load and process each position in the block
            for pos_in_block in range(block_size):
                token_pos = start_pos + pos_in_block
                
                # Skip if beyond context length
                if token_pos >= context_len:
                    continue
                
                # Load key vector
                k_offset = k_base + pos_in_block * stride_ks
                k = tl.load(
                    key_cache_ptr + k_offset + tl.arange(0, BLOCK_DMODEL),
                    mask=tl.arange(0, BLOCK_DMODEL) < head_dim,
                    other=0.0
                )
                
                # Compute attention score: q @ k^T * scale
                qk = tl.sum(q * k) * scale
                
                # Online softmax update
                m_new = tl.maximum(m_i, qk)
                alpha = tl.exp(m_i - m_new)
                beta = tl.exp(qk - m_new)
                
                # Update accumulator with rescaling
                acc = acc * alpha
                
                # Load value vector
                v_offset = k_base + pos_in_block * stride_ks
                v = tl.load(
                    value_cache_ptr + v_offset + tl.arange(0, BLOCK_DMODEL),
                    mask=tl.arange(0, BLOCK_DMODEL) < head_dim,
                    other=0.0
                )
                
                acc = acc + beta * v
                l_i = l_i * alpha + beta
                m_i = m_new
        
        # Normalize by sum of exp values
        acc = acc / l_i
        
        # Store output
        out_offset = seq_idx * stride_qn + head_idx * stride_qh
        tl.store(
            output_ptr + out_offset + tl.arange(0, BLOCK_DMODEL),
            acc.to(output_ptr.dtype.element_ty),
            mask=tl.arange(0, BLOCK_DMODEL) < head_dim
        )


    @triton.jit
    def _paged_attention_v2_kernel(
        # Pointers
        exp_sums_ptr,         # [num_seqs, num_heads, max_num_partitions]
        max_logits_ptr,       # [num_seqs, num_heads, max_num_partitions]
        tmp_output_ptr,       # [num_seqs, num_heads, max_num_partitions, head_dim]
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lens_ptr,
        # Sizes
        num_seqs,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        max_num_blocks_per_seq,
        partition_size: tl.constexpr,
        # Strides
        stride_qn,
        stride_qh,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_btn,
        stride_esn,
        stride_esh,
        stride_ton,
        stride_toh,
        stride_top,
        # Scale
        scale,
        # Block sizes
        BLOCK_DMODEL: tl.constexpr,
    ):
        """
        Paged Attention V2 - Partitioned computation for long sequences.
        
        For very long sequences, we partition the KV blocks and compute
        attention in parallel across partitions, then reduce.
        
        This enables:
        1. Better GPU utilization for long contexts
        2. Reduced memory bandwidth pressure
        3. Better scaling with sequence length
        """
        seq_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        partition_idx = tl.program_id(2)
        
        kv_head_idx = head_idx // (num_heads // num_kv_heads)
        
        context_len = tl.load(context_lens_ptr + seq_idx)
        num_blocks = (context_len + block_size - 1) // block_size
        
        # Calculate which blocks this partition handles
        blocks_per_partition = partition_size // block_size
        start_block = partition_idx * blocks_per_partition
        end_block = tl.minimum(start_block + blocks_per_partition, num_blocks)
        
        # Early exit if this partition has no blocks
        if start_block >= num_blocks:
            # Store sentinel values
            tl.store(exp_sums_ptr + seq_idx * stride_esn + head_idx * stride_esh + partition_idx, 0.0)
            tl.store(max_logits_ptr + seq_idx * stride_esn + head_idx * stride_esh + partition_idx, float("-inf"))
            return
        
        # Load query
        q_offset = seq_idx * stride_qn + head_idx * stride_qh
        q = tl.load(
            query_ptr + q_offset + tl.arange(0, BLOCK_DMODEL),
            mask=tl.arange(0, BLOCK_DMODEL) < head_dim,
            other=0.0
        )
        
        # Process blocks in this partition
        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
        
        for block_idx in range(start_block, end_block):
            block_table_offset = seq_idx * stride_btn + block_idx
            physical_block_num = tl.load(block_tables_ptr + block_table_offset)
            
            start_pos = block_idx * block_size
            k_base = physical_block_num * stride_kb + kv_head_idx * stride_kh
            
            for pos_in_block in range(block_size):
                token_pos = start_pos + pos_in_block
                if token_pos >= context_len:
                    continue
                    
                k_offset = k_base + pos_in_block * stride_ks
                k = tl.load(
                    key_cache_ptr + k_offset + tl.arange(0, BLOCK_DMODEL),
                    mask=tl.arange(0, BLOCK_DMODEL) < head_dim,
                    other=0.0
                )
                
                qk = tl.sum(q * k) * scale
                m_new = tl.maximum(m_i, qk)
                alpha = tl.exp(m_i - m_new)
                beta = tl.exp(qk - m_new)
                
                acc = acc * alpha
                
                v = tl.load(
                    value_cache_ptr + k_offset + tl.arange(0, BLOCK_DMODEL),
                    mask=tl.arange(0, BLOCK_DMODEL) < head_dim,
                    other=0.0
                )
                
                acc = acc + beta * v
                l_i = l_i * alpha + beta
                m_i = m_new
        
        # Store partition results for reduction
        tl.store(exp_sums_ptr + seq_idx * stride_esn + head_idx * stride_esh + partition_idx, l_i)
        tl.store(max_logits_ptr + seq_idx * stride_esn + head_idx * stride_esh + partition_idx, m_i)
        
        # Store unnormalized output
        tmp_out_offset = seq_idx * stride_ton + head_idx * stride_toh + partition_idx * stride_top
        tl.store(
            tmp_output_ptr + tmp_out_offset + tl.arange(0, BLOCK_DMODEL),
            acc.to(tmp_output_ptr.dtype.element_ty),
            mask=tl.arange(0, BLOCK_DMODEL) < head_dim
        )


    @triton.jit
    def _flash_attention_kernel(
        # Pointers
        output_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        # Sizes
        seq_len,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        # Strides
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        # Scale
        scale,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
    ):
        """
        Flash Attention Triton Kernel
        
        For prefill phase where we compute attention over the full context.
        Uses tiling to reduce memory bandwidth.
        """
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        start_m = tl.program_id(2)
        
        # Offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        
        # Pointers to query, key, value for this (batch, head)
        q_ptrs = (
            query_ptr + 
            batch_idx * stride_qb + 
            head_idx * stride_qh + 
            offs_m[:, None] * stride_qs + 
            offs_d[None, :] * stride_qd
        )
        k_ptrs = (
            key_ptr + 
            batch_idx * stride_kb + 
            head_idx * stride_kh + 
            offs_n[:, None] * stride_ks + 
            offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            value_ptr + 
            batch_idx * stride_vb + 
            head_idx * stride_vh + 
            offs_n[:, None] * stride_vs + 
            offs_d[None, :] * stride_vd
        )
        
        # Load query block
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
        
        # Initialize accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        
        # Iterate over key/value blocks
        for start_n in range(0, seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            
            # Load key and value blocks
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim), other=0.0)
            
            # Compute attention scores
            qk = tl.dot(q, tl.trans(k)) * scale
            
            # Causal mask (for autoregressive models)
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
            
            # Online softmax
            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            
            # Update accumulators
            l_new = alpha * l_i + tl.sum(p, 1)
            acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
            
            m_i = m_new
            l_i = l_new
            
            # Advance pointers
            k_ptrs += BLOCK_N * stride_ks
            v_ptrs += BLOCK_N * stride_vs
        
        # Final normalization
        acc = acc / l_i[:, None]
        
        # Store output
        o_ptrs = (
            output_ptr + 
            batch_idx * stride_ob + 
            head_idx * stride_oh + 
            offs_m[:, None] * stride_os + 
            offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim))


def paged_attention_v1(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int,
) -> None:
    """
    Paged Attention V1 - Basic paged attention for decode phase.
    
    Args:
        output: Output tensor [num_seqs, num_heads, head_dim]
        query: Query tensor [num_seqs, num_heads, head_dim]
        key_cache: Key cache [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Value cache [num_blocks, num_kv_heads, block_size, head_dim]
        block_tables: Block table [num_seqs, max_blocks]
        context_lens: Context lengths [num_seqs]
        scale: Attention scale (1/sqrt(head_dim))
        block_size: KV cache block size
    """
    check_triton_available()
    
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    max_num_blocks_per_seq = block_tables.shape[1]
    
    # Launch kernel
    grid = (num_seqs, num_heads)
    
    _paged_attention_v1_kernel[grid](
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        num_seqs,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        max_num_blocks_per_seq,
        # Query strides
        query.stride(0),
        query.stride(1),
        # Key cache strides
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        # Block table stride
        block_tables.stride(0),
        # Scale
        scale,
        # Block sizes
        BLOCK_M=16,
        BLOCK_DMODEL=head_dim,
    )


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Flash Attention for prefill phase.
    
    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, num_heads, seq_len, head_dim]
        value: [batch, num_heads, seq_len, head_dim]
        scale: Attention scale (default: 1/sqrt(head_dim))
    
    Returns:
        Output tensor [batch, num_heads, seq_len, head_dim]
    """
    check_triton_available()
    
    batch, num_heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    output = torch.empty_like(query)
    
    # Block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim
    
    # Grid
    num_m_blocks = (seq_len + BLOCK_M - 1) // BLOCK_M
    grid = (batch, num_heads, num_m_blocks)
    
    _flash_attention_kernel[grid](
        output,
        query,
        key,
        value,
        seq_len,
        num_heads,
        head_dim,
        # Query strides
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        # Key strides
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        # Value strides
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return output


# Fallback implementations for when Triton is not available
def paged_attention_v1_torch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    block_size: int,
) -> None:
    """
    PyTorch fallback for paged attention (slow, for testing only).
    """
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    
    # GQA factor
    gqa_factor = num_heads // num_kv_heads
    
    for seq_idx in range(num_seqs):
        context_len = context_lens[seq_idx].item()
        num_blocks = (context_len + block_size - 1) // block_size
        
        for head_idx in range(num_heads):
            kv_head_idx = head_idx // gqa_factor
            q = query[seq_idx, head_idx]  # [head_dim]
            
            # Gather keys and values from blocks
            keys = []
            values = []
            
            for block_idx in range(num_blocks):
                physical_block = block_tables[seq_idx, block_idx].item()
                start_pos = block_idx * block_size
                end_pos = min(start_pos + block_size, context_len)
                num_tokens = end_pos - start_pos
                
                keys.append(key_cache[physical_block, kv_head_idx, :num_tokens])
                values.append(value_cache[physical_block, kv_head_idx, :num_tokens])
            
            if keys:
                k = torch.cat(keys, dim=0)  # [context_len, head_dim]
                v = torch.cat(values, dim=0)  # [context_len, head_dim]
                
                # Attention: softmax(q @ k^T / sqrt(d)) @ v
                attn_weights = torch.softmax(q @ k.T * scale, dim=-1)
                output[seq_idx, head_idx] = attn_weights @ v


def flash_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    PyTorch fallback for flash attention (slow, for testing only).
    """
    batch, num_heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Standard attention with causal mask
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Causal mask
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
        diagonal=1
    )
    attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
    
    attn_weights = torch.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, value)
    
    return output
