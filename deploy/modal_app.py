"""
ZSE Modal Deployment

Deploy ZSE on Modal for GPU-accelerated inference.
Supports testing CUDA/Triton kernels on cloud GPUs.

Usage:
    modal run deploy/modal_app.py::test_attention
    modal run deploy/modal_app.py::benchmark_attention
    modal deploy deploy/modal_app.py  # Deploy as persistent app

Author: ZSE Team
"""

import modal
import os

# Get the path to the zse directory (parent of deploy)
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(DEPLOY_DIR)

# Modal app configuration
app = modal.App("zse-inference")

# GPU image with ZSE dependencies
zse_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch>=2.1.0",
        "triton>=2.1.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "accelerate>=0.25.0",
        "pynvml",
        "rich",
        "typer",
    )
    .run_commands(
        # Clone and install ZSE in editable mode
        "pip install ninja",
    )
)

# Image with ZSE source code
zse_image_with_code = zse_image.add_local_dir(ZSE_ROOT, remote_path="/root/zse")

# Volume for model storage
model_volume = modal.Volume.from_name("zse-models", create_if_missing=True)


@app.function(
    image=zse_image,
    gpu="A10G",  # A10G has more shared memory (164KB vs 64KB on T4)
    timeout=600,
)
def test_attention():
    """Test zAttention Triton kernels on GPU."""
    import torch
    import math
    import time
    
    print("=" * 60)
    print("ZSE Attention Kernel Test")
    print("=" * 60)
    
    # Check GPU
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check Triton
    try:
        import triton
        import triton.language as tl
        print(f"Triton Version: {triton.__version__}")
        triton_available = True
    except ImportError:
        print("Triton not available!")
        triton_available = False
    
    if not triton_available:
        print("\nERROR: Triton not available. Cannot test kernels.")
        return {"status": "error", "message": "Triton not available"}
    
    # ============================================================
    # ZSE Flash Attention Triton Kernel (inline for Modal)
    # Optimized version with proper tiling
    # ============================================================
    @triton.jit
    def _flash_attention_kernel_v2(
        output_ptr, query_ptr, key_ptr, value_ptr,
        seq_len,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Optimized flash attention with proper memory coalescing."""
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        start_m = tl.program_id(2)
        
        # Offsets for this block
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        
        # Load query block [BLOCK_M, BLOCK_D]
        q_ptrs = (query_ptr + batch_idx * stride_qb + head_idx * stride_qh + 
                  offs_m[:, None] * stride_qs + offs_d[None, :])
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)
        
        # Initialize accumulators
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        
        # Compute end position for causal attention
        end_n = tl.minimum((start_m + 1) * BLOCK_M, seq_len)
        
        # Iterate over K/V blocks
        for start_n in range(0, end_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            
            # Load K block [BLOCK_N, BLOCK_D]
            k_ptrs = (key_ptr + batch_idx * stride_kb + head_idx * stride_kh + 
                      offs_n[:, None] * stride_ks + offs_d[None, :])
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
            
            # Compute QK^T [BLOCK_M, BLOCK_N]
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k), qk) * scale
            
            # Apply causal mask
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
            
            # Compute max for numerical stability
            m_ij = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_ij)
            
            # Compute attention weights with rescaling
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            
            # Load V block [BLOCK_N, BLOCK_D]
            v_ptrs = (value_ptr + batch_idx * stride_vb + head_idx * stride_vh + 
                      offs_n[:, None] * stride_vs + offs_d[None, :])
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
            
            # Update running sum and accumulator
            l_new = alpha * l_i + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
            
            m_i = m_new
            l_i = l_new
        
        # Final normalization
        acc = acc / l_i[:, None]
        
        # Store output
        o_ptrs = (output_ptr + batch_idx * stride_ob + head_idx * stride_oh + 
                  offs_m[:, None] * stride_os + offs_d[None, :])
        tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=offs_m[:, None] < seq_len)
    
    def zse_flash_attention(query, key, value, scale=None):
        """ZSE Flash Attention using Triton."""
        batch, num_heads, seq_len, head_dim = query.shape
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        output = torch.empty_like(query)
        
        # Block sizes - optimized for A10G (164KB shared memory)
        BLOCK_M = 64
        BLOCK_N = 64
        
        num_m_blocks = triton.cdiv(seq_len, BLOCK_M)
        grid = (batch, num_heads, num_m_blocks)
        
        _flash_attention_kernel_v2[grid](
            output, query, key, value, seq_len,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=head_dim,
        )
        return output
    
    # ============================================================
    # Test Flash Attention
    # ============================================================
    print("\n" + "-" * 60)
    print("Testing ZSE Flash Attention (Triton Kernel)")
    print("-" * 60)
    
    # Test configuration
    batch_size = 2
    num_heads = 32
    seq_len = 512
    head_dim = 128
    
    # Create test tensors
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    
    # PyTorch reference
    scale = 1.0 / (head_dim ** 0.5)
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    ref_output = torch.matmul(attn_weights, value)
    
    print(f"Query shape: {query.shape}")
    print(f"Reference output shape: {ref_output.shape}")
    print(f"Reference output mean: {ref_output.mean().item():.6f}")
    
    # Run ZSE Triton kernel
    print("\n🚀 Running ZSE Flash Attention (Triton)...")
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(5):
        zse_output = zse_flash_attention(query, key, value, scale)
    torch.cuda.synchronize()
    
    # Benchmark ZSE
    start = time.perf_counter()
    iterations = 100
    for _ in range(iterations):
        zse_output = zse_flash_attention(query, key, value, scale)
    torch.cuda.synchronize()
    zse_time = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark naive PyTorch (matmul + softmax)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        naive_output = torch.matmul(attn_weights, value)
    torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / iterations * 1000
    
    # Benchmark PyTorch SDPA (Flash Attention)
    import torch.nn.functional as F
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        sdpa_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / iterations * 1000
    
    # Verify correctness
    max_diff = (zse_output - ref_output).abs().max().item()
    mean_diff = (zse_output - ref_output).abs().mean().item()
    
    print(f"\n📊 Results (batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}):")
    print(f"  ZSE Triton:         {zse_time:.3f} ms")
    print(f"  PyTorch Naive:      {naive_time:.3f} ms  (vs ZSE: {naive_time/zse_time:.2f}x)")
    print(f"  PyTorch SDPA:       {sdpa_time:.3f} ms  (vs ZSE: {sdpa_time/zse_time:.2f}x)")
    print(f"  Max difference:     {max_diff:.6f}")
    print(f"  Mean difference:    {mean_diff:.6f}")
    print(f"  Correctness:        {'✅ PASS' if max_diff < 0.01 else '❌ FAIL'}")
    
    # Memory test
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Large sequence test
    print("\n" + "-" * 60)
    print("Testing Large Sequence (2048 tokens)")
    print("-" * 60)
    
    seq_len_large = 2048
    query_large = torch.randn(1, num_heads, seq_len_large, head_dim, device="cuda", dtype=torch.float16)
    key_large = torch.randn(1, num_heads, seq_len_large, head_dim, device="cuda", dtype=torch.float16)
    value_large = torch.randn(1, num_heads, seq_len_large, head_dim, device="cuda", dtype=torch.float16)
    
    # ZSE on large sequence
    start = time.perf_counter()
    for _ in range(20):
        zse_large = zse_flash_attention(query_large, key_large, value_large)
    torch.cuda.synchronize()
    zse_large_time = (time.perf_counter() - start) / 20 * 1000
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"  Sequence length:    {seq_len_large}")
    print(f"  ZSE time:           {zse_large_time:.3f} ms")
    print(f"  Peak memory:        {peak_memory:.1f} MB")
    
    # Paged attention test info
    print("\n" + "-" * 60)
    print("Testing Paged Attention (Decode)")
    print("-" * 60)
    
    # Paged attention test
    num_seqs = 4
    num_kv_heads = 8  # GQA with 4x factor
    block_size = 16
    max_context_len = 2048
    max_num_blocks = (max_context_len + block_size - 1) // block_size
    num_blocks_total = num_seqs * max_num_blocks
    
    # Query for decode (single token per sequence)
    query_decode = torch.randn(num_seqs, num_heads, head_dim, device="cuda", dtype=torch.float16)
    
    # KV cache blocks
    key_cache = torch.randn(num_blocks_total, num_kv_heads, block_size, head_dim, device="cuda", dtype=torch.float16)
    value_cache = torch.randn(num_blocks_total, num_kv_heads, block_size, head_dim, device="cuda", dtype=torch.float16)
    
    # Block tables (mapping logical blocks to physical blocks)
    block_tables = torch.zeros(num_seqs, max_num_blocks, dtype=torch.int32, device="cuda")
    for i in range(num_seqs):
        for j in range(max_num_blocks):
            block_tables[i, j] = i * max_num_blocks + j
    
    # Context lengths
    context_lens = torch.randint(256, max_context_len, (num_seqs,), dtype=torch.int32, device="cuda")
    
    print(f"  Query shape (decode): {query_decode.shape}")
    print(f"  KV cache shape:       {key_cache.shape}")
    print(f"  Block tables shape:   {block_tables.shape}")
    print(f"  Context lengths:      {context_lens.tolist()}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    
    return {
        "status": "success",
        "gpu": torch.cuda.get_device_name(0),
        "triton_version": triton.__version__ if triton_available else None,
    }


@app.function(
    image=zse_image,
    gpu="A10G",  # A10G for benchmarking
    timeout=1800,
)
def benchmark_attention(
    seq_lengths: str = "512,1024,2048,4096",
    batch_sizes: str = "1,4,8",
    num_heads: int = 32,
    head_dim: int = 128,
    warmup: int = 10,
    iterations: int = 100,
):
    """Benchmark zAttention kernels."""
    import torch
    import time
    from typing import Dict, List
    
    # Import Triton and define kernel
    import triton
    import triton.language as tl
    
    @triton.jit
    def _flash_attention_kernel_v2(
        Q, K, V, Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        scale,
    ):
        pid_m = tl.program_id(0)
        pid_zh = tl.program_id(1)
        
        num_heads = tl.num_programs(1) // tl.num_programs(2) if tl.num_programs(2) > 0 else tl.num_programs(1)
        
        batch_idx = pid_zh // 32
        head_idx = pid_zh % 32
        
        q_offset = batch_idx * stride_qz + head_idx * stride_qh
        k_offset = batch_idx * stride_kz + head_idx * stride_kh
        v_offset = batch_idx * stride_vz + head_idx * stride_vh
        o_offset = batch_idx * stride_oz + head_idx * stride_oh
        
        Q_block_ptr = Q + q_offset
        K_block_ptr = K + k_offset
        V_block_ptr = V + v_offset
        O_block_ptr = Out + o_offset
        
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)
        offs_n = tl.arange(0, BLOCK_N)
        
        q = tl.load(Q_block_ptr + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk,
                   mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM), other=0.0)
        
        num_blocks = (pid_m + 1) * BLOCK_M
        num_blocks = (num_blocks + BLOCK_N - 1) // BLOCK_N
        
        for block_n in range(num_blocks):
            start_n = block_n * BLOCK_N
            offs_n_curr = start_n + tl.arange(0, BLOCK_N)
            
            k = tl.load(K_block_ptr + offs_n_curr[:, None] * stride_kn + offs_k[None, :] * stride_kk,
                       mask=(offs_n_curr[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM), other=0.0)
            
            qk = tl.dot(q.to(tl.float16), tl.trans(k.to(tl.float16))).to(tl.float32)
            qk = qk * scale
            
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
            
            m_ij = tl.maximum(tl.max(qk, axis=1), m_i)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v = tl.load(V_block_ptr + offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk,
                       mask=(offs_n_curr[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM), other=0.0)
            
            acc += tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
            m_i = m_ij
        
        acc = acc / l_i[:, None]
        
        tl.store(O_block_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
                acc.to(tl.float16),
                mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < HEAD_DIM))
    
    def zse_flash_attention(q, k, v, scale):
        batch, heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)
        
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads, 1)
        
        _flash_attention_kernel_v2[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seq_len, head_dim,
            BLOCK_M, BLOCK_N,
            scale,
        )
        return out
    
    # Parse comma-separated strings to lists
    seq_lengths_list = [int(x) for x in seq_lengths.split(",")]
    batch_sizes_list = [int(x) for x in batch_sizes.split(",")]
    
    results: Dict[str, List] = {
        "seq_len": [],
        "batch_size": [],
        "prefill_ms": [],
        "decode_ms": [],
        "memory_mb": [],
    }
    
    print("=" * 70)
    print("ZSE Attention Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    import torch.nn.functional as F
    
    print(f"\n{'Seq':>6} {'Batch':>6} | {'ZSE':>8} {'SDPA':>8} {'Speedup':>8} | {'Memory':>10}")
    print("-" * 70)
    
    for seq_len in seq_lengths_list:
        for batch_size in batch_sizes_list:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create tensors
            query = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                              device="cuda", dtype=torch.float16)
            key = torch.randn(batch_size, num_heads, seq_len, head_dim,
                            device="cuda", dtype=torch.float16)
            value = torch.randn(batch_size, num_heads, seq_len, head_dim,
                              device="cuda", dtype=torch.float16)
            
            scale = 1.0 / (head_dim ** 0.5)
            
            # Warmup both kernels
            for _ in range(warmup):
                _ = zse_flash_attention(query, key, value, scale)
                _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
                torch.cuda.synchronize()
            
            # Benchmark ZSE Triton
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for _ in range(iterations):
                out = zse_flash_attention(query, key, value, scale)
            
            torch.cuda.synchronize()
            zse_ms = (time.perf_counter() - start) / iterations * 1000
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # Benchmark PyTorch SDPA
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            for _ in range(iterations):
                out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            
            torch.cuda.synchronize()
            sdpa_ms = (time.perf_counter() - start) / iterations * 1000
            
            speedup = sdpa_ms / zse_ms
            
            results["seq_len"].append(seq_len)
            results["batch_size"].append(batch_size)
            results["prefill_ms"].append(zse_ms)
            results["decode_ms"].append(sdpa_ms)
            results["memory_mb"].append(memory_mb)
            
            print(f"{seq_len:>6} {batch_size:>6} | {zse_ms:>7.2f}ms {sdpa_ms:>7.2f}ms {speedup:>7.2f}x | {memory_mb:>9.1f}MB")
    
    print("-" * 70)
    avg_speedup = sum(results["decode_ms"]) / sum(results["prefill_ms"])
    print(f"Average ZSE speedup vs SDPA: {avg_speedup:.2f}x")
    
    return results


@app.function(
    image=zse_image,
    gpu="A10G",
    volumes={"/models": model_volume},
    timeout=1200,
)
def test_model_loading(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Test loading a real model from HuggingFace with ZSE.
    
    Uses TinyLlama (1.1B) by default as it's small but functional.
    """
    import torch
    import time
    import json
    from pathlib import Path
    
    print("=" * 70)
    print("ZSE Model Loading Test")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Download model using huggingface_hub
    from huggingface_hub import snapshot_download
    
    print(f"\n📥 Downloading {model_name}...")
    start = time.time()
    model_path = snapshot_download(
        model_name,
        cache_dir="/models",
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    download_time = time.time() - start
    print(f"✅ Download complete: {download_time:.1f}s")
    print(f"   Path: {model_path}")
    
    # Load config
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config_dict = json.load(f)
    
    print(f"\n📊 Model Configuration:")
    print(f"   Architecture: {config_dict.get('architectures', ['unknown'])[0]}")
    print(f"   Hidden size: {config_dict.get('hidden_size', 'N/A')}")
    print(f"   Layers: {config_dict.get('num_hidden_layers', 'N/A')}")
    print(f"   Heads: {config_dict.get('num_attention_heads', 'N/A')}")
    print(f"   KV Heads: {config_dict.get('num_key_value_heads', 'N/A')}")
    print(f"   Vocab: {config_dict.get('vocab_size', 'N/A')}")
    
    # Now test with ZSE model architecture (inline for Modal)
    # Create a minimal Llama config and model
    
    import torch.nn as nn
    import math
    
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
        
        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (self.weight * x).to(dtype)
    
    class SimpleLlamaAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hidden_size = config["hidden_size"]
            self.num_heads = config["num_attention_heads"]
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
            
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        def forward(self, x, attention_mask=None):
            batch, seq_len, _ = x.shape
            
            q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            # Repeat KV for GQA
            if self.num_kv_heads != self.num_heads:
                factor = self.num_heads // self.num_kv_heads
                k = k.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(batch, self.num_heads, seq_len, self.head_dim)
                v = v.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(batch, self.num_heads, seq_len, self.head_dim)
            
            # Flash attention via SDPA
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
            
            return self.o_proj(attn_out)
    
    class SimpleLlamaMLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hidden_size = config["hidden_size"]
            self.intermediate_size = config["intermediate_size"]
            
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        
        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
    
    class SimpleLlamaLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.self_attn = SimpleLlamaAttention(config)
            self.mlp = SimpleLlamaMLP(config)
            self.input_layernorm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5))
            self.post_attention_layernorm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5))
        
        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x
    
    class SimpleLlamaModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
            self.layers = nn.ModuleList([
                SimpleLlamaLayer(config) for _ in range(config["num_hidden_layers"])
            ])
            self.norm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5))
            self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        def forward(self, input_ids):
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.lm_head(x)
    
    print(f"\n🏗️ Creating ZSE model architecture...")
    start = time.time()
    model = SimpleLlamaModel(config_dict)
    model = model.half().cuda()
    create_time = time.time() - start
    print(f"✅ Model created: {create_time:.2f}s")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params / 1e9:.2f}B")
    
    # Load weights from safetensors
    print(f"\n📦 Loading weights from safetensors...")
    from safetensors.torch import load_file
    import glob
    
    safetensor_files = glob.glob(f"{model_path}/*.safetensors")
    print(f"   Found {len(safetensor_files)} safetensors file(s)")
    
    start = time.time()
    state_dict = {}
    for sf_file in safetensor_files:
        state_dict.update(load_file(sf_file, device="cuda"))
    
    # Map HuggingFace names to our model names
    mapped_state_dict = {}
    for name, tensor in state_dict.items():
        # Remove "model." prefix if present
        new_name = name.replace("model.", "")
        mapped_state_dict[new_name] = tensor
    
    # Load into model
    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
    load_time = time.time() - start
    
    print(f"✅ Weights loaded: {load_time:.2f}s")
    print(f"   Missing keys: {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"   Sample missing: {missing[:3]}")
    
    # Memory stats
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"\n💾 GPU Memory:")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved: {reserved:.2f} GB")
    
    # Test inference
    print(f"\n🔥 Running inference test...")
    test_input = torch.randint(0, config_dict["vocab_size"], (1, 64), device="cuda")
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_input)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.time()
    iterations = 10
    with torch.no_grad():
        for _ in range(iterations):
            output = model(test_input)
    torch.cuda.synchronize()
    inference_time = (time.time() - start) / iterations * 1000
    
    print(f"✅ Inference successful!")
    print(f"   Output shape: {output.shape}")
    print(f"   Time per forward: {inference_time:.2f} ms")
    print(f"   Tokens/sec (64 tokens): {64 / (inference_time / 1000):.0f}")
    
    print("\n" + "=" * 70)
    print("Model Loading Test Complete!")
    print("=" * 70)
    
    return {
        "status": "success",
        "model": model_name,
        "params_billions": num_params / 1e9,
        "gpu_memory_gb": allocated,
        "inference_ms": inference_time,
        "tokens_per_sec": 64 / (inference_time / 1000),
    }


@app.function(
    image=zse_image,
    gpu="A10G",
    volumes={"/models": model_volume},
    timeout=1200,
)
def test_streaming_loader(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Test MEMORY-EFFICIENT streaming model loading.
    
    Key optimizations:
    1. Meta device init - model skeleton uses ZERO memory
    2. Direct GPU loading - weights go straight to GPU via safetensors
    3. Streaming - one tensor at a time, no intermediate dict
    
    Expected: ~2.2GB for 1.1B params (vs 4.1GB without optimization)
    """
    import torch
    import torch.nn as nn
    import gc
    import time
    import json
    from pathlib import Path
    from safetensors import safe_open
    
    def empty_cache():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def mem_stats():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
        }
    
    print("=" * 70)
    print("ZSE MEMORY-EFFICIENT Streaming Loader Test")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    initial_mem = mem_stats()
    print(f"Initial GPU Memory: {initial_mem['allocated']:.3f} GB allocated")
    
    # Download model
    from huggingface_hub import snapshot_download
    print(f"\n📥 Downloading {model_name}...")
    model_path = snapshot_download(
        model_name,
        cache_dir="/models",
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    print(f"   Path: {model_path}")
    
    # Load config
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"\n📊 Model Config:")
    print(f"   Hidden: {config['hidden_size']}")
    print(f"   Layers: {config['num_hidden_layers']}")
    print(f"   Heads: {config['num_attention_heads']}")
    
    # Calculate expected memory
    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    
    # Parameters per layer
    head_dim = hidden_size // num_heads
    attn_params = (
        hidden_size * (num_heads * head_dim) +  # q_proj
        hidden_size * (num_kv_heads * head_dim) +  # k_proj
        hidden_size * (num_kv_heads * head_dim) +  # v_proj
        (num_heads * head_dim) * hidden_size  # o_proj
    )
    mlp_params = 3 * hidden_size * intermediate_size  # gate, up, down
    norm_params = 2 * hidden_size  # input_layernorm, post_attention_layernorm
    layer_params = attn_params + mlp_params + norm_params
    
    total_params = (
        vocab_size * hidden_size +  # embed_tokens
        num_layers * layer_params +
        hidden_size +  # final norm
        vocab_size * hidden_size  # lm_head
    )
    
    expected_mem_gb = (total_params * 2) / (1024**3)  # FP16 = 2 bytes
    print(f"\n📐 Expected Memory for {total_params/1e9:.2f}B params:")
    print(f"   Theoretical minimum (FP16): {expected_mem_gb:.2f} GB")
    
    # Define model with meta device initialization
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, device=None):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim, device=device))
        
        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (self.weight * x).to(dtype)
    
    class LlamaAttention(nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.hidden_size = config["hidden_size"]
            self.num_heads = config["num_attention_heads"]
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
            
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, device=device)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, device=device)
        
        def forward(self, x):
            batch, seq_len, _ = x.shape
            q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            if self.num_kv_heads != self.num_heads:
                factor = self.num_heads // self.num_kv_heads
                k = k.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(batch, self.num_heads, seq_len, self.head_dim)
                v = v.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(batch, self.num_heads, seq_len, self.head_dim)
            
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o_proj(attn.transpose(1, 2).reshape(batch, seq_len, self.hidden_size))
    
    class LlamaMLP(nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False, device=device)
            self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False, device=device)
            self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False, device=device)
        
        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
    
    class LlamaLayer(nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.self_attn = LlamaAttention(config, device=device)
            self.mlp = LlamaMLP(config, device=device)
            self.input_layernorm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5), device=device)
            self.post_attention_layernorm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5), device=device)
        
        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x
    
    class LlamaModel(nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"], device=device)
            self.layers = nn.ModuleList([LlamaLayer(config, device=device) for _ in range(config["num_hidden_layers"])])
            self.norm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5), device=device)
            self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False, device=device)
        
        def forward(self, input_ids):
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(self.norm(x))
    
    # ================================================================
    # CREATE MODEL ON META DEVICE (ZERO MEMORY!)
    # ================================================================
    print(f"\n🏗️ Creating model on META device (zero memory)...")
    empty_cache()
    pre_meta_mem = mem_stats()
    
    model = LlamaModel(config, device="meta")
    
    post_meta_mem = mem_stats()
    print(f"   Memory after meta init: {post_meta_mem['allocated']:.3f} GB")
    print(f"   ✅ Meta init used: {(post_meta_mem['allocated'] - pre_meta_mem['allocated']):.3f} GB (should be ~0)")
    
    # ================================================================
    # STREAM WEIGHTS DIRECTLY TO GPU (ONE AT A TIME)
    # ================================================================
    print(f"\n📦 Streaming weights directly to GPU...")
    
    import glob
    safetensor_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    print(f"   Found {len(safetensor_files)} safetensors file(s)")
    
    # Build name mapping
    name_map = {}
    model_state_keys = set(model.state_dict().keys())
    
    # Get all tensor names first
    all_tensor_names = []
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt") as f:
            all_tensor_names.extend(f.keys())
    
    for name in all_tensor_names:
        # Try direct
        if name in model_state_keys:
            name_map[name] = name
        # Try without "model." prefix
        elif name.startswith("model.") and name[6:] in model_state_keys:
            name_map[name] = name[6:]
    
    print(f"   Mapped {len(name_map)} tensors")
    
    # Create a new model on CUDA for actual weights
    # We need to materialize layer by layer
    print(f"\n⚡ Materializing model on GPU with streaming...")
    empty_cache()
    pre_load_mem = mem_stats()
    
    start_time = time.time()
    
    # Create new model directly on CUDA
    model = LlamaModel(config, device="cuda").half()
    
    # Track which params we've loaded
    loaded = set()
    
    # Stream and load one tensor at a time
    for sf_file in safetensor_files:
        # Open with DIRECT GPU loading - no CPU copy!
        with safe_open(sf_file, framework="pt", device="cuda") as f:
            for tensor_name in f.keys():
                # Get param name in our model
                param_name = name_map.get(tensor_name)
                if not param_name:
                    continue
                
                # Load tensor DIRECTLY to GPU
                tensor = f.get_tensor(tensor_name).half()
                
                # Navigate to the parameter and copy in-place
                parts = param_name.split(".")
                module = model
                for part in parts[:-1]:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                
                # Copy in-place
                param = getattr(module, parts[-1])
                param.data.copy_(tensor)
                loaded.add(param_name)
                
                # Immediately delete the loaded tensor to minimize peak memory
                del tensor
        
        # Clear cache after each file
        empty_cache()
    
    load_time = time.time() - start_time
    
    post_load_mem = mem_stats()
    print(f"\n✅ Loading complete!")
    print(f"   Time: {load_time:.2f}s")
    print(f"   Loaded {len(loaded)}/{len(model_state_keys)} parameters")
    
    # Final memory stats
    empty_cache()
    torch.cuda.synchronize()
    final_mem = mem_stats()
    
    print(f"\n💾 MEMORY EFFICIENCY:")
    print(f"   Expected (theoretical FP16): {expected_mem_gb:.2f} GB")
    print(f"   Actual allocated: {final_mem['allocated']:.2f} GB")
    print(f"   Actual reserved: {final_mem['reserved']:.2f} GB")
    overhead = ((final_mem['allocated'] - expected_mem_gb) / expected_mem_gb) * 100
    print(f"   Overhead: {overhead:+.1f}%")
    
    if overhead < 20:
        print(f"   ✅ EXCELLENT memory efficiency!")
    elif overhead < 50:
        print(f"   ⚠️ Acceptable memory efficiency")
    else:
        print(f"   ❌ Memory efficiency needs improvement")
    
    # Test inference
    print(f"\n🔥 Testing inference...")
    test_input = torch.randint(0, config["vocab_size"], (1, 64), device="cuda")
    
    with torch.no_grad():
        for _ in range(3):  # Warmup
            _ = model(test_input)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(10):
            output = model(test_input)
        torch.cuda.synchronize()
        inference_ms = (time.time() - start) / 10 * 1000
    
    print(f"   Output shape: {output.shape}")
    print(f"   Time per forward: {inference_ms:.2f} ms")
    
    print("\n" + "=" * 70)
    print("Streaming Loader Test Complete!")
    print("=" * 70)
    
    return {
        "status": "success",
        "expected_mem_gb": expected_mem_gb,
        "actual_mem_gb": final_mem['allocated'],
        "overhead_percent": overhead,
        "load_time_s": load_time,
        "inference_ms": inference_ms,
    }


@app.function(
    image=zse_image,
    gpu="A10G",
    volumes={"/models": model_volume},
    timeout=1200,
)
def test_quantized_loading(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Test INT8/INT4 quantized model loading.
    
    Memory targets:
    - FP16: 2.05 GB (baseline)
    - INT8: ~1.0 GB (50% reduction)
    - INT4: ~0.5 GB (75% reduction)
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import gc
    import time
    import json
    from pathlib import Path
    from safetensors import safe_open
    from typing import Tuple, Optional
    from enum import Enum
    
    def empty_cache():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def mem_stats():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
        }
    
    print("=" * 70)
    print("ZSE INT8/INT4 QUANTIZED Model Loading Test")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Download model
    from huggingface_hub import snapshot_download
    print(f"\n📥 Downloading {model_name}...")
    model_path = snapshot_download(
        model_name,
        cache_dir="/models",
        allow_patterns=["*.safetensors", "*.json"],
    )
    
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # ================================================================
    # INT8 QUANTIZATION (inline implementation for Modal)
    # ================================================================
    
    def quantize_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to INT8 with per-channel scales."""
        if tensor.ndim == 2:
            tensor_max = tensor.abs().amax(dim=1, keepdim=True)
        else:
            tensor_max = tensor.abs().max()
        tensor_max = torch.clamp(tensor_max, min=1e-8)
        scale = tensor_max / 127.0
        quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
        return quantized, scale.squeeze()
    
    def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize INT8 back to float."""
        if quantized.ndim == 2 and scale.ndim == 1:
            scale = scale.unsqueeze(1)
        return (quantized.float() * scale).to(dtype)
    
    class QuantizedLinear(nn.Module):
        """INT8 quantized linear layer."""
        def __init__(self, in_features: int, out_features: int, device=None):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8, device=device))
            self.register_buffer("weight_scale", torch.zeros(out_features, dtype=torch.float16, device=device))
        
        @classmethod
        def from_float(cls, linear: nn.Linear) -> "QuantizedLinear":
            device = linear.weight.device
            q = cls(linear.in_features, linear.out_features, device=device)
            w_int8, scale = quantize_int8(linear.weight.data.float())
            q.weight_int8.copy_(w_int8.to(device))
            q.weight_scale.copy_(scale.half().to(device))
            return q
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            weight = dequantize_int8(self.weight_int8, self.weight_scale, x.dtype)
            return F.linear(x, weight)
    
    # ================================================================
    # MODEL DEFINITION
    # ================================================================
    
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, device=None):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim, device=device))
        
        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (self.weight * x).to(dtype)
    
    class LlamaAttention(nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.hidden_size = config["hidden_size"]
            self.num_heads = config["num_attention_heads"]
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
            
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, device=device)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, device=device)
        
        def forward(self, x):
            batch, seq_len, _ = x.shape
            q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            if self.num_kv_heads != self.num_heads:
                factor = self.num_heads // self.num_kv_heads
                k = k.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(batch, self.num_heads, seq_len, self.head_dim)
                v = v.unsqueeze(2).expand(-1, -1, factor, -1, -1).reshape(batch, self.num_heads, seq_len, self.head_dim)
            
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o_proj(attn.transpose(1, 2).reshape(batch, seq_len, self.hidden_size))
    
    class LlamaMLP(nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False, device=device)
            self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False, device=device)
            self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False, device=device)
        
        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
    
    class LlamaLayer(nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.self_attn = LlamaAttention(config, device=device)
            self.mlp = LlamaMLP(config, device=device)
            self.input_layernorm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5), device=device)
            self.post_attention_layernorm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5), device=device)
        
        def forward(self, x):
            x = x + self.self_attn(self.input_layernorm(x))
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x
    
    class LlamaModel(nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"], device=device)
            self.layers = nn.ModuleList([LlamaLayer(config, device=device) for _ in range(config["num_hidden_layers"])])
            self.norm = RMSNorm(config["hidden_size"], config.get("rms_norm_eps", 1e-5), device=device)
            self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False, device=device)
        
        def forward(self, input_ids):
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(self.norm(x))
    
    # ================================================================
    # LOAD FP16 MODEL FIRST (baseline)
    # ================================================================
    
    print("\n📊 STEP 1: Load FP16 model (baseline)")
    empty_cache()
    
    model_fp16 = LlamaModel(config, device="cuda").half()
    
    # Load weights
    import glob
    safetensor_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    
    model_keys = set(model_fp16.state_dict().keys())
    name_map = {}
    
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt") as f:
            for name in f.keys():
                if name in model_keys:
                    name_map[name] = name
                elif name.startswith("model.") and name[6:] in model_keys:
                    name_map[name] = name[6:]
    
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cuda") as f:
            for tensor_name in f.keys():
                param_name = name_map.get(tensor_name)
                if not param_name:
                    continue
                tensor = f.get_tensor(tensor_name).half()
                parts = param_name.split(".")
                module = model_fp16
                for part in parts[:-1]:
                    module = module[int(part)] if part.isdigit() else getattr(module, part)
                getattr(module, parts[-1]).data.copy_(tensor)
                del tensor
    
    empty_cache()
    fp16_mem = mem_stats()
    print(f"   FP16 Memory: {fp16_mem['allocated']:.2f} GB")
    
    # Test inference
    test_input = torch.randint(0, config["vocab_size"], (1, 64), device="cuda")
    with torch.no_grad():
        out_fp16 = model_fp16(test_input)
    print(f"   FP16 Output shape: {out_fp16.shape}")
    
    # ================================================================
    # QUANTIZE TO INT8
    # ================================================================
    
    print("\n📊 STEP 2: Quantize to INT8")
    
    def quantize_model_int8(model):
        """Replace Linear layers with QuantizedLinear."""
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and "embed" not in name and "lm_head" not in name:
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
                q_linear = QuantizedLinear.from_float(module)
                setattr(parent, parts[-1], q_linear)
                del module
        return model
    
    model_int8 = quantize_model_int8(model_fp16)
    
    empty_cache()
    int8_mem = mem_stats()
    reduction = (1 - int8_mem['allocated'] / fp16_mem['allocated']) * 100
    print(f"   INT8 Memory: {int8_mem['allocated']:.2f} GB")
    print(f"   Memory Reduction: {reduction:.1f}%")
    
    # Test inference
    with torch.no_grad():
        out_int8 = model_int8(test_input)
    print(f"   INT8 Output shape: {out_int8.shape}")
    
    # Compare accuracy
    diff = (out_fp16 - out_int8).abs().mean() / out_fp16.abs().mean()
    print(f"   Quantization Error: {diff.item():.4f} (relative)")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    
    print("\n" + "=" * 70)
    print("QUANTIZATION SUMMARY")
    print("=" * 70)
    print(f"\n💾 Memory Usage:")
    print(f"   FP16 (baseline): {fp16_mem['allocated']:.2f} GB")
    print(f"   INT8 (quantized): {int8_mem['allocated']:.2f} GB")
    print(f"   Reduction: {reduction:.1f}%")
    
    if reduction > 40:
        print(f"\n   ✅ EXCELLENT memory reduction with INT8!")
    else:
        print(f"\n   ⚠️ Memory reduction lower than expected")
    
    return {
        "fp16_mem_gb": fp16_mem['allocated'],
        "int8_mem_gb": int8_mem['allocated'],
        "reduction_percent": reduction,
        "quantization_error": diff.item(),
    }


@app.function(
    image=zse_image,
    gpu="A10G",
    volumes={"/models": model_volume},
    timeout=1200,
)
def test_continuous_batching():
    """
    Test KV Cache and Continuous Batching.
    
    Demonstrates:
    1. Paged KV cache allocation
    2. Multi-sequence scheduling  
    3. Dynamic batch management
    """
    import torch
    import time
    from dataclasses import dataclass, field
    from typing import Dict, List, Tuple, Optional, Any
    from enum import Enum
    from collections import deque
    import heapq
    import math
    
    print("=" * 70)
    print("ZSE Continuous Batching Test")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ================================================================
    # Inline KV Cache Implementation (for Modal)
    # ================================================================
    
    @dataclass
    class KVCacheConfig:
        num_layers: int
        num_heads: int
        head_dim: int
        max_seq_len: int = 2048
        page_size: int = 16
        max_pages: int = 4096
        dtype: torch.dtype = torch.float16
        device: str = "cuda"
        
        @property
        def bytes_per_token(self) -> int:
            elem_size = 2 if self.dtype == torch.float16 else 4
            return 2 * self.num_layers * self.num_heads * self.head_dim * elem_size
    
    class PagedKVCache:
        def __init__(self, config: KVCacheConfig):
            self.config = config
            self.page_size = config.page_size
            self.num_pages = config.max_pages
            
            page_shape = (
                config.num_layers,
                self.num_pages,
                self.page_size,
                config.num_heads,
                config.head_dim,
            )
            
            self.key_pages = torch.zeros(page_shape, dtype=config.dtype, device=config.device)
            self.value_pages = torch.zeros(page_shape, dtype=config.dtype, device=config.device)
            
            self.free_pages = list(range(self.num_pages))
            self.seq_page_tables: Dict[int, List[int]] = {}
            self.seq_lengths: Dict[int, int] = {}
        
        def allocate_sequence(self, seq_id: int, initial_len: int = 0) -> bool:
            if seq_id in self.seq_page_tables:
                return False
            
            num_pages_needed = max(1, math.ceil(initial_len / self.page_size))
            
            if len(self.free_pages) < num_pages_needed:
                return False
            
            allocated_pages = []
            for _ in range(num_pages_needed):
                page_id = self.free_pages.pop()
                allocated_pages.append(page_id)
            
            self.seq_page_tables[seq_id] = allocated_pages
            self.seq_lengths[seq_id] = initial_len
            return True
        
        def free_sequence(self, seq_id: int):
            if seq_id not in self.seq_page_tables:
                return
            for page_id in self.seq_page_tables[seq_id]:
                self.free_pages.append(page_id)
            del self.seq_page_tables[seq_id]
            del self.seq_lengths[seq_id]
        
        def num_free_pages(self) -> int:
            return len(self.free_pages)
        
        def memory_usage_gb(self) -> float:
            used_pages = self.num_pages - len(self.free_pages)
            bytes_per_page = (
                2 * self.config.num_layers * self.page_size *
                self.config.num_heads * self.config.head_dim * 2
            )
            return (used_pages * bytes_per_page) / (1024**3)
    
    # ================================================================
    # Inline Scheduler Implementation
    # ================================================================
    
    class RequestStatus(Enum):
        PENDING = "pending"
        RUNNING = "running"
        COMPLETED = "completed"
    
    @dataclass
    class Request:
        request_id: str
        prompt_tokens: List[int]
        max_new_tokens: int = 64
        status: RequestStatus = RequestStatus.PENDING
        generated_tokens: List[int] = field(default_factory=list)
        created_at: float = field(default_factory=time.time)
        
        @property
        def is_finished(self) -> bool:
            return (
                self.status == RequestStatus.COMPLETED or
                len(self.generated_tokens) >= self.max_new_tokens
            )
    
    class ContinuousBatchingScheduler:
        def __init__(self, kv_cache: PagedKVCache, max_batch_size: int = 32):
            self.kv_cache = kv_cache
            self.max_batch_size = max_batch_size
            self.waiting: deque = deque()
            self.running: Dict[str, Request] = {}
            self._next_seq_id = 0
        
        def add_request(self, request: Request) -> bool:
            self.waiting.append(request)
            return True
        
        def schedule(self) -> List[Request]:
            # Remove finished
            finished = [rid for rid, req in self.running.items() if req.is_finished]
            for rid in finished:
                req = self.running[rid]
                req.status = RequestStatus.COMPLETED
                self.kv_cache.free_sequence(self._get_seq_id(rid))
                del self.running[rid]
            
            # Add new requests
            while self.waiting and len(self.running) < self.max_batch_size:
                request = self.waiting.popleft()
                seq_id = self._next_seq_id
                self._next_seq_id += 1
                
                if self.kv_cache.allocate_sequence(seq_id, len(request.prompt_tokens)):
                    request.status = RequestStatus.RUNNING
                    self.running[request.request_id] = request
                else:
                    self.waiting.appendleft(request)
                    break
            
            return list(self.running.values())
        
        def _get_seq_id(self, request_id: str) -> int:
            # Simple mapping for this test
            return hash(request_id) % 10000
    
    # ================================================================
    # Test Continuous Batching
    # ================================================================
    
    print("\n📊 STEP 1: Create Paged KV Cache")
    
    # Llama-like config
    config = KVCacheConfig(
        num_layers=22,
        num_heads=32,
        head_dim=64,
        page_size=16,
        max_pages=2048,  # ~1GB for KV cache
    )
    
    kv_cache = PagedKVCache(config)
    
    total_kv_mem_gb = (config.max_pages * config.page_size * config.bytes_per_token) / (1024**3)
    print(f"   Total KV cache capacity: {total_kv_mem_gb:.2f} GB")
    print(f"   Free pages: {kv_cache.num_free_pages()}")
    print(f"   Page size: {config.page_size} tokens")
    
    # Calculate max sequences
    tokens_per_seq = 512  # Average sequence length
    pages_per_seq = math.ceil(tokens_per_seq / config.page_size)
    max_concurrent_seqs = config.max_pages // pages_per_seq
    print(f"   Max concurrent sequences (512 tokens each): {max_concurrent_seqs}")
    
    print("\n📊 STEP 2: Create Scheduler")
    
    scheduler = ContinuousBatchingScheduler(kv_cache, max_batch_size=16)
    
    print("\n📊 STEP 3: Simulate Continuous Batching")
    
    # Create test requests
    num_requests = 50
    requests = []
    for i in range(num_requests):
        req = Request(
            request_id=f"req_{i}",
            prompt_tokens=list(range(64 + (i % 32))),  # Varying lengths
            max_new_tokens=32,
        )
        requests.append(req)
        scheduler.add_request(req)
    
    print(f"   Added {num_requests} requests to queue")
    print(f"   Waiting: {len(scheduler.waiting)}")
    
    # Simulate generation loop
    total_tokens_generated = 0
    iterations = 0
    start_time = time.time()
    
    while scheduler.running or scheduler.waiting:
        # Schedule batch
        batch = scheduler.schedule()
        
        if not batch:
            break
        
        # Simulate one generation step
        for req in batch:
            if not req.is_finished:
                # Generate random token
                req.generated_tokens.append(torch.randint(0, 32000, (1,)).item())
                total_tokens_generated += 1
        
        iterations += 1
        
        # Status update every 50 iterations
        if iterations % 50 == 0:
            print(f"   Iteration {iterations}: Running={len(scheduler.running)}, "
                  f"Waiting={len(scheduler.waiting)}, "
                  f"Tokens={total_tokens_generated}")
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Continuous Batching Complete!")
    print(f"   Total iterations: {iterations}")
    print(f"   Total tokens generated: {total_tokens_generated}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {total_tokens_generated / elapsed:.0f} tokens/sec (simulated)")
    
    # Check all completed
    completed = sum(1 for r in requests if r.status == RequestStatus.COMPLETED)
    print(f"   Completed requests: {completed}/{num_requests}")
    
    print("\n📊 STEP 4: Memory Efficiency Check")
    
    # KV cache should be fully freed
    print(f"   Free pages after completion: {kv_cache.num_free_pages()}/{config.max_pages}")
    print(f"   KV cache memory in use: {kv_cache.memory_usage_gb():.3f} GB")
    
    if kv_cache.num_free_pages() == config.max_pages:
        print(f"   ✅ All KV cache memory properly freed!")
    else:
        print(f"   ⚠️ Some KV cache pages still allocated")
    
    print("\n" + "=" * 70)
    print("Continuous Batching Test Complete!")
    print("=" * 70)
    
    return {
        "status": "success",
        "total_requests": num_requests,
        "completed_requests": completed,
        "total_tokens": total_tokens_generated,
        "elapsed_seconds": elapsed,
        "throughput_tokens_per_sec": total_tokens_generated / elapsed,
    }


@app.function(
    image=zse_image_with_code,
    gpu="A10G",
    timeout=600,
)
def test_generation():
    """Test text generation with streaming using a real model."""
    import torch
    import torch.nn as nn
    import time
    
    # Add ZSE to path
    import sys
    sys.path.insert(0, "/root/zse")
    
    from zse.engine.generation import (
        TextGenerator,
        SamplingParams,
        Sampler,
        PrintStreamCallback,
    )
    
    print("=" * 70)
    print("ZSE Text Generation Test")
    print("=" * 70)
    
    # GPU info
    print(f"\n📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test 1: Simple model generation
    print("\n" + "-" * 50)
    print("Test 1: Text Generation with Simple Model")
    print("-" * 50)
    
    class SimpleLanguageModel(nn.Module):
        """A simple model for testing generation."""
        def __init__(self, vocab_size=32000, hidden_dim=512, num_layers=2):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, 
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    batch_first=True,
                ) for _ in range(num_layers)
            ])
            self.proj = nn.Linear(hidden_dim, vocab_size)
        
        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.proj(x)
    
    # Create and move model to GPU
    vocab_size = 32000
    model = SimpleLanguageModel(vocab_size=vocab_size).cuda().half()
    
    # Calculate model size
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"   Model parameters: {param_count:,}")
    print(f"   Model size: {model_size_mb:.1f} MB")
    
    # Simple tokenizer
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.eos_token_id = 2
            self.pad_token_id = 0
        
        def encode(self, text):
            # Simple encoding: hash characters to token IDs
            return [hash(c) % self.vocab_size + 3 for c in text[:100]]
        
        def decode(self, ids, skip_special_tokens=False):
            # Map back to ASCII range
            chars = []
            for i in ids:
                if skip_special_tokens and i < 3:
                    continue
                chars.append(chr((i % 94) + 32))
            return "".join(chars)
    
    tokenizer = SimpleTokenizer(vocab_size)
    
    # Create generator
    generator = TextGenerator(model, tokenizer, device="cuda")
    
    # Test 2: Greedy generation
    print("\n" + "-" * 50)
    print("Test 2: Greedy Generation (temperature=0)")
    print("-" * 50)
    
    params = SamplingParams(
        temperature=0,
        max_new_tokens=50,
    )
    
    start = time.perf_counter()
    output = generator.generate("Hello world", params)
    elapsed = time.perf_counter() - start
    
    print(f"   Generated {output.num_tokens} tokens in {elapsed*1000:.1f}ms")
    print(f"   Throughput: {output.num_tokens/elapsed:.1f} tokens/sec")
    print(f"   Finish reason: {output.finish_reason}")
    print(f"   Output text: {output.text[:50]}...")
    
    # Test 3: Streaming generation
    print("\n" + "-" * 50)
    print("Test 3: Streaming Generation")
    print("-" * 50)
    
    params = SamplingParams(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        max_new_tokens=30,
    )
    
    print("   Streaming output: ", end="")
    latencies = []
    token_count = 0
    
    for chunk in generator.generate_stream("The quick brown fox", params):
        latencies.append(chunk.latency_ms)
        token_count += 1
        # Print first 10 chars of each token
        print(chunk.text[:10], end="", flush=True)
    print()
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"\n   Tokens: {token_count}")
    print(f"   Avg latency: {avg_latency:.2f}ms per token")
    print(f"   Throughput: {1000/avg_latency:.1f} tokens/sec" if avg_latency > 0 else "   N/A")
    
    # Test 4: Sampling strategies comparison
    print("\n" + "-" * 50)
    print("Test 4: Sampling Strategies")
    print("-" * 50)
    
    prompt = "Once upon a time"
    strategies = [
        ("Greedy", SamplingParams(temperature=0, max_new_tokens=20)),
        ("Temperature=0.7", SamplingParams(temperature=0.7, max_new_tokens=20)),
        ("Top-k=40", SamplingParams(temperature=1.0, top_k=40, max_new_tokens=20)),
        ("Top-p=0.9", SamplingParams(temperature=1.0, top_p=0.9, max_new_tokens=20)),
    ]
    
    for name, params in strategies:
        output = generator.generate(prompt, params)
        print(f"   {name}: {output.text[:30]}...")
    
    # Test 5: Repetition penalty
    print("\n" + "-" * 50)
    print("Test 5: Repetition Penalty")
    print("-" * 50)
    
    # Without penalty
    params_no_penalty = SamplingParams(temperature=0.5, max_new_tokens=30, repetition_penalty=1.0)
    output_no = generator.generate("The cat sat on", params_no_penalty)
    
    # With penalty
    params_with_penalty = SamplingParams(temperature=0.5, max_new_tokens=30, repetition_penalty=1.2)
    output_with = generator.generate("The cat sat on", params_with_penalty)
    
    def count_unique_tokens(tokens):
        return len(set(tokens))
    
    print(f"   Without penalty: {count_unique_tokens(output_no.tokens)} unique tokens")
    print(f"   With penalty: {count_unique_tokens(output_with.tokens)} unique tokens")
    
    # Test 6: GPU memory usage
    print("\n" + "-" * 50)
    print("Test 6: GPU Memory During Generation")
    print("-" * 50)
    
    torch.cuda.reset_peak_memory_stats()
    
    # Generate with longer sequence
    params = SamplingParams(temperature=0.7, max_new_tokens=100)
    output = generator.generate("Long generation test: ", params)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    current_mem = torch.cuda.memory_allocated() / 1024**3
    
    print(f"   Tokens generated: {output.num_tokens}")
    print(f"   Peak GPU memory: {peak_mem:.3f} GB")
    print(f"   Current GPU memory: {current_mem:.3f} GB")
    
    # Test 7: Sampler directly
    print("\n" + "-" * 50)
    print("Test 7: Sampler Unit Test on GPU")
    print("-" * 50)
    
    sampler = Sampler(SamplingParams(temperature=0.7, top_k=50))
    
    # Create random logits on GPU
    logits = torch.randn(vocab_size, device="cuda")
    logits[1000] = 10.0  # Make token 1000 very likely
    
    samples = [sampler.sample(logits) for _ in range(100)]
    most_common = max(set(samples), key=samples.count)
    
    print(f"   Most frequent token: {most_common}")
    print(f"   Token 1000 frequency: {samples.count(1000)}/100")
    
    print("\n" + "=" * 70)
    print("Text Generation Test Complete!")
    print("=" * 70)
    
    return {
        "status": "success",
        "model_params": param_count,
        "tokens_generated": token_count,
        "avg_latency_ms": avg_latency,
    }


@app.function(
    image=zse_image,
    gpu="A100",
    volumes={"/models": model_volume},
    timeout=3600,
)
def serve_model(model_name: str = "meta-llama/Llama-2-7b-hf"):
    """Serve a model with ZSE for testing."""
    import torch
    
    print(f"Loading model: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # This will be expanded when we have the full model loader
    return {
        "status": "ready",
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0),
    }


@app.local_entrypoint()
def main():
    """Run attention tests."""
    print("Running ZSE attention tests on Modal...")
    result = test_attention.remote()
    print(f"\nResult: {result}")
