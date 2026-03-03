#!/usr/bin/env python3
"""
ZSE Feature Benchmark v2 - Robust version
Tests each feature safely with proper error handling
"""

import os
import sys
import time
import json
import gc
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*70)
print("ZSE FEATURE BENCHMARK v2")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Started: {datetime.now()}")
print()

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_vram():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0

# ============================================================================
# TEST 1: Check what's available in zse package
# ============================================================================
print("\n" + "="*60)
print("TEST 1: Package Import Check")
print("="*60)

try:
    import zse
    print(f"  zse version: {getattr(zse, '__version__', 'unknown')}")
    print(f"  zse location: {zse.__file__}")
except ImportError as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Check submodules
modules_to_check = [
    "zse.efficiency.quantization",
    "zse.efficiency.triton_quant_kernels",
    "zse.core.zkv",
    "zse.core.zattention",
    "zse.core.zstream",
    "zse.format",
    "zse.engine.orchestrator",
]

for mod in modules_to_check:
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
    except ImportError as e:
        print(f"  [MISSING] {mod}: {e}")

# ============================================================================
# TEST 2: Quantization Functions
# ============================================================================
print("\n" + "="*60)
print("TEST 2: Quantization Functions")
print("="*60)

try:
    from zse.efficiency.quantization import (
        quantize_tensor_int8,
        dequantize_tensor_int8,
        quantize_tensor_int4,
        dequantize_tensor_int4,
        TRITON_AVAILABLE,
    )
    print(f"  TRITON_AVAILABLE: {TRITON_AVAILABLE}")
    
    # Test INT8
    x = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
    q, scale, zp = quantize_tensor_int8(x, per_channel=True, symmetric=True)
    reconstructed = dequantize_tensor_int8(q, scale, zp)
    error_int8 = (x - reconstructed).abs().mean().item()
    print(f"  INT8 quantize/dequantize: OK (MAE={error_int8:.6f})")
    
    # Test INT4
    q4, scale4 = quantize_tensor_int4(x, group_size=128)
    reconstructed4 = dequantize_tensor_int4(q4, scale4, group_size=128)
    error_int4 = (x - reconstructed4).abs().mean().item()
    print(f"  INT4 quantize/dequantize: OK (MAE={error_int4:.6f})")
    
    del x, q, scale, zp, reconstructed, q4, scale4, reconstructed4
    clear_gpu()
    
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 3: Fused Triton Kernels (INT8)
# ============================================================================
print("\n" + "="*60)
print("TEST 3: Fused INT8 Matmul Kernel")
print("="*60)

try:
    from zse.efficiency.triton_quant_kernels import int8_fused_matmul
    from zse.efficiency.quantization import quantize_tensor_int8, dequantize_tensor_int8
    
    M, K, N = 1024, 4096, 4096
    
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    W = torch.randn(N, K, dtype=torch.float32, device="cuda")
    
    # Quantize weights
    W_int8, scale, _ = quantize_tensor_int8(W, per_channel=True, symmetric=True)
    W_int8 = W_int8.cuda()
    scale = scale.cuda().half()
    
    # Test fused kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        out_fused = int8_fused_matmul(x, W_int8, scale)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 10 * 1000
    
    # Test unfused (baseline)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        W_fp16 = dequantize_tensor_int8(W_int8, scale)
        out_unfused = F.linear(x, W_fp16)
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / 10 * 1000
    
    print(f"  Fused:   {fused_time:.2f} ms")
    print(f"  Unfused: {unfused_time:.2f} ms")
    print(f"  Speedup: {unfused_time/fused_time:.2f}x")
    
    del x, W, W_int8, scale, out_fused, out_unfused
    clear_gpu()
    
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: Fused Triton Kernels (INT4)
# ============================================================================
print("\n" + "="*60)
print("TEST 4: Fused INT4 Matmul Kernel")
print("="*60)

try:
    from zse.efficiency.triton_quant_kernels import int4_fused_matmul
    from zse.efficiency.quantization import quantize_tensor_int4, dequantize_tensor_int4
    
    M, K, N = 1024, 4096, 4096
    group_size = 128
    
    x = torch.randn(M, K, dtype=torch.float16, device="cuda")
    W = torch.randn(N, K, dtype=torch.float32, device="cuda")
    
    # Quantize weights
    W_packed, scale = quantize_tensor_int4(W, group_size=group_size)
    W_packed = W_packed.cuda()
    scale = scale.cuda().half()
    
    # Test fused kernel
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        out_fused = int4_fused_matmul(x, W_packed, scale, group_size=group_size)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 10 * 1000
    
    # Test unfused
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        W_fp16 = dequantize_tensor_int4(W_packed, scale, group_size=group_size)
        out_unfused = F.linear(x, W_fp16)
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / 10 * 1000
    
    print(f"  Fused:   {fused_time:.2f} ms")
    print(f"  Unfused: {unfused_time:.2f} ms")
    print(f"  Speedup: {unfused_time/fused_time:.2f}x")
    
    del x, W, W_packed, scale, out_fused, out_unfused
    clear_gpu()
    
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 5: QuantizedLinear Layer
# ============================================================================
print("\n" + "="*60)
print("TEST 5: QuantizedLinear Layer")
print("="*60)

try:
    from zse.efficiency.quantization import QuantizedLinear, QuantType
    
    in_features, out_features = 4096, 4096
    
    # Create FP16 linear
    linear_fp16 = nn.Linear(in_features, out_features, bias=True).half().cuda()
    
    # Create INT8 quantized
    print("  Creating INT8 QuantizedLinear...")
    linear_int8 = QuantizedLinear.from_float(linear_fp16, QuantType.INT8)
    
    # Create INT4 quantized
    print("  Creating INT4 QuantizedLinear...")
    linear_int4 = QuantizedLinear.from_float(linear_fp16, QuantType.INT4, group_size=128)
    
    # Test input (smaller batch to be safe)
    x = torch.randn(1, 512, in_features, dtype=torch.float16, device="cuda")
    
    print("  Testing forward pass...")
    with torch.no_grad():
        out_fp16 = linear_fp16(x)
        out_int8 = linear_int8(x)
        out_int4 = linear_int4(x)
    
    # Memory
    fp16_mem = sum(p.numel() * 2 for p in linear_fp16.parameters()) / 1e6
    int8_mem = linear_int8.memory_bytes() / 1e6 if hasattr(linear_int8, 'memory_bytes') else -1
    int4_mem = linear_int4.memory_bytes() / 1e6 if hasattr(linear_int4, 'memory_bytes') else -1
    
    print(f"  FP16 memory:  {fp16_mem:.1f} MB")
    print(f"  INT8 memory:  {int8_mem:.1f} MB")
    print(f"  INT4 memory:  {int4_mem:.1f} MB")
    print(f"  INT8 error:   {(out_fp16 - out_int8).abs().mean().item():.6f}")
    print(f"  INT4 error:   {(out_fp16 - out_int4).abs().mean().item():.6f}")
    
    del linear_fp16, linear_int8, linear_int4, x
    clear_gpu()
    
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 6: zKVCache
# ============================================================================
print("\n" + "="*60)
print("TEST 6: zKVCache")
print("="*60)

try:
    from zse.core.zkv.cache import zKVCache, KVCacheConfig
    
    # Check what's available
    import inspect
    print(f"  KVCacheConfig fields: {list(inspect.signature(KVCacheConfig).parameters.keys())[:5]}...")
    
    # Try to create a simple config
    config = KVCacheConfig(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        block_size=16,
        max_num_blocks=256,
        device="cuda",
    )
    
    cache = zKVCache(config)
    print(f"  Cache created successfully")
    print(f"  Total capacity: {cache.max_num_blocks * cache.block_size if hasattr(cache, 'max_num_blocks') else 'unknown'} tokens")
    
    del cache
    clear_gpu()
    
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 7: Orchestrator (End-to-End)
# ============================================================================
print("\n" + "="*60)
print("TEST 7: Intelligence Orchestrator")
print("="*60)

try:
    from zse.engine.orchestrator import IntelligenceOrchestrator
    
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"  Loading {model_id} in FP16...")
    clear_gpu()
    mem_before = get_vram()
    
    start = time.perf_counter()
    orch = IntelligenceOrchestrator.max_speed(model_id)
    orch.load(verbose=False)
    load_time = time.perf_counter() - start
    
    mem_after = get_vram()
    print(f"  Load time: {load_time:.2f}s")
    print(f"  VRAM used: {mem_after - mem_before:.2f} GB")
    
    # Generate
    print("  Generating...")
    output = orch.generate("Hello!", max_tokens=10, stream=False)
    print(f"  Output: {output[:50]}...")
    
    del orch
    clear_gpu()
    
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("BENCHMARK COMPLETE")
print("="*70)
print(f"Finished: {datetime.now()}")
