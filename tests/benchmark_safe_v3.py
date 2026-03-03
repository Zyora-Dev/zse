#!/usr/bin/env python3
"""
ZSE Safe Benchmark v3 - Skip broken Triton kernels
"""

import sys
import time
import gc
from datetime import datetime

import torch
import torch.nn as nn

print("="*70)
print("ZSE SAFE BENCHMARK v3")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"PyTorch: {torch.__version__}")
print()

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_vram():
    return torch.cuda.memory_allocated() / 1e9

# ============================================================================
# TEST 1: Basic imports
# ============================================================================
print("TEST 1: Imports")
print("-"*40)
try:
    import zse
    print(f"  zse: OK (v{getattr(zse, '__version__', '?')})")
    from zse.efficiency.quantization import quantize_tensor_int8, dequantize_tensor_int8
    print("  quantization: OK")
    from zse.engine.orchestrator import IntelligenceOrchestrator
    print("  orchestrator: OK")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: Quantization (no Triton)
# ============================================================================
print("\nTEST 2: Quantization Functions (CPU/CUDA, no Triton)")
print("-"*40)
try:
    x = torch.randn(4096, 4096, device="cuda")
    
    # INT8
    q8, s8, z8 = quantize_tensor_int8(x, per_channel=True)
    r8 = dequantize_tensor_int8(q8, s8, z8)
    e8 = (x - r8).abs().mean().item()
    print(f"  INT8: MAE={e8:.6f}, Memory={q8.numel()/1e6:.1f}MB vs {x.numel()*4/1e6:.1f}MB")
    
    del x, q8, s8, z8, r8
    clear_gpu()
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")

# ============================================================================
# TEST 3: Orchestrator with 0.5B model
# ============================================================================
print("\nTEST 3: Orchestrator (0.5B model)")
print("-"*40)
try:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    clear_gpu()
    
    print(f"  Loading {model_id}...")
    start = time.perf_counter()
    orch = IntelligenceOrchestrator.max_speed(model_id)
    orch.load(verbose=False)
    load_time = time.perf_counter() - start
    vram = get_vram()
    
    print(f"  Load: {load_time:.1f}s, VRAM: {vram:.2f}GB")
    
    # Generate
    out = orch.generate("What is 2+2?", max_tokens=20, stream=False)
    print(f"  Output: {out[:60]}...")
    
    del orch
    clear_gpu()
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: Orchestrator with 14B model (the real test)
# ============================================================================
print("\nTEST 4: Orchestrator (14B model)")
print("-"*40)
try:
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    
    # FP16
    print(f"  [FP16] Loading {model_id}...")
    clear_gpu()
    start = time.perf_counter()
    orch = IntelligenceOrchestrator.max_speed(model_id)
    orch.load(verbose=False)
    fp16_load = time.perf_counter() - start
    fp16_vram = get_vram()
    print(f"  [FP16] Load: {fp16_load:.1f}s, VRAM: {fp16_vram:.1f}GB")
    
    start = time.perf_counter()
    out = orch.generate("Hello!", max_tokens=50, stream=False)
    fp16_gen = time.perf_counter() - start
    print(f"  [FP16] Gen: {fp16_gen:.1f}s")
    
    del orch
    clear_gpu()
    
    # INT4 (bitsandbytes)
    print(f"  [INT4/bnb] Loading {model_id}...")
    start = time.perf_counter()
    orch = IntelligenceOrchestrator.min_memory(model_id)
    orch.load(verbose=False)
    int4_load = time.perf_counter() - start
    int4_vram = get_vram()
    print(f"  [INT4/bnb] Load: {int4_load:.1f}s, VRAM: {int4_vram:.1f}GB")
    
    start = time.perf_counter()
    out = orch.generate("Hello!", max_tokens=50, stream=False)
    int4_gen = time.perf_counter() - start
    print(f"  [INT4/bnb] Gen: {int4_gen:.1f}s")
    
    del orch
    clear_gpu()
    
    print(f"\n  SUMMARY:")
    print(f"    FP16:     {fp16_load:.1f}s load, {fp16_vram:.1f}GB VRAM")
    print(f"    INT4/bnb: {int4_load:.1f}s load, {int4_vram:.1f}GB VRAM")
    print(f"    Memory savings: {fp16_vram/int4_vram:.1f}x")
    
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# KEY FINDING
# ============================================================================
print("\n" + "="*70)
print("KEY FINDING")
print("="*70)
print("""
ZSE's fused Triton kernels (int8_fused_matmul, int4_fused_matmul) have bugs
causing SEGFAULT. The orchestrator works but uses bitsandbytes, not ZSE's
own quantization kernels.

TO FIX:
1. Debug and fix triton_quant_kernels.py segfault
2. Wire fixed kernels into QuantizedLinear.forward()
3. Use QuantizedLinear in orchestrator instead of bitsandbytes
""")
print("="*70)
