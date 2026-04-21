"""ZSE Orchestrator — GPU integration test on Modal A100.

Tests real GPU kernel compilation, weight upload, VRAM allocation,
and the full inference pipeline on an A100.

Run: modal run tests/test_modal_orchestrator.py
"""

import modal
import sys

app = modal.App("zse-orchestrator-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
)


@app.function(gpu="A100", image=zse_image, timeout=600)
def test_orchestrator_gpu():
    """Test orchestrator components on real A100 GPU."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import time
    import struct

    print("=" * 60)
    print("ZSE ORCHESTRATOR — A100 GPU INTEGRATION TEST")
    print("=" * 60)

    # Initialize CUDA context
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    ret = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert ret == 0, f"cuCtxCreate failed: {ret}"

    from zse_compiler.runtime.memory import GPUMemory
    from zse_compiler.runtime.device import get_devices
    from zse_compiler.types.dtypes import float16, int32
    from zse_engine.format.config import ModelConfig

    gpu_mem = GPUMemory(backend="cuda")
    devices = get_devices("cuda")
    device = devices[0]
    print(f"Device: {device.name}")
    print(f"VRAM: {device.vram_total_gb:.1f} GB")

    results = {}

    # ------------------------------------------------------------------ #
    # Test 1: VRAM Allocator Planning
    # ------------------------------------------------------------------ #
    print("\n[TEST 1] VRAM Allocator")
    from zse_engine.orchestrator.vram_allocator import VRAMAllocator

    config = ModelConfig(
        arch="llama", num_layers=4, num_heads=8, num_kv_heads=4,
        head_dim=64, hidden_size=512, intermediate_size=1376,
        vocab_size=32000, max_seq_len=2048,
    )

    alloc = VRAMAllocator(gpu_mem, device)
    model_size = config.estimate_model_size_bytes()
    plan = alloc.plan_allocation(model_size, config)
    print(plan.summary())

    assert plan.total_vram > 30 * 1024**3  # A100 has 40GB+
    assert plan.kv_cache_bytes > 0
    assert plan.max_batch_tokens > 0
    results["vram_plan"] = "PASS"
    print("  ✓ VRAM plan computed correctly")

    # ------------------------------------------------------------------ #
    # Test 2: Scratch Buffer Allocation on GPU
    # ------------------------------------------------------------------ #
    print("\n[TEST 2] Scratch Buffer GPU Allocation")

    scratch = alloc.allocate_scratch(config, max_seq_len=512)
    assert scratch.hidden is not None
    assert scratch.hidden.data_ptr != 0
    assert scratch.qkv is not None
    assert scratch.logits is not None
    assert scratch.total_bytes > 0
    print(f"  Scratch total: {scratch.total_bytes / 1024**2:.1f}MB")
    print(f"  Hidden buffer: ptr={scratch.hidden.data_ptr:#x}")
    results["scratch_alloc"] = "PASS"
    print("  ✓ Scratch buffers allocated on GPU")

    # ------------------------------------------------------------------ #
    # Test 3: Kernel Compilation
    # ------------------------------------------------------------------ #
    print("\n[TEST 3] Kernel Compilation (NVRTC)")
    from zse_engine.orchestrator.kernels import InferenceKernels

    kernels = InferenceKernels(backend="cuda")

    compile_start = time.monotonic()
    kernels.compile_all()
    compile_time = time.monotonic() - compile_start

    assert kernels.num_compiled == 29
    print(f"  Compiled {kernels.num_compiled} kernels in {compile_time:.2f}s")

    for name in kernels.kernel_names:
        assert kernels.is_compiled(name), f"Kernel {name} not compiled"
        print(f"    ✓ {name}")

    results["kernel_compile"] = "PASS"
    results["kernel_compile_time"] = f"{compile_time:.2f}s"

    # ------------------------------------------------------------------ #
    # Test 4: RMSNorm Kernel Execution
    # ------------------------------------------------------------------ #
    print("\n[TEST 4] RMSNorm Kernel Execution")

    hidden_size = 512
    seq_len = 4

    # Create input: [seq_len, hidden_size] = all 1.0 in fp16
    input_data = struct.pack(f'<{seq_len * hidden_size}e',
                             *([1.0] * (seq_len * hidden_size)))
    input_t = gpu_mem.allocate((seq_len, hidden_size), float16)
    gpu_mem.copy_host_to_device(input_data, input_t)

    # Weight: all 1.0
    weight_data = struct.pack(f'<{hidden_size}e', *([1.0] * hidden_size))
    weight_t = gpu_mem.allocate((hidden_size,), float16)
    gpu_mem.copy_host_to_device(weight_data, weight_t)

    # Output buffer
    out_t = gpu_mem.allocate((seq_len, hidden_size), float16)
    gpu_mem.memset(out_t, 0)

    # Launch: rmsnorm(out, input, weight, hidden_size, eps)
    kernels.launch("rmsnorm", (seq_len,), (256,),
                   out_t, input_t, weight_t, hidden_size, 1e-5)

    # Download result
    result = gpu_mem.copy_device_to_host(out_t)
    values = [struct.unpack_from('<e', result, i * 2)[0] for i in range(seq_len * hidden_size)]

    # RMSNorm of all-1s with weight=1: each value should be ~1.0
    # rms = sqrt(mean(1^2) + eps) ≈ 1.0, so out ≈ 1.0 * 1/1.0 * 1.0 = 1.0
    avg_val = sum(values) / len(values)
    print(f"  Input: all 1.0, Weight: all 1.0")
    print(f"  Output avg: {avg_val:.4f} (expected ~1.0)")
    assert 0.95 < avg_val < 1.05, f"RMSNorm output wrong: {avg_val}"
    results["rmsnorm"] = "PASS"
    print("  ✓ RMSNorm kernel correct")

    gpu_mem.free(input_t)
    gpu_mem.free(weight_t)
    gpu_mem.free(out_t)

    # ------------------------------------------------------------------ #
    # Test 5: SiLU Mul Kernel
    # ------------------------------------------------------------------ #
    print("\n[TEST 5] SiLU Mul Kernel")

    n = 1024
    # gate = 2.0, up = 3.0 → silu(2.0) * 3.0 = 2.0 * sigmoid(2.0) * 3.0
    gate_data = struct.pack(f'<{n}e', *([2.0] * n))
    up_data = struct.pack(f'<{n}e', *([3.0] * n))

    gate_t = gpu_mem.allocate((n,), float16)
    up_t = gpu_mem.allocate((n,), float16)
    out_t = gpu_mem.allocate((n,), float16)

    gpu_mem.copy_host_to_device(gate_data, gate_t)
    gpu_mem.copy_host_to_device(up_data, up_t)

    kernels.launch("silu_mul", ((n + 255) // 256,), (256,),
                   out_t, gate_t, up_t, n)

    result = gpu_mem.copy_device_to_host(out_t)
    val = struct.unpack_from('<e', result, 0)[0]

    import math
    expected = 2.0 * (1.0 / (1.0 + math.exp(-2.0))) * 3.0  # ≈ 5.238
    print(f"  silu(2.0) * 3.0 = {val:.3f} (expected {expected:.3f})")
    assert abs(val - expected) < 0.1, f"SiLU output wrong: {val} vs {expected}"
    results["silu_mul"] = "PASS"
    print("  ✓ SiLU Mul kernel correct")

    gpu_mem.free(gate_t)
    gpu_mem.free(up_t)
    gpu_mem.free(out_t)

    # ------------------------------------------------------------------ #
    # Test 6: Embedding Lookup
    # ------------------------------------------------------------------ #
    print("\n[TEST 6] Embedding Lookup")

    vocab_size = 100
    hidden = 64
    seq = 3

    # Create embedding table: token_id * 0.01 for each dim
    table_vals = []
    for tok in range(vocab_size):
        for d in range(hidden):
            table_vals.append(tok * 0.01)
    table_data = struct.pack(f'<{len(table_vals)}e', *table_vals)
    table_t = gpu_mem.allocate((vocab_size, hidden), float16)
    gpu_mem.copy_host_to_device(table_data, table_t)

    # Token IDs: [5, 10, 50]
    tokens = [5, 10, 50]
    tok_data = struct.pack(f'<{seq}i', *tokens)
    tok_t = gpu_mem.allocate((seq,), int32)
    gpu_mem.copy_host_to_device(tok_data, tok_t)

    out_t = gpu_mem.allocate((seq, hidden), float16)

    kernels.launch("embedding_lookup",
                   ((seq * hidden + 255) // 256,), (256,),
                   out_t, table_t, tok_t, hidden, seq)

    result = gpu_mem.copy_device_to_host(out_t)
    # Check first value of each row
    for i, tok in enumerate(tokens):
        val = struct.unpack_from('<e', result, i * hidden * 2)[0]
        expected = tok * 0.01
        assert abs(val - expected) < 0.01, f"Embedding [{i}] = {val}, expected {expected}"
        print(f"  Token {tok}: first val = {val:.4f} (expected {expected:.4f}) ✓")

    results["embedding"] = "PASS"
    print("  ✓ Embedding lookup correct")

    gpu_mem.free(table_t)
    gpu_mem.free(tok_t)
    gpu_mem.free(out_t)

    # ------------------------------------------------------------------ #
    # Test 7: Residual Add
    # ------------------------------------------------------------------ #
    print("\n[TEST 7] Residual Add")

    n = 512
    a_data = struct.pack(f'<{n}e', *([1.5] * n))
    b_data = struct.pack(f'<{n}e', *([2.5] * n))

    a_t = gpu_mem.allocate((n,), float16)
    b_t = gpu_mem.allocate((n,), float16)
    out_t = gpu_mem.allocate((n,), float16)

    gpu_mem.copy_host_to_device(a_data, a_t)
    gpu_mem.copy_host_to_device(b_data, b_t)

    kernels.launch("residual_add", ((n + 255) // 256,), (256,),
                   out_t, a_t, b_t, n)

    result = gpu_mem.copy_device_to_host(out_t)
    val = struct.unpack_from('<e', result, 0)[0]
    assert abs(val - 4.0) < 0.01, f"Residual add: {val} != 4.0"
    results["residual_add"] = "PASS"
    print(f"  1.5 + 2.5 = {val:.3f} ✓")

    gpu_mem.free(a_t)
    gpu_mem.free(b_t)
    gpu_mem.free(out_t)

    # ------------------------------------------------------------------ #
    # Test 8: FP16 Matmul
    # ------------------------------------------------------------------ #
    print("\n[TEST 8] FP16 Matmul")

    M, N, K = 2, 4, 8
    # A = all 1.0, B (transposed, [N, K]) = all 0.5
    # Result: each element = sum(1.0 * 0.5, K times) = K * 0.5 = 4.0
    a_data = struct.pack(f'<{M * K}e', *([1.0] * (M * K)))
    b_data = struct.pack(f'<{N * K}e', *([0.5] * (N * K)))

    a_t = gpu_mem.allocate((M, K), float16)
    b_t = gpu_mem.allocate((N, K), float16)
    out_t = gpu_mem.allocate((M, N), float16)

    gpu_mem.copy_host_to_device(a_data, a_t)
    gpu_mem.copy_host_to_device(b_data, b_t)
    gpu_mem.memset(out_t, 0)

    kernels.launch("fp16_matmul", ((N + 255) // 256, M), (256,),
                   out_t, a_t, b_t, M, N, K)

    result = gpu_mem.copy_device_to_host(out_t)
    val = struct.unpack_from('<e', result, 0)[0]
    expected = K * 0.5  # 4.0
    print(f"  [2x8] @ [8x4]^T: first element = {val:.2f} (expected {expected:.2f})")
    assert abs(val - expected) < 0.1, f"Matmul wrong: {val} vs {expected}"
    results["fp16_matmul"] = "PASS"
    print("  ✓ FP16 matmul correct")

    gpu_mem.free(a_t)
    gpu_mem.free(b_t)
    gpu_mem.free(out_t)

    # ------------------------------------------------------------------ #
    # Test 9: Softmax
    # ------------------------------------------------------------------ #
    print("\n[TEST 9] Softmax")

    rows, cols = 1, 4
    # logits: [1.0, 2.0, 3.0, 4.0]
    logits = [1.0, 2.0, 3.0, 4.0]
    logit_data = struct.pack(f'<{rows * cols}e', *logits)

    in_t = gpu_mem.allocate((rows, cols), float16)
    out_t = gpu_mem.allocate((rows, cols), float16)

    gpu_mem.copy_host_to_device(logit_data, in_t)

    kernels.launch("softmax", (rows,), (256,),
                   out_t, in_t, rows, cols)

    result = gpu_mem.copy_device_to_host(out_t)
    probs = [struct.unpack_from('<e', result, i * 2)[0] for i in range(cols)]
    total = sum(probs)
    print(f"  Logits: {logits}")
    print(f"  Probs:  [{', '.join(f'{p:.4f}' for p in probs)}]")
    print(f"  Sum:    {total:.4f}")
    assert abs(total - 1.0) < 0.02, f"Softmax doesn't sum to 1: {total}"
    # probs should be increasing
    assert probs[3] > probs[2] > probs[1] > probs[0]
    results["softmax"] = "PASS"
    print("  ✓ Softmax correct")

    gpu_mem.free(in_t)
    gpu_mem.free(out_t)

    # Clean up scratch
    scratch.destroy(gpu_mem)

    # ------------------------------------------------------------------ #
    # Test 10: @zse.kernel Dequant Matmul INT4 (Portable Kernel)
    # ------------------------------------------------------------------ #
    print("\n[TEST 10] @zse.kernel Dequant Matmul INT4 (Python → CUDA C)")

    from zse_engine.orchestrator.portable_kernels import dequant_matmul_int4

    # Verify source generation
    cuda_src = dequant_matmul_int4.source("cuda")
    assert "half*" in cuda_src
    assert "__half2float" in cuda_src
    assert "#include <cuda_fp16.h>" in cuda_src
    print("  ✓ CUDA C source generated with half support")

    # Compile on GPU
    compiled = dequant_matmul_int4.compile("cuda")
    assert compiled is not None
    print(f"  ✓ Compiled via NVRTC: {compiled}")

    # Run a small test: 2x4 matmul with trivial INT4 weights
    # Weight: all nibbles = 8, scale = 0.1, zero = 8 → dequant = 0.0
    # So output should be ~0.0 for zero weights
    # Better: nibbles = 9, zero = 8 → dequant = (9-8)*0.1 = 0.1
    # out[m,n] = sum_k(0.1 * input[m,k]) = 0.1 * sum(input row)
    M_test, N_test, K_test = 1, 2, 4
    group_size_test = 4

    # Input: all 1.0 in fp16
    inp_data = struct.pack(f'<{M_test * K_test}e', *([1.0] * (M_test * K_test)))
    inp_t = gpu_mem.allocate((M_test * K_test,), float16)
    gpu_mem.copy_host_to_device(inp_data, inp_t)

    # Weight: packed uint8, each byte = 0x99 (nibble 9 for both positions)
    w_bytes = N_test * K_test // 2  # 4 bytes
    w_data = bytes([0x99] * w_bytes)
    w_t = gpu_mem.allocate((w_bytes,))
    gpu_mem.copy_host_to_device(w_data, w_t)

    # Scales: 0.1 for each group
    num_groups = 1  # K_test / group_size_test = 1
    scales_data = struct.pack(f'<{N_test * num_groups}e', *([0.1] * (N_test * num_groups)))
    scales_t = gpu_mem.allocate((N_test * num_groups,), float16)
    gpu_mem.copy_host_to_device(scales_data, scales_t)

    # Zeros: 8 for each group → dequant = (9 - 8) * 0.1 = 0.1
    zeros_data = struct.pack(f'<{N_test * num_groups}e', *([8.0] * (N_test * num_groups)))
    zeros_t = gpu_mem.allocate((N_test * num_groups,), float16)
    gpu_mem.copy_host_to_device(zeros_data, zeros_t)

    # Output
    out_t = gpu_mem.allocate((M_test, N_test), float16)
    gpu_mem.memset(out_t, 0)

    # Launch
    from zse_compiler.runtime.launcher import KernelLauncher, LaunchConfig
    launcher = KernelLauncher()
    config = LaunchConfig(grid=((N_test + 255) // 256, M_test), block=(256,))
    launcher.launch(compiled, config,
                    out_t, w_t, scales_t, zeros_t, inp_t,
                    M_test, N_test, K_test, group_size_test)

    # Check result: each output = sum_k(0.1 * 1.0) = K * 0.1 = 0.4
    result = gpu_mem.copy_device_to_host(out_t)
    val = struct.unpack_from('<e', result, 0)[0]
    expected = K_test * 0.1  # 0.4
    print(f"  Output[0,0] = {val:.4f} (expected {expected:.4f})")
    assert abs(val - expected) < 0.05, f"@zse.kernel dequant_matmul wrong: {val} vs {expected}"
    results["portable_dequant_int4"] = "PASS"
    print("  ✓ @zse.kernel dequant_matmul_int4 correct on A100!")

    gpu_mem.free(inp_t)
    gpu_mem.free(w_t)
    gpu_mem.free(scales_t)
    gpu_mem.free(zeros_t)
    gpu_mem.free(out_t)

    # ------------------------------------------------------------------ #
    # Test 11: @zse.kernel Paged Attention (Portable Kernel)
    # ------------------------------------------------------------------ #
    print("\n[TEST 11] @zse.kernel Paged Attention (Python → CUDA C)")

    from zse_engine.orchestrator.portable_kernels import paged_attention

    # Verify source generation
    cuda_src = paged_attention.source("cuda")
    assert "half*" in cuda_src
    assert "extern __shared__" in cuda_src
    assert "__half2float" in cuda_src
    print("  ✓ CUDA C source generated with half + shared memory support")

    # Compile on GPU
    compiled_pa = paged_attention.compile("cuda")
    assert compiled_pa is not None
    print(f"  ✓ Compiled via NVRTC: {compiled_pa}")

    # Small functional test: 1 sequence, 2 heads, head_dim=4, 2 tokens in 1 block
    pa_num_seqs = 1
    pa_num_heads = 2
    pa_num_kv_heads = 2
    pa_head_dim = 4
    pa_block_size = 16
    pa_max_blocks = 1
    pa_num_layers = 1
    pa_layer_idx = 0
    pa_seq_len = 2

    # Q: [1, 2, 4] — all 1.0
    q_size = pa_num_seqs * pa_num_heads * pa_head_dim
    q_data = struct.pack(f'<{q_size}e', *([1.0] * q_size))
    q_t = gpu_mem.allocate((q_size,), float16)
    gpu_mem.copy_host_to_device(q_data, q_t)

    # KV cache block: layout = [block_total] per block
    # block_total = num_layers * 2 * num_kv_heads * block_size * head_dim
    kv_block_total = pa_num_layers * 2 * pa_num_kv_heads * pa_block_size * pa_head_dim
    # K values: all 1.0, V values: all 0.5
    kv_vals = []
    for layer in range(pa_num_layers):
        # K: num_kv_heads * block_size * head_dim
        for h in range(pa_num_kv_heads):
            for t in range(pa_block_size):
                for d in range(pa_head_dim):
                    kv_vals.append(1.0)  # K = 1.0
        # V: num_kv_heads * block_size * head_dim
        for h in range(pa_num_kv_heads):
            for t in range(pa_block_size):
                for d in range(pa_head_dim):
                    kv_vals.append(0.5)  # V = 0.5
    kv_data = struct.pack(f'<{len(kv_vals)}e', *kv_vals)
    kv_t = gpu_mem.allocate((len(kv_vals),), float16)
    gpu_mem.copy_host_to_device(kv_data, kv_t)

    # Block table: [1, 1] — sequence 0 uses block 0
    bt_data = struct.pack('<1i', 0)
    bt_t = gpu_mem.allocate((1,), int32)
    gpu_mem.copy_host_to_device(bt_data, bt_t)

    # Seq lens: [2]
    sl_data = struct.pack('<1i', pa_seq_len)
    sl_t = gpu_mem.allocate((1,), int32)
    gpu_mem.copy_host_to_device(sl_data, sl_t)

    # Output: [1, 2, 4]
    out_size = pa_num_seqs * pa_num_heads * pa_head_dim
    out_t = gpu_mem.allocate((out_size,), float16)
    gpu_mem.memset(out_t, 0)

    # scale = 1/sqrt(head_dim) = 1/2 = 0.5
    pa_scale = 1.0 / (pa_head_dim ** 0.5)

    # Launch with dynamic shared memory
    from zse_compiler.runtime.launcher import KernelLauncher, LaunchConfig
    launcher2 = KernelLauncher()
    # Grid: (num_seqs, num_heads), Block: (1,) since tid==0 does all work
    shared_bytes = pa_seq_len * 4  # float per token for scores
    config2 = LaunchConfig(
        grid=(pa_num_seqs, pa_num_heads),
        block=(1,),
        shared_mem_bytes=shared_bytes,
    )
    launcher2.launch(compiled_pa, config2,
                     out_t, q_t, kv_t, bt_t, sl_t,
                     pa_num_heads, pa_num_kv_heads, pa_head_dim,
                     pa_block_size, pa_max_blocks, pa_num_layers,
                     pa_layer_idx, pa_scale)

    # Check result: Q·K = sum(1.0 * 1.0, head_dim) * scale = 4 * 0.5 = 2.0 for each token
    # softmax(2.0, 2.0) = (0.5, 0.5)
    # V weighted sum = 0.5 * 0.5 + 0.5 * 0.5 = 0.5 for each dim
    result = gpu_mem.copy_device_to_host(out_t)
    pa_vals = [struct.unpack_from('<e', result, i * 2)[0] for i in range(out_size)]
    print(f"  Output (first head): {[f'{v:.3f}' for v in pa_vals[:pa_head_dim]]}")
    expected_val = 0.5  # V values
    for i in range(out_size):
        assert abs(pa_vals[i] - expected_val) < 0.1, f"Paged attention output[{i}] = {pa_vals[i]}, expected ~{expected_val}"

    results["portable_paged_attention"] = "PASS"
    print("  ✓ @zse.kernel paged_attention correct on A100!")

    gpu_mem.free(q_t)
    gpu_mem.free(kv_t)
    gpu_mem.free(bt_t)
    gpu_mem.free(sl_t)
    gpu_mem.free(out_t)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")
        if result != "PASS" and not name.endswith("_time"):
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    assert all_pass, "Some GPU tests failed"

    return results


@app.local_entrypoint()
def main():
    print("Launching ZSE Orchestrator GPU tests on Modal A100...")
    results = test_orchestrator_gpu.remote()
    print("\nRemote results:", results)
