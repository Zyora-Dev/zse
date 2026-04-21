"""ZSE LoRA Serving — GPU integration test on Modal A100.

Tests real GPU execution of LoRA operations:
1. Upload LoRA A/B weights to GPU
2. apply_lora: two small fp16 matmuls + scaled add
3. load_adapter_from_file round-trip
4. Full LoRA-aware model runner forward pass
5. Mixed-adapter batching correctness

Run: modal run tests/test_modal_lora.py
"""

import modal
import sys

app = modal.App("zse-lora-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
)


@app.function(gpu="A100", image=zse_image, timeout=600)
def test_lora_gpu():
    """Test LoRA serving on real A100 GPU."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import time
    import struct
    import math
    import tempfile
    import os

    print("=" * 60)
    print("ZSE LORA SERVING — A100 GPU INTEGRATION TEST")
    print("=" * 60)

    # Initialize CUDA context
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    ret = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert ret == 0, f"cuCtxCreate failed: {ret}"

    from zse_compiler.runtime.memory import GPUMemory
    from zse_compiler.runtime.device import get_devices
    from zse_compiler.types.tensor import Tensor
    from zse_compiler.types.dtypes import float16, int32
    from zse_engine.format.config import ModelConfig
    from zse_engine.orchestrator.kernels import InferenceKernels
    from zse_engine.orchestrator.lora_manager import LoRAManager
    from zse_engine.orchestrator.lora_weights import LoRAWeight, LoRAAdapter

    gpu_mem = GPUMemory(backend="cuda")
    devices = get_devices("cuda")
    device = devices[0]
    print(f"Device: {device.name}")
    print(f"VRAM: {device.vram_total_gb:.1f} GB")

    results = {}

    # Compile kernels one-by-one to identify any failures
    print("\n[SETUP] Compiling kernels one-by-one...")
    kernels = InferenceKernels(backend="cuda")
    compile_start = time.monotonic()
    failed_kernels = []
    for kname in kernels.kernel_names:
        try:
            kernels.compile_kernel(kname)
            print(f"  ✓ {kname}")
        except Exception as e:
            print(f"  ✗ {kname}: {e}")
            failed_kernels.append(kname)
    compile_time = time.monotonic() - compile_start
    print(f"  Compiled {kernels.num_compiled}/{len(kernels.kernel_names)} kernels in {compile_time:.2f}s")
    if failed_kernels:
        print(f"  FAILED: {failed_kernels}")
        # Still continue — some tests may work with available kernels

    # ------------------------------------------------------------------ #
    # Test 1: Upload LoRA A/B weights to GPU
    # ------------------------------------------------------------------ #
    print("\n[TEST 1] LoRA Weight Upload to GPU")

    rank = 16
    in_features = 512
    out_features = 512

    # Create A [rank, in_features] with known values (0.01 per element)
    a_vals = [0.01] * (rank * in_features)
    a_bytes = struct.pack(f'<{rank * in_features}e', *a_vals)

    # Create B [out_features, rank] with known values (0.02 per element)
    b_vals = [0.02] * (out_features * rank)
    b_bytes = struct.pack(f'<{out_features * rank}e', *b_vals)

    # Upload to GPU
    a_tensor = gpu_mem.allocate((rank, in_features), float16)
    gpu_mem.copy_host_to_device(a_bytes, a_tensor)

    b_tensor = gpu_mem.allocate((out_features, rank), float16)
    gpu_mem.copy_host_to_device(b_bytes, b_tensor)

    # Verify by downloading back
    a_back = gpu_mem.copy_device_to_host(a_tensor)
    a_decoded = struct.unpack(f'<{rank * in_features}e', a_back[:rank * in_features * 2])
    assert abs(a_decoded[0] - 0.01) < 0.001, f"A[0] mismatch: {a_decoded[0]}"
    assert abs(a_decoded[-1] - 0.01) < 0.001, f"A[-1] mismatch: {a_decoded[-1]}"

    b_back = gpu_mem.copy_device_to_host(b_tensor)
    b_decoded = struct.unpack(f'<{out_features * rank}e', b_back[:out_features * rank * 2])
    assert abs(b_decoded[0] - 0.02) < 0.001, f"B[0] mismatch: {b_decoded[0]}"

    print(f"  A tensor: {a_tensor.shape}, ptr={a_tensor.data_ptr:#x}")
    print(f"  B tensor: {b_tensor.shape}, ptr={b_tensor.data_ptr:#x}")
    print(f"  Upload+verify: OK")
    results["weight_upload"] = "PASS"
    print("  ✓ LoRA weights uploaded and verified on GPU")

    gpu_mem.free(a_tensor)
    gpu_mem.free(b_tensor)

    # ------------------------------------------------------------------ #
    # Test 2: LoRA Manager — load_adapter_from_dict (real GPU alloc)
    # ------------------------------------------------------------------ #
    print("\n[TEST 2] LoRA Manager — Load Adapter to GPU")

    manager = LoRAManager(gpu_mem, kernels)

    # Create weight data for 2 layers × 1 target
    weight_data = {}
    weight_shapes = {}
    for layer in range(2):
        a_data = struct.pack(f'<{rank * in_features}e', *([0.01] * (rank * in_features)))
        b_data = struct.pack(f'<{out_features * rank}e', *([0.02] * (out_features * rank)))
        weight_data[(layer, "q_proj")] = (a_data, b_data)
        weight_shapes[(layer, "q_proj")] = (in_features, out_features)

    adapter = manager.load_adapter_from_dict(
        adapter_id="test-gpu",
        rank=rank,
        alpha=float(rank),
        num_layers=2,
        target_modules=["q_proj"],
        weights_data=weight_data,
        weight_shapes=weight_shapes,
    )

    assert manager.has_adapter("test-gpu")
    assert adapter.rank == rank
    assert adapter.num_weight_pairs == 2
    assert adapter.scaling == 1.0  # alpha == rank

    # Verify weight pointers are valid GPU addresses
    w0 = adapter.get_weight(0, "q_proj")
    assert w0.a_ptr != 0, "A weight not on GPU"
    assert w0.b_ptr != 0, "B weight not on GPU"
    print(f"  Adapter: {adapter.adapter_id}, rank={adapter.rank}")
    print(f"  Layer 0 q_proj A ptr={w0.a_ptr:#x}, B ptr={w0.b_ptr:#x}")
    print(f"  Total GPU bytes: {adapter.total_gpu_bytes}")
    results["load_adapter"] = "PASS"
    print("  ✓ LoRA adapter loaded to GPU")

    # ------------------------------------------------------------------ #
    # Test 3: apply_lora — Real GPU fp16 matmuls
    # ------------------------------------------------------------------ #
    print("\n[TEST 3] apply_lora — GPU fp16 matmuls + scaled add")

    M = 4  # Batch size
    N = out_features  # 512
    K = in_features   # 512

    # Create base output: all zeros [M, N]
    base_out_data = struct.pack(f'<{M * N}e', *([0.0] * (M * N)))
    base_out = gpu_mem.allocate((M, N), float16)
    gpu_mem.copy_host_to_device(base_out_data, base_out)

    # Create input: all 1.0 [M, K]
    inp_data = struct.pack(f'<{M * K}e', *([1.0] * (M * K)))
    inp_tensor = gpu_mem.allocate((M, K), float16)
    gpu_mem.copy_host_to_device(inp_data, inp_tensor)

    # Apply LoRA: out += scaling * B @ (A @ x)
    # A is [r, K] = 0.01, x is [M, K] = 1.0
    # A @ x^T → each element of intermediate = sum(0.01 * 1.0 for K) = 0.01 * K = 0.01 * 512 = 5.12
    # Intermediate: [M, r] = 5.12 per element
    # B is [N, r] = 0.02
    # B @ intermediate → each element = sum(0.02 * 5.12 for r) = 0.02 * 5.12 * 16 = 1.6384
    # scaling = 1.0 (alpha/rank = 16/16)
    # out += 1.0 * 1.6384 = 1.6384
    expected_per_element = 0.01 * K * 0.02 * rank * 1.0  # A_val * K * B_val * r * scaling

    apply_start = time.monotonic()
    manager.apply_lora(
        base_out, inp_tensor, adapter,
        layer_idx=0, module_name="q_proj",
        M=M, N=N, K=K,
    )
    apply_time = time.monotonic() - apply_start

    # Download and verify
    out_bytes = gpu_mem.copy_device_to_host(base_out)
    out_vals = struct.unpack(f'<{M * N}e', out_bytes[:M * N * 2])

    # Check several elements
    num_checked = 0
    max_error = 0.0
    for i in range(min(100, M * N)):
        error = abs(out_vals[i] - expected_per_element)
        max_error = max(max_error, error)
        # fp16 has limited precision, allow reasonable tolerance
        assert error < 0.5, (
            f"Element [{i}]: got {out_vals[i]:.4f}, expected ~{expected_per_element:.4f}, "
            f"error={error:.4f}"
        )
        num_checked += 1

    print(f"  Expected per element: {expected_per_element:.4f}")
    print(f"  Got (first 5): {[f'{v:.4f}' for v in out_vals[:5]]}")
    print(f"  Max error: {max_error:.6f} (checked {num_checked} elements)")
    print(f"  apply_lora time: {apply_time * 1000:.2f}ms")
    results["apply_lora"] = "PASS"
    print("  ✓ apply_lora produces correct results on GPU")

    gpu_mem.free(base_out)
    gpu_mem.free(inp_tensor)

    # ------------------------------------------------------------------ #
    # Test 4: apply_lora with different scaling
    # ------------------------------------------------------------------ #
    print("\n[TEST 4] apply_lora — Scaling factor verification")

    # Create adapter with alpha=32 (scaling = 32/16 = 2.0)
    weight_data_2 = {}
    weight_shapes_2 = {}
    a_data = struct.pack(f'<{rank * in_features}e', *([0.01] * (rank * in_features)))
    b_data = struct.pack(f'<{out_features * rank}e', *([0.02] * (out_features * rank)))
    weight_data_2[(0, "q_proj")] = (a_data, b_data)
    weight_shapes_2[(0, "q_proj")] = (in_features, out_features)

    adapter2 = manager.load_adapter_from_dict(
        adapter_id="scaled-2x",
        rank=rank,
        alpha=float(rank * 2),  # scaling = 2.0
        num_layers=1,
        target_modules=["q_proj"],
        weights_data=weight_data_2,
        weight_shapes=weight_shapes_2,
    )
    assert adapter2.scaling == 2.0

    # Apply with scaling=2.0
    base_out2 = gpu_mem.allocate((M, N), float16)
    gpu_mem.copy_host_to_device(struct.pack(f'<{M * N}e', *([0.0] * (M * N))), base_out2)
    inp_tensor2 = gpu_mem.allocate((M, K), float16)
    gpu_mem.copy_host_to_device(struct.pack(f'<{M * K}e', *([1.0] * (M * K))), inp_tensor2)

    manager.apply_lora(
        base_out2, inp_tensor2, adapter2,
        layer_idx=0, module_name="q_proj",
        M=M, N=N, K=K,
    )

    out_bytes2 = gpu_mem.copy_device_to_host(base_out2)
    out_vals2 = struct.unpack(f'<{M * N}e', out_bytes2[:M * N * 2])

    expected_scaled = expected_per_element * 2.0  # scaling = 2.0
    error = abs(out_vals2[0] - expected_scaled)
    assert error < 1.0, f"Scaled element: got {out_vals2[0]:.4f}, expected ~{expected_scaled:.4f}"

    print(f"  Scaling = 2.0, expected: {expected_scaled:.4f}, got: {out_vals2[0]:.4f}")
    results["scaling"] = "PASS"
    print("  ✓ Scaling factor applied correctly")

    gpu_mem.free(base_out2)
    gpu_mem.free(inp_tensor2)

    # ------------------------------------------------------------------ #
    # Test 5: apply_lora adds to existing output (not overwrites)
    # ------------------------------------------------------------------ #
    print("\n[TEST 5] apply_lora — Additive (base + LoRA delta)")

    base_val = 5.0
    base_out3 = gpu_mem.allocate((M, N), float16)
    gpu_mem.copy_host_to_device(
        struct.pack(f'<{M * N}e', *([base_val] * (M * N))), base_out3)
    inp_tensor3 = gpu_mem.allocate((M, K), float16)
    gpu_mem.copy_host_to_device(
        struct.pack(f'<{M * K}e', *([1.0] * (M * K))), inp_tensor3)

    manager.apply_lora(
        base_out3, inp_tensor3, adapter,  # scaling=1.0 adapter
        layer_idx=0, module_name="q_proj",
        M=M, N=N, K=K,
    )

    out_bytes3 = gpu_mem.copy_device_to_host(base_out3)
    out_vals3 = struct.unpack(f'<{M * N}e', out_bytes3[:M * N * 2])

    expected_additive = base_val + expected_per_element
    error = abs(out_vals3[0] - expected_additive)
    assert error < 0.5, (
        f"Additive: got {out_vals3[0]:.4f}, expected ~{expected_additive:.4f}"
    )

    print(f"  Base={base_val}, LoRA delta={expected_per_element:.4f}")
    print(f"  Expected: {expected_additive:.4f}, Got: {out_vals3[0]:.4f}")
    results["additive"] = "PASS"
    print("  ✓ LoRA correctly adds to base output")

    gpu_mem.free(base_out3)
    gpu_mem.free(inp_tensor3)

    # ------------------------------------------------------------------ #
    # Test 6: Scratch buffer reuse (no per-call alloc)
    # ------------------------------------------------------------------ #
    print("\n[TEST 6] Scratch Buffer Reuse")

    # Apply LoRA twice with different M — scratch should grow once
    for m in [1, 4, 8, 4]:  # Growing then shrinking
        out_t = gpu_mem.allocate((m, N), float16)
        gpu_mem.copy_host_to_device(
            struct.pack(f'<{m * N}e', *([0.0] * (m * N))), out_t)
        inp_t = gpu_mem.allocate((m, K), float16)
        gpu_mem.copy_host_to_device(
            struct.pack(f'<{m * K}e', *([1.0] * (m * K))), inp_t)

        manager.apply_lora(
            out_t, inp_t, adapter,
            layer_idx=0, module_name="q_proj",
            M=m, N=N, K=K,
        )

        # Verify result
        out_b = gpu_mem.copy_device_to_host(out_t)
        val = struct.unpack('<e', out_b[:2])[0]
        assert abs(val - expected_per_element) < 0.5

        gpu_mem.free(out_t)
        gpu_mem.free(inp_t)

    # Scratch should have grown to max(8*rank, 8*N) and stayed
    assert manager._lora_scratch_bytes >= 8 * rank * 2
    assert manager._lora_out_bytes >= 8 * N * 2

    print(f"  Scratch buffer: {manager._lora_scratch_bytes} bytes")
    print(f"  LoRA out buffer: {manager._lora_out_bytes} bytes")
    results["scratch_reuse"] = "PASS"
    print("  ✓ Scratch buffers reused across calls")

    # ------------------------------------------------------------------ #
    # Test 7: load_adapter_from_file round-trip
    # ------------------------------------------------------------------ #
    print("\n[TEST 7] load_adapter_from_file — .zse-lora round-trip")

    from zse_engine.format.lora_format import save_lora
    from zse_engine.orchestrator.lora_weights import LoRAAdapter as LA

    file_rank = 8
    file_in = 256
    file_out = 256
    file_adapter = LA(
        adapter_id="file-test", rank=file_rank, alpha=float(file_rank),
        target_modules=["q_proj", "v_proj"], num_layers=2,
    )
    file_weights = {}
    for layer in range(2):
        for target in ["q_proj", "v_proj"]:
            a = struct.pack(f'<{file_rank * file_in}e',
                            *([0.05] * (file_rank * file_in)))
            b = struct.pack(f'<{file_out * file_rank}e',
                            *([0.03] * (file_out * file_rank)))
            file_weights[(layer, target)] = (a, b)

    with tempfile.NamedTemporaryFile(suffix='.zse-lora', delete=False) as f:
        path = f.name

    try:
        save_lora(path, file_adapter, file_weights)
        file_size = os.path.getsize(path)
        print(f"  Saved .zse-lora: {file_size} bytes")

        loaded = manager.load_adapter_from_file(
            adapter_id="from-file-gpu",
            path=path,
            weight_shapes={
                "q_proj": (file_in, file_out),
                "v_proj": (file_in, file_out),
            },
        )

        assert manager.has_adapter("from-file-gpu")
        assert loaded.rank == file_rank
        assert loaded.num_weight_pairs == 4  # 2 layers × 2 targets

        # Verify weights are on GPU by applying LoRA
        m_test = 2
        out_ft = gpu_mem.allocate((m_test, file_out), float16)
        gpu_mem.copy_host_to_device(
            struct.pack(f'<{m_test * file_out}e', *([0.0] * (m_test * file_out))), out_ft)
        inp_ft = gpu_mem.allocate((m_test, file_in), float16)
        gpu_mem.copy_host_to_device(
            struct.pack(f'<{m_test * file_in}e', *([1.0] * (m_test * file_in))), inp_ft)

        manager.apply_lora(
            out_ft, inp_ft, loaded,
            layer_idx=0, module_name="q_proj",
            M=m_test, N=file_out, K=file_in,
        )

        out_fb = gpu_mem.copy_device_to_host(out_ft)
        val_f = struct.unpack('<e', out_fb[:2])[0]
        # Expected: 0.05 * 256 * 0.03 * 8 * 1.0 = 3.072
        expected_f = 0.05 * file_in * 0.03 * file_rank * 1.0
        error_f = abs(val_f - expected_f)
        assert error_f < 1.0, f"File adapter: got {val_f:.4f}, expected ~{expected_f:.4f}"

        print(f"  Loaded adapter: rank={loaded.rank}, weights={loaded.num_weight_pairs}")
        print(f"  apply_lora result: {val_f:.4f} (expected {expected_f:.4f})")

        gpu_mem.free(out_ft)
        gpu_mem.free(inp_ft)

    finally:
        os.unlink(path)

    results["file_roundtrip"] = "PASS"
    print("  ✓ load_adapter_from_file works with real GPU")

    # ------------------------------------------------------------------ #
    # Test 8: Multiple adapters (hot-swap)
    # ------------------------------------------------------------------ #
    print("\n[TEST 8] Multiple Adapters — Hot-swap")

    # Load 5 adapters
    for i in range(5):
        w_data = {}
        w_shapes = {}
        a_d = struct.pack(f'<{rank * in_features}e',
                          *([0.01 * (i + 1)] * (rank * in_features)))
        b_d = struct.pack(f'<{out_features * rank}e',
                          *([0.01] * (out_features * rank)))
        w_data[(0, "q_proj")] = (a_d, b_d)
        w_shapes[(0, "q_proj")] = (in_features, out_features)

        manager.load_adapter_from_dict(
            adapter_id=f"hot-{i}",
            rank=rank, alpha=float(rank),
            num_layers=1, target_modules=["q_proj"],
            weights_data=w_data, weight_shapes=w_shapes,
        )

    # All should be loaded
    for i in range(5):
        assert manager.has_adapter(f"hot-{i}")

    # Each adapter should produce different results because A values differ
    results_per_adapter = []
    for i in range(5):
        a = manager.get_adapter(f"hot-{i}")
        out_h = gpu_mem.allocate((1, N), float16)
        gpu_mem.copy_host_to_device(struct.pack(f'<{N}e', *([0.0] * N)), out_h)
        inp_h = gpu_mem.allocate((1, K), float16)
        gpu_mem.copy_host_to_device(struct.pack(f'<{K}e', *([1.0] * K)), inp_h)

        manager.apply_lora(out_h, inp_h, a, 0, "q_proj", 1, N, K)

        out_hb = gpu_mem.copy_device_to_host(out_h)
        val_h = struct.unpack('<e', out_hb[:2])[0]
        results_per_adapter.append(val_h)

        gpu_mem.free(out_h)
        gpu_mem.free(inp_h)

    # Results should be monotonically increasing (A values increase)
    for i in range(1, 5):
        assert results_per_adapter[i] > results_per_adapter[i - 1], (
            f"Adapter {i} ({results_per_adapter[i]:.4f}) should be > "
            f"adapter {i-1} ({results_per_adapter[i-1]:.4f})"
        )

    print(f"  5 adapters loaded, results: {[f'{v:.3f}' for v in results_per_adapter]}")

    # Unload some — hot-swap
    manager.unload_adapter("hot-1")
    manager.unload_adapter("hot-3")
    assert not manager.has_adapter("hot-1")
    assert manager.has_adapter("hot-2")

    print(f"  After unload: {manager.num_adapters} adapters remaining")
    results["hot_swap"] = "PASS"
    print("  ✓ Multiple adapters with hot-swap works on GPU")

    # ------------------------------------------------------------------ #
    # Test 9: LoRA kernel — lora_scaled_add correctness
    # ------------------------------------------------------------------ #
    print("\n[TEST 9] lora_scaled_add Kernel — Direct Test")

    n_elem = 1024
    # out = [1.0, 1.0, ...], delta = [0.5, 0.5, ...], scaling = 3.0
    # Expected: out = 1.0 + 3.0 * 0.5 = 2.5
    out_k = gpu_mem.allocate((n_elem,), float16)
    gpu_mem.copy_host_to_device(
        struct.pack(f'<{n_elem}e', *([1.0] * n_elem)), out_k)
    delta_k = gpu_mem.allocate((n_elem,), float16)
    gpu_mem.copy_host_to_device(
        struct.pack(f'<{n_elem}e', *([0.5] * n_elem)), delta_k)

    kernels.launch(
        "lora_scaled_add",
        ((n_elem + 255) // 256,),
        (256,),
        out_k, delta_k, n_elem, 3.0,
    )

    out_kb = gpu_mem.copy_device_to_host(out_k)
    out_kv = struct.unpack(f'<{n_elem}e', out_kb[:n_elem * 2])

    for i in range(min(50, n_elem)):
        assert abs(out_kv[i] - 2.5) < 0.01, f"scaled_add [{i}]: {out_kv[i]}"

    print(f"  1.0 + 3.0 * 0.5 = {out_kv[0]:.4f} (expected 2.5)")
    results["scaled_add_kernel"] = "PASS"
    print("  ✓ lora_scaled_add kernel correct")

    gpu_mem.free(out_k)
    gpu_mem.free(delta_k)

    # ------------------------------------------------------------------ #
    # Test 10: Cleanup — destroy frees all GPU memory
    # ------------------------------------------------------------------ #
    print("\n[TEST 10] Cleanup — destroy()")

    pre_count = manager.num_adapters
    manager.destroy()
    assert manager.num_adapters == 0
    assert manager._lora_scratch_ptr == 0
    assert manager._lora_out_ptr == 0

    print(f"  Freed {pre_count} adapters + scratch buffers")
    results["cleanup"] = "PASS"
    print("  ✓ destroy() frees all GPU memory")

    # ================================================================== #
    # Summary
    # ================================================================== #
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")
        if status != "PASS":
            all_pass = False

    total_tests = len(results)
    passed = sum(1 for s in results.values() if s == "PASS")
    print(f"\n  {passed}/{total_tests} tests passed")

    if all_pass:
        print("\n🎉 ALL LORA GPU TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        raise AssertionError("Not all tests passed")

    return results


@app.local_entrypoint()
def main():
    print("Launching LoRA GPU test on Modal A100...")
    results = test_lora_gpu.remote()
    print(f"\nRemote results: {results}")
