"""ZSE Compiler — End-to-end GPU test on Modal (A100/H100).

Run: modal run tests/test_modal_gpu.py
"""

import modal
import sys

app = modal.App("zse-compiler-gpu-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
)


@app.function(gpu="A100", image=zse_image, timeout=300)
def test_gpu_end_to_end():
    """Full end-to-end test: Python → CUDA C → NVRTC → GPU execution → verify."""
    sys.path.insert(0, "/root/zse-compiler")

    import zse_compiler as zse

    print("=" * 60)
    print("ZSE COMPILER — GPU END-TO-END TEST")
    print("=" * 60)

    # 1. Device detection
    print("\n[1] Device Detection")
    backend = zse.detect_backend()
    print(f"    Backend: {backend}")

    devices = zse.get_devices(backend)
    for d in devices:
        print(f"    Device: {d.name}")
        print(f"    Compute: sm_{d.compute_capability}")
        print(f"    VRAM: {d.vram_total_gb:.1f} GB")
        print(f"    SMs: {d.multiprocessor_count}")

    assert backend == "cuda", f"Expected CUDA backend on A100, got {backend}"
    assert len(devices) > 0, "No CUDA devices found"
    print("    ✓ Device detection passed")

    # 2. Code generation
    print("\n[2] Code Generation")

    @zse.kernel
    def vector_add(a: zse.Tensor, b: zse.Tensor, out: zse.Tensor):
        idx = zse.global_id(0)
        out[idx] = a[idx] + b[idx]

    cuda_source = vector_add.source("cuda")
    print(f"    Generated CUDA source ({len(cuda_source)} chars)")
    print(f"    {cuda_source[:100]}...")
    assert "__global__" in cuda_source
    assert "vector_add" in cuda_source
    print("    ✓ Code generation passed")

    # 3. Runtime compilation via NVRTC
    print("\n[3] NVRTC Compilation")
    compiled = vector_add.compile("cuda")
    print(f"    Compiled: {compiled}")
    assert compiled.function is not None
    print("    ✓ NVRTC compilation passed")

    # 4. GPU memory allocation
    print("\n[4] GPU Memory Allocation")
    N = 1024 * 1024  # 1M elements
    mem = zse.GPUMemory(backend="cuda")

    a = mem.allocate((N,), zse.float32)
    b = mem.allocate((N,), zse.float32)
    out = mem.allocate((N,), zse.float32)

    print(f"    Allocated 3 tensors: {a.nbytes / 1024:.0f} KB each")
    print(f"    Free VRAM: {mem.get_free_memory() / (1024**3):.1f} GB")
    assert a.data_ptr != 0
    assert b.data_ptr != 0
    assert out.data_ptr != 0
    print("    ✓ GPU memory allocation passed")

    # 5. Upload test data
    print("\n[5] Data Upload")
    import struct
    a_host = struct.pack(f'{N}f', *([1.0] * N))
    b_host = struct.pack(f'{N}f', *([2.0] * N))

    mem.copy_host_to_device(a_host, a)
    mem.copy_host_to_device(b_host, b)
    mem.memset(out, 0)
    print(f"    Uploaded {len(a_host) + len(b_host)} bytes to GPU")
    print("    ✓ Data upload passed")

    # 6. Kernel launch
    print("\n[6] Kernel Launch")
    block_size = 256
    grid_size = N // block_size

    vector_add.launch(
        grid=(grid_size,),
        block=(block_size,),
        args=(a, b, out),
        backend="cuda",
    )
    print(f"    Launched: grid=({grid_size},), block=({block_size},)")
    print("    ✓ Kernel launch passed")

    # 7. Verify results
    print("\n[7] Result Verification")
    result_bytes = mem.copy_device_to_host(out)
    results = struct.unpack(f'{N}f', result_bytes)

    all_correct = True
    for i in range(10):
        if abs(results[i] - 3.0) > 1e-5:
            print(f"    FAIL: out[{i}] = {results[i]}, expected 3.0")
            all_correct = False
    for i in range(N - 10, N):
        if abs(results[i] - 3.0) > 1e-5:
            print(f"    FAIL: out[{i}] = {results[i]}, expected 3.0")
            all_correct = False

    assert all_correct, "Result verification failed!"
    print(f"    out[0:5] = {results[:5]}")
    print(f"    out[-5:] = {results[-5:]}")
    print(f"    All {N} elements = 3.0 ✓")
    print("    ✓ Result verification passed")

    # 8. Cleanup
    mem.free(a)
    mem.free(b)
    mem.free(out)

    print("\n" + "=" * 60)
    print("✅ ALL GPU TESTS PASSED — ZSE COMPILER WORKS ON REAL HARDWARE")
    print("=" * 60)

    return "PASS"


@app.local_entrypoint()
def main():
    result = test_gpu_end_to_end.remote()
    print(f"\nModal result: {result}")
