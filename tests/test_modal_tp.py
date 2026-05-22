"""ZSE Tensor Parallelism Test — 2x A100-80GB on Modal.

Tests:
1. Multi-GPU detection (2 GPUs visible)
2. NCCL communicator init via ncclCommInitAll (no sockets needed)
3. NCCL all-reduce fp32 correctness
4. Multi-device GPUMemory
5. Weight sharding dimensions
6. NCCL all-reduce fp16
7. Full TP model inference (if model available)

Run: modal run tests/test_modal_tp.py
"""

import sys
import modal

app = modal.App("zse-tp-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
    .pip_install("huggingface_hub")
)

hf_cache = modal.Volume.from_name("zse-hf-cache", create_if_missing=True)
zse_cache = modal.Volume.from_name("zse-model-cache", create_if_missing=True)


@app.function(
    gpu="A100-80GB:2",
    image=zse_image,
    timeout=3600,
    volumes={"/root/hf_cache": hf_cache, "/root/zse_cache": zse_cache},
)
def test_tp():
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    import os
    import time
    import struct
    os.environ["NCCL_DEBUG"] = "WARN"

    results = {}

    print("=" * 70)
    print("ZSE TENSOR PARALLELISM TEST — 2x A100-80GB")
    print("=" * 70)

    # ================================================================
    # TEST 1: Multi-GPU detection
    # ================================================================
    print("\n--- TEST 1: Multi-GPU Detection ---")

    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    count = ctypes.c_int(0)
    libcuda.cuDeviceGetCount(ctypes.byref(count))
    num_gpus = count.value
    print(f"  GPUs detected: {num_gpus}")

    for i in range(num_gpus):
        name_buf = ctypes.create_string_buffer(256)
        dev = ctypes.c_int(0)
        libcuda.cuDeviceGet(ctypes.byref(dev), i)
        libcuda.cuDeviceGetName(name_buf, 256, dev)
        total_mem = ctypes.c_size_t(0)
        libcuda.cuDeviceTotalMem_v2(ctypes.byref(total_mem), dev)
        print(f"  GPU {i}: {name_buf.value.decode()} ({total_mem.value / 1024**3:.1f}GB)")

    assert num_gpus >= 2, f"Need 2 GPUs, got {num_gpus}"
    results["test1_gpu_count"] = num_gpus
    print("  ✅ PASS: 2 GPUs detected")

    # ================================================================
    # TEST 2: NCCL comm_init_all (no sockets — works in containers)
    # ================================================================
    print("\n--- TEST 2: NCCL CommInitAll ---")

    from zse_compiler.runtime.nccl import is_nccl_available, comm_init_all

    assert is_nccl_available("cuda"), "NCCL not found!"
    print("  NCCL available: ✅")

    comms = comm_init_all(2, backend="cuda")
    assert len(comms) == 2
    assert comms[0].rank == 0
    assert comms[1].rank == 1
    print(f"  Comm 0: {comms[0]}")
    print(f"  Comm 1: {comms[1]}")
    results["test2_comm_init_all"] = "PASS"
    print("  ✅ PASS: ncclCommInitAll (no socket bootstrap)")

    # ================================================================
    # TEST 3: NCCL all-reduce fp32
    # ================================================================
    print("\n--- TEST 3: NCCL All-Reduce FP32 ---")

    cudart = ctypes.CDLL("libcudart.so.12")

    # Create CUDA streams per device (NCCL needs per-device streams)
    streams = []
    bufs = []
    values_per_rank = [[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]
    count = 4
    nbytes = count * 4

    for rank in range(2):
        cudart.cudaSetDevice(rank)
        stream = ctypes.c_void_p(0)
        cudart.cudaStreamCreate(ctypes.byref(stream))
        streams.append(stream)

        buf_ptr = ctypes.c_void_p(0)
        cudart.cudaMalloc(ctypes.byref(buf_ptr), ctypes.c_size_t(nbytes))
        host_data = struct.pack(f'<{count}f', *values_per_rank[rank])
        src = ctypes.c_char_p(host_data)
        cudart.cudaMemcpy(buf_ptr, src, ctypes.c_size_t(nbytes), 1)
        bufs.append(buf_ptr)
        print(f"  Rank {rank}: buf_ptr={buf_ptr.value:#x}, stream={stream.value}")

    # Use ncclGroupStart/End to issue all-reduce from single thread
    nccl_lib = comms[0]._lib
    print("  Calling ncclGroupStart...")
    status = nccl_lib.ncclGroupStart()
    print(f"  ncclGroupStart status: {status}")

    for rank in range(2):
        print(f"  Issuing ncclAllReduce for rank {rank}, comm={comms[rank]._comm}")
        status = nccl_lib.ncclAllReduce(
            bufs[rank], bufs[rank],
            ctypes.c_size_t(count),
            ctypes.c_int(7),  # NCCL_FLOAT32
            ctypes.c_int(0),  # NCCL_SUM
            comms[rank]._comm,
            streams[rank],
        )
        print(f"  ncclAllReduce rank {rank} status: {status}")

    print("  Calling ncclGroupEnd...")
    status = nccl_lib.ncclGroupEnd()
    print(f"  ncclGroupEnd status: {status}")

    # Sync and readback
    expected = [11.0, 22.0, 33.0, 44.0]
    for rank in range(2):
        cudart.cudaSetDevice(rank)
        cudart.cudaStreamSynchronize(streams[rank])
        out_buf = ctypes.create_string_buffer(nbytes)
        cudart.cudaMemcpy(out_buf, bufs[rank], ctypes.c_size_t(nbytes), 2)
        result = list(struct.unpack(f'<{count}f', out_buf.raw))
        match = all(abs(a - b) < 0.01 for a, b in zip(result, expected))
        print(f"  Rank {rank}: {result} correct={match}")
        assert match, f"Rank {rank} wrong: {result}"
        cudart.cudaFree(bufs[rank])
        cudart.cudaStreamDestroy(streams[rank])

    results["test3_nccl_fp32"] = "PASS"
    print("  ✅ PASS: NCCL all-reduce fp32 ([1+10, 2+20, 3+30, 4+40] = [11, 22, 33, 44])")

    # ================================================================
    # TEST 4: Multi-device GPUMemory
    # ================================================================
    print("\n--- TEST 4: Multi-device GPUMemory ---")

    from zse_compiler.runtime.memory import GPUMemory
    from zse_compiler.types.dtypes import float32 as dt_f32

    gpu0 = GPUMemory(backend="cuda", device_index=0)
    gpu1 = GPUMemory(backend="cuda", device_index=1)

    t0 = gpu0.allocate((1024,), dt_f32)
    t1 = gpu1.allocate((1024,), dt_f32)

    assert t0.data_ptr != 0, "GPU 0 alloc failed"
    assert t1.data_ptr != 0, "GPU 1 alloc failed"

    data0 = struct.pack('<1024f', *[1.0] * 1024)
    data1 = struct.pack('<1024f', *[2.0] * 1024)
    gpu0.ensure_context()
    gpu0.copy_host_to_device(data0, t0)
    gpu1.ensure_context()
    gpu1.copy_host_to_device(data1, t1)

    gpu0.ensure_context()
    out0 = gpu0.copy_device_to_host(t0)
    gpu1.ensure_context()
    out1 = gpu1.copy_device_to_host(t1)

    vals0 = struct.unpack('<1024f', out0)
    vals1 = struct.unpack('<1024f', out1)
    assert abs(vals0[0] - 1.0) < 0.01, f"GPU 0 data wrong: {vals0[0]}"
    assert abs(vals1[0] - 2.0) < 0.01, f"GPU 1 data wrong: {vals1[0]}"

    gpu0.ensure_context()
    gpu0.free(t0)
    gpu1.ensure_context()
    gpu1.free(t1)

    results["test4_multi_device_memory"] = "PASS"
    print(f"  GPU 0: alloc + write 1.0 + readback ✅")
    print(f"  GPU 1: alloc + write 2.0 + readback ✅")
    print("  ✅ PASS: Independent GPU memory on 2 devices")

    # ================================================================
    # TEST 5: Weight sharding dimensions
    # ================================================================
    print("\n--- TEST 5: Weight Sharding ---")

    from zse_engine.orchestrator.tensor_parallel import (
        TensorParallelGroup, TPConfig, COLUMN_PARALLEL, ROW_PARALLEL,
    )

    tp_cfg = TPConfig(tp_size=2, backend="cuda")
    tp_cfg.validate(32, 8, 11008)
    print("  Config validation (32h, 8kv, 11008i, tp=2): ✅")

    def _make_tp(tp_size, rank):
        tp = TensorParallelGroup.__new__(TensorParallelGroup)
        tp.tp_size = tp_size
        tp.rank = rank
        tp.backend = "cuda"
        tp._stream = 0
        tp._comm = None
        return tp

    t0 = _make_tp(2, 0)
    t1 = _make_tp(2, 1)

    assert t0.compute_shard_range(4096, COLUMN_PARALLEL) == (0, 2048)
    assert t1.compute_shard_range(4096, COLUMN_PARALLEL) == (2048, 4096)
    print(f"  Q_proj column split: ✅")

    assert t0.compute_shard_range(4096, ROW_PARALLEL) == (0, 2048)
    assert t1.compute_shard_range(4096, ROW_PARALLEL) == (2048, 4096)
    print(f"  O_proj row split: ✅")

    assert t0.compute_shard_range(11008, COLUMN_PARALLEL) == (0, 5504)
    assert t1.compute_shard_range(11008, COLUMN_PARALLEL) == (5504, 11008)
    print(f"  Gate_proj column split: ✅")

    results["test5_weight_sharding"] = "PASS"
    print("  ✅ PASS: Weight sharding dimensions correct")

    # ================================================================
    # TEST 6: NCCL all-reduce fp16
    # ================================================================
    print("\n--- TEST 6: NCCL All-Reduce FP16 ---")

    # Fresh comms
    for c in comms:
        c.destroy()
    comms = comm_init_all(2, backend="cuda")

    fp16_count = 256
    fp16_nbytes = fp16_count * 2
    # fp16: 1.0 = 0x3C00, 2.0 = 0x4000
    fp16_vals = [0x3C00, 0x4000]

    fp16_bufs = []
    for rank in range(2):
        cudart.cudaSetDevice(rank)
        buf_ptr = ctypes.c_void_p(0)
        cudart.cudaMalloc(ctypes.byref(buf_ptr), ctypes.c_size_t(fp16_nbytes))
        host_data = struct.pack(f'<{fp16_count}H', *([fp16_vals[rank]] * fp16_count))
        src = ctypes.c_char_p(host_data)
        cudart.cudaMemcpy(buf_ptr, src, ctypes.c_size_t(fp16_nbytes), 1)
        fp16_bufs.append(buf_ptr)

    nccl_lib = comms[0]._lib
    nccl_lib.ncclGroupStart()
    for rank in range(2):
        cudart.cudaSetDevice(rank)
        nccl_lib.ncclAllReduce(
            fp16_bufs[rank], fp16_bufs[rank],
            ctypes.c_size_t(fp16_count),
            ctypes.c_int(6),  # NCCL_FLOAT16
            ctypes.c_int(0),  # NCCL_SUM
            comms[rank]._comm,
            ctypes.c_void_p(0),
        )
    nccl_lib.ncclGroupEnd()

    expected_fp16 = 0x4200  # 3.0 in fp16
    for rank in range(2):
        cudart.cudaSetDevice(rank)
        cudart.cudaDeviceSynchronize()
        out_buf = ctypes.create_string_buffer(fp16_nbytes)
        cudart.cudaMemcpy(out_buf, fp16_bufs[rank], ctypes.c_size_t(fp16_nbytes), 2)
        raw = struct.unpack(f'<{fp16_count}H', out_buf.raw)
        num_correct = sum(1 for v in raw if v == expected_fp16)
        print(f"  Rank {rank}: {num_correct}/{fp16_count} elements correct")
        assert num_correct == fp16_count, f"Rank {rank}: only {num_correct}/{fp16_count}"
        cudart.cudaFree(fp16_bufs[rank])

    results["test6_nccl_fp16"] = "PASS"
    print("  ✅ PASS: NCCL all-reduce fp16 (256 elements, 1.0 + 2.0 = 3.0)")

    # Cleanup comms
    for c in comms:
        c.destroy()

    # ================================================================
    # TEST 7: Full TP inference
    # ================================================================
    print("\n--- TEST 7: Full TP Inference ---")

    model_path = "/root/zse_cache/qwen2.5-7b-int4.zse"

    if not os.path.exists(model_path):
        print(f"  Model not cached, downloading + converting...")
        try:
            from huggingface_hub import snapshot_download
            hf_dir = snapshot_download("Qwen/Qwen2.5-7B-Instruct", cache_dir="/root/hf_cache")
            print(f"  Downloaded: {hf_dir}")

            sys.argv = ["zse-convert", hf_dir, model_path,
                         "--quant", "int4", "--arch", "qwen2", "--quiet"]
            from zse_engine.format.__main__ import main as convert_main
            convert_main()
            print(f"  Converted: {model_path}")
        except Exception as e:
            print(f"  ⚠️ Download/convert failed: {e}")
            results["test7_tp_inference"] = f"SKIP: {e}"

    if os.path.exists(model_path):
        try:
            from zse_engine.orchestrator.tp_engine import TPEngine

            t_start = time.monotonic()
            engine = TPEngine(model_path, tp_size=2, quiet=False)
            init_time = time.monotonic() - t_start
            print(f"  TP Engine init: {init_time:.2f}s")

            t_start = time.monotonic()
            text = engine.generate("The capital of France is", max_tokens=20, temperature=0.0)
            gen_time = time.monotonic() - t_start
            print(f"  Generated ({gen_time:.2f}s): {text[:200]}")

            engine.destroy()
            results["test7_tp_inference"] = "PASS"
            results["test7_init_time"] = round(init_time, 2)
            results["test7_gen_time"] = round(gen_time, 2)
            print("  ✅ PASS: 2-GPU TP inference")

        except Exception as e:
            print(f"  ⚠️ TP inference failed: {e}")
            import traceback
            traceback.print_exc()
            results["test7_tp_inference"] = f"FAIL: {e}"

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("TENSOR PARALLELISM TEST SUMMARY")
    print("=" * 70)
    for test, result in results.items():
        status = "✅" if result == "PASS" or (isinstance(result, (int, float)) and result > 0) else "❌"
        print(f"  {status} {test}: {result}")

    pass_count = sum(1 for k, v in results.items()
                     if k.startswith("test") and (v == "PASS" or (isinstance(v, (int, float)) and v > 0)))
    total_count = len([k for k in results if k.startswith("test") and not k.endswith("_time")])
    print(f"\n  {pass_count}/{total_count} tests passed")

    return results


@app.local_entrypoint()
def main():
    results = test_tp.remote()
    print("\n📊 Results received from Modal:")
    for k, v in results.items():
        print(f"  {k}: {v}")
