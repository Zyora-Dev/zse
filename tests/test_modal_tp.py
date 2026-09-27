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
import modal.experimental

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
def main(full_inference: bool = False, sustained_load: bool = False, multi_node: bool = False):
    if multi_node:
        import json
        import secrets
        from pathlib import Path
        run_id = secrets.token_hex(8)
        result = test_multi_node.remote(secrets.token_bytes(32), run_id)
        path = Path(__file__).with_name("modal_tp_multi_node.json")
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {path}", flush=True)
        if not result.get("passed"):
            raise RuntimeError(result.get("error", "Cross-host TP validation failed"))
        return
    results = test_full_inference.remote(sustained_load) if full_inference or sustained_load else test_tp.remote()
    if full_inference or sustained_load:
        import json
        from pathlib import Path
        result_name = "modal_tp_sustained_load.json" if sustained_load else "modal_tp_full_inference.json"
        Path(__file__).with_name(result_name).write_text(
            json.dumps(results, indent=2) + "\n"
        )
    print("\n📊 Results received from Modal:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    if (full_inference or sustained_load) and not results["passed"]:
        raise RuntimeError(results.get("error", "Full TP inference validation failed"))


@app.function(
    gpu="A100-80GB:2",
    image=zse_image,
    timeout=900,
    volumes={"/root/zse_cache": zse_cache, "/root/hf_cache": hf_cache},
)
def test_full_inference(sustained_load: bool = False):
    import os
    import sys
    import time
    import traceback
    from pathlib import Path

    sys.path[:0] = ["/root/zse-compiler", "/root/zse-engine"]
    os.environ["NCCL_DEBUG"] = "INFO"
    results = {"passed": False, "tp_size": 2, "requests": []}
    engine = None
    try:
        from zse_engine.orchestrator.tp_engine import TPEngine

        candidates = list(Path("/root/zse_cache").glob(
            "*a09a35458c702b33eeacc393d103063234e8bc28_int4_rowmajor.zse"
        ))
        assert len(candidates) == 1, f"Expected corrected Qwen7B artifact, found {candidates}"
        results["model_path"] = str(candidates[0])
        print(f"Full TP inference model: {candidates[0]}", flush=True)
        started = time.monotonic()
        engine = TPEngine(str(candidates[0]), tp_size=2, quiet=False)
        results["init_seconds"] = time.monotonic() - started
        results["worker_pids"] = [worker.pid for worker in engine._workers]
        assert len(set(results["worker_pids"])) == 2
        for question, expected in (
            ("What is the capital of France? Answer briefly.", "paris"),
            ("What is the capital of Japan? Answer briefly.", "tokyo"),
        ):
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            before = engine._total_tokens
            text = engine.generate(prompt, max_tokens=32, temperature=0.0)
            tokens = engine._total_tokens - before
            results["requests"].append({"question": question, "text": text, "tokens": tokens})
            print(f"TP generated {tokens} tokens: {text!r}", flush=True)
            assert tokens > 1, "Decode was not exercised"
            assert expected in text.lower(), f"Incorrect model answer: {text!r}"
            assert tokens < 32, "Brief answer did not terminate at EOS"
        assert all(worker.is_alive() for worker in engine._workers)
        if sustained_load:
            results["load"] = run_sustained_load(engine)
        results["passed"] = True
    except Exception as error:
        results["error"] = str(error)
        results["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if engine is not None:
            workers = list(engine._workers)
            engine.destroy()
            results["workers_stopped"] = all(not worker.is_alive() for worker in workers)
            results["passed"] = results["passed"] and results["workers_stopped"]
    print(f"Full TP results: {results}", flush=True)
    return results


def run_sustained_load(engine, vram_reader=None):
    import csv
    import io
    import statistics
    import subprocess
    import time

    def device_memory():
        if vram_reader is not None:
            return vram_reader()
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=15)
        return {row[0].strip(): int(row[1]) for row in csv.reader(io.StringIO(output))}

    prompts = [
        ("What is the capital of France? Answer briefly.", "paris"),
        ("What is the capital of Japan? Answer briefly.", "tokyo"),
        (("This is background text for a repeated-request test. " * 64)
         + "What is the capital of France? Answer briefly.", "paris"),
        (("This is background text for a repeated-request test. " * 128)
         + "What is the capital of Japan? Answer briefly.", "tokyo"),
    ]
    baseline = {}
    latencies = []
    completed = 0
    cancelled = 0
    streamed = 0
    memory_start = None
    started = time.monotonic()
    while completed < 100 or time.monotonic() - started < 300:
        if time.monotonic() - started > 600:
            raise RuntimeError("Sustained workload exceeded its 600-second bound")
        prompt_index = completed % len(prompts)
        question, expected = prompts[prompt_index]
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        request_start = time.monotonic()
        if completed % 5 == 4:
            stream = engine.stream_generate(prompt, max_tokens=48, temperature=0.0)
            try:
                assert next(stream), "Cancelled stream returned no first token"
            finally:
                stream.close()
            cancelled += 1
        else:
            if completed % 5 == 3:
                pieces = list(engine.stream_generate(prompt, max_tokens=48, temperature=0.0))
                assert 1 < len(pieces) < 47, "Stream did not decode and terminate early"
                text = "".join(pieces)
                streamed += 1
            else:
                before = engine._total_tokens
                text = engine.generate(prompt, max_tokens=48, temperature=0.0)
                assert 1 < engine._total_tokens - before < 48, "Generation did not terminate early"
            assert expected in text.lower(), f"Incorrect sustained answer: {text!r}"
            assert "<|" not in text, f"Special token leaked: {text!r}"
            if prompt_index not in baseline:
                baseline[prompt_index] = text
            assert text == baseline[prompt_index], "Greedy answer changed across repeated requests"
        latencies.append(time.monotonic() - request_start)
        assert set(engine._last_release_stats) == {0, 1}
        for cache in engine._last_release_stats.values():
            assert cache["num_sequences"] == 0, f"Live sequence leaked: {cache}"
            assert cache["allocated_blocks"] == 0, f"KV blocks leaked: {cache}"
            assert cache["free_blocks"] == cache["total_blocks"], f"KV capacity not restored: {cache}"
        assert all(worker.is_alive() for worker in engine._workers), "Worker exited during load"
        completed += 1
        if completed == 20:
            memory_start = device_memory()
        if completed % 20 == 0:
            print(f"TP load: {completed} requests, {time.monotonic() - started:.1f}s, cache reclaimed on both ranks", flush=True)
    memory_end = device_memory()
    assert memory_start is not None and len(memory_start) == (1 if vram_reader is not None else 2)
    assert memory_start.keys() == memory_end.keys()
    growth = {device: memory_end[device] - memory_start[device] for device in memory_start}
    assert max(growth.values()) <= 128, f"Post-warmup VRAM growth exceeds 128 MiB: {growth}"
    return {
        "passed": True,
        "duration_seconds": time.monotonic() - started,
        "requests": completed,
        "cancelled_streams": cancelled,
        "completed_streams": streamed,
        "concurrency": 1,
        "minimum_requests": 100,
        "minimum_duration_seconds": 300,
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
        "answer_samples": baseline,
        "post_warmup_vram_mib": memory_start,
        "final_vram_mib": memory_end,
        "vram_growth_mib": growth,
        "final_cache": engine._last_release_stats,
    }


@app.function(
    image=zse_image,
    gpu="A100-80GB:8",
    timeout=900,
    retries=0,
    volumes={"/root/zse_cache": zse_cache, "/root/hf_cache": hf_cache},
)
@modal.experimental.clustered(size=2)
def test_multi_node(authkey: bytes, run_id: str):
    import hashlib
    import json
    import os
    import socket
    import subprocess
    import time
    import traceback
    from pathlib import Path
    from modal.experimental import get_cluster_info

    sys.path[:0] = ["/root/zse-compiler", "/root/zse-engine"]
    from zse_engine.orchestrator.tp_engine import TPEngine
    from zse_engine.orchestrator.tp_transport import TPRemoteEndpoint, serve_tp_worker

    cluster = get_cluster_info()
    rank = cluster.rank
    addresses = list(cluster.container_ips)
    result = {"passed": False, "run_id": run_id, "node_rank": rank,
              "cluster_id": cluster.cluster_id, "node_addresses": addresses,
              "allocated_gpus": 16, "active_inference_gpus": 2, "tp_size": 2}
    engine = None
    workers = []
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET6"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

    def gpu_snapshot():
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,name,memory.used", "--format=csv,noheader,nounits",
        ], text=True, timeout=15)
        return [{"uuid": fields[0].strip(), "name": fields[1].strip(), "used_mib": int(fields[2])}
                for line in output.strip().splitlines() for fields in [line.split(",")]]

    try:
        assert len(addresses) == 2 and len(set(addresses)) == 2, addresses
        gpus = gpu_snapshot()
        assert len(gpus) == 8 and all("A100" in gpu["name"] for gpu in gpus), gpus
        candidates = list(Path("/root/zse_cache").glob(
            "*a09a35458c702b33eeacc393d103063234e8bc28_int4_rowmajor.zse"
        ))
        assert len(candidates) == 1, f"Expected corrected Qwen7B artifact, found {candidates}"
        model_path = candidates[0]
        digest = hashlib.sha256()
        with model_path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        node = {"hostname": socket.gethostname(), "address": addresses[rank], "gpus": gpus,
                "model_sha256": digest.hexdigest(), "model_bytes": model_path.stat().st_size}
        result["node"] = node
        endpoint = TPRemoteEndpoint(addresses[1], 29571, authkey)
        print(f"Node {rank} ready: {node['hostname']}, eight A100 GPUs, model {node['model_sha256']}", flush=True)
        if rank == 1:
            result.update(serve_tp_worker(endpoint, str(model_path), 1, 2, local_rank=0,
                                          timeout=840, node_metadata=node))
            result["passed"] = True
        else:
            engine = TPEngine(str(model_path), tp_size=2, quiet=True,
                              remote_endpoints=[endpoint])
            workers = list(engine._workers)
            follower = engine._ready_info[1]["node"]
            assert follower["model_sha256"] == node["model_sha256"], "Model files differ between nodes"
            assert follower["hostname"] != node["hostname"], "Node hostnames are identical"
            assert not ({gpu["uuid"] for gpu in gpus} & {gpu["uuid"] for gpu in follower["gpus"]})
            assert engine._ready_info[0]["local_rank"] == engine._ready_info[1]["local_rank"] == 0
            result["rank_readiness"] = engine._ready_info

            def active_vram():
                snapshot = gpu_snapshot()[0]
                return {snapshot["uuid"]: snapshot["used_mib"]}

            result["sustained_load"] = run_sustained_load(engine, vram_reader=active_vram)
            result["vram_scope"] = "leader active GPU only; follower logical KV reclamation checked per request"
            result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if engine is not None:
            engine.destroy()
            result["worker_exitcodes"] = [worker.exitcode for worker in workers]
            result["workers_stopped"] = all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
            if not result["workers_stopped"]:
                result["passed"] = False
                result.setdefault("error", "Missing clean local/remote worker shutdown")
        result["finished_at"] = time.time()
        path = Path(f"/root/zse_cache/tp_multi_node_{run_id}_rank{rank}.json")
        path.write_text(json.dumps(result, indent=2) + "\n")
        zse_cache.commit()
        print(json.dumps(result, indent=2), flush=True)
    return result
