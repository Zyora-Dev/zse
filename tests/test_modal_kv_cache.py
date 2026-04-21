"""ZSE KV Cache — GPU integration test on Modal A100.

Tests real GPU memory allocation, KV cache management, and memory tracking.

Run: modal run tests/test_modal_kv_cache.py
"""

import modal
import sys

app = modal.App("zse-kv-cache-test")

zse_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .add_local_dir("zse-compiler", remote_path="/root/zse-compiler", copy=True)
    .add_local_dir("zse-engine", remote_path="/root/zse-engine", copy=True)
)


@app.function(gpu="A100", image=zse_image, timeout=300)
def test_kv_cache_gpu():
    """Test KV cache with real GPU memory on A100."""
    sys.path.insert(0, "/root/zse-engine")
    sys.path.insert(0, "/root/zse-compiler")

    import ctypes
    from zse_engine.format.config import ModelConfig
    from zse_engine.cache import KVCacheManager, BlockPool
    from zse_compiler.runtime.memory import GPUMemory
    from zse_compiler.runtime.device import get_devices

    print("=" * 60)
    print("ZSE KV CACHE — GPU INTEGRATION TEST")
    print("=" * 60)

    # Initialize CUDA context (required before any GPU allocation)
    import ctypes
    libcuda = ctypes.CDLL("libcuda.so.1")
    libcuda.cuInit(0)
    ctx = ctypes.c_void_p()
    ret = libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, 0)
    assert ret == 0, f"cuCtxCreate failed: {ret}"

    results = {}

    # ------------------------------------------------------------------ #
    # Test 1: GPU Memory Allocation
    # ------------------------------------------------------------------ #
    print("\n[TEST 1] GPU Memory Allocation")

    devices = get_devices("cuda")
    print(f"  Device: {devices[0].name}")
    print(f"  VRAM: {devices[0].vram_total_gb:.1f} GB")

    gpu_mem = GPUMemory(backend="cuda")
    free_before = gpu_mem.get_free_memory()
    total = gpu_mem.get_total_memory()
    print(f"  Free VRAM: {free_before / 1024**3:.1f} GB / {total / 1024**3:.1f} GB")

    # Allocate a 1GB slab for KV cache
    config = ModelConfig(
        arch="llama", num_layers=32, num_heads=32, num_kv_heads=8,
        head_dim=128, hidden_size=4096, intermediate_size=11008,
        vocab_size=32000, max_seq_len=4096,
    )

    budget = 1 * 1024**3  # 1 GB
    pool = BlockPool(
        gpu_mem=gpu_mem, total_bytes=budget,
        block_size_tokens=16, num_kv_heads=8,
        head_dim=128, num_layers=32,
    )

    free_after = gpu_mem.get_free_memory()
    allocated = free_before - free_after
    print(f"  Allocated: {allocated / 1024**2:.0f} MB")
    print(f"  Blocks: {pool.num_blocks} × {pool.block_bytes:,} bytes")
    print(f"  Capacity: {pool.num_blocks * 16:,} tokens")

    assert pool.num_blocks > 0
    assert allocated > 0

    pool.destroy()
    results["gpu_alloc"] = "PASS"
    print("  ✅ GPU allocation: PASS")

    # ------------------------------------------------------------------ #
    # Test 2: Full CacheManager on GPU
    # ------------------------------------------------------------------ #
    print("\n[TEST 2] CacheManager on GPU (7B model config)")

    manager = KVCacheManager(
        config=config, gpu_mem=gpu_mem,
        budget_bytes=2 * 1024**3,  # 2 GB
        block_size=16,
        eviction_policy="smart",
    )

    stats = manager.stats()
    print(f"  Total blocks: {stats.total_blocks}")
    print(f"  Block size: {stats.block_size_tokens} tokens × {stats.block_bytes:,} bytes")
    print(f"  Max capacity: {manager.max_tokens_capacity():,} tokens")

    # Allocate multiple sequences
    num_seqs = 8
    prompt_len = 128
    for i in range(num_seqs):
        handle = manager.allocate_sequence(i, list(range(prompt_len)))
        assert handle.num_tokens == prompt_len

    stats = manager.stats()
    print(f"  Sequences: {stats.num_sequences}")
    print(f"  Cached tokens: {stats.total_cached_tokens:,}")
    print(f"  Utilization: {stats.utilization:.1%}")

    # Extend each by 50 tokens (decode)
    for i in range(num_seqs):
        manager.extend_sequence(i, 50)

    stats = manager.stats()
    print(f"  After decode: {stats.total_cached_tokens:,} tokens, {stats.utilization:.1%}")

    # Get attention metadata for batch
    meta = manager.get_attention_metadata(list(range(num_seqs)))
    assert meta.num_seqs == num_seqs
    assert all(sl == prompt_len + 50 for sl in meta.seq_lengths)

    # Verify metadata can be packed for GPU
    bt_bytes = meta.pack_block_tables()
    sl_bytes = meta.pack_seq_lengths()
    print(f"  Block tables: {len(bt_bytes):,} bytes")
    print(f"  Seq lengths: {len(sl_bytes)} bytes")

    # Upload metadata to GPU
    from zse_compiler.types.tensor import Tensor
    from zse_compiler.types.dtypes import int32
    bt_tensor = gpu_mem.allocate(shape=(meta.num_seqs * meta.max_blocks_per_seq,), dtype=int32)
    gpu_mem.copy_host_to_device(bt_bytes, bt_tensor)
    roundtrip = gpu_mem.copy_device_to_host(bt_tensor)
    assert roundtrip == bt_bytes, "Block table GPU roundtrip failed!"
    gpu_mem.free(bt_tensor)
    print(f"  Block table GPU roundtrip: ✓")

    # Test fork (beam search)
    fork_handle = manager.fork_sequence(0, 100)
    assert fork_handle.num_tokens == prompt_len + 50
    print(f"  Fork: seq 0 → 100, {fork_handle.num_tokens} tokens (COW)")

    # Print summary
    print(f"\n{manager.summary()}")

    # Cleanup
    for i in range(num_seqs):
        manager.free_sequence(i)
    manager.free_sequence(100)
    manager.destroy()

    results["cache_manager"] = "PASS"
    print("  ✅ CacheManager GPU: PASS")

    # ------------------------------------------------------------------ #
    # Test 3: Memory pressure & eviction
    # ------------------------------------------------------------------ #
    print("\n[TEST 3] Eviction under memory pressure")

    # Tiny budget to force eviction
    small_config = ModelConfig(
        arch="llama", num_layers=4, num_heads=8, num_kv_heads=2,
        head_dim=64, hidden_size=512, intermediate_size=1376,
        vocab_size=1000, max_seq_len=256,
    )

    manager = KVCacheManager(
        config=small_config, gpu_mem=gpu_mem,
        budget_bytes=256 * 1024,  # 256 KB — very small
        block_size=8,
        eviction_policy="smart",
        enable_dedup=False,
    )

    print(f"  Budget: 256 KB, blocks: {manager.stats().total_blocks}")

    # Fill up
    filled = 0
    while manager.num_free_blocks > 2:
        manager.allocate_sequence(filled, list(range(32)))
        manager.get_attention_metadata([filled])  # register access
        manager.mark_idle(filled)
        filled += 1

    print(f"  Filled {filled} sequences, free: {manager.num_free_blocks}")

    # Allocate more — should trigger eviction
    manager.allocate_sequence(999, list(range(32)))

    stats = manager.stats()
    print(f"  After pressure: evictions={stats.eviction_count}, "
          f"tokens_evicted={stats.tokens_evicted}")
    assert stats.eviction_count > 0

    manager.destroy()
    results["eviction"] = "PASS"
    print("  ✅ Eviction: PASS")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = all(v == "PASS" for v in results.values())
    for name, status in results.items():
        symbol = "✅" if status == "PASS" else "❌"
        print(f"  {symbol} {name}: {status}")
    print("=" * 60)
    print("✅ ALL KV CACHE GPU TESTS PASSED" if all_pass else "❌ SOME FAILED")

    libcuda.cuCtxDestroy_v2(ctx)
    return "PASS" if all_pass else "FAIL"


@app.local_entrypoint()
def main():
    result = test_kv_cache_gpu.remote()
    print(f"\nModal result: {result}")
