"""ZSE KV Cache — Unit tests.

Tests all components: BlockPool, PageTable, Evictor, Dedup, CacheManager.
All tests run CPU-only (gpu_mem=None) — GPU integration tested on Modal.
"""

import os
import sys
import time
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zse-engine'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zse-compiler'))

from zse_engine.format.config import ModelConfig
from zse_engine.cache.block_pool import BlockPool, Block
from zse_engine.cache.page_table import PageTable
from zse_engine.cache.evictor import Evictor, EvictionCandidate
from zse_engine.cache.dedup import BlockDedup
from zse_engine.cache.cache_manager import KVCacheManager, KVCacheHandle
from zse_engine.cache.attention_metadata import build_attention_metadata


# --- Helper: tiny model config for testing ---

def _tiny_config() -> ModelConfig:
    return ModelConfig(
        arch="llama",
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=512,
        intermediate_size=1376,
        vocab_size=1000,
        max_seq_len=256,
    )


# ================================================================
# BlockPool tests
# ================================================================

def test_block_pool_alloc_free():
    """Test basic block allocation and freeing."""
    config = _tiny_config()
    # Budget: enough for ~10 blocks
    # block_bytes = 16 * 2 * 64 * 2 * 2 * 4 = 131072 per block (with 4 layers)
    pool = BlockPool(
        gpu_mem=None, total_bytes=2 * 1024 * 1024,
        block_size_tokens=16, num_kv_heads=2, head_dim=64, num_layers=4,
    )

    assert pool.num_blocks > 0
    assert pool.num_free == pool.num_blocks
    assert pool.num_allocated == 0

    # Allocate all blocks
    blocks = []
    for _ in range(pool.num_blocks):
        b = pool.alloc()
        blocks.append(b)
        assert b.ref_count == 1

    assert pool.num_free == 0
    assert pool.num_allocated == pool.num_blocks

    # Should fail to allocate more
    try:
        pool.alloc()
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass

    # Free all
    for b in blocks:
        pool.free(b)
    assert pool.num_free == pool.num_blocks
    assert pool.num_allocated == 0

    print("  [PASS] block_pool alloc/free")


def test_block_pool_ref_count():
    """Test reference counting for COW."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=16, num_kv_heads=2, head_dim=64, num_layers=4,
    )

    block = pool.alloc()
    assert block.ref_count == 1

    pool.increment_ref(block)
    assert block.ref_count == 2
    assert block.is_shared

    # First free just decrements ref
    freed = pool.free(block)
    assert not freed
    assert block.ref_count == 1

    # Second free actually frees
    freed = pool.free(block)
    assert freed
    assert block.ref_count == 0

    print("  [PASS] block_pool ref_count")


def test_block_pool_memory_layout():
    """Test block GPU offset calculations."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=16, num_kv_heads=2, head_dim=64, num_layers=4,
    )

    b0 = pool.alloc()
    b1 = pool.alloc()

    # Block 1 should start right after block 0
    assert b1.gpu_offset == pool.block_bytes
    assert b0.gpu_offset == 0

    print("  [PASS] block_pool memory layout")


# ================================================================
# PageTable tests
# ================================================================

def test_page_table_basic():
    """Test creating sequences and appending tokens."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=4, num_kv_heads=2, head_dim=64, num_layers=4,
    )
    pt = PageTable(pool)

    pt.create_sequence(0)
    assert pt.num_tokens(0) == 0

    # Add 10 tokens (should need 3 blocks with block_size=4)
    pt.append_tokens(0, 10)
    assert pt.num_tokens(0) == 10
    assert pt.num_blocks(0) == 3  # ceil(10/4)

    block_ids = pt.get_block_ids(0)
    assert len(block_ids) == 3

    print("  [PASS] page_table basic")


def test_page_table_fork():
    """Test copy-on-write forking."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=4, num_kv_heads=2, head_dim=64, num_layers=4,
    )
    pt = PageTable(pool)

    pt.create_sequence(0)
    pt.append_tokens(0, 8)  # 2 blocks

    # Fork
    pt.fork_sequence(0, 1)
    assert pt.num_tokens(1) == 8
    assert pt.num_blocks(1) == 2

    # Blocks should be shared (same IDs)
    assert pt.get_block_ids(0) == pt.get_block_ids(1)

    # Blocks should have ref_count 2
    for block in pt.get_blocks(0):
        assert block.ref_count == 2

    # Removing one sequence should decrement, not free
    pt.remove_sequence(1)
    for block in pt.get_blocks(0):
        assert block.ref_count == 1

    print("  [PASS] page_table fork (COW)")


def test_page_table_evict_block():
    """Test token-level eviction (removing individual blocks)."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=4, num_kv_heads=2, head_dim=64, num_layers=4,
    )
    pt = PageTable(pool)

    pt.create_sequence(0)
    pt.append_tokens(0, 12)  # 3 blocks
    free_before = pool.num_free

    # Evict middle block
    evicted = pt.evict_block(0, 1)
    assert evicted is not None
    assert pool.num_free == free_before + 1

    print("  [PASS] page_table evict_block")


# ================================================================
# Evictor tests
# ================================================================

def test_evictor_lru():
    """Test LRU eviction policy."""
    evictor = Evictor(policy="lru")

    # Record accesses at different times
    evictor.record_access(seq_id=0, block_idx=0, block_id=0, num_tokens=16)
    time.sleep(0.01)
    evictor.record_access(seq_id=0, block_idx=1, block_id=1, num_tokens=16)
    time.sleep(0.01)
    evictor.record_access(seq_id=1, block_idx=0, block_id=2, num_tokens=16)

    # LRU: oldest (block 0) should be evicted first
    victims = evictor.select_victims(1)
    assert len(victims) == 1
    assert victims[0].block_id == 0

    print("  [PASS] evictor LRU")


def test_evictor_smart():
    """Test smart eviction policy."""
    evictor = Evictor(policy="smart")

    # Block 0: prompt, accessed many times (high value)
    for _ in range(10):
        evictor.record_access(seq_id=0, block_idx=0, block_id=0,
                              num_tokens=16, is_prompt=True)

    # Block 1: decode, accessed once (low value)
    evictor.record_access(seq_id=0, block_idx=1, block_id=1,
                          num_tokens=16, is_prompt=False)

    # Block 2: decode, accessed once (low value)
    evictor.record_access(seq_id=1, block_idx=0, block_id=2,
                          num_tokens=16, is_prompt=False)

    # Smart: low-frequency decode blocks should be evicted first
    victims = evictor.select_victims(2)
    assert len(victims) == 2
    # Block 0 (prompt, high freq) should NOT be in first 2 victims
    victim_ids = {v.block_id for v in victims}
    assert 0 not in victim_ids, f"Prompt block evicted too early: {victim_ids}"

    print("  [PASS] evictor smart")


def test_evictor_exclude():
    """Test excluding sequences from eviction."""
    evictor = Evictor(policy="lru")

    evictor.record_access(seq_id=0, block_idx=0, block_id=0, num_tokens=16)
    evictor.record_access(seq_id=1, block_idx=0, block_id=1, num_tokens=16)

    # Exclude seq 0
    victims = evictor.select_victims(1, exclude_seq_ids={0})
    assert len(victims) == 1
    assert victims[0].seq_id == 1

    print("  [PASS] evictor exclude")


# ================================================================
# Dedup tests
# ================================================================

def test_dedup_basic():
    """Test block deduplication."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=4, num_kv_heads=2, head_dim=64, num_layers=4,
    )
    dedup = BlockDedup(pool)

    b0 = pool.alloc()
    b1 = pool.alloc()

    tokens = [100, 200, 300, 400]

    # Register block 0 — first time, no match
    existing = dedup.register_block(b0.block_id, tokens)
    assert existing is None

    # Register block 1 with same tokens — should match block 0
    existing = dedup.register_block(b1.block_id, tokens)
    assert existing == b0.block_id
    assert b0.ref_count == 2  # incremented

    assert dedup.dedup_hits == 1
    assert dedup.dedup_misses == 1
    assert dedup.hit_rate == 0.5

    print("  [PASS] dedup basic")


def test_dedup_lookup():
    """Test hash lookup without registering."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=4, num_kv_heads=2, head_dim=64, num_layers=4,
    )
    dedup = BlockDedup(pool)

    b0 = pool.alloc()
    tokens = [10, 20, 30, 40]

    dedup.register_block(b0.block_id, tokens)

    assert dedup.lookup(tokens) == b0.block_id
    assert dedup.lookup([99, 99, 99, 99]) is None

    print("  [PASS] dedup lookup")


# ================================================================
# CacheManager tests
# ================================================================

def test_cache_manager_basic():
    """Test end-to-end cache manager workflow."""
    config = _tiny_config()

    manager = KVCacheManager(
        config=config,
        gpu_mem=None,
        budget_bytes=2 * 1024 * 1024,
        block_size=4,
        eviction_policy="smart",
    )

    # Allocate sequence with prompt
    prompt = list(range(20))  # 20 tokens
    handle = manager.allocate_sequence(0, prompt_tokens=prompt)
    assert handle.seq_id == 0
    assert handle.num_tokens == 20
    assert handle.num_blocks == 5  # ceil(20/4)

    # Extend by 1 token (decode step)
    manager.extend_sequence(0, 1)

    # Get attention metadata
    meta = manager.get_attention_metadata([0])
    assert meta.num_seqs == 1
    assert meta.seq_lengths[0] == 21
    assert len(meta.block_tables[0]) == 6  # 5 + 1 new block

    # Free
    manager.free_sequence(0)
    assert manager.num_sequences == 0

    print("  [PASS] cache_manager basic")


def test_cache_manager_multi_seq():
    """Test multiple concurrent sequences."""
    config = _tiny_config()
    manager = KVCacheManager(
        config=config, gpu_mem=None,
        budget_bytes=4 * 1024 * 1024, block_size=4,
    )

    # Allocate 3 sequences
    for i in range(3):
        manager.allocate_sequence(i, prompt_tokens=list(range(10)))

    assert manager.num_sequences == 3

    # Get batch metadata
    meta = manager.get_attention_metadata([0, 1, 2])
    assert meta.num_seqs == 3
    assert len(meta.seq_lengths) == 3

    # Free one
    manager.free_sequence(1)
    assert manager.num_sequences == 2

    print("  [PASS] cache_manager multi_seq")


def test_cache_manager_eviction():
    """Test eviction under memory pressure."""
    config = _tiny_config()
    # block_bytes for this config = 4 * 2 * 64 * 2 * 2 * 4 = 8192
    manager = KVCacheManager(
        config=config, gpu_mem=None,
        budget_bytes=8192 * 4,  # Only 4 blocks total
        block_size=4,
        eviction_policy="lru",
        enable_dedup=False,  # Disable dedup to test pure eviction
    )

    # Fill up completely: 16 tokens = 4 blocks = all blocks
    manager.allocate_sequence(0, list(range(8)))   # 2 blocks
    manager.allocate_sequence(1, list(range(8)))   # 2 blocks
    assert manager.num_free_blocks == 0

    # Access them so evictor tracks them
    manager.get_attention_metadata([0, 1])

    # Mark both as idle
    manager.mark_idle(0)
    manager.mark_idle(1)

    # Now allocate a new seq — must evict to make room
    manager.allocate_sequence(2, list(range(4)))  # needs 1 block

    stats = manager.stats()
    assert stats.eviction_count > 0

    print(f"  [PASS] cache_manager eviction ({stats.eviction_count} evictions)")


def test_cache_manager_fork():
    """Test beam search forking."""
    config = _tiny_config()
    manager = KVCacheManager(
        config=config, gpu_mem=None,
        budget_bytes=2 * 1024 * 1024, block_size=4,
    )

    manager.allocate_sequence(0, list(range(8)))
    handle = manager.fork_sequence(0, 1)
    assert handle.num_tokens == 8

    # Both should work
    meta = manager.get_attention_metadata([0, 1])
    assert meta.num_seqs == 2
    assert meta.seq_lengths[0] == meta.seq_lengths[1] == 8

    print("  [PASS] cache_manager fork")


def test_cache_manager_stats():
    """Test stats and summary."""
    config = _tiny_config()
    manager = KVCacheManager(
        config=config, gpu_mem=None,
        budget_bytes=2 * 1024 * 1024, block_size=4,
    )

    manager.allocate_sequence(0, list(range(16)))
    stats = manager.stats()

    assert stats.total_blocks > 0
    assert stats.allocated_blocks == 4  # 16/4
    assert stats.num_sequences == 1
    assert stats.total_cached_tokens == 16
    assert stats.block_size_tokens == 4

    summary = manager.summary()
    assert "ZSE KV Cache" in summary
    assert "Blocks:" in summary

    print(f"  [PASS] cache_manager stats")


# ================================================================
# AttentionMetadata tests
# ================================================================

def test_attention_metadata_packing():
    """Test block table and seq length packing for GPU."""
    meta = build_attention_metadata(
        block_ids_per_seq=[[0, 1, 2], [3, 4]],
        seq_lengths=[12, 8],
        block_size=4,
    )

    assert meta.num_seqs == 2
    assert meta.max_blocks_per_seq == 3
    assert meta.max_seq_len == 12

    # Pack and verify
    bt_bytes = meta.pack_block_tables()
    # 2 seqs × 3 blocks × 4 bytes = 24 bytes
    assert len(bt_bytes) == 24

    # Verify values
    vals = struct.unpack(f'<{6}i', bt_bytes)
    assert vals == (0, 1, 2, 3, 4, -1)  # second row padded with -1

    sl_bytes = meta.pack_seq_lengths()
    assert len(sl_bytes) == 8
    seq_lens = struct.unpack('<2i', sl_bytes)
    assert seq_lens == (12, 8)

    print("  [PASS] attention_metadata packing")


def test_auto_block_size():
    """Test adaptive block size selection including GQA."""
    config = _tiny_config()

    # Short context
    config.max_seq_len = 2048
    config.num_heads = 8
    config.num_kv_heads = 8  # MHA
    assert KVCacheManager._auto_block_size(config) == 8

    # Medium context, MHA
    config.max_seq_len = 8192
    assert KVCacheManager._auto_block_size(config) == 16

    # Long context, MHA
    config.max_seq_len = 131072
    assert KVCacheManager._auto_block_size(config) == 32

    # Medium context with heavy GQA (ratio 0.25) → halved
    config.max_seq_len = 8192
    config.num_heads = 32
    config.num_kv_heads = 8  # ratio = 0.25
    assert KVCacheManager._auto_block_size(config) == 8  # 16 halved to 8

    # Long context with heavy GQA → halved
    config.max_seq_len = 131072
    assert KVCacheManager._auto_block_size(config) == 16  # 32 halved to 16

    print("  [PASS] auto_block_size (with GQA)")


def test_double_free_protection():
    """Critical #4: double free should not corrupt free list."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=16, num_kv_heads=2, head_dim=64, num_layers=4,
    )
    block = pool.alloc()
    initial_free = pool.num_free

    pool.free(block)
    assert pool.num_free == initial_free + 1

    # Second free should be no-op (block already freed)
    pool.free(block)
    assert pool.num_free == initial_free + 1  # Should NOT increase again

    print("  [PASS] double_free_protection")


def test_eviction_no_crash_on_get_block_ids():
    """Critical #3: get_block_ids should return -1 for evicted blocks, not crash."""
    pool = BlockPool(
        gpu_mem=None, total_bytes=1024 * 1024,
        block_size_tokens=4, num_kv_heads=2, head_dim=64, num_layers=4,
    )
    pt = PageTable(pool)

    pt.create_sequence(0)
    pt.append_tokens(0, 12)  # 3 blocks
    original_ids = pt.get_block_ids(0)
    assert len(original_ids) == 3

    # Evict middle block
    pt.evict_block(0, 1)

    # This should NOT crash — returns -1 for evicted position
    ids_after = pt.get_block_ids(0)
    assert len(ids_after) == 3
    assert ids_after[0] == original_ids[0]
    assert ids_after[1] == -1  # Evicted
    assert ids_after[2] == original_ids[2]

    print("  [PASS] eviction_no_crash_on_get_block_ids")


def test_max_seq_len_enforcement():
    """Gap #7: sequences should not exceed max_seq_len."""
    config = _tiny_config()
    config.max_seq_len = 16  # Very short

    manager = KVCacheManager(
        config=config, gpu_mem=None,
        budget_bytes=2 * 1024 * 1024, block_size=4,
    )

    # Prompt too long
    try:
        manager.allocate_sequence(0, list(range(20)))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exceeds max_seq_len" in str(e)

    # Extend beyond limit
    manager.allocate_sequence(1, list(range(12)))
    try:
        manager.extend_sequence(1, 10)  # 12 + 10 = 22 > 16
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exceeds max_seq_len" in str(e)

    print("  [PASS] max_seq_len_enforcement")


def test_zero_token_extend():
    """Edge case: extending by 0 tokens should be no-op."""
    config = _tiny_config()
    manager = KVCacheManager(
        config=config, gpu_mem=None,
        budget_bytes=2 * 1024 * 1024, block_size=4,
    )
    manager.allocate_sequence(0, list(range(8)))
    before = manager.stats().total_cached_tokens
    manager.extend_sequence(0, 0)
    after = manager.stats().total_cached_tokens
    assert before == after
    print("  [PASS] zero_token_extend")


# ================================================================
# Runner
# ================================================================

def main():
    print("=" * 60)
    print("ZSE KV Cache — Unit Tests")
    print("=" * 60)

    tests = [
        test_block_pool_alloc_free,
        test_block_pool_ref_count,
        test_block_pool_memory_layout,
        test_page_table_basic,
        test_page_table_fork,
        test_page_table_evict_block,
        test_evictor_lru,
        test_evictor_smart,
        test_evictor_exclude,
        test_dedup_basic,
        test_dedup_lookup,
        test_cache_manager_basic,
        test_cache_manager_multi_seq,
        test_cache_manager_eviction,
        test_cache_manager_fork,
        test_cache_manager_stats,
        test_attention_metadata_packing,
        test_auto_block_size,
        test_double_free_protection,
        test_eviction_no_crash_on_get_block_ids,
        test_max_seq_len_enforcement,
        test_zero_token_extend,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
