"""ZSE KV Cache Manager — Main orchestration API for paged KV cache.

Thread-safe orchestration of BlockPool, PageTable, Evictor, and BlockDedup.

Advantages over vLLM:
- Adaptive block sizing (not fixed 16)
- Token-level eviction (not sequence-level)
- Smart eviction (frequency + recency + recompute cost, not LRU)
- Block deduplication for shared prefixes
- GPU-only (no CPU swap)
- Sliding window support
- Copy-on-write with actual GPU data transfer

Usage:
    manager = KVCacheManager(config, gpu_mem, budget_bytes=30*1024**3)
    handle = manager.allocate_sequence(0, prompt_token_ids)
    manager.extend_sequence(0, num_new_tokens=1)
    metadata = manager.get_attention_metadata([0, 1, 2])
    manager.free_sequence(0)
"""

import threading
from dataclasses import dataclass
from typing import List, Optional, Dict, Set

from zse_engine.format.config import ModelConfig
from zse_engine.cache.block_pool import BlockPool
from zse_engine.cache.page_table import PageTable
from zse_engine.cache.evictor import Evictor
from zse_engine.cache.dedup import BlockDedup
from zse_engine.cache.attention_metadata import AttentionMetadata, build_attention_metadata


@dataclass
class KVCacheHandle:
    """Handle returned when a sequence is allocated."""
    seq_id: int
    num_tokens: int
    num_blocks: int


@dataclass
class CacheStats:
    """KV cache statistics."""
    # Pool
    total_blocks: int
    allocated_blocks: int
    free_blocks: int
    total_bytes: int
    used_bytes: int
    utilization: float
    # Sequences
    num_sequences: int
    total_cached_tokens: int
    # Eviction
    eviction_count: int
    tokens_evicted: int
    eviction_policy: str
    # Dedup
    dedup_hits: int
    dedup_hit_rate: float
    # Config
    block_size_tokens: int
    block_bytes: int


class KVCacheManager:
    """High-level KV cache manager for LLM inference.

    Thread-safe: all public methods acquire a lock.

    Args:
        config: Model config (provides KV dimensions)
        gpu_mem: GPUMemory instance (None for CPU-only testing)
        budget_bytes: Total GPU memory budget for KV cache
        block_size: Tokens per block (default: auto-select)
        eviction_policy: "smart", "lru", or "lfu"
        enable_dedup: Enable block deduplication for shared prefixes
    """

    def __init__(
        self,
        config: ModelConfig,
        gpu_mem=None,
        budget_bytes: int = 0,
        block_size: int = 0,
        eviction_policy: str = "smart",
        enable_dedup: bool = True,
    ):
        self._config = config
        self._gpu_mem = gpu_mem
        self._lock = threading.Lock()

        # Auto-select block size if not specified
        if block_size <= 0:
            block_size = self._auto_block_size(config)
        self._block_size = block_size

        # Auto-budget if not specified
        if budget_bytes <= 0 and gpu_mem is not None:
            free = gpu_mem.get_free_memory()
            model_size = config.estimate_model_size_bytes()
            budget_bytes = int((free - model_size) * 0.9)
            budget_bytes = max(budget_bytes, 1024 * 1024)

        # Create components
        self._pool = BlockPool(
            gpu_mem=gpu_mem,
            total_bytes=budget_bytes,
            block_size_tokens=block_size,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            num_layers=config.num_layers,
        )

        # GPU copy function for COW
        gpu_copy_fn = None
        if gpu_mem is not None:
            def _do_gpu_copy(src_block, dst_block):
                src_ptr = self._pool.gpu_ptr_for_block(src_block)
                dst_ptr = self._pool.gpu_ptr_for_block(dst_block)
                gpu_mem.copy_device_to_device(src_ptr, dst_ptr, self._pool.block_bytes)
            gpu_copy_fn = _do_gpu_copy

        self._page_table = PageTable(self._pool, gpu_copy_fn=gpu_copy_fn)
        self._evictor = Evictor(policy=eviction_policy)
        self._dedup = BlockDedup(self._pool) if enable_dedup else None

        # Track prompt lengths for eviction scoring
        self._prompt_lengths: Dict[int, int] = {}
        # Track sequences currently generating (protected from eviction)
        self._active_seq_ids: Set[int] = set()

    @staticmethod
    def _auto_block_size(config: ModelConfig) -> int:
        """Auto-select block size based on model characteristics.

        Considers both context length and GQA ratio:
        - GQA models (num_kv_heads << num_heads) have smaller per-token KV,
          so smaller blocks waste less memory per unfilled slot.
        - Long context models benefit from larger blocks (less metadata).
        """
        max_seq = config.max_seq_len
        gqa_ratio = config.num_kv_heads / config.num_heads  # 1.0 = MHA, <1.0 = GQA

        if max_seq <= 4096:
            base = 8
        elif max_seq <= 32768:
            base = 16
        else:
            base = 32

        # GQA with heavy grouping (e.g., 8 KV heads / 32 heads = 0.25)
        # → smaller KV per token → can afford smaller blocks with less waste
        if gqa_ratio <= 0.25 and base > 8:
            base = base // 2

        return base

    def allocate_sequence(
        self,
        seq_id: int,
        prompt_tokens: Optional[List[int]] = None,
    ) -> KVCacheHandle:
        """Allocate KV cache for a new sequence.

        Args:
            seq_id: Unique sequence identifier
            prompt_tokens: Token IDs of the prompt (for dedup + eviction scoring)

        Returns:
            KVCacheHandle with allocation info
        """
        with self._lock:
            num_tokens = len(prompt_tokens) if prompt_tokens else 0

            # Gap #7: max_seq_len enforcement
            if num_tokens > self._config.max_seq_len:
                raise ValueError(
                    f"Prompt length {num_tokens} exceeds max_seq_len "
                    f"{self._config.max_seq_len}"
                )

            self._prompt_lengths[seq_id] = num_tokens

            # Check if we have enough blocks
            blocks_needed = (num_tokens + self._block_size - 1) // self._block_size if num_tokens > 0 else 0
            if blocks_needed > self._pool.num_free:
                shortage = blocks_needed - self._pool.num_free
                if not self._evict_if_needed_locked(shortage):
                    raise RuntimeError(
                        f"Cannot allocate {blocks_needed} blocks for seq {seq_id}: "
                        f"only {self._pool.num_free} free, eviction failed"
                    )

            # Create sequence and allocate blocks
            self._page_table.create_sequence(seq_id)
            if num_tokens > 0:
                self._page_table.append_tokens(seq_id, num_tokens)

            # Dedup: properly replace duplicate blocks
            if self._dedup and prompt_tokens:
                self._try_dedup_blocks(seq_id, prompt_tokens)

            return KVCacheHandle(
                seq_id=seq_id,
                num_tokens=self._page_table.num_tokens(seq_id),
                num_blocks=self._page_table.num_blocks(seq_id),
            )

    def extend_sequence(self, seq_id: int, num_new_tokens: int = 1):
        """Extend a sequence by new tokens (decode step)."""
        if num_new_tokens <= 0:
            return

        with self._lock:
            # Gap #7: max_seq_len check
            current = self._page_table.num_tokens(seq_id)
            if current + num_new_tokens > self._config.max_seq_len:
                raise ValueError(
                    f"Extending seq {seq_id} to {current + num_new_tokens} tokens "
                    f"exceeds max_seq_len {self._config.max_seq_len}"
                )

            # Critical #5: accurate block need calculation
            blocks_before = self._page_table.num_blocks(seq_id)
            total_after = current + num_new_tokens
            blocks_after = (total_after + self._block_size - 1) // self._block_size
            new_blocks_needed = blocks_after - blocks_before

            if new_blocks_needed > 0 and new_blocks_needed > self._pool.num_free:
                if not self._evict_if_needed_locked(new_blocks_needed - self._pool.num_free):
                    raise RuntimeError(
                        f"Cannot extend seq {seq_id}: need {new_blocks_needed} blocks, "
                        f"only {self._pool.num_free} free"
                    )

            self._page_table.append_tokens(seq_id, num_new_tokens)

            # Gap #10: sliding window — trim old blocks beyond window
            if self._config.sliding_window and self._config.sliding_window > 0:
                self._trim_sliding_window(seq_id)

    def free_sequence(self, seq_id: int):
        """Free all KV cache for a sequence."""
        with self._lock:
            # Clean up dedup entries
            if self._dedup:
                for block in self._page_table.get_blocks(seq_id):
                    self._dedup.deregister_block(block.block_id)

            # Clean up evictor entries
            for block in self._page_table.get_blocks(seq_id):
                self._evictor.remove_block(block.block_id)

            self._page_table.remove_sequence(seq_id)
            self._prompt_lengths.pop(seq_id, None)
            self._active_seq_ids.discard(seq_id)

    def fork_sequence(self, src_id: int, dst_id: int) -> KVCacheHandle:
        """Fork a sequence (copy-on-write) for beam search.

        Shared blocks have ref_count > 1. On next write to a shared block,
        PageTable copies the GPU data before modifying.
        """
        with self._lock:
            self._page_table.fork_sequence(src_id, dst_id)
            self._prompt_lengths[dst_id] = self._prompt_lengths.get(src_id, 0)
            self._active_seq_ids.add(dst_id)

            return KVCacheHandle(
                seq_id=dst_id,
                num_tokens=self._page_table.num_tokens(dst_id),
                num_blocks=self._page_table.num_blocks(dst_id),
            )

    def get_attention_metadata(
        self,
        seq_ids: List[int],
        gpu_upload: bool = False,
    ) -> AttentionMetadata:
        """Build GPU attention metadata for a batch of sequences.

        Args:
            seq_ids: Sequences in this batch
            gpu_upload: If True and gpu_mem available, upload block tables
                        and seq lengths to GPU tensors.

        Also records access for eviction scoring.
        """
        with self._lock:
            block_ids_per_seq = []
            seq_lengths = []

            for sid in seq_ids:
                bids = self._page_table.get_block_ids(sid)
                block_ids_per_seq.append(bids)
                num_tok = self._page_table.num_tokens(sid)
                seq_lengths.append(num_tok)

                # Record access for eviction scoring
                prompt_len = self._prompt_lengths.get(sid, 0)
                self._evictor.record_batch_access(
                    sid, bids, self._block_size, prompt_len,
                    total_seq_tokens=num_tok,
                )

            meta = build_attention_metadata(
                block_ids_per_seq, seq_lengths, self._block_size,
            )

            # Gap #8: upload metadata to GPU
            if gpu_upload and self._gpu_mem is not None:
                self._upload_metadata_to_gpu(meta)

            return meta

    def _upload_metadata_to_gpu(self, meta: AttentionMetadata):
        """Upload block tables and seq lengths to GPU memory."""
        from zse_compiler.types.dtypes import int32

        bt_bytes = meta.pack_block_tables()
        sl_bytes = meta.pack_seq_lengths()

        bt_tensor = self._gpu_mem.allocate(
            shape=(meta.num_seqs * meta.max_blocks_per_seq,), dtype=int32
        )
        self._gpu_mem.copy_host_to_device(bt_bytes, bt_tensor)
        meta.block_tables_gpu_ptr = bt_tensor.data_ptr

        sl_tensor = self._gpu_mem.allocate(
            shape=(meta.num_seqs,), dtype=int32
        )
        self._gpu_mem.copy_host_to_device(sl_bytes, sl_tensor)
        meta.seq_lengths_gpu_ptr = sl_tensor.data_ptr

    def evict_if_needed(self, blocks_needed: int) -> bool:
        """Try to free blocks via eviction (thread-safe)."""
        with self._lock:
            return self._evict_if_needed_locked(blocks_needed)

    def _evict_if_needed_locked(self, blocks_needed: int) -> bool:
        """Try to free blocks (must be called with lock held)."""
        if self._pool.num_free >= blocks_needed:
            return True

        shortage = blocks_needed - self._pool.num_free
        victims = self._evictor.select_victims(
            shortage, exclude_seq_ids=self._active_seq_ids,
        )

        if len(victims) < shortage:
            victims = self._evictor.select_victims(shortage)

        freed = 0
        for victim in victims:
            # Deregister dedup BEFORE freeing (removes extra ref_count)
            if self._dedup:
                self._dedup.deregister_block(victim.block_id)
            # Use block_id to find the right block_idx (Gap #9: stale index fix)
            block = self._evict_by_block_id(victim.seq_id, victim.block_id)
            if block is not None:
                self._evictor.remove_block(block.block_id)
                freed += 1

        return self._pool.num_free >= blocks_needed

    def _evict_by_block_id(self, seq_id: int, block_id: int):
        """Evict a block by its ID (not index) to avoid stale index bugs."""
        blocks = self._page_table.get_blocks(seq_id)
        # Find the actual index of this block_id in the sequence
        entry = self._page_table.get_entry_or_none(seq_id)
        if entry is None:
            return None
        for idx, block in enumerate(entry.blocks):
            if idx not in entry.evicted_indices and block.block_id == block_id:
                return self._page_table.evict_block(seq_id, idx)
        return None

    def _trim_sliding_window(self, seq_id: int):
        """Trim blocks outside the sliding window (Gap #10)."""
        window = self._config.sliding_window
        total_tokens = self._page_table.num_tokens(seq_id)
        if total_tokens <= window:
            return

        # How many tokens are outside the window?
        excess_tokens = total_tokens - window
        # How many full blocks to trim from the start?
        blocks_to_trim = excess_tokens // self._block_size

        entry = self._page_table.get_entry_or_none(seq_id)
        if entry is None:
            return

        trimmed = 0
        for idx in range(len(entry.blocks)):
            if trimmed >= blocks_to_trim:
                break
            if idx in entry.evicted_indices:
                continue
            self._page_table.evict_block(seq_id, idx)
            trimmed += 1

    def mark_active(self, seq_id: int):
        """Mark a sequence as actively generating (protected from eviction)."""
        with self._lock:
            self._active_seq_ids.add(seq_id)

    def mark_idle(self, seq_id: int):
        """Mark a sequence as idle (eligible for eviction)."""
        with self._lock:
            self._active_seq_ids.discard(seq_id)

    def truncate_sequence(self, seq_id: int, keep_tokens: int):
        """Truncate a sequence's KV cache to keep_tokens length.

        Used by speculative decoding to roll back rejected draft tokens.
        Frees any blocks that are no longer needed after truncation.

        Args:
            seq_id: Sequence to truncate
            keep_tokens: Number of tokens to keep (must be <= current length)
        """
        with self._lock:
            entry = self._page_table.get_entry_or_none(seq_id)
            if entry is None:
                return

            current = entry.num_tokens
            if keep_tokens >= current:
                return  # Nothing to truncate

            # How many blocks needed for keep_tokens
            blocks_needed = (keep_tokens + self._block_size - 1) // self._block_size if keep_tokens > 0 else 0

            # Free excess blocks from the end
            while len(entry.blocks) > blocks_needed:
                block = entry.blocks.pop()
                idx = len(entry.blocks)
                if idx not in entry.evicted_indices:
                    # Deregister from dedup before freeing
                    if self._dedup:
                        self._dedup.deregister_block(block.block_id)
                    self._evictor.remove_block(block.block_id)
                    self._pool.free(block)
                entry.evicted_indices.discard(idx)

            # Adjust token count in the last remaining block
            if entry.blocks and keep_tokens > 0:
                tokens_in_prior = (blocks_needed - 1) * self._block_size
                entry.blocks[-1].num_tokens = keep_tokens - tokens_in_prior

            entry.num_tokens = keep_tokens

    def _try_dedup_blocks(self, seq_id: int, tokens: List[int]):
        """Try to deduplicate full blocks against existing cached blocks.

        Critical #2 fix: properly free duplicate blocks and update page table.
        """
        entry = self._page_table.get_entry_or_none(seq_id)
        if entry is None:
            return

        offset = 0
        for i in range(len(entry.blocks)):
            if i in entry.evicted_indices:
                continue
            block = entry.blocks[i]
            end = min(offset + self._block_size, len(tokens))
            block_tokens = tokens[offset:end]

            if len(block_tokens) == self._block_size:  # Only dedup full blocks
                existing_id = self._dedup.register_block(block.block_id, block_tokens)
                if existing_id is not None and existing_id != block.block_id:
                    # Duplicate found — replace this block with the existing one
                    existing_block = self._pool.get_block(existing_id)
                    # Free the duplicate block (our newly allocated one)
                    self._pool.free(block)
                    # Point page table entry to the shared block
                    entry.blocks[i] = existing_block
            offset = end

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_free_blocks(self) -> int:
        return self._pool.num_free

    @property
    def num_sequences(self) -> int:
        return self._page_table.num_sequences

    def max_tokens_capacity(self) -> int:
        """Maximum total tokens the cache can hold."""
        return self._pool.num_blocks * self._block_size

    def stats(self) -> CacheStats:
        pool = self._pool.stats()
        pt = self._page_table.stats()
        ev = self._evictor.stats()
        dd = self._dedup.stats() if self._dedup else {"dedup_hits": 0, "hit_rate": 0.0}

        return CacheStats(
            total_blocks=pool["num_blocks"],
            allocated_blocks=pool["num_allocated"],
            free_blocks=pool["num_free"],
            total_bytes=pool["total_bytes"],
            used_bytes=pool["used_bytes"],
            utilization=pool["utilization"],
            num_sequences=pt["num_sequences"],
            total_cached_tokens=pt["total_tokens"],
            eviction_count=ev["eviction_count"],
            tokens_evicted=ev["total_tokens_evicted"],
            eviction_policy=ev["policy"],
            dedup_hits=dd["dedup_hits"],
            dedup_hit_rate=dd["hit_rate"],
            block_size_tokens=self._block_size,
            block_bytes=pool["block_bytes"],
        )

    def summary(self) -> str:
        """Human-readable cache summary."""
        s = self.stats()
        return (
            f"ZSE KV Cache:\n"
            f"  Blocks: {s.allocated_blocks}/{s.total_blocks} "
            f"({s.utilization:.1%} used)\n"
            f"  Memory: {s.used_bytes / 1024**2:.1f}MB / "
            f"{s.total_bytes / 1024**2:.1f}MB\n"
            f"  Sequences: {s.num_sequences}, "
            f"Tokens: {s.total_cached_tokens:,}\n"
            f"  Block size: {s.block_size_tokens} tokens "
            f"({s.block_bytes:,} bytes/block)\n"
            f"  Evictions: {s.eviction_count} ({s.tokens_evicted:,} tokens)\n"
            f"  Dedup hit rate: {s.dedup_hit_rate:.1%}\n"
            f"  Policy: {s.eviction_policy}"
        )

    def destroy(self):
        """Release all GPU memory."""
        self._pool.destroy()
