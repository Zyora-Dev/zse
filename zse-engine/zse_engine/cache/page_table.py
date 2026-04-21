"""ZSE Page Table — Per-sequence block mapping for paged KV cache.

Maps (sequence_id, token_position) → block. Each sequence has an ordered
list of blocks forming its KV cache.

Supports copy-on-write forking for beam search: shared blocks have
ref_count > 1, and writes trigger a copy with GPU data transfer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

from zse_engine.cache.block_pool import BlockPool, Block


@dataclass
class SequenceEntry:
    """Tracks one sequence's KV cache blocks."""
    seq_id: int
    blocks: List[Block] = field(default_factory=list)
    num_tokens: int = 0
    # Tracks which block indices have been evicted (holes)
    evicted_indices: set = field(default_factory=set)

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_live_blocks(self) -> int:
        return len(self.blocks) - len(self.evicted_indices)


class PageTable:
    """Manages block→token mappings for all active sequences.

    Usage:
        pt = PageTable(block_pool)
        pt.create_sequence(0)
        pt.append_tokens(0, 5)   # allocate blocks for 5 new tokens
        block_ids = pt.get_block_ids(0)
        pt.remove_sequence(0)    # free all blocks
    """

    def __init__(self, pool: BlockPool, gpu_copy_fn: Optional[Callable] = None):
        """
        Args:
            pool: Block allocator
            gpu_copy_fn: Optional fn(src_block, dst_block) for GPU D2D copy.
                         Called during COW. If None, COW does metadata-only copy.
        """
        self._pool = pool
        self._sequences: Dict[int, SequenceEntry] = {}
        self._gpu_copy_fn = gpu_copy_fn

    def create_sequence(self, seq_id: int):
        """Create a new empty sequence."""
        if seq_id in self._sequences:
            raise ValueError(f"Sequence {seq_id} already exists")
        self._sequences[seq_id] = SequenceEntry(seq_id=seq_id)

    def has_sequence(self, seq_id: int) -> bool:
        return seq_id in self._sequences

    def get_entry(self, seq_id: int) -> 'SequenceEntry':
        """Get the sequence entry for a given seq_id. Raises KeyError if not found."""
        return self._sequences[seq_id]

    def get_entry_or_none(self, seq_id: int):
        """Get the sequence entry, or None if not found."""
        return self._sequences.get(seq_id)

    def append_tokens(self, seq_id: int, num_tokens: int) -> List[int]:
        """Extend a sequence by num_tokens, allocating blocks as needed.

        Returns list of block_ids that were newly allocated.
        """
        if num_tokens <= 0:
            return []

        entry = self._sequences[seq_id]
        new_block_ids = []

        for _ in range(num_tokens):
            # Do we need a new block?
            if not entry.blocks or self._pool.is_block_full(entry.blocks[-1]):
                block = self._pool.alloc(seq_id=seq_id)
                block.token_start = entry.num_tokens
                entry.blocks.append(block)
                new_block_ids.append(block.block_id)

            # The last block gets one more token
            last = entry.blocks[-1]

            # COW check: if shared, we need our own copy before writing
            if last.is_shared:
                new_block = self._pool.alloc(seq_id=seq_id)
                new_block.token_start = last.token_start
                new_block.num_tokens = last.num_tokens

                # GPU data copy (Critical fix: actually copy the data)
                if self._gpu_copy_fn is not None:
                    self._gpu_copy_fn(last, new_block)

                self._pool.free(last)  # decrement ref on shared block
                entry.blocks[-1] = new_block
                last = new_block
                new_block_ids.append(new_block.block_id)

            last.num_tokens += 1
            entry.num_tokens += 1

        return new_block_ids

    def get_block_ids(self, seq_id: int) -> List[int]:
        """Get ordered list of block IDs for a sequence.

        Evicted blocks (holes) return -1 to signal the attention kernel
        that those positions need recomputation.
        """
        entry = self._sequences[seq_id]
        result = []
        for i, b in enumerate(entry.blocks):
            if i in entry.evicted_indices:
                result.append(-1)
            else:
                result.append(b.block_id)
        return result

    def get_blocks(self, seq_id: int) -> List[Block]:
        """Get ordered list of live blocks (no evicted holes)."""
        entry = self._sequences[seq_id]
        return [b for i, b in enumerate(entry.blocks) if i not in entry.evicted_indices]

    def num_tokens(self, seq_id: int) -> int:
        return self._sequences[seq_id].num_tokens

    def num_blocks(self, seq_id: int) -> int:
        return self._sequences[seq_id].num_blocks

    def remove_sequence(self, seq_id: int) -> List[Block]:
        """Remove a sequence and free its blocks. Returns freed blocks."""
        entry = self._sequences.pop(seq_id)
        freed = []
        for i, block in enumerate(entry.blocks):
            if i in entry.evicted_indices:
                continue  # Already freed
            actually_freed = self._pool.free(block)
            if actually_freed:
                freed.append(block)
        return freed

    def fork_sequence(self, src_id: int, dst_id: int):
        """Fork a sequence (copy-on-write) for beam search.

        The new sequence shares all blocks with the source.
        Blocks get ref_count incremented; on next write they'll be copied.
        """
        if dst_id in self._sequences:
            raise ValueError(f"Sequence {dst_id} already exists")

        src = self._sequences[src_id]
        dst_entry = SequenceEntry(
            seq_id=dst_id,
            blocks=list(src.blocks),  # Shallow copy — same Block objects
            num_tokens=src.num_tokens,
            evicted_indices=set(src.evicted_indices),
        )

        # Increment ref counts on all live shared blocks
        for i, block in enumerate(dst_entry.blocks):
            if i not in dst_entry.evicted_indices:
                self._pool.increment_ref(block)

        self._sequences[dst_id] = dst_entry

    def evict_block(self, seq_id: int, block_idx: int) -> Optional[Block]:
        """Evict a specific block from a sequence (token-level eviction).

        Marks the block as evicted (hole) rather than removing it, preserving
        positional mapping. get_block_ids returns -1 for evicted positions.

        Returns the evicted block, or None if invalid/already evicted.
        """
        entry = self._sequences.get(seq_id)
        if entry is None or block_idx >= len(entry.blocks):
            return None
        if block_idx in entry.evicted_indices:
            return None  # Already evicted

        block = entry.blocks[block_idx]
        tokens_in_block = block.num_tokens
        entry.evicted_indices.add(block_idx)
        entry.num_tokens -= tokens_in_block

        self._pool.free(block)
        return block

    def get_all_sequences(self) -> List[int]:
        """Get all active sequence IDs."""
        return list(self._sequences.keys())

    @property
    def num_sequences(self) -> int:
        return len(self._sequences)

    def stats(self) -> dict:
        total_tokens = sum(e.num_tokens for e in self._sequences.values())
        total_blocks = sum(e.num_live_blocks for e in self._sequences.values())
        return {
            "num_sequences": self.num_sequences,
            "total_tokens": total_tokens,
            "total_blocks": total_blocks,
        }
