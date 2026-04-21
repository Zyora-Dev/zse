"""ZSE Attention Metadata — Packed GPU metadata for PagedAttention kernel.

The attention kernel needs to know which blocks belong to which sequence
and how many tokens each sequence has. This struct packs that info into
GPU-resident tensors.
"""

import struct
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AttentionMetadata:
    """Metadata for a batch of sequences, consumed by the attention kernel.

    Attributes:
        block_tables: Flat list of block IDs, shape [num_seqs, max_blocks_per_seq].
                      Padded with -1 for sequences with fewer blocks.
        seq_lengths: Number of tokens per sequence, shape [num_seqs].
        num_seqs: Batch size.
        max_seq_len: Maximum sequence length in this batch.
        max_blocks_per_seq: Maximum number of blocks per sequence.
        block_size: Tokens per block.

    GPU tensors are packed as int32 arrays. The caller uploads them
    to GPU via GPUMemory.copy_host_to_device().
    """
    block_tables: List[List[int]]  # [num_seqs][max_blocks_per_seq]
    seq_lengths: List[int]         # [num_seqs]
    num_seqs: int
    max_seq_len: int
    max_blocks_per_seq: int
    block_size: int
    # Prefill/decode phase tracking (Gap #12)
    is_prefill: List[bool] = None  # Per-sequence: True = prefill, False = decode

    # GPU pointers (set after upload)
    block_tables_gpu_ptr: int = 0
    seq_lengths_gpu_ptr: int = 0

    def pack_block_tables(self) -> bytes:
        """Pack block_tables as flat int32 array for GPU upload.

        Layout: row-major [num_seqs, max_blocks_per_seq], padded with -1.
        """
        parts = []
        for row in self.block_tables:
            padded = row + [-1] * (self.max_blocks_per_seq - len(row))
            parts.append(struct.pack(f'<{self.max_blocks_per_seq}i', *padded))
        return b''.join(parts)

    def pack_seq_lengths(self) -> bytes:
        """Pack seq_lengths as int32 array for GPU upload."""
        return struct.pack(f'<{self.num_seqs}i', *self.seq_lengths)

    @property
    def block_tables_nbytes(self) -> int:
        return self.num_seqs * self.max_blocks_per_seq * 4

    @property
    def seq_lengths_nbytes(self) -> int:
        return self.num_seqs * 4


def build_attention_metadata(
    block_ids_per_seq: List[List[int]],
    seq_lengths: List[int],
    block_size: int,
) -> AttentionMetadata:
    """Build AttentionMetadata from page table state.

    Args:
        block_ids_per_seq: block IDs for each sequence
        seq_lengths: token count per sequence
        block_size: tokens per block
    """
    num_seqs = len(seq_lengths)
    max_blocks = max((len(b) for b in block_ids_per_seq), default=0)
    max_seq_len = max(seq_lengths, default=0)

    return AttentionMetadata(
        block_tables=block_ids_per_seq,
        seq_lengths=seq_lengths,
        num_seqs=num_seqs,
        max_seq_len=max_seq_len,
        max_blocks_per_seq=max_blocks,
        block_size=block_size,
    )
