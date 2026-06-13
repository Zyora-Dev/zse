"""ZSE Block Pool — GPU memory block allocator for KV cache.

Pre-allocates a contiguous GPU memory slab, divides into fixed-size blocks.
Each block holds KV cache data for `block_size` tokens across ALL layers.

Unlike vLLM (one block per layer), we pack all layers into one block:
    block_bytes = block_size * num_kv_heads * head_dim * 2 (fp16) * 2 (K+V) * num_layers

This gives better spatial locality when the attention kernel accesses KV
for a token range across all layers.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Deque
from collections import deque


@dataclass
class Block:
    """A single KV cache block in GPU memory."""
    block_id: int
    gpu_offset: int          # Byte offset into the GPU slab
    ref_count: int = 1       # >1 means shared (COW)
    content_hash: Optional[int] = None  # For dedup
    seq_id: Optional[int] = None        # Which sequence owns this
    token_start: int = 0     # First token index in this block
    num_tokens: int = 0      # How many tokens written so far

    @property
    def is_shared(self) -> bool:
        return self.ref_count > 1


class BlockPool:
    """GPU memory pool divided into fixed-size blocks.

    Usage:
        pool = BlockPool(gpu_mem, total_bytes=2*1024**3, block_size_tokens=16, config=config)
        block = pool.alloc()
        pool.free(block)
    """

    def __init__(
        self,
        gpu_mem,  # GPUMemory instance (or None for CPU-only testing)
        total_bytes: int,
        block_size_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        kv_dtype_bytes: int = 2,  # float16 = 2 bytes
        per_layer_kv_elems: Optional[List[int]] = None,
    ):
        self._gpu_mem = gpu_mem
        self._block_size_tokens = block_size_tokens
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._num_layers = num_layers
        self._kv_dtype_bytes = kv_dtype_bytes

        # Per-layer KV element count (K only; V is the same size) PER TOKEN.
        # Uniform models (Llama/Qwen/Gemma2): every layer = num_kv_heads*head_dim.
        # Gemma 4: sliding layers = 8*256, full layers = 1*512 — varies per layer.
        if per_layer_kv_elems is None:
            per_layer_kv_elems = [num_kv_heads * head_dim] * num_layers
        if len(per_layer_kv_elems) != num_layers:
            raise ValueError(
                f"per_layer_kv_elems has {len(per_layer_kv_elems)} entries, "
                f"expected num_layers={num_layers}"
            )
        self._per_layer_kv_elems = list(per_layer_kv_elems)

        # Per-layer bytes per token (K + V). Cumulative offsets give each layer's
        # slot start within a block (used by the per-layer cache-write / attention
        # kernels). For uniform models these collapse to layer * uniform_stride.
        self._per_layer_token_bytes = [
            e * kv_dtype_bytes * 2 for e in self._per_layer_kv_elems  # *2 = K and V
        ]
        self._layer_byte_offsets: List[int] = []
        acc = 0
        for tb in self._per_layer_token_bytes:
            self._layer_byte_offsets.append(acc)
            acc += tb * block_size_tokens
        self._block_bytes = acc  # block_size * sum(per-layer K+V bytes)

        # Back-compat scalar fields (uniform case). For non-uniform models these
        # describe layer 0 only; callers needing exact per-layer math use the
        # per-layer accessors below.
        self._bytes_per_token = self._per_layer_token_bytes[0]
        self._bytes_per_token_all_layers = sum(self._per_layer_token_bytes)

        # Cap budget to actual free GPU memory (minus 512MB safety margin)
        # to handle fragmentation after weight uploads
        if gpu_mem is not None:
            try:
                actual_free = gpu_mem.get_free_memory()
                safe_budget = max(0, actual_free - 512 * 1024**2)
                if total_bytes > safe_budget:
                    total_bytes = safe_budget
            except Exception:
                pass

        # How many blocks fit in the budget
        self._num_blocks = total_bytes // self._block_bytes
        if self._num_blocks == 0:
            raise ValueError(
                f"Budget {total_bytes:,} bytes too small for even 1 block "
                f"({self._block_bytes:,} bytes/block)"
            )

        self._total_bytes = self._num_blocks * self._block_bytes

        # Allocate GPU slab (if gpu_mem provided)
        self._gpu_base_ptr = 0
        if gpu_mem is not None:
            from zse_compiler.types.dtypes import uint8
            from zse_compiler.types.tensor import Tensor
            self._slab = gpu_mem.allocate(
                shape=(self._total_bytes,), dtype=uint8
            )
            self._gpu_base_ptr = self._slab.data_ptr
            gpu_mem.memset(self._slab, 0)

        # Create all blocks and put them on the free list
        self._blocks: List[Block] = []
        self._free_list: Deque[int] = deque()

        for i in range(self._num_blocks):
            block = Block(
                block_id=i,
                gpu_offset=i * self._block_bytes,
            )
            self._blocks.append(block)
            self._free_list.append(i)

        self._num_allocated = 0

    @property
    def block_size_tokens(self) -> int:
        return self._block_size_tokens

    @property
    def block_bytes(self) -> int:
        return self._block_bytes

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free_list)

    @property
    def num_allocated(self) -> int:
        return self._num_allocated

    @property
    def bytes_per_token_all_layers(self) -> int:
        return self._bytes_per_token_all_layers

    @property
    def is_uniform_kv(self) -> bool:
        """True if every layer has the same KV size (Llama/Qwen/Gemma2)."""
        return len(set(self._per_layer_kv_elems)) == 1

    def layer_kv_elems(self, layer: int) -> int:
        """K-element count per token for a given layer (V is the same size)."""
        return self._per_layer_kv_elems[layer]

    def layer_byte_offset(self, layer: int) -> int:
        """Byte offset of a layer's KV slot within a block (K+V interleaved)."""
        return self._layer_byte_offsets[layer]

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def used_bytes(self) -> int:
        return self._num_allocated * self._block_bytes

    @property
    def utilization(self) -> float:
        if self._num_blocks == 0:
            return 0.0
        return self._num_allocated / self._num_blocks

    def alloc(self, seq_id: Optional[int] = None) -> Block:
        """Allocate a block from the free list. O(1).

        Raises RuntimeError if no blocks available.
        """
        if not self._free_list:
            raise RuntimeError(
                f"KV cache exhausted: all {self._num_blocks} blocks in use "
                f"({self._total_bytes / 1024**2:.0f} MB)"
            )

        block_id = self._free_list.popleft()
        block = self._blocks[block_id]
        block.ref_count = 1
        block.content_hash = None
        block.seq_id = seq_id
        block.token_start = 0
        block.num_tokens = 0
        self._num_allocated += 1
        return block

    def free(self, block: Block) -> bool:
        """Free a block. Decrements ref_count; actually frees when it hits 0.

        Returns True if the block was actually freed (ref_count reached 0).
        """
        if block.ref_count <= 0:
            return False  # Already freed — prevent double-free
        block.ref_count -= 1
        if block.ref_count <= 0:
            block.ref_count = 0
            block.content_hash = None
            block.seq_id = None
            block.token_start = 0
            block.num_tokens = 0
            self._free_list.append(block.block_id)
            self._num_allocated -= 1
            return True
        return False

    def get_block(self, block_id: int) -> Block:
        """Get block by ID."""
        return self._blocks[block_id]

    def is_block_full(self, block: Block) -> bool:
        """Check if a block has reached its token capacity."""
        return block.num_tokens >= self._block_size_tokens

    def gpu_ptr_for_block(self, block: Block) -> int:
        """Get the raw GPU pointer for a block's start address."""
        return self._gpu_base_ptr + block.gpu_offset

    def gpu_ptr_for_block_layer(self, block: Block, layer: int) -> int:
        """Get GPU pointer for a specific layer within a block.

        Block layout: [layer0_K | layer0_V | layer1_K | layer1_V | ...]
        Each layer section: block_size_tokens * num_kv_heads * head_dim * dtype_bytes
        """
        layer_offset = layer * self._block_size_tokens * self._bytes_per_token
        return self._gpu_base_ptr + block.gpu_offset + layer_offset

    def gpu_ptr_for_kv(self, block: Block, layer: int, is_value: bool) -> int:
        """Get GPU pointer for K or V within a specific layer of a block.

        Layout per layer: [K: block_size * kv_heads * head_dim | V: same]
        """
        layer_offset = layer * self._block_size_tokens * self._bytes_per_token
        kv_section_bytes = self._block_size_tokens * self._num_kv_heads * self._head_dim * self._kv_dtype_bytes
        v_offset = kv_section_bytes if is_value else 0
        return self._gpu_base_ptr + block.gpu_offset + layer_offset + v_offset

    def increment_ref(self, block: Block):
        """Increment reference count (for COW sharing)."""
        block.ref_count += 1

    def stats(self) -> dict:
        """Return pool statistics."""
        return {
            "num_blocks": self._num_blocks,
            "num_allocated": self._num_allocated,
            "num_free": self.num_free,
            "block_size_tokens": self._block_size_tokens,
            "block_bytes": self._block_bytes,
            "total_bytes": self._total_bytes,
            "used_bytes": self.used_bytes,
            "utilization": self.utilization,
        }

    def destroy(self):
        """Free the GPU slab."""
        if self._gpu_mem is not None and self._gpu_base_ptr != 0:
            self._gpu_mem.free(self._slab)
            self._gpu_base_ptr = 0
            self._slab = None
