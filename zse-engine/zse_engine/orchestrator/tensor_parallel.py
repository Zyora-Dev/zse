"""ZSE Tensor Parallel Group — Multi-GPU coordination for tensor parallelism.

Manages a group of GPUs running in parallel, each holding a shard of the model.
Uses NCCL/RCCL for inter-GPU communication via pure ctypes.

Architecture (Megatron-LM style):
    - Column parallel: QKV, Gate, Up projections (split output dim)
    - Row parallel: O, Down projections (split input dim, all-reduce after)
    - Replicated: LayerNorm, Embedding, Residuals

Usage:
    # On each process/rank:
    tp = TensorParallelGroup(tp_size=2, rank=0, backend="cuda", unique_id=uid_bytes)
    tp.all_reduce_inplace(buf_ptr, num_elements, dtype="float16")
    tp.destroy()

For multi-process:
    tp_launch(tp_size=2, rank_fn=my_init_fn, backend="cuda")
"""

import ctypes
import os
import struct
import tempfile
import multiprocessing
from typing import Optional, Callable, Any, List
from dataclasses import dataclass

from zse_compiler.runtime.nccl import (
    NcclCommunicator, get_unique_id, is_nccl_available,
    NCCL_UNIQUE_ID_BYTES, _DTYPE_SIZE, _DTYPE_MAP,
)


# Weight split strategies
COLUMN_PARALLEL = "column"  # Split along output dim (N)
ROW_PARALLEL = "row"        # Split along input dim (K)
REPLICATED = "replicated"   # Full copy on each GPU

# Standard Llama/Qwen weight split map
WEIGHT_SPLIT_MAP = {
    # Attention
    "self_attn.q_proj.weight": COLUMN_PARALLEL,
    "self_attn.k_proj.weight": COLUMN_PARALLEL,
    "self_attn.v_proj.weight": COLUMN_PARALLEL,
    "self_attn.q_proj.bias": COLUMN_PARALLEL,
    "self_attn.k_proj.bias": COLUMN_PARALLEL,
    "self_attn.v_proj.bias": COLUMN_PARALLEL,
    "self_attn.o_proj.weight": ROW_PARALLEL,
    # MLP
    "mlp.gate_proj.weight": COLUMN_PARALLEL,
    "mlp.up_proj.weight": COLUMN_PARALLEL,
    "mlp.down_proj.weight": ROW_PARALLEL,
    # Norms — replicated
    "input_layernorm.weight": REPLICATED,
    "post_attention_layernorm.weight": REPLICATED,
}

# Non-layer weights
NON_LAYER_SPLIT_MAP = {
    "embed_tokens.weight": REPLICATED,
    "model.norm.weight": REPLICATED,
    "lm_head.weight": COLUMN_PARALLEL,
}


@dataclass
class TPConfig:
    """Tensor parallelism configuration."""
    tp_size: int = 1           # Number of GPUs
    backend: str = "cuda"      # "cuda" or "rocm"

    def is_enabled(self) -> bool:
        return self.tp_size > 1

    def validate(self, num_heads: int, num_kv_heads: int, intermediate_size: int):
        """Validate that model dimensions are divisible by tp_size."""
        if self.tp_size <= 1:
            return
        if num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by tp_size ({self.tp_size})"
            )
        if num_kv_heads % self.tp_size != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by tp_size ({self.tp_size})"
            )
        if intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by tp_size ({self.tp_size})"
            )


class TensorParallelGroup:
    """Manages inter-GPU communication for tensor parallelism.

    One instance per GPU rank. All ranks must call collective operations
    in the same order.

    Args:
        tp_size: Number of GPUs in the group
        rank: This GPU's index (0 .. tp_size-1)
        backend: "cuda" or "rocm"
        unique_id: Shared NCCL unique ID bytes (from get_unique_id on rank 0)
        stream: GPU stream for async NCCL operations (0 = default)
    """

    def __init__(
        self,
        tp_size: int,
        rank: int,
        backend: str = "cuda",
        unique_id: Optional[bytes] = None,
        stream: int = 0,
    ):
        self.tp_size = tp_size
        self.rank = rank
        self.backend = backend
        self._stream = stream
        self._comm = None

        if tp_size <= 1:
            return  # No-op for single GPU

        if unique_id is None:
            raise ValueError("unique_id required for tp_size > 1")

        self._comm = NcclCommunicator(
            nranks=tp_size,
            rank=rank,
            unique_id=unique_id,
            backend=backend,
            stream=stream,
        )

    def all_reduce_inplace(
        self,
        buf_ptr: int,
        count: int,
        dtype: str = "float16",
        op: str = "sum",
    ):
        """In-place all-reduce: sum partial results across all ranks.

        Used after row-parallel matmuls (O projection, down projection).
        Each rank has a partial sum; after all-reduce, each rank has the full sum.

        Args:
            buf_ptr: GPU pointer to buffer (read + overwritten with result)
            count: Number of elements (not bytes)
            dtype: "float16" or "float32"
            op: "sum" (standard for TP)
        """
        if self.tp_size <= 1:
            return  # No-op
        self._comm.all_reduce_inplace(buf_ptr, count, dtype, op)

    def all_gather(
        self,
        send_ptr: int,
        recv_ptr: int,
        sendcount: int,
        dtype: str = "float16",
    ):
        """All-gather: concatenate shards from all ranks.

        Used for gathering LM head logits when vocab is split across GPUs.

        Args:
            send_ptr: GPU pointer to this rank's shard
            recv_ptr: GPU pointer to full output (sendcount * tp_size elements)
            sendcount: Elements per rank
        """
        if self.tp_size <= 1:
            return
        self._comm.all_gather(send_ptr, recv_ptr, sendcount, dtype)

    def broadcast(
        self,
        buf_ptr: int,
        count: int,
        dtype: str = "int32",
        root: int = 0,
    ):
        """Broadcast from root rank to all.

        Used to distribute token IDs from rank 0 to all ranks.
        """
        if self.tp_size <= 1:
            return
        self._comm.broadcast(buf_ptr, count, dtype, root)

    def barrier(self):
        """Synchronize all ranks."""
        if self.tp_size <= 1:
            return
        self._comm.barrier()

    def sync_stream(self):
        """Wait for all NCCL operations on this rank's stream to complete."""
        if self.tp_size <= 1:
            return
        self._comm.stream_synchronize()

    def get_split_strategy(self, weight_name: str) -> str:
        """Determine how a weight should be split for TP.

        Args:
            weight_name: Full weight name (e.g. "model.layers.0.self_attn.q_proj.weight")

        Returns:
            COLUMN_PARALLEL, ROW_PARALLEL, or REPLICATED
        """
        # Check non-layer weights first
        for key, strategy in NON_LAYER_SPLIT_MAP.items():
            if weight_name.endswith(key) or weight_name == key:
                return strategy

        # Check layer weights (strip "model.layers.N." prefix)
        for key, strategy in WEIGHT_SPLIT_MAP.items():
            if key in weight_name:
                return strategy

        # Default: replicate unknown weights
        return REPLICATED

    def compute_shard_range(
        self,
        full_size: int,
        strategy: str,
    ) -> tuple:
        """Compute (start, end) indices for this rank's shard.

        Args:
            full_size: Full dimension size (N for column, K for row)
            strategy: COLUMN_PARALLEL or ROW_PARALLEL

        Returns:
            (start_idx, end_idx) — slice range for this rank
        """
        if strategy == REPLICATED or self.tp_size <= 1:
            return (0, full_size)

        shard_size = full_size // self.tp_size
        start = self.rank * shard_size
        end = start + shard_size
        return (start, end)

    def shard_size(self, full_size: int, strategy: str) -> int:
        """Get the size of this rank's shard for a given dimension."""
        if strategy == REPLICATED or self.tp_size <= 1:
            return full_size
        return full_size // self.tp_size

    def destroy(self):
        """Clean up NCCL communicator."""
        if self._comm is not None:
            self._comm.destroy()
            self._comm = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"TensorParallelGroup(tp={self.tp_size}, rank={self.rank}, backend={self.backend})"


def tp_launch(
    tp_size: int,
    rank_fn: Callable[[int, int, bytes, str], Any],
    backend: str = "cuda",
) -> List[Any]:
    """Launch tensor parallel processes.

    Creates tp_size processes, each running rank_fn with its rank.
    Handles NCCL unique ID distribution automatically.

    Args:
        tp_size: Number of GPUs
        rank_fn: Function(rank, tp_size, unique_id_bytes, backend) → result
        backend: "cuda" or "rocm"

    Returns:
        List of results from each rank (via mp.Queue)
    """
    if tp_size <= 1:
        raise ValueError("tp_launch requires tp_size >= 2")

    if not is_nccl_available(backend):
        lib_name = "RCCL" if backend == "rocm" else "NCCL"
        raise RuntimeError(f"{lib_name} not available. Install it for multi-GPU support.")

    # Generate unique ID on main process
    uid_bytes = get_unique_id(backend)

    # Share via temp file (works across processes)
    uid_path = os.path.join(tempfile.gettempdir(), f"zse_nccl_uid_{os.getpid()}.bin")
    with open(uid_path, "wb") as f:
        f.write(uid_bytes)

    result_queue = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()

    def _worker(rank, tp_size, uid_path, backend, result_q, error_q):
        try:
            with open(uid_path, "rb") as f:
                uid = f.read()
            result = rank_fn(rank, tp_size, uid, backend)
            result_q.put((rank, result))
        except Exception as e:
            error_q.put((rank, str(e)))

    processes = []
    for rank in range(tp_size):
        p = multiprocessing.Process(
            target=_worker,
            args=(rank, tp_size, uid_path, backend, result_queue, error_queue),
        )
        p.start()
        processes.append(p)

    # Wait for all processes
    for p in processes:
        p.join(timeout=600)  # 10 min timeout

    # Clean up
    try:
        os.unlink(uid_path)
    except OSError:
        pass

    # Check for errors
    errors = []
    while not error_queue.empty():
        rank, msg = error_queue.get_nowait()
        errors.append(f"Rank {rank}: {msg}")

    if errors:
        raise RuntimeError(f"Tensor parallel launch failed:\n" + "\n".join(errors))

    # Collect results
    results = [None] * tp_size
    while not result_queue.empty():
        rank, result = result_queue.get_nowait()
        results[rank] = result

    return results
