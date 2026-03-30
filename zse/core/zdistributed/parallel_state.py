"""
Distributed Process Group Management

Initializes and manages NCCL process groups for tensor parallelism.
Handles multi-GPU communication on a single node.

Usage:
    from zse.core.zdistributed.parallel_state import (
        init_tensor_parallel,
        get_tp_group,
        get_tp_rank,
        get_tp_world_size,
        destroy_tensor_parallel,
    )
    
    init_tensor_parallel(tp_size=4)
    rank = get_tp_rank()       # 0, 1, 2, or 3
    world = get_tp_world_size() # 4
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


# Global state
_TP_GROUP: Optional[dist.ProcessGroup] = None
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1
_INITIALIZED: bool = False


def is_initialized() -> bool:
    return _INITIALIZED


def get_tp_group() -> Optional[dist.ProcessGroup]:
    return _TP_GROUP


def get_tp_rank() -> int:
    return _TP_RANK


def get_tp_world_size() -> int:
    return _TP_WORLD_SIZE


def init_tensor_parallel(
    tp_size: int,
    rank: Optional[int] = None,
    backend: str = "nccl",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> None:
    """
    Initialize tensor parallel process group.
    
    For single-node multi-GPU, this is called once per process.
    Each GPU runs as a separate process with a unique rank.
    
    Args:
        tp_size: Number of GPUs for tensor parallelism
        rank: This process's rank (auto-detected from env if None)
        backend: Communication backend ("nccl" for GPU)
        master_addr: Address of rank 0 process
        master_port: Port for rendezvous
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE, _INITIALIZED
    
    if _INITIALIZED:
        return
    
    if tp_size <= 1:
        _TP_RANK = 0
        _TP_WORLD_SIZE = 1
        _INITIALIZED = True
        return
    
    # Set env vars for torch.distributed
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            world_size=tp_size,
            rank=rank,
        )
    
    # Create TP group with all ranks
    tp_ranks = list(range(tp_size))
    _TP_GROUP = dist.new_group(tp_ranks)
    _TP_RANK = rank
    _TP_WORLD_SIZE = tp_size
    _INITIALIZED = True
    
    # Bind this process to its GPU
    torch.cuda.set_device(rank)


def destroy_tensor_parallel() -> None:
    """Clean up distributed state."""
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE, _INITIALIZED
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    _TP_GROUP = None
    _TP_RANK = 0
    _TP_WORLD_SIZE = 1
    _INITIALIZED = False


def init_single_process_tp(tp_size: int) -> None:
    """
    Initialize TP state for single-process multi-GPU (no spawning).
    
    This is used when the orchestrator manages all GPUs from one process
    (e.g., for serving). Communication uses direct CUDA operations instead
    of NCCL process groups.
    
    Args:
        tp_size: Number of GPUs to use
    """
    global _TP_RANK, _TP_WORLD_SIZE, _INITIALIZED
    
    _TP_RANK = 0  # Single process is always rank 0
    _TP_WORLD_SIZE = tp_size
    _INITIALIZED = True
