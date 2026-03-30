"""
zDistributed - Distributed Inference

Implements multi-device inference:
- Tensor parallelism (split layers across GPUs)
- Pipeline parallelism (split model across GPUs)
- Hybrid CPU offload (GPU + CPU memory)
- Multi-node support (multiple machines)

Key Features:
- Seamless multi-GPU scaling
- Smart layer placement based on memory
- Async communication to hide latency
- NCCL backend for GPU-GPU transfer
"""

from zse.core.zdistributed.tensor_parallel import (
    TensorParallel,
    apply_tensor_parallel,
    ColumnParallelLinear,
    RowParallelLinear,
    ColumnParallelQuantized,
    RowParallelQuantized,
    VocabParallelEmbedding,
)
from zse.core.zdistributed.parallel_state import (
    init_tensor_parallel,
    init_single_process_tp,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    is_initialized,
    destroy_tensor_parallel,
)
from zse.core.zdistributed.pipeline_parallel import PPCoordinator, compute_stage_assignments
from zse.core.zdistributed.hybrid_offload import HybridOffloadCoordinator, OffloadModelWrapper
from zse.core.zdistributed.tp_pp_parallel import TPPPCoordinator
from zse.core.zdistributed.worker import TPCoordinator
from zse.core.zdistributed.model_wrapper import TPModelWrapper, PPModelWrapper

__all__ = [
    # Tensor parallelism
    "TensorParallel",
    "apply_tensor_parallel",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "ColumnParallelQuantized",
    "RowParallelQuantized",
    "VocabParallelEmbedding",
    # Process group management
    "init_tensor_parallel",
    "init_single_process_tp",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_world_size",
    "is_initialized",
    "destroy_tensor_parallel",
    # Multi-process coordination
    "TPCoordinator",
    "TPModelWrapper",
    # Pipeline parallelism
    "PPCoordinator",
    "PPModelWrapper",
    "compute_stage_assignments",
    # Hybrid CPU offload
    "HybridOffloadCoordinator",
    "OffloadModelWrapper",
    # Combined TP+PP
    "TPPPCoordinator",
]
