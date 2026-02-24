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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zse.core.zdistributed.tensor_parallel import TensorParallel
    from zse.core.zdistributed.pipeline_parallel import PipelineParallel
    from zse.core.zdistributed.hybrid_offload import HybridOffload

__all__ = [
    "TensorParallel",
    "PipelineParallel",
    "HybridOffload",
]
