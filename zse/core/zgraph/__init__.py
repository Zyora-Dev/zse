"""
zGraph - CUDA Graph Execution

Captures and replays CUDA graphs for maximum inference throughput:
- Eliminates kernel launch overhead
- Reduces decode latency by 20-30%
- Consistent latency for real-time applications

Usage:
    from zse.core.zgraph import CUDAGraphRunner
    
    runner = CUDAGraphRunner(model, max_batch_size=32)
    
    # First call captures, subsequent calls replay
    logits = runner.decode_step(input_ids, position_ids)
"""

from .cuda_graph import (
    CUDAGraphRunner,
    BatchedGraphRunner,
    CapturedGraph,
    GraphState,
    capture_model_graph,
    is_cuda_graph_compatible,
)

__all__ = [
    "CUDAGraphRunner",
    "BatchedGraphRunner", 
    "CapturedGraph",
    "GraphState",
    "capture_model_graph",
    "is_cuda_graph_compatible",
]
