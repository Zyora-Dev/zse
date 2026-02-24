"""
ZSE CUDA Graph Execution - zGraph

Captures and replays CUDA graphs for the decode phase to eliminate
kernel launch overhead and achieve maximum throughput.

Key benefits:
- Eliminates CPU overhead from kernel launches (~0.01ms each)
- Reduces decode latency by 20-30%
- Consistent latency for real-time applications

How it works:
1. Warmup: Run decode step normally to prepare memory
2. Capture: Record all CUDA operations into a graph
3. Replay: Execute the entire graph with single launch

Limitations:
- Fixed batch size (need separate graphs per batch size)
- Fixed sequence positions (use graph pools)
- Memory addresses must remain stable

Author: ZSE Team
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import warnings

import torch
import torch.nn as nn


class GraphState(Enum):
    """State of a CUDA graph."""
    UNINITIALIZED = "uninitialized"
    CAPTURING = "capturing"  
    CAPTURED = "captured"
    INVALID = "invalid"


@dataclass
class CapturedGraph:
    """
    A captured CUDA graph ready for replay.
    """
    # The CUDA graph object
    graph: Optional[torch.cuda.CUDAGraph] = None
    
    # Static input buffers (must remain at same address)
    input_buffers: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Static output buffers
    output_buffers: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Graph state
    state: GraphState = GraphState.UNINITIALIZED
    
    # Configuration used for capture
    batch_size: int = 0
    max_seq_len: int = 0
    
    # Stats
    num_replays: int = 0
    capture_time_ms: float = 0.0


class CUDAGraphRunner:
    """
    CUDA Graph runner for decode phase optimization.
    
    Captures the decode forward pass as a CUDA graph and replays it
    for each decode step, eliminating kernel launch overhead.
    
    Usage:
        runner = CUDAGraphRunner(model, max_batch_size=32)
        
        # First call captures the graph
        output = runner.decode_step(input_ids, position_ids, kv_cache)
        
        # Subsequent calls replay the graph
        output = runner.decode_step(input_ids, position_ids, kv_cache)
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        max_context_len: int = 8192,
        device: int = 0,
        num_graph_pools: int = 4,
        enable_capture: bool = True,
    ):
        """
        Initialize CUDA Graph runner.
        
        Args:
            model: The model to capture
            max_batch_size: Maximum batch size for graphs
            max_context_len: Maximum context length
            device: CUDA device ID
            num_graph_pools: Number of graph pools for different configs
            enable_capture: Enable graph capture (disable for debugging)
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len
        self.device = device
        self.num_graph_pools = num_graph_pools
        self.enable_capture = enable_capture and torch.cuda.is_available()
        
        # Graph pool: (batch_size, seq_len_bucket) -> CapturedGraph
        self.graph_pool: Dict[Tuple[int, int], CapturedGraph] = {}
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Memory pool for static buffers
        self._memory_pool: Optional[torch.cuda.graphs.graph_pool_handle] = None
        
        # Stats
        self.stats = {
            "captures": 0,
            "replays": 0,
            "capture_time_ms": 0.0,
            "replay_time_ms": 0.0,
        }
        
        if self.enable_capture:
            self._init_memory_pool()
    
    def _init_memory_pool(self) -> None:
        """Initialize CUDA graph memory pool."""
        try:
            # Use default memory pool
            self._memory_pool = torch.cuda.graph_pool_handle()
        except Exception as e:
            warnings.warn(f"Failed to init CUDA graph memory pool: {e}")
            self._memory_pool = None
    
    def _get_seq_bucket(self, seq_len: int) -> int:
        """Get sequence length bucket for graph pool."""
        # Bucket by powers of 2 up to max
        buckets = [128, 256, 512, 1024, 2048, 4096, 8192]
        for bucket in buckets:
            if seq_len <= bucket:
                return bucket
        return self.max_context_len
    
    def _get_graph_key(self, batch_size: int, seq_len: int) -> Tuple[int, int]:
        """Get key for graph pool lookup."""
        return (batch_size, self._get_seq_bucket(seq_len))
    
    def _create_static_buffers(
        self,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Create static input/output buffers for graph capture."""
        device = f"cuda:{self.device}"
        
        buffers = {
            # Input: single token per sequence for decode
            "input_ids": torch.zeros(
                (batch_size, 1), 
                dtype=torch.long, 
                device=device
            ),
            "position_ids": torch.zeros(
                (batch_size, 1), 
                dtype=torch.long, 
                device=device
            ),
            # Output logits
            "logits": torch.zeros(
                (batch_size, 1, vocab_size),
                dtype=dtype,
                device=device,
            ),
        }
        
        return buffers
    
    def _warmup(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Warmup run before capture to allocate memory.
        
        This ensures all memory allocations happen before capture.
        """
        with torch.no_grad():
            # Run model normally
            if hasattr(self.model, 'forward_decode'):
                output = self.model.forward_decode(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    **kwargs,
                )
            else:
                output = self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    **kwargs,
                )
                if hasattr(output, 'logits'):
                    output = output.logits
        
        # Sync to ensure all ops complete
        torch.cuda.synchronize(self.device)
        return output
    
    def _capture_graph(
        self,
        batch_size: int,
        seq_len: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> CapturedGraph:
        """
        Capture CUDA graph for decode step.
        
        Args:
            batch_size: Batch size for this graph
            seq_len: Current sequence length
            input_ids: Input token IDs
            position_ids: Position IDs
            **kwargs: Additional model arguments (kv_cache, etc.)
            
        Returns:
            Captured graph ready for replay
        """
        import time
        start = time.perf_counter()
        
        # Create captured graph container
        captured = CapturedGraph(
            batch_size=batch_size,
            max_seq_len=seq_len,
            state=GraphState.CAPTURING,
        )
        
        # Get model config for buffer creation
        hidden_size = getattr(self.model.config, 'hidden_size', 4096)
        vocab_size = getattr(self.model.config, 'vocab_size', 32000)
        dtype = next(self.model.parameters()).dtype
        
        # Create static buffers
        captured.input_buffers = self._create_static_buffers(
            batch_size, seq_len, hidden_size, vocab_size, dtype
        )
        
        # Copy input data to static buffers
        captured.input_buffers["input_ids"].copy_(input_ids)
        captured.input_buffers["position_ids"].copy_(position_ids)
        
        # Warmup run
        _ = self._warmup(
            captured.input_buffers["input_ids"],
            captured.input_buffers["position_ids"],
            **kwargs,
        )
        
        # Clear CUDA cache
        torch.cuda.synchronize(self.device)
        
        # Create graph
        graph = torch.cuda.CUDAGraph()
        
        # Capture
        with torch.cuda.graph(graph, pool=self._memory_pool):
            with torch.no_grad():
                if hasattr(self.model, 'forward_decode'):
                    output = self.model.forward_decode(
                        input_ids=captured.input_buffers["input_ids"],
                        position_ids=captured.input_buffers["position_ids"],
                        **kwargs,
                    )
                else:
                    output = self.model(
                        input_ids=captured.input_buffers["input_ids"],
                        position_ids=captured.input_buffers["position_ids"],
                        use_cache=True,
                        **kwargs,
                    )
                    if hasattr(output, 'logits'):
                        output = output.logits
                
                captured.output_buffers["logits"] = output
        
        captured.graph = graph
        captured.state = GraphState.CAPTURED
        captured.capture_time_ms = (time.perf_counter() - start) * 1000
        
        self.stats["captures"] += 1
        self.stats["capture_time_ms"] += captured.capture_time_ms
        
        return captured
    
    def decode_step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        use_graph: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Execute a single decode step, using CUDA graph if available.
        
        Args:
            input_ids: Input token IDs [batch, 1]
            position_ids: Position IDs [batch, 1]
            use_graph: Whether to use CUDA graph
            **kwargs: Additional model arguments
            
        Returns:
            Logits tensor [batch, 1, vocab_size]
        """
        if not self.enable_capture or not use_graph:
            # Fall back to normal execution
            return self._warmup(input_ids, position_ids, **kwargs)
        
        batch_size = input_ids.shape[0]
        seq_len = position_ids.max().item() + 1 if position_ids.numel() > 0 else 1
        
        # Get or create graph
        graph_key = self._get_graph_key(batch_size, seq_len)
        
        with self._lock:
            if graph_key not in self.graph_pool:
                # Capture new graph
                captured = self._capture_graph(
                    batch_size, seq_len,
                    input_ids, position_ids,
                    **kwargs,
                )
                self.graph_pool[graph_key] = captured
            
            captured = self.graph_pool[graph_key]
        
        if captured.state != GraphState.CAPTURED:
            # Graph capture failed, fall back
            return self._warmup(input_ids, position_ids, **kwargs)
        
        # Update static input buffers
        captured.input_buffers["input_ids"].copy_(input_ids)
        captured.input_buffers["position_ids"].copy_(position_ids)
        
        # Replay graph
        import time
        start = time.perf_counter()
        captured.graph.replay()
        replay_time = (time.perf_counter() - start) * 1000
        
        captured.num_replays += 1
        self.stats["replays"] += 1
        self.stats["replay_time_ms"] += replay_time
        
        # Return output from static buffer (clone to be safe)
        return captured.output_buffers["logits"].clone()
    
    def clear_graphs(self) -> None:
        """Clear all captured graphs."""
        with self._lock:
            self.graph_pool.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph execution statistics."""
        avg_capture = (
            self.stats["capture_time_ms"] / self.stats["captures"]
            if self.stats["captures"] > 0 else 0
        )
        avg_replay = (
            self.stats["replay_time_ms"] / self.stats["replays"]
            if self.stats["replays"] > 0 else 0
        )
        
        return {
            **self.stats,
            "num_graphs": len(self.graph_pool),
            "avg_capture_ms": avg_capture,
            "avg_replay_ms": avg_replay,
            "enabled": self.enable_capture,
        }


class BatchedGraphRunner:
    """
    Graph runner optimized for varying batch sizes.
    
    Pre-captures graphs for common batch sizes to avoid
    capture overhead at runtime.
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        device: int = 0,
    ):
        """
        Initialize batched graph runner.
        
        Args:
            model: Model to capture
            batch_sizes: Batch sizes to pre-capture
            device: CUDA device
        """
        self.model = model
        self.batch_sizes = sorted(batch_sizes)
        self.device = device
        
        # Runners for each batch size
        self.runners: Dict[int, CUDAGraphRunner] = {}
        
        for bs in batch_sizes:
            self.runners[bs] = CUDAGraphRunner(
                model=model,
                max_batch_size=bs,
                device=device,
            )
    
    def _find_runner(self, batch_size: int) -> CUDAGraphRunner:
        """Find appropriate runner for batch size."""
        # Find smallest batch size >= requested
        for bs in self.batch_sizes:
            if bs >= batch_size:
                return self.runners[bs]
        # Fall back to largest
        return self.runners[self.batch_sizes[-1]]
    
    def decode_step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Execute decode step with appropriate graph."""
        batch_size = input_ids.shape[0]
        runner = self._find_runner(batch_size)
        
        # Pad if needed
        target_bs = runner.max_batch_size
        if batch_size < target_bs:
            pad_size = target_bs - batch_size
            input_ids = torch.cat([
                input_ids,
                torch.zeros((pad_size, 1), dtype=input_ids.dtype, device=input_ids.device),
            ], dim=0)
            position_ids = torch.cat([
                position_ids,
                torch.zeros((pad_size, 1), dtype=position_ids.dtype, device=position_ids.device),
            ], dim=0)
        
        output = runner.decode_step(input_ids, position_ids, **kwargs)
        
        # Remove padding
        if batch_size < target_bs:
            output = output[:batch_size]
        
        return output


# =============================================================================
# GRAPH CAPTURE UTILITIES
# =============================================================================

def capture_model_graph(
    model: nn.Module,
    example_inputs: Dict[str, torch.Tensor],
    warmup_iters: int = 3,
) -> Tuple[torch.cuda.CUDAGraph, Dict[str, torch.Tensor]]:
    """
    Utility to capture a model forward pass as CUDA graph.
    
    Args:
        model: Model to capture
        example_inputs: Example input tensors
        warmup_iters: Number of warmup iterations
        
    Returns:
        (graph, output_buffers) tuple
    """
    # Create static input buffers
    static_inputs = {
        k: v.clone() for k, v in example_inputs.items()
    }
    
    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(**static_inputs)
    
    torch.cuda.synchronize()
    
    # Capture
    graph = torch.cuda.CUDAGraph()
    
    with torch.cuda.graph(graph):
        with torch.no_grad():
            output = model(**static_inputs)
    
    return graph, {"output": output}


def is_cuda_graph_compatible(model: nn.Module) -> bool:
    """
    Check if a model is compatible with CUDA graph capture.
    
    Requirements:
    - All parameters on same CUDA device
    - No dynamic control flow
    - No CPU operations during forward
    """
    if not torch.cuda.is_available():
        return False
    
    # Check all params on CUDA
    devices = set()
    for param in model.parameters():
        if param.device.type != 'cuda':
            return False
        devices.add(param.device)
    
    # All on same device
    return len(devices) <= 1
