"""ZSE CUDA Graph Capture — Record & replay the forward pass.

The core throughput bottleneck: 480 Python→ctypes kernel launches per decode step.
Even async, each launch costs ~0.5ms Python overhead → 240ms/step → 4 tok/s.

Solution: CUDA Graphs.
1. Record the entire decode forward pass (all 480 launches) into a CUDA graph
2. Replay the graph per step with cuGraphLaunch — ONE Python→C call
3. Update only the changing inputs (token, position, block_table) via graph node update

Expected: ~0.5ms host overhead per step → 50-100+ tok/s (GPU-bound, not CPU-bound)

CUDA Graph API (driver level):
- cuStreamBeginCapture / cuStreamEndCapture → cuGraph
- cuGraphInstantiate → cuGraphExec
- cuGraphLaunch(exec, stream) — replays entire recorded sequence
- cuGraphExecKernelNodeSetParams — update kernel args without re-recording
"""

import ctypes
from typing import Optional


class CUDAGraphRunner:
    """Records and replays the decode forward pass as a CUDA graph.

    Usage:
        runner = CUDAGraphRunner(driver)

        # Record phase (first decode step)
        runner.begin_capture(stream)
        ... launch all kernels normally ...
        runner.end_capture()

        # Replay phase (all subsequent steps)
        runner.replay(stream)  # One call, replays all 480 launches
    """

    def __init__(self, driver):
        self._driver = driver
        self._graph = None         # CUgraph
        self._graph_exec = None    # CUgraphExec
        self._stream = None        # Capture stream
        self._captured = False

    def begin_capture(self, stream=None):
        """Start recording kernel launches into a CUDA graph."""
        if stream is None:
            # Create a dedicated stream for graph capture
            self._stream = ctypes.c_void_p()
            status = self._driver.cuStreamCreate(
                ctypes.byref(self._stream), 0  # CU_STREAM_DEFAULT
            )
            if status != 0:
                raise RuntimeError(f"cuStreamCreate failed: {status}")
        else:
            self._stream = stream

        # Begin capture — all subsequent launches on this stream are recorded
        status = self._driver.cuStreamBeginCapture(
            self._stream,
            0,  # CU_STREAM_CAPTURE_MODE_GLOBAL
        )
        if status != 0:
            raise RuntimeError(f"cuStreamBeginCapture failed: {status}")

    def end_capture(self):
        """Finish recording and instantiate the executable graph."""
        graph = ctypes.c_void_p()
        status = self._driver.cuStreamEndCapture(
            self._stream,
            ctypes.byref(graph),
        )
        if status != 0:
            raise RuntimeError(f"cuStreamEndCapture failed: {status}")

        self._graph = graph

        # Instantiate — compile the graph into an executable
        graph_exec = ctypes.c_void_p()
        status = self._driver.cuGraphInstantiate(
            ctypes.byref(graph_exec),
            self._graph,
            ctypes.c_void_p(0),  # error node (NULL)
            ctypes.c_char_p(0),  # log buffer (NULL)
            0,                   # buffer size
        )
        if status != 0:
            raise RuntimeError(f"cuGraphInstantiate failed: {status}")

        self._graph_exec = graph_exec
        self._captured = True

    def replay(self, stream=None):
        """Replay the captured graph — one call for the entire forward pass."""
        if not self._captured:
            raise RuntimeError("No graph captured yet")

        s = stream or self._stream
        status = self._driver.cuGraphLaunch(self._graph_exec, s)
        if status != 0:
            raise RuntimeError(f"cuGraphLaunch failed: {status}")

    @property
    def is_captured(self) -> bool:
        return self._captured

    def create_stream(self):
        """Create a dedicated stream for graph operations."""
        stream = ctypes.c_void_p()
        status = self._driver.cuStreamCreate(ctypes.byref(stream), 0)
        if status != 0:
            raise RuntimeError(f"cuStreamCreate failed: {status}")
        return stream

    def sync(self, stream=None):
        """Synchronize the graph stream — wait for all GPU work to complete."""
        s = stream or self._stream
        if s is not None:
            status = self._driver.cuStreamSynchronize(s)
            if status != 0:
                raise RuntimeError(f"cuStreamSynchronize failed: {status}")
        else:
            # Fallback: sync entire device
            status = self._driver.cuCtxSynchronize()
            if status != 0:
                raise RuntimeError(f"cuCtxSynchronize failed: {status}")

    def destroy(self):
        """Release graph resources."""
        if self._graph_exec is not None:
            self._driver.cuGraphExecDestroy(self._graph_exec)
            self._graph_exec = None
        if self._graph is not None:
            self._driver.cuGraphDestroy(self._graph)
            self._graph = None
