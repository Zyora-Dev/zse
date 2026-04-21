"""ZSE HIP Graph Runner — Record & replay the decode forward pass on AMD.

The decode path launches ~960 kernels per token via Python→ctypes FFI calls.
Each hipModuleLaunchKernel has ~7μs Python overhead → ~7ms wasted per token.
At 29 tok/s (34.5ms/token), this is 20% overhead just in launch dispatch.

Solution: HIP Graphs (AMD equivalent of CUDA Graphs).
1. First decode step runs normally (warmup)
2. Second decode step: captured into a HIP graph
3. All subsequent steps: replay the graph with ONE hipGraphLaunch call

The only values that change per step:
- token_id (1 int, written to pre-allocated GPU buffer)
- position (1 int, written to pre-allocated GPU buffer)
- seq_lens (increments by 1 — written to GPU buffer)
- block_table (rarely changes — only when new KV block allocated)

These are handled by writing to the GPU buffers BEFORE graph replay.
The graph references the buffer ADDRESSES (which don't change), so the
kernels automatically see the updated values. No graph node update needed!

HIP API (ctypes):
- hipStreamCreate → stream
- hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal)
- ... launch all kernels on this stream ...
- hipStreamEndCapture(stream) → hipGraph_t
- hipGraphInstantiate(graph) → hipGraphExec_t
- hipGraphLaunch(exec, stream) — replays everything
"""

import ctypes
from typing import Optional


class HIPGraphRunner:
    """Records and replays the decode forward pass as a HIP graph.

    Usage:
        runner = HIPGraphRunner(hip_driver)

        # Warmup (first step — no capture)
        ... run decode normally ...

        # Capture (second step)
        runner.begin_capture()
        ... run decode normally (all launches go into graph) ...
        runner.end_capture()

        # Replay (all subsequent steps)
        # Write updated values to GPU buffers first, then:
        runner.replay()
    """

    def __init__(self, driver):
        self._driver = driver
        self._graph = None         # hipGraph_t
        self._graph_exec = None    # hipGraphExec_t
        self._stream = None        # Capture/replay stream
        self._captured = False

    def create_stream(self):
        """Create a dedicated stream for graph capture and replay."""
        self._stream = ctypes.c_void_p()
        status = self._driver.hipStreamCreate(
            ctypes.byref(self._stream), 0  # hipStreamDefault
        )
        if status != 0:
            raise RuntimeError(f"hipStreamCreate failed: {status}")
        return self._stream

    @property
    def stream(self):
        """The stream to use for kernel launches during capture."""
        return self._stream

    def begin_capture(self):
        """Start recording kernel launches into a HIP graph.

        All subsequent hipModuleLaunchKernel calls on this stream
        will be captured into the graph (not executed).
        """
        if self._stream is None:
            self.create_stream()

        # hipStreamCaptureModeGlobal = 0
        status = self._driver.hipStreamBeginCapture(
            self._stream, 0
        )
        if status != 0:
            raise RuntimeError(f"hipStreamBeginCapture failed: {status}")

    def end_capture(self):
        """Finish capture and instantiate the executable graph."""
        graph = ctypes.c_void_p()
        status = self._driver.hipStreamEndCapture(
            self._stream, ctypes.byref(graph)
        )
        if status != 0:
            raise RuntimeError(f"hipStreamEndCapture failed: {status}")
        self._graph = graph

        # Instantiate — optimize the graph for repeated execution
        graph_exec = ctypes.c_void_p()
        status = self._driver.hipGraphInstantiate(
            ctypes.byref(graph_exec),
            self._graph,
            ctypes.c_void_p(0),  # error node (NULL)
            ctypes.c_char_p(0),  # log buffer (NULL)
            0,                   # buffer size
        )
        if status != 0:
            raise RuntimeError(f"hipGraphInstantiate failed: {status}")

        self._graph_exec = graph_exec
        self._captured = True

    def replay(self):
        """Replay the captured graph — one FFI call for ~960 kernel launches."""
        if not self._captured:
            raise RuntimeError("No graph captured yet. Call begin/end_capture first.")

        status = self._driver.hipGraphLaunch(self._graph_exec, self._stream)
        if status != 0:
            raise RuntimeError(f"hipGraphLaunch failed: {status}")

    def sync(self):
        """Wait for graph execution to complete."""
        status = self._driver.hipStreamSynchronize(self._stream)
        if status != 0:
            raise RuntimeError(f"hipStreamSynchronize failed: {status}")

    @property
    def is_captured(self) -> bool:
        return self._captured

    def destroy(self):
        """Release graph resources."""
        if self._graph_exec is not None:
            self._driver.hipGraphExecDestroy(self._graph_exec)
            self._graph_exec = None
        if self._graph is not None:
            self._driver.hipGraphDestroy(self._graph)
            self._graph = None
        if self._stream is not None:
            self._driver.hipStreamDestroy(self._stream)
            self._stream = None
        self._captured = False


class CUDAGraphRunner:
    """Records and replays the decode forward pass as a CUDA graph.

    Same API as HIPGraphRunner but uses CUDA driver API.
    """

    def __init__(self, driver):
        self._driver = driver
        self._graph = None
        self._graph_exec = None
        self._stream = None
        self._captured = False

    def create_stream(self):
        """Create a dedicated stream for graph capture and replay."""
        self._stream = ctypes.c_void_p()
        status = self._driver.cuStreamCreate(
            ctypes.byref(self._stream), 0
        )
        if status != 0:
            raise RuntimeError(f"cuStreamCreate failed: {status}")
        return self._stream

    @property
    def stream(self):
        return self._stream

    def begin_capture(self):
        if self._stream is None:
            self.create_stream()
        # CU_STREAM_CAPTURE_MODE_GLOBAL = 0
        status = self._driver.cuStreamBeginCapture(self._stream, 0)
        if status != 0:
            raise RuntimeError(f"cuStreamBeginCapture failed: {status}")

    def end_capture(self):
        graph = ctypes.c_void_p()
        status = self._driver.cuStreamEndCapture(
            self._stream, ctypes.byref(graph)
        )
        if status != 0:
            raise RuntimeError(f"cuStreamEndCapture failed: {status}")
        self._graph = graph

        graph_exec = ctypes.c_void_p()
        status = self._driver.cuGraphInstantiate(
            ctypes.byref(graph_exec), self._graph,
            ctypes.c_void_p(0), ctypes.c_char_p(0), 0,
        )
        if status != 0:
            raise RuntimeError(f"cuGraphInstantiate failed: {status}")
        self._graph_exec = graph_exec
        self._captured = True

    def replay(self):
        if not self._captured:
            raise RuntimeError("No graph captured")
        status = self._driver.cuGraphLaunch(self._graph_exec, self._stream)
        if status != 0:
            raise RuntimeError(f"cuGraphLaunch failed: {status}")

    def sync(self):
        self._driver.cuStreamSynchronize(self._stream)

    @property
    def is_captured(self) -> bool:
        return self._captured

    def destroy(self):
        if self._graph_exec is not None:
            self._driver.cuGraphExecDestroy(self._graph_exec)
            self._graph_exec = None
        if self._graph is not None:
            self._driver.cuGraphDestroy(self._graph)
            self._graph = None
        if self._stream is not None:
            self._driver.cuStreamDestroy_v2(self._stream)
            self._stream = None
        self._captured = False
