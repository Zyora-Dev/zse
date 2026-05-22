"""ZSE Metal Dispatch — Python wrapper for the Metal C bridge.

Auto-compiles metal_dispatch.m → .dylib on first use.
Only needs Command Line Tools (clang), NOT full Xcode.

Zero Python dependencies — pure ctypes.
"""

import ctypes
import os
import platform
import subprocess
from typing import Optional, List, Tuple


def _get_cache_dir() -> str:
    """Get or create the ZSE cache directory."""
    cache_dir = os.path.expanduser("~/.zse/cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_dylib_path() -> str:
    return os.path.join(_get_cache_dir(), "metal_dispatch.dylib")


def _get_source_path() -> str:
    return os.path.join(os.path.dirname(__file__), "metal_dispatch.m")


def _compile_bridge() -> str:
    """Compile metal_dispatch.m → .dylib using clang (Command Line Tools only)."""
    src = _get_source_path()
    dylib = _get_dylib_path()

    if not os.path.exists(src):
        raise FileNotFoundError(f"Metal bridge source not found: {src}")

    # Recompile if source is newer than dylib
    if os.path.exists(dylib):
        if os.path.getmtime(dylib) >= os.path.getmtime(src):
            return dylib

    try:
        result = subprocess.run(
            ["clang", "-O2", "-shared", "-o", dylib, src,
             "-framework", "Metal", "-framework", "Foundation",
             "-fobjc-arc"],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "clang not found. Install Command Line Tools: xcode-select --install"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to compile Metal bridge:\n{e.stderr}")

    return dylib


def _load_bridge() -> ctypes.CDLL:
    """Load the Metal C bridge dylib."""
    dylib = _compile_bridge()
    lib = ctypes.cdll.LoadLibrary(dylib)

    # --- Device Init ---
    lib.zse_metal_init.restype = ctypes.c_void_p
    lib.zse_metal_init.argtypes = []

    lib.zse_metal_create_queue.restype = ctypes.c_void_p
    lib.zse_metal_create_queue.argtypes = [ctypes.c_void_p]

    # --- Device Info ---
    lib.zse_metal_device_memory.restype = ctypes.c_uint64
    lib.zse_metal_device_memory.argtypes = [ctypes.c_void_p]

    lib.zse_metal_device_name.restype = ctypes.c_char_p
    lib.zse_metal_device_name.argtypes = [ctypes.c_void_p]

    # --- Compile ---
    lib.zse_metal_compile.restype = ctypes.c_void_p
    lib.zse_metal_compile.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]

    lib.zse_metal_compile_error.restype = ctypes.c_char_p
    lib.zse_metal_compile_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    # --- Buffer ---
    lib.zse_metal_alloc_buffer.restype = ctypes.c_void_p
    lib.zse_metal_alloc_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

    lib.zse_metal_buffer_contents.restype = ctypes.c_void_p
    lib.zse_metal_buffer_contents.argtypes = [ctypes.c_void_p]

    lib.zse_metal_buffer_length.restype = ctypes.c_uint64
    lib.zse_metal_buffer_length.argtypes = [ctypes.c_void_p]

    lib.zse_metal_free_buffer.restype = None
    lib.zse_metal_free_buffer.argtypes = [ctypes.c_void_p]

    # --- Dispatch ---
    lib.zse_metal_dispatch.restype = ctypes.c_double
    lib.zse_metal_dispatch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    lib.zse_metal_dispatch_async.restype = None
    lib.zse_metal_dispatch_async.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    lib.zse_metal_sync.restype = ctypes.c_double
    lib.zse_metal_sync.argtypes = []

    return lib


class MetalRuntime:
    """Metal GPU runtime — compile MSL, allocate buffers, dispatch kernels.

    Auto-compiles the Objective-C bridge on first use (needs clang).
    After that, everything goes through the cached .dylib.

    Usage:
        rt = MetalRuntime()
        print(rt.device_name, rt.device_memory_bytes)
        pipeline = rt.compile_msl(source, "my_kernel")
        buf = rt.alloc_buffer(1024)
        rt.dispatch(pipeline, [buf], grid=(4,1,1), block=(256,1,1))
    """

    def __init__(self):
        if platform.system() != "Darwin":
            raise RuntimeError("Metal backend requires macOS")

        self._lib = _load_bridge()
        self._device = self._lib.zse_metal_init()
        if not self._device:
            raise RuntimeError("No Metal device found")
        self._queue = self._lib.zse_metal_create_queue(self._device)
        if not self._queue:
            raise RuntimeError("Failed to create Metal command queue")

        # Pipeline cache: (source_hash, name) → pipeline_ptr
        self._pipeline_cache = {}

    @property
    def device_name(self) -> str:
        name = self._lib.zse_metal_device_name(self._device)
        return name.decode() if name else "Apple GPU"

    @property
    def device_memory_bytes(self) -> int:
        return self._lib.zse_metal_device_memory(self._device)

    @property
    def device_memory_mb(self) -> int:
        return self.device_memory_bytes // (1024 * 1024)

    def compile_msl(self, source: str, kernel_name: str) -> ctypes.c_void_p:
        """Compile MSL source into a compute pipeline. Cached by content hash."""
        cache_key = (hash(source), kernel_name)
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        pipeline = self._lib.zse_metal_compile(
            self._device,
            source.encode("utf-8"),
            kernel_name.encode("utf-8"),
        )
        if not pipeline:
            # Get error message
            err = self._lib.zse_metal_compile_error(
                self._device, source.encode("utf-8")
            )
            err_msg = err.decode() if err else "Unknown error"
            raise RuntimeError(f"Metal MSL compilation failed for '{kernel_name}':\n{err_msg}")

        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def alloc_buffer(self, nbytes: int) -> ctypes.c_void_p:
        """Allocate a shared-memory Metal buffer."""
        buf = self._lib.zse_metal_alloc_buffer(self._device, nbytes)
        if not buf:
            raise MemoryError(f"Failed to allocate Metal buffer of {nbytes} bytes")
        return buf

    def buffer_contents(self, buffer: ctypes.c_void_p) -> int:
        """Get CPU-accessible pointer to buffer contents (unified memory)."""
        return self._lib.zse_metal_buffer_contents(buffer)

    def buffer_length(self, buffer: ctypes.c_void_p) -> int:
        """Get buffer size in bytes."""
        return self._lib.zse_metal_buffer_length(buffer)

    def free_buffer(self, buffer: ctypes.c_void_p):
        """Release a Metal buffer."""
        if buffer:
            self._lib.zse_metal_free_buffer(buffer)

    def dispatch(
        self,
        pipeline: ctypes.c_void_p,
        buffers: List[ctypes.c_void_p],
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
    ) -> float:
        """Dispatch a compute kernel. Returns GPU time in milliseconds."""
        # Pad to 3D
        while len(grid) < 3:
            grid = grid + (1,)
        while len(block) < 3:
            block = block + (1,)

        # Build buffer pointer array
        n = len(buffers)
        buf_arr = (ctypes.c_void_p * n)(*buffers)

        return self._lib.zse_metal_dispatch(
            self._queue, pipeline,
            buf_arr, n,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
        )

    def dispatch_async(
        self,
        pipeline: ctypes.c_void_p,
        buffers: List[ctypes.c_void_p],
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
    ):
        """Dispatch a compute kernel asynchronously (non-blocking)."""
        while len(grid) < 3:
            grid = grid + (1,)
        while len(block) < 3:
            block = block + (1,)

        n = len(buffers)
        buf_arr = (ctypes.c_void_p * n)(*buffers)

        self._lib.zse_metal_dispatch_async(
            self._queue, pipeline,
            buf_arr, n,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
        )

    def sync(self) -> float:
        """Wait for pending async work. Returns GPU time in ms."""
        return self._lib.zse_metal_sync()


# Global singleton
_runtime: Optional[MetalRuntime] = None


def get_metal_runtime() -> MetalRuntime:
    """Get or create the global Metal runtime."""
    global _runtime
    if _runtime is None:
        _runtime = MetalRuntime()
    return _runtime
