"""ZSE GPU Memory Management — Direct GPU memory allocation via ctypes.

No PyTorch, no pycuda. Pure ctypes → CUDA driver API / HIP API.
"""

import ctypes
from typing import Optional
from zse_compiler.types.tensor import Tensor
from zse_compiler.types.dtypes import DType, float32


class GPUMemory:
    """Direct GPU memory allocator — zero dependency.

    Args:
        backend: "cuda", "rocm", or "metal"
        device_index: GPU device index (default 0). For multi-GPU tensor
                      parallelism, each rank creates GPUMemory with its own index.
    """

    def __init__(self, backend: str = "cuda", device_index: int = 0):
        self.backend = backend
        self.device_index = device_index
        self._driver = None
        self._allocations = {}  # ptr → size
        self._ctx = None  # GPU context handle (for multi-device)

        if backend == "cuda":
            self._driver = self._load_cuda_driver()
            if self._driver is None:
                raise RuntimeError("CUDA driver not found")
            self._driver.cuInit(0)
            # Select specific device and create context
            device = ctypes.c_int(0)
            self._driver.cuDeviceGet(ctypes.byref(device), device_index)
            # Check if context already exists for this device
            ctx = ctypes.c_void_p()
            self._driver.cuCtxGetCurrent(ctypes.byref(ctx))
            if ctx.value is None or ctx.value == 0:
                ctx = ctypes.c_void_p()
                self._driver.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
                self._ctx = ctx
            elif device_index > 0:
                # Need a new context for a different device
                ctx = ctypes.c_void_p()
                self._driver.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
                self._ctx = ctx
        elif backend == "rocm":
            self._driver = self._load_hip_runtime()
            if self._driver is None:
                raise RuntimeError("HIP runtime not found")
            # Select specific device
            self._driver.hipSetDevice(device_index)
        elif backend == "metal":
            from zse_compiler.runtime.metal_dispatch import get_metal_runtime
            self._metal_rt = get_metal_runtime()
            # Map buffer_handle → contents_ptr for unified memory access
            self._metal_buffers = {}  # ptr_as_int → buffer_handle

    def ensure_context(self):
        """Ensure this device's CUDA/HIP context is current.

        Call before GPU operations when multiple GPUMemory instances
        exist (tensor parallelism). No-op for single-GPU.
        """
        if self._ctx and self.backend == "cuda":
            self._driver.cuCtxSetCurrent(self._ctx)
        elif self.backend == "rocm" and self.device_index > 0:
            self._driver.hipSetDevice(self.device_index)

    def allocate(self, shape: tuple, dtype: DType = float32) -> Tensor:
        """Allocate GPU memory and return a ZSE Tensor."""
        t = Tensor(shape=shape, dtype=dtype)
        nbytes = t.nbytes

        if self.backend == "cuda":
            ptr = self._cuda_malloc(nbytes)
        elif self.backend == "rocm":
            ptr = self._hip_malloc(nbytes)
        elif self.backend == "metal":
            ptr = self._metal_alloc(nbytes)
        else:
            raise ValueError(f"Memory allocation not supported for backend: {self.backend}")

        t._data_ptr = ptr
        t._device = f"{self.backend}:0"
        t._nbytes = nbytes
        self._allocations[ptr] = nbytes
        return t

    def free(self, tensor: Tensor):
        """Free GPU memory."""
        if tensor.data_ptr == 0:
            return

        if self.backend == "cuda":
            self._cuda_free(tensor.data_ptr)
        elif self.backend == "rocm":
            self._hip_free(tensor.data_ptr)
        elif self.backend == "metal":
            self._metal_free(tensor.data_ptr)

        self._allocations.pop(tensor.data_ptr, None)
        tensor._data_ptr = 0

    def copy_host_to_device(self, host_data: bytes, tensor: Tensor):
        """Copy data from host (CPU) to device (GPU)."""
        nbytes = len(host_data)
        src = ctypes.c_char_p(host_data)

        if self.backend == "cuda":
            self._driver.cuMemcpyHtoD_v2(
                ctypes.c_void_p(tensor.data_ptr),
                src,
                ctypes.c_size_t(nbytes)
            )
        elif self.backend == "rocm":
            self._driver.hipMemcpyHtoD(
                ctypes.c_void_p(tensor.data_ptr),
                src,
                ctypes.c_size_t(nbytes)
            )
        elif self.backend == "metal":
            # Unified memory — just memcpy to buffer contents
            contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[tensor.data_ptr])
            )
            ctypes.memmove(contents, host_data, nbytes)

    def copy_device_to_host(self, tensor: Tensor) -> bytes:
        """Copy data from device (GPU) to host (CPU)."""
        nbytes = tensor.nbytes
        buf = ctypes.create_string_buffer(nbytes)

        if self.backend == "cuda":
            self._driver.cuMemcpyDtoH_v2(
                buf,
                ctypes.c_void_p(tensor.data_ptr),
                ctypes.c_size_t(nbytes)
            )
        elif self.backend == "rocm":
            self._driver.hipMemcpyDtoH(
                buf,
                ctypes.c_void_p(tensor.data_ptr),
                ctypes.c_size_t(nbytes)
            )
        elif self.backend == "metal":
            # Unified memory — just memcpy from buffer contents
            contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[tensor.data_ptr])
            )
            ctypes.memmove(buf, contents, nbytes)

        return buf.raw

    def memset(self, tensor: Tensor, value: int = 0):
        """Zero out GPU memory."""
        if self.backend == "cuda":
            self._driver.cuMemsetD8_v2(
                ctypes.c_void_p(tensor.data_ptr),
                ctypes.c_ubyte(value),
                ctypes.c_size_t(tensor.nbytes)
            )
        elif self.backend == "rocm":
            self._driver.hipMemset(
                ctypes.c_void_p(tensor.data_ptr),
                ctypes.c_int(value),
                ctypes.c_size_t(tensor.nbytes)
            )
        elif self.backend == "metal":
            contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[tensor.data_ptr])
            )
            ctypes.memset(contents, value, tensor.nbytes)

    def malloc_raw(self, nbytes: int) -> int:
        """Allocate raw GPU memory, return pointer. No Tensor wrapper."""
        if nbytes == 0:
            return 0
        if self.backend == "cuda":
            ptr = self._cuda_malloc(nbytes)
        elif self.backend == "rocm":
            ptr = self._hip_malloc(nbytes)
        elif self.backend == "metal":
            ptr = self._metal_alloc(nbytes)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        self._allocations[ptr] = nbytes
        return ptr

    def free_raw(self, ptr: int):
        """Free raw GPU pointer."""
        if ptr == 0:
            return
        if self.backend == "cuda":
            self._cuda_free(ptr)
        elif self.backend == "rocm":
            self._hip_free(ptr)
        elif self.backend == "metal":
            self._metal_free(ptr)
        self._allocations.pop(ptr, None)

    def copy_host_to_device_raw(self, host_data: bytes, dst_ptr: int, nbytes: int):
        """Copy bytes to a raw GPU pointer."""
        src = ctypes.c_char_p(host_data)
        if self.backend == "cuda":
            self._driver.cuMemcpyHtoD_v2(
                ctypes.c_void_p(dst_ptr), src, ctypes.c_size_t(nbytes)
            )
        elif self.backend == "rocm":
            self._driver.hipMemcpyHtoD(
                ctypes.c_void_p(dst_ptr), src, ctypes.c_size_t(nbytes)
            )
        elif self.backend == "metal":
            contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[dst_ptr])
            )
            ctypes.memmove(contents, host_data, nbytes)

    def copy_host_to_device_raw_ptr(self, src_ptr: int, dst_ptr: int, nbytes: int):
        """Copy from a raw host pointer to a raw GPU pointer (synchronous).

        Use when the source is already a contiguous host buffer (e.g., pinned
        memory, mmap region, or ctypes array). Avoids the implicit copy that
        happens when passing a Python `bytes` object.
        """
        if nbytes == 0:
            return
        if self.backend == "cuda":
            self._driver.cuMemcpyHtoD_v2(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
            )
        elif self.backend == "rocm":
            self._driver.hipMemcpyHtoD(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
            )
        elif self.backend == "metal":
            contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[dst_ptr])
            )
            ctypes.memmove(contents, ctypes.c_void_p(src_ptr), nbytes)

    def copy_host_to_device_async_raw_ptr(
        self, src_ptr: int, dst_ptr: int, nbytes: int, stream: int
    ):
        """Async HtoD copy from raw host pointer (must be pinned for true async).

        Returns immediately; the copy runs on the stream. Caller must
        synchronize the stream before reusing the source buffer.
        """
        if nbytes == 0:
            return
        if self.backend == "cuda":
            self._driver.cuMemcpyHtoDAsync_v2(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream) if stream else None,
            )
        elif self.backend == "rocm":
            self._driver.hipMemcpyHtoDAsync(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_void_p(stream) if stream else None,
            )
        elif self.backend == "metal":
            # Metal unified memory — no separate HtoD; just memcpy.
            contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[dst_ptr])
            )
            ctypes.memmove(contents, ctypes.c_void_p(src_ptr), nbytes)

    # --- Pinned host memory (for fast async HtoD) ---

    def pinned_alloc(self, nbytes: int) -> int:
        """Allocate page-locked (pinned) host memory. Returns raw pointer (int).

        Pinned memory enables true async HtoD copies and ~2x higher PCIe
        throughput vs pageable memory. Falls back to malloc on Metal (unified).
        """
        if nbytes == 0:
            return 0
        if self.backend == "cuda":
            ptr = ctypes.c_void_p(0)
            # cuMemHostAlloc flags: 0 = portable=false, mapped=false
            rc = self._driver.cuMemHostAlloc(
                ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(0)
            )
            if rc != 0:
                raise RuntimeError(f"cuMemHostAlloc failed: rc={rc}, nbytes={nbytes}")
            return ptr.value
        elif self.backend == "rocm":
            ptr = ctypes.c_void_p(0)
            rc = self._driver.hipHostMalloc(
                ctypes.byref(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(0)
            )
            if rc != 0:
                raise RuntimeError(f"hipHostMalloc failed: rc={rc}, nbytes={nbytes}")
            return ptr.value
        elif self.backend == "metal":
            # No pinned concept; just plain malloc'd ctypes buffer.
            buf = (ctypes.c_char * nbytes)()
            return ctypes.addressof(buf)
        raise ValueError(f"pinned_alloc unsupported for backend: {self.backend}")

    def pinned_free(self, ptr: int):
        """Free pinned host memory."""
        if ptr == 0:
            return
        if self.backend == "cuda":
            self._driver.cuMemFreeHost(ctypes.c_void_p(ptr))
        elif self.backend == "rocm":
            self._driver.hipHostFree(ctypes.c_void_p(ptr))
        # metal: ctypes-owned, GC handles it

    # --- Streams ---

    def create_stream(self) -> int:
        """Create a stream for async ops. Returns stream handle as int."""
        if self.backend == "cuda":
            stream = ctypes.c_void_p(0)
            rc = self._driver.cuStreamCreate(ctypes.byref(stream), ctypes.c_uint(0))
            if rc != 0:
                raise RuntimeError(f"cuStreamCreate failed: rc={rc}")
            return stream.value or 0
        elif self.backend == "rocm":
            stream = ctypes.c_void_p(0)
            rc = self._driver.hipStreamCreate(ctypes.byref(stream))
            if rc != 0:
                raise RuntimeError(f"hipStreamCreate failed: rc={rc}")
            return stream.value or 0
        return 0  # Metal: no streams

    def destroy_stream(self, stream: int):
        """Destroy a stream."""
        if stream == 0:
            return
        if self.backend == "cuda":
            self._driver.cuStreamDestroy_v2(ctypes.c_void_p(stream))
        elif self.backend == "rocm":
            self._driver.hipStreamDestroy(ctypes.c_void_p(stream))

    def synchronize_stream(self, stream: int):
        """Block until all operations on stream complete."""
        if self.backend == "cuda":
            self._driver.cuStreamSynchronize(ctypes.c_void_p(stream) if stream else None)
        elif self.backend == "rocm":
            self._driver.hipStreamSynchronize(ctypes.c_void_p(stream) if stream else None)
        # metal: ops are synchronous

    def synchronize(self):
        """Block until all GPU operations on the current context complete."""
        if self.backend == "cuda":
            self._driver.cuCtxSynchronize()
        elif self.backend == "rocm":
            self._driver.hipDeviceSynchronize()

    def copy_device_to_device(self, src_ptr: int, dst_ptr: int, nbytes: int):
        """Copy data between two GPU memory locations."""
        if self.backend == "cuda":
            self._driver.cuMemcpy(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
            )
        elif self.backend == "rocm":
            self._driver.hipMemcpyDtoD(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
            )
        elif self.backend == "metal":
            # Unified memory — memcpy between buffer contents
            src_contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[src_ptr])
            )
            dst_contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[dst_ptr])
            )
            ctypes.memmove(dst_contents, src_contents, nbytes)

    def copy_device_to_device_async(self, src_ptr: int, dst_ptr: int, nbytes: int, stream):
        """Copy data between GPU memory locations on a stream (for graph capture)."""
        if self.backend == "cuda":
            self._driver.cuMemcpyDtoDAsync_v2(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
                stream,
            )
        elif self.backend == "rocm":
            self._driver.hipMemcpyDtoDAsync(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
                stream,
            )
        elif self.backend == "metal":
            # Metal unified memory — sync memcpy (no stream concept)
            src_contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[src_ptr])
            )
            dst_contents = self._metal_rt.buffer_contents(
                ctypes.c_void_p(self._metal_buffers[dst_ptr])
            )
            ctypes.memmove(dst_contents, src_contents, nbytes)

    def get_free_memory(self) -> int:
        """Get available GPU memory in bytes."""
        if self.backend == "cuda":
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            self._driver.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
            return free.value
        elif self.backend == "rocm":
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            self._driver.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            return free.value
        elif self.backend == "metal":
            total = self._metal_rt.device_memory_bytes
            used = sum(self._allocations.values())
            return total - used
        return 0

    def get_total_memory(self) -> int:
        """Get total GPU memory in bytes."""
        if self.backend == "cuda":
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            self._driver.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
            return total.value
        elif self.backend == "rocm":
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            self._driver.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            return total.value
        elif self.backend == "metal":
            return self._metal_rt.device_memory_bytes
        return 0

    # --- Internal allocation ---

    def _cuda_malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p(0)
        status = self._driver.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if status != 0:
            raise RuntimeError(f"cuMemAlloc failed with status {status} for {nbytes} bytes")
        return ptr.value

    def _cuda_free(self, ptr: int):
        self._driver.cuMemFree_v2(ctypes.c_void_p(ptr))

    def _hip_malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p(0)
        status = self._driver.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes))
        if status != 0:
            raise RuntimeError(f"hipMalloc failed with status {status}")
        return ptr.value

    def _hip_free(self, ptr: int):
        self._driver.hipFree(ctypes.c_void_p(ptr))

    @staticmethod
    def _load_cuda_driver():
        names = ["libcuda.so.1", "libcuda.so", "nvcuda.dll", "cuda"]
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        path = ctypes.util.find_library("cuda")
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass
        return None

    @staticmethod
    def _load_hip_runtime():
        names = ["libamdhip64.so", "amdhip64.dll"]
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        return None

    def _metal_alloc(self, nbytes: int) -> int:
        """Allocate a Metal buffer. Returns buffer handle as int.

        The Tensor.data_ptr stores the buffer handle (not contents ptr).
        For CPU access, use buffer_contents() on the handle.
        """
        buf = self._metal_rt.alloc_buffer(nbytes)
        if not buf:
            raise MemoryError(f"Metal buffer allocation failed for {nbytes} bytes")
        # Store as int for consistency with CUDA/ROCm ptr pattern
        handle = ctypes.cast(buf, ctypes.c_void_p).value
        self._metal_buffers[handle] = handle
        return handle

    def _metal_free(self, ptr: int):
        """Free a Metal buffer by handle."""
        if ptr in self._metal_buffers:
            self._metal_rt.free_buffer(ctypes.c_void_p(ptr))
            del self._metal_buffers[ptr]

    def __del__(self):
        """Free all remaining allocations."""
        for ptr in list(self._allocations.keys()):
            try:
                if self.backend == "cuda":
                    self._cuda_free(ptr)
                elif self.backend == "rocm":
                    self._hip_free(ptr)
                elif self.backend == "metal":
                    self._metal_free(ptr)
            except Exception:
                pass
        self._allocations.clear()
