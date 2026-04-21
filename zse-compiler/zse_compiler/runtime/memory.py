"""ZSE GPU Memory Management — Direct GPU memory allocation via ctypes.

No PyTorch, no pycuda. Pure ctypes → CUDA driver API / HIP API.
"""

import ctypes
from typing import Optional
from zse_compiler.types.tensor import Tensor
from zse_compiler.types.dtypes import DType, float32


class GPUMemory:
    """Direct GPU memory allocator — zero dependency."""

    def __init__(self, backend: str = "cuda"):
        self.backend = backend
        self._driver = None
        self._allocations = {}  # ptr → size

        if backend == "cuda":
            self._driver = self._load_cuda_driver()
            if self._driver is None:
                raise RuntimeError("CUDA driver not found")
            self._driver.cuInit(0)
        elif backend == "rocm":
            self._driver = self._load_hip_runtime()
            if self._driver is None:
                raise RuntimeError("HIP runtime not found")

    def allocate(self, shape: tuple, dtype: DType = float32) -> Tensor:
        """Allocate GPU memory and return a ZSE Tensor."""
        t = Tensor(shape=shape, dtype=dtype)
        nbytes = t.nbytes

        if self.backend == "cuda":
            ptr = self._cuda_malloc(nbytes)
        elif self.backend == "rocm":
            ptr = self._hip_malloc(nbytes)
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

    def malloc_raw(self, nbytes: int) -> int:
        """Allocate raw GPU memory, return pointer. No Tensor wrapper."""
        if nbytes == 0:
            return 0
        if self.backend == "cuda":
            ptr = self._cuda_malloc(nbytes)
        elif self.backend == "rocm":
            ptr = self._hip_malloc(nbytes)
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

    def __del__(self):
        """Free all remaining allocations."""
        for ptr in list(self._allocations.keys()):
            try:
                if self.backend == "cuda":
                    self._cuda_free(ptr)
                elif self.backend == "rocm":
                    self._hip_free(ptr)
            except Exception:
                pass
        self._allocations.clear()
