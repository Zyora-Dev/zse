"""ZSE Kernel Launcher — Launch compiled kernels on GPU.

Handles grid/block configuration and argument passing via ctypes.
"""

import ctypes
from typing import Tuple, List, Any, Optional
from dataclasses import dataclass

from zse_compiler.runtime.compiler import CompiledKernel
from zse_compiler.types.tensor import Tensor


@dataclass
class LaunchConfig:
    """Kernel launch configuration."""
    grid: Tuple[int, ...]   # (grid_x, grid_y, grid_z)
    block: Tuple[int, ...]  # (block_x, block_y, block_z)
    shared_mem_bytes: int = 0
    stream: Optional[ctypes.c_void_p] = None

    def __post_init__(self):
        # Pad to 3D
        while len(self.grid) < 3:
            self.grid = self.grid + (1,)
        while len(self.block) < 3:
            self.block = self.block + (1,)

    def validate(self):
        """Validate launch configuration."""
        # Grid dimensions must be positive
        for i, g in enumerate(self.grid):
            if g <= 0:
                raise ValueError(f"Grid dimension {i} must be > 0, got {g}")

        # Block dimensions must be positive
        for i, b in enumerate(self.block):
            if b <= 0:
                raise ValueError(f"Block dimension {i} must be > 0, got {b}")

        # Total threads per block limit (1024 for most GPUs)
        total_threads = self.block[0] * self.block[1] * self.block[2]
        if total_threads > 1024:
            raise ValueError(
                f"Total threads per block ({total_threads}) exceeds maximum (1024). "
                f"block=({self.block[0]}, {self.block[1]}, {self.block[2]})"
            )

        # Block dimensions individual limits
        if self.block[0] > 1024:
            raise ValueError(f"block.x ({self.block[0]}) exceeds max (1024)")
        if self.block[1] > 1024:
            raise ValueError(f"block.y ({self.block[1]}) exceeds max (1024)")
        if self.block[2] > 64:
            raise ValueError(f"block.z ({self.block[2]}) exceeds max (64)")

        # Grid dimension limits (2^31 - 1 for x, 65535 for y/z)
        if self.grid[0] > 2147483647:
            raise ValueError(f"grid.x ({self.grid[0]}) exceeds max (2^31-1)")
        if self.grid[1] > 65535:
            raise ValueError(f"grid.y ({self.grid[1]}) exceeds max (65535)")
        if self.grid[2] > 65535:
            raise ValueError(f"grid.z ({self.grid[2]}) exceeds max (65535)")

        # Shared memory limit (typically 48KB default, 100KB+ with opt-in)
        if self.shared_mem_bytes > 166912:  # 163KB max on H100
            raise ValueError(
                f"Shared memory ({self.shared_mem_bytes} bytes) exceeds max (163KB)"
            )


class KernelLauncher:
    """Launches compiled kernels on GPU hardware."""

    def launch(self, kernel: CompiledKernel, config: LaunchConfig, *args):
        """Launch a compiled kernel with given configuration and arguments.

        Kernels are launched asynchronously — no GPU sync after launch.
        GPU results are available after the next host←device transfer
        (cuMemcpyDtoH is synchronous and waits for all prior work).
        """
        if kernel.backend == "cuda":
            self._launch_cuda(kernel, config, args)
        elif kernel.backend == "rocm":
            self._launch_rocm(kernel, config, args)
        elif kernel.backend == "metal":
            self._launch_metal(kernel, config, args)
        else:
            raise ValueError(f"Cannot launch on backend: {kernel.backend}")

    def launch_prepacked(self, kernel: CompiledKernel, config: LaunchConfig,
                         prepacked: 'PrepackedArgs'):
        """Launch with pre-packed arguments — zero allocation per call.

        Used in the decode hot path where the same kernel is launched 960× per token.
        The PrepackedArgs object's values are mutated in-place between calls.
        """
        if kernel.backend == "cuda":
            status = kernel._driver.cuLaunchKernel(
                kernel.function,
                config.grid[0], config.grid[1], config.grid[2],
                config.block[0], config.block[1], config.block[2],
                config.shared_mem_bytes,
                config.stream or ctypes.c_void_p(0),
                prepacked.arg_array,
                ctypes.c_void_p(0),
            )
            if status != 0:
                kernel._driver.cuCtxSynchronize()
                raise RuntimeError(f"cuLaunchKernel failed: {status}")
        elif kernel.backend == "rocm":
            status = kernel._driver.hipModuleLaunchKernel(
                kernel.function,
                config.grid[0], config.grid[1], config.grid[2],
                config.block[0], config.block[1], config.block[2],
                config.shared_mem_bytes,
                config.stream or ctypes.c_void_p(0),
                prepacked.arg_array,
                ctypes.c_void_p(0),
            )
            if status != 0:
                kernel._driver.hipDeviceSynchronize()
                raise RuntimeError(f"hipModuleLaunchKernel failed: {status}")

    def sync(self, kernel: CompiledKernel):
        """Explicit GPU synchronization — only call when you need results on CPU."""
        if kernel.backend == "cuda":
            kernel._driver.cuCtxSynchronize()
        elif kernel.backend == "rocm":
            kernel._driver.hipDeviceSynchronize()

    def _launch_cuda(self, kernel: CompiledKernel, config: LaunchConfig, args: tuple):
        """Launch CUDA kernel via cuLaunchKernel (async — no sync)."""
        driver = kernel._driver
        if driver is None:
            raise RuntimeError("CUDA driver not available")

        kernel_args = self._prepare_args_cuda(args)

        status = driver.cuLaunchKernel(
            kernel.function,
            config.grid[0], config.grid[1], config.grid[2],
            config.block[0], config.block[1], config.block[2],
            config.shared_mem_bytes,
            config.stream or ctypes.c_void_p(0),
            kernel_args,
            ctypes.c_void_p(0),
        )

        if status != 0:
            # Sync to get the real error
            driver.cuCtxSynchronize()
            raise RuntimeError(f"cuLaunchKernel failed with status {status}")

    def _launch_rocm(self, kernel: CompiledKernel, config: LaunchConfig, args: tuple):
        """Launch HIP kernel via hipModuleLaunchKernel (async — no sync)."""
        hip = kernel._driver
        if hip is None:
            raise RuntimeError("HIP runtime not available")

        kernel_args = self._prepare_args_cuda(args)

        status = hip.hipModuleLaunchKernel(
            kernel.function,
            config.grid[0], config.grid[1], config.grid[2],
            config.block[0], config.block[1], config.block[2],
            config.shared_mem_bytes,
            config.stream or ctypes.c_void_p(0),
            kernel_args,
            ctypes.c_void_p(0),
        )

        if status != 0:
            hip.hipDeviceSynchronize()
            raise RuntimeError(f"hipModuleLaunchKernel failed with status {status}")

    def _launch_metal(self, kernel: CompiledKernel, config: LaunchConfig, args: tuple):
        """Launch Metal kernel — requires Metal Python bridge or ctypes to ObjC."""
        # Metal launch is more complex — requires command buffer, encoder, etc.
        # This is a placeholder for the Metal dispatch pipeline
        raise NotImplementedError(
            "Metal kernel launch requires the Metal framework bridge. "
            "Use zse_compiler.runtime.metal_bridge for macOS dispatch."
        )

    def _prepare_args_cuda(self, args: tuple):
        """Convert Python arguments to CUDA kernel argument array."""
        arg_ptrs = []
        # Keep references alive
        self._arg_refs = []

        for arg in args:
            if isinstance(arg, Tensor):
                # Pass device pointer
                ptr = ctypes.c_void_p(arg.data_ptr)
                self._arg_refs.append(ptr)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(ptr), ctypes.c_void_p))
            elif isinstance(arg, int):
                val = ctypes.c_int(arg)
                self._arg_refs.append(val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(val), ctypes.c_void_p))
            elif isinstance(arg, float):
                val = ctypes.c_float(arg)
                self._arg_refs.append(val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(val), ctypes.c_void_p))
            else:
                raise TypeError(f"Unsupported kernel argument type: {type(arg)}")

        # Create array of void pointers
        arr = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
        self._arg_refs.append(arr)
        return arr


class PrepackedArgs:
    """Pre-allocated kernel argument array for zero-allocation launches.

    CUDA/HIP kernel launch expects void** — array of pointers, each pointing
    to the value to pass. We pre-allocate a contiguous buffer of values and
    a stable pointer array. Updates mutate values in-place without allocation.

    Usage:
        packed = PrepackedArgs.from_args(out_t, inp_t, 5120, 1e-5)
        packed.set_int(2, new_value)  # fast in-place update
        launcher.launch_prepacked(kernel, config, packed)
    """

    __slots__ = ('_num_args', '_int_slots', '_float_slots', '_ptr_slots', '_slot_ptrs', '_arr')

    def __init__(self, num_args: int):
        self._num_args = num_args
        # Each arg gets a stable ctypes slot (c_void_p for ptrs, c_int for ints, c_float for floats)
        # We use c_void_p for all since kernel API just reads bytes from the address
        self._int_slots = []    # (index, c_int)
        self._float_slots = []  # (index, c_float)
        self._ptr_slots = []    # (index, c_void_p)

        # Pre-allocate storage: one c_void_p per arg for the pointer value
        # and one array of void* pointing to each slot
        self._slot_ptrs = [None] * num_args  # ctypes objects holding values
        self._arr = None  # built after all slots set

    @classmethod
    def from_args(cls, *args):
        """Build a PrepackedArgs from initial argument values."""
        n = len(args)
        packed = cls(n)
        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                slot = ctypes.c_void_p(arg.data_ptr)
                packed._slot_ptrs[i] = slot
            elif isinstance(arg, int):
                slot = ctypes.c_int(arg)
                packed._slot_ptrs[i] = slot
            elif isinstance(arg, float):
                slot = ctypes.c_float(arg)
                packed._slot_ptrs[i] = slot
            else:
                raise TypeError(f"Unsupported arg type: {type(arg)}")
        # Build the void** array (pointers to each slot)
        ptr_array = (ctypes.c_void_p * n)()
        for i in range(n):
            ptr_array[i] = ctypes.cast(ctypes.pointer(packed._slot_ptrs[i]), ctypes.c_void_p)
        packed._arr = ptr_array
        return packed

    def set_ptr(self, index: int, ptr_value: int):
        """Update a pointer argument in-place."""
        self._slot_ptrs[index].value = ptr_value

    def set_int(self, index: int, value: int):
        """Update an integer argument in-place."""
        self._slot_ptrs[index].value = value

    def set_float(self, index: int, value: float):
        """Update a float argument in-place."""
        self._slot_ptrs[index].value = value

    @property
    def arg_array(self):
        """Get the pre-built void** argument array for launch."""
        return self._arr
