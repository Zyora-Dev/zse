"""ZSE Kernel Profiler — Measure kernel execution time and throughput.

Zero dependency — uses ctypes to CUDA/HIP event APIs.
"""

import ctypes
import time
from dataclasses import dataclass
from typing import Optional

from zse_compiler.runtime.compiler import CompiledKernel
from zse_compiler.runtime.launcher import KernelLauncher, LaunchConfig


@dataclass
class KernelProfile:
    """Profiling results for a kernel execution."""
    kernel_name: str
    backend: str
    elapsed_ms: float  # GPU execution time in milliseconds
    grid: tuple
    block: tuple
    num_runs: int = 1

    @property
    def throughput_gflops(self) -> Optional[float]:
        """Estimated throughput — caller must set total_flops."""
        return None

    def bandwidth_gb_s(self, total_bytes: int) -> float:
        """Effective memory bandwidth in GB/s."""
        if self.elapsed_ms == 0:
            return 0.0
        return (total_bytes / 1e9) / (self.elapsed_ms / 1e3)

    def __repr__(self) -> str:
        return (f"KernelProfile('{self.kernel_name}', {self.elapsed_ms:.3f}ms, "
                f"grid={self.grid}, block={self.block})")


class KernelProfiler:
    """Profile kernel execution using GPU events for accurate timing."""

    def __init__(self):
        self._launcher = KernelLauncher()

    def profile(
        self,
        kernel: CompiledKernel,
        config: LaunchConfig,
        args: tuple,
        warmup_runs: int = 3,
        profile_runs: int = 10,
    ) -> KernelProfile:
        """Profile a kernel with warmup and multiple runs."""

        if kernel.backend == "cuda":
            return self._profile_cuda(kernel, config, args, warmup_runs, profile_runs)
        else:
            # Fallback: wall-clock timing
            return self._profile_wallclock(kernel, config, args, warmup_runs, profile_runs)

    def _profile_cuda(
        self, kernel, config, args, warmup_runs, profile_runs
    ) -> KernelProfile:
        """Profile using CUDA events for accurate GPU timing."""
        driver = kernel._driver
        if driver is None:
            return self._profile_wallclock(kernel, config, args, warmup_runs, profile_runs)

        # Warmup
        for _ in range(warmup_runs):
            self._launcher.launch(kernel, config, *args)

        # Create events
        start_event = ctypes.c_void_p()
        end_event = ctypes.c_void_p()
        driver.cuEventCreate(ctypes.byref(start_event), 0)
        driver.cuEventCreate(ctypes.byref(end_event), 0)

        # Record start
        driver.cuEventRecord(start_event, ctypes.c_void_p(0))

        # Profile runs
        for _ in range(profile_runs):
            self._launcher.launch(kernel, config, *args)

        # Record end
        driver.cuEventRecord(end_event, ctypes.c_void_p(0))
        driver.cuEventSynchronize(end_event)

        # Get elapsed time
        elapsed = ctypes.c_float(0)
        driver.cuEventElapsedTime(ctypes.byref(elapsed), start_event, end_event)

        # Cleanup
        driver.cuEventDestroy_v2(start_event)
        driver.cuEventDestroy_v2(end_event)

        return KernelProfile(
            kernel_name=kernel.name,
            backend=kernel.backend,
            elapsed_ms=elapsed.value / profile_runs,
            grid=config.grid,
            block=config.block,
            num_runs=profile_runs,
        )

    def _profile_wallclock(
        self, kernel, config, args, warmup_runs, profile_runs
    ) -> KernelProfile:
        """Fallback profiling using wall-clock time."""
        for _ in range(warmup_runs):
            self._launcher.launch(kernel, config, *args)

        start = time.perf_counter()
        for _ in range(profile_runs):
            self._launcher.launch(kernel, config, *args)
        elapsed = time.perf_counter() - start

        return KernelProfile(
            kernel_name=kernel.name,
            backend=kernel.backend,
            elapsed_ms=(elapsed * 1000) / profile_runs,
            grid=config.grid,
            block=config.block,
            num_runs=profile_runs,
        )
