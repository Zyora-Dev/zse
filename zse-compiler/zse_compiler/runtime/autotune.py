"""ZSE Auto-Tuner — Benchmark different launch configurations, pick the fastest.

Tries multiple block sizes for a kernel and selects the one with lowest execution time.
Uses the KernelProfiler for accurate GPU timing.

Usage:
    tuner = AutoTuner()
    best_config = tuner.tune(
        kernel=my_kernel,
        args=(a, b, out),
        total_threads=N,
    )
    print(f"Best: block={best_config.block}, {best_config.elapsed_ms:.3f}ms")
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from zse_compiler.runtime.compiler import CompiledKernel
from zse_compiler.runtime.launcher import LaunchConfig
from zse_compiler.runtime.profiler import KernelProfiler, KernelProfile


@dataclass
class TuneResult:
    """Result of auto-tuning a kernel."""
    kernel_name: str
    best_config: LaunchConfig
    best_time_ms: float
    all_results: List[KernelProfile]

    def __repr__(self) -> str:
        return (f"TuneResult('{self.kernel_name}', "
                f"block={self.best_config.block[:self._nonone_dims()]}, "
                f"{self.best_time_ms:.3f}ms, "
                f"tested={len(self.all_results)} configs)")

    def _nonone_dims(self) -> int:
        dims = 3
        while dims > 1 and self.best_config.block[dims-1] == 1:
            dims -= 1
        return dims

    def summary(self) -> str:
        """Print a summary table of all tested configurations."""
        lines = [f"Auto-tune results for '{self.kernel_name}':"]
        lines.append(f"{'Config':<20} {'Time (ms)':<12} {'Speedup':<10}")
        lines.append("-" * 42)

        worst = max(r.elapsed_ms for r in self.all_results) if self.all_results else 1.0
        for r in sorted(self.all_results, key=lambda x: x.elapsed_ms):
            block_str = f"{r.block[0]}"
            if r.block[1] > 1:
                block_str += f"x{r.block[1]}"
            speedup = worst / r.elapsed_ms if r.elapsed_ms > 0 else 0
            marker = " ← best" if r.elapsed_ms == self.best_time_ms else ""
            lines.append(f"  block={block_str:<12} {r.elapsed_ms:<12.3f} {speedup:<10.2f}x{marker}")

        return "\n".join(lines)


# Common block sizes to try for 1D kernels
BLOCK_SIZES_1D = [32, 64, 128, 256, 512, 1024]

# Common block sizes for 2D kernels (block_x, block_y)
BLOCK_SIZES_2D = [
    (8, 8), (16, 8), (8, 16), (16, 16),
    (32, 8), (8, 32), (32, 16), (16, 32), (32, 32),
]


class AutoTuner:
    """Auto-tune kernel launch configurations for best performance."""

    def __init__(self, warmup_runs: int = 3, profile_runs: int = 10):
        self._profiler = KernelProfiler()
        self._warmup_runs = warmup_runs
        self._profile_runs = profile_runs

    def tune_1d(
        self,
        kernel: CompiledKernel,
        args: tuple,
        total_threads: int,
        block_sizes: Optional[List[int]] = None,
    ) -> TuneResult:
        """Tune a 1D kernel by trying different block sizes."""
        if block_sizes is None:
            block_sizes = BLOCK_SIZES_1D

        results = []
        for bs in block_sizes:
            if bs > total_threads:
                continue
            if total_threads % bs != 0:
                continue

            grid_size = total_threads // bs
            config = LaunchConfig(grid=(grid_size,), block=(bs,))

            try:
                config.validate()
                profile = self._profiler.profile(
                    kernel, config, args,
                    warmup_runs=self._warmup_runs,
                    profile_runs=self._profile_runs,
                )
                results.append(profile)
            except (ValueError, RuntimeError):
                continue

        if not results:
            raise RuntimeError(f"No valid configurations found for {total_threads} threads")

        best = min(results, key=lambda r: r.elapsed_ms)
        best_config = LaunchConfig(grid=best.grid, block=best.block)

        return TuneResult(
            kernel_name=kernel.name,
            best_config=best_config,
            best_time_ms=best.elapsed_ms,
            all_results=results,
        )

    def tune_2d(
        self,
        kernel: CompiledKernel,
        args: tuple,
        grid_shape: Tuple[int, int],
        block_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> TuneResult:
        """Tune a 2D kernel by trying different block configurations."""
        if block_sizes is None:
            block_sizes = BLOCK_SIZES_2D

        rows, cols = grid_shape
        results = []

        for bx, by in block_sizes:
            if bx * by > 1024:
                continue
            if rows % by != 0 or cols % bx != 0:
                continue

            grid = (cols // bx, rows // by)
            config = LaunchConfig(grid=grid, block=(bx, by))

            try:
                config.validate()
                profile = self._profiler.profile(
                    kernel, config, args,
                    warmup_runs=self._warmup_runs,
                    profile_runs=self._profile_runs,
                )
                results.append(profile)
            except (ValueError, RuntimeError):
                continue

        if not results:
            raise RuntimeError(f"No valid 2D configurations found for {grid_shape}")

        best = min(results, key=lambda r: r.elapsed_ms)
        best_config = LaunchConfig(grid=best.grid, block=best.block)

        return TuneResult(
            kernel_name=kernel.name,
            best_config=best_config,
            best_time_ms=best.elapsed_ms,
            all_results=results,
        )
