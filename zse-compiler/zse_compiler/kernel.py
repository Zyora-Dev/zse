"""ZSE @kernel decorator — The main entry point for writing GPU kernels.

Usage:
    import zse_compiler as zse

    @zse.kernel
    def vector_add(a: zse.Tensor, b: zse.Tensor, out: zse.Tensor):
        idx = zse.global_id(0)
        out[idx] = a[idx] + b[idx]

    # Inspect generated code
    print(vector_add.source("cuda"))

    # Compile for a backend
    vector_add.compile(backend="cuda")

    # Launch
    vector_add.launch(grid=(N // 256,), block=(256,), args=(a, b, out))
"""

from typing import Optional, Tuple, Callable, Dict
from zse_compiler.ast_parser.parser import KernelParser
from zse_compiler.ast_parser.validator import validate_kernel
from zse_compiler.ir.nodes import IRFunction
from zse_compiler.codegen.cuda import CUDACodegen
from zse_compiler.codegen.rocm import ROCmCodegen
from zse_compiler.codegen.metal import MetalCodegen
from zse_compiler.runtime.compiler import RuntimeCompiler, CompiledKernel
from zse_compiler.runtime.launcher import KernelLauncher, LaunchConfig
from zse_compiler.runtime.device import detect_backend


CODEGEN_MAP = {
    "cuda": CUDACodegen,
    "rocm": ROCmCodegen,
    "metal": MetalCodegen,
}


class KernelFunction:
    """A compiled GPU kernel function."""

    def __init__(self, func: Callable):
        self._func = func
        self._name = func.__name__
        self._ir: Optional[IRFunction] = None
        self._sources: Dict[str, str] = {}  # backend → source code
        self._compiled: Dict[str, CompiledKernel] = {}  # backend → compiled kernel
        self._compiler = RuntimeCompiler()
        self._launcher = KernelLauncher()

        # Validate
        warnings = validate_kernel(func)
        for w in warnings:
            print(f"[zse] warning: {w}")

        # Parse to IR immediately
        parser = KernelParser()
        self._ir = parser.parse(func)

    @property
    def name(self) -> str:
        return self._name

    @property
    def ir(self) -> IRFunction:
        return self._ir

    def source(self, backend: Optional[str] = None) -> str:
        """Generate source code for the given backend."""
        if backend is None:
            backend = detect_backend()

        if backend not in self._sources:
            if backend not in CODEGEN_MAP:
                raise ValueError(f"Unknown backend: {backend}. Available: {list(CODEGEN_MAP.keys())}")
            codegen = CODEGEN_MAP[backend]()
            self._sources[backend] = codegen.generate(self._ir)

        return self._sources[backend]

    def compile(self, backend: Optional[str] = None) -> CompiledKernel:
        """Compile the kernel for the given backend."""
        if backend is None:
            backend = detect_backend()

        if backend not in self._compiled:
            src = self.source(backend)
            self._compiled[backend] = self._compiler.compile(src, self._name, backend)

        return self._compiled[backend]

    def launch(
        self,
        grid: Tuple[int, ...],
        block: Tuple[int, ...],
        args: tuple,
        backend: Optional[str] = None,
        shared_mem_bytes: int = 0,
    ):
        """Compile (if needed) and launch the kernel."""
        if backend is None:
            backend = detect_backend()

        compiled = self.compile(backend)
        config = LaunchConfig(grid=grid, block=block, shared_mem_bytes=shared_mem_bytes)
        self._launcher.launch(compiled, config, *args)

    def __call__(self, *args, **kwargs):
        """Calling the kernel directly is not supported — use .launch()"""
        raise RuntimeError(
            f"Cannot call @zse.kernel '{self._name}' directly. "
            f"Use {self._name}.launch(grid=..., block=..., args=...) instead."
        )

    def __repr__(self) -> str:
        backends = list(self._compiled.keys()) or ["not compiled"]
        return f"zse.KernelFunction('{self._name}', compiled={backends})"


def kernel(func: Callable) -> KernelFunction:
    """Decorator to mark a Python function as a GPU kernel.

    @zse.kernel
    def my_kernel(a: zse.Tensor, b: zse.Tensor, out: zse.Tensor):
        idx = zse.global_id(0)
        out[idx] = a[idx] + b[idx]
    """
    return KernelFunction(func)


def fuse(kernels: list, name: str = None) -> 'KernelFunction':
    """Fuse multiple element-wise kernels into a single kernel.

    Eliminates intermediate global memory reads/writes between kernels.

    Usage:
        fused = zse.fuse([add_bias, silu], name="add_bias_silu")
        fused.launch(grid=..., block=..., args=(...))
    """
    from zse_compiler.ir.fusion import FusionPass

    fusion = FusionPass()
    fused_ir = fusion.fuse(kernels, name=name)

    # Create a KernelFunction wrapper for the fused IR
    fused = object.__new__(KernelFunction)
    fused._func = None
    fused._name = fused_ir.name
    fused._ir = fused_ir
    fused._sources = {}
    fused._compiled = {}
    fused._compiler = RuntimeCompiler()
    fused._launcher = KernelLauncher()
    return fused
