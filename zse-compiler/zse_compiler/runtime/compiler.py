"""ZSE Runtime Compiler — Compiles generated source code using platform compilers.

Uses ctypes to call:
- nvrtc (NVIDIA Runtime Compilation) for CUDA
- hiprtc (AMD) for ROCm
- Metal compiler framework for Apple

Zero dependency — no pycuda, no PyTorch.
"""

import ctypes
import ctypes.util
import tempfile
import hashlib
import os
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class CompiledKernel:
    """A compiled GPU kernel ready to launch."""
    name: str
    backend: str
    module: ctypes.c_void_p  # GPU module handle
    function: ctypes.c_void_p  # GPU function handle
    source: str  # Original source for debugging
    _driver: object = None  # Keep reference to driver lib

    def __repr__(self) -> str:
        return f"CompiledKernel(name='{self.name}', backend='{self.backend}')"


class RuntimeCompiler:
    """Compiles kernel source code to GPU binary at runtime."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir or os.path.expanduser("~/.zse/cache"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._compiled_cache: Dict[str, CompiledKernel] = {}

    def compile(self, source: str, kernel_name: str, backend: str) -> CompiledKernel:
        """Compile source code to a GPU kernel."""
        # Check cache
        cache_key = self._cache_key(source, backend)
        if cache_key in self._compiled_cache:
            return self._compiled_cache[cache_key]

        if backend == "cuda":
            result = self._compile_cuda(source, kernel_name)
        elif backend == "rocm":
            result = self._compile_rocm(source, kernel_name)
        elif backend == "metal":
            result = self._compile_metal(source, kernel_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._compiled_cache[cache_key] = result
        return result

    def _cache_key(self, source: str, backend: str) -> str:
        return hashlib.sha256(f"{backend}:{source}".encode()).hexdigest()

    # --- CUDA Compilation via NVRTC ---

    def _compile_cuda(self, source: str, kernel_name: str) -> CompiledKernel:
        """Compile CUDA C source using nvrtc → cuModuleLoadData → cuModuleGetFunction."""
        nvrtc = self._load_nvrtc()
        driver = self._load_cuda_driver()

        if nvrtc is None:
            raise RuntimeError("NVRTC not found. Install CUDA toolkit or ensure libnvrtc is in PATH.")
        if driver is None:
            raise RuntimeError("CUDA driver not found.")

        # Initialize CUDA
        driver.cuInit(0)

        # Use existing context if available, otherwise create one
        ctx = ctypes.c_void_p()
        driver.cuCtxGetCurrent(ctypes.byref(ctx))
        device = ctypes.c_int(0)
        if ctx.value is None or ctx.value == 0:
            driver.cuDeviceGet(ctypes.byref(device), 0)
            driver.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
        else:
            # Get device backing the current context
            driver.cuCtxGetDevice(ctypes.byref(device))

        # Query compute capability of the current device so we don't hardcode SM 80
        arch_str = self._detect_cuda_arch(driver, device)

        # Create NVRTC program
        source_bytes = source.encode("utf-8")
        prog = ctypes.c_void_p()
        status = nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog),
            source_bytes,
            b"zse_kernel.cu",
            0, None, None
        )
        if status != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed with status {status}")

        # Compile — arch matches the actual device (T4=sm_75, V100=sm_70, A100=sm_80, H100=sm_90, etc.)
        options = [f"--gpu-architecture={arch_str}".encode(), b"-default-device"]
        # Add CUDA include paths for cuda_fp16.h etc.
        for inc_path in self._find_cuda_include_paths():
            options.append(f"--include-path={inc_path}".encode())
        opts_array = (ctypes.c_char_p * len(options))(*options)
        status = nvrtc.nvrtcCompileProgram(prog, len(options), opts_array)

        if status != 0:
            # Get compile log
            log_size = ctypes.c_size_t(0)
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            nvrtc.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError(f"NVRTC compilation failed:\n{log_buf.value.decode('utf-8')}")

        # Get PTX
        ptx_size = ctypes.c_size_t(0)
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        nvrtc.nvrtcGetPTX(prog, ptx_buf)

        # Load module from PTX
        module = ctypes.c_void_p()
        status = driver.cuModuleLoadData(ctypes.byref(module), ptx_buf)
        if status != 0:
            raise RuntimeError(f"cuModuleLoadData failed with status {status}")

        # Get function
        func = ctypes.c_void_p()
        status = driver.cuModuleGetFunction(ctypes.byref(func), module, kernel_name.encode("utf-8"))
        if status != 0:
            raise RuntimeError(f"cuModuleGetFunction failed for '{kernel_name}' with status {status}")

        # Cleanup
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

        return CompiledKernel(
            name=kernel_name,
            backend="cuda",
            module=module,
            function=func,
            source=source,
            _driver=driver,
        )

    # --- ROCm Compilation via HIPRTC ---

    def _compile_rocm(self, source: str, kernel_name: str) -> CompiledKernel:
        """Compile HIP C source using hiprtc."""
        hiprtc = self._load_hiprtc()
        hip = self._load_hip_runtime()

        if hiprtc is None:
            raise RuntimeError("HIPRTC not found. Install ROCm toolkit.")
        if hip is None:
            raise RuntimeError("HIP runtime not found.")

        # Create program
        source_bytes = source.encode("utf-8")
        prog = ctypes.c_void_p()
        hiprtc.hiprtcCreateProgram(
            ctypes.byref(prog),
            source_bytes,
            b"zse_kernel.hip",
            0, None, None
        )

        # Compile with include paths
        options = []
        for inc_path in self._find_rocm_include_paths():
            options.append(f"-I{inc_path}".encode())
        if options:
            opts_array = (ctypes.c_char_p * len(options))(*options)
            status = hiprtc.hiprtcCompileProgram(prog, len(options), opts_array)
        else:
            status = hiprtc.hiprtcCompileProgram(prog, 0, None)
        if status != 0:
            log_size = ctypes.c_size_t(0)
            hiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(log_size))
            log_buf = ctypes.create_string_buffer(log_size.value)
            hiprtc.hiprtcGetProgramLog(prog, log_buf)
            raise RuntimeError(f"HIPRTC compilation failed:\n{log_buf.value.decode('utf-8')}")

        # Get code
        code_size = ctypes.c_size_t(0)
        hiprtc.hiprtcGetCodeSize(prog, ctypes.byref(code_size))
        code_buf = ctypes.create_string_buffer(code_size.value)
        hiprtc.hiprtcGetCode(prog, code_buf)

        # Load module
        module = ctypes.c_void_p()
        hip.hipModuleLoadData(ctypes.byref(module), code_buf)

        # Get function
        func = ctypes.c_void_p()
        hip.hipModuleGetFunction(ctypes.byref(func), module, kernel_name.encode("utf-8"))

        hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))

        return CompiledKernel(
            name=kernel_name,
            backend="rocm",
            module=module,
            function=func,
            source=source,
            _driver=hip,
        )

    # --- Metal Compilation ---

    def _compile_metal(self, source: str, kernel_name: str) -> CompiledKernel:
        """Compile Metal source at runtime using Metal's GPU driver compiler.

        Uses the C bridge (metal_dispatch.m) which calls newLibraryWithSource:
        — compiles MSL on-device, needs only Command Line Tools (no Xcode).
        """
        from .metal_dispatch import get_metal_runtime

        rt = get_metal_runtime()
        pipeline = rt.compile_msl(source, kernel_name)

        return CompiledKernel(
            name=kernel_name,
            backend="metal",
            module=pipeline,       # MTLComputePipelineState ptr
            function=pipeline,     # Same ptr — pipeline IS the callable
            source=source,
            _driver=rt,            # Keep runtime alive
        )

    # --- Library Loading ---

    # Cache: device ordinal → "compute_XY" string
    _arch_cache: Dict[int, str] = {}

    @classmethod
    def _detect_cuda_arch(cls, driver, device) -> str:
        """Query compute capability of a CUDA device and return the NVRTC arch string.

        Falls back to compute_80 if the query fails (matches old hardcoded behavior).
        """
        try:
            ordinal = device.value if hasattr(device, "value") else int(device)
        except Exception:
            ordinal = 0
        if ordinal in cls._arch_cache:
            return cls._arch_cache[ordinal]

        try:
            major = ctypes.c_int(0)
            minor = ctypes.c_int(0)
            # 75 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            # 76 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
            rc1 = driver.cuDeviceGetAttribute(ctypes.byref(major), 75, device)
            rc2 = driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)
            if rc1 == 0 and rc2 == 0 and major.value > 0:
                arch = f"compute_{major.value}{minor.value}"
            else:
                arch = "compute_80"
        except Exception:
            arch = "compute_80"
        cls._arch_cache[ordinal] = arch
        return arch

    @staticmethod
    def _find_cuda_include_paths():
        """Find CUDA toolkit include directories for cuda_fp16.h etc."""
        import glob
        candidates = [
            "/usr/local/cuda/include",
            "/usr/local/cuda-12/include",
            "/usr/local/cuda-12.4/include",
        ]
        # Also check CUDA_HOME / CUDA_PATH env vars
        for env in ["CUDA_HOME", "CUDA_PATH"]:
            val = os.environ.get(env)
            if val:
                candidates.insert(0, os.path.join(val, "include"))
        # Search /usr/local/cuda-* pattern
        for d in glob.glob("/usr/local/cuda-*/include"):
            candidates.append(d)
        paths = []
        for c in candidates:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "cuda_fp16.h")):
                paths.append(c)
                break  # One is enough
        return paths

    @staticmethod
    def _find_rocm_include_paths():
        """Find ROCm include directories for hip/hip_runtime.h."""
        candidates = [
            "/opt/rocm/include",
            "/opt/rocm-7.2.0/include",
        ]
        for env in ["ROCM_PATH", "HIP_PATH"]:
            val = os.environ.get(env)
            if val:
                candidates.insert(0, os.path.join(val, "include"))
        import glob
        for d in glob.glob("/opt/rocm-*/include"):
            candidates.append(d)
        paths = []
        for c in candidates:
            if os.path.isdir(c) and os.path.isdir(os.path.join(c, "hip")):
                paths.append(c)
                break
        return paths

    @staticmethod
    def _load_nvrtc():
        names = ["libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll"]
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        path = ctypes.util.find_library("nvrtc")
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass
        return None

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
    def _load_hiprtc():
        names = ["libhiprtc.so", "hiprtc.dll"]
        for name in names:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
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

