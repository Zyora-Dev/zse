"""
ZSE Setup - CUDA Extension Compilation

This setup.py handles compilation of custom CUDA kernels for:
- zAttention: Paged, Flash, and Sparse attention
- zQuantize: INT2/3/4/8 GEMM kernels
- zKV: Quantized KV cache operations

On systems without CUDA, this gracefully falls back to Python-only installation.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext


def is_cuda_available() -> bool:
    """Check if CUDA is available for building extensions."""
    # Check CUDA_HOME environment variable
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and os.path.exists(cuda_home):
        return True
    
    # Try to find nvcc
    try:
        subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check common CUDA paths
    common_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
    ]
    for path in common_paths:
        if os.path.exists(path):
            os.environ["CUDA_HOME"] = path
            return True
    
    return False


# Early CUDA detection
CUDA_AVAILABLE = is_cuda_available()

if not CUDA_AVAILABLE:
    print("=" * 60)
    print("INFO: CUDA not found. Building Python-only version.")
    print("CUDA kernels (zAttention, zQuantize, zKV) will not be compiled.")
    print("GPU acceleration will use Triton or PyTorch fallbacks.")
    print("=" * 60)


def get_cuda_version() -> tuple[int, int] | None:
    """Detect CUDA version from nvcc."""
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        # Parse version like "release 12.1"
        for line in output.split("\n"):
            if "release" in line:
                parts = line.split("release")[-1].strip().split(",")[0].split(".")
                return int(parts[0]), int(parts[1])
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
        return None


def get_torch_cuda_arch_list() -> list[str]:
    """Get CUDA architectures supported by current PyTorch installation."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get compute capability of current GPU
            cc = torch.cuda.get_device_capability()
            arch = f"{cc[0]}.{cc[1]}"
            # Common architectures to support
            archs = ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]
            # Filter to supported and include current
            if arch not in archs:
                archs.append(arch)
            return sorted(set(archs))
    except ImportError:
        pass
    # Default architectures if torch not available
    return ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]


class BuildExtension(build_ext):
    """Custom build extension for CUDA kernels."""
    
    def build_extensions(self) -> None:
        """Build CUDA extensions if CUDA is available."""
        cuda_version = get_cuda_version()
        
        if cuda_version is None:
            print("=" * 60)
            print("WARNING: CUDA not found. Building CPU-only version.")
            print("CUDA kernels (zAttention, zQuantize, zKV) will not be available.")
            print("GPU acceleration will fall back to PyTorch/Triton implementations.")
            print("=" * 60)
            # Skip CUDA extensions
            self.extensions = [
                ext for ext in self.extensions 
                if not getattr(ext, 'is_cuda', False)
            ]
        else:
            print(f"Found CUDA {cuda_version[0]}.{cuda_version[1]}")
            arch_list = get_torch_cuda_arch_list()
            print(f"Building for architectures: {arch_list}")
            
            # Set architecture flags
            os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(arch_list)
        
        super().build_extensions()


def get_extensions():
    """Get list of extensions to build."""
    extensions = []
    
    # Skip CUDA extensions if CUDA is not available
    if not CUDA_AVAILABLE:
        return extensions
    
    # Check if CUDA source directory exists
    csrc_dir = Path(__file__).parent / "csrc"
    
    if not csrc_dir.exists():
        print("No csrc/ directory found. Skipping CUDA extensions.")
        return extensions
    
    try:
        from torch.utils.cpp_extension import CUDAExtension, CppExtension
        
        # Common compilation flags
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--use_fast_math",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
            ],
        }
        
        # zAttention CUDA extension
        attention_sources = list((csrc_dir / "attention").glob("*.cu"))
        attention_sources += list((csrc_dir / "attention").glob("*.cpp"))
        
        if attention_sources:
            ext = CUDAExtension(
                name="zse._C.attention",
                sources=[str(s) for s in attention_sources],
                extra_compile_args=extra_compile_args,
            )
            ext.is_cuda = True
            extensions.append(ext)
        
        # zQuantize CUDA extension
        quantize_sources = list((csrc_dir / "quantize").glob("*.cu"))
        quantize_sources += list((csrc_dir / "quantize").glob("*.cpp"))
        
        if quantize_sources:
            ext = CUDAExtension(
                name="zse._C.quantize",
                sources=[str(s) for s in quantize_sources],
                extra_compile_args=extra_compile_args,
            )
            ext.is_cuda = True
            extensions.append(ext)
        
        # zKV CUDA extension
        kv_sources = list((csrc_dir / "kv_cache").glob("*.cu"))
        kv_sources += list((csrc_dir / "kv_cache").glob("*.cpp"))
        
        if kv_sources:
            ext = CUDAExtension(
                name="zse._C.kv_cache",
                sources=[str(s) for s in kv_sources],
                extra_compile_args=extra_compile_args,
            )
            ext.is_cuda = True
            extensions.append(ext)
            
    except ImportError:
        print("PyTorch not found. Skipping CUDA extensions.")
    
    return extensions


# Only run setup if this file is executed directly
# (pyproject.toml handles the main build configuration)
if __name__ == "__main__":
    setup(
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )
