"""ZSE Device Detection — Auto-detect GPU backend and capabilities.

Zero dependency — uses ctypes to directly call driver APIs.
No PyTorch, no pycuda, no pyopencl.
"""

import ctypes
import ctypes.util
import platform
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DeviceInfo:
    backend: str  # "cuda", "rocm", "metal", "cpu"
    name: str
    index: int
    compute_capability: Optional[str] = None  # e.g. "8.0" for A100
    vram_total_bytes: int = 0
    vram_free_bytes: int = 0
    max_threads_per_block: int = 1024
    max_shared_memory_bytes: int = 49152  # 48KB default
    warp_size: int = 32
    multiprocessor_count: int = 0

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_bytes / (1024 ** 3)

    @property
    def vram_free_gb(self) -> float:
        return self.vram_free_bytes / (1024 ** 3)

    def __repr__(self) -> str:
        return (f"DeviceInfo(backend='{self.backend}', name='{self.name}', "
                f"vram={self.vram_total_gb:.1f}GB, free={self.vram_free_gb:.1f}GB)")


def detect_backend() -> str:
    """Detect best available GPU backend."""
    if _has_cuda():
        return "cuda"
    if _has_rocm():
        return "rocm"
    if _has_metal():
        return "metal"
    return "cpu"


def get_devices(backend: Optional[str] = None) -> List[DeviceInfo]:
    """Get all available GPU devices."""
    if backend is None:
        backend = detect_backend()

    if backend == "cuda":
        return _get_cuda_devices()
    elif backend == "rocm":
        return _get_rocm_devices()
    elif backend == "metal":
        return _get_metal_devices()
    return []


# --- CUDA Detection (via ctypes → libcuda / libnvrtc) ---

def _has_cuda() -> bool:
    try:
        lib = _load_cuda_driver()
        return lib is not None
    except Exception:
        return False


def _load_cuda_driver():
    """Load CUDA driver library via ctypes."""
    names = ["libcuda.so.1", "libcuda.so", "libcuda.dylib", "nvcuda.dll", "cuda"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    # Try ctypes.util
    path = ctypes.util.find_library("cuda")
    if path:
        try:
            return ctypes.CDLL(path)
        except OSError:
            pass
    return None


def _load_cuda_runtime():
    """Load CUDA runtime library."""
    names = ["libcudart.so.12", "libcudart.so.11", "libcudart.so", "cudart64_12.dll", "cudart64_11.dll"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    path = ctypes.util.find_library("cudart")
    if path:
        try:
            return ctypes.CDLL(path)
        except OSError:
            pass
    return None


def _get_cuda_devices() -> List[DeviceInfo]:
    """Query CUDA devices via driver API."""
    devices = []
    try:
        driver = _load_cuda_driver()
        if driver is None:
            return []

        # cuInit
        result = driver.cuInit(0)
        if result != 0:
            return []

        # cuDeviceGetCount
        count = ctypes.c_int(0)
        driver.cuDeviceGetCount(ctypes.byref(count))

        for i in range(count.value):
            device = ctypes.c_int(0)
            driver.cuDeviceGet(ctypes.byref(device), i)

            # Device name
            name_buf = ctypes.create_string_buffer(256)
            driver.cuDeviceGetName(name_buf, 256, device)
            name = name_buf.value.decode("utf-8")

            # Compute capability
            major = ctypes.c_int(0)
            minor = ctypes.c_int(0)
            driver.cuDeviceGetAttribute(ctypes.byref(major), 75, device)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

            # Total memory
            total_mem = ctypes.c_size_t(0)
            driver.cuDeviceTotalMem_v2(ctypes.byref(total_mem), device)

            # Multiprocessor count
            sm_count = ctypes.c_int(0)
            driver.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, device)  # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT

            # Max threads per block
            max_threads = ctypes.c_int(0)
            driver.cuDeviceGetAttribute(ctypes.byref(max_threads), 1, device)  # CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK

            # Max shared memory per block
            max_smem = ctypes.c_int(0)
            driver.cuDeviceGetAttribute(ctypes.byref(max_smem), 8, device)  # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK

            # Warp size
            warp_sz = ctypes.c_int(0)
            driver.cuDeviceGetAttribute(ctypes.byref(warp_sz), 10, device)  # CU_DEVICE_ATTRIBUTE_WARP_SIZE

            devices.append(DeviceInfo(
                backend="cuda",
                name=name,
                index=i,
                compute_capability=f"{major.value}.{minor.value}",
                vram_total_bytes=total_mem.value,
                vram_free_bytes=total_mem.value,  # Approximate — need context for exact free
                max_threads_per_block=max_threads.value,
                max_shared_memory_bytes=max_smem.value,
                warp_size=warp_sz.value,
                multiprocessor_count=sm_count.value,
            ))

    except Exception:
        pass

    return devices


# --- ROCm Detection ---

def _has_rocm() -> bool:
    try:
        lib = _load_hip_runtime()
        return lib is not None
    except Exception:
        return False


def _load_hip_runtime():
    """Load HIP runtime library."""
    names = ["libamdhip64.so", "libamdhip64.so.5", "amdhip64.dll"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    path = ctypes.util.find_library("amdhip64")
    if path:
        try:
            return ctypes.CDLL(path)
        except OSError:
            pass
    return None


def _get_rocm_devices() -> List[DeviceInfo]:
    """Query ROCm/HIP devices."""
    devices = []
    try:
        hip = _load_hip_runtime()
        if hip is None:
            return []

        hip.hipInit(0)

        count = ctypes.c_int(0)
        hip.hipGetDeviceCount(ctypes.byref(count))

        for i in range(count.value):
            # Device name
            name_buf = ctypes.create_string_buffer(256)
            hip.hipDeviceGetName(name_buf, 256, i)
            name = name_buf.value.decode("utf-8")

            # Total memory
            total_mem = ctypes.c_size_t(0)
            hip.hipDeviceTotalMem(ctypes.byref(total_mem), i)

            # Attributes (HIP enum values, alphabetical order)
            def _hip_attr(attr_id, dev=i):
                val = ctypes.c_int(0)
                hip.hipDeviceGetAttribute(ctypes.byref(val), attr_id, dev)
                return val.value

            devices.append(DeviceInfo(
                backend="rocm",
                name=name,
                index=i,
                vram_total_bytes=total_mem.value,
                vram_free_bytes=total_mem.value,  # Approximate
                max_threads_per_block=_hip_attr(56) or 1024,
                max_shared_memory_bytes=_hip_attr(70) or 65536,
                warp_size=_hip_attr(87) or 64,
                multiprocessor_count=_hip_attr(63),
            ))

    except Exception:
        pass
    return devices


# --- Metal Detection ---

def _has_metal() -> bool:
    return platform.system() == "Darwin"


def _get_metal_devices() -> List[DeviceInfo]:
    """Detect Metal devices on macOS.

    Uses the Metal ObjC bridge for the real device name + recommendedMaxWorkingSetSize.
    Falls back to a generic entry if the bridge isn't available yet (e.g. clang
    missing on first run).
    """
    if not _has_metal():
        return []

    name = "Apple GPU"
    vram_bytes = 0
    try:
        from zse_compiler.runtime.metal_dispatch import MetalRuntime
        rt = MetalRuntime()
        name = rt.device_name or "Apple GPU"
        vram_bytes = int(rt.device_memory_bytes or 0)
    except Exception:
        # Bridge unavailable (e.g. no clang yet). Keep generic entry.
        pass

    # Apple Silicon uses unified memory: total == free for our purposes.
    devices = [DeviceInfo(
        backend="metal",
        name=name,
        index=0,
        warp_size=32,  # Apple calls these "SIMD groups", width=32
        vram_total_bytes=vram_bytes,
        vram_free_bytes=vram_bytes,
        max_threads_per_block=1024,
        max_shared_memory_bytes=32768,  # 32KB threadgroup memory on Apple7+
    )]
    return devices
