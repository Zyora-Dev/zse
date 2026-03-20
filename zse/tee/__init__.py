"""
TEE (Trusted Execution Environment) Module

Hardware-based privacy protection for GPUs that support it.

Supported TEE implementations:
- NVIDIA Confidential Computing (H100, H200, GH200)
- AMD SEV (CPU-side memory encryption)
- AMD GPU TEE (future MI300X+)

The TEE layer provides hardware-enforced privacy:
- Memory is encrypted in hardware
- Even the system administrator cannot read data
- Code execution is verified by hardware
"""

from .base import (
    TEEBackend,
    TEECapabilities,
    TEESession,
    TEEError,
    TEEType,
    SoftwareTEEBackend,
)
from .detector import (
    TEEDetector,
    TEEDetectionResult,
    TEESelectionStrategy,
    detect_tee,
    get_best_tee,
    print_tee_status,
)
from .nvidia_cc import NvidiaCCBackend
from .amd_sev import AMDSEVBackend
from .amd_gpu import AMDGPUTEEBackend

__all__ = [
    # Base
    "TEEBackend",
    "TEECapabilities", 
    "TEESession",
    "TEEError",
    "TEEType",
    "SoftwareTEEBackend",
    # Detection
    "TEEDetector",
    "TEEDetectionResult",
    "TEESelectionStrategy",
    "detect_tee",
    "get_best_tee",
    "print_tee_status",
    # Backends
    "NvidiaCCBackend",
    "AMDSEVBackend",
    "AMDGPUTEEBackend",
]
