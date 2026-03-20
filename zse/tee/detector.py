"""
TEE Detector - Auto-detect TEE capabilities across vendors.

Provides a unified interface for detecting available TEE backends
and selecting the best option for the current hardware.
"""

import logging
from typing import Optional, List, Dict, Type
from dataclasses import dataclass, field
from enum import Enum

from .base import TEEBackend, TEECapabilities, TEESession, TEEError, TEEType, SoftwareTEEBackend
from .nvidia_cc import NvidiaCCBackend
from .amd_sev import AMDSEVBackend
from .amd_gpu import AMDGPUTEEBackend

logger = logging.getLogger(__name__)


class TEESelectionStrategy(Enum):
    """Strategy for selecting TEE backend."""
    
    # Use the most secure option available
    MOST_SECURE = "most_secure"
    
    # Prefer GPU-based TEE for compute
    PREFER_GPU = "prefer_gpu"
    
    # Prefer CPU-based TEE (more mature)
    PREFER_CPU = "prefer_cpu"
    
    # Use specific backend
    NVIDIA_ONLY = "nvidia_only"
    AMD_ONLY = "amd_only"
    
    # Always use software fallback
    SOFTWARE_ONLY = "software_only"


@dataclass
class TEEDetectionResult:
    """Result of TEE detection."""
    
    # All backends detected
    backends: Dict[TEEType, TEECapabilities] = field(default_factory=dict)
    
    # Best available backend
    recommended: Optional[TEEType] = None
    recommended_backend: Optional[TEEBackend] = None
    
    # Detection summary
    has_hardware_tee: bool = False
    has_gpu_tee: bool = False
    has_cpu_tee: bool = False
    
    # Errors encountered
    errors: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Get detection summary."""
        lines = ["TEE Detection Results:", "=" * 40]
        
        for tee_type, caps in self.backends.items():
            status = "✓ Available" if caps.available else "✗ Not Available"
            lines.append(f"\n{tee_type.value}:")
            lines.append(f"  Status: {status}")
            lines.append(f"  Device: {caps.device_name}")
            if caps.available:
                lines.append(f"  Encryption: {caps.encryption_algorithm or 'N/A'}")
                lines.append(f"  Remote Attestation: {caps.remote_attestation}")
            if caps.error_message:
                lines.append(f"  Note: {caps.error_message}")
        
        lines.append("\n" + "=" * 40)
        lines.append(f"Hardware TEE: {'✓' if self.has_hardware_tee else '✗'}")
        lines.append(f"GPU TEE: {'✓' if self.has_gpu_tee else '✗'}")
        lines.append(f"CPU TEE: {'✓' if self.has_cpu_tee else '✗'}")
        
        if self.recommended:
            lines.append(f"\nRecommended: {self.recommended.value}")
        
        if self.errors:
            lines.append(f"\nWarnings: {len(self.errors)}")
        
        return "\n".join(lines)


class TEEDetector:
    """
    Unified TEE detector for all supported backends.
    
    Usage:
        detector = TEEDetector()
        result = detector.detect_all()
        
        # Get best backend
        backend = detector.get_best_backend()
        
        # Get specific backend
        nvidia = detector.get_backend(TEEType.NVIDIA_CC)
    """
    
    # Backend registry
    BACKEND_CLASSES: Dict[TEEType, Type[TEEBackend]] = {
        TEEType.NVIDIA_CC: NvidiaCCBackend,
        TEEType.AMD_SEV: AMDSEVBackend,
        TEEType.AMD_GPU_TEE: AMDGPUTEEBackend,
        TEEType.SOFTWARE: SoftwareTEEBackend,
    }
    
    # Priority order (higher = more secure)
    TEE_PRIORITY = {
        TEEType.NVIDIA_CC: 100,      # Hardware GPU TEE - most secure
        TEEType.AMD_GPU_TEE: 95,      # AMD GPU TEE
        TEEType.AMD_SEV: 80,          # CPU memory encryption
        TEEType.SOFTWARE: 10,         # Software fallback
    }
    
    def __init__(
        self,
        strategy: TEESelectionStrategy = TEESelectionStrategy.MOST_SECURE,
        device_index: int = 0
    ):
        self.strategy = strategy
        self.device_index = device_index
        self._detection_result: Optional[TEEDetectionResult] = None
        self._backends: Dict[TEEType, TEEBackend] = {}
    
    def detect_all(self, force: bool = False) -> TEEDetectionResult:
        """
        Detect all available TEE backends.
        
        Args:
            force: Force re-detection even if cached
            
        Returns:
            TEEDetectionResult with all detected backends
        """
        if self._detection_result and not force:
            return self._detection_result
        
        result = TEEDetectionResult()
        
        # Backends that don't take device_index
        no_device_backends = {TEEType.SOFTWARE, TEEType.AMD_SEV}
        
        # Detect each backend
        for tee_type, backend_class in self.BACKEND_CLASSES.items():
            try:
                if tee_type in no_device_backends:
                    backend = backend_class()
                else:
                    backend = backend_class(self.device_index)
                
                caps = backend.detect()
                result.backends[tee_type] = caps
                self._backends[tee_type] = backend
                
                if caps.available:
                    if tee_type != TEEType.SOFTWARE:
                        result.has_hardware_tee = True
                    
                    if tee_type in (TEEType.NVIDIA_CC, TEEType.AMD_GPU_TEE):
                        result.has_gpu_tee = True
                    elif tee_type == TEEType.AMD_SEV:
                        result.has_cpu_tee = True
                
                logger.info(f"TEE detection {tee_type.value}: {caps.available}")
                
            except Exception as e:
                error_msg = f"Error detecting {tee_type.value}: {e}"
                result.errors.append(error_msg)
                logger.warning(error_msg)
                
                # Add placeholder with required fields
                result.backends[tee_type] = TEECapabilities(
                    tee_type=tee_type,
                    vendor="unknown",
                    device_name="unknown",
                    available=False,
                    error_message=str(e)
                )
        
        # Select recommended backend
        result.recommended = self._select_best(result)
        if result.recommended:
            result.recommended_backend = self._backends.get(result.recommended)
        
        self._detection_result = result
        return result
    
    def _select_best(self, result: TEEDetectionResult) -> Optional[TEEType]:
        """Select best backend based on strategy."""
        available = [
            tee_type for tee_type, caps in result.backends.items()
            if caps.available
        ]
        
        if not available:
            return None
        
        if self.strategy == TEESelectionStrategy.SOFTWARE_ONLY:
            return TEEType.SOFTWARE
        
        if self.strategy == TEESelectionStrategy.NVIDIA_ONLY:
            if TEEType.NVIDIA_CC in available:
                return TEEType.NVIDIA_CC
            return None
        
        if self.strategy == TEESelectionStrategy.AMD_ONLY:
            if TEEType.AMD_GPU_TEE in available:
                return TEEType.AMD_GPU_TEE
            if TEEType.AMD_SEV in available:
                return TEEType.AMD_SEV
            return None
        
        if self.strategy == TEESelectionStrategy.PREFER_GPU:
            for tee_type in [TEEType.NVIDIA_CC, TEEType.AMD_GPU_TEE]:
                if tee_type in available:
                    return tee_type
        
        if self.strategy == TEESelectionStrategy.PREFER_CPU:
            if TEEType.AMD_SEV in available:
                return TEEType.AMD_SEV
        
        # Default: MOST_SECURE - select highest priority
        return max(available, key=lambda t: self.TEE_PRIORITY.get(t, 0))
    
    def get_best_backend(self) -> TEEBackend:
        """
        Get the best available TEE backend.
        
        Returns:
            Best TEEBackend, falls back to software if no hardware TEE
        """
        if not self._detection_result:
            self.detect_all()
        
        if self._detection_result.recommended_backend:
            return self._detection_result.recommended_backend
        
        # Fallback to software
        return self._backends.get(TEEType.SOFTWARE, SoftwareTEEBackend())
    
    def get_backend(self, tee_type: TEEType) -> Optional[TEEBackend]:
        """Get a specific TEE backend."""
        if not self._detection_result:
            self.detect_all()
        
        return self._backends.get(tee_type)
    
    def get_capabilities(self, tee_type: TEEType) -> Optional[TEECapabilities]:
        """Get capabilities for a specific TEE type."""
        if not self._detection_result:
            self.detect_all()
        
        return self._detection_result.backends.get(tee_type)
    
    def is_hardware_tee_available(self) -> bool:
        """Check if any hardware TEE is available."""
        if not self._detection_result:
            self.detect_all()
        return self._detection_result.has_hardware_tee
    
    def is_gpu_tee_available(self) -> bool:
        """Check if GPU TEE is available."""
        if not self._detection_result:
            self.detect_all()
        return self._detection_result.has_gpu_tee


def detect_tee(
    strategy: TEESelectionStrategy = TEESelectionStrategy.MOST_SECURE,
    device_index: int = 0
) -> TEEDetectionResult:
    """
    Convenience function to detect TEE capabilities.
    
    Args:
        strategy: Backend selection strategy
        device_index: GPU device index
        
    Returns:
        TEEDetectionResult
    """
    detector = TEEDetector(strategy, device_index)
    return detector.detect_all()


def get_best_tee(
    strategy: TEESelectionStrategy = TEESelectionStrategy.MOST_SECURE,
    device_index: int = 0
) -> TEEBackend:
    """
    Get the best available TEE backend.
    
    Args:
        strategy: Backend selection strategy
        device_index: GPU device index
        
    Returns:
        Best available TEEBackend
    """
    detector = TEEDetector(strategy, device_index)
    return detector.get_best_backend()


def print_tee_status():
    """Print TEE status to console."""
    detector = TEEDetector()
    result = detector.detect_all()
    print(result.summary())


# CLI interface
if __name__ == "__main__":
    print_tee_status()
