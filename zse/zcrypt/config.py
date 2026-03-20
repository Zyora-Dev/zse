"""
Privacy Configuration for zCrypt

Defines privacy levels, configurations, and hardware detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import os


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    
    NONE = "none"
    """No privacy protection (not recommended for production)."""
    
    STANDARD = "standard"
    """zCrypt software protection (works on all GPUs)."""
    
    ENHANCED = "enhanced"
    """zCrypt + additional hardening (reduced performance)."""
    
    MAXIMUM = "maximum"
    """zCrypt + Hardware TEE (requires H100/MI300X)."""


class GPUVendor(Enum):
    """GPU vendor detection."""
    NVIDIA = "nvidia"
    AMD = "amd"
    UNKNOWN = "unknown"


@dataclass
class PrivacyConfig:
    """
    Configuration for zCrypt privacy layer.
    
    Attributes:
        level: Privacy protection level
        encryption_key: Optional pre-shared key (generated if not provided)
        dp_epsilon: Differential privacy epsilon (lower = more private)
        dp_delta: Differential privacy delta
        dp_max_grad_norm: Gradient clipping threshold for DP
        split_cut_layer: Layer index to cut for split learning
        num_shares: Number of shares for secret sharing
        attestation_enabled: Enable worker attestation
        tee_required: Require hardware TEE (fails if unavailable)
    """
    
    # Core settings
    level: PrivacyLevel = PrivacyLevel.STANDARD
    encryption_key: Optional[bytes] = None
    
    # Differential Privacy settings
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: Optional[float] = None  # Auto-computed if None
    
    # Split Learning settings
    split_learning_enabled: bool = True
    split_cut_layer: int = 1  # Cut after embedding layer
    activation_encryption: bool = True
    
    # Secure Aggregation settings
    secure_aggregation_enabled: bool = True
    num_shares: int = 3  # For Shamir's secret sharing
    threshold: int = 2  # Minimum shares to reconstruct
    
    # Attestation settings
    attestation_enabled: bool = True
    attestation_timeout: int = 30  # seconds
    
    # Hardware TEE settings
    tee_required: bool = False
    tee_fallback_to_zcrypt: bool = True
    
    # Performance tuning
    async_encryption: bool = True
    batch_size: int = 1024
    num_workers: int = 4
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dp_epsilon <= 0:
            raise ValueError(f"dp_epsilon must be positive, got {self.dp_epsilon}")
        if self.dp_delta <= 0 or self.dp_delta >= 1:
            raise ValueError(f"dp_delta must be in (0, 1), got {self.dp_delta}")
        if self.threshold > self.num_shares:
            raise ValueError(f"threshold ({self.threshold}) > num_shares ({self.num_shares})")
        if self.level == PrivacyLevel.MAXIMUM and not self.tee_required:
            self.tee_required = True
    
    @classmethod
    def for_level(cls, level: PrivacyLevel) -> "PrivacyConfig":
        """Create config preset for a privacy level."""
        if level == PrivacyLevel.NONE:
            return cls(
                level=level,
                dp_enabled=False,
                split_learning_enabled=False,
                secure_aggregation_enabled=False,
                attestation_enabled=False,
            )
        elif level == PrivacyLevel.STANDARD:
            return cls(
                level=level,
                dp_epsilon=1.0,
                dp_max_grad_norm=1.0,
            )
        elif level == PrivacyLevel.ENHANCED:
            return cls(
                level=level,
                dp_epsilon=0.5,  # Stricter privacy
                dp_max_grad_norm=0.5,
                activation_encryption=True,
                num_shares=5,
                threshold=3,
            )
        elif level == PrivacyLevel.MAXIMUM:
            return cls(
                level=level,
                dp_epsilon=0.1,  # Very strict
                dp_max_grad_norm=0.3,
                activation_encryption=True,
                num_shares=7,
                threshold=4,
                tee_required=True,
            )
        return cls(level=level)


@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities for privacy."""
    
    vendor: GPUVendor = GPUVendor.UNKNOWN
    gpu_name: str = "unknown"
    tee_available: bool = False
    tee_type: Optional[str] = None
    compute_capability: Optional[tuple] = None
    driver_version: Optional[str] = None
    
    # Specific TEE features
    nvidia_cc: bool = False  # NVIDIA Confidential Computing
    amd_sev: bool = False    # AMD SEV (CPU-side)
    amd_sme: bool = False    # AMD SME (memory encryption)
    
    @classmethod
    def detect(cls) -> "HardwareCapabilities":
        """Auto-detect hardware capabilities."""
        caps = cls()
        
        # Try NVIDIA first
        try:
            import torch
            if torch.cuda.is_available():
                caps.vendor = GPUVendor.NVIDIA
                caps.gpu_name = torch.cuda.get_device_name(0)
                caps.compute_capability = torch.cuda.get_device_capability(0)
                
                # Check for Confidential Computing (H100+)
                # H100 has compute capability 9.0+
                if caps.compute_capability and caps.compute_capability[0] >= 9:
                    caps.tee_available = True
                    caps.tee_type = "nvidia_cc"
                    caps.nvidia_cc = True
                
                # Get driver version
                caps.driver_version = torch.version.cuda
                
        except ImportError:
            pass
        
        # Try AMD ROCm
        if caps.vendor == GPUVendor.UNKNOWN:
            try:
                # Check for ROCm
                if os.path.exists("/opt/rocm"):
                    import torch
                    if hasattr(torch, 'hip') or 'rocm' in str(torch.__config__).lower():
                        caps.vendor = GPUVendor.AMD
                        
                        # Try to get GPU name
                        try:
                            import subprocess
                            result = subprocess.run(
                                ["rocm-smi", "--showproductname"],
                                capture_output=True, text=True
                            )
                            if result.returncode == 0:
                                for line in result.stdout.split('\n'):
                                    if 'GPU' in line:
                                        caps.gpu_name = line.split(':')[-1].strip()
                                        break
                        except:
                            caps.gpu_name = "AMD GPU"
                        
                        # Check for MI300X (future TEE support)
                        if "MI300" in caps.gpu_name or "MI350" in caps.gpu_name:
                            # MI300X may have TEE in future firmware
                            caps.tee_available = False  # Not yet available
                            caps.tee_type = "amd_gpu_tee"
                        
                        # Check for AMD SEV (CPU-side encryption)
                        if os.path.exists("/sys/module/kvm_amd/parameters/sev"):
                            caps.amd_sev = True
                            
            except Exception:
                pass
        
        return caps
    
    def supports_tee(self) -> bool:
        """Check if hardware supports any TEE."""
        return self.tee_available or self.amd_sev
    
    def get_best_tee(self) -> Optional[str]:
        """Get the best available TEE type."""
        if self.nvidia_cc:
            return "nvidia_cc"
        if self.amd_sev:
            return "amd_sev"
        return None
    
    def __str__(self) -> str:
        return (
            f"HardwareCapabilities(\n"
            f"  vendor={self.vendor.value},\n"
            f"  gpu={self.gpu_name},\n"
            f"  tee_available={self.tee_available},\n"
            f"  tee_type={self.tee_type}\n"
            f")"
        )
