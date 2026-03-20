"""
AMD SEV (Secure Encrypted Virtualization) Backend

CPU-side memory encryption for AMD systems.

AMD SEV provides:
- Main memory encryption (protects against physical attacks)
- VM isolation (SEV-ES, SEV-SNP)
- Attestation for secure boot

Note: This is CPU-side encryption. GPU memory is NOT encrypted
by SEV, but the data path between CPU and GPU can be protected.

Requirements:
- AMD EPYC CPU with SEV support
- Linux kernel with SEV enabled
- KVM with SEV support (for VMs)
"""

import secrets
import os
from typing import Optional
import torch

from .base import TEEBackend, TEECapabilities, TEESession, TEEError, TEEType


class AMDSEVBackend(TEEBackend):
    """
    AMD SEV backend for CPU-side memory encryption.
    
    Provides memory encryption at the CPU/system level.
    GPU memory is protected during CPU-GPU transfers.
    """
    
    def __init__(self):
        super().__init__()
        self._sev_enabled = False
        self._encryption_key: Optional[bytes] = None
    
    @property
    def tee_type(self) -> TEEType:
        return TEEType.AMD_SEV
    
    @property
    def vendor(self) -> str:
        return "amd"
    
    def detect(self) -> TEECapabilities:
        """Detect AMD SEV availability."""
        caps = TEECapabilities(
            tee_type=TEEType.AMD_SEV,
            vendor="amd",
            device_name="CPU",
            available=False
        )
        
        try:
            # Check for AMD CPU
            if not self._is_amd_cpu():
                caps.error_message = "Not an AMD CPU"
                return caps
            
            # Check for SEV support in CPU
            sev_caps = self._check_sev_support()
            
            if not sev_caps["sev"]:
                caps.error_message = "SEV not supported by CPU"
                return caps
            
            # Check if SEV is enabled in kernel
            if not self._check_sev_kernel():
                caps.error_message = "SEV not enabled in kernel"
                return caps
            
            # SEV is available
            caps.available = True
            caps.memory_encryption = True
            caps.encryption_algorithm = "AES-128-XTS"
            caps.remote_attestation = sev_caps.get("sev_snp", False)
            caps.local_attestation = True
            caps.supports_gpu_compute = True  # CPU-side protection
            caps.supports_training = True
            
            # Update device name with CPU model
            caps.device_name = self._get_cpu_model()
            
        except Exception as e:
            caps.error_message = str(e)
        
        self._capabilities = caps
        return caps
    
    def _is_amd_cpu(self) -> bool:
        """Check if running on AMD CPU."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read().lower()
                return "amd" in content or "authenticamd" in content
        except:
            return False
    
    def _get_cpu_model(self) -> str:
        """Get CPU model name."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":")[1].strip()
        except:
            pass
        return "AMD CPU"
    
    def _check_sev_support(self) -> dict:
        """Check SEV capabilities."""
        caps = {
            "sev": False,
            "sev_es": False,
            "sev_snp": False,
        }
        
        try:
            # Check /sys/module/kvm_amd/parameters/sev
            if os.path.exists("/sys/module/kvm_amd/parameters/sev"):
                with open("/sys/module/kvm_amd/parameters/sev", "r") as f:
                    caps["sev"] = f.read().strip() in ("1", "Y", "y")
            
            # Check for SEV-ES
            if os.path.exists("/sys/module/kvm_amd/parameters/sev_es"):
                with open("/sys/module/kvm_amd/parameters/sev_es", "r") as f:
                    caps["sev_es"] = f.read().strip() in ("1", "Y", "y")
            
            # Check for SEV-SNP
            if os.path.exists("/sys/module/kvm_amd/parameters/sev_snp"):
                with open("/sys/module/kvm_amd/parameters/sev_snp", "r") as f:
                    caps["sev_snp"] = f.read().strip() in ("1", "Y", "y")
            
            # Alternative: check CPU flags
            if not caps["sev"]:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read().lower()
                    caps["sev"] = "sev" in content
                    caps["sev_es"] = "sev_es" in content
                    caps["sev_snp"] = "sev_snp" in content
                    
        except Exception:
            pass
        
        return caps
    
    def _check_sev_kernel(self) -> bool:
        """Check if SEV is enabled in kernel."""
        try:
            # Check dmesg for SEV initialization
            import subprocess
            result = subprocess.run(
                ["dmesg"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                if "SEV supported" in result.stdout or "SEV enabled" in result.stdout:
                    return True
            
            # Alternative: check if we're in an SEV VM
            if os.path.exists("/sys/kernel/security/sev"):
                return True
                
        except:
            pass
        
        # If we have the module parameter set, consider it enabled
        if os.path.exists("/sys/module/kvm_amd/parameters/sev"):
            try:
                with open("/sys/module/kvm_amd/parameters/sev", "r") as f:
                    return f.read().strip() in ("1", "Y", "y")
            except:
                pass
        
        return False
    
    def initialize(self) -> TEESession:
        """Initialize AMD SEV session."""
        if not self.is_available():
            raise TEEError(f"AMD SEV not available: {self.capabilities.error_message}")
        
        try:
            # Generate session key
            self._encryption_key = secrets.token_bytes(32)
            
            # Create session
            self._session = TEESession(
                session_id=secrets.token_hex(16),
                tee_type=TEEType.AMD_SEV,
                capabilities=self.capabilities
            )
            
            self._sev_enabled = True
            
            # Generate attestation
            self._session.attestation_report = self.generate_attestation()
            
            return self._session
            
        except Exception as e:
            raise TEEError(f"Failed to initialize AMD SEV: {e}")
    
    def encrypt_tensor(
        self,
        tensor: torch.Tensor,
        context: Optional[str] = None
    ) -> bytes:
        """
        Encrypt tensor with SEV protection.
        
        Note: SEV encrypts main memory automatically.
        This adds application-level encryption for defense in depth.
        """
        if not self._sev_enabled:
            raise TEEError("SEV session not initialized")
        
        from ..zcrypt.encryption import AESCipher, encrypt_tensor as sw_encrypt
        
        # Ensure on CPU (SEV protects CPU memory)
        cpu_tensor = tensor.cpu()
        
        cipher = AESCipher(self._encryption_key)
        encrypted = sw_encrypt(
            cpu_tensor,
            cipher,
            context.encode() if context else None
        )
        
        if self._session:
            self._session.encrypted_bytes += len(encrypted.encrypted_data)
        
        return encrypted.to_bytes()
    
    def decrypt_tensor(
        self,
        encrypted: bytes,
        shape: tuple,
        dtype: torch.dtype,
        device: str,
        context: Optional[str] = None
    ) -> torch.Tensor:
        """Decrypt tensor."""
        if not self._sev_enabled:
            raise TEEError("SEV session not initialized")
        
        from ..zcrypt.encryption import AESCipher, EncryptedTensor, decrypt_tensor as sw_decrypt
        
        cipher = AESCipher(self._encryption_key)
        enc_tensor = EncryptedTensor.from_bytes(encrypted)
        tensor = sw_decrypt(enc_tensor, cipher)
        
        if self._session:
            self._session.decrypted_bytes += len(encrypted)
        
        return tensor.to(device=device, dtype=dtype).reshape(shape)
    
    def generate_attestation(self) -> bytes:
        """Generate AMD SEV attestation report."""
        import json
        import time
        import hashlib
        
        report = {
            "type": "amd_sev",
            "timestamp": time.time(),
            "cpu": self._get_cpu_model(),
            "sev_enabled": self._sev_enabled,
            "session_id": self._session.session_id if self._session else None,
        }
        
        # Add SEV-specific info if available
        sev_caps = self._check_sev_support()
        report["capabilities"] = sev_caps
        
        # Sign report
        report_bytes = json.dumps(report, sort_keys=True).encode()
        signature = hashlib.sha256(
            report_bytes + (self._encryption_key or b'')
        ).digest()
        
        return report_bytes + signature
    
    def verify_attestation(
        self,
        report: bytes,
        expected_measurement: Optional[bytes] = None
    ) -> bool:
        """Verify AMD SEV attestation report."""
        import json
        import hashlib
        
        try:
            report_bytes = report[:-32]
            signature = report[-32:]
            
            expected_sig = hashlib.sha256(
                report_bytes + (self._encryption_key or b'')
            ).digest()
            
            if signature != expected_sig:
                return False
            
            report_data = json.loads(report_bytes)
            
            if report_data.get("type") != "amd_sev":
                return False
            
            if not report_data.get("sev_enabled"):
                return False
            
            return True
            
        except Exception:
            return False
    
    def shutdown(self):
        """Shutdown SEV session."""
        self._sev_enabled = False
        self._encryption_key = None
        
        if self._session:
            self._session.is_active = False
        self._session = None
