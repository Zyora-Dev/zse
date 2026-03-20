"""
Base TEE Interface

Abstract interface for Trusted Execution Environment backends.
All vendor-specific implementations inherit from this.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
import torch


class TEEError(Exception):
    """TEE-related error."""
    pass


class TEEType(Enum):
    """Types of TEE implementations."""
    NONE = "none"
    SOFTWARE = "software"        # Software fallback (no hardware TEE)
    NVIDIA_CC = "nvidia_cc"      # NVIDIA Confidential Computing
    AMD_SEV = "amd_sev"          # AMD Secure Encrypted Virtualization
    AMD_SME = "amd_sme"          # AMD Secure Memory Encryption
    AMD_GPU_TEE = "amd_gpu_tee"  # AMD GPU TEE (future)
    INTEL_SGX = "intel_sgx"      # Intel SGX (CPU)
    INTEL_TDX = "intel_tdx"      # Intel TDX (VM)


@dataclass
class TEECapabilities:
    """Capabilities of a TEE implementation."""
    
    tee_type: TEEType
    vendor: str
    device_name: str
    
    # Memory encryption
    memory_encryption: bool = False
    encryption_algorithm: Optional[str] = None
    
    # Attestation
    remote_attestation: bool = False
    local_attestation: bool = False
    
    # Capabilities
    max_enclave_size: Optional[int] = None  # bytes
    supports_gpu_compute: bool = False
    supports_training: bool = False
    
    # Status
    available: bool = False
    error_message: Optional[str] = None
    
    def __str__(self) -> str:
        status = "available" if self.available else "not available"
        return (
            f"TEE: {self.tee_type.value} ({status})\n"
            f"  Vendor: {self.vendor}\n"
            f"  Device: {self.device_name}\n"
            f"  Memory Encryption: {self.memory_encryption}\n"
            f"  Attestation: remote={self.remote_attestation}, local={self.local_attestation}\n"
            f"  GPU Compute: {self.supports_gpu_compute}"
        )


@dataclass
class TEESession:
    """Active TEE session."""
    
    session_id: str
    tee_type: TEEType
    capabilities: TEECapabilities
    
    # Session state
    is_active: bool = True
    attestation_report: Optional[bytes] = None
    
    # Metrics
    encrypted_bytes: int = 0
    decrypted_bytes: int = 0


class TEEBackend(ABC):
    """
    Abstract base class for TEE backends.
    
    Each vendor-specific implementation must implement these methods.
    """
    
    def __init__(self):
        self._session: Optional[TEESession] = None
        self._capabilities: Optional[TEECapabilities] = None
    
    @property
    @abstractmethod
    def tee_type(self) -> TEEType:
        """Return the TEE type this backend implements."""
        pass
    
    @property
    @abstractmethod
    def vendor(self) -> str:
        """Return the vendor name."""
        pass
    
    @abstractmethod
    def detect(self) -> TEECapabilities:
        """
        Detect TEE availability and capabilities.
        
        Returns:
            TEECapabilities with availability info
        """
        pass
    
    @abstractmethod
    def initialize(self) -> TEESession:
        """
        Initialize TEE session.
        
        Returns:
            Active TEE session
        
        Raises:
            TEEError: If initialization fails
        """
        pass
    
    @abstractmethod
    def encrypt_tensor(
        self,
        tensor: torch.Tensor,
        context: Optional[str] = None
    ) -> bytes:
        """
        Encrypt tensor within TEE.
        
        Args:
            tensor: Tensor to encrypt
            context: Optional context for key derivation
        
        Returns:
            Encrypted tensor bytes
        """
        pass
    
    @abstractmethod
    def decrypt_tensor(
        self,
        encrypted: bytes,
        shape: tuple,
        dtype: torch.dtype,
        device: str,
        context: Optional[str] = None
    ) -> torch.Tensor:
        """
        Decrypt tensor within TEE.
        
        Args:
            encrypted: Encrypted bytes
            shape: Original tensor shape
            dtype: Original tensor dtype
            device: Target device
            context: Optional context for key derivation
        
        Returns:
            Decrypted tensor
        """
        pass
    
    @abstractmethod
    def generate_attestation(self) -> bytes:
        """
        Generate attestation report.
        
        Returns:
            Attestation report bytes
        """
        pass
    
    @abstractmethod
    def verify_attestation(
        self,
        report: bytes,
        expected_measurement: Optional[bytes] = None
    ) -> bool:
        """
        Verify attestation report.
        
        Args:
            report: Attestation report bytes
            expected_measurement: Expected enclave measurement
        
        Returns:
            True if verification passes
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """Shutdown TEE session and cleanup."""
        pass
    
    # Optional methods with default implementations
    
    def is_available(self) -> bool:
        """Check if TEE is currently available."""
        if self._capabilities is None:
            self._capabilities = self.detect()
        return self._capabilities.available
    
    @property
    def capabilities(self) -> TEECapabilities:
        """Get cached capabilities."""
        if self._capabilities is None:
            self._capabilities = self.detect()
        return self._capabilities
    
    @property
    def session(self) -> Optional[TEESession]:
        """Get current session."""
        return self._session
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False


class SoftwareTEEBackend(TEEBackend):
    """
    Software fallback when no hardware TEE is available.
    
    Provides the same API but uses software encryption.
    This is NOT secure against a malicious host but provides
    defense against accidental data exposure.
    """
    
    def __init__(self):
        super().__init__()
        self._key: Optional[bytes] = None
    
    @property
    def tee_type(self) -> TEEType:
        return TEEType.NONE
    
    @property
    def vendor(self) -> str:
        return "software"
    
    def detect(self) -> TEECapabilities:
        return TEECapabilities(
            tee_type=TEEType.NONE,
            vendor="software",
            device_name="CPU",
            memory_encryption=False,
            remote_attestation=False,
            local_attestation=False,
            supports_gpu_compute=True,
            supports_training=True,
            available=True,  # Always available as fallback
            error_message="Software fallback - no hardware TEE"
        )
    
    def initialize(self) -> TEESession:
        import secrets
        
        self._key = secrets.token_bytes(32)
        self._session = TEESession(
            session_id=secrets.token_hex(16),
            tee_type=TEEType.NONE,
            capabilities=self.detect()
        )
        return self._session
    
    def encrypt_tensor(
        self,
        tensor: torch.Tensor,
        context: Optional[str] = None
    ) -> bytes:
        from ..zcrypt.encryption import AESCipher, encrypt_tensor as crypto_encrypt
        
        if self._key is None:
            raise TEEError("Session not initialized")
        
        cipher = AESCipher(self._key)
        encrypted = crypto_encrypt(tensor, cipher, context.encode() if context else None)
        
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
        from ..zcrypt.encryption import AESCipher, EncryptedTensor, decrypt_tensor as crypto_decrypt
        
        if self._key is None:
            raise TEEError("Session not initialized")
        
        cipher = AESCipher(self._key)
        enc_tensor = EncryptedTensor.from_bytes(encrypted)
        tensor = crypto_decrypt(enc_tensor, cipher)
        
        if self._session:
            self._session.decrypted_bytes += len(encrypted)
        
        return tensor.to(device=device, dtype=dtype).reshape(shape)
    
    def generate_attestation(self) -> bytes:
        """Software attestation (not hardware-backed)."""
        import json
        import time
        import hashlib
        
        report = {
            "type": "software",
            "timestamp": time.time(),
            "session_id": self._session.session_id if self._session else None,
            "warning": "Software attestation - not hardware backed"
        }
        
        report_bytes = json.dumps(report).encode()
        signature = hashlib.sha256(report_bytes + (self._key or b'')).digest()
        
        return report_bytes + signature
    
    def verify_attestation(
        self,
        report: bytes,
        expected_measurement: Optional[bytes] = None
    ) -> bool:
        """Verify software attestation."""
        import json
        import hashlib
        
        try:
            report_bytes = report[:-32]
            signature = report[-32:]
            
            expected_sig = hashlib.sha256(report_bytes + (self._key or b'')).digest()
            
            return signature == expected_sig
        except:
            return False
    
    def shutdown(self):
        """Cleanup session."""
        self._key = None
        if self._session:
            self._session.is_active = False
        self._session = None
