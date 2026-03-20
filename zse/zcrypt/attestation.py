"""
Worker Attestation for zCrypt

Verifies that workers in ZMesh are running trusted code.
Prevents malicious workers from tampering with training.

Attestation levels:
1. Software: Hash verification of worker binary
2. TPM: Hardware-backed attestation (CPU)
3. TEE: Enclave attestation (GPU - H100+)

Protocol:
1. Worker generates attestation report
2. Report includes code hash, hardware info, nonce
3. Coordinator verifies report against known-good values
4. Verified workers are marked as trusted
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum
import json
import base64


class AttestationLevel(Enum):
    """Attestation security levels."""
    NONE = "none"
    SOFTWARE = "software"  # Code hash verification
    TPM = "tpm"            # Hardware TPM attestation
    TEE = "tee"            # TEE/Enclave attestation


@dataclass
class AttestationReport:
    """Attestation report from a worker."""
    
    worker_id: str
    level: AttestationLevel
    timestamp: float
    nonce: bytes
    
    # Software attestation
    code_hash: Optional[str] = None
    python_version: Optional[str] = None
    torch_version: Optional[str] = None
    
    # Hardware info
    gpu_vendor: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_uuid: Optional[str] = None
    
    # TEE-specific
    tee_type: Optional[str] = None
    enclave_measurement: Optional[str] = None
    platform_info: Optional[Dict[str, Any]] = None
    
    # Signature
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "worker_id": self.worker_id,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "nonce": base64.b64encode(self.nonce).decode(),
            "code_hash": self.code_hash,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "gpu_vendor": self.gpu_vendor,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "tee_type": self.tee_type,
            "enclave_measurement": self.enclave_measurement,
            "platform_info": self.platform_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttestationReport":
        """Create from dictionary."""
        return cls(
            worker_id=data["worker_id"],
            level=AttestationLevel(data["level"]),
            timestamp=data["timestamp"],
            nonce=base64.b64decode(data["nonce"]),
            code_hash=data.get("code_hash"),
            python_version=data.get("python_version"),
            torch_version=data.get("torch_version"),
            gpu_vendor=data.get("gpu_vendor"),
            gpu_name=data.get("gpu_name"),
            gpu_uuid=data.get("gpu_uuid"),
            tee_type=data.get("tee_type"),
            enclave_measurement=data.get("enclave_measurement"),
            platform_info=data.get("platform_info"),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "AttestationReport":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class WorkerAttestation:
    """
    Worker-side attestation generator.
    
    Collects system information and generates attestation
    reports for verification.
    """
    
    worker_id: str
    signing_key: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    
    def generate_report(
        self,
        nonce: bytes,
        code_paths: Optional[List[str]] = None
    ) -> AttestationReport:
        """
        Generate attestation report.
        
        Args:
            nonce: Challenge nonce from verifier
            code_paths: Paths to hash for code verification
        
        Returns:
            Attestation report
        """
        import sys
        
        # Determine attestation level
        level = self._detect_attestation_level()
        
        # Collect code hash
        code_hash = None
        if code_paths:
            code_hash = self._compute_code_hash(code_paths)
        
        # Collect GPU info
        gpu_info = self._collect_gpu_info()
        
        # Create report
        report = AttestationReport(
            worker_id=self.worker_id,
            level=level,
            timestamp=time.time(),
            nonce=nonce,
            code_hash=code_hash,
            python_version=sys.version.split()[0],
            torch_version=self._get_torch_version(),
            gpu_vendor=gpu_info.get("vendor"),
            gpu_name=gpu_info.get("name"),
            gpu_uuid=gpu_info.get("uuid"),
            tee_type=gpu_info.get("tee_type"),
            enclave_measurement=gpu_info.get("enclave_measurement"),
        )
        
        # Sign report
        report.signature = self._sign_report(report)
        
        return report
    
    def _detect_attestation_level(self) -> AttestationLevel:
        """Detect available attestation level."""
        # Check for TEE
        gpu_info = self._collect_gpu_info()
        
        if gpu_info.get("tee_available"):
            return AttestationLevel.TEE
        
        # Check for TPM
        if self._has_tpm():
            return AttestationLevel.TPM
        
        return AttestationLevel.SOFTWARE
    
    def _compute_code_hash(self, paths: List[str]) -> str:
        """Compute hash of code files."""
        import os
        
        hasher = hashlib.sha256()
        
        for path in sorted(paths):
            if os.path.isfile(path):
                with open(path, 'rb') as f:
                    hasher.update(f.read())
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for filename in sorted(files):
                        if filename.endswith('.py'):
                            filepath = os.path.join(root, filename)
                            with open(filepath, 'rb') as f:
                                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def _collect_gpu_info(self) -> Dict[str, Any]:
        """Collect GPU information."""
        info = {}
        
        try:
            import torch
            if torch.cuda.is_available():
                info["vendor"] = "nvidia"
                info["name"] = torch.cuda.get_device_name(0)
                
                # Get UUID if available
                try:
                    import subprocess
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        info["uuid"] = result.stdout.strip().split('\n')[0]
                except:
                    pass
                
                # Check for Confidential Computing (H100+)
                cc = torch.cuda.get_device_capability(0)
                if cc[0] >= 9:  # Hopper architecture
                    info["tee_available"] = True
                    info["tee_type"] = "nvidia_cc"
                    info["enclave_measurement"] = self._get_nvidia_cc_measurement()
        except:
            pass
        
        # Check for AMD
        if not info:
            try:
                import os
                if os.path.exists("/opt/rocm"):
                    info["vendor"] = "amd"
                    # ROCm detection
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["rocm-smi", "--showproductname"],
                            capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                if 'GPU' in line:
                                    info["name"] = line.split(':')[-1].strip()
                                    break
                    except:
                        info["name"] = "AMD GPU"
            except:
                pass
        
        return info
    
    def _get_nvidia_cc_measurement(self) -> Optional[str]:
        """Get NVIDIA Confidential Computing measurement."""
        # This would interface with NVIDIA CC SDK in production
        # For now, return placeholder
        try:
            import torch
            device_props = torch.cuda.get_device_properties(0)
            # Create deterministic measurement from device properties
            measurement = hashlib.sha256(
                f"{device_props.name}:{device_props.total_memory}".encode()
            ).hexdigest()
            return measurement
        except:
            return None
    
    def _get_torch_version(self) -> Optional[str]:
        """Get PyTorch version."""
        try:
            import torch
            return torch.__version__
        except:
            return None
    
    def _has_tpm(self) -> bool:
        """Check if TPM is available."""
        import os
        # Linux TPM device
        return os.path.exists("/dev/tpm0") or os.path.exists("/dev/tpmrm0")
    
    def _sign_report(self, report: AttestationReport) -> bytes:
        """Sign attestation report."""
        report_bytes = report.to_json().encode()
        return hmac.new(self.signing_key, report_bytes, hashlib.sha256).digest()


class AttestationVerifier:
    """
    Coordinator-side attestation verifier.
    
    Verifies worker attestation reports against known-good values.
    """
    
    def __init__(
        self,
        trusted_code_hashes: Optional[List[str]] = None,
        trusted_gpu_uuids: Optional[List[str]] = None,
        max_clock_skew: float = 300.0,  # 5 minutes
        require_tee: bool = False,
    ):
        """
        Initialize verifier.
        
        Args:
            trusted_code_hashes: List of trusted code hashes
            trusted_gpu_uuids: List of trusted GPU UUIDs (optional)
            max_clock_skew: Maximum allowed timestamp difference
            require_tee: Require TEE attestation
        """
        self.trusted_code_hashes = set(trusted_code_hashes or [])
        self.trusted_gpu_uuids = set(trusted_gpu_uuids or [])
        self.max_clock_skew = max_clock_skew
        self.require_tee = require_tee
        
        # Active challenges
        self._challenges: Dict[str, bytes] = {}
        
        # Verified workers
        self._verified: Dict[str, AttestationReport] = {}
    
    def generate_challenge(self, worker_id: str) -> bytes:
        """
        Generate attestation challenge for worker.
        
        Args:
            worker_id: Worker to challenge
        
        Returns:
            Challenge nonce
        """
        nonce = secrets.token_bytes(32)
        self._challenges[worker_id] = nonce
        return nonce
    
    def verify(
        self,
        report: AttestationReport,
        signing_key: Optional[bytes] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verify attestation report.
        
        Args:
            report: Attestation report to verify
            signing_key: Worker's signing key (for signature verification)
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check challenge nonce
        expected_nonce = self._challenges.get(report.worker_id)
        if expected_nonce is None:
            issues.append("No active challenge for worker")
        elif report.nonce != expected_nonce:
            issues.append("Nonce mismatch")
        
        # Check timestamp
        current_time = time.time()
        if abs(current_time - report.timestamp) > self.max_clock_skew:
            issues.append(f"Timestamp too old (skew: {abs(current_time - report.timestamp):.1f}s)")
        
        # Check TEE requirement
        if self.require_tee and report.level != AttestationLevel.TEE:
            issues.append(f"TEE required but got {report.level.value}")
        
        # Check code hash
        if self.trusted_code_hashes:
            if report.code_hash not in self.trusted_code_hashes:
                issues.append(f"Untrusted code hash: {report.code_hash}")
        
        # Check GPU UUID (if enforced)
        if self.trusted_gpu_uuids:
            if report.gpu_uuid and report.gpu_uuid not in self.trusted_gpu_uuids:
                issues.append(f"Untrusted GPU: {report.gpu_uuid}")
        
        # Verify signature
        if signing_key and report.signature:
            # Recreate report without signature for verification
            report_copy = AttestationReport(
                worker_id=report.worker_id,
                level=report.level,
                timestamp=report.timestamp,
                nonce=report.nonce,
                code_hash=report.code_hash,
                python_version=report.python_version,
                torch_version=report.torch_version,
                gpu_vendor=report.gpu_vendor,
                gpu_name=report.gpu_name,
                gpu_uuid=report.gpu_uuid,
                tee_type=report.tee_type,
                enclave_measurement=report.enclave_measurement,
                platform_info=report.platform_info,
            )
            expected_sig = hmac.new(
                signing_key,
                report_copy.to_json().encode(),
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(report.signature, expected_sig):
                issues.append("Invalid signature")
        
        # Mark as verified if no issues
        is_valid = len(issues) == 0
        if is_valid:
            self._verified[report.worker_id] = report
            # Clear challenge
            self._challenges.pop(report.worker_id, None)
        
        return is_valid, issues
    
    def is_verified(self, worker_id: str) -> bool:
        """Check if worker is verified."""
        return worker_id in self._verified
    
    def get_verification(self, worker_id: str) -> Optional[AttestationReport]:
        """Get verification report for worker."""
        return self._verified.get(worker_id)
    
    def revoke(self, worker_id: str):
        """Revoke worker verification."""
        self._verified.pop(worker_id, None)
    
    def add_trusted_hash(self, code_hash: str):
        """Add a trusted code hash."""
        self.trusted_code_hashes.add(code_hash)
    
    def remove_trusted_hash(self, code_hash: str):
        """Remove a trusted code hash."""
        self.trusted_code_hashes.discard(code_hash)


def generate_attestation(
    worker_id: str,
    nonce: bytes,
    code_paths: Optional[List[str]] = None
) -> AttestationReport:
    """
    Convenience function to generate attestation report.
    
    Args:
        worker_id: Worker identifier
        nonce: Challenge nonce from verifier
        code_paths: Paths to include in code hash
    
    Returns:
        Attestation report
    """
    worker = WorkerAttestation(worker_id)
    return worker.generate_report(nonce, code_paths)


def verify_attestation(
    report: AttestationReport,
    verifier: AttestationVerifier,
    signing_key: Optional[bytes] = None
) -> Tuple[bool, List[str]]:
    """
    Convenience function to verify attestation report.
    
    Args:
        report: Report to verify
        verifier: Attestation verifier
        signing_key: Worker's signing key
    
    Returns:
        (is_valid, issues)
    """
    return verifier.verify(report, signing_key)
