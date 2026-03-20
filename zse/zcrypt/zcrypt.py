"""
ZCrypt - Unified Privacy API for ZSE

The main entry point for privacy protection in ZSE.
Provides a unified interface that automatically selects
the best available privacy mechanisms:

1. Software Privacy (zCrypt) - Works on ALL GPUs
   - AES-256-GCM encryption
   - Differential privacy
   - Split learning
   - Secure aggregation
   
2. Hardware Privacy (TEE) - Requires supported hardware
   - NVIDIA Confidential Computing (H100+)
   - AMD SEV (EPYC CPUs)
   - AMD GPU TEE (future MI300X+)

Usage:
    from zse.zcrypt import ZCrypt, PrivacyLevel
    
    # Auto-detect best privacy
    privacy = ZCrypt()
    
    # Explicit privacy level
    privacy = ZCrypt(level=PrivacyLevel.MAXIMUM)
    
    # Encrypt data
    encrypted = privacy.encrypt(tensor)
    decrypted = privacy.decrypt(encrypted)
    
    # Private gradient computation
    private_grads = privacy.private_gradients(grads, batch_size)
"""

import secrets
import logging
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from .config import PrivacyConfig, PrivacyLevel, HardwareCapabilities
from .encryption import (
    AESCipher,
    EncryptedTensor,
    encrypt_tensor,
    decrypt_tensor,
    generate_key,
    derive_key,
)
from .differential_privacy import (
    GaussianMechanism,
    LaplaceMechanism,
    PrivacyAccountant,
    clip_gradients as dp_clip_gradients,
    add_noise as dp_add_noise,
)
from .split_learning import SplitLearningClient, SplitLearningWorker, ActivationEncryptor
from .secure_aggregation import SecureAggregator, FederatedAggregator
from .attestation import WorkerAttestation, AttestationVerifier

from ..tee import (
    TEEBackend,
    TEEDetector,
    TEESelectionStrategy,
    detect_tee,
    get_best_tee,
    TEEType,
)

logger = logging.getLogger(__name__)


@dataclass
class PrivacyMetrics:
    """Metrics tracking privacy guarantees."""
    
    # Encryption stats
    tensors_encrypted: int = 0
    tensors_decrypted: int = 0
    bytes_encrypted: int = 0
    bytes_decrypted: int = 0
    
    # DP stats
    dp_queries: int = 0
    epsilon_spent: float = 0.0
    delta_spent: float = 0.0
    gradients_clipped: int = 0
    
    # TEE stats
    tee_operations: int = 0
    attestations_generated: int = 0
    attestations_verified: int = 0
    
    # Aggregation stats
    aggregations_performed: int = 0
    
    def privacy_budget_remaining(self, epsilon_budget: float) -> float:
        """Calculate remaining privacy budget."""
        return max(0, epsilon_budget - self.epsilon_spent)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "encryption": {
                "tensors_encrypted": self.tensors_encrypted,
                "tensors_decrypted": self.tensors_decrypted,
                "bytes_encrypted": self.bytes_encrypted,
                "bytes_decrypted": self.bytes_decrypted,
            },
            "differential_privacy": {
                "queries": self.dp_queries,
                "epsilon_spent": self.epsilon_spent,
                "delta_spent": self.delta_spent,
                "gradients_clipped": self.gradients_clipped,
            },
            "tee": {
                "operations": self.tee_operations,
                "attestations_generated": self.attestations_generated,
                "attestations_verified": self.attestations_verified,
            },
            "aggregation": {
                "aggregations_performed": self.aggregations_performed,
            },
        }


class ZCrypt:
    """
    Unified Privacy Manager for ZSE.
    
    Automatically selects the best available privacy mechanisms
    based on hardware capabilities and configured privacy level.
    
    Privacy Levels:
        - NONE: No privacy protection (fastest)
        - STANDARD: AES encryption + basic DP
        - ENHANCED: Stronger DP + secure aggregation
        - MAXIMUM: Hardware TEE + all software protections
    
    Args:
        level: Privacy level (auto-detected if not specified)
        config: Optional PrivacyConfig for fine-grained control
        tee_strategy: TEE selection strategy
        device: Default device for operations
        
    Example:
        >>> privacy = ZCrypt(level=PrivacyLevel.ENHANCED)
        >>> 
        >>> # Encrypt model weights
        >>> encrypted = privacy.encrypt(model.state_dict())
        >>> 
        >>> # Privatize gradients
        >>> for batch in dataloader:
        ...     grads = compute_gradients(batch)
        ...     private_grads = privacy.private_gradients(grads, len(batch))
        ...     apply_gradients(private_grads)
    """
    
    def __init__(
        self,
        level: Optional[PrivacyLevel] = None,
        config: Optional[PrivacyConfig] = None,
        tee_strategy: TEESelectionStrategy = TEESelectionStrategy.MOST_SECURE,
        device: str = "cuda",
    ):
        # Detect hardware capabilities
        self.hw_caps = HardwareCapabilities.detect()
        
        # Detect TEE
        self._tee_detector = TEEDetector(tee_strategy)
        self._tee_result = self._tee_detector.detect_all()
        self._tee_backend: Optional[TEEBackend] = None
        
        # Determine privacy level
        if level is None:
            level = self._auto_detect_level()
        self.level = level
        
        # Load or create config
        if config is None:
            config = PrivacyConfig(level=level)
        self.config = config
        
        # Initialize components based on level
        self._cipher: Optional[AESCipher] = None
        self._dp_mechanism: Optional[GaussianMechanism] = None
        self._epsilon_spent: float = 0.0
        self._delta_spent: float = 0.0
        self._dp_steps: int = 0
        self._aggregator: Optional[SecureAggregator] = None
        self._verifier: Optional[AttestationVerifier] = None
        
        self._initialize_components()
        
        # Metrics tracking
        self.metrics = PrivacyMetrics()
        
        # Default device
        self.device = device
        
        logger.info(f"ZCrypt initialized with level={level.name}, TEE={self._tee_result.recommended}")
    
    def _auto_detect_level(self) -> PrivacyLevel:
        """Auto-detect appropriate privacy level."""
        if self._tee_result.has_hardware_tee:
            return PrivacyLevel.MAXIMUM
        elif self.hw_caps.has_gpu:
            return PrivacyLevel.ENHANCED
        else:
            return PrivacyLevel.STANDARD
    
    def _initialize_components(self):
        """Initialize privacy components based on level."""
        # Always initialize encryption for non-NONE levels
        if self.level != PrivacyLevel.NONE:
            key = generate_key()
            self._cipher = AESCipher(key)
        
        # Initialize DP for STANDARD and above
        if self.level in (PrivacyLevel.STANDARD, PrivacyLevel.ENHANCED, PrivacyLevel.MAXIMUM):
            # Use Gaussian mechanism for gradient noise
            self._dp_mechanism = GaussianMechanism(
                sensitivity=self.config.dp_max_grad_norm,
                epsilon=self.config.dp_epsilon,
                delta=self.config.dp_delta,
            )
            # Track privacy budget (simplified accountant)
            self._epsilon_spent = 0.0
            self._delta_spent = 0.0
            self._dp_steps = 0
        
        # Initialize secure aggregation for ENHANCED and above
        if self.level in (PrivacyLevel.ENHANCED, PrivacyLevel.MAXIMUM):
            self._aggregator = SecureAggregator(
                num_workers=self.config.num_shares,
                threshold=self.config.threshold,
            )
        
        # Initialize TEE for MAXIMUM level
        if self.level == PrivacyLevel.MAXIMUM and self._tee_result.has_hardware_tee:
            self._tee_backend = self._tee_detector.get_best_backend()
            if self._tee_backend:
                try:
                    self._tee_backend.initialize()
                    logger.info(f"Hardware TEE initialized: {self._tee_backend.tee_type.value}")
                except Exception as e:
                    logger.warning(f"Failed to initialize TEE: {e}")
                    self._tee_backend = None
        
        # Initialize attestation verifier
        self._verifier = AttestationVerifier()
    
    # =========================================
    # Encryption API
    # =========================================
    
    def encrypt(
        self,
        data: Union[torch.Tensor, Dict[str, torch.Tensor], nn.Module],
        context: Optional[str] = None,
    ) -> Union[bytes, Dict[str, bytes], bytes]:
        """
        Encrypt tensor, state dict, or model.
        
        Args:
            data: Tensor, state dict, or model to encrypt
            context: Optional context for authenticated encryption
            
        Returns:
            Encrypted bytes (or dict of bytes for state dict)
        """
        if self.level == PrivacyLevel.NONE:
            raise ValueError("Encryption disabled at NONE privacy level")
        
        if isinstance(data, nn.Module):
            data = data.state_dict()
        
        if isinstance(data, dict):
            return self._encrypt_state_dict(data, context)
        else:
            return self._encrypt_tensor(data, context)
    
    def _encrypt_tensor(
        self,
        tensor: torch.Tensor,
        context: Optional[str] = None
    ) -> bytes:
        """Encrypt a single tensor."""
        # Use TEE if available for MAXIMUM level
        if self._tee_backend and self._tee_backend.is_available():
            encrypted = self._tee_backend.encrypt_tensor(tensor, context)
            self.metrics.tee_operations += 1
        else:
            # Use software encryption
            context_bytes = context.encode() if context else None
            enc_tensor = encrypt_tensor(tensor, self._cipher, context_bytes)
            encrypted = enc_tensor.to_bytes()
        
        self.metrics.tensors_encrypted += 1
        self.metrics.bytes_encrypted += len(encrypted)
        
        return encrypted
    
    def _encrypt_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        context: Optional[str] = None
    ) -> Dict[str, bytes]:
        """Encrypt a state dict."""
        return {
            key: self._encrypt_tensor(tensor, f"{context}_{key}" if context else key)
            for key, tensor in state_dict.items()
        }
    
    def decrypt(
        self,
        encrypted: Union[bytes, Dict[str, bytes]],
        context: Optional[str] = None,
        shape: Optional[tuple] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted: Encrypted bytes (or dict)
            context: Context used during encryption
            shape: Tensor shape (for TEE decryption)
            dtype: Tensor dtype (for TEE decryption)
            
        Returns:
            Decrypted tensor or state dict
        """
        if self.level == PrivacyLevel.NONE:
            raise ValueError("Decryption disabled at NONE privacy level")
        
        if isinstance(encrypted, dict):
            return self._decrypt_state_dict(encrypted, context)
        else:
            return self._decrypt_tensor(encrypted, context, shape, dtype)
    
    def _decrypt_tensor(
        self,
        encrypted: bytes,
        context: Optional[str] = None,
        shape: Optional[tuple] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Decrypt a single tensor."""
        # Use TEE if available
        if self._tee_backend and self._tee_backend.is_available() and shape and dtype:
            tensor = self._tee_backend.decrypt_tensor(
                encrypted, shape, dtype, self.device, context
            )
            self.metrics.tee_operations += 1
        else:
            # Use software decryption
            enc_tensor = EncryptedTensor.from_bytes(encrypted)
            tensor = decrypt_tensor(enc_tensor, self._cipher)
        
        self.metrics.tensors_decrypted += 1
        self.metrics.bytes_decrypted += len(encrypted)
        
        return tensor
    
    def _decrypt_state_dict(
        self,
        encrypted: Dict[str, bytes],
        context: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Decrypt state dict."""
        return {
            key: self._decrypt_tensor(data, f"{context}_{key}" if context else key)
            for key, data in encrypted.items()
        }
    
    # =========================================
    # Differential Privacy API
    # =========================================
    
    def private_gradients(
        self,
        gradients: Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]],
        batch_size: int,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Apply differential privacy to gradients.
        
        Args:
            gradients: Raw gradients (tensor, dict, or list)
            batch_size: Batch size for DP accounting
            
        Returns:
            Private gradients with same type as input
        """
        if self.level == PrivacyLevel.NONE:
            return gradients
        
        if self._dp_mechanism is None:
            return gradients
        
        def _privatize_single(grad: torch.Tensor) -> torch.Tensor:
            """Clip and add noise to a single gradient."""
            # Clip gradient
            clipped = dp_clip_gradients(grad, self.config.dp_max_grad_norm)
            # Add noise
            noisy = self._dp_mechanism.apply(clipped)
            # Update epsilon tracking (simplified accounting)
            import math
            self._dp_steps += 1
            step_eps = 2 * (1.0 / batch_size) * math.sqrt(2 * math.log(1.25 / self.config.dp_delta)) / self._dp_mechanism.sigma
            self._epsilon_spent += step_eps
            return noisy
        
        if isinstance(gradients, dict):
            self.metrics.dp_queries += len(gradients)
            return {
                key: _privatize_single(grad)
                for key, grad in gradients.items()
            }
        elif isinstance(gradients, list):
            self.metrics.dp_queries += len(gradients)
            return [_privatize_single(g) for g in gradients]
        else:
            result = _privatize_single(gradients)
            self.metrics.dp_queries += 1
            return result
    
    def clip_gradients(
        self,
        parameters: Union[nn.Module, List[torch.Tensor]],
    ) -> float:
        """
        Clip per-sample gradients.
        
        Args:
            parameters: Model or list of parameters
            
        Returns:
            Total gradient norm before clipping
        """
        if self._dp_mechanism is None:
            return 0.0
        
        if isinstance(parameters, nn.Module):
            grads = [p.grad for p in parameters.parameters() if p.grad is not None]
        else:
            grads = [p.grad if hasattr(p, 'grad') and p.grad is not None else p for p in parameters]
        
        if not grads:
            return 0.0
        
        # Compute total norm before clipping
        total_norm = torch.norm(
            torch.stack([torch.norm(g.flatten(), 2) for g in grads]), 2
        ).item()
        
        # Clip
        clipped = dp_clip_gradients(grads, self.config.dp_max_grad_norm)
        
        # Update gradients in-place
        for original, clip in zip(grads, clipped):
            original.copy_(clip)
        
        self.metrics.gradients_clipped += 1
        
        return total_norm
    
    def add_noise(
        self,
        tensor: torch.Tensor,
        sensitivity: float = 1.0,
    ) -> torch.Tensor:
        """
        Add calibrated noise for differential privacy.
        
        Args:
            tensor: Input tensor
            sensitivity: Query sensitivity
            
        Returns:
            Noisy tensor
        """
        if self._dp_mechanism is None:
            return tensor
        
        # Scale noise by sensitivity
        noise_scale = self._dp_mechanism.sigma * sensitivity / self.config.dp_max_grad_norm
        noise = torch.randn_like(tensor) * noise_scale
        return tensor + noise
    
    @property
    def privacy_spent(self) -> Tuple[float, float]:
        """Get privacy budget spent (epsilon, delta)."""
        return (self._epsilon_spent, self._delta_spent)
    
    @property
    def privacy_remaining(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        epsilon, delta = self.privacy_spent
        return (
            max(0, self.config.dp_epsilon - epsilon),
            max(0, self.config.dp_delta - delta),
        )
    
    # =========================================
    # Secure Aggregation API
    # =========================================
    
    def create_aggregator(
        self,
        num_workers: int,
        threshold: int = None,
    ) -> FederatedAggregator:
        """
        Create a federated aggregator for secure gradient aggregation.
        
        Args:
            num_workers: Total number of workers
            threshold: Minimum workers required (default: majority)
            
        Returns:
            FederatedAggregator instance
        """
        if threshold is None:
            threshold = num_workers // 2 + 1
        
        return FederatedAggregator(
            num_workers=num_workers,
            threshold=threshold,
            encryption_key=self._cipher._key if self._cipher else None,
        )
    
    def secure_aggregate(
        self,
        gradients: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Securely aggregate gradients from multiple workers.
        
        Args:
            gradients: List of gradient dicts from workers
            
        Returns:
            Aggregated gradients
        """
        if self._aggregator is None or len(gradients) < 2:
            # Simple average if no aggregator
            keys = gradients[0].keys()
            return {
                key: torch.stack([g[key] for g in gradients]).mean(dim=0)
                for key in keys
            }
        
        aggregated = self._aggregator.aggregate(gradients)
        self.metrics.aggregations_performed += 1
        
        return aggregated
    
    # =========================================
    # Attestation API
    # =========================================
    
    def generate_attestation(
        self,
        worker_id: str,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Generate attestation report for a worker.
        
        Args:
            worker_id: Unique worker identifier
            capabilities: Worker capabilities
            
        Returns:
            Attestation report bytes
        """
        attestation = WorkerAttestation(
            worker_id=worker_id,
            capabilities=capabilities or {},
        )
        
        # Use TEE attestation if available
        if self._tee_backend and self._tee_backend.is_available():
            attestation.tee_report = self._tee_backend.generate_attestation()
        
        report = attestation.generate_report(include_tee=True)
        self.metrics.attestations_generated += 1
        
        return report
    
    def verify_attestation(
        self,
        report: bytes,
        expected_capabilities: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a worker's attestation report.
        
        Args:
            report: Attestation report bytes
            expected_capabilities: Expected worker capabilities
            
        Returns:
            (is_valid, error_message)
        """
        is_valid, error = self._verifier.verify(report, expected_capabilities)
        self.metrics.attestations_verified += 1
        
        return is_valid, error
    
    # =========================================
    # TEE API
    # =========================================
    
    @property
    def has_hardware_tee(self) -> bool:
        """Check if hardware TEE is available."""
        return self._tee_result.has_hardware_tee
    
    @property
    def tee_type(self) -> Optional[TEEType]:
        """Get active TEE type."""
        if self._tee_backend:
            return self._tee_backend.tee_type
        return None
    
    def get_tee_status(self) -> Dict[str, Any]:
        """Get TEE status and capabilities."""
        return {
            "available": self._tee_result.has_hardware_tee,
            "gpu_tee": self._tee_result.has_gpu_tee,
            "cpu_tee": self._tee_result.has_cpu_tee,
            "recommended": self._tee_result.recommended.value if self._tee_result.recommended else None,
            "backends": {
                tee_type.value: {
                    "available": caps.available,
                    "device": caps.device_name,
                    "encryption": caps.encryption_algorithm,
                }
                for tee_type, caps in self._tee_result.backends.items()
            },
        }
    
    # =========================================
    # Split Learning API
    # =========================================
    
    def create_split_client(
        self,
        model: nn.Module,
        cut_layer: int,
    ) -> SplitLearningClient:
        """
        Create a split learning client.
        
        The client keeps embedding layers local.
        
        Args:
            model: Full model
            cut_layer: Layer index to split at
            
        Returns:
            SplitLearningClient
        """
        return SplitLearningClient(
            model=model,
            cut_layer=cut_layer,
            encryptor=ActivationEncryptor(self._cipher) if self._cipher else None,
        )
    
    def create_split_worker(
        self,
        model: nn.Module,
        cut_layer: int,
    ) -> SplitLearningWorker:
        """
        Create a split learning worker.
        
        The worker handles computation on encrypted activations.
        
        Args:
            model: Full model
            cut_layer: Start layer for worker
            
        Returns:
            SplitLearningWorker
        """
        return SplitLearningWorker(
            model=model,
            start_layer=cut_layer,
            encryptor=ActivationEncryptor(self._cipher) if self._cipher else None,
        )
    
    # =========================================
    # Utility Methods
    # =========================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get privacy metrics."""
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset privacy metrics."""
        self.metrics = PrivacyMetrics()
    
    def get_status(self) -> Dict[str, Any]:
        """Get full status report."""
        return {
            "privacy_level": self.level.name,
            "hardware": {
                "has_gpu": self.hw_caps.has_gpu,
                "gpu_name": self.hw_caps.gpu_name,
                "cuda_version": self.hw_caps.cuda_version,
            },
            "tee": self.get_tee_status(),
            "config": {
                "epsilon": self.config.dp_epsilon,
                "delta": self.config.dp_delta,
                "noise_multiplier": self.config.dp_noise_multiplier,
                "max_grad_norm": self.config.dp_max_grad_norm,
            },
            "privacy_spent": {
                "epsilon": self.privacy_spent[0],
                "delta": self.privacy_spent[1],
            },
            "metrics": self.get_metrics(),
        }
    
    def __repr__(self) -> str:
        tee_info = f", TEE={self.tee_type.value}" if self.tee_type else ""
        return f"ZCrypt(level={self.level.name}{tee_info})"
    
    def shutdown(self):
        """Shutdown and cleanup resources."""
        if self._tee_backend:
            self._tee_backend.shutdown()
        
        # Clear keys
        if self._cipher:
            self._cipher._key = b'\x00' * 32  # Overwrite key
        
        logger.info("ZCrypt shutdown complete")


# Convenience function
def create_privacy_manager(
    level: Optional[PrivacyLevel] = None,
    **kwargs
) -> ZCrypt:
    """
    Create a privacy manager with optimal settings.
    
    Args:
        level: Privacy level (auto-detected if None)
        **kwargs: Additional arguments for ZCrypt
        
    Returns:
        Configured ZCrypt instance
    """
    return ZCrypt(level=level, **kwargs)
