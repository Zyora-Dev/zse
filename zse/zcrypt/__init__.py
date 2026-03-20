"""
zCrypt - Universal Privacy Layer for ZMesh GPU Farming

Software-based privacy protection that works on ALL GPUs (NVIDIA + AMD).
Provides defense-in-depth through multiple cryptographic techniques.

Layers:
1. Transport Security: TLS 1.3 + mTLS for all communications
2. Data Encryption: AES-256-GCM for data at rest and in transit
3. Split Learning: Client keeps sensitive embeddings, worker sees only activations
4. Differential Privacy: Calibrated noise prevents model inversion attacks
5. Secure Aggregation: Multi-worker gradient combination hides individual contributions

Usage:
    from zse.zcrypt import ZCrypt, PrivacyConfig
    
    # Initialize with automatic hardware detection
    zcrypt = ZCrypt()
    
    # Encrypt training data
    encrypted = zcrypt.encrypt(training_data)
    
    # Protect gradients with differential privacy
    safe_grads = zcrypt.protect_gradients(gradients, epsilon=1.0)
    
    # Secure aggregation across workers
    combined = zcrypt.secure_aggregate([grad1, grad2, grad3])

Author: ZSE Team
License: Apache 2.0
"""

from .config import PrivacyConfig, PrivacyLevel
from .encryption import (
    AESCipher,
    encrypt_tensor,
    decrypt_tensor,
    generate_key,
    derive_key,
)
from .split_learning import (
    SplitLearningClient,
    SplitLearningWorker,
    ActivationEncryptor,
)
from .differential_privacy import (
    DPEngine,
    GaussianMechanism,
    LaplaceMechanism,
    clip_gradients,
    add_noise,
    compute_privacy_budget,
)
from .secure_aggregation import (
    SecureAggregator,
    SecretSharing,
    MaskedGradients,
)
from .attestation import (
    WorkerAttestation,
    AttestationVerifier,
    generate_attestation,
    verify_attestation,
)
from .zcrypt import ZCrypt

__all__ = [
    # Core
    "ZCrypt",
    "PrivacyConfig",
    "PrivacyLevel",
    # Encryption
    "AESCipher",
    "encrypt_tensor",
    "decrypt_tensor",
    "generate_key",
    "derive_key",
    # Split Learning
    "SplitLearningClient",
    "SplitLearningWorker",
    "ActivationEncryptor",
    # Differential Privacy
    "DPEngine",
    "GaussianMechanism",
    "LaplaceMechanism",
    "clip_gradients",
    "add_noise",
    "compute_privacy_budget",
    # Secure Aggregation
    "SecureAggregator",
    "SecretSharing",
    "MaskedGradients",
    # Attestation
    "WorkerAttestation",
    "AttestationVerifier",
    "generate_attestation",
    "verify_attestation",
]
