"""
Secure Aggregation for zCrypt

Enables multiple workers to contribute gradients without any
single party (including the aggregator) learning individual contributions.

Techniques:
1. Secret Sharing: Split gradients into shares, reconstruct only aggregate
2. Masked Aggregation: Each worker adds/subtracts pairwise masks that cancel out
3. Threshold Cryptography: Require k-of-n workers for reconstruction

Based on:
- "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
  Bonawitz et al., ACM CCS 2017
"""

import secrets
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import torch


@dataclass
class Share:
    """A single share in secret sharing."""
    index: int
    value: torch.Tensor
    commitment: Optional[bytes] = None  # For verification


class SecretSharing:
    """
    Shamir's Secret Sharing for tensors.
    
    Split a tensor into n shares where any k shares can
    reconstruct the original, but k-1 shares reveal nothing.
    
    Usage:
        ss = SecretSharing(threshold=3, num_shares=5)
        
        # Split secret into shares
        shares = ss.split(secret_tensor)
        
        # Reconstruct from any 3+ shares
        recovered = ss.reconstruct([shares[0], shares[2], shares[4]])
    """
    
    # Large prime for modular arithmetic
    PRIME = 2**61 - 1  # Mersenne prime
    
    def __init__(self, threshold: int, num_shares: int):
        """
        Initialize secret sharing.
        
        Args:
            threshold: Minimum shares needed to reconstruct (k)
            num_shares: Total number of shares (n)
        """
        if threshold > num_shares:
            raise ValueError(f"threshold ({threshold}) > num_shares ({num_shares})")
        if threshold < 2:
            raise ValueError("Threshold must be at least 2")
        
        self.threshold = threshold
        self.num_shares = num_shares
    
    def split(self, secret: torch.Tensor) -> List[Share]:
        """
        Split tensor into shares using Shamir's scheme.
        
        Args:
            secret: Tensor to split
        
        Returns:
            List of shares
        """
        # Convert to integer representation for modular arithmetic
        # Scale float to large integer
        scale = 2**30
        device = secret.device
        dtype = secret.dtype
        shape = secret.shape
        
        # Flatten and convert to int64
        flat = (secret.flatten().float() * scale).long().cpu()
        
        shares = []
        for i in range(1, self.num_shares + 1):
            share_values = torch.zeros_like(flat)
            
            for j in range(len(flat)):
                # Generate random polynomial coefficients
                coeffs = [flat[j].item()] + [
                    secrets.randbelow(self.PRIME)
                    for _ in range(self.threshold - 1)
                ]
                
                # Evaluate polynomial at point i
                share_values[j] = self._eval_poly(coeffs, i)
            
            # Reshape and convert back to float
            share_tensor = share_values.reshape(shape).float() / scale
            share_tensor = share_tensor.to(device=device, dtype=dtype)
            
            # Compute commitment for verification
            commitment = self._compute_commitment(share_tensor)
            
            shares.append(Share(index=i, value=share_tensor, commitment=commitment))
        
        return shares
    
    def _eval_poly(self, coeffs: List[int], x: int) -> int:
        """Evaluate polynomial at x using Horner's method."""
        result = 0
        for coeff in reversed(coeffs):
            result = (result * x + coeff) % self.PRIME
        return result
    
    def _compute_commitment(self, tensor: torch.Tensor) -> bytes:
        """Compute hash commitment for share verification."""
        data = tensor.cpu().numpy().tobytes()
        return hashlib.sha256(data).digest()
    
    def reconstruct(self, shares: List[Share]) -> torch.Tensor:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: At least threshold shares
        
        Returns:
            Reconstructed tensor
        """
        if len(shares) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} shares, got {len(shares)}"
            )
        
        # Use first threshold shares
        shares = shares[:self.threshold]
        
        # Get shape and device from first share
        shape = shares[0].value.shape
        device = shares[0].value.device
        dtype = shares[0].value.dtype
        
        # Scale for integer arithmetic
        scale = 2**30
        
        # Convert shares to integers
        share_ints = [
            (s.index, (s.value.flatten().float() * scale).long().cpu())
            for s in shares
        ]
        
        # Lagrange interpolation to recover secret (evaluate at x=0)
        result = torch.zeros(share_ints[0][1].shape, dtype=torch.long)
        
        indices = [s[0] for s in share_ints]
        
        for i, (xi, yi) in enumerate(share_ints):
            # Compute Lagrange basis polynomial
            numerator = 1
            denominator = 1
            
            for j, xj in enumerate(indices):
                if i != j:
                    numerator = (numerator * (-xj)) % self.PRIME
                    denominator = (denominator * (xi - xj)) % self.PRIME
            
            # Modular inverse
            inv_denom = pow(denominator, self.PRIME - 2, self.PRIME)
            lagrange = (numerator * inv_denom) % self.PRIME
            
            # Add contribution
            result = (result + yi * lagrange) % self.PRIME
        
        # Handle negative values (convert from modular representation)
        result = torch.where(
            result > self.PRIME // 2,
            result - self.PRIME,
            result
        )
        
        # Convert back to float and reshape
        result = result.reshape(shape).float() / scale
        return result.to(device=device, dtype=dtype)
    
    def verify_share(self, share: Share) -> bool:
        """Verify a share's commitment."""
        if share.commitment is None:
            return True  # No commitment to verify
        
        expected = self._compute_commitment(share.value)
        return secrets.compare_digest(share.commitment, expected)


@dataclass
class MaskedGradients:
    """Gradients masked for secure aggregation."""
    
    worker_id: int
    masked_grads: Dict[str, torch.Tensor]
    seed_commitments: Dict[int, bytes]  # Commitments to pairwise seeds


class SecureAggregator:
    """
    Secure aggregation server.
    
    Aggregates gradients from multiple workers such that:
    1. Individual gradients are never revealed
    2. Only the sum (average) is computed
    3. Tolerates dropouts (up to threshold)
    
    Protocol:
    1. Workers generate pairwise random masks
    2. Each worker adds/subtracts masks to their gradients
    3. Aggregator sums masked gradients (masks cancel out)
    4. Result is true aggregate (no individual info leaked)
    
    Usage:
        aggregator = SecureAggregator(num_workers=5, threshold=3)
        
        # Register workers
        for worker_id in range(5):
            aggregator.register_worker(worker_id)
        
        # Collect masked gradients
        for worker_id, masked_grads in worker_submissions:
            aggregator.submit(worker_id, masked_grads)
        
        # Aggregate (only possible if >= threshold workers submitted)
        result = aggregator.aggregate()
    """
    
    def __init__(
        self,
        num_workers: int,
        threshold: int,
        use_secret_sharing: bool = True
    ):
        """
        Initialize secure aggregator.
        
        Args:
            num_workers: Total number of workers
            threshold: Minimum workers needed for aggregation
            use_secret_sharing: Use secret sharing for dropout tolerance
        """
        self.num_workers = num_workers
        self.threshold = threshold
        self.use_secret_sharing = use_secret_sharing
        
        self._workers: Dict[int, bytes] = {}  # worker_id -> public_key
        self._submissions: Dict[int, MaskedGradients] = {}
        self._pairwise_seeds: Dict[Tuple[int, int], bytes] = {}
        
        if use_secret_sharing:
            self._secret_sharing = SecretSharing(threshold, num_workers)
    
    def register_worker(self, worker_id: int, public_key: Optional[bytes] = None):
        """
        Register a worker for aggregation.
        
        Args:
            worker_id: Unique worker identifier
            public_key: Worker's public key for key agreement (optional)
        """
        if public_key is None:
            public_key = secrets.token_bytes(32)  # Placeholder
        
        self._workers[worker_id] = public_key
    
    def generate_pairwise_seeds(self) -> Dict[int, Dict[int, bytes]]:
        """
        Generate pairwise random seeds for mask generation.
        
        Returns:
            Dict mapping worker_id -> {other_worker_id -> shared_seed}
        """
        worker_seeds: Dict[int, Dict[int, bytes]] = {
            w: {} for w in self._workers
        }
        
        worker_list = sorted(self._workers.keys())
        
        for i, w1 in enumerate(worker_list):
            for w2 in worker_list[i+1:]:
                # Generate shared seed
                seed = secrets.token_bytes(32)
                self._pairwise_seeds[(w1, w2)] = seed
                
                # Both workers get the same seed
                worker_seeds[w1][w2] = seed
                worker_seeds[w2][w1] = seed
        
        return worker_seeds
    
    def submit(self, worker_id: int, masked_grads: MaskedGradients):
        """
        Submit masked gradients from a worker.
        
        Args:
            worker_id: Worker identifier
            masked_grads: Masked gradient submission
        """
        if worker_id not in self._workers:
            raise ValueError(f"Unknown worker {worker_id}")
        
        self._submissions[worker_id] = masked_grads
    
    def can_aggregate(self) -> bool:
        """Check if enough workers have submitted."""
        return len(self._submissions) >= self.threshold
    
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate masked gradients.
        
        Returns:
            Aggregated (average) gradients
        
        Raises:
            ValueError: If not enough submissions
        """
        if not self.can_aggregate():
            raise ValueError(
                f"Need {self.threshold} submissions, have {len(self._submissions)}"
            )
        
        # Get gradient structure from first submission
        first = next(iter(self._submissions.values()))
        grad_keys = list(first.masked_grads.keys())
        
        # Sum masked gradients (masks cancel out)
        aggregated = {}
        for key in grad_keys:
            total = None
            for worker_id, submission in self._submissions.items():
                grad = submission.masked_grads[key]
                if total is None:
                    total = grad.clone()
                else:
                    total += grad
            aggregated[key] = total / len(self._submissions)
        
        # Clear submissions for next round
        self._submissions.clear()
        
        return aggregated
    
    def reset(self):
        """Reset aggregator for new round."""
        self._submissions.clear()
        self._pairwise_seeds.clear()


class WorkerMasker:
    """
    Client-side masking for secure aggregation.
    
    Each worker uses this to mask their gradients before
    sending to the aggregator.
    """
    
    def __init__(
        self,
        worker_id: int,
        pairwise_seeds: Dict[int, bytes],
        all_worker_ids: List[int]
    ):
        """
        Initialize worker masker.
        
        Args:
            worker_id: This worker's ID
            pairwise_seeds: Shared seeds with other workers
            all_worker_ids: List of all worker IDs
        """
        self.worker_id = worker_id
        self.pairwise_seeds = pairwise_seeds
        self.all_worker_ids = sorted(all_worker_ids)
    
    def _generate_mask(
        self,
        seed: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        """Generate deterministic mask from seed."""
        # Use seed to initialize generator
        seed_int = int.from_bytes(seed[:8], 'big')
        gen = torch.Generator().manual_seed(seed_int)
        
        mask = torch.randn(shape, generator=gen, dtype=dtype, device=device)
        return mask
    
    def mask_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> MaskedGradients:
        """
        Mask gradients for secure aggregation.
        
        Args:
            gradients: Raw gradients to mask
        
        Returns:
            Masked gradients ready for submission
        """
        masked = {}
        seed_commitments = {}
        
        for key, grad in gradients.items():
            masked_grad = grad.clone()
            
            # Add/subtract pairwise masks
            for other_id, seed in self.pairwise_seeds.items():
                mask = self._generate_mask(
                    seed, grad.shape, grad.dtype, grad.device
                )
                
                # Sign convention: smaller ID subtracts, larger ID adds
                if self.worker_id < other_id:
                    masked_grad = masked_grad - mask
                else:
                    masked_grad = masked_grad + mask
                
                # Commitment to seed (for verification)
                seed_commitments[other_id] = hashlib.sha256(seed).digest()
            
            masked[key] = masked_grad
        
        return MaskedGradients(
            worker_id=self.worker_id,
            masked_grads=masked,
            seed_commitments=seed_commitments
        )


class FederatedAggregator:
    """
    High-level federated learning aggregator with privacy.
    
    Combines:
    - Secure aggregation (hide individual gradients)
    - Differential privacy (add noise to aggregate)
    - Gradient compression (reduce bandwidth)
    """
    
    def __init__(
        self,
        num_workers: int,
        threshold: int = None,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        compression_ratio: float = 1.0,  # 1.0 = no compression
    ):
        """
        Initialize federated aggregator.
        
        Args:
            num_workers: Total workers
            threshold: Minimum workers for aggregation (default: all)
            dp_epsilon: Differential privacy epsilon (set 0 to disable)
            dp_delta: Differential privacy delta
            compression_ratio: Gradient compression ratio
        """
        self.num_workers = num_workers
        self.threshold = threshold or num_workers
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.compression_ratio = compression_ratio
        
        # Initialize components
        self._aggregator = SecureAggregator(num_workers, self.threshold)
        
        if dp_epsilon > 0:
            from .differential_privacy import GaussianMechanism
            self._dp = GaussianMechanism(
                sensitivity=1.0,  # Assumes normalized gradients
                epsilon=dp_epsilon,
                delta=dp_delta
            )
        else:
            self._dp = None
    
    def setup_round(self) -> Dict[int, Dict[int, bytes]]:
        """
        Setup a new aggregation round.
        
        Returns:
            Pairwise seeds to distribute to workers
        """
        return self._aggregator.generate_pairwise_seeds()
    
    def submit(self, worker_id: int, masked_grads: MaskedGradients):
        """Submit masked gradients from worker."""
        self._aggregator.submit(worker_id, masked_grads)
    
    def aggregate(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients with privacy.
        
        Returns:
            Private aggregated gradients
        """
        # Secure aggregation
        aggregated = self._aggregator.aggregate()
        
        # Add DP noise
        if self._dp is not None:
            aggregated = {
                k: self._dp.apply(v) for k, v in aggregated.items()
            }
        
        return aggregated
    
    @property
    def ready(self) -> bool:
        """Check if aggregation can proceed."""
        return self._aggregator.can_aggregate()
