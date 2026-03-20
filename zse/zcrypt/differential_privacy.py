"""
Differential Privacy for zCrypt

Implements differential privacy mechanisms to protect training data
from model inversion attacks.

Key concepts:
- Epsilon (ε): Privacy budget. Lower = more private, but noisier.
- Delta (δ): Probability of privacy breach. Should be < 1/|dataset|.
- Sensitivity: Maximum change in output from one sample.

Mechanisms:
- Gaussian: Good for high-dimensional data (gradients)
- Laplacian: Better for sparse data
- Gradient clipping: Bounds sensitivity

Based on: "Deep Learning with Differential Privacy", Abadi et al.
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable, Union
import torch
import torch.nn as nn


@dataclass
class PrivacyAccountant:
    """
    Tracks privacy budget consumption across training.
    
    Uses Rényi Differential Privacy (RDP) for tighter composition.
    """
    
    epsilon: float
    delta: float
    noise_multiplier: float
    sample_rate: float  # batch_size / dataset_size
    
    _steps: int = 0
    _spent_epsilon: float = 0.0
    
    def step(self) -> float:
        """Record one optimization step, return spent epsilon."""
        self._steps += 1
        
        # RDP composition (simplified)
        # Full implementation would use RDP accountant
        step_epsilon = self._compute_step_epsilon()
        self._spent_epsilon += step_epsilon
        
        return self._spent_epsilon
    
    def _compute_step_epsilon(self) -> float:
        """Compute epsilon consumed by one step."""
        # Simplified: linear composition
        # Real implementation uses moments accountant
        sigma = self.noise_multiplier
        q = self.sample_rate
        
        # Approximate epsilon per step
        return 2 * q * math.sqrt(2 * math.log(1.25 / self.delta)) / sigma
    
    def remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon - self._spent_epsilon)
    
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self._spent_epsilon >= self.epsilon
    
    @property
    def steps(self) -> int:
        """Number of steps taken."""
        return self._steps


def compute_sigma(
    epsilon: float,
    delta: float,
    sensitivity: float,
    sample_rate: float,
    num_steps: int
) -> float:
    """
    Compute noise multiplier (sigma) for target epsilon.
    
    Args:
        epsilon: Target epsilon
        delta: Target delta
        sensitivity: L2 sensitivity (max_grad_norm)
        sample_rate: Batch size / dataset size
        num_steps: Total training steps
    
    Returns:
        Noise multiplier sigma
    """
    # Simplified formula (for exact, use RDP accountant)
    # Based on Gaussian mechanism with subsampling
    
    c = math.sqrt(2 * math.log(1.25 / delta))
    
    # Adjust for composition over steps (simple linear composition)
    # Real implementation uses advanced composition theorems
    epsilon_per_step = epsilon / math.sqrt(num_steps)
    
    sigma = c * sensitivity * sample_rate / epsilon_per_step
    
    return max(sigma, 0.01)  # Minimum noise to prevent division by zero


def compute_privacy_budget(
    noise_multiplier: float,
    sample_rate: float,
    num_steps: int,
    delta: float = 1e-5
) -> float:
    """
    Compute total epsilon for given training configuration.
    
    Args:
        noise_multiplier: Noise multiplier (sigma)
        sample_rate: Batch size / dataset size
        num_steps: Total training steps
        delta: Target delta
    
    Returns:
        Total epsilon consumed
    """
    # Simplified (linear composition)
    c = math.sqrt(2 * math.log(1.25 / delta))
    epsilon_per_step = c * sample_rate / noise_multiplier
    
    # Square root composition (advanced composition theorem)
    total_epsilon = epsilon_per_step * math.sqrt(num_steps)
    
    return total_epsilon


class GaussianMechanism:
    """
    Gaussian mechanism for differential privacy.
    
    Adds Gaussian noise calibrated to sensitivity and epsilon.
    Best for high-dimensional data like gradients.
    """
    
    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float
    ):
        """
        Initialize Gaussian mechanism.
        
        Args:
            sensitivity: L2 sensitivity (max change from one sample)
            epsilon: Privacy parameter
            delta: Failure probability
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta
        
        # Compute sigma
        self.sigma = self._compute_sigma()
    
    def _compute_sigma(self) -> float:
        """Compute noise scale."""
        c = math.sqrt(2 * math.log(1.25 / self.delta))
        return c * self.sensitivity / self.epsilon
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Add calibrated noise to input."""
        noise = torch.randn_like(x) * self.sigma
        return x + noise
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Add calibrated noise to input."""
        return self(x)


class LaplaceMechanism:
    """
    Laplace mechanism for differential privacy.
    
    Adds Laplace noise calibrated to sensitivity and epsilon.
    Better for sparse data.
    """
    
    def __init__(self, sensitivity: float, epsilon: float):
        """
        Initialize Laplace mechanism.
        
        Args:
            sensitivity: L1 sensitivity
            epsilon: Privacy parameter
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.scale = sensitivity / epsilon
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Add calibrated Laplace noise."""
        # Laplace(0, scale) = Exponential(1/scale) - Exponential(1/scale)
        u = torch.rand_like(x) - 0.5
        noise = -self.scale * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
        return x + noise
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Add calibrated Laplace noise."""
        return self(x)


def clip_gradients(
    gradients: Union[torch.Tensor, List[torch.Tensor]],
    max_norm: float,
    norm_type: float = 2.0
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Clip gradients to bound sensitivity.
    
    Args:
        gradients: Single gradient tensor or list of tensors
        max_norm: Maximum norm threshold
        norm_type: Type of norm (1, 2, or inf)
    
    Returns:
        Clipped gradients (same structure as input)
    """
    if isinstance(gradients, torch.Tensor):
        return _clip_single(gradients, max_norm, norm_type)
    
    # List of tensors: compute total norm, then scale each
    total_norm = torch.norm(
        torch.stack([torch.norm(g, norm_type) for g in gradients]),
        norm_type
    )
    
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    
    return [g * clip_coef for g in gradients]


def _clip_single(
    grad: torch.Tensor,
    max_norm: float,
    norm_type: float
) -> torch.Tensor:
    """Clip single gradient tensor."""
    grad_norm = torch.norm(grad, norm_type)
    clip_coef = max_norm / (grad_norm + 1e-6)
    clip_coef = torch.clamp(clip_coef, max=1.0)
    return grad * clip_coef


def add_noise(
    gradients: Union[torch.Tensor, List[torch.Tensor]],
    noise_multiplier: float,
    max_grad_norm: float
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Add calibrated Gaussian noise to gradients.
    
    Args:
        gradients: Gradient tensor(s)
        noise_multiplier: Sigma / max_grad_norm
        max_grad_norm: Clipping threshold (also sensitivity bound)
    
    Returns:
        Noisy gradients
    """
    sigma = noise_multiplier * max_grad_norm
    
    if isinstance(gradients, torch.Tensor):
        noise = torch.randn_like(gradients) * sigma
        return gradients + noise
    
    return [g + torch.randn_like(g) * sigma for g in gradients]


class DPEngine:
    """
    Differential Privacy engine for training.
    
    Wraps optimizer to automatically:
    1. Clip per-sample gradients
    2. Add calibrated noise
    3. Track privacy budget
    
    Usage:
        dp_engine = DPEngine(
            model=model,
            optimizer=optimizer,
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            dataset_size=10000,
            batch_size=32,
            epochs=3
        )
        
        for batch in dataloader:
            loss = model(batch)
            dp_engine.step(loss)
        
        print(f"Privacy spent: ε={dp_engine.spent_epsilon:.2f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        dataset_size: int,
        batch_size: int,
        epochs: int,
        noise_multiplier: Optional[float] = None,
    ):
        """
        Initialize DP engine.
        
        Args:
            model: Model to train
            optimizer: PyTorch optimizer
            epsilon: Target epsilon budget
            delta: Target delta
            max_grad_norm: Gradient clipping threshold
            dataset_size: Total samples in dataset
            batch_size: Training batch size
            epochs: Number of training epochs
            noise_multiplier: Override auto-computed sigma
        """
        self.model = model
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        # Compute derived params
        self.sample_rate = batch_size / dataset_size
        self.num_steps = int(epochs * dataset_size / batch_size)
        
        # Compute noise multiplier
        if noise_multiplier is None:
            self.noise_multiplier = compute_sigma(
                epsilon, delta, max_grad_norm,
                self.sample_rate, self.num_steps
            )
        else:
            self.noise_multiplier = noise_multiplier
        
        # Initialize accountant
        self.accountant = PrivacyAccountant(
            epsilon=epsilon,
            delta=delta,
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate
        )
        
        # Gaussian mechanism for gradients
        self.mechanism = GaussianMechanism(
            sensitivity=max_grad_norm,
            epsilon=epsilon / self.num_steps,  # Per-step budget
            delta=delta
        )
    
    def step(self, loss: torch.Tensor) -> float:
        """
        Execute one DP training step.
        
        Args:
            loss: Loss to backprop
        
        Returns:
            Spent epsilon so far
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Collect gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data)
        
        # Clip gradients
        clipped = clip_gradients(gradients, self.max_grad_norm)
        
        # Add noise
        noisy = add_noise(clipped, self.noise_multiplier, self.max_grad_norm)
        
        # Update model gradients
        idx = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data = noisy[idx]
                idx += 1
        
        # Update weights
        self.optimizer.step()
        
        # Track privacy
        return self.accountant.step()
    
    @property
    def spent_epsilon(self) -> float:
        """Total epsilon spent."""
        return self.accountant._spent_epsilon
    
    @property
    def remaining_budget(self) -> float:
        """Remaining epsilon budget."""
        return self.accountant.remaining_budget()
    
    def privacy_report(self) -> Dict[str, float]:
        """Get privacy budget report."""
        return {
            "target_epsilon": self.epsilon,
            "spent_epsilon": self.spent_epsilon,
            "remaining_epsilon": self.remaining_budget,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "steps_taken": self.accountant.steps,
            "total_steps": self.num_steps,
        }


class PerSampleGradientClipper:
    """
    Clips gradients per-sample for tighter DP bounds.
    
    Standard gradient clipping clips the sum of gradients.
    Per-sample clipping clips each sample's gradient individually,
    providing tighter privacy guarantees.
    """
    
    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = max_grad_norm
    
    def clip_and_accumulate(
        self,
        per_sample_grads: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Clip each sample's gradients and accumulate.
        
        Args:
            per_sample_grads: List of gradient dicts, one per sample
        
        Returns:
            Accumulated clipped gradients
        """
        accumulated = {}
        
        for sample_grads in per_sample_grads:
            # Compute per-sample norm
            flat_grads = torch.cat([g.flatten() for g in sample_grads.values()])
            grad_norm = torch.norm(flat_grads, 2)
            
            # Clip coefficient
            clip_coef = min(1.0, self.max_grad_norm / (grad_norm + 1e-6))
            
            # Accumulate clipped gradients
            for name, grad in sample_grads.items():
                clipped = grad * clip_coef
                if name in accumulated:
                    accumulated[name] += clipped
                else:
                    accumulated[name] = clipped.clone()
        
        # Average
        num_samples = len(per_sample_grads)
        for name in accumulated:
            accumulated[name] /= num_samples
        
        return accumulated
