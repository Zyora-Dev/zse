"""
Split Learning for zCrypt

Implements split learning where:
1. Client keeps embedding layer (raw data never leaves client)
2. Worker receives only intermediate activations (encrypted)
3. Gradients flow back through the cut point

This ensures the worker (GPU farmer) never sees the raw training data.

Paper: https://arxiv.org/abs/1812.00564
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict, Any
import torch
import torch.nn as nn

from .encryption import AESCipher, encrypt_tensor, decrypt_tensor, EncryptedTensor


@dataclass
class SplitConfig:
    """Configuration for split learning."""
    
    cut_layer: int = 1
    """Layer index to split at (0 = after embedding)."""
    
    encrypt_activations: bool = True
    """Encrypt activations before sending to worker."""
    
    compress_activations: bool = False
    """Apply compression to activations (reduces bandwidth)."""
    
    compression_ratio: float = 0.5
    """Target compression ratio if compression enabled."""


class ActivationEncryptor:
    """
    Encrypts intermediate activations for split learning.
    
    The client computes embeddings locally, then encrypts the
    activations before sending to the worker.
    """
    
    def __init__(self, cipher: Optional[AESCipher] = None):
        """
        Initialize with optional cipher.
        
        Args:
            cipher: AES cipher (created if not provided)
        """
        self.cipher = cipher or AESCipher()
    
    def encrypt_activation(
        self,
        activation: torch.Tensor,
        layer_name: str
    ) -> EncryptedTensor:
        """
        Encrypt activation tensor.
        
        Args:
            activation: Intermediate activation from forward pass
            layer_name: Name of the layer (for AAD)
        
        Returns:
            Encrypted activation
        """
        return encrypt_tensor(
            activation,
            self.cipher,
            associated_data=layer_name.encode('utf-8')
        )
    
    def decrypt_activation(self, encrypted: EncryptedTensor) -> torch.Tensor:
        """Decrypt activation tensor."""
        return decrypt_tensor(encrypted, self.cipher)
    
    def encrypt_gradient(
        self,
        gradient: torch.Tensor,
        layer_name: str
    ) -> EncryptedTensor:
        """Encrypt gradient for backprop through split."""
        return encrypt_tensor(
            gradient,
            self.cipher,
            associated_data=f"grad_{layer_name}".encode('utf-8')
        )
    
    def decrypt_gradient(self, encrypted: EncryptedTensor) -> torch.Tensor:
        """Decrypt gradient."""
        return decrypt_tensor(encrypted, self.cipher)


class SplitLearningClient:
    """
    Client-side split learning component.
    
    Keeps the embedding layer local and only sends encrypted
    activations to the worker.
    
    Usage:
        client = SplitLearningClient(model, cut_layer=1)
        
        # Forward: compute local, send encrypted activation
        encrypted_act = client.forward(input_ids)
        
        # Send encrypted_act to worker...
        # Worker computes rest of forward + backward...
        
        # Backward: receive encrypted gradient, compute local
        client.backward(encrypted_grad)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[SplitConfig] = None,
        cipher: Optional[AESCipher] = None,
    ):
        """
        Initialize client-side split learning.
        
        Args:
            model: Full model (we'll use only first layers)
            config: Split learning configuration
            cipher: Encryption cipher (shared with worker for gradients)
        """
        self.config = config or SplitConfig()
        self.encryptor = ActivationEncryptor(cipher)
        
        # Extract client-side layers (up to cut point)
        self.client_layers = self._extract_client_layers(model)
        self._activation_cache: Optional[torch.Tensor] = None
    
    def _extract_client_layers(self, model: nn.Module) -> nn.Module:
        """Extract layers up to cut point."""
        # For transformer models, typically cut after embedding
        if hasattr(model, 'model'):
            base = model.model
        else:
            base = model
        
        # Get embedding layer
        if hasattr(base, 'embed_tokens'):
            return base.embed_tokens
        elif hasattr(base, 'wte'):
            return base.wte
        elif hasattr(base, 'embeddings'):
            return base.embeddings
        else:
            # Fallback: use first layer
            layers = list(model.children())
            if layers:
                return layers[0]
            raise ValueError("Could not identify embedding layer")
    
    def forward(self, input_ids: torch.Tensor) -> EncryptedTensor:
        """
        Compute client-side forward pass.
        
        Args:
            input_ids: Input token IDs
        
        Returns:
            Encrypted activation to send to worker
        """
        # Compute embeddings locally
        with torch.no_grad():  # Embeddings don't need grad on client
            activation = self.client_layers(input_ids)
        
        # Cache for backward pass
        self._activation_cache = activation.detach().requires_grad_(True)
        
        # Encrypt before sending
        if self.config.encrypt_activations:
            return self.encryptor.encrypt_activation(activation, "embedding")
        else:
            return activation
    
    def backward(
        self,
        encrypted_grad: EncryptedTensor,
        update_embeddings: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Compute client-side backward pass.
        
        Args:
            encrypted_grad: Encrypted gradient from worker
            update_embeddings: Whether to update embedding weights
        
        Returns:
            Gradient w.r.t input (if needed)
        """
        if self._activation_cache is None:
            raise RuntimeError("Must call forward() before backward()")
        
        # Decrypt gradient
        if isinstance(encrypted_grad, EncryptedTensor):
            grad = self.encryptor.decrypt_gradient(encrypted_grad)
        else:
            grad = encrypted_grad
        
        # Backprop through embedding
        if update_embeddings and self._activation_cache.grad_fn is not None:
            self._activation_cache.backward(grad)
        
        # Clear cache
        self._activation_cache = None
        
        return None  # Usually don't need input gradient
    
    @property
    def cipher_key(self) -> bytes:
        """Get cipher key for sharing with worker."""
        return self.encryptor.cipher.key


class SplitLearningWorker:
    """
    Worker-side split learning component.
    
    Receives encrypted activations, computes forward/backward
    on remaining layers, returns encrypted gradients.
    
    The worker never sees the raw input data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[SplitConfig] = None,
        cipher_key: Optional[bytes] = None,
    ):
        """
        Initialize worker-side split learning.
        
        Args:
            model: Full model (we'll use layers after cut point)
            config: Split learning configuration
            cipher_key: Shared cipher key from client
        """
        self.config = config or SplitConfig()
        
        # Create cipher with shared key
        if cipher_key:
            self.encryptor = ActivationEncryptor(AESCipher(cipher_key))
        else:
            self.encryptor = ActivationEncryptor()
        
        # Extract worker-side layers (after cut point)
        self.worker_model = self._extract_worker_layers(model)
        self._input_activation: Optional[torch.Tensor] = None
    
    def _extract_worker_layers(self, model: nn.Module) -> nn.Module:
        """Extract layers after cut point."""
        # For transformer models, return everything except embedding
        # This is model-specific, but we try common patterns
        
        class WorkerModel(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model
                
                # Find the main layers
                if hasattr(base_model, 'model'):
                    self.actual_model = base_model.model
                else:
                    self.actual_model = base_model
            
            def forward(self, hidden_states):
                # Skip embedding, start from hidden states
                model = self.actual_model
                
                # Apply transformer layers
                if hasattr(model, 'layers'):
                    for layer in model.layers:
                        hidden_states = layer(hidden_states)[0]
                elif hasattr(model, 'h'):
                    for layer in model.h:
                        hidden_states = layer(hidden_states)[0]
                
                # Apply final norm
                if hasattr(model, 'norm'):
                    hidden_states = model.norm(hidden_states)
                elif hasattr(model, 'ln_f'):
                    hidden_states = model.ln_f(hidden_states)
                
                # Apply LM head
                if hasattr(self.base, 'lm_head'):
                    logits = self.base.lm_head(hidden_states)
                elif hasattr(self.base, 'output'):
                    logits = self.base.output(hidden_states)
                else:
                    logits = hidden_states
                
                return logits
        
        return WorkerModel(model)
    
    def forward(
        self,
        encrypted_activation: EncryptedTensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute worker-side forward pass.
        
        Args:
            encrypted_activation: Encrypted activation from client
            labels: Optional labels for loss computation
        
        Returns:
            (logits, loss) - loss is None if no labels provided
        """
        # Decrypt activation
        if isinstance(encrypted_activation, EncryptedTensor):
            activation = self.encryptor.decrypt_activation(encrypted_activation)
        else:
            activation = encrypted_activation
        
        # Enable grad for backward
        activation = activation.detach().requires_grad_(True)
        self._input_activation = activation
        
        # Forward through worker layers
        logits = self.worker_model(activation)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Reshape for loss computation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def backward(self, loss: torch.Tensor) -> EncryptedTensor:
        """
        Compute worker-side backward and return encrypted gradient.
        
        Args:
            loss: Loss to backprop from
        
        Returns:
            Encrypted gradient to send back to client
        """
        if self._input_activation is None:
            raise RuntimeError("Must call forward() before backward()")
        
        # Backward pass
        loss.backward()
        
        # Get gradient at split point
        grad = self._input_activation.grad
        
        if grad is None:
            raise RuntimeError("No gradient computed at split point")
        
        # Encrypt gradient before sending back
        encrypted_grad = self.encryptor.encrypt_gradient(grad, "embedding")
        
        # Clear cache
        self._input_activation = None
        
        return encrypted_grad


class SplitLearningSession:
    """
    Manages a complete split learning training session.
    
    Coordinates between client and worker, handling encryption,
    communication, and gradient synchronization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[SplitConfig] = None,
    ):
        """
        Initialize split learning session.
        
        Args:
            model: Full model to split
            config: Split learning configuration
        """
        self.config = config or SplitConfig()
        
        # Create shared cipher
        self._cipher = AESCipher()
        
        # Create client and worker
        self.client = SplitLearningClient(model, config, self._cipher)
        self.worker = SplitLearningWorker(
            model, config, 
            cipher_key=self._cipher.key
        )
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Execute one training step with split learning.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels
        
        Returns:
            Loss value
        """
        # Client forward (local embedding + encrypt)
        encrypted_act = self.client.forward(input_ids)
        
        # Worker forward (decrypt + rest of model)
        logits, loss = self.worker.forward(encrypted_act, labels)
        
        if loss is None:
            raise ValueError("Loss is None - labels may be invalid")
        
        # Worker backward (compute + encrypt gradient)
        encrypted_grad = self.worker.backward(loss)
        
        # Client backward (decrypt + update embedding)
        self.client.backward(encrypted_grad)
        
        return loss.item()
    
    @property
    def session_key(self) -> bytes:
        """Get session encryption key."""
        return self._cipher.key
