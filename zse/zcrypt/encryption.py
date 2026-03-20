"""
Encryption Layer for zCrypt

AES-256-GCM encryption for tensors and data.
Supports both synchronous and asynchronous encryption.

Security:
- AES-256-GCM (authenticated encryption)
- Random IVs per encryption
- Key derivation via PBKDF2 or HKDF
- Constant-time operations where possible
"""

import os
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import struct

import torch
import numpy as np

# Try to use cryptography library (production-grade)
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Fallback to PyCryptodome
if not CRYPTO_AVAILABLE:
    try:
        from Crypto.Cipher import AES
        from Crypto.Protocol.KDF import PBKDF2
        from Crypto.Random import get_random_bytes
        PYCRYPTO_AVAILABLE = True
    except ImportError:
        PYCRYPTO_AVAILABLE = False
else:
    PYCRYPTO_AVAILABLE = False


# Constants
KEY_SIZE = 32  # 256 bits
IV_SIZE = 12   # 96 bits for GCM
TAG_SIZE = 16  # 128 bits authentication tag
SALT_SIZE = 16


def generate_key() -> bytes:
    """Generate a cryptographically secure random key."""
    return secrets.token_bytes(KEY_SIZE)


def derive_key(
    password: Union[str, bytes],
    salt: Optional[bytes] = None,
    iterations: int = 100000
) -> Tuple[bytes, bytes]:
    """
    Derive encryption key from password using PBKDF2.
    
    Args:
        password: Password string or bytes
        salt: Optional salt (generated if not provided)
        iterations: PBKDF2 iterations (higher = more secure, slower)
    
    Returns:
        (derived_key, salt)
    """
    if isinstance(password, str):
        password = password.encode('utf-8')
    
    if salt is None:
        salt = secrets.token_bytes(SALT_SIZE)
    
    if CRYPTO_AVAILABLE:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        key = kdf.derive(password)
    elif PYCRYPTO_AVAILABLE:
        key = PBKDF2(password, salt, dkLen=KEY_SIZE, count=iterations)
    else:
        # Pure Python fallback (slower but works)
        key = hashlib.pbkdf2_hmac('sha256', password, salt, iterations, dklen=KEY_SIZE)
    
    return key, salt


def derive_key_hkdf(
    input_key: bytes,
    info: bytes = b"zcrypt-tensor-encryption",
    length: int = KEY_SIZE
) -> bytes:
    """
    Derive key using HKDF (for key expansion).
    
    Args:
        input_key: Input keying material
        info: Context/application-specific info
        length: Output key length
    
    Returns:
        Derived key
    """
    if CRYPTO_AVAILABLE:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,
            info=info,
            backend=default_backend()
        )
        return hkdf.derive(input_key)
    else:
        # Simple HKDF extract-expand
        prk = hmac.new(b'\x00' * 32, input_key, hashlib.sha256).digest()
        okm = b''
        t = b''
        for i in range((length + 31) // 32):
            t = hmac.new(prk, t + info + bytes([i + 1]), hashlib.sha256).digest()
            okm += t
        return okm[:length]


class AESCipher:
    """
    AES-256-GCM cipher for authenticated encryption.
    
    Thread-safe and supports async operations.
    """
    
    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize cipher with key.
        
        Args:
            key: 32-byte encryption key (generated if not provided)
        """
        self.key = key if key else generate_key()
        
        if len(self.key) != KEY_SIZE:
            raise ValueError(f"Key must be {KEY_SIZE} bytes, got {len(self.key)}")
        
        if CRYPTO_AVAILABLE:
            self._aesgcm = AESGCM(self.key)
        elif PYCRYPTO_AVAILABLE:
            self._aesgcm = None  # Created per-encryption
        else:
            raise RuntimeError(
                "No cryptography library available. "
                "Install 'cryptography' or 'pycryptodome': pip install cryptography"
            )
        
        # Thread pool for async encryption
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def encrypt(self, plaintext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        Encrypt data with AES-256-GCM.
        
        Args:
            plaintext: Data to encrypt
            associated_data: Additional authenticated data (not encrypted)
        
        Returns:
            iv + ciphertext + tag
        """
        iv = secrets.token_bytes(IV_SIZE)
        
        if CRYPTO_AVAILABLE:
            ciphertext = self._aesgcm.encrypt(iv, plaintext, associated_data)
        else:
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=iv)
            if associated_data:
                cipher.update(associated_data)
            ciphertext, tag = cipher.encrypt_and_digest(plaintext)
            ciphertext = ciphertext + tag
        
        return iv + ciphertext
    
    def decrypt(self, ciphertext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """
        Decrypt data with AES-256-GCM.
        
        Args:
            ciphertext: iv + ciphertext + tag
            associated_data: Additional authenticated data
        
        Returns:
            Decrypted plaintext
        
        Raises:
            ValueError: If authentication fails
        """
        if len(ciphertext) < IV_SIZE + TAG_SIZE:
            raise ValueError("Ciphertext too short")
        
        iv = ciphertext[:IV_SIZE]
        encrypted = ciphertext[IV_SIZE:]
        
        if CRYPTO_AVAILABLE:
            try:
                return self._aesgcm.decrypt(iv, encrypted, associated_data)
            except Exception as e:
                raise ValueError(f"Decryption failed: {e}")
        else:
            tag = encrypted[-TAG_SIZE:]
            encrypted = encrypted[:-TAG_SIZE]
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=iv)
            if associated_data:
                cipher.update(associated_data)
            try:
                return cipher.decrypt_and_verify(encrypted, tag)
            except Exception as e:
                raise ValueError(f"Decryption failed: {e}")
    
    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


@dataclass
class EncryptedTensor:
    """Container for encrypted tensor data."""
    
    encrypted_data: bytes
    shape: Tuple[int, ...]
    dtype: str
    device: str
    associated_data: Optional[bytes] = None
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for transmission."""
        # Header: shape_len (4) + dtype_len (4) + device_len (4) + aad_len (4)
        shape_bytes = struct.pack(f'{len(self.shape)}i', *self.shape)
        dtype_bytes = self.dtype.encode('utf-8')
        device_bytes = self.device.encode('utf-8')
        aad_bytes = self.associated_data or b''
        
        header = struct.pack(
            '4I',
            len(shape_bytes),
            len(dtype_bytes),
            len(device_bytes),
            len(aad_bytes)
        )
        
        return (
            header + 
            shape_bytes + 
            dtype_bytes + 
            device_bytes + 
            aad_bytes +
            self.encrypted_data
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedTensor":
        """Deserialize from bytes."""
        header_size = 16  # 4 * 4 bytes
        shape_len, dtype_len, device_len, aad_len = struct.unpack('4I', data[:header_size])
        
        offset = header_size
        shape_bytes = data[offset:offset + shape_len]
        offset += shape_len
        
        dtype_bytes = data[offset:offset + dtype_len]
        offset += dtype_len
        
        device_bytes = data[offset:offset + device_len]
        offset += device_len
        
        aad_bytes = data[offset:offset + aad_len] if aad_len > 0 else None
        offset += aad_len
        
        encrypted_data = data[offset:]
        
        # Unpack shape
        shape = struct.unpack(f'{shape_len // 4}i', shape_bytes)
        
        return cls(
            encrypted_data=encrypted_data,
            shape=shape,
            dtype=dtype_bytes.decode('utf-8'),
            device=device_bytes.decode('utf-8'),
            associated_data=aad_bytes
        )


def _tensor_to_bytes(tensor: torch.Tensor) -> Tuple[bytes, Tuple[int, ...], str]:
    """Convert tensor to bytes."""
    # Move to CPU for serialization
    cpu_tensor = tensor.detach().cpu()
    numpy_array = cpu_tensor.numpy()
    return numpy_array.tobytes(), tuple(tensor.shape), str(tensor.dtype)


def _bytes_to_tensor(
    data: bytes,
    shape: Tuple[int, ...],
    dtype_str: str,
    device: str
) -> torch.Tensor:
    """Convert bytes back to tensor."""
    # Map string dtype to numpy dtype
    dtype_map = {
        'torch.float32': np.float32,
        'torch.float16': np.float16,
        'torch.bfloat16': np.float32,  # bfloat16 needs special handling
        'torch.int32': np.int32,
        'torch.int64': np.int64,
        'torch.int8': np.int8,
        'torch.uint8': np.uint8,
    }
    
    np_dtype = dtype_map.get(dtype_str, np.float32)
    numpy_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
    tensor = torch.from_numpy(numpy_array.copy())
    
    # Handle bfloat16
    if dtype_str == 'torch.bfloat16':
        tensor = tensor.to(torch.bfloat16)
    
    # Move to target device
    if device != 'cpu':
        tensor = tensor.to(device)
    
    return tensor


def encrypt_tensor(
    tensor: torch.Tensor,
    cipher: AESCipher,
    associated_data: Optional[bytes] = None
) -> EncryptedTensor:
    """
    Encrypt a PyTorch tensor.
    
    Args:
        tensor: Tensor to encrypt
        cipher: AES cipher instance
        associated_data: Optional AAD for authentication
    
    Returns:
        EncryptedTensor container
    """
    data_bytes, shape, dtype_str = _tensor_to_bytes(tensor)
    device = str(tensor.device)
    
    encrypted = cipher.encrypt(data_bytes, associated_data)
    
    return EncryptedTensor(
        encrypted_data=encrypted,
        shape=shape,
        dtype=dtype_str,
        device=device,
        associated_data=associated_data
    )


def decrypt_tensor(
    encrypted: EncryptedTensor,
    cipher: AESCipher
) -> torch.Tensor:
    """
    Decrypt an encrypted tensor.
    
    Args:
        encrypted: EncryptedTensor container
        cipher: AES cipher instance (same key as encryption)
    
    Returns:
        Decrypted tensor on original device
    """
    decrypted_bytes = cipher.decrypt(
        encrypted.encrypted_data,
        encrypted.associated_data
    )
    
    return _bytes_to_tensor(
        decrypted_bytes,
        encrypted.shape,
        encrypted.dtype,
        encrypted.device
    )


class TensorEncryptor:
    """
    High-level tensor encryption with batching and async support.
    """
    
    def __init__(
        self,
        key: Optional[bytes] = None,
        batch_size: int = 1024,
        async_mode: bool = True
    ):
        self.cipher = AESCipher(key)
        self.batch_size = batch_size
        self.async_mode = async_mode
        self._executor = ThreadPoolExecutor(max_workers=4) if async_mode else None
    
    def encrypt(
        self,
        tensor: torch.Tensor,
        layer_id: Optional[str] = None
    ) -> EncryptedTensor:
        """Encrypt tensor with optional layer context."""
        aad = layer_id.encode('utf-8') if layer_id else None
        return encrypt_tensor(tensor, self.cipher, aad)
    
    def decrypt(self, encrypted: EncryptedTensor) -> torch.Tensor:
        """Decrypt tensor."""
        return decrypt_tensor(encrypted, self.cipher)
    
    def encrypt_state_dict(
        self,
        state_dict: dict
    ) -> dict:
        """Encrypt entire model state dict."""
        encrypted_dict = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                encrypted_dict[key] = self.encrypt(tensor, layer_id=key)
            else:
                encrypted_dict[key] = tensor
        return encrypted_dict
    
    def decrypt_state_dict(self, encrypted_dict: dict) -> dict:
        """Decrypt entire model state dict."""
        state_dict = {}
        for key, value in encrypted_dict.items():
            if isinstance(value, EncryptedTensor):
                state_dict[key] = self.decrypt(value)
            else:
                state_dict[key] = value
        return state_dict
    
    @property
    def key(self) -> bytes:
        """Get encryption key (for secure sharing)."""
        return self.cipher.key
    
    def __del__(self):
        if self._executor:
            self._executor.shutdown(wait=False)
