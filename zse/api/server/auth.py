"""
ZSE API Authentication

API key authentication for securing the ZSE server.
Supports multiple API keys with rate limiting and permissions.
"""

import os
import secrets
import hashlib
import json
import time
import threading
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import deque

from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader, APIKeyQuery


# API key header/query parameter names
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)
BEARER_HEADER = APIKeyHeader(name="Authorization", auto_error=False)

# Default config path
DEFAULT_KEYS_FILE = Path.home() / ".zse" / "api_keys.json"


@dataclass
class APIKey:
    """API key with metadata."""
    key_hash: str  # SHA-256 hash of the key (for storage)
    name: str
    created_at: str
    permissions: List[str] = field(default_factory=lambda: ["*"])  # ["chat", "models", "admin"]
    rate_limit: Optional[int] = None  # Requests per minute, None = unlimited
    last_used: Optional[str] = None
    request_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKey":
        return cls(**data)


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for API keys.
    
    Uses a sliding window algorithm to track requests per minute.
    Thread-safe for concurrent access.
    """
    
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._requests: Dict[str, deque] = {}  # key_hash -> timestamps
        self._lock = threading.Lock()
    
    def check_and_record(self, key_hash: str, limit: int) -> tuple:
        """
        Check if request is allowed and record it.
        
        Args:
            key_hash: Hash of the API key
            limit: Maximum requests per window
            
        Returns:
            (allowed: bool, current_count: int, reset_seconds: int)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        with self._lock:
            # Initialize deque if needed
            if key_hash not in self._requests:
                self._requests[key_hash] = deque()
            
            timestamps = self._requests[key_hash]
            
            # Remove expired timestamps
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()
            
            current_count = len(timestamps)
            
            # Calculate reset time (when oldest request expires)
            if timestamps:
                reset_seconds = int(timestamps[0] - window_start)
            else:
                reset_seconds = self.window_seconds
            
            # Check if under limit
            if current_count >= limit:
                return False, current_count, reset_seconds
            
            # Record this request
            timestamps.append(now)
            return True, current_count + 1, reset_seconds
    
    def get_usage(self, key_hash: str) -> Dict[str, Any]:
        """Get current usage stats for a key."""
        now = time.time()
        window_start = now - self.window_seconds
        
        with self._lock:
            if key_hash not in self._requests:
                return {"current": 0, "window_seconds": self.window_seconds}
            
            timestamps = self._requests[key_hash]
            
            # Count only active timestamps
            count = sum(1 for ts in timestamps if ts >= window_start)
            
            return {
                "current": count,
                "window_seconds": self.window_seconds,
            }
    
    def reset(self, key_hash: str) -> None:
        """Reset rate limit for a key."""
        with self._lock:
            if key_hash in self._requests:
                self._requests[key_hash].clear()
    
    def clear_expired(self) -> None:
        """Clear all expired timestamps (call periodically for cleanup)."""
        now = time.time()
        window_start = now - self.window_seconds
        
        with self._lock:
            for timestamps in self._requests.values():
                while timestamps and timestamps[0] < window_start:
                    timestamps.popleft()


# Global rate limiter instance
_rate_limiter: Optional[SlidingWindowRateLimiter] = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SlidingWindowRateLimiter(window_seconds=60)
    return _rate_limiter


class APIKeyManager:
    """Manages API keys for the ZSE server."""
    
    def __init__(self, keys_file: Optional[Path] = None):
        self.keys_file = keys_file or DEFAULT_KEYS_FILE
        self.keys: Dict[str, APIKey] = {}  # hash -> APIKey
        self._enabled = True
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load keys from file."""
        if self.keys_file.exists():
            try:
                with open(self.keys_file) as f:
                    data = json.load(f)
                    self.keys = {
                        k: APIKey.from_dict(v) 
                        for k, v in data.get("keys", {}).items()
                    }
                    self._enabled = data.get("enabled", True)
            except Exception as e:
                print(f"Warning: Could not load API keys: {e}")
    
    def _save_keys(self) -> None:
        """Save keys to file."""
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.keys_file, "w") as f:
            json.dump({
                "enabled": self._enabled,
                "keys": {k: v.to_dict() for k, v in self.keys.items()}
            }, f, indent=2)
        # Secure the file
        os.chmod(self.keys_file, 0o600)
    
    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new API key."""
        return f"zse-{secrets.token_urlsafe(32)}"
    
    def create_key(self, name: str, permissions: Optional[List[str]] = None, 
                   rate_limit: Optional[int] = None) -> str:
        """Create a new API key and return it (only shown once)."""
        key = self.generate_key()
        key_hash = self._hash_key(key)
        
        self.keys[key_hash] = APIKey(
            key_hash=key_hash,
            name=name,
            created_at=datetime.utcnow().isoformat(),
            permissions=permissions or ["*"],
            rate_limit=rate_limit
        )
        self._save_keys()
        return key
    
    def delete_key(self, name: str) -> bool:
        """Delete an API key by name."""
        for key_hash, api_key in list(self.keys.items()):
            if api_key.name == name:
                del self.keys[key_hash]
                self._save_keys()
                return True
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all API keys (without the actual key)."""
        return [
            {
                "name": k.name,
                "created_at": k.created_at,
                "last_used": k.last_used,
                "request_count": k.request_count,
                "permissions": k.permissions,
                "rate_limit": k.rate_limit
            }
            for k in self.keys.values()
        ]
    
    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key and return its metadata."""
        if not self._enabled:
            return APIKey(key_hash="", name="anonymous", created_at="")
        
        key_hash = self._hash_key(key)
        api_key = self.keys.get(key_hash)
        
        if api_key:
            # Update usage stats
            api_key.last_used = datetime.utcnow().isoformat()
            api_key.request_count += 1
            self._save_keys()
        
        return api_key
    
    def is_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable API key authentication."""
        self._enabled = True
        self._save_keys()
    
    def disable(self) -> None:
        """Disable API key authentication (allow all requests)."""
        self._enabled = False
        self._save_keys()


# Global manager instance
_key_manager: Optional[APIKeyManager] = None


def get_key_manager() -> APIKeyManager:
    """Get or create the global API key manager."""
    global _key_manager
    if _key_manager is None:
        _key_manager = APIKeyManager()
    return _key_manager


async def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
    api_key_query: Optional[str] = Security(API_KEY_QUERY),
    bearer_header: Optional[str] = Security(BEARER_HEADER),
) -> Optional[APIKey]:
    """
    Verify API key from header, query param, or Bearer token.
    
    Supports:
    - X-API-Key header
    - ?api_key= query parameter
    - Authorization: Bearer <key> header
    """
    manager = get_key_manager()
    
    # If auth is disabled, allow all
    if not manager.is_enabled():
        return None
    
    # Try to get key from various sources
    key = None
    
    if api_key_header:
        key = api_key_header
    elif api_key_query:
        key = api_key_query
    elif bearer_header and bearer_header.startswith("Bearer "):
        key = bearer_header[7:]
    
    if not key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via X-API-Key header, api_key query param, or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    api_key = manager.validate_key(key)
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Store API key info in request state for audit logging
    request.state.api_key_name = api_key.name
    request.state.api_key_hash = api_key.key_hash
    
    # Enforce rate limiting
    if api_key.rate_limit is not None and api_key.rate_limit > 0:
        rate_limiter = get_rate_limiter()
        allowed, current, reset_seconds = rate_limiter.check_and_record(
            api_key.key_hash, 
            api_key.rate_limit
        )
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Limit: {api_key.rate_limit}/min, Current: {current}. Try again in {reset_seconds}s.",
                headers={
                    "X-RateLimit-Limit": str(api_key.rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_seconds),
                    "Retry-After": str(reset_seconds),
                }
            )
        
        # Add rate limit headers to response (via request state)
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(api_key.rate_limit),
            "X-RateLimit-Remaining": str(api_key.rate_limit - current),
            "X-RateLimit-Reset": str(reset_seconds),
        }
    
    return api_key


def require_permission(permission: str):
    """Decorator factory to require specific permission."""
    async def check_permission(api_key: Optional[APIKey] = Security(verify_api_key)):
        if api_key and "*" not in api_key.permissions and permission not in api_key.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied. Required: {permission}"
            )
        return api_key
    return check_permission


# =============================================================================
# Rate Limit Utilities
# =============================================================================

def get_rate_limit_status(api_key: APIKey) -> Dict[str, Any]:
    """Get rate limit status for an API key."""
    if api_key.rate_limit is None or api_key.rate_limit <= 0:
        return {
            "limited": False,
            "limit": None,
            "current": 0,
            "remaining": None,
            "window_seconds": 60,
        }
    
    rate_limiter = get_rate_limiter()
    usage = rate_limiter.get_usage(api_key.key_hash)
    
    return {
        "limited": True,
        "limit": api_key.rate_limit,
        "current": usage["current"],
        "remaining": max(0, api_key.rate_limit - usage["current"]),
        "window_seconds": usage["window_seconds"],
    }


def reset_rate_limit(key_name: str) -> bool:
    """Reset rate limit for an API key by name."""
    manager = get_key_manager()
    
    for key_hash, api_key in manager.keys.items():
        if api_key.name == key_name:
            rate_limiter = get_rate_limiter()
            rate_limiter.reset(key_hash)
            return True
    
    return False
