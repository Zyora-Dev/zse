"""ZSE Server Auth — API key validation and rate limiting.

Validates Bearer tokens against the database, enforces rate limits
using a sliding window, and checks model access permissions.
"""

import time
from dataclasses import dataclass
from typing import Optional

from zse_engine.server.database import ServerDatabase, APIKey


@dataclass
class AuthResult:
    """Result of authentication check."""
    authenticated: bool
    key: Optional[APIKey] = None
    error: Optional[str] = None
    status_code: int = 200


class AuthManager:
    """API key authentication and rate limiting.

    Args:
        db: ServerDatabase instance
        admin_key: Master admin key (bypasses normal auth for admin routes)
    """

    def __init__(self, db: ServerDatabase, admin_key: Optional[str] = None):
        self._db = db
        self._admin_key = admin_key

    def authenticate(self, authorization: Optional[str]) -> AuthResult:
        """Validate a request's Authorization header.

        Args:
            authorization: The Authorization header value (e.g., "Bearer sk-zse-...")

        Returns:
            AuthResult with authentication status
        """
        if not authorization:
            return AuthResult(False, error="Missing Authorization header", status_code=401)

        # Parse "Bearer <token>"
        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0] != "Bearer":
            return AuthResult(False, error="Invalid Authorization format. Use: Bearer <key>", status_code=401)

        token = parts[1].strip()
        if not token:
            return AuthResult(False, error="Empty API key", status_code=401)

        # Check admin key
        if self._admin_key and token == self._admin_key:
            return AuthResult(True, key=APIKey(
                id=0, key_hash="", key_prefix="admin", name="admin",
                created_at=0, expires_at=None, is_active=True,
                rate_limit_rpm=0, rate_limit_tpm=0, allowed_models="*",
                total_requests=0, total_tokens=0,
            ))

        # Validate against database
        api_key = self._db.validate_key(token)
        if api_key is None:
            return AuthResult(False, error="Invalid or expired API key", status_code=401)

        return AuthResult(True, key=api_key)

    def check_rate_limit(self, key: APIKey) -> AuthResult:
        """Check if a key is within its rate limits.

        Args:
            key: Validated API key

        Returns:
            AuthResult — authenticated=True if within limits
        """
        if key.id == 0:  # Admin key — no limits
            return AuthResult(True, key=key)

        if key.rate_limit_rpm == 0 and key.rate_limit_tpm == 0:
            return AuthResult(True, key=key)  # Unlimited

        req_count, token_count = self._db.get_usage_window(key.id, 60)

        if key.rate_limit_rpm > 0 and req_count >= key.rate_limit_rpm:
            return AuthResult(
                False, key=key,
                error=f"Rate limit exceeded: {req_count}/{key.rate_limit_rpm} requests/min",
                status_code=429,
            )

        if key.rate_limit_tpm > 0 and token_count >= key.rate_limit_tpm:
            return AuthResult(
                False, key=key,
                error=f"Token limit exceeded: {token_count}/{key.rate_limit_tpm} tokens/min",
                status_code=429,
            )

        return AuthResult(True, key=key)

    def check_model_access(self, key: APIKey, model: str) -> AuthResult:
        """Check if a key has access to a specific model.

        Args:
            key: Validated API key
            model: Model name to check

        Returns:
            AuthResult — authenticated=True if model is allowed
        """
        if key.allowed_models == "*":
            return AuthResult(True, key=key)

        allowed = [m.strip() for m in key.allowed_models.split(",")]
        if model in allowed:
            return AuthResult(True, key=key)

        return AuthResult(
            False, key=key,
            error=f"Model '{model}' not allowed for this key. Allowed: {key.allowed_models}",
            status_code=403,
        )

    def is_admin(self, authorization: Optional[str]) -> bool:
        """Check if the request is from admin (for admin-only routes)."""
        if not authorization or not self._admin_key:
            return False
        parts = authorization.split(" ", 1)
        if len(parts) != 2:
            return False
        return parts[1].strip() == self._admin_key

    def full_auth_check(
        self, authorization: Optional[str], model: str = ""
    ) -> AuthResult:
        """Full authentication pipeline: auth → rate limit → model access.

        Args:
            authorization: Authorization header value
            model: Model name (optional, skips model check if empty)

        Returns:
            AuthResult — authenticated=True if all checks pass
        """
        # Step 1: Authenticate
        auth = self.authenticate(authorization)
        if not auth.authenticated:
            return auth

        # Step 2: Rate limit
        rate = self.check_rate_limit(auth.key)
        if not rate.authenticated:
            return rate

        # Step 3: Model access
        if model:
            access = self.check_model_access(auth.key, model)
            if not access.authenticated:
                return access

        return AuthResult(True, key=auth.key)
