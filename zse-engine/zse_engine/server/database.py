"""ZSE Server Database — SQLite storage for API keys, usage, and chat history.

Zero dependencies — uses Python stdlib sqlite3.

Tables:
    api_keys     — API key management (hashed keys, rate limits, expiration)
    usage        — Per-request token usage tracking
    chat_history — Chat message history per session
"""

import sqlite3
import hashlib
import secrets
import time
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path


@dataclass
class APIKey:
    """API key record."""
    id: int
    key_hash: str
    name: str
    created_at: float
    expires_at: Optional[float]
    is_active: bool
    rate_limit_rpm: int       # Requests per minute (0 = unlimited)
    rate_limit_tpm: int       # Tokens per minute (0 = unlimited)
    allowed_models: str       # Comma-separated model names ("*" = all)
    key_prefix: str           # First 8 chars for display (sk-zse-xxxx...)
    total_requests: int
    total_tokens: int


@dataclass
class UsageRecord:
    """Token usage for a single request."""
    id: int
    key_id: int
    timestamp: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    lora_id: Optional[str]
    request_id: str
    latency_ms: float


@dataclass
class ChatMessage:
    """A single chat message."""
    id: int
    key_id: int
    session_id: str
    role: str               # "system", "user", "assistant"
    content: str
    model: str
    lora_id: Optional[str]
    created_at: float


class ServerDatabase:
    """SQLite database for ZSE Server.

    Thread-safe via sqlite3's built-in serialization (check_same_thread=False).

    Args:
        db_path: Path to SQLite database file. Creates if not exists.
    """

    def __init__(self, db_path: str = "~/.zse/server.db"):
        self._db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode
        )
        self._conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent reads
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_hash TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                name TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                expires_at REAL,
                is_active INTEGER NOT NULL DEFAULT 1,
                rate_limit_rpm INTEGER NOT NULL DEFAULT 60,
                rate_limit_tpm INTEGER NOT NULL DEFAULT 100000,
                allowed_models TEXT NOT NULL DEFAULT '*',
                total_requests INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT '',
                lora_id TEXT,
                request_id TEXT NOT NULL DEFAULT '',
                latency_ms REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (key_id) REFERENCES api_keys(id)
            );

            CREATE INDEX IF NOT EXISTS idx_usage_key_ts
                ON usage(key_id, timestamp);

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                lora_id TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (key_id) REFERENCES api_keys(id)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_session
                ON chat_history(session_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_chat_key
                ON chat_history(key_id, created_at);
        """)

    # ------------------------------------------------------------------
    # API Key Management
    # ------------------------------------------------------------------

    @staticmethod
    def hash_key(key: str) -> str:
        """SHA-256 hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new API key: sk-zse-{64 hex chars}."""
        return f"sk-zse-{secrets.token_hex(32)}"

    def create_key(
        self,
        name: str = "",
        rate_limit_rpm: int = 60,
        rate_limit_tpm: int = 100000,
        allowed_models: str = "*",
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key.

        Returns:
            (plaintext_key, APIKey record). Plaintext is shown ONCE.
        """
        plaintext = self.generate_key()
        key_hash = self.hash_key(plaintext)
        key_prefix = plaintext[:12] + "..."
        now = time.time()
        expires_at = now + (expires_in_days * 86400) if expires_in_days else None

        cursor = self._conn.execute(
            """INSERT INTO api_keys
               (key_hash, key_prefix, name, created_at, expires_at,
                is_active, rate_limit_rpm, rate_limit_tpm, allowed_models)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)""",
            (key_hash, key_prefix, name, now, expires_at,
             rate_limit_rpm, rate_limit_tpm, allowed_models),
        )
        key_record = APIKey(
            id=cursor.lastrowid,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            created_at=now,
            expires_at=expires_at,
            is_active=True,
            rate_limit_rpm=rate_limit_rpm,
            rate_limit_tpm=rate_limit_tpm,
            allowed_models=allowed_models,
            total_requests=0,
            total_tokens=0,
        )
        return plaintext, key_record

    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key. Returns APIKey if valid, None if invalid.

        Checks: exists, is_active, not expired.
        """
        key_hash = self.hash_key(key)
        row = self._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()

        if row is None:
            return None

        api_key = self._row_to_api_key(row)

        if not api_key.is_active:
            return None

        if api_key.expires_at and time.time() > api_key.expires_at:
            return None

        return api_key

    def list_keys(self) -> List[APIKey]:
        """List all API keys (for admin panel)."""
        rows = self._conn.execute(
            "SELECT * FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_api_key(row) for row in rows]

    def revoke_key(self, key_id: int) -> bool:
        """Revoke an API key by ID."""
        cursor = self._conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,)
        )
        return cursor.rowcount > 0

    def revoke_key_by_prefix(self, prefix: str) -> bool:
        """Revoke an API key by its prefix."""
        cursor = self._conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE key_prefix = ?", (prefix,)
        )
        return cursor.rowcount > 0

    def update_key_limits(
        self, key_id: int,
        rate_limit_rpm: Optional[int] = None,
        rate_limit_tpm: Optional[int] = None,
        allowed_models: Optional[str] = None,
    ) -> bool:
        """Update rate limits for a key."""
        updates = []
        params = []
        if rate_limit_rpm is not None:
            updates.append("rate_limit_rpm = ?")
            params.append(rate_limit_rpm)
        if rate_limit_tpm is not None:
            updates.append("rate_limit_tpm = ?")
            params.append(rate_limit_tpm)
        if allowed_models is not None:
            updates.append("allowed_models = ?")
            params.append(allowed_models)
        if not updates:
            return False
        params.append(key_id)
        cursor = self._conn.execute(
            f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ?", params
        )
        return cursor.rowcount > 0

    def _row_to_api_key(self, row) -> APIKey:
        return APIKey(
            id=row[0], key_hash=row[1], key_prefix=row[2], name=row[3],
            created_at=row[4], expires_at=row[5], is_active=bool(row[6]),
            rate_limit_rpm=row[7], rate_limit_tpm=row[8],
            allowed_models=row[9], total_requests=row[10], total_tokens=row[11],
        )

    # ------------------------------------------------------------------
    # Usage Tracking
    # ------------------------------------------------------------------

    def record_usage(
        self,
        key_id: int,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
        lora_id: Optional[str] = None,
        request_id: str = "",
        latency_ms: float = 0,
    ):
        """Record token usage for a request."""
        total = prompt_tokens + completion_tokens
        now = time.time()
        self._conn.execute(
            """INSERT INTO usage
               (key_id, timestamp, prompt_tokens, completion_tokens,
                total_tokens, model, lora_id, request_id, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (key_id, now, prompt_tokens, completion_tokens,
             total, model, lora_id, request_id, latency_ms),
        )
        # Update aggregate counters on api_keys
        self._conn.execute(
            """UPDATE api_keys
               SET total_requests = total_requests + 1,
                   total_tokens = total_tokens + ?
               WHERE id = ?""",
            (total, key_id),
        )

    def get_usage_window(
        self, key_id: int, window_seconds: float = 60
    ) -> Tuple[int, int]:
        """Get request count and token count in a sliding window.

        Returns:
            (request_count, token_count) in the last window_seconds.
        """
        cutoff = time.time() - window_seconds
        row = self._conn.execute(
            """SELECT COUNT(*), COALESCE(SUM(total_tokens), 0)
               FROM usage WHERE key_id = ? AND timestamp > ?""",
            (key_id, cutoff),
        ).fetchone()
        return (row[0], row[1])

    def get_usage_stats(self, key_id: Optional[int] = None) -> Dict:
        """Get usage statistics."""
        if key_id:
            rows = self._conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0),
                          COALESCE(SUM(completion_tokens), 0),
                          COALESCE(SUM(total_tokens), 0),
                          COALESCE(AVG(latency_ms), 0)
                   FROM usage WHERE key_id = ?""",
                (key_id,),
            ).fetchone()
        else:
            rows = self._conn.execute(
                """SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0),
                          COALESCE(SUM(completion_tokens), 0),
                          COALESCE(SUM(total_tokens), 0),
                          COALESCE(AVG(latency_ms), 0)
                   FROM usage""",
            ).fetchone()
        return {
            "total_requests": rows[0],
            "total_prompt_tokens": rows[1],
            "total_completion_tokens": rows[2],
            "total_tokens": rows[3],
            "avg_latency_ms": round(rows[4], 2),
        }

    # ------------------------------------------------------------------
    # Chat History
    # ------------------------------------------------------------------

    def save_message(
        self,
        key_id: int,
        session_id: str,
        role: str,
        content: str,
        model: str = "",
        lora_id: Optional[str] = None,
    ):
        """Save a chat message."""
        self._conn.execute(
            """INSERT INTO chat_history
               (key_id, session_id, role, content, model, lora_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (key_id, session_id, role, content, model, lora_id, time.time()),
        )

    def get_session_messages(
        self, session_id: str, limit: int = 100
    ) -> List[ChatMessage]:
        """Get messages for a chat session."""
        rows = self._conn.execute(
            """SELECT * FROM chat_history
               WHERE session_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [self._row_to_chat_message(row) for row in rows]

    def list_sessions(
        self, key_id: int, limit: int = 50
    ) -> List[Dict]:
        """List chat sessions for a key (most recent first)."""
        rows = self._conn.execute(
            """SELECT session_id,
                      MIN(created_at) as started,
                      MAX(created_at) as last_msg,
                      COUNT(*) as msg_count,
                      (SELECT content FROM chat_history ch2
                       WHERE ch2.session_id = ch.session_id AND ch2.role = 'user'
                       ORDER BY created_at ASC LIMIT 1) as first_user_msg
               FROM chat_history ch
               WHERE key_id = ?
               GROUP BY session_id
               ORDER BY last_msg DESC
               LIMIT ?""",
            (key_id, limit),
        ).fetchall()
        return [
            {
                "session_id": r[0],
                "started_at": r[1],
                "last_message_at": r[2],
                "message_count": r[3],
                "preview": (r[4] or "")[:80],
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> int:
        """Delete all messages in a session. Returns count deleted."""
        cursor = self._conn.execute(
            "DELETE FROM chat_history WHERE session_id = ?", (session_id,)
        )
        return cursor.rowcount

    def _row_to_chat_message(self, row) -> ChatMessage:
        return ChatMessage(
            id=row[0], key_id=row[1], session_id=row[2], role=row[3],
            content=row[4], model=row[5], lora_id=row[6], created_at=row[7],
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_old_usage(self, days: int = 30):
        """Delete usage records older than N days."""
        cutoff = time.time() - (days * 86400)
        self._conn.execute("DELETE FROM usage WHERE timestamp < ?", (cutoff,))

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()
