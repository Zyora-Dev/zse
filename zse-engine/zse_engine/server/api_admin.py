"""ZSE Server Admin API — API key management and server statistics.

Endpoints:
    POST /v1/admin/keys/create   — Create new API key
    GET  /v1/admin/keys/list     — List all API keys
    POST /v1/admin/keys/revoke   — Revoke an API key
    POST /v1/admin/keys/update   — Update key rate limits
    GET  /v1/admin/stats         — Server statistics
    GET  /v1/admin/usage         — Usage statistics
"""

import time
from typing import Optional

from zse_engine.server.router import Request, Response
from zse_engine.server.database import ServerDatabase
from zse_engine.server.auth import AuthManager


class AdminAPI:
    """Admin endpoint handlers.

    All admin routes require the admin key (set at server startup).

    Args:
        db: ServerDatabase instance
        auth: AuthManager instance
        engine: ZStreamerEngine instance (for stats)
    """

    def __init__(self, db: ServerDatabase, auth: AuthManager, engine=None):
        self._db = db
        self._auth = auth
        self._engine = engine

    def register(self, router):
        """Register admin routes."""
        router.post("/v1/admin/keys/create", self.handle_create_key)
        router.get("/v1/admin/keys/list", self.handle_list_keys)
        router.post("/v1/admin/keys/revoke", self.handle_revoke_key)
        router.post("/v1/admin/keys/update", self.handle_update_key)
        router.get("/v1/admin/stats", self.handle_stats)
        router.get("/v1/admin/usage", self.handle_usage)

        # Dashboard API endpoints (authenticated via any valid key or admin)
        router.get("/v1/dashboard/sessions", self.handle_list_sessions)
        router.get("/v1/dashboard/session/{session_id}", self.handle_get_session)
        router.delete("/v1/dashboard/session/{session_id}", self.handle_delete_session)
        router.post("/v1/dashboard/session/save", self.handle_save_message)

    def _require_admin(self, request: Request) -> Optional[Response]:
        """Check admin auth. Returns error Response if not admin, None if OK."""
        if not self._auth.is_admin(request.authorization):
            return Response.error("Admin access required", 403)
        return None

    # ------------------------------------------------------------------
    # Key Management
    # ------------------------------------------------------------------

    async def handle_create_key(self, request: Request) -> Response:
        """Create a new API key."""
        err = self._require_admin(request)
        if err:
            return err

        body = request.json
        name = body.get("name", "")
        rate_limit_rpm = int(body.get("rate_limit_rpm", 60))
        rate_limit_tpm = int(body.get("rate_limit_tpm", 100000))
        allowed_models = body.get("allowed_models", "*")
        expires_in_days = body.get("expires_in_days")

        plaintext, key_record = self._db.create_key(
            name=name,
            rate_limit_rpm=rate_limit_rpm,
            rate_limit_tpm=rate_limit_tpm,
            allowed_models=allowed_models,
            expires_in_days=expires_in_days,
        )

        return Response.json({
            "key": plaintext,  # Shown ONCE
            "id": key_record.id,
            "prefix": key_record.key_prefix,
            "name": key_record.name,
            "created_at": key_record.created_at,
            "expires_at": key_record.expires_at,
            "rate_limit_rpm": key_record.rate_limit_rpm,
            "rate_limit_tpm": key_record.rate_limit_tpm,
            "allowed_models": key_record.allowed_models,
        }, status=201)

    async def handle_list_keys(self, request: Request) -> Response:
        """List all API keys (masked)."""
        err = self._require_admin(request)
        if err:
            return err

        keys = self._db.list_keys()
        return Response.json({
            "keys": [{
                "id": k.id,
                "prefix": k.key_prefix,
                "name": k.name,
                "created_at": k.created_at,
                "expires_at": k.expires_at,
                "is_active": k.is_active,
                "rate_limit_rpm": k.rate_limit_rpm,
                "rate_limit_tpm": k.rate_limit_tpm,
                "allowed_models": k.allowed_models,
                "total_requests": k.total_requests,
                "total_tokens": k.total_tokens,
            } for k in keys],
        })

    async def handle_revoke_key(self, request: Request) -> Response:
        """Revoke an API key."""
        err = self._require_admin(request)
        if err:
            return err

        body = request.json
        key_id = body.get("id")
        prefix = body.get("prefix")

        if key_id:
            success = self._db.revoke_key(int(key_id))
        elif prefix:
            success = self._db.revoke_key_by_prefix(prefix)
        else:
            return Response.error("Provide 'id' or 'prefix' to revoke")

        if success:
            return Response.json({"status": "revoked"})
        return Response.error("Key not found", 404)

    async def handle_update_key(self, request: Request) -> Response:
        """Update key rate limits."""
        err = self._require_admin(request)
        if err:
            return err

        body = request.json
        key_id = body.get("id")
        if not key_id:
            return Response.error("'id' field required")

        success = self._db.update_key_limits(
            int(key_id),
            rate_limit_rpm=body.get("rate_limit_rpm"),
            rate_limit_tpm=body.get("rate_limit_tpm"),
            allowed_models=body.get("allowed_models"),
        )

        if success:
            return Response.json({"status": "updated"})
        return Response.error("Key not found or no updates", 404)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def handle_stats(self, request: Request) -> Response:
        """Server statistics."""
        # Auth: allow admin or any valid key
        auth = self._auth.authenticate(request.authorization)
        if not auth.authenticated:
            return Response.error(auth.error, auth.status_code)

        stats = {}

        # Engine stats
        if self._engine:
            try:
                engine_stats = self._engine.stats()
                stats["engine"] = {
                    "total_requests": engine_stats.total_requests,
                    "total_tokens": engine_stats.total_tokens_generated,
                    "tokens_per_sec": round(engine_stats.tokens_per_sec, 1),
                    "requests_per_sec": round(engine_stats.requests_per_sec, 2),
                    "avg_batch_size": round(engine_stats.avg_batch_size, 1),
                    "avg_ttft_ms": round(engine_stats.avg_ttft_ms, 1),
                    "avg_tpot_ms": round(engine_stats.avg_tpot_ms, 1),
                    "p99_ttft_ms": round(engine_stats.p99_ttft_ms, 1),
                    "queue_depth": engine_stats.queue_depth,
                    "active_requests": engine_stats.active_requests,
                    "memory_utilization": round(engine_stats.memory_utilization, 3),
                    "uptime_s": round(engine_stats.uptime_s, 1),
                }
            except Exception:
                stats["engine"] = {"status": "unavailable"}

        # LoRA stats
        if self._engine and self._engine.lora_manager:
            lora_stats = self._engine.lora_manager.stats()
            stats["lora"] = lora_stats

        # Database usage stats
        stats["usage"] = self._db.get_usage_stats()

        # Key count
        keys = self._db.list_keys()
        stats["api_keys"] = {
            "total": len(keys),
            "active": sum(1 for k in keys if k.is_active),
        }

        return Response.json(stats)

    async def handle_usage(self, request: Request) -> Response:
        """Usage statistics (admin only)."""
        err = self._require_admin(request)
        if err:
            return err

        key_id = request.query_params.get("key_id")
        if key_id:
            stats = self._db.get_usage_stats(int(key_id))
        else:
            stats = self._db.get_usage_stats()

        return Response.json(stats)

    # ------------------------------------------------------------------
    # Dashboard Session Endpoints
    # ------------------------------------------------------------------

    async def handle_list_sessions(self, request: Request) -> Response:
        """List chat sessions for the authenticated user."""
        auth = self._auth.authenticate(request.authorization)
        if not auth.authenticated:
            return Response.error(auth.error, auth.status_code)

        sessions = self._db.list_sessions(auth.key.id)
        return Response.json({"sessions": sessions})

    async def handle_get_session(self, request: Request) -> Response:
        """Get messages for a chat session."""
        auth = self._auth.authenticate(request.authorization)
        if not auth.authenticated:
            return Response.error(auth.error, auth.status_code)

        session_id = request.path_params.get("session_id", "")
        messages = self._db.get_session_messages(session_id)
        return Response.json({
            "session_id": session_id,
            "messages": [{
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at,
            } for m in messages],
        })

    async def handle_delete_session(self, request: Request) -> Response:
        """Delete a chat session."""
        auth = self._auth.authenticate(request.authorization)
        if not auth.authenticated:
            return Response.error(auth.error, auth.status_code)

        session_id = request.path_params.get("session_id", "")
        count = self._db.delete_session(session_id)
        return Response.json({"deleted": count})

    async def handle_save_message(self, request: Request) -> Response:
        """Save a chat message to a session."""
        auth = self._auth.authenticate(request.authorization)
        if not auth.authenticated:
            return Response.error(auth.error, auth.status_code)

        body = request.json
        session_id = body.get("session_id", "")
        role = body.get("role", "")
        content = body.get("content", "")
        model = body.get("model", "")
        lora_id = body.get("lora_id")

        if not session_id or not role or not content:
            return Response.error("session_id, role, and content are required")

        self._db.save_message(
            key_id=auth.key.id,
            session_id=session_id,
            role=role,
            content=content,
            model=model,
            lora_id=lora_id,
        )
        return Response.json({"ok": True})
