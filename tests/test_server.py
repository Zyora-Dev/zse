"""ZSE Server — Unit tests.

Tests database, auth, router, SSE, and API endpoints without GPU.
"""

import os
import sys
import json
import time
import tempfile
import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zse-engine'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zse-compiler'))

from zse_engine.server.database import ServerDatabase, APIKey
from zse_engine.server.auth import AuthManager, AuthResult
from zse_engine.server.router import Router, Request, Response, parse_http_request, format_http_response
from zse_engine.server.sse import sse_event, sse_done, sse_chat_chunk, sse_completion_chunk


# ======================================================================
# Database Tests
# ======================================================================

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = ServerDatabase(self.db_path)

    def tearDown(self):
        self.db.close()

    def test_create_key(self):
        plaintext, key = self.db.create_key(name="test-key")
        assert plaintext.startswith("sk-zse-")
        assert len(plaintext) > 20
        assert key.name == "test-key"
        assert key.is_active
        assert key.id > 0

    def test_validate_key(self):
        plaintext, _ = self.db.create_key(name="valid")
        key = self.db.validate_key(plaintext)
        assert key is not None
        assert key.name == "valid"

    def test_validate_invalid_key(self):
        key = self.db.validate_key("sk-zse-invalid")
        assert key is None

    def test_revoke_key(self):
        plaintext, key = self.db.create_key(name="revoke-me")
        assert self.db.revoke_key(key.id)
        assert self.db.validate_key(plaintext) is None

    def test_expired_key(self):
        plaintext, key = self.db.create_key(expires_in_days=-1)  # Already expired
        result = self.db.validate_key(plaintext)
        assert result is None

    def test_list_keys(self):
        self.db.create_key(name="a")
        self.db.create_key(name="b")
        keys = self.db.list_keys()
        assert len(keys) >= 2
        names = [k.name for k in keys]
        assert "a" in names
        assert "b" in names

    def test_key_prefix(self):
        plaintext, key = self.db.create_key()
        assert key.key_prefix == plaintext[:12] + "..."

    def test_record_usage(self):
        _, key = self.db.create_key()
        self.db.record_usage(key.id, prompt_tokens=10, completion_tokens=20, model="test")
        stats = self.db.get_usage_stats(key.id)
        assert stats["total_requests"] == 1
        assert stats["total_prompt_tokens"] == 10
        assert stats["total_completion_tokens"] == 20

    def test_usage_window(self):
        _, key = self.db.create_key()
        self.db.record_usage(key.id, prompt_tokens=5, completion_tokens=10)
        self.db.record_usage(key.id, prompt_tokens=5, completion_tokens=10)
        req_count, token_count = self.db.get_usage_window(key.id, 60)
        assert req_count == 2
        assert token_count == 30

    def test_chat_history(self):
        _, key = self.db.create_key()
        session = "test-session-1"
        self.db.save_message(key.id, session, "user", "Hello")
        self.db.save_message(key.id, session, "assistant", "Hi there!")
        messages = self.db.get_session_messages(session)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].content == "Hi there!"

    def test_list_sessions(self):
        _, key = self.db.create_key()
        self.db.save_message(key.id, "s1", "user", "First chat")
        self.db.save_message(key.id, "s2", "user", "Second chat")
        sessions = self.db.list_sessions(key.id)
        assert len(sessions) == 2

    def test_delete_session(self):
        _, key = self.db.create_key()
        self.db.save_message(key.id, "del-me", "user", "Delete this")
        count = self.db.delete_session("del-me")
        assert count == 1
        assert len(self.db.get_session_messages("del-me")) == 0

    def test_update_key_limits(self):
        _, key = self.db.create_key(rate_limit_rpm=60)
        self.db.update_key_limits(key.id, rate_limit_rpm=120)
        keys = self.db.list_keys()
        updated = [k for k in keys if k.id == key.id][0]
        assert updated.rate_limit_rpm == 120

    def test_key_hash(self):
        h = ServerDatabase.hash_key("test-key")
        assert len(h) == 64  # SHA-256 hex
        assert h == ServerDatabase.hash_key("test-key")  # Deterministic

    def test_generate_key_format(self):
        key = ServerDatabase.generate_key()
        assert key.startswith("sk-zse-")
        assert len(key) == 7 + 64  # "sk-zse-" + 32 bytes hex


# ======================================================================
# Auth Tests
# ======================================================================

class TestAuth(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = ServerDatabase(os.path.join(self.tmpdir, "auth.db"))
        self.auth = AuthManager(self.db, admin_key="admin-secret")

    def tearDown(self):
        self.db.close()

    def test_authenticate_valid(self):
        plaintext, _ = self.db.create_key()
        result = self.auth.authenticate(f"Bearer {plaintext}")
        assert result.authenticated
        assert result.key is not None

    def test_authenticate_invalid(self):
        result = self.auth.authenticate("Bearer invalid-key")
        assert not result.authenticated
        assert result.status_code == 401

    def test_authenticate_missing(self):
        result = self.auth.authenticate(None)
        assert not result.authenticated

    def test_authenticate_bad_format(self):
        result = self.auth.authenticate("Token something")
        assert not result.authenticated

    def test_admin_key(self):
        result = self.auth.authenticate("Bearer admin-secret")
        assert result.authenticated
        assert result.key.key_prefix == "admin"

    def test_is_admin(self):
        assert self.auth.is_admin("Bearer admin-secret")
        assert not self.auth.is_admin("Bearer other")
        assert not self.auth.is_admin(None)

    def test_rate_limit_check(self):
        _, key = self.db.create_key(rate_limit_rpm=2)
        self.db.record_usage(key.id, 10, 10)
        self.db.record_usage(key.id, 10, 10)
        result = self.auth.check_rate_limit(key)
        # Refresh key data to get updated counter
        key_refreshed = self.db.validate_key.__wrapped__ if hasattr(self.db.validate_key, '__wrapped__') else None
        # The rate limit checks usage window, not the key counter
        # After 2 requests with rpm=2, should be at limit
        result = self.auth.check_rate_limit(key)
        assert result.status_code == 429

    def test_model_access_wildcard(self):
        key = APIKey(id=1, key_hash="", key_prefix="", name="",
                     created_at=0, expires_at=None, is_active=True,
                     rate_limit_rpm=0, rate_limit_tpm=0,
                     allowed_models="*", total_requests=0, total_tokens=0)
        result = self.auth.check_model_access(key, "any-model")
        assert result.authenticated

    def test_model_access_restricted(self):
        key = APIKey(id=1, key_hash="", key_prefix="", name="",
                     created_at=0, expires_at=None, is_active=True,
                     rate_limit_rpm=0, rate_limit_tpm=0,
                     allowed_models="model-a,model-b", total_requests=0, total_tokens=0)
        assert self.auth.check_model_access(key, "model-a").authenticated
        assert not self.auth.check_model_access(key, "model-c").authenticated

    def test_full_auth_check(self):
        plaintext, _ = self.db.create_key()
        result = self.auth.full_auth_check(f"Bearer {plaintext}", model="test")
        assert result.authenticated


# ======================================================================
# Router Tests
# ======================================================================

class TestRouter(unittest.TestCase):
    def test_route_matching(self):
        from zse_engine.server.router import Route
        route = Route("GET", "/v1/models", None)
        assert route.match("GET", "/v1/models") is not None
        assert route.match("POST", "/v1/models") is None
        assert route.match("GET", "/v1/other") is None

    def test_path_params(self):
        from zse_engine.server.router import Route
        route = Route("GET", "/v1/session/{session_id}", None)
        params = route.match("GET", "/v1/session/abc123")
        assert params == {"session_id": "abc123"}

    def test_parse_http_request(self):
        raw = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test\r\n\r\n"
        req = parse_http_request(raw)
        assert req is not None
        assert req.method == "GET"
        assert req.path == "/v1/models"
        assert req.authorization == "Bearer test"

    def test_parse_http_post(self):
        body = b'{"model":"test"}'
        raw = f"POST /v1/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
        req = parse_http_request(raw)
        assert req is not None
        assert req.method == "POST"
        assert req.json == {"model": "test"}

    def test_parse_query_params(self):
        raw = b"GET /v1/stats?key_id=5&format=json HTTP/1.1\r\n\r\n"
        req = parse_http_request(raw)
        assert req.query_params == {"key_id": "5", "format": "json"}

    def test_format_response(self):
        resp = Response.json({"status": "ok"})
        raw = format_http_response(resp)
        assert b"HTTP/1.1 200 OK" in raw
        assert b"application/json" in raw
        assert b'"status"' in raw

    def test_error_response(self):
        resp = Response.error("not found", 404)
        data = json.loads(resp.body)
        assert data["error"]["message"] == "not found"
        assert resp.status == 404

    def test_streaming_response(self):
        resp = Response.streaming()
        assert resp._is_streaming
        assert "text/event-stream" in resp.headers["Content-Type"]

    def test_router_handle(self):
        router = Router()

        async def handler(req):
            return Response.json({"hello": "world"})

        router.get("/test", handler)

        req = Request(method="GET", path="/test", headers={})
        resp = asyncio.get_event_loop().run_until_complete(router.handle(req))
        assert resp.status == 200

    def test_router_404(self):
        router = Router()
        req = Request(method="GET", path="/nonexistent", headers={})
        resp = asyncio.get_event_loop().run_until_complete(router.handle(req))
        assert resp.status == 404


# ======================================================================
# SSE Tests
# ======================================================================

class TestSSE(unittest.TestCase):
    def test_sse_event(self):
        data = sse_event('{"test": true}')
        assert data == b'data: {"test": true}\n\n'

    def test_sse_done(self):
        assert sse_done() == b'data: [DONE]\n\n'

    def test_sse_chat_chunk(self):
        chunk = sse_chat_chunk("id-1", "model", "Hello")
        text = chunk.decode()
        assert text.startswith("data: ")
        data = json.loads(text.split("data: ")[1])
        assert data["id"] == "id-1"
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_sse_chat_chunk_with_role(self):
        chunk = sse_chat_chunk("id-1", "model", "Hi", role="assistant")
        data = json.loads(chunk.decode().split("data: ")[1])
        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_sse_chat_chunk_finish(self):
        chunk = sse_chat_chunk("id-1", "model", "", finish_reason="stop")
        data = json.loads(chunk.decode().split("data: ")[1])
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_sse_completion_chunk(self):
        chunk = sse_completion_chunk("id-2", "model", "World")
        data = json.loads(chunk.decode().split("data: ")[1])
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "World"


# ======================================================================
# API OpenAI Tests
# ======================================================================

class TestOpenAIAPI(unittest.TestCase):
    def setUp(self):
        self.mock_engine = MagicMock()
        self.mock_engine.lora_manager = MagicMock()
        self.mock_engine.lora_manager.list_adapters.return_value = ["adapter-1"]
        self.mock_engine._tokenizer = None

        from zse_engine.server.api_openai import OpenAIAPI
        self.api = OpenAIAPI(self.mock_engine, model_name="test-model")

    def test_list_models(self):
        req = Request(method="GET", path="/v1/models", headers={})
        resp = asyncio.get_event_loop().run_until_complete(
            self.api.handle_list_models(req)
        )
        data = json.loads(resp.body)
        assert data["object"] == "list"
        assert len(data["data"]) == 2  # base model + 1 adapter
        assert data["data"][0]["id"] == "test-model"

    def test_chat_completions_missing_messages(self):
        req = Request(method="POST", path="/v1/chat/completions",
                      headers={}, body=b'{"model":"test"}')
        resp = asyncio.get_event_loop().run_until_complete(
            self.api.handle_chat_completions(req)
        )
        assert resp.status == 400

    def test_chat_completions_empty_body(self):
        req = Request(method="POST", path="/v1/chat/completions",
                      headers={}, body=b'')
        resp = asyncio.get_event_loop().run_until_complete(
            self.api.handle_chat_completions(req)
        )
        assert resp.status == 400

    def test_format_chat_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = self.api._format_chat_messages(messages)
        assert "system" in prompt
        assert "user" in prompt
        assert "Hello" in prompt
        assert "assistant" in prompt


# ======================================================================
# Admin API Tests
# ======================================================================

class TestAdminAPI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = ServerDatabase(os.path.join(self.tmpdir, "admin.db"))
        self.auth = AuthManager(self.db, admin_key="admin-key")

        from zse_engine.server.api_admin import AdminAPI
        self.admin = AdminAPI(self.db, self.auth, engine=None)

    def tearDown(self):
        self.db.close()

    def test_create_key_requires_admin(self):
        req = Request(method="POST", path="/v1/admin/keys/create",
                      headers={"authorization": "Bearer wrong"},
                      body=b'{"name":"test"}')
        resp = asyncio.get_event_loop().run_until_complete(
            self.admin.handle_create_key(req)
        )
        assert resp.status == 403

    def test_create_key_success(self):
        req = Request(method="POST", path="/v1/admin/keys/create",
                      headers={"authorization": "Bearer admin-key"},
                      body=b'{"name":"my-key","rate_limit_rpm":100}')
        resp = asyncio.get_event_loop().run_until_complete(
            self.admin.handle_create_key(req)
        )
        assert resp.status == 201
        data = json.loads(resp.body)
        assert data["key"].startswith("sk-zse-")
        assert data["name"] == "my-key"

    def test_list_keys_success(self):
        self.db.create_key(name="k1")
        req = Request(method="GET", path="/v1/admin/keys/list",
                      headers={"authorization": "Bearer admin-key"})
        resp = asyncio.get_event_loop().run_until_complete(
            self.admin.handle_list_keys(req)
        )
        data = json.loads(resp.body)
        assert len(data["keys"]) >= 1

    def test_revoke_key_success(self):
        _, key = self.db.create_key(name="revoke-test")
        req = Request(method="POST", path="/v1/admin/keys/revoke",
                      headers={"authorization": "Bearer admin-key"},
                      body=json.dumps({"id": key.id}).encode())
        resp = asyncio.get_event_loop().run_until_complete(
            self.admin.handle_revoke_key(req)
        )
        assert resp.status == 200

    def test_stats_requires_auth(self):
        req = Request(method="GET", path="/v1/admin/stats", headers={})
        resp = asyncio.get_event_loop().run_until_complete(
            self.admin.handle_stats(req)
        )
        assert resp.status == 401


# ======================================================================
# LoRA API Tests
# ======================================================================

class TestLoRAAPI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db = ServerDatabase(os.path.join(self.tmpdir, "lora.db"))
        self.auth = AuthManager(self.db, admin_key="admin-key")

        self.mock_engine = MagicMock()
        self.mock_lora_manager = MagicMock()
        self.mock_lora_manager.list_adapters.return_value = []
        self.mock_engine.lora_manager = self.mock_lora_manager

        from zse_engine.server.api_lora import LoRAAPI
        self.lora_api = LoRAAPI(self.auth, engine=self.mock_engine)

    def tearDown(self):
        self.db.close()

    def test_list_requires_auth(self):
        req = Request(method="GET", path="/v1/lora/list", headers={})
        resp = asyncio.get_event_loop().run_until_complete(
            self.lora_api.handle_list(req)
        )
        assert resp.status == 401

    def test_list_empty(self):
        plaintext, _ = self.db.create_key()
        req = Request(method="GET", path="/v1/lora/list",
                      headers={"authorization": f"Bearer {plaintext}"})
        resp = asyncio.get_event_loop().run_until_complete(
            self.lora_api.handle_list(req)
        )
        assert resp.status == 200
        data = json.loads(resp.body)
        assert data["adapters"] == []

    def test_load_requires_admin(self):
        plaintext, _ = self.db.create_key()
        req = Request(method="POST", path="/v1/lora/load",
                      headers={"authorization": f"Bearer {plaintext}"},
                      body=b'{"adapter_id":"test","path":"/tmp/a.zse-lora"}')
        resp = asyncio.get_event_loop().run_until_complete(
            self.lora_api.handle_load(req)
        )
        assert resp.status == 403

    def test_unload_requires_admin(self):
        plaintext, _ = self.db.create_key()
        req = Request(method="POST", path="/v1/lora/unload",
                      headers={"authorization": f"Bearer {plaintext}"},
                      body=b'{"adapter_id":"test"}')
        resp = asyncio.get_event_loop().run_until_complete(
            self.lora_api.handle_unload(req)
        )
        assert resp.status == 403


# ======================================================================
# Integration: Server Setup
# ======================================================================

class TestServerSetup(unittest.TestCase):
    def test_server_creates_without_model(self):
        """Server can start in test/mock mode without a model."""
        from zse_engine.server.app import ZSEServer
        tmpdir = tempfile.mkdtemp()
        server = ZSEServer(
            model_path=None,
            port=0,
            admin_key="test-admin",
            db_path=os.path.join(tmpdir, "test.db"),
            quiet=True,
        )
        assert server._engine is None
        assert server._db is not None
        assert server._auth is not None
        server.destroy()

    def test_static_dir_exists(self):
        static_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'zse-engine', 'zse_engine', 'server', 'static'
        )
        assert os.path.isdir(static_dir)

    def test_index_html_exists(self):
        index = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'zse-engine', 'zse_engine', 'server', 'static', 'index.html'
        )
        assert os.path.isfile(index)


if __name__ == "__main__":
    unittest.main()
