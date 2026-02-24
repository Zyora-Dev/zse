"""
ZSE API Server

FastAPI-based REST server with OpenAI-compatible API.

Endpoints:
- POST /v1/chat/completions: Chat completion
- POST /v1/completions: Text completion
- GET /v1/models: List available models
- GET /health: Health check
- GET /api/stats: System statistics
- GET /api/analytics: Request analytics
- WS /ws/chat: WebSocket chat
- WS /ws/stats: Real-time stats stream
- GET /dashboard: Monitoring dashboard

Features:
- Streaming responses (SSE)
- WebSocket support
- Real-time monitoring
- Analytics tracking

Usage:
    from zse.api.server import create_app, run_server
    
    # Run server
    run_server(host="0.0.0.0", port=8000)
    
    # Or get app for custom deployment
    app = create_app()
"""

from zse.api.server.app import create_app, run_server, app
from zse.api.server.state import server_state, ServerState
from zse.api.server.models import (
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    LoadModelRequest, LoadModelResponse,
    HealthResponse, SystemStats, AnalyticsOverview
)
from zse.api.server.audit import (
    AuditLogger, AuditLogEntry, AuditMiddleware,
    get_audit_logger, configure_audit_logger, add_audit_middleware
)

__all__ = [
    "create_app",
    "run_server", 
    "app",
    "server_state",
    "ServerState",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "CompletionRequest",
    "CompletionResponse",
    "LoadModelRequest",
    "LoadModelResponse",
    "HealthResponse",
    "SystemStats",
    "AnalyticsOverview",
    # Audit
    "AuditLogger",
    "AuditLogEntry", 
    "AuditMiddleware",
    "get_audit_logger",
    "configure_audit_logger",
    "add_audit_middleware",
]

