"""ZSE Server Router — URL routing, middleware, and HTTP helpers.

Pure Python HTTP request/response handling. No frameworks, no dependencies.
"""

import json
import os
import mimetypes
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, Any, Awaitable, Tuple


@dataclass
class Request:
    """Parsed HTTP request."""
    method: str
    path: str
    headers: Dict[str, str]
    body: bytes = b""
    query_params: Dict[str, str] = field(default_factory=dict)
    # Set by auth middleware
    api_key: Any = None  # APIKey dataclass
    path_params: Dict[str, str] = field(default_factory=dict)

    @property
    def json(self) -> dict:
        """Parse body as JSON."""
        if not self.body:
            return {}
        return json.loads(self.body.decode("utf-8"))

    @property
    def authorization(self) -> Optional[str]:
        return self.headers.get("authorization")

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")


@dataclass
class Response:
    """HTTP response to send."""
    status: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    _is_streaming: bool = False

    @staticmethod
    def json(data: dict, status: int = 200) -> "Response":
        """Create a JSON response."""
        body = json.dumps(data).encode("utf-8")
        return Response(
            status=status,
            headers={"Content-Type": "application/json"},
            body=body,
        )

    @staticmethod
    def error(message: str, status: int = 400, error_type: str = "invalid_request_error") -> "Response":
        """Create an OpenAI-compatible error response."""
        return Response.json({
            "error": {
                "message": message,
                "type": error_type,
                "code": status,
            }
        }, status=status)

    @staticmethod
    def text(content: str, status: int = 200) -> "Response":
        return Response(
            status=status,
            headers={"Content-Type": "text/plain"},
            body=content.encode("utf-8"),
        )

    @staticmethod
    def html(content: str, status: int = 200) -> "Response":
        return Response(
            status=status,
            headers={"Content-Type": "text/html; charset=utf-8"},
            body=content.encode("utf-8"),
        )

    @staticmethod
    def streaming() -> "Response":
        """Create a streaming response (headers only, body sent later)."""
        return Response(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
            _is_streaming=True,
        )


# Handler type: async function that takes Request and returns Response
Handler = Callable[["Request"], Awaitable[Response]]
# Middleware type: wraps a handler
Middleware = Callable[[Handler], Handler]


class Route:
    """A registered URL route."""

    def __init__(self, method: str, path: str, handler: Handler):
        self.method = method.upper()
        self.path = path
        self.handler = handler
        # Parse path segments for matching
        self._segments = path.strip("/").split("/") if path != "/" else []

    def match(self, method: str, path: str) -> Optional[Dict[str, str]]:
        """Check if method+path matches this route. Returns path params or None."""
        if self.method != method.upper() and self.method != "*":
            return None

        segments = path.strip("/").split("/") if path != "/" else []

        if len(segments) != len(self._segments):
            return None

        params = {}
        for route_seg, req_seg in zip(self._segments, segments):
            if route_seg.startswith("{") and route_seg.endswith("}"):
                params[route_seg[1:-1]] = req_seg
            elif route_seg != req_seg:
                return None

        return params


class Router:
    """URL router with middleware support.

    Usage:
        router = Router()
        router.add("GET", "/v1/models", handle_models)
        router.add("POST", "/v1/chat/completions", handle_chat)

        # Add middleware
        router.use(cors_middleware)
        router.use(auth_middleware)

        # Route a request
        response = await router.handle(request)
    """

    def __init__(self):
        self._routes: List[Route] = []
        self._middlewares: List[Middleware] = []
        self._static_dir: Optional[str] = None

    def add(self, method: str, path: str, handler: Handler):
        """Register a route."""
        self._routes.append(Route(method, path, handler))

    def get(self, path: str, handler: Handler):
        self.add("GET", path, handler)

    def post(self, path: str, handler: Handler):
        self.add("POST", path, handler)

    def delete(self, path: str, handler: Handler):
        self.add("DELETE", path, handler)

    def put(self, path: str, handler: Handler):
        self.add("PUT", path, handler)

    def use(self, middleware: Middleware):
        """Add middleware (applied in order: first added = outermost)."""
        self._middlewares.append(middleware)

    def set_static_dir(self, path: str):
        """Set directory for serving static files (dashboard)."""
        self._static_dir = path

    async def handle(self, request: Request) -> Response:
        """Route a request to its handler."""
        # Strip query string from path
        path = request.path.split("?")[0]

        # Try static files first for non-API paths
        if self._static_dir and not path.startswith("/v1/"):
            static_resp = self._serve_static(path)
            if static_resp:
                return static_resp

        # Find matching route
        for route in self._routes:
            params = route.match(request.method, path)
            if params is not None:
                request.path_params = params
                # Build handler chain with middleware
                handler = route.handler
                for mw in reversed(self._middlewares):
                    handler = mw(handler)
                try:
                    return await handler(request)
                except json.JSONDecodeError:
                    return Response.error("Invalid JSON in request body", 400)
                except Exception as e:
                    return Response.error(f"Internal server error: {str(e)}", 500)

        # 404
        return Response.error(f"Not found: {request.method} {path}", 404)

    def _serve_static(self, path: str) -> Optional[Response]:
        """Serve a static file from the static directory."""
        if not self._static_dir:
            return None

        # Default to index.html
        if path == "/" or path == "":
            path = "/index.html"

        # Strip /static/ prefix if present (HTML references /static/foo.css)
        if path.startswith("/static/"):
            path = path[7:]  # "/static/foo.css" -> "/foo.css"

        # Security: prevent path traversal
        clean = os.path.normpath(path.lstrip("/"))
        if clean.startswith(".."):
            return None

        filepath = os.path.join(self._static_dir, clean)
        if not os.path.isfile(filepath):
            return None

        # Read file
        with open(filepath, "rb") as f:
            content = f.read()

        # Determine content type
        mime, _ = mimetypes.guess_type(filepath)
        if mime is None:
            mime = "application/octet-stream"

        return Response(
            status=200,
            headers={
                "Content-Type": mime,
                "Cache-Control": "no-cache",  # Dev-friendly
            },
            body=content,
        )


# ------------------------------------------------------------------
# HTTP Parsing Helpers
# ------------------------------------------------------------------

def parse_http_request(raw: bytes) -> Optional[Request]:
    """Parse raw HTTP/1.1 bytes into a Request.

    Returns None if the request is incomplete.
    """
    # Split headers from body
    header_end = raw.find(b"\r\n\r\n")
    if header_end == -1:
        return None

    header_bytes = raw[:header_end]
    body = raw[header_end + 4:]

    # Parse request line
    lines = header_bytes.decode("utf-8", errors="replace").split("\r\n")
    if not lines:
        return None

    parts = lines[0].split(" ", 2)
    if len(parts) < 2:
        return None

    method = parts[0]
    full_path = parts[1]

    # Parse query string
    query_params = {}
    if "?" in full_path:
        path, query_string = full_path.split("?", 1)
        for param in query_string.split("&"):
            if "=" in param:
                k, v = param.split("=", 1)
                query_params[k] = v
    else:
        path = full_path

    # Parse headers
    headers = {}
    for line in lines[1:]:
        if ": " in line:
            key, value = line.split(": ", 1)
            headers[key.lower()] = value

    # Check Content-Length for body completeness
    content_length = int(headers.get("content-length", "0"))
    if len(body) < content_length:
        return None  # Incomplete body

    body = body[:content_length]

    return Request(
        method=method,
        path=path,
        headers=headers,
        body=body,
        query_params=query_params,
    )


def format_http_response(response: Response) -> bytes:
    """Format a Response into raw HTTP/1.1 bytes."""
    status_text = HTTP_STATUS_TEXT.get(response.status, "Unknown")
    lines = [f"HTTP/1.1 {response.status} {status_text}"]

    # Add headers
    headers = dict(response.headers)
    if not response._is_streaming:
        headers["Content-Length"] = str(len(response.body))
    headers.setdefault("Access-Control-Allow-Origin", "*")
    headers.setdefault("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
    headers.setdefault("Access-Control-Allow-Headers", "Content-Type, Authorization")

    for key, value in headers.items():
        lines.append(f"{key}: {value}")

    header_bytes = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
    if response._is_streaming:
        return header_bytes  # Body sent separately for SSE
    return header_bytes + response.body


HTTP_STATUS_TEXT = {
    200: "OK",
    201: "Created",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    429: "Too Many Requests",
    500: "Internal Server Error",
}
