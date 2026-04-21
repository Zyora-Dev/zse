"""ZSE Server SSE — Server-Sent Events for streaming responses.

Implements OpenAI-compatible SSE format:
    data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"token"}}]}\n\n
    data: [DONE]\n\n
"""

import json
import time
from typing import Optional


def sse_event(data: str) -> bytes:
    """Format a single SSE event."""
    return f"data: {data}\n\n".encode("utf-8")


def sse_done() -> bytes:
    """Format the SSE termination signal."""
    return b"data: [DONE]\n\n"


def sse_chat_chunk(
    completion_id: str,
    model: str,
    token_text: str,
    finish_reason: Optional[str] = None,
    role: Optional[str] = None,
) -> bytes:
    """Format a streaming chat completion chunk (OpenAI-compatible).

    Args:
        completion_id: Unique completion ID (chatcmpl-...)
        model: Model name
        token_text: Decoded token text (empty string for finish chunk)
        finish_reason: "stop", "length", or None (still generating)
        role: Set to "assistant" on first chunk only
    """
    delta = {}
    if role:
        delta["role"] = role
    if token_text:
        delta["content"] = token_text

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return sse_event(json.dumps(chunk))


def sse_completion_chunk(
    completion_id: str,
    model: str,
    token_text: str,
    finish_reason: Optional[str] = None,
) -> bytes:
    """Format a streaming text completion chunk (OpenAI-compatible)."""
    chunk = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": token_text,
            "finish_reason": finish_reason,
        }],
    }
    return sse_event(json.dumps(chunk))


SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # Disable nginx buffering
}
