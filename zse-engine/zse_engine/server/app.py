"""ZSE Server — Async HTTP server for LLM inference.

Pure Python HTTP/1.1 server using asyncio. Zero dependencies.

Usage:
    python -m zse_engine.server --model model.zse --port 8000 --admin-key mysecret
"""

import asyncio
import os
import signal
import time
import threading
from typing import Optional

from zse_engine.server.database import ServerDatabase
from zse_engine.server.auth import AuthManager
from zse_engine.server.router import Router, Request, Response, parse_http_request, format_http_response
from zse_engine.server.api_openai import OpenAIAPI
from zse_engine.server.api_admin import AdminAPI
from zse_engine.server.api_lora import LoRAAPI
from zse_engine.server.api_rag import RAGAPI
from zse_engine.server.sse import sse_chat_chunk, sse_completion_chunk, sse_done


class ZSEServer:
    """ZSE HTTP inference server.

    Wraps ZStreamerEngine with an OpenAI-compatible HTTP API,
    admin key management, and a web dashboard.

    Args:
        model_path: Path to .zse model file (None for test/mock mode)
        host: Bind address
        port: Bind port
        admin_key: Master admin key for key management
        db_path: Path to SQLite database
        model_name: Model name to report in /v1/models
        quiet: Suppress output
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        admin_key: Optional[str] = None,
        db_path: str = "~/.zse/server.db",
        model_name: str = "zse-model",
        quiet: bool = False,
        tp_size: int = 1,
    ):
        self._host = host
        self._port = port
        self._quiet = quiet
        self._model_name = model_name
        self._tp_size = tp_size
        self._server = None
        self._engine = None
        self._engine_thread = None

        # Database
        self._db = ServerDatabase(db_path)

        # Auth
        self._auth = AuthManager(self._db, admin_key=admin_key)

        # Load engine if model provided
        if model_path:
            self._init_engine(model_path, tp_size=tp_size)

        # Router
        self._router = Router()
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        self._router.set_static_dir(static_dir)

        # Register API endpoints
        tokenizer = self._engine._tokenizer if self._engine else None
        openai_api = OpenAIAPI(self._engine, model_name=model_name, tokenizer=tokenizer)
        admin_api = AdminAPI(self._db, self._auth, engine=self._engine)
        lora_api = LoRAAPI(self._auth, engine=self._engine)

        openai_api.register(self._router)
        admin_api.register(self._router)
        lora_api.register(self._router)

        # RAG engine
        from zse_engine.rag.store import RAGStore
        from zse_engine.rag.engine import RAGEngine
        rag_store = RAGStore(self._db)
        self._rag_engine = RAGEngine(
            store=rag_store,
            tokenizer=tokenizer,
        )
        # Wire the inference LLM into RAG for dense embeddings + LLM reranking.
        # Zero extra deps / VRAM \u2014 reuses the model already loaded for serving.
        try:
            model_runner = getattr(self._engine, "_model_runner", None)
            if model_runner is not None:
                self._rag_engine.set_model_runner(model_runner)
        except Exception:
            pass
        rag_api = RAGAPI(self._auth, rag_engine=self._rag_engine)
        rag_api.register(self._router)

        # Pass RAG engine to OpenAI API for chat augmentation
        openai_api._rag_engine = self._rag_engine

        # Health check (no auth)
        self._router.get("/health", self._handle_health)
        self._router.get("/v1/health", self._handle_health)

        # CORS preflight
        self._router.add("OPTIONS", "/*", self._handle_options)

    def _init_engine(self, model_path: str, tp_size: int = 1):
        """Initialize the inference engine.

        Uses TPEngine for multi-GPU (tp_size > 1), ZStreamerEngine for single GPU.
        """
        if tp_size > 1:
            from zse_engine.orchestrator.tp_engine import TPEngine
            self._engine = TPEngine(
                model_path=model_path,
                tp_size=tp_size,
                quiet=self._quiet,
            )
        else:
            from zse_engine.zstreamer.engine import ZStreamerEngine
            self._engine = ZStreamerEngine(
                model_path=model_path,
                quiet=self._quiet,
            )

    async def _handle_health(self, request: Request) -> Response:
        """Health check endpoint."""
        return Response.json({
            "status": "ok",
            "model": self._model_name,
            "engine": "running" if self._engine else "mock",
        })

    async def _handle_options(self, request: Request) -> Response:
        """Handle CORS preflight."""
        return Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "86400",
            },
        )

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single HTTP connection."""
        try:
            # Read request (up to 1MB)
            raw = b""
            while True:
                chunk = await asyncio.wait_for(reader.read(65536), timeout=30)
                if not chunk:
                    return
                raw += chunk
                # Try to parse
                request = parse_http_request(raw)
                if request is not None:
                    break
                if len(raw) > 1048576:  # 1MB limit
                    writer.write(format_http_response(
                        Response.error("Request too large", 413)
                    ))
                    await writer.drain()
                    return

            # Route and handle
            response = await self._router.handle(request)

            # Check if this is a streaming response
            if response._is_streaming and hasattr(response, '_token_queue'):
                # Send headers
                writer.write(format_http_response(response))
                await writer.drain()

                # Stream SSE events from token queue
                await self._stream_sse(writer, response)
            else:
                writer.write(format_http_response(response))
                await writer.drain()

        except asyncio.TimeoutError:
            pass
        except ConnectionResetError:
            pass
        except Exception as e:
            try:
                writer.write(format_http_response(
                    Response.error(f"Internal error: {e}", 500)
                ))
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _stream_sse(self, writer: asyncio.StreamWriter, response: Response):
        """Stream SSE events from a token queue to the client."""
        token_queue = response._token_queue
        completion_id = response._completion_id
        model = response._model
        stream_type = response._stream_type
        tokenizer = self._engine._tokenizer if self._engine else None

        first_chunk = True
        try:
            while True:
                try:
                    event_type, payload = await asyncio.wait_for(
                        token_queue.get(), timeout=120
                    )
                except asyncio.TimeoutError:
                    break

                if event_type == "token":
                    # Decode token to text
                    token_text = ""
                    if tokenizer and hasattr(tokenizer, "decode"):
                        token_text = tokenizer.decode([payload])
                    else:
                        token_text = str(payload)

                    # Skip chat turn markers in streaming output
                    if token_text in ("<|im_end|>", "<|im_start|>", "<|eot_id|>",
                                      "<end_of_turn>", "</s>"):
                        continue

                    if stream_type == "chat":
                        chunk = sse_chat_chunk(
                            completion_id, model, token_text,
                            role="assistant" if first_chunk else None,
                        )
                    else:
                        chunk = sse_completion_chunk(
                            completion_id, model, token_text,
                        )
                    writer.write(chunk)
                    await writer.drain()
                    first_chunk = False

                elif event_type == "done":
                    finish_reason = payload
                    # Send final chunk with finish_reason
                    if stream_type == "chat":
                        chunk = sse_chat_chunk(
                            completion_id, model, "",
                            finish_reason=finish_reason,
                        )
                    else:
                        chunk = sse_completion_chunk(
                            completion_id, model, "",
                            finish_reason=finish_reason,
                        )
                    writer.write(chunk)
                    writer.write(sse_done())
                    await writer.drain()
                    break

        except (ConnectionResetError, BrokenPipeError):
            # Client disconnected — cancel the request
            if hasattr(response, '_request_id') and self._engine:
                self._engine.cancel_request(response._request_id)

    def _run_engine_loop(self):
        """Run the engine's continuous batching loop in a background thread."""
        if self._engine:
            self._engine.run(idle_sleep_ms=1.0)

    async def start(self):
        """Start the HTTP server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self._host,
            self._port,
        )

        # Start engine loop in background thread
        if self._engine:
            self._engine_thread = threading.Thread(
                target=self._run_engine_loop,
                daemon=True,
                name="zse-engine",
            )
            self._engine_thread.start()

        if not self._quiet:
            print(f"\n{'=' * 60}")
            print(f"ZSE Server running on http://{self._host}:{self._port}")
            print(f"{'=' * 60}")
            print(f"  Model:     {self._model_name}")
            print(f"  Dashboard: http://{self._host}:{self._port}/")
            print(f"  API:       http://{self._host}:{self._port}/v1/")
            print(f"  Health:    http://{self._host}:{self._port}/health")
            print(f"{'=' * 60}\n")

    async def run_forever(self):
        """Start and run until interrupted."""
        await self.start()
        try:
            await self._server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Stop the server and cleanup."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        if self._engine:
            self._engine.stop()

        self._db.close()

        if not self._quiet:
            print("\n[ZSE Server] Shutdown complete.")

    def destroy(self):
        """Synchronous cleanup."""
        if self._engine:
            self._engine.stop()
            self._engine.destroy()
        self._db.close()
