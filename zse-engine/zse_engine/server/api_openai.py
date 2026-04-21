"""ZSE Server OpenAI API — OpenAI-compatible inference endpoints.

Endpoints:
    POST /v1/chat/completions   — Chat completion (streaming + non-streaming)
    POST /v1/completions        — Text completion (streaming + non-streaming)
    GET  /v1/models             — List available models
"""

import json
import time
import asyncio
import secrets
from typing import Optional, List, Dict, Any

from zse_engine.server.router import Request, Response
from zse_engine.server.sse import (
    sse_chat_chunk, sse_completion_chunk, sse_done,
)


class OpenAIAPI:
    """OpenAI-compatible API endpoint handlers.

    Args:
        engine: ZStreamerEngine instance
        model_name: Name to report for this model (e.g., "llama-7b")
        tokenizer: Tokenizer for decoding output tokens
    """

    def __init__(self, engine, model_name: str = "zse-model", tokenizer=None):
        self._engine = engine
        self._model_name = model_name
        self._tokenizer = tokenizer
        # Detect chat template from model architecture
        self._chat_template = "chatml"  # Default: ChatML (Qwen2, etc.)
        if engine and hasattr(engine, '_config'):
            arch = getattr(engine._config, 'arch', '').lower()
            if 'llama' in arch or 'mistral' in arch:
                self._chat_template = "llama"
            elif 'gemma' in arch:
                self._chat_template = "gemma"
            elif 'phi' in arch:
                self._chat_template = "phi"

    def register(self, router):
        """Register OpenAI-compatible routes on the router."""
        router.post("/v1/chat/completions", self.handle_chat_completions)
        router.post("/v1/completions", self.handle_completions)
        router.get("/v1/models", self.handle_list_models)

    # ------------------------------------------------------------------
    # POST /v1/chat/completions
    # ------------------------------------------------------------------

    async def handle_chat_completions(self, request: Request) -> Response:
        """Handle chat completion requests."""
        body = request.json
        if not body:
            return Response.error("Request body required")

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return Response.error("'messages' field is required and must be a list")

        # Extract parameters
        model = body.get("model", self._model_name)
        temperature = float(body.get("temperature", 1.0))
        top_p = float(body.get("top_p", 0.9))
        top_k = int(body.get("top_k", 50))
        max_tokens = int(body.get("max_tokens", 128))
        stream = bool(body.get("stream", False))
        seed = body.get("seed")
        lora_id = body.get("lora_id")
        repetition_penalty = float(body.get("repetition_penalty", 1.0))

        # RAG augmentation — inject relevant context if enabled
        use_rag = bool(body.get("rag", False))
        if use_rag and hasattr(self, '_rag_engine') and self._rag_engine:
            messages = self._rag_engine.augment_messages(messages)

        # Build prompt from messages
        prompt = self._format_chat_messages(messages)

        completion_id = f"chatcmpl-{secrets.token_hex(12)}"

        if stream:
            return await self._stream_chat_response(
                request, prompt, completion_id, model,
                temperature, top_p, top_k, max_tokens,
                seed, lora_id, repetition_penalty,
            )
        else:
            return await self._sync_chat_response(
                request, prompt, completion_id, model,
                temperature, top_p, top_k, max_tokens,
                seed, lora_id, repetition_penalty,
            )

    async def _sync_chat_response(
        self, request, prompt, completion_id, model,
        temperature, top_p, top_k, max_tokens,
        seed, lora_id, repetition_penalty,
    ) -> Response:
        """Non-streaming chat completion."""
        # Collect all tokens
        tokens: List[int] = []
        done_event = asyncio.Event()
        loop = asyncio.get_event_loop()

        def on_token(token_id):
            tokens.append(token_id)

        def on_finish(output):
            loop.call_soon_threadsafe(done_event.set)

        request_id = self._engine.add_request(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            lora_id=lora_id,
            on_token=on_token,
            on_finish=on_finish,
        )

        if request_id is None:
            return Response.error("Server overloaded, try again later", 503)

        # Wait for completion (with timeout)
        try:
            await asyncio.wait_for(done_event.wait(), timeout=120)
        except asyncio.TimeoutError:
            self._engine.cancel_request(request_id)
            return Response.error("Request timed out", 504)

        # Get result
        result = self._engine.get_result(request_id)
        output_text = self._decode_tokens(tokens)
        prompt_tokens = len(self._tokenizer.encode(prompt)) if self._tokenizer else 0

        finish_reason = "stop"
        if result and result.finish_reason:
            finish_reason = result.finish_reason.value

        return Response.json({
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens),
            },
        })

    async def _stream_chat_response(
        self, request, prompt, completion_id, model,
        temperature, top_p, top_k, max_tokens,
        seed, lora_id, repetition_penalty,
    ):
        """Streaming chat completion — returns a streaming Response.

        The actual SSE data is written by the app layer using the token queue.
        """
        # Create a queue for token streaming
        token_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def on_token(token_id):
            loop.call_soon_threadsafe(token_queue.put_nowait, ("token", token_id))

        def on_finish(output):
            finish_reason = "stop"
            if output and output.finish_reason:
                finish_reason = output.finish_reason.value
            loop.call_soon_threadsafe(token_queue.put_nowait, ("done", finish_reason))

        request_id = self._engine.add_request(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            lora_id=lora_id,
            on_token=on_token,
            on_finish=on_finish,
        )

        if request_id is None:
            return Response.error("Server overloaded, try again later", 503)

        # Return streaming response with token_queue attached
        resp = Response.streaming()
        resp._token_queue = token_queue
        resp._completion_id = completion_id
        resp._model = model
        resp._request_id = request_id
        resp._stream_type = "chat"
        return resp

    # ------------------------------------------------------------------
    # POST /v1/completions
    # ------------------------------------------------------------------

    async def handle_completions(self, request: Request) -> Response:
        """Handle text completion requests."""
        body = request.json
        if not body:
            return Response.error("Request body required")

        prompt = body.get("prompt")
        if not prompt:
            return Response.error("'prompt' field is required")

        model = body.get("model", self._model_name)
        temperature = float(body.get("temperature", 1.0))
        top_p = float(body.get("top_p", 0.9))
        top_k = int(body.get("top_k", 50))
        max_tokens = int(body.get("max_tokens", 128))
        stream = bool(body.get("stream", False))
        seed = body.get("seed")
        lora_id = body.get("lora_id")

        completion_id = f"cmpl-{secrets.token_hex(12)}"

        if stream:
            token_queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def on_token(token_id):
                loop.call_soon_threadsafe(token_queue.put_nowait, ("token", token_id))

            def on_finish(output):
                finish_reason = "stop"
                if output and output.finish_reason:
                    finish_reason = output.finish_reason.value
                loop.call_soon_threadsafe(token_queue.put_nowait, ("done", finish_reason))

            request_id = self._engine.add_request(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                lora_id=lora_id,
                on_token=on_token,
                on_finish=on_finish,
            )
            if request_id is None:
                return Response.error("Server overloaded", 503)

            resp = Response.streaming()
            resp._token_queue = token_queue
            resp._completion_id = completion_id
            resp._model = model
            resp._request_id = request_id
            resp._stream_type = "completion"
            return resp
        else:
            # Sync completion
            tokens = []
            done_event = asyncio.Event()
            loop = asyncio.get_event_loop()

            def on_token(token_id):
                tokens.append(token_id)

            def on_finish(output):
                loop.call_soon_threadsafe(done_event.set)

            request_id = self._engine.add_request(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                lora_id=lora_id,
                on_token=on_token,
                on_finish=on_finish,
            )
            if request_id is None:
                return Response.error("Server overloaded", 503)

            try:
                await asyncio.wait_for(done_event.wait(), timeout=120)
            except asyncio.TimeoutError:
                self._engine.cancel_request(request_id)
                return Response.error("Request timed out", 504)

            result = self._engine.get_result(request_id)
            output_text = self._decode_tokens(tokens)
            prompt_tokens = len(self._tokenizer.encode(prompt)) if self._tokenizer else 0

            finish_reason = "stop"
            if result and result.finish_reason:
                finish_reason = result.finish_reason.value

            return Response.json({
                "id": completion_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "text": output_text,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(tokens),
                    "total_tokens": prompt_tokens + len(tokens),
                },
            })

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------

    async def handle_list_models(self, request: Request) -> Response:
        """List available models."""
        models = [{
            "id": self._model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "zse",
            "permission": [],
        }]

        # Add LoRA adapters as model variants
        if self._engine.lora_manager:
            for adapter_id in self._engine.lora_manager.list_adapters():
                models.append({
                    "id": f"{self._model_name}:{adapter_id}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "zse",
                    "permission": [],
                })

        return Response.json({
            "object": "list",
            "data": models,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_chat_messages(self, messages: List[Dict]) -> str:
        """Convert chat messages to a single prompt string.

        Supports multiple chat templates based on model architecture:
        - ChatML (Qwen2, default): <|im_start|>role\ncontent<|im_end|>
        - Llama/Mistral: [INST] content [/INST]
        - Gemma: <start_of_turn>role\ncontent<end_of_turn>
        """
        if self._chat_template == "llama":
            return self._format_llama(messages)
        elif self._chat_template == "gemma":
            return self._format_gemma(messages)
        else:
            # ChatML (Qwen2, Phi, default)
            return self._format_chatml(messages)

    def _format_chatml(self, messages: List[Dict]) -> str:
        """ChatML format (Qwen2, Phi)."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    def _format_llama(self, messages: List[Dict]) -> str:
        """Llama/Mistral format."""
        parts = []
        system_msg = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_msg = content
            elif role == "user":
                if system_msg:
                    parts.append(f"[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{content} [/INST]")
                    system_msg = ""
                else:
                    parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(f" {content} </s>")
        return "".join(parts)

    def _format_gemma(self, messages: List[Dict]) -> str:
        """Gemma format."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)

    def _decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs to text, stripping chat turn markers."""
        if self._tokenizer and hasattr(self._tokenizer, "decode"):
            text = self._tokenizer.decode(token_ids)
        else:
            text = " ".join(str(t) for t in token_ids)
        # Strip trailing chat markers that may have been generated
        for marker in ["<|im_end|>", "<|im_start|>", "<|eot_id|>",
                       "<end_of_turn>", "</s>"]:
            if marker in text:
                text = text.split(marker)[0]
        return text.rstrip()
