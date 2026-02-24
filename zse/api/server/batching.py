"""
ZSE Batched Completion Endpoints

High-throughput completion endpoints using the async batching engine.

When batching is enabled, concurrent requests are automatically batched
for 5-10x throughput improvement.

Author: ZSE Team
"""

import time
import json
import asyncio
from typing import Optional, AsyncGenerator

from fastapi import HTTPException, Security

from zse.api.server.auth import verify_api_key, APIKey
from zse.api.server.models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    ChatCompletionChunk, ChatMessage, CompletionRequest, CompletionResponse,
    CompletionChoice, UsageStats
)
from zse.api.server.state import server_state, LoadedModel
from zse.engine.batching import BatchingEngine, BatchConfig, get_batching_engine


# =============================================================================
# BATCHING STATE
# =============================================================================

class BatchingState:
    """State for batched inference."""
    
    def __init__(self):
        self._enabled = False
        self._engines: dict[str, BatchingEngine] = {}
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def enable(self):
        self._enabled = True
    
    def disable(self):
        self._enabled = False
    
    async def get_engine(self, model: LoadedModel) -> Optional[BatchingEngine]:
        """Get or create batching engine for a model."""
        if not self._enabled:
            return None
        
        if model.model_id not in self._engines:
            # Create new engine
            orch = model.orchestrator
            if not orch:
                print(f"[Batching] No orchestrator for model {model.model_id}")
                return None
            
            # Check if model and tokenizer are loaded
            if not hasattr(orch, 'model') or orch.model is None:
                print(f"[Batching] Model not loaded in orchestrator for {model.model_id}")
                return None
            
            if not hasattr(orch, 'tokenizer') or orch.tokenizer is None:
                print(f"[Batching] Tokenizer not loaded in orchestrator for {model.model_id}")
                return None
            
            try:
                config = BatchConfig(
                    max_batch_size=32,
                    max_tokens_per_batch=4096,
                    batch_wait_timeout_ms=50,
                )
                
                engine = BatchingEngine(orch.model, orch.tokenizer, config)
                await engine.start()
                self._engines[model.model_id] = engine
                print(f"[Batching] Created engine for {model.model_id}")
            except Exception as e:
                print(f"[Batching] Failed to create engine: {e}")
                return None
        
        return self._engines[model.model_id]
        return self._engines[model.model_id]
    
    async def remove_engine(self, model_id: str):
        """Remove engine for a model."""
        if model_id in self._engines:
            await self._engines[model_id].stop()
            del self._engines[model_id]
    
    async def shutdown(self):
        """Shutdown all engines."""
        for engine in self._engines.values():
            await engine.stop()
        self._engines.clear()
    
    def stats(self) -> dict:
        """Get batching stats."""
        return {
            "enabled": self._enabled,
            "engines": {
                model_id: engine.stats()
                for model_id, engine in self._engines.items()
            }
        }


# Global state
_batching_state = BatchingState()


def get_batching_state() -> BatchingState:
    """Get global batching state."""
    return _batching_state


# =============================================================================
# BATCHED COMPLETION FUNCTIONS
# =============================================================================

async def batched_chat_completion(
    request: ChatCompletionRequest,
    api_key: Optional[APIKey] = None
) -> ChatCompletionResponse:
    """
    Process chat completion with batching.
    
    Falls back to non-batched if batching is disabled or unavailable.
    """
    request_id = server_state.generate_request_id()
    start_time = time.time()
    
    try:
        model = _get_model(request.model)
        
        # Build prompt from messages
        prompt = _build_chat_prompt(request.messages)
        prompt_tokens = len(prompt.split()) * 1.3  # rough estimate
        
        # Try batched inference
        state = get_batching_state()
        engine = await state.get_engine(model)
        
        if engine:
            # Use batching engine
            output_text = await engine.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        else:
            # Fallback to sequential
            orch = model.orchestrator
            output_text = ""
            for text in orch.generate(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                output_text += text
        
        completion_tokens = len(output_text.split())
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics
        server_state.record_request(
            request_id=request_id,
            model=request.model,
            endpoint="/v1/chat/completions",
            prompt_tokens=int(prompt_tokens),
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            status="success",
            user=request.user
        )
        
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageStats(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=completion_tokens,
                total_tokens=int(prompt_tokens) + completion_tokens
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        server_state.record_request(
            request_id=request_id,
            model=request.model,
            endpoint="/v1/chat/completions",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            status="error",
            user=request.user
        )
        raise HTTPException(status_code=500, detail=str(e))


async def batched_stream_chat_completion(
    request_id: str,
    request: ChatCompletionRequest,
    model: LoadedModel,
    prompt: str,
    prompt_tokens: int,
    start_time: float
) -> AsyncGenerator[str, None]:
    """Stream chat completion with batching support."""
    
    state = get_batching_state()
    engine = await state.get_engine(model)
    
    completion_tokens = 0
    
    try:
        if engine:
            # Use batching engine streaming
            async for text_chunk in engine.generate_stream(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                completion_tokens += 1
                
                chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None
                    }]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        else:
            # Fallback to sequential
            orch = model.orchestrator
            for text_chunk in orch.generate(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                completion_tokens += 1
                
                chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {"content": text_chunk},
                        "finish_reason": None
                    }]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0)
        
        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=request_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        server_state.record_request(
            request_id=request_id,
            model=request.model,
            endpoint="/v1/chat/completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            status="success",
            user=request.user
        )
        
    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


async def batched_text_completion(
    request: CompletionRequest,
    api_key: Optional[APIKey] = None
) -> CompletionResponse:
    """Process text completion with batching."""
    request_id = server_state.generate_request_id()
    start_time = time.time()
    
    try:
        model = _get_model(request.model)
        
        # Handle prompt
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        prompt_tokens = len(prompt.split())
        
        # Try batched inference
        state = get_batching_state()
        engine = await state.get_engine(model)
        
        if engine:
            output_text = await engine.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        else:
            orch = model.orchestrator
            output_text = ""
            for text in orch.generate(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                output_text += text
        
        completion_tokens = len(output_text.split())
        latency_ms = (time.time() - start_time) * 1000
        
        server_state.record_request(
            request_id=request_id,
            model=request.model,
            endpoint="/v1/completions",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            status="success"
        )
        
        if request.echo:
            output_text = prompt + output_text
        
        return CompletionResponse(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=output_text,
                    finish_reason="stop"
                )
            ],
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HELPERS
# =============================================================================

def _get_model(model_name: str) -> LoadedModel:
    """Get a loaded model or raise error."""
    model = server_state.get_model_by_name(model_name)
    if not model:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not loaded. Load it first with POST /api/models/load"
        )
    return model


def _build_chat_prompt(messages: list) -> str:
    """Build prompt string from chat messages."""
    parts = []
    for msg in messages:
        role = msg.role.capitalize()
        parts.append(f"{role}: {msg.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)
