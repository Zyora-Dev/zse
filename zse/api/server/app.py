"""
ZSE FastAPI Server

OpenAI-compatible API with monitoring, analytics, and WebSocket streaming.
"""

import os
import time
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional, Union, AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from zse.api.server.auth import verify_api_key, get_key_manager, APIKey
from zse.api.server.audit import add_audit_middleware, get_audit_logger

from zse.api.server.models import (
    # OpenAI-compatible
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    ChatCompletionChunk, ChatMessage, CompletionRequest, CompletionResponse,
    CompletionChoice, UsageStats, ModelInfo, ModelListResponse, ErrorResponse,
    # ZSE-specific
    LoadModelRequest, LoadModelResponse, UnloadModelRequest,
    HealthResponse, SystemStats, AnalyticsOverview, AnalyticsTimeSeries,
    RequestMetrics
)
from zse.api.server.state import server_state, LoadedModel
from zse.api.server.batching import (
    get_batching_state,
    batched_chat_completion,
    batched_stream_chat_completion,
    batched_text_completion,
)
from zse.version import __version__


# =============================================================================
# App Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle handler."""
    print(f"🚀 ZSE API Server v{__version__} starting...")
    print(f"   Docs: http://localhost:8000/docs")
    print(f"   Dashboard: http://localhost:8000/dashboard")
    yield
    print("👋 ZSE API Server shutting down...")
    # Cleanup batching engines
    await get_batching_state().shutdown()
    # Cleanup loaded models
    for model in server_state.list_models():
        if model.orchestrator:
            model.orchestrator.unload()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="ZSE API",
        description="Z Server Engine - Ultra memory-efficient LLM inference API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS for web UI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add audit logging middleware (before routes, so it catches all requests)
    add_audit_middleware(app, enabled=True)
    
    # Register routes
    _register_health_routes(app)
    _register_model_routes(app)
    _register_api_key_routes(app)
    _register_batching_routes(app)
    _register_registry_routes(app)
    _register_completion_routes(app)
    _register_monitoring_routes(app)
    _register_analytics_routes(app)
    _register_websocket_routes(app)
    _register_dashboard_routes(app)
    _register_audit_routes(app)
    
    return app


# =============================================================================
# Health Routes
# =============================================================================

def _register_health_routes(app: FastAPI):
    """Register health check endpoints."""
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        import torch
        return HealthResponse(
            status="healthy",
            version=__version__,
            uptime_seconds=server_state.uptime_seconds,
            models_loaded=server_state.model_count(),
            gpu_available=torch.cuda.is_available()
        )
    
    @app.get("/", tags=["Health"])
    async def root():
        """Root endpoint - redirect to docs."""
        return {
            "name": "ZSE API",
            "version": __version__,
            "docs": "/docs",
            "dashboard": "/dashboard"
        }


# =============================================================================
# Model Management Routes  
# =============================================================================

def _register_model_routes(app: FastAPI):
    """Register model management endpoints."""
    
    @app.get("/v1/models", response_model=ModelListResponse, tags=["Models"])
    async def list_models():
        """List all available/loaded models (OpenAI-compatible)."""
        models = server_state.list_models()
        return ModelListResponse(
            object="list",
            data=[
                ModelInfo(
                    id=m.model_id,
                    object="model",
                    created=int(m.load_time.timestamp()),
                    owned_by="zse"
                )
                for m in models
            ]
        )
    
    @app.post("/api/models/load", response_model=LoadModelResponse, tags=["Models"])
    async def load_model(request: LoadModelRequest):
        """Load a model into memory."""
        import torch
        from zse.engine.orchestrator import IntelligenceOrchestrator
        
        start_time = time.time()
        
        try:
            # Check if model already loaded
            existing = server_state.get_model_by_name(request.model_name)
            if existing:
                return LoadModelResponse(
                    success=True,
                    model_id=existing.model_id,
                    model_name=existing.model_name,
                    quantization=existing.quantization,
                    vram_used_gb=existing.vram_used_gb,
                    load_time_sec=0,
                    message="Model already loaded"
                )
            
            # Create orchestrator based on quantization
            if request.quantization == "auto":
                if request.target_vram_gb:
                    orch = IntelligenceOrchestrator.for_vram(
                        request.target_vram_gb, 
                        request.model_name,
                        device=request.device
                    )
                else:
                    orch = IntelligenceOrchestrator.auto(
                        request.model_name,
                        device=request.device
                    )
            elif request.quantization == "int4":
                orch = IntelligenceOrchestrator.min_memory(
                    request.model_name,
                    device=request.device
                )
            elif request.quantization == "int8":
                orch = IntelligenceOrchestrator.balanced(
                    request.model_name,
                    device=request.device
                )
            else:  # fp16
                orch = IntelligenceOrchestrator.max_speed(
                    request.model_name,
                    device=request.device
                )
            
            # Load the model
            orch.load(verbose=True)
            
            # Get memory usage
            if torch.cuda.is_available() and request.device != "cpu":
                memory_used = torch.cuda.memory_allocated() / (1024**3)
            else:
                # CPU mode: report from orchestrator config
                import psutil
                process = psutil.Process()
                memory_used = process.memory_info().rss / (1024**3)  # RSS in GB
            
            load_time = time.time() - start_time
            
            # Register model
            model_id = server_state.generate_model_id(request.model_name)
            server_state.add_model(
                model_id=model_id,
                model_name=request.model_name,
                quantization=orch.quantization,
                vram_used_gb=memory_used,
                orchestrator=orch
            )
            
            device_info = f" on {request.device.upper()}" if request.device == "cpu" else ""
            return LoadModelResponse(
                success=True,
                model_id=model_id,
                model_name=request.model_name,
                quantization=orch.quantization,
                device=orch.device,
                vram_used_gb=round(memory_used, 2),
                load_time_sec=round(load_time, 2),
                message=f"Model loaded successfully in {load_time:.1f}s{device_info}"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/models/unload", tags=["Models"])
    async def unload_model(request: UnloadModelRequest):
        """Unload a model from memory."""
        import torch
        import gc
        
        loaded = server_state.remove_model(request.model_id)
        if not loaded:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Cleanup
        if loaded.orchestrator:
            loaded.orchestrator.unload()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {"success": True, "message": f"Model {request.model_id} unloaded"}


# =============================================================================
# API Key Management Routes
# =============================================================================

def _register_api_key_routes(app: FastAPI):
    """Register API key management endpoints."""
    
    @app.get("/api/keys", tags=["API Keys"])
    async def list_api_keys():
        """List all API keys (without revealing actual keys)."""
        manager = get_key_manager()
        return {
            "enabled": manager.is_enabled(),
            "keys": manager.list_keys()
        }
    
    @app.post("/api/keys", tags=["API Keys"])
    async def create_api_key(request: dict):
        """Create a new API key."""
        name = request.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="Key name is required")
        
        manager = get_key_manager()
        key = manager.create_key(name)
        return {"success": True, "name": name, "key": key}
    
    @app.delete("/api/keys/{name}", tags=["API Keys"])
    async def delete_api_key(name: str):
        """Delete an API key by name."""
        manager = get_key_manager()
        if manager.delete_key(name):
            return {"success": True, "message": f"Key '{name}' deleted"}
        raise HTTPException(status_code=404, detail=f"Key '{name}' not found")
    
    @app.post("/api/keys/auth", tags=["API Keys"])
    async def toggle_auth(request: dict):
        """Enable or disable API key authentication."""
        enabled = request.get("enabled", True)
        manager = get_key_manager()
        if enabled:
            manager.enable()
        else:
            manager.disable()
        return {"success": True, "enabled": enabled}


# =============================================================================
# Batching Routes
# =============================================================================

def _register_batching_routes(app: FastAPI):
    """Register batching configuration endpoints."""
    
    @app.get("/api/batching", tags=["Batching"])
    async def get_batching_status():
        """Get batching configuration and stats."""
        state = get_batching_state()
        return state.stats()
    
    @app.post("/api/batching/enable", tags=["Batching"])
    async def enable_batching():
        """
        Enable request batching for high throughput.
        
        When enabled, concurrent requests are automatically batched
        for 5-10x throughput improvement.
        """
        state = get_batching_state()
        state.enable()
        return {"success": True, "enabled": True, "message": "Batching enabled - concurrent requests will be batched for higher throughput"}
    
    @app.post("/api/batching/disable", tags=["Batching"])
    async def disable_batching():
        """Disable request batching (process sequentially)."""
        state = get_batching_state()
        state.disable()
        return {"success": True, "enabled": False, "message": "Batching disabled - requests processed sequentially"}


# =============================================================================
# Model Registry & Discovery Routes
# =============================================================================

def _register_registry_routes(app: FastAPI):
    """Register model registry and discovery endpoints."""
    
    from zse.models.registry import get_registry, ModelCategory, ModelSize
    from zse.models.discovery import get_discovery
    
    @app.get("/api/models/registry", tags=["Model Registry"])
    async def list_registry(
        category: Optional[str] = Query(None, description="Filter by category: chat, instruct, code, reasoning"),
        size: Optional[str] = Query(None, description="Filter by size: tiny, small, medium, large, xlarge, xxl"),
        max_vram: Optional[float] = Query(None, description="Max VRAM in GB (filters by INT8 requirement)"),
        recommended: bool = Query(False, description="Show only recommended models"),
    ):
        """
        List curated models from the ZSE registry.
        
        Returns tested, known-compatible models with VRAM estimates and recommended settings.
        """
        registry = get_registry()
        
        if recommended:
            models = registry.get_recommended(max_vram_gb=max_vram)
        elif max_vram:
            models = registry.filter_by_vram(max_vram, quantization="int8")
        elif category:
            try:
                cat = ModelCategory(category.lower())
                models = registry.filter_by_category(cat)
            except ValueError:
                raise HTTPException(400, f"Invalid category. Valid: {[c.value for c in ModelCategory]}")
        elif size:
            try:
                sz = ModelSize(size.lower())
                models = registry.filter_by_size(sz)
            except ValueError:
                raise HTTPException(400, f"Invalid size. Valid: {[s.value for s in ModelSize]}")
        else:
            models = registry.list_all()
        
        return {
            "count": len(models),
            "models": [m.to_dict() for m in models],
        }
    
    @app.get("/api/models/registry/{model_id:path}", tags=["Model Registry"])
    async def get_registry_model(model_id: str):
        """Get detailed info for a specific model from the registry."""
        registry = get_registry()
        model = registry.get(model_id)
        if not model:
            raise HTTPException(404, f"Model '{model_id}' not in registry. Try /api/models/search")
        return model.to_dict()
    
    @app.get("/api/models/search", tags=["Model Discovery"])
    async def search_huggingface(
        q: str = Query("", description="Search query"),
        author: Optional[str] = Query(None, description="Filter by author (e.g., 'meta-llama')"),
        limit: int = Query(20, ge=1, le=50, description="Max results"),
        sort: str = Query("downloads", description="Sort by: downloads, likes, created_at"),
    ):
        """
        Search HuggingFace Hub for compatible models.
        
        Discovers models compatible with ZSE from HuggingFace.
        Use this to find new models not yet in the registry.
        """
        try:
            discovery = get_discovery()
            models = discovery.search(
                query=q,
                author=author,
                limit=limit,
                sort=sort,
                only_compatible=True,
            )
            return {
                "count": len(models),
                "query": q,
                "models": [m.to_dict() for m in models],
            }
        except ImportError as e:
            raise HTTPException(500, f"Discovery requires httpx: pip install httpx")
        except Exception as e:
            raise HTTPException(500, f"Search failed: {str(e)}")
    
    @app.get("/api/models/check/{model_id:path}", tags=["Model Discovery"])
    async def check_model_compatibility(model_id: str):
        """
        Check if a model is compatible with ZSE.
        
        Fetches model info from HuggingFace and checks architecture,
        file formats, and estimates VRAM requirements.
        """
        try:
            discovery = get_discovery()
            result = discovery.check_compatibility(model_id)
            return result
        except ImportError:
            raise HTTPException(500, "Discovery requires httpx: pip install httpx")
        except Exception as e:
            raise HTTPException(500, f"Check failed: {str(e)}")


# =============================================================================
# Completion Routes (OpenAI-compatible)
# =============================================================================

def _register_completion_routes(app: FastAPI):
    """Register OpenAI-compatible completion endpoints."""
    
    def _get_model(model_name: str) -> LoadedModel:
        """Get a loaded model or raise error."""
        model = server_state.get_model_by_name(model_name)
        if not model:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not loaded. Load it first with POST /api/models/load"
            )
        return model
    
    @app.post("/v1/chat/completions", tags=["Completions"])
    async def chat_completion(
        request: ChatCompletionRequest,
        api_key: Optional[APIKey] = Security(verify_api_key)
    ):
        """
        Create a chat completion (OpenAI-compatible).
        
        Supports streaming with stream=true.
        Requires API key when authentication is enabled.
        When batching is enabled, concurrent requests are automatically batched.
        """
        # Check if batching is enabled for non-streaming
        batching_state = get_batching_state()
        
        if not request.stream and batching_state.enabled:
            # Use batched completion
            return await batched_chat_completion(request, api_key)
        
        request_id = server_state.generate_request_id()
        start_time = time.time()
        
        try:
            model = _get_model(request.model)
            orch = model.orchestrator
            
            # Build prompt from messages
            prompt = _build_chat_prompt(request.messages)
            
            # Count prompt tokens (approximate)
            prompt_tokens = len(prompt.split()) * 1.3  # rough estimate
            
            if request.stream:
                # Streaming response (uses batching if enabled)
                return StreamingResponse(
                    batched_stream_chat_completion(
                        request_id, request, model, prompt, int(prompt_tokens), start_time
                    ) if batching_state.enabled else _stream_chat_completion(
                        request_id, request, model, prompt, int(prompt_tokens), start_time
                    ),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming response - pass parameters directly
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
    
    @app.post("/v1/completions", tags=["Completions"])
    async def text_completion(
        request: CompletionRequest,
        api_key: Optional[APIKey] = Security(verify_api_key)
    ):
        """Create a text completion (OpenAI-compatible). Requires API key when auth is enabled."""
        # Check if batching is enabled for non-streaming
        batching_state = get_batching_state()
        
        if not request.stream and batching_state.enabled:
            return await batched_text_completion(request, api_key)
        
        request_id = server_state.generate_request_id()
        start_time = time.time()
        
        try:
            model = _get_model(request.model)
            orch = model.orchestrator
            
            # Handle prompt (can be string or list)
            prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
            prompt_tokens = len(prompt.split())
            
            if request.stream:
                return StreamingResponse(
                    _stream_completion(request_id, request, model, prompt, prompt_tokens, start_time),
                    media_type="text/event-stream"
                )
            else:
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
                
                # Prepend prompt if echo=True
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


def _build_chat_prompt(messages: list) -> str:
    """Build prompt string from chat messages."""
    # Simple format - can be improved with model-specific templates
    parts = []
    for msg in messages:
        role = msg.role.capitalize()
        parts.append(f"{role}: {msg.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


async def _stream_chat_completion(
    request_id: str,
    request: ChatCompletionRequest,
    model: LoadedModel,
    prompt: str,
    prompt_tokens: int,
    start_time: float
) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens."""
    orch = model.orchestrator
    completion_tokens = 0
    
    try:
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
            await asyncio.sleep(0)  # Allow other tasks
        
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


async def _stream_completion(
    request_id: str,
    request: CompletionRequest,
    model: LoadedModel,
    prompt: str,
    prompt_tokens: int,
    start_time: float
) -> AsyncGenerator[str, None]:
    """Stream text completion tokens."""
    orch = model.orchestrator
    completion_tokens = 0
    
    for text_chunk in orch.generate(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    ):
        completion_tokens += 1
        
        data = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "text": text_chunk,
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0)
    
    yield "data: [DONE]\n\n"
    
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


# =============================================================================
# Monitoring Routes
# =============================================================================

def _register_monitoring_routes(app: FastAPI):
    """Register monitoring endpoints."""
    
    @app.get("/api/stats", response_model=SystemStats, tags=["Monitoring"])
    async def get_system_stats():
        """Get system statistics (CPU, memory, GPU)."""
        return server_state.get_system_stats()
    
    @app.get("/api/stats/gpu", tags=["Monitoring"])
    async def get_gpu_stats():
        """Get GPU statistics."""
        return {"gpus": server_state.get_gpu_stats()}
    
    @app.get("/api/stats/models", tags=["Monitoring"])
    async def get_model_stats():
        """Get statistics for loaded models."""
        models = server_state.list_models()
        return {
            "models": [
                {
                    "model_id": m.model_id,
                    "model_name": m.model_name,
                    "quantization": m.quantization,
                    "vram_used_gb": m.vram_used_gb,
                    "load_time": m.load_time.isoformat(),
                    "request_count": m.request_count,
                    "tokens_generated": m.tokens_generated
                }
                for m in models
            ]
        }


# =============================================================================
# Analytics Routes
# =============================================================================

def _register_analytics_routes(app: FastAPI):
    """Register analytics endpoints."""
    
    @app.get("/api/analytics", response_model=AnalyticsOverview, tags=["Analytics"])
    async def get_analytics_overview():
        """Get analytics overview."""
        return server_state.get_analytics_overview()
    
    @app.get("/api/analytics/timeseries", response_model=AnalyticsTimeSeries, tags=["Analytics"])
    async def get_analytics_timeseries():
        """Get time series analytics data (last 60 minutes)."""
        return server_state.get_analytics_timeseries()
    
    @app.get("/api/analytics/requests", tags=["Analytics"])
    async def get_recent_requests(
        limit: int = Query(default=100, le=1000)
    ):
        """Get recent requests."""
        requests = server_state.get_recent_requests(limit)
        return {
            "requests": [r.model_dump() for r in requests]
        }


# =============================================================================
# WebSocket Routes
# =============================================================================

def _register_websocket_routes(app: FastAPI):
    """Register WebSocket endpoints."""
    
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket endpoint for real-time chat.
        
        Send: {"model": "...", "messages": [...], "max_tokens": 512}
        Receive: {"type": "token", "content": "..."} or {"type": "done"}
        """
        await websocket.accept()
        
        try:
            while True:
                data = await websocket.receive_json()
                
                model_name = data.get("model")
                messages = data.get("messages", [])
                max_tokens = data.get("max_tokens", 512)
                temperature = data.get("temperature", 0.7)
                
                model = server_state.get_model_by_name(model_name)
                if not model:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Model '{model_name}' not loaded"
                    })
                    continue
                
                # Build prompt
                prompt = _build_chat_prompt([
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in messages
                ])
                
                # Generate
                full_response = ""
                for text_chunk in model.orchestrator.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                ):
                    full_response += text_chunk
                    await websocket.send_json({
                        "type": "token",
                        "content": text_chunk
                    })
                
                await websocket.send_json({
                    "type": "done",
                    "full_response": full_response
                })
                
        except WebSocketDisconnect:
            pass
    
    @app.websocket("/ws/stats")
    async def websocket_stats(websocket: WebSocket):
        """
        WebSocket endpoint for real-time stats updates.
        
        Sends system stats every 2 seconds.
        """
        await websocket.accept()
        
        try:
            while True:
                stats = server_state.get_system_stats()
                analytics = server_state.get_analytics_overview()
                
                await websocket.send_json({
                    "type": "stats",
                    "system": stats.model_dump(),
                    "analytics": analytics.model_dump()
                })
                
                await asyncio.sleep(2)
                
        except WebSocketDisconnect:
            pass


# =============================================================================
# Dashboard Routes
# =============================================================================

def _register_dashboard_routes(app: FastAPI):
    """Register web dashboard routes."""
    
    @app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
    async def dashboard():
        """Serve the monitoring dashboard."""
        return _get_dashboard_html()


def _get_dashboard_html() -> str:
    """Generate dashboard HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZSE | Control Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #111111;
            --bg-card: #161616;
            --bg-card-hover: #1a1a1a;
            --border: #2a2a2a;
            --text-primary: #ffffff;
            --text-secondary: #888888;
            --text-muted: #555555;
            --accent: #ffffff;
            --accent-dim: rgba(255,255,255,0.1);
            --success: #22c55e;
            --warning: #eab308;
            --error: #ef4444;
            --glow: rgba(255,255,255,0.05);
        }
        
        body {
            font-family: 'Montserrat', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        /* Floating Sidebar */
        .sidebar {
            position: fixed;
            left: 24px;
            top: 50%;
            transform: translateY(-50%);
            width: 72px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 16px 0;
            z-index: 1000;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }
        
        .sidebar-item {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            margin: 8px auto;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.2s ease;
            color: var(--text-secondary);
        }
        
        .sidebar-item:hover, .sidebar-item.active {
            background: var(--accent-dim);
            color: var(--text-primary);
        }
        
        .sidebar-item.active {
            background: var(--accent);
            color: var(--bg-primary);
        }
        
        .sidebar-item svg {
            width: 22px;
            height: 22px;
        }
        
        .sidebar-logo {
            text-align: center;
            padding: 12px 0 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 12px;
        }
        
        .sidebar-logo span {
            font-weight: 700;
            font-size: 18px;
            letter-spacing: 2px;
        }
        
        /* Main Content */
        .main {
            margin-left: 120px;
            padding: 32px 40px;
            max-width: 1400px;
        }
        
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 28px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        
        .header-status {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 30px;
            font-size: 13px;
            font-weight: 500;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Section Title */
        .section-title {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 20px;
            padding-left: 4px;
        }
        
        /* Cards */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            background: var(--bg-card-hover);
            border-color: #333;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            text-align: center;
            padding: 28px 20px;
        }
        
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -1px;
        }
        
        .stat-label {
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--text-secondary);
        }
        
        /* Two Column Grid */
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .card-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card-title svg {
            width: 18px;
            height: 18px;
            opacity: 0.6;
        }
        
        /* GPU Progress */
        .gpu-item {
            margin-bottom: 20px;
        }
        
        .gpu-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 13px;
        }
        
        .gpu-name {
            font-weight: 500;
        }
        
        .gpu-usage {
            color: var(--text-secondary);
        }
        
        .progress-bar {
            height: 6px;
            background: var(--border);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent) 0%, #888 100%);
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        
        /* Model List */
        .model-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 0;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }
        
        .model-item:last-child {
            border-bottom: none;
        }
        
        .model-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
        }
        
        /* Charts */
        .chart-container {
            height: 200px;
            margin-top: 10px;
        }
        
        /* Playground */
        .playground {
            margin-bottom: 40px;
        }
        
        .playground-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: var(--text-secondary);
            margin-bottom: 10px;
            display: block;
        }
        
        .input-select, .input-textarea {
            width: 100%;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px 16px;
            color: var(--text-primary);
            font-family: 'Montserrat', sans-serif;
            font-size: 14px;
            transition: border-color 0.2s;
        }
        
        .input-select:focus, .input-textarea:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .input-textarea {
            resize: none;
            height: 120px;
        }
        
        .btn {
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            padding: 14px 28px;
            border-radius: 10px;
            font-family: 'Montserrat', sans-serif;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.5px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(255,255,255,0.2);
        }
        
        .btn-outline {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text-primary);
        }
        
        .btn-outline:hover {
            background: var(--accent-dim);
            border-color: var(--accent);
        }
        
        .btn-sm {
            padding: 8px 16px;
            font-size: 12px;
        }
        
        .btn-danger {
            background: var(--error);
        }
        
        .chat-output {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            min-height: 200px;
            font-size: 14px;
            line-height: 1.7;
            color: var(--text-secondary);
            white-space: pre-wrap;
        }
        
        /* API Keys Section */
        .api-keys-section {
            margin-bottom: 40px;
        }
        
        .key-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .key-item:last-child {
            border-bottom: none;
        }
        
        .key-info {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .key-icon {
            width: 40px;
            height: 40px;
            background: var(--accent-dim);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .key-icon svg {
            width: 18px;
            height: 18px;
        }
        
        .key-name {
            font-weight: 500;
            font-size: 14px;
            margin-bottom: 4px;
        }
        
        .key-meta {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .key-stats {
            text-align: right;
            margin-right: 20px;
        }
        
        .key-requests {
            font-size: 14px;
            font-weight: 500;
        }
        
        .key-last-used {
            font-size: 12px;
            color: var(--text-secondary);
        }
        
        .create-key-form {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }
        
        .create-key-form input {
            flex: 1;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 16px;
            color: var(--text-primary);
            font-family: 'Montserrat', sans-serif;
            font-size: 14px;
        }
        
        .create-key-form input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .auth-toggle {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px 20px;
            background: var(--bg-secondary);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .toggle-switch {
            width: 48px;
            height: 26px;
            background: var(--border);
            border-radius: 13px;
            position: relative;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .toggle-switch.active {
            background: var(--success);
        }
        
        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            top: 3px;
            left: 3px;
            transition: transform 0.3s;
        }
        
        .toggle-switch.active::after {
            transform: translateX(22px);
        }
        
        .toggle-label {
            font-size: 14px;
            font-weight: 500;
        }
        
        .new-key-display {
            background: var(--bg-secondary);
            border: 1px solid var(--success);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .new-key-display code {
            font-family: monospace;
            background: var(--bg-primary);
            padding: 12px 16px;
            border-radius: 8px;
            display: block;
            margin: 12px 0;
            font-size: 13px;
            word-break: break-all;
        }
        
        .new-key-warning {
            color: var(--warning);
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        /* Page Sections */
        .page { display: none; }
        .page.active { display: block; }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }
        
        .empty-state svg {
            width: 48px;
            height: 48px;
            margin-bottom: 16px;
            opacity: 0.3;
        }
        
        /* Documentation Styles */
        .doc-content {
            color: var(--text-secondary);
            line-height: 1.8;
            font-size: 14px;
        }
        
        .doc-content p {
            margin-bottom: 20px;
        }
        
        .doc-steps {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        
        .doc-step {
            display: flex;
            align-items: flex-start;
            gap: 16px;
            padding: 16px;
            background: var(--bg-secondary);
            border-radius: 10px;
        }
        
        .step-num {
            width: 32px;
            height: 32px;
            background: var(--accent);
            color: var(--bg-primary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            flex-shrink: 0;
        }
        
        .step-content {
            flex: 1;
        }
        
        .step-content strong {
            display: block;
            color: var(--text-primary);
            margin-bottom: 8px;
        }
        
        .step-content code {
            display: block;
            background: var(--bg-primary);
            padding: 10px 14px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 8px;
        }
        
        .step-note {
            display: block;
            color: var(--text-muted);
            font-size: 12px;
            margin-top: 4px;
        }
        
        .step-note code {
            display: inline;
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
        }
        
        .endpoint-list {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .endpoint-group-title {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 12px;
        }
        
        .endpoint {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
        }
        
        .endpoint:last-child {
            border-bottom: none;
        }
        
        .method {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            min-width: 50px;
            text-align: center;
        }
        
        .method.get { background: #166534; color: #86efac; }
        .method.post { background: #1e40af; color: #93c5fd; }
        .method.delete { background: #991b1b; color: #fca5a5; }
        .method.ws { background: #6b21a8; color: #d8b4fe; }
        
        .path {
            font-family: monospace;
            color: var(--text-primary);
            flex-shrink: 0;
        }
        
        .desc {
            color: var(--text-secondary);
            margin-left: auto;
        }
        
        .code-block {
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 13px;
            line-height: 1.6;
            overflow-x: auto;
            margin: 0;
            white-space: pre;
        }
        
        .code-keyword { color: #c792ea; }
        .code-string { color: #c3e88d; }
        .code-comment { color: #546e7a; }
        .code-number { color: #f78c6c; }
        
        .cli-table {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .cli-row {
            display: flex;
            align-items: center;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 8px;
            gap: 20px;
        }
        
        .cli-cmd {
            font-family: monospace;
            font-size: 13px;
            color: var(--text-primary);
            min-width: 280px;
        }
        
        .cli-desc {
            color: var(--text-secondary);
            font-size: 13px;
        }
        
        .cli-group-title {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            color: var(--text-muted);
            padding: 8px 0 4px;
        }
        
        .memory-table {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .memory-header, .memory-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
        }
        
        .memory-header {
            background: var(--bg-secondary);
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 1px;
        }
        
        .memory-row {
            border: 1px solid var(--border);
        }
        
        .mem-fp16 { color: var(--error); }
        .mem-zse { color: var(--success); }
        .mem-save { color: var(--warning); font-weight: 600; }
        
        /* Load Model Form */
        .load-model-form {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }
        
        .input-row {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        
        .model-input {
            flex: 1;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            color: var(--text-primary);
            font-family: 'Montserrat', sans-serif;
            font-size: 13px;
        }
        
        .model-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .quant-select {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 16px;
            color: var(--text-primary);
            font-family: 'Montserrat', sans-serif;
            font-size: 13px;
            min-width: 100px;
        }
        
        .quant-select:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .load-status {
            margin-top: 12px;
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .load-status.loading {
            color: var(--warning);
        }
        
        .load-status.success {
            color: var(--success);
        }
        
        .load-status.error {
            color: var(--error);
        }
        
        .model-item-with-actions {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
        }
        
        .model-item-with-actions:last-child {
            border-bottom: none;
        }
        
        .model-info-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .btn-unload {
            padding: 6px 12px;
            font-size: 11px;
            background: transparent;
            border: 1px solid var(--error);
            color: var(--error);
            border-radius: 6px;
            cursor: pointer;
            font-family: 'Montserrat', sans-serif;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .btn-unload:hover {
            background: var(--error);
            color: white;
        }
        
        .empty-state {
            color: var(--text-muted);
            font-size: 13px;
            padding: 12px 0;
            text-align: center;
        }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .grid-2 { grid-template-columns: 1fr; }
            .playground-grid { grid-template-columns: 1fr; }
            .input-row { flex-wrap: wrap; }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                left: 12px;
                width: 60px;
            }
            .main {
                margin-left: 90px;
                padding: 20px;
            }
            .stats-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <!-- Floating Sidebar -->
    <nav class="sidebar">
        <div class="sidebar-logo">
            <span>ZSE</span>
        </div>
        <div class="sidebar-item active" onclick="showPage('dashboard')" title="Dashboard">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"/>
            </svg>
        </div>
        <div class="sidebar-item" onclick="showPage('playground')" title="Playground">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
            </svg>
        </div>
        <div class="sidebar-item" onclick="showPage('apikeys')" title="API Keys">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
            </svg>
        </div>
        <div class="sidebar-item" onclick="showPage('docs')" title="Documentation">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
            </svg>
        </div>
        <div class="sidebar-item" onclick="window.open('/docs', '_blank')" title="Swagger API">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
            </svg>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="main">
        <!-- Dashboard Page -->
        <div id="page-dashboard" class="page active">
            <div class="header">
                <h1>Dashboard</h1>
                <div class="header-status">
                    <span class="status-dot" id="status-dot"></span>
                    <span id="status-text">Connected</span>
                </div>
            </div>

            <div class="section-title">Overview</div>
            <div class="stats-grid">
                <div class="card stat-card">
                    <div class="stat-value" id="total-requests">0</div>
                    <div class="stat-label">Requests</div>
                </div>
                <div class="card stat-card">
                    <div class="stat-value" id="total-tokens">0</div>
                    <div class="stat-label">Tokens</div>
                </div>
                <div class="card stat-card">
                    <div class="stat-value" id="avg-latency">0<span style="font-size:16px;opacity:0.5">ms</span></div>
                    <div class="stat-label">Latency</div>
                </div>
                <div class="card stat-card">
                    <div class="stat-value" id="tokens-per-sec">0</div>
                    <div class="stat-label">Tokens/sec</div>
                </div>
            </div>

            <div class="section-title">System</div>
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
                        </svg>
                        GPU Memory
                    </div>
                    <div id="gpu-stats">
                        <div class="empty-state">
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
                            </svg>
                            <div>No GPU detected</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                        </svg>
                        Loaded Models
                    </div>
                    <div id="models-list">
                        <div class="empty-state">
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/>
                            </svg>
                            <div>No models loaded</div>
                        </div>
                    </div>
                    
                    <!-- Load Model Form -->
                    <div class="load-model-form">
                        <div class="input-row">
                            <input type="text" id="model-name-input" placeholder="HuggingFace ID or local path" class="model-input">
                            <select id="quant-select" class="quant-select">
                                <option value="auto">Auto</option>
                                <option value="fp16">FP16</option>
                                <option value="int8">INT8</option>
                                <option value="int4">INT4</option>
                            </select>
                            <button class="btn" onclick="loadModel()">Load</button>
                        </div>
                        <div id="load-status" class="load-status"></div>
                    </div>
                </div>
            </div>

            <div class="section-title">Performance</div>
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">Requests / Minute</div>
                    <div class="chart-container">
                        <canvas id="requests-chart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">Memory Usage (GB)</div>
                    <div class="chart-container">
                        <canvas id="memory-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Playground Page -->
        <div id="page-playground" class="page">
            <div class="header">
                <h1>Playground</h1>
            </div>

            <div class="card playground">
                <div class="playground-grid">
                    <div>
                        <div class="input-group">
                            <label class="input-label">Model</label>
                            <select id="chat-model" class="input-select">
                                <option value="">Select a model...</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label class="input-label">Message</label>
                            <textarea id="chat-input" class="input-textarea" placeholder="Enter your message..."></textarea>
                        </div>
                        <button class="btn" onclick="sendChat()">Generate</button>
                    </div>
                    <div>
                        <label class="input-label">Response</label>
                        <div id="chat-output" class="chat-output">Response will appear here...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- API Keys Page -->
        <div id="page-apikeys" class="page">
            <div class="header">
                <h1>API Keys</h1>
            </div>

            <div class="card api-keys-section">
                <div class="auth-toggle">
                    <div id="auth-toggle" class="toggle-switch active" onclick="toggleAuth()"></div>
                    <span class="toggle-label">Authentication <span id="auth-status">Enabled</span></span>
                </div>

                <div class="card-title">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
                    </svg>
                    Your API Keys
                </div>

                <div id="keys-list">
                    <div class="empty-state">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
                        </svg>
                        <div>No API keys created yet</div>
                    </div>
                </div>

                <div id="new-key-display" class="new-key-display" style="display:none;">
                    <strong>New API Key Created</strong>
                    <code id="new-key-value"></code>
                    <div class="new-key-warning">
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                        </svg>
                        Save this key now - it won't be shown again!
                    </div>
                </div>

                <div class="create-key-form">
                    <input type="text" id="new-key-name" placeholder="Key name (e.g., my-app)">
                    <button class="btn" onclick="createKey()">Create Key</button>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Usage Examples</div>
                <div style="font-size:13px; color: var(--text-secondary); line-height: 1.8;">
                    <p style="margin-bottom:16px;"><strong style="color:var(--text-primary)">Python (OpenAI SDK):</strong></p>
                    <code style="display:block; background:var(--bg-secondary); padding:16px; border-radius:8px; margin-bottom:20px; white-space:pre; overflow-x:auto;">from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello!"}]
)</code>
                    <p style="margin-bottom:16px;"><strong style="color:var(--text-primary)">cURL:</strong></p>
                    <code style="display:block; background:var(--bg-secondary); padding:16px; border-radius:8px; overflow-x:auto;">curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-api-key" \\
  -d '{"model": "your-model", "messages": [{"role": "user", "content": "Hello!"}]}'</code>
                </div>
            </div>
        </div>

        <!-- Documentation Page -->
        <div id="page-docs" class="page">
            <div class="header">
                <h1>API Documentation</h1>
                <p style="color:var(--text-muted);margin-top:8px;">Complete reference for ZSE inference server</p>
            </div>

            <!-- Quick Start -->
            <div class="card" style="margin-bottom:24px;">
                <div class="card-title">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                    Quick Start
                </div>
                <div class="doc-content">
                    <p>ZSE (Z Server Engine) is an ultra memory-efficient LLM inference engine with OpenAI-compatible API.</p>
                    <div class="doc-steps">
                        <div class="doc-step">
                            <div class="step-num">1</div>
                            <div class="step-content">
                                <strong>Start the Server</strong>
                                <code>zse serve</code>
                                <span class="step-note">Or with a pre-loaded model: <code>zse serve &lt;model&gt;</code></span>
                            </div>
                        </div>
                        <div class="doc-step">
                            <div class="step-num">2</div>
                            <div class="step-content">
                                <strong>Load a Model</strong>
                                <span class="step-note">Via Dashboard or API:</span>
                                <code>POST /api/models/load {"model_path": "&lt;model&gt;"}</code>
                            </div>
                        </div>
                        <div class="doc-step">
                            <div class="step-num">3</div>
                            <div class="step-content">
                                <strong>Create API Key (Optional)</strong>
                                <code>zse api-key create &lt;name&gt;</code>
                            </div>
                        </div>
                        <div class="doc-step">
                            <div class="step-num">4</div>
                            <div class="step-content">
                                <strong>Make Requests</strong>
                                <code>POST /v1/chat/completions</code>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- API Endpoints -->
            <div class="card" style="margin-bottom:24px;">
                <div class="card-title">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                    </svg>
                    API Endpoints
                </div>
                <div class="endpoint-list">
                    <div class="endpoint-group">
                        <div class="endpoint-group-title">Chat & Completions</div>
                        <div class="endpoint">
                            <span class="method post">POST</span>
                            <span class="path">/v1/chat/completions</span>
                            <span class="desc">Create chat completion (OpenAI-compatible)</span>
                        </div>
                        <div class="endpoint">
                            <span class="method post">POST</span>
                            <span class="path">/v1/completions</span>
                            <span class="desc">Create text completion</span>
                        </div>
                    </div>
                    <div class="endpoint-group">
                        <div class="endpoint-group-title">Models</div>
                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <span class="path">/v1/models</span>
                            <span class="desc">List loaded models</span>
                        </div>
                        <div class="endpoint">
                            <span class="method post">POST</span>
                            <span class="path">/api/models/load</span>
                            <span class="desc">Load a model into memory</span>
                        </div>
                        <div class="endpoint">
                            <span class="method post">POST</span>
                            <span class="path">/api/models/unload</span>
                            <span class="desc">Unload a model</span>
                        </div>
                    </div>
                    <div class="endpoint-group">
                        <div class="endpoint-group-title">System</div>
                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <span class="path">/health</span>
                            <span class="desc">Health check</span>
                        </div>
                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <span class="path">/api/stats</span>
                            <span class="desc">System statistics</span>
                        </div>
                        <div class="endpoint">
                            <span class="method get">GET</span>
                            <span class="path">/api/analytics</span>
                            <span class="desc">Usage analytics</span>
                        </div>
                    </div>
                    <div class="endpoint-group">
                        <div class="endpoint-group-title">WebSocket</div>
                        <div class="endpoint">
                            <span class="method ws">WS</span>
                            <span class="path">/ws/chat</span>
                            <span class="desc">Streaming chat via WebSocket</span>
                        </div>
                        <div class="endpoint">
                            <span class="method ws">WS</span>
                            <span class="path">/ws/stats</span>
                            <span class="desc">Real-time stats stream</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Code Examples -->
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">Python (OpenAI SDK)</div>
                    <pre class="code-block"><span class="code-keyword">from</span> openai <span class="code-keyword">import</span> OpenAI

client = OpenAI(
    base_url=<span class="code-string">"http://localhost:8000/v1"</span>,
    api_key=<span class="code-string">"your-api-key"</span>  <span class="code-comment"># or "sk-" if auth disabled</span>
)

<span class="code-comment"># Get available models</span>
models = client.models.list()
model_name = models.data[<span class="code-number">0</span>].id

<span class="code-comment"># Chat completion</span>
response = client.chat.completions.create(
    model=model_name,
    messages=[{<span class="code-string">"role"</span>: <span class="code-string">"user"</span>, <span class="code-string">"content"</span>: <span class="code-string">"Hello!"</span>}],
    stream=<span class="code-keyword">True</span>
)

<span class="code-keyword">for</span> chunk <span class="code-keyword">in</span> response:
    <span class="code-keyword">if</span> chunk.choices[<span class="code-number">0</span>].delta.content:
        <span class="code-keyword">print</span>(chunk.choices[<span class="code-number">0</span>].delta.content, end=<span class="code-string">""</span>)</pre>
                </div>
                <div class="card">
                    <div class="card-title">cURL</div>
                    <pre class="code-block"><span class="code-comment"># List models</span>
curl http://localhost:8000/v1/models

<span class="code-comment"># Chat completion</span>
curl http://localhost:8000/v1/chat/completions \
  -H <span class="code-string">"Content-Type: application/json"</span> \
  -H <span class="code-string">"X-API-Key: your-api-key"</span> \
  -d <span class="code-string">'{"model": "your-model", "messages": [{"role": "user", "content": "Hello"}]}'</span>

<span class="code-comment"># Load a model</span>
curl -X POST http://localhost:8000/api/models/load \
  -H <span class="code-string">"Content-Type: application/json"</span> \
  -d <span class="code-string">'{"model_path": "your-model-id-or-path"}'</span>

<span class="code-comment"># Unload a model</span>
curl -X POST http://localhost:8000/api/models/unload \
  -H <span class="code-string">"Content-Type: application/json"</span> \
  -d <span class="code-string">'{"model_id": "model-uuid"}'</span></pre>
                </div>
            </div>

            <!-- JavaScript Example -->
            <div class="card" style="margin-top:24px;">
                <div class="card-title">JavaScript / TypeScript</div>
                <pre class="code-block"><span class="code-comment">// Fetch API</span>
<span class="code-keyword">const</span> response = <span class="code-keyword">await</span> fetch(<span class="code-string">'http://localhost:8000/v1/chat/completions'</span>, {
  method: <span class="code-string">'POST'</span>,
  headers: {
    <span class="code-string">'Content-Type'</span>: <span class="code-string">'application/json'</span>,
    <span class="code-string">'X-API-Key'</span>: <span class="code-string">'your-api-key'</span>
  },
  body: JSON.stringify({
    model: <span class="code-string">'your-loaded-model'</span>,
    messages: [{ role: <span class="code-string">'user'</span>, content: <span class="code-string">'Hello!'</span> }],
    stream: <span class="code-keyword">true</span>
  })
});

<span class="code-comment">// Streaming response</span>
<span class="code-keyword">const</span> reader = response.body.getReader();
<span class="code-keyword">const</span> decoder = <span class="code-keyword">new</span> TextDecoder();

<span class="code-keyword">while</span> (<span class="code-keyword">true</span>) {
  <span class="code-keyword">const</span> { done, value } = <span class="code-keyword">await</span> reader.read();
  <span class="code-keyword">if</span> (done) <span class="code-keyword">break</span>;
  console.log(decoder.decode(value));
}</pre>
            </div>

            <!-- CLI Reference -->
            <div class="card" style="margin-top:24px;">
                <div class="card-title">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                    </svg>
                    CLI Reference
                </div>
                <div class="cli-table">
                    <div class="cli-group-title">Server</div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse serve</code>
                        <span class="cli-desc">Start server (load models via dashboard)</span>
                    </div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse serve &lt;model&gt;</code>
                        <span class="cli-desc">Start server with model pre-loaded</span>
                    </div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse serve &lt;model&gt; -m 16GB</code>
                        <span class="cli-desc">Limit VRAM usage</span>
                    </div>
                    <div class="cli-group-title" style="margin-top:16px;">Model Operations</div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse chat &lt;model&gt;</code>
                        <span class="cli-desc">Interactive chat session</span>
                    </div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse info &lt;model&gt;</code>
                        <span class="cli-desc">Show model info & memory requirements</span>
                    </div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse benchmark &lt;model&gt;</code>
                        <span class="cli-desc">Run inference benchmarks</span>
                    </div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse convert &lt;model&gt; -o out.zse</code>
                        <span class="cli-desc">Convert to optimized format</span>
                    </div>
                    <div class="cli-group-title" style="margin-top:16px;">System</div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse hardware</code>
                        <span class="cli-desc">Show GPU and memory info</span>
                    </div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse api-key create &lt;name&gt;</code>
                        <span class="cli-desc">Create new API key</span>
                    </div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse api-key list</code>
                        <span class="cli-desc">List all API keys</span>
                    </div>
                    <div class="cli-row">
                        <code class="cli-cmd">zse api-key delete &lt;name&gt;</code>
                        <span class="cli-desc">Delete an API key</span>
                    </div>
                </div>
            </div>

            <!-- Authentication -->
            <div class="card" style="margin-top:24px;">
                <div class="card-title">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
                    </svg>
                    Authentication
                </div>
                <div class="doc-content">
                    <p>API key authentication is optional but recommended for production. Pass your key via header:</p>
                    <pre class="code-block">X-API-Key: your-api-key</pre>
                    <p style="margin-top:16px;">Or use OpenAI SDK format:</p>
                    <pre class="code-block">Authorization: Bearer your-api-key</pre>
                    <p style="margin-top:16px;color:var(--text-muted);font-size:13px;">
                        Manage authentication on the <strong>API Keys</strong> page or via CLI commands.
                        When auth is disabled, any value (or empty) is accepted.
                    </p>
                </div>
            </div>

            <!-- Memory Efficiency -->
            <div class="card" style="margin-top:24px;">
                <div class="card-title">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                    </svg>
                    Memory Efficiency Targets
                </div>
                <div class="memory-table">
                    <div class="memory-header">
                        <span>Model</span>
                        <span>FP16</span>
                        <span>ZSE Target</span>
                        <span>Savings</span>
                    </div>
                    <div class="memory-row">
                        <span>7B</span>
                        <span class="mem-fp16">14+ GB</span>
                        <span class="mem-zse">3-3.5 GB</span>
                        <span class="mem-save">~75%</span>
                    </div>
                    <div class="memory-row">
                        <span>14B</span>
                        <span class="mem-fp16">28+ GB</span>
                        <span class="mem-zse">6 GB</span>
                        <span class="mem-save">~78%</span>
                    </div>
                    <div class="memory-row">
                        <span>32B</span>
                        <span class="mem-fp16">64+ GB</span>
                        <span class="mem-zse">16-20 GB</span>
                        <span class="mem-save">~70%</span>
                    </div>
                    <div class="memory-row">
                        <span>70B</span>
                        <span class="mem-fp16">140+ GB</span>
                        <span class="mem-zse">24-32 GB</span>
                        <span class="mem-save">~77%</span>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        // State
        let ws = null;
        let chatWs = null;
        let requestsChart = null;
        let memoryChart = null;
        let authEnabled = true;
        
        // Page Navigation
        function showPage(page) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
            document.getElementById('page-' + page).classList.add('active');
            event.target.closest('.sidebar-item').classList.add('active');
            
            if (page === 'apikeys') loadApiKeys();
        }
        
        // Charts
        function initCharts() {
            const chartOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: '#222' }, ticks: { color: '#666', maxTicksLimit: 6 } },
                    y: { grid: { color: '#222' }, ticks: { color: '#666' }, beginAtZero: true }
                }
            };
            
            requestsChart = new Chart(document.getElementById('requests-chart').getContext('2d'), {
                type: 'line',
                data: { labels: [], datasets: [{ data: [], borderColor: '#fff', borderWidth: 2, tension: 0.3, fill: false, pointRadius: 0 }] },
                options: chartOptions
            });
            
            memoryChart = new Chart(document.getElementById('memory-chart').getContext('2d'), {
                type: 'line',
                data: { labels: [], datasets: [{ data: [], borderColor: '#fff', borderWidth: 2, tension: 0.3, fill: false, pointRadius: 0 }] },
                options: chartOptions
            });
        }
        
        // WebSocket
        function connectStats() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/stats`);
            
            ws.onmessage = (event) => updateStats(JSON.parse(event.data));
            ws.onerror = () => {
                document.getElementById('status-dot').style.background = 'var(--error)';
                document.getElementById('status-text').textContent = 'Disconnected';
            };
            ws.onclose = () => setTimeout(connectStats, 3000);
        }
        
        function updateStats(data) {
            const { system, analytics } = data;
            
            document.getElementById('total-requests').textContent = analytics.total_requests.toLocaleString();
            document.getElementById('total-tokens').textContent = analytics.total_tokens_generated.toLocaleString();
            document.getElementById('avg-latency').innerHTML = `${analytics.avg_latency_ms.toFixed(0)}<span style="font-size:16px;opacity:0.5">ms</span>`;
            document.getElementById('tokens-per-sec').textContent = analytics.avg_tokens_per_sec.toFixed(1);
            
            // GPU
            if (system.gpus && system.gpus.length > 0) {
                document.getElementById('gpu-stats').innerHTML = system.gpus.map(gpu => `
                    <div class="gpu-item">
                        <div class="gpu-header">
                            <span class="gpu-name">${gpu.name}</span>
                            <span class="gpu-usage">${gpu.used_memory_gb.toFixed(1)} / ${gpu.total_memory_gb.toFixed(1)} GB</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${gpu.utilization_percent}%"></div>
                        </div>
                    </div>
                `).join('');
                
                // Memory chart
                const now = new Date().toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit'});
                memoryChart.data.labels.push(now);
                memoryChart.data.datasets[0].data.push(system.gpus[0].used_memory_gb);
                if (memoryChart.data.labels.length > 20) {
                    memoryChart.data.labels.shift();
                    memoryChart.data.datasets[0].data.shift();
                }
                memoryChart.update('none');
            }
            
            // Models
            if (system.models_loaded && system.models_loaded.length > 0) {
                document.getElementById('models-list').innerHTML = system.models_loaded.map(m => `
                    <div class="model-item-with-actions">
                        <div class="model-info-row">
                            <span class="model-dot"></span>
                            <span>${m.name}</span>
                        </div>
                        <button class="btn-unload" onclick="unloadModel('${m.id}')" title="Unload model">×</button>
                    </div>
                `).join('');
                
                const select = document.getElementById('chat-model');
                const currentValue = select.value;
                select.innerHTML = '<option value="">Select a model...</option>' + 
                    system.models_loaded.map(m => `<option value="${m.name}">${m.name}</option>`).join('');
                select.value = currentValue;
            } else {
                document.getElementById('models-list').innerHTML = '<div class="empty-state">No models loaded</div>';
            }
            
            // Requests chart
            const now = new Date().toLocaleTimeString('en-US', {hour12: false, hour: '2-digit', minute: '2-digit'});
            requestsChart.data.labels.push(now);
            requestsChart.data.datasets[0].data.push(analytics.requests_per_minute);
            if (requestsChart.data.labels.length > 20) {
                requestsChart.data.labels.shift();
                requestsChart.data.datasets[0].data.shift();
            }
            requestsChart.update('none');
        }
        
        // Chat
        function sendChat() {
            const model = document.getElementById('chat-model').value;
            const input = document.getElementById('chat-input').value;
            const output = document.getElementById('chat-output');
            
            if (!model) { output.textContent = 'Please select a model first'; return; }
            if (!input.trim()) { output.textContent = 'Please enter a message'; return; }
            
            output.textContent = 'Generating...';
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            chatWs = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
            
            chatWs.onopen = () => {
                chatWs.send(JSON.stringify({ model, messages: [{ role: 'user', content: input }], max_tokens: 512 }));
                output.textContent = '';
            };
            
            chatWs.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'token') output.textContent += data.content;
                else if (data.type === 'done') chatWs.close();
                else if (data.type === 'error') output.textContent = 'Error: ' + data.message;
            };
        }
        
        // Model Management
        async function loadModel() {
            const modelName = document.getElementById('model-name-input').value.trim();
            const quant = document.getElementById('quant-select').value;
            const status = document.getElementById('load-status');
            
            if (!modelName) {
                status.textContent = 'Please enter a model name';
                status.className = 'load-status error';
                return;
            }
            
            status.textContent = 'Loading model... This may take a few minutes.';
            status.className = 'load-status loading';
            
            try {
                const res = await fetch('/api/models/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model_name: modelName,
                        quantization: quant
                    })
                });
                
                const data = await res.json();
                
                if (res.ok && data.success) {
                    status.textContent = `Model loaded! VRAM: ${data.vram_used_gb.toFixed(2)} GB, Time: ${data.load_time_sec.toFixed(1)}s`;
                    status.className = 'load-status success';
                    document.getElementById('model-name-input').value = '';
                } else {
                    status.textContent = 'Error: ' + (data.detail || data.message || 'Failed to load model');
                    status.className = 'load-status error';
                }
            } catch (e) {
                status.textContent = 'Error: ' + e.message;
                status.className = 'load-status error';
            }
        }
        
        async function unloadModel(modelId) {
            if (!confirm('Unload this model?')) return;
            
            try {
                const res = await fetch('/api/models/unload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_id: modelId })
                });
                
                if (res.ok) {
                    document.getElementById('load-status').textContent = 'Model unloaded';
                    document.getElementById('load-status').className = 'load-status success';
                }
            } catch (e) {
                alert('Failed to unload: ' + e.message);
            }
        }
        
        // API Keys
        async function loadApiKeys() {
            try {
                const res = await fetch('/api/keys');
                const data = await res.json();
                authEnabled = data.enabled;
                
                const toggle = document.getElementById('auth-toggle');
                toggle.classList.toggle('active', authEnabled);
                document.getElementById('auth-status').textContent = authEnabled ? 'Enabled' : 'Disabled';
                
                if (data.keys.length === 0) {
                    document.getElementById('keys-list').innerHTML = `
                        <div class="empty-state">
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
                            </svg>
                            <div>No API keys created yet</div>
                        </div>
                    `;
                } else {
                    document.getElementById('keys-list').innerHTML = data.keys.map(key => `
                        <div class="key-item">
                            <div class="key-info">
                                <div class="key-icon">
                                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
                                    </svg>
                                </div>
                                <div>
                                    <div class="key-name">${key.name}</div>
                                    <div class="key-meta">Created ${key.created_at.split('T')[0]}</div>
                                </div>
                            </div>
                            <div class="key-stats">
                                <div class="key-requests">${key.request_count.toLocaleString()} requests</div>
                                <div class="key-last-used">Last used: ${key.last_used ? key.last_used.split('T')[0] : 'Never'}</div>
                            </div>
                            <button class="btn btn-outline btn-sm" onclick="deleteKey('${key.name}')">Delete</button>
                        </div>
                    `).join('');
                }
            } catch (e) {
                console.error('Failed to load API keys:', e);
            }
        }
        
        async function createKey() {
            const name = document.getElementById('new-key-name').value.trim();
            if (!name) { alert('Please enter a key name'); return; }
            
            try {
                const res = await fetch('/api/keys', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
                const data = await res.json();
                
                document.getElementById('new-key-value').textContent = data.key;
                document.getElementById('new-key-display').style.display = 'block';
                document.getElementById('new-key-name').value = '';
                loadApiKeys();
            } catch (e) {
                alert('Failed to create key: ' + e.message);
            }
        }
        
        async function deleteKey(name) {
            if (!confirm(`Delete API key "${name}"?`)) return;
            
            try {
                await fetch(`/api/keys/${name}`, { method: 'DELETE' });
                loadApiKeys();
            } catch (e) {
                alert('Failed to delete key: ' + e.message);
            }
        }
        
        async function toggleAuth() {
            authEnabled = !authEnabled;
            try {
                await fetch('/api/keys/auth', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: authEnabled })
                });
                document.getElementById('auth-toggle').classList.toggle('active', authEnabled);
                document.getElementById('auth-status').textContent = authEnabled ? 'Enabled' : 'Disabled';
            } catch (e) {
                authEnabled = !authEnabled;
                alert('Failed to toggle auth: ' + e.message);
            }
        }
        
        // Init
        initCharts();
        connectStats();
    </script>
</body>
</html>'''


# =============================================================================
# Audit Routes
# =============================================================================

def _register_audit_routes(app: FastAPI):
    """Register audit logging endpoints."""
    
    @app.get("/api/audit/summary", tags=["Audit"])
    async def get_audit_summary(
        hours: int = Query(default=24, ge=1, le=720, description="Hours to look back")
    ):
        """
        Get audit log summary for the specified time period.
        
        Returns aggregated statistics including:
        - Total requests
        - Unique API keys
        - Top endpoints
        - Status code distribution
        - Average latency
        """
        logger = get_audit_logger()
        return logger.get_summary(hours=hours, include_rotated=True)
    
    @app.get("/api/audit/recent", tags=["Audit"])
    async def get_recent_audit_logs(
        limit: int = Query(default=50, ge=1, le=500, description="Number of entries")
    ):
        """Get recent audit log entries from memory buffer."""
        logger = get_audit_logger()
        entries = logger.get_recent(limit)
        return {
            "count": len(entries),
            "entries": [e.to_dict() for e in entries]
        }
    
    @app.get("/api/audit/query", tags=["Audit"])
    async def query_audit_logs(
        hours: Optional[int] = Query(default=None, description="Hours to look back"),
        api_key: Optional[str] = Query(default=None, description="Filter by API key name"),
        path: Optional[str] = Query(default=None, description="Filter by path prefix"),
        status: Optional[int] = Query(default=None, description="Filter by status code"),
        method: Optional[str] = Query(default=None, description="Filter by HTTP method"),
        min_latency: Optional[float] = Query(default=None, description="Min latency in ms"),
        limit: int = Query(default=100, ge=1, le=1000, description="Max results"),
    ):
        """
        Query audit logs with filters.
        
        Searches both current and rotated log files.
        """
        from datetime import datetime, timedelta
        
        logger = get_audit_logger()
        start_time = datetime.now() - timedelta(hours=hours) if hours else None
        status_codes = [status] if status else None
        
        entries = logger.query(
            start_time=start_time,
            api_key_name=api_key,
            path_prefix=path,
            status_codes=status_codes,
            method=method,
            min_latency_ms=min_latency,
            limit=limit,
            include_rotated=True,
        )
        
        return {
            "count": len(entries),
            "entries": entries
        }
    
    @app.get("/api/audit/stats", tags=["Audit"])
    async def get_audit_stats():
        """Get audit logging system statistics."""
        logger = get_audit_logger()
        return logger.get_stats()
    
    @app.delete("/api/audit/clear", tags=["Audit"])
    async def clear_audit_logs(
        all_logs: bool = Query(default=False, description="Clear rotated logs too")
    ):
        """
        Clear audit logs.
        
        WARNING: This permanently deletes log data.
        """
        logger = get_audit_logger()
        logger.clear(include_rotated=all_logs)
        return {"status": "cleared", "all_logs": all_logs}


# Create the app instance
app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
):
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "zse.api.server.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers
    )


if __name__ == "__main__":
    run_server()
