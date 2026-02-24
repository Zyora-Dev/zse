"""
Pydantic models for ZSE API.

OpenAI-compatible request/response models plus ZSE-specific models.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum


# ============================================================================
# OpenAI-Compatible Models
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    echo: bool = False


class ChatCompletionChoice(BaseModel):
    """Single choice in chat completion."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class CompletionChoice(BaseModel):
    """Single choice in completion."""
    index: int
    text: str
    finish_reason: Optional[str] = None


class UsageStats(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageStats


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageStats


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for chat completion."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "zse"


class ModelListResponse(BaseModel):
    """List of available models."""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# ZSE-Specific Models
# ============================================================================

class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_config = {"protected_namespaces": ()}  # Allow model_ fields
    
    model_name: str
    quantization: Literal["auto", "fp16", "int8", "int4"] = "auto"
    target_vram_gb: Optional[float] = None
    device: str = "auto"  # "auto", "cuda", "cpu", or "cuda:N"


class LoadModelResponse(BaseModel):
    """Response after loading a model."""
    model_config = {"protected_namespaces": ()}  # Allow model_ fields
    
    success: bool
    model_id: str
    model_name: str
    quantization: str
    device: str = "cuda"  # "cuda", "cpu", or "cuda:N"
    vram_used_gb: float  # VRAM (GPU) or RAM (CPU) in GB
    load_time_sec: float
    message: str


class UnloadModelRequest(BaseModel):
    """Request to unload a model."""
    model_config = {"protected_namespaces": ()}  # Allow model_ fields
    
    model_id: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    uptime_seconds: float
    models_loaded: int
    gpu_available: bool


class GPUStats(BaseModel):
    """GPU statistics."""
    name: str
    index: int
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float
    utilization_percent: float
    temperature_celsius: Optional[float] = None


class LoadedModelInfo(BaseModel):
    """Info about a loaded model for dashboard."""
    id: str
    name: str


class SystemStats(BaseModel):
    """System statistics."""
    hostname: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpus: List[GPUStats]
    models_loaded: List[LoadedModelInfo]
    uptime_seconds: float


class RequestMetrics(BaseModel):
    """Metrics for a single request."""
    request_id: str
    model: str
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    tokens_per_sec: float
    timestamp: datetime
    status: str
    user: Optional[str] = None


class AnalyticsOverview(BaseModel):
    """Analytics overview."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_generated: int
    total_prompt_tokens: int
    avg_latency_ms: float
    avg_tokens_per_sec: float
    requests_per_minute: float
    peak_memory_gb: float
    uptime_seconds: float
    models_used: Dict[str, int]  # model -> request count


class AnalyticsTimeSeries(BaseModel):
    """Time series analytics data."""
    timestamps: List[datetime]
    requests_per_minute: List[float]
    tokens_per_second: List[float]
    memory_usage_gb: List[float]
    latency_ms: List[float]


class ErrorResponse(BaseModel):
    """Error response."""
    error: Dict[str, Any]
