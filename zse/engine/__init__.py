"""
ZSE Engine Module

Core execution engine components:
- kv_cache: Paged KV cache for efficient generation
- scheduler: Continuous batching for high throughput
- generation: Text generation with streaming output
- executor: Model execution and token generation
- router: Route requests to native engine or GGUF backend
- orchestrator: Smart memory recommendations and auto-configuration
"""

# KV Cache
from zse.engine.kv_cache import (
    KVCache,
    KVCacheConfig,
    KVCacheManager,
    PagedKVCache,
)

# Continuous Batching Scheduler
from zse.engine.scheduler import (
    InferenceRequest,
    RequestStatus,
    GenerationConfig,
    SchedulerConfig,
    ContinuousBatchingScheduler,
    InferenceEngine,
)

# Text Generation
from zse.engine.generation import (
    SamplingParams,
    GenerationOutput,
    StreamChunk,
    Sampler,
    StopChecker,
    TextGenerator,
    CachedTextGenerator,
    StreamingCallback,
    PrintStreamCallback,
    BatchGenerator,
    BatchGenerationRequest,
    BatchGenerationOutput,
)

# Intelligence Orchestrator
from zse.engine.orchestrator import (
    OptimizationMode,
    ModelConfig,
    InferenceStats,
    IntelligenceOrchestrator,
    load_model,
    estimate_requirements,
)

# Async Batching Engine
from zse.engine.batching import (
    BatchRequest,
    BatchRequestStatus,
    BatchConfig,
    BatchingEngine,
    ServerBatchingEngine,
    get_batching_engine,
)

__all__ = [
    # KV Cache
    "KVCache",
    "KVCacheConfig",
    "KVCacheManager",
    "PagedKVCache",
    # Scheduler
    "InferenceRequest",
    "RequestStatus",
    "GenerationConfig",
    "SchedulerConfig",
    "ContinuousBatchingScheduler",
    "InferenceEngine",
    # Generation
    "SamplingParams",
    "GenerationOutput",
    "StreamChunk",
    "Sampler",
    "StopChecker",
    "TextGenerator",
    "CachedTextGenerator",
    "StreamingCallback",
    "PrintStreamCallback",
    "BatchGenerator",
    "BatchGenerationRequest",
    "BatchGenerationOutput",
    # Orchestrator
    "OptimizationMode",
    "ModelConfig",
    "InferenceStats",
    "IntelligenceOrchestrator",
    "load_model",
    "estimate_requirements",
    # Batching Engine
    "BatchRequest",
    "BatchRequestStatus",
    "BatchConfig",
    "BatchingEngine",
    "ServerBatchingEngine",
    "get_batching_engine",
    # Legacy
    "executor",
    "router",
]
