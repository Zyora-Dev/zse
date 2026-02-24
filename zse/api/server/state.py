"""
Server State Management

Manages loaded models, request tracking, and analytics.
Thread-safe state management for the API server.
"""

import time
import threading
import uuid
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import torch

from zse.api.server.models import (
    RequestMetrics, GPUStats, SystemStats, 
    AnalyticsOverview, AnalyticsTimeSeries
)


@dataclass
class LoadedModel:
    """Information about a loaded model."""
    model_id: str
    model_name: str
    quantization: str
    vram_used_gb: float
    load_time: datetime
    orchestrator: Any  # IntelligenceOrchestrator
    request_count: int = 0
    tokens_generated: int = 0


class ServerState:
    """
    Thread-safe server state management.
    
    Tracks:
    - Loaded models
    - Request history
    - Analytics metrics
    """
    
    def __init__(self, max_history: int = 10000):
        self._lock = threading.RLock()
        self._models: Dict[str, LoadedModel] = {}
        self._request_history: deque = deque(maxlen=max_history)
        self._start_time = time.time()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._peak_memory_gb = 0.0
        
        # Time series data (1 minute buckets, last 60 minutes)
        self._time_series_buckets: deque = deque(maxlen=60)
        self._current_bucket_start = time.time()
        self._current_bucket_requests = 0
        self._current_bucket_tokens = 0
        self._current_bucket_latency_sum = 0.0
        
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time
    
    def generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"zse-{uuid.uuid4().hex[:12]}"
    
    def generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID."""
        # Sanitize model name for ID
        safe_name = model_name.replace("/", "-").replace(".", "-").lower()
        return f"{safe_name}-{uuid.uuid4().hex[:6]}"
    
    # =========================================================================
    # Model Management
    # =========================================================================
    
    def add_model(
        self, 
        model_id: str,
        model_name: str, 
        quantization: str,
        vram_used_gb: float,
        orchestrator: Any
    ) -> LoadedModel:
        """Register a loaded model."""
        with self._lock:
            loaded = LoadedModel(
                model_id=model_id,
                model_name=model_name,
                quantization=quantization,
                vram_used_gb=vram_used_gb,
                load_time=datetime.now(),
                orchestrator=orchestrator
            )
            self._models[model_id] = loaded
            return loaded
    
    def remove_model(self, model_id: str) -> Optional[LoadedModel]:
        """Remove a model from state."""
        with self._lock:
            return self._models.pop(model_id, None)
    
    def get_model(self, model_id: str) -> Optional[LoadedModel]:
        """Get a loaded model by ID."""
        with self._lock:
            return self._models.get(model_id)
    
    def get_model_by_name(self, model_name: str) -> Optional[LoadedModel]:
        """Get a loaded model by name (returns first match)."""
        with self._lock:
            for model in self._models.values():
                if model.model_name == model_name or model.model_id == model_name:
                    return model
            return None
    
    def list_models(self) -> List[LoadedModel]:
        """List all loaded models."""
        with self._lock:
            return list(self._models.values())
    
    def model_count(self) -> int:
        """Get number of loaded models."""
        with self._lock:
            return len(self._models)
    
    # =========================================================================
    # Request Tracking
    # =========================================================================
    
    def record_request(
        self,
        request_id: str,
        model: str,
        endpoint: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        status: str,
        user: Optional[str] = None
    ):
        """Record a completed request."""
        with self._lock:
            metrics = RequestMetrics(
                request_id=request_id,
                model=model,
                endpoint=endpoint,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                tokens_per_sec=completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0,
                timestamp=datetime.now(),
                status=status,
                user=user
            )
            
            self._request_history.append(metrics)
            self._total_requests += 1
            
            if status == "success":
                self._successful_requests += 1
                self._total_tokens += completion_tokens
                self._total_prompt_tokens += prompt_tokens
            else:
                self._failed_requests += 1
            
            # Update model stats
            loaded_model = self.get_model_by_name(model)
            if loaded_model:
                loaded_model.request_count += 1
                loaded_model.tokens_generated += completion_tokens
            
            # Update time series bucket
            self._update_time_bucket(latency_ms, completion_tokens)
            
            # Track peak memory
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / (1024**3)
                self._peak_memory_gb = max(self._peak_memory_gb, current_mem)
    
    def _update_time_bucket(self, latency_ms: float, tokens: int):
        """Update current time series bucket."""
        now = time.time()
        bucket_duration = 60  # 1 minute buckets
        
        # Check if we need to start a new bucket
        if now - self._current_bucket_start >= bucket_duration:
            # Save current bucket
            if self._current_bucket_requests > 0:
                avg_latency = self._current_bucket_latency_sum / self._current_bucket_requests
            else:
                avg_latency = 0
            
            self._time_series_buckets.append({
                "timestamp": datetime.fromtimestamp(self._current_bucket_start),
                "requests": self._current_bucket_requests,
                "tokens": self._current_bucket_tokens,
                "avg_latency_ms": avg_latency,
                "memory_gb": self._get_gpu_memory_used()
            })
            
            # Start new bucket
            self._current_bucket_start = now
            self._current_bucket_requests = 0
            self._current_bucket_tokens = 0
            self._current_bucket_latency_sum = 0.0
        
        # Update current bucket
        self._current_bucket_requests += 1
        self._current_bucket_tokens += tokens
        self._current_bucket_latency_sum += latency_ms
    
    def get_recent_requests(self, limit: int = 100) -> List[RequestMetrics]:
        """Get recent requests."""
        with self._lock:
            return list(self._request_history)[-limit:]
    
    # =========================================================================
    # Analytics
    # =========================================================================
    
    def get_analytics_overview(self) -> AnalyticsOverview:
        """Get analytics overview."""
        with self._lock:
            # Calculate requests per minute
            uptime_minutes = self.uptime_seconds / 60
            rpm = self._total_requests / uptime_minutes if uptime_minutes > 0 else 0
            
            # Calculate averages from recent requests
            recent = list(self._request_history)[-1000:]
            if recent:
                avg_latency = sum(r.latency_ms for r in recent) / len(recent)
                avg_tps = sum(r.tokens_per_sec for r in recent) / len(recent)
            else:
                avg_latency = 0
                avg_tps = 0
            
            # Model usage
            models_used = {}
            for r in self._request_history:
                models_used[r.model] = models_used.get(r.model, 0) + 1
            
            return AnalyticsOverview(
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                total_tokens_generated=self._total_tokens,
                total_prompt_tokens=self._total_prompt_tokens,
                avg_latency_ms=avg_latency,
                avg_tokens_per_sec=avg_tps,
                requests_per_minute=rpm,
                peak_memory_gb=self._peak_memory_gb,
                uptime_seconds=self.uptime_seconds,
                models_used=models_used
            )
    
    def get_analytics_timeseries(self) -> AnalyticsTimeSeries:
        """Get time series analytics data."""
        with self._lock:
            buckets = list(self._time_series_buckets)
            
            return AnalyticsTimeSeries(
                timestamps=[b["timestamp"] for b in buckets],
                requests_per_minute=[b["requests"] for b in buckets],
                tokens_per_second=[b["tokens"] / 60 for b in buckets],  # tokens per sec
                memory_usage_gb=[b["memory_gb"] for b in buckets],
                latency_ms=[b["avg_latency_ms"] for b in buckets]
            )
    
    # =========================================================================
    # System Stats
    # =========================================================================
    
    def _get_gpu_memory_used(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def get_gpu_stats(self) -> List[GPUStats]:
        """Get GPU statistics."""
        stats = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_info = torch.cuda.mem_get_info(i)
                
                free_gb = mem_info[0] / (1024**3)
                total_gb = mem_info[1] / (1024**3)
                used_gb = total_gb - free_gb
                
                stats.append(GPUStats(
                    name=props.name,
                    index=i,
                    total_memory_gb=round(total_gb, 2),
                    used_memory_gb=round(used_gb, 2),
                    free_memory_gb=round(free_gb, 2),
                    utilization_percent=round((used_gb / total_gb) * 100, 1),
                    temperature_celsius=None  # Would need pynvml
                ))
        
        return stats
    
    def get_system_stats(self) -> SystemStats:
        """Get system statistics."""
        import socket
        from .models import LoadedModelInfo
        
        memory = psutil.virtual_memory()
        
        return SystemStats(
            hostname=socket.gethostname(),
            cpu_percent=psutil.cpu_percent(),
            memory_percent=memory.percent,
            memory_used_gb=round(memory.used / (1024**3), 2),
            memory_total_gb=round(memory.total / (1024**3), 2),
            gpus=self.get_gpu_stats(),
            models_loaded=[LoadedModelInfo(id=m.model_id, name=m.model_name) for m in self.list_models()],
            uptime_seconds=self.uptime_seconds
        )


# Global server state instance
server_state = ServerState()
