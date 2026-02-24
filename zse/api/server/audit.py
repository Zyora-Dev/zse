"""
ZSE Request Audit Logging

Comprehensive request logging for compliance, debugging, and analytics.
Stores structured logs in JSON Lines format with automatic rotation.
"""

import os
import json
import time
import uuid
import gzip
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import deque
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

# Default paths
DEFAULT_AUDIT_DIR = Path.home() / ".zse" / "audit"
DEFAULT_LOG_FILE = DEFAULT_AUDIT_DIR / "requests.jsonl"
MAX_BODY_LOG_SIZE = 4096  # Maximum request/response body size to log
MAX_LOG_FILE_SIZE = 50 * 1024 * 1024  # 50MB before rotation
MAX_ROTATED_FILES = 10  # Keep last 10 rotated files


@dataclass
class AuditLogEntry:
    """
    A single audit log entry capturing request/response details.
    """
    # Request identification
    request_id: str
    timestamp: str
    timestamp_unix: float
    
    # Request details
    method: str
    path: str
    query_params: Dict[str, str]
    client_ip: str
    user_agent: str
    
    # Authentication
    api_key_name: Optional[str] = None
    api_key_hash_prefix: Optional[str] = None  # First 8 chars of hash for correlation
    
    # Request body (truncated if large)
    request_body: Optional[str] = None
    request_body_truncated: bool = False
    request_content_type: Optional[str] = None
    request_content_length: Optional[int] = None
    
    # Response details
    status_code: int = 0
    response_content_type: Optional[str] = None
    response_content_length: Optional[int] = None
    
    # Timing
    latency_ms: float = 0.0
    
    # Additional context
    model_id: Optional[str] = None
    endpoint_type: Optional[str] = None  # "chat", "completion", "models", etc.
    error_message: Optional[str] = None
    
    # Token usage (for inference endpoints)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'))


class AuditLogger:
    """
    Thread-safe audit logger with file rotation and compression.
    
    Features:
    - JSON Lines format for easy parsing
    - Automatic log rotation when file exceeds size limit
    - Optional gzip compression of rotated files
    - In-memory buffer for recent entries
    - Thread-safe writes
    """
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_file_size: int = MAX_LOG_FILE_SIZE,
        max_rotated_files: int = MAX_ROTATED_FILES,
        compress_rotated: bool = True,
        buffer_size: int = 1000,
        enabled: bool = True,
        log_request_body: bool = True,
        log_sensitive_fields: bool = False,
    ):
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_AUDIT_DIR
        self.log_file = self.log_dir / "requests.jsonl"
        self.max_file_size = max_file_size
        self.max_rotated_files = max_rotated_files
        self.compress_rotated = compress_rotated
        self.buffer_size = buffer_size
        self.enabled = enabled
        self.log_request_body = log_request_body
        self.log_sensitive_fields = log_sensitive_fields
        
        # Thread safety
        self._lock = threading.Lock()
        
        # In-memory buffer for recent entries
        self._buffer: deque[AuditLogEntry] = deque(maxlen=buffer_size)
        
        # Statistics
        self._stats = {
            "total_logged": 0,
            "errors": 0,
            "rotations": 0,
        }
        
        # Ensure directory exists
        self._ensure_dir()
    
    def _ensure_dir(self):
        """Ensure the audit log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _should_rotate(self) -> bool:
        """Check if log file should be rotated."""
        if not self.log_file.exists():
            return False
        return self.log_file.stat().st_size >= self.max_file_size
    
    def _rotate_log(self):
        """Rotate the current log file."""
        if not self.log_file.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.compress_rotated:
            rotated_path = self.log_dir / f"requests_{timestamp}.jsonl.gz"
            with open(self.log_file, 'rb') as f_in:
                with gzip.open(rotated_path, 'wb') as f_out:
                    f_out.writelines(f_in)
        else:
            rotated_path = self.log_dir / f"requests_{timestamp}.jsonl"
            self.log_file.rename(rotated_path)
        
        # Remove old rotated files
        self._cleanup_old_logs()
        
        self._stats["rotations"] += 1
    
    def _cleanup_old_logs(self):
        """Remove old rotated log files beyond the limit."""
        pattern = "requests_*.jsonl*"
        rotated_files = sorted(self.log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        
        for old_file in rotated_files[self.max_rotated_files:]:
            try:
                old_file.unlink()
            except OSError:
                pass
    
    def log(self, entry: AuditLogEntry):
        """
        Log an audit entry.
        
        Thread-safe. Handles rotation automatically.
        """
        if not self.enabled:
            return
        
        with self._lock:
            try:
                # Add to in-memory buffer
                self._buffer.append(entry)
                
                # Check rotation
                if self._should_rotate():
                    self._rotate_log()
                
                # Write to file
                self._ensure_dir()
                with open(self.log_file, 'a') as f:
                    f.write(entry.to_json() + '\n')
                
                self._stats["total_logged"] += 1
                
            except Exception as e:
                self._stats["errors"] += 1
                # Don't raise - audit logging should not affect request processing
    
    def get_recent(self, count: int = 100) -> List[AuditLogEntry]:
        """Get recent entries from the in-memory buffer."""
        with self._lock:
            entries = list(self._buffer)
            return entries[-count:] if len(entries) > count else entries
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        api_key_name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        status_codes: Optional[List[int]] = None,
        method: Optional[str] = None,
        min_latency_ms: Optional[float] = None,
        limit: int = 1000,
        include_rotated: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs with filters.
        
        Args:
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            api_key_name: Filter by API key name
            path_prefix: Filter by path prefix
            status_codes: Filter by status codes
            method: Filter by HTTP method
            min_latency_ms: Filter by minimum latency
            limit: Maximum entries to return
            include_rotated: Include rotated (older) log files
            
        Returns:
            List of matching log entries as dicts
        """
        results = []
        files_to_search = [self.log_file] if self.log_file.exists() else []
        
        if include_rotated:
            rotated = sorted(self.log_dir.glob("requests_*.jsonl*"), key=lambda p: p.stat().st_mtime, reverse=True)
            files_to_search.extend(rotated)
        
        for log_path in files_to_search:
            if len(results) >= limit:
                break
            
            try:
                opener = gzip.open if log_path.suffix == '.gz' else open
                mode = 'rt' if log_path.suffix == '.gz' else 'r'
                
                with opener(log_path, mode) as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                        
                        try:
                            entry = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue
                        
                        # Apply filters
                        if start_time and entry.get("timestamp_unix", 0) < start_time.timestamp():
                            continue
                        if end_time and entry.get("timestamp_unix", 0) > end_time.timestamp():
                            continue
                        if api_key_name and entry.get("api_key_name") != api_key_name:
                            continue
                        if path_prefix and not entry.get("path", "").startswith(path_prefix):
                            continue
                        if status_codes and entry.get("status_code") not in status_codes:
                            continue
                        if method and entry.get("method") != method:
                            continue
                        if min_latency_ms and entry.get("latency_ms", 0) < min_latency_ms:
                            continue
                        
                        results.append(entry)
                        
            except Exception:
                continue
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit logging statistics."""
        with self._lock:
            stats = dict(self._stats)
            stats["buffer_size"] = len(self._buffer)
            stats["log_file"] = str(self.log_file)
            stats["log_file_size_mb"] = (
                self.log_file.stat().st_size / (1024 * 1024)
                if self.log_file.exists() else 0
            )
            
            # Count rotated files
            rotated = list(self.log_dir.glob("requests_*.jsonl*"))
            stats["rotated_files"] = len(rotated)
            
            return stats
    
    def get_summary(
        self,
        hours: int = 24,
        include_rotated: bool = False,
    ) -> Dict[str, Any]:
        """
        Get a summary of recent audit logs.
        
        Returns aggregated statistics for the time period.
        """
        start_time = datetime.now() - timedelta(hours=hours)
        entries = self.query(start_time=start_time, limit=100000, include_rotated=include_rotated)
        
        if not entries:
            return {
                "period_hours": hours,
                "total_requests": 0,
                "unique_keys": 0,
                "endpoints": {},
                "status_codes": {},
                "avg_latency_ms": 0,
                "total_tokens": 0,
                "errors": 0,
            }
        
        # Aggregate
        endpoints: Dict[str, int] = {}
        status_codes: Dict[int, int] = {}
        api_keys: set = set()
        total_latency = 0.0
        total_tokens = 0
        
        for entry in entries:
            path = entry.get("path", "unknown")
            endpoints[path] = endpoints.get(path, 0) + 1
            
            status = entry.get("status_code", 0)
            status_codes[status] = status_codes.get(status, 0) + 1
            
            if entry.get("api_key_name"):
                api_keys.add(entry["api_key_name"])
            
            total_latency += entry.get("latency_ms", 0)
            total_tokens += entry.get("total_tokens", 0) or 0
        
        return {
            "period_hours": hours,
            "total_requests": len(entries),
            "unique_keys": len(api_keys),
            "endpoints": dict(sorted(endpoints.items(), key=lambda x: x[1], reverse=True)[:10]),
            "status_codes": status_codes,
            "avg_latency_ms": round(total_latency / len(entries), 2) if entries else 0,
            "total_tokens": total_tokens,
            "errors": status_codes.get(500, 0) + status_codes.get(400, 0) + status_codes.get(429, 0),
        }
    
    def clear(self, include_rotated: bool = False):
        """Clear audit logs."""
        with self._lock:
            self._buffer.clear()
            
            if self.log_file.exists():
                self.log_file.unlink()
            
            if include_rotated:
                for rotated in self.log_dir.glob("requests_*.jsonl*"):
                    try:
                        rotated.unlink()
                    except OSError:
                        pass
            
            self._stats["total_logged"] = 0
    
    def export(
        self,
        output_path: Path,
        format: str = "jsonl",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        compress: bool = False,
    ) -> int:
        """
        Export audit logs to a file.
        
        Args:
            output_path: Output file path
            format: "jsonl" or "csv"
            start_time: Filter start time
            end_time: Filter end time
            compress: Gzip compress output
            
        Returns:
            Number of entries exported
        """
        entries = self.query(
            start_time=start_time,
            end_time=end_time,
            limit=1000000,
            include_rotated=True
        )
        
        if not entries:
            return 0
        
        opener = gzip.open if compress else open
        mode = 'wt' if compress else 'w'
        
        with opener(output_path, mode) as f:
            if format == "csv":
                import csv
                if entries:
                    writer = csv.DictWriter(f, fieldnames=entries[0].keys())
                    writer.writeheader()
                    writer.writerows(entries)
            else:
                for entry in entries:
                    f.write(json.dumps(entry, separators=(',', ':')) + '\n')
        
        return len(entries)


# =============================================================================
# Middleware
# =============================================================================

class AuditMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic request auditing.
    
    Captures request/response details and logs them via AuditLogger.
    """
    
    # Paths to exclude from logging
    EXCLUDE_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }
    
    # Paths with sensitive content (don't log body)
    SENSITIVE_PATHS = {
        "/v1/api-keys",
    }
    
    def __init__(self, app: ASGIApp, logger: AuditLogger):
        super().__init__(app)
        self.logger = logger
    
    def _classify_endpoint(self, path: str, method: str) -> str:
        """Classify the endpoint type for analytics."""
        if "/chat/completions" in path:
            return "chat"
        elif "/completions" in path:
            return "completion"
        elif "/models" in path:
            if method == "POST":
                return "model_load"
            elif method == "DELETE":
                return "model_unload"
            return "models"
        elif "/embeddings" in path:
            return "embeddings"
        elif "/health" in path:
            return "health"
        elif "/metrics" in path or "/analytics" in path:
            return "monitoring"
        elif "/api-key" in path:
            return "auth"
        else:
            return "other"
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process the request and log audit details."""
        
        # Skip excluded paths
        if request.url.path in self.EXCLUDE_PATHS:
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Extract request details
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        query_params = dict(request.query_params)
        
        # Get API key info from request state (set by auth middleware)
        api_key_name = getattr(request.state, "api_key_name", None) if hasattr(request, "state") else None
        api_key_hash = getattr(request.state, "api_key_hash", None) if hasattr(request, "state") else None
        
        # Read request body if needed
        request_body = None
        request_body_truncated = False
        content_type = request.headers.get("content-type", "")
        content_length = int(request.headers.get("content-length", 0) or 0)
        
        if (
            self.logger.log_request_body
            and request.url.path not in self.SENSITIVE_PATHS
            and content_length > 0
            and content_length <= MAX_BODY_LOG_SIZE
            and "json" in content_type
        ):
            try:
                body_bytes = await request.body()
                request_body = body_bytes.decode("utf-8", errors="replace")
                if len(request_body) > MAX_BODY_LOG_SIZE:
                    request_body = request_body[:MAX_BODY_LOG_SIZE]
                    request_body_truncated = True
            except Exception:
                pass
        elif content_length > MAX_BODY_LOG_SIZE:
            request_body_truncated = True
        
        # Call the actual endpoint
        response = None
        error_message = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract model ID from path or body
            model_id = None
            if request_body:
                try:
                    body_json = json.loads(request_body)
                    model_id = body_json.get("model")
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            # Extract token usage from response (if available in request state)
            prompt_tokens = getattr(request.state, "prompt_tokens", None) if hasattr(request, "state") else None
            completion_tokens = getattr(request.state, "completion_tokens", None) if hasattr(request, "state") else None
            total_tokens = getattr(request.state, "total_tokens", None) if hasattr(request, "state") else None
            
            # Create audit entry
            entry = AuditLogEntry(
                request_id=request_id,
                timestamp=timestamp,
                timestamp_unix=start_time,
                method=request.method,
                path=request.url.path,
                query_params=query_params,
                client_ip=client_ip,
                user_agent=user_agent,
                api_key_name=api_key_name,
                api_key_hash_prefix=api_key_hash[:8] if api_key_hash else None,
                request_body=request_body if not self.SENSITIVE_PATHS else None,
                request_body_truncated=request_body_truncated,
                request_content_type=content_type if content_type else None,
                request_content_length=content_length if content_length else None,
                status_code=response.status_code if response else 500,
                response_content_type=response.headers.get("content-type") if response else None,
                latency_ms=round(latency_ms, 2),
                model_id=model_id,
                endpoint_type=self._classify_endpoint(request.url.path, request.method),
                error_message=error_message,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            
            # Log asynchronously to not block response
            self.logger.log(entry)
        
        return response


# =============================================================================
# Global Instance
# =============================================================================

_audit_logger: Optional[AuditLogger] = None
_audit_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        with _audit_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger()
    return _audit_logger


def configure_audit_logger(
    enabled: bool = True,
    log_dir: Optional[Path] = None,
    log_request_body: bool = True,
    max_file_size_mb: int = 50,
    max_rotated_files: int = 10,
) -> AuditLogger:
    """Configure and return the global audit logger."""
    global _audit_logger
    with _audit_lock:
        _audit_logger = AuditLogger(
            enabled=enabled,
            log_dir=log_dir,
            log_request_body=log_request_body,
            max_file_size=max_file_size_mb * 1024 * 1024,
            max_rotated_files=max_rotated_files,
        )
    return _audit_logger


def add_audit_middleware(app, enabled: bool = True):
    """Add audit middleware to a FastAPI app."""
    if enabled:
        logger = get_audit_logger()
        logger.enabled = enabled
        app.add_middleware(AuditMiddleware, logger=logger)


# =============================================================================
# CLI helpers
# =============================================================================

def get_audit_summary(hours: int = 24) -> Dict[str, Any]:
    """Get audit summary for CLI."""
    return get_audit_logger().get_summary(hours=hours, include_rotated=True)


def get_recent_requests(count: int = 50) -> List[Dict[str, Any]]:
    """Get recent requests for CLI."""
    entries = get_audit_logger().get_recent(count)
    return [e.to_dict() for e in entries]


def query_audit_logs(
    hours: Optional[int] = None,
    api_key: Optional[str] = None,
    path: Optional[str] = None,
    status: Optional[int] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Query audit logs for CLI."""
    start_time = datetime.now() - timedelta(hours=hours) if hours else None
    status_codes = [status] if status else None
    
    return get_audit_logger().query(
        start_time=start_time,
        api_key_name=api_key,
        path_prefix=path,
        status_codes=status_codes,
        limit=limit,
        include_rotated=True,
    )


def export_audit_logs(
    output: str,
    format: str = "jsonl",
    hours: Optional[int] = None,
    compress: bool = False,
) -> int:
    """Export audit logs for CLI."""
    start_time = datetime.now() - timedelta(hours=hours) if hours else None
    output_path = Path(output)
    
    return get_audit_logger().export(
        output_path=output_path,
        format=format,
        start_time=start_time,
        compress=compress,
    )


def clear_audit_logs(all_logs: bool = False):
    """Clear audit logs."""
    get_audit_logger().clear(include_rotated=all_logs)
