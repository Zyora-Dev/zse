"""
Async Prefetcher for zStream

Implements intelligent prefetching strategies to hide data transfer latency
by overlapping compute with memory transfers.

Key Strategies:
- Sequential prefetch: Next N layers
- Pattern-based: Learn access patterns
- Priority-based: High-value layers first
"""

import torch
import threading
import queue
from typing import Optional, Set, List, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque


class PrefetchStrategy(Enum):
    """Prefetch strategies."""
    SEQUENTIAL = "sequential"    # Prefetch next N layers
    ADAPTIVE = "adaptive"        # Learn from access patterns
    PRIORITY = "priority"        # Prioritize certain layers


@dataclass
class PrefetchRequest:
    """A prefetch request."""
    layer_idx: int
    priority: int = 0  # Higher = more urgent
    deadline: float = 0.0  # When this layer will be needed
    
    def __lt__(self, other):
        # Higher priority first, earlier deadline first
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.deadline < other.deadline


class AsyncPrefetcher:
    """
    Async layer prefetcher with intelligent strategies.
    
    Features:
    - Multiple CUDA streams for parallel transfers
    - Priority queue for urgent prefetches
    - Adaptive prefetch based on access patterns
    - Bandwidth-aware scheduling
    
    The key insight: GPU compute takes ~10ms per layer,
    CPU->GPU transfer takes ~5ms per layer.
    By prefetching 2 layers ahead, we can hide ALL transfer latency!
    """
    
    def __init__(
        self,
        load_fn: Callable[[int], None],
        is_loaded_fn: Callable[[int], bool],
        num_layers: int,
        num_streams: int = 2,
        queue_size: int = 10,
        strategy: PrefetchStrategy = PrefetchStrategy.SEQUENTIAL,
        device: int = 0,
    ):
        """
        Args:
            load_fn: Function to load a layer to GPU
            is_loaded_fn: Function to check if layer is loaded
            num_layers: Total number of layers
            num_streams: Number of CUDA transfer streams
            queue_size: Size of prefetch queue
            strategy: Prefetching strategy
            device: GPU device
        """
        self.load_fn = load_fn
        self.is_loaded_fn = is_loaded_fn
        self.num_layers = num_layers
        self.strategy = strategy
        self.device = device
        
        # CUDA streams for async transfers
        self.streams: List[torch.cuda.Stream] = []
        if torch.cuda.is_available():
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream(device=device))
        
        # Priority queue for requests
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self._pending: Set[int] = set()
        self._pending_lock = threading.Lock()
        
        # Access pattern tracking (for adaptive strategy)
        self._access_history: deque = deque(maxlen=100)
        self._pattern_model: Optional[Dict[int, List[int]]] = None
        
        # Worker threads
        self._workers: List[threading.Thread] = []
        self._running = False
        
        # Stats
        self.stats = {
            "prefetches_requested": 0,
            "prefetches_completed": 0,
            "prefetches_skipped": 0,
            "avg_prefetch_time": 0.0,
        }
    
    def start(self):
        """Start prefetch workers."""
        if self._running:
            return
        
        self._running = True
        
        # One worker per stream
        for i, stream in enumerate(self.streams):
            t = threading.Thread(
                target=self._worker,
                args=(i, stream),
                daemon=True,
                name=f"prefetcher-{i}",
            )
            t.start()
            self._workers.append(t)
        
        if not self.streams:
            # CPU fallback - single worker
            t = threading.Thread(
                target=self._worker,
                args=(0, None),
                daemon=True,
                name="prefetcher-cpu",
            )
            t.start()
            self._workers.append(t)
    
    def stop(self):
        """Stop prefetch workers."""
        self._running = False
        
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for workers
        for t in self._workers:
            t.join(timeout=1.0)
        self._workers.clear()
    
    def request_prefetch(
        self,
        layer_idx: int,
        priority: int = 0,
        deadline: float = 0.0,
    ):
        """
        Request prefetch of a layer.
        
        Args:
            layer_idx: Layer to prefetch
            priority: Higher = more urgent
            deadline: When this layer will be needed
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        
        # Check if already loaded or pending
        if self.is_loaded_fn(layer_idx):
            return
        
        with self._pending_lock:
            if layer_idx in self._pending:
                return
            self._pending.add(layer_idx)
        
        # Queue request
        request = PrefetchRequest(layer_idx, priority, deadline)
        try:
            self._queue.put_nowait(request)
            self.stats["prefetches_requested"] += 1
        except queue.Full:
            with self._pending_lock:
                self._pending.discard(layer_idx)
    
    def notify_access(self, layer_idx: int):
        """
        Notify that a layer was accessed.
        
        Used for adaptive prefetch strategy.
        """
        self._access_history.append((time.time(), layer_idx))
        
        # Trigger prefetch based on strategy
        if self.strategy == PrefetchStrategy.SEQUENTIAL:
            self._prefetch_sequential(layer_idx)
        elif self.strategy == PrefetchStrategy.ADAPTIVE:
            self._prefetch_adaptive(layer_idx)
    
    def _prefetch_sequential(self, current_idx: int, lookahead: int = 2):
        """Prefetch next N layers sequentially."""
        for offset in range(1, lookahead + 1):
            next_idx = current_idx + offset
            if next_idx < self.num_layers:
                self.request_prefetch(next_idx, priority=lookahead - offset + 1)
    
    def _prefetch_adaptive(self, current_idx: int):
        """Prefetch based on learned patterns."""
        # Analyze recent access patterns
        if len(self._access_history) < 10:
            # Not enough history, fall back to sequential
            self._prefetch_sequential(current_idx)
            return
        
        # Find common next-layer patterns
        # For now, simple sequential detection
        recent = [idx for _, idx in list(self._access_history)[-10:]]
        
        if len(recent) >= 2:
            # Check if sequential
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            avg_diff = sum(diffs) / len(diffs)
            
            if 0.5 < avg_diff < 1.5:
                # Sequential pattern
                for offset in range(1, 3):
                    self.request_prefetch(current_idx + offset)
            elif avg_diff < -0.5:
                # Reverse pattern (happens during some attention patterns)
                for offset in range(1, 3):
                    self.request_prefetch(current_idx - offset)
    
    def _worker(self, worker_id: int, stream: Optional[torch.cuda.Stream]):
        """Background worker that processes prefetch requests."""
        while self._running:
            try:
                request = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            layer_idx = request.layer_idx
            
            # Check if still needed
            if self.is_loaded_fn(layer_idx):
                with self._pending_lock:
                    self._pending.discard(layer_idx)
                self.stats["prefetches_skipped"] += 1
                continue
            
            # Perform prefetch
            start_time = time.perf_counter()
            
            try:
                if stream:
                    with torch.cuda.stream(stream):
                        self.load_fn(layer_idx)
                else:
                    self.load_fn(layer_idx)
                
                elapsed = time.perf_counter() - start_time
                self.stats["prefetches_completed"] += 1
                
                # Update average time
                n = self.stats["prefetches_completed"]
                self.stats["avg_prefetch_time"] = (
                    (self.stats["avg_prefetch_time"] * (n-1) + elapsed) / n
                )
                
            except Exception as e:
                print(f"[Prefetcher] Error prefetching layer {layer_idx}: {e}")
            finally:
                with self._pending_lock:
                    self._pending.discard(layer_idx)
    
    def wait_for_layer(self, layer_idx: int, timeout: float = 5.0) -> bool:
        """
        Wait for a layer to be prefetched.
        
        Returns True if layer is loaded, False if timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.is_loaded_fn(layer_idx):
                return True
            time.sleep(0.001)  # 1ms polling
        return False
    
    def get_queue_size(self) -> int:
        """Get number of pending prefetch requests."""
        return self._queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prefetcher statistics."""
        return {
            **self.stats,
            "pending": len(self._pending),
            "queue_size": self._queue.qsize(),
            "hit_rate": (
                self.stats["prefetches_completed"] / 
                self.stats["prefetches_requested"]
                if self.stats["prefetches_requested"] > 0 else 0
            ),
        }


class BandwidthEstimator:
    """
    Estimates CPU->GPU transfer bandwidth for scheduling.
    
    Uses running average of transfer times to predict
    how long prefetches will take.
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._history: deque = deque(maxlen=window_size)
        
        # Initial estimate: ~10 GB/s for PCIe 4.0 x16
        self._default_bandwidth = 10 * 1024**3  # bytes/sec
    
    def record_transfer(self, size_bytes: int, duration_sec: float):
        """Record a transfer for bandwidth estimation."""
        if duration_sec > 0:
            bandwidth = size_bytes / duration_sec
            self._history.append(bandwidth)
    
    def estimate_bandwidth(self) -> float:
        """Get estimated bandwidth in bytes/sec."""
        if not self._history:
            return self._default_bandwidth
        return sum(self._history) / len(self._history)
    
    def estimate_transfer_time(self, size_bytes: int) -> float:
        """Estimate transfer time for a given size."""
        bandwidth = self.estimate_bandwidth()
        return size_bytes / bandwidth if bandwidth > 0 else float('inf')
