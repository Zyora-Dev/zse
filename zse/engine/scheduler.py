"""
ZSE Continuous Batching Scheduler

Dynamic batch scheduling for high-throughput LLM inference.

Key innovation: Unlike static batching that waits for all sequences to finish,
continuous batching:
- Removes finished sequences immediately
- Adds new requests to fill slots
- Maximizes GPU utilization

This is the core of vLLM, TGI, and other production serving systems.

Author: ZSE Team
"""

import torch
import time
import threading
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, PriorityQueue
import heapq
from collections import deque

from zse.engine.kv_cache import KVCacheManager, KVCacheConfig


class RequestStatus(Enum):
    """Status of an inference request."""
    PENDING = "pending"          # Waiting in queue
    RUNNING = "running"          # Currently being processed
    COMPLETED = "completed"      # Generation finished
    FAILED = "failed"            # Error occurred
    CANCELLED = "cancelled"      # Cancelled by user


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = field(default_factory=list)
    eos_token_id: Optional[int] = None
    

@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: str
    prompt_tokens: List[int]
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    
    # Tracking
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # State
    status: RequestStatus = RequestStatus.PENDING
    generated_tokens: List[int] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # Internal
    seq_id: Optional[int] = None  # KV cache sequence ID
    
    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)
    
    @property
    def output_len(self) -> int:
        return len(self.generated_tokens)
    
    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len
    
    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED)
    
    def __lt__(self, other):
        """For priority queue - shorter prompts first (FCFS by default)."""
        return self.created_at < other.created_at


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""
    max_batch_size: int = 32          # Max sequences in a batch
    max_total_tokens: int = 4096      # Max tokens across all sequences
    max_waiting_requests: int = 1000  # Max pending requests
    
    # Scheduling policy
    policy: str = "fcfs"              # fcfs, shortest_first, or priority
    
    # Preemption
    enable_preemption: bool = True    # Allow preempting long sequences
    preemption_threshold: float = 0.8 # Preempt when memory > 80%
    
    # Chunked prefill
    enable_chunked_prefill: bool = True
    chunk_size: int = 512             # Max tokens per prefill chunk


class RequestQueue:
    """
    Priority queue for pending requests.
    
    Supports different scheduling policies:
    - FCFS (First Come First Serve)
    - Shortest Job First
    - Priority-based
    """
    
    def __init__(self, policy: str = "fcfs", max_size: int = 1000):
        self.policy = policy
        self.max_size = max_size
        self._queue: List[Tuple[float, InferenceRequest]] = []
        self._lock = threading.Lock()
    
    def _get_priority(self, request: InferenceRequest) -> float:
        """Calculate priority (lower = higher priority)."""
        if self.policy == "fcfs":
            return request.created_at
        elif self.policy == "shortest_first":
            return request.prompt_len
        else:
            return request.created_at
    
    def add(self, request: InferenceRequest) -> bool:
        """Add request to queue."""
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False
            priority = self._get_priority(request)
            heapq.heappush(self._queue, (priority, request))
            return True
    
    def pop(self) -> Optional[InferenceRequest]:
        """Get highest priority request."""
        with self._lock:
            if not self._queue:
                return None
            _, request = heapq.heappop(self._queue)
            return request
    
    def peek(self) -> Optional[InferenceRequest]:
        """Look at highest priority request without removing."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0][1]
    
    def size(self) -> int:
        """Number of pending requests."""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0


class SequenceGroup:
    """
    Group of sequences being processed together.
    
    In continuous batching, a batch is a dynamic group of sequences
    at different stages of generation.
    """
    
    def __init__(self, request: InferenceRequest):
        self.request = request
        self.seq_id = request.seq_id
        
        # Current state
        self.is_prefill = True  # True during prompt processing
        self.num_computed_tokens = 0
        
        # For chunked prefill
        self.prefill_remaining = request.prompt_len
    
    @property
    def is_finished(self) -> bool:
        return self.request.is_finished
    
    def get_next_tokens(self, chunk_size: int = 512) -> List[int]:
        """Get tokens for next forward pass."""
        if self.is_prefill:
            # Return chunk of prompt tokens
            start = self.num_computed_tokens
            end = min(start + chunk_size, self.request.prompt_len)
            return self.request.prompt_tokens[start:end]
        else:
            # Return last generated token
            if self.request.generated_tokens:
                return [self.request.generated_tokens[-1]]
            return []


class ContinuousBatchingScheduler:
    """
    Continuous Batching Scheduler for LLM inference.
    
    Key concepts:
    1. Running batch: Sequences currently being processed
    2. Waiting queue: Pending requests
    3. Preemption: Pause long sequences to admit new ones
    
    Each iteration:
    1. Check for finished sequences, remove them
    2. Fill empty slots with waiting requests
    3. Run one forward pass on the batch
    4. Sample next tokens
    5. Update KV caches
    """
    
    def __init__(
        self,
        config: SchedulerConfig,
        kv_cache_manager: KVCacheManager,
    ):
        self.config = config
        self.kv_manager = kv_cache_manager
        
        # Request queues
        self.waiting_queue = RequestQueue(
            policy=config.policy,
            max_size=config.max_waiting_requests,
        )
        
        # Currently running sequences
        self.running: Dict[str, SequenceGroup] = {}
        
        # Completed requests (for retrieval)
        self.completed: Dict[str, InferenceRequest] = {}
        
        # Stats
        self.total_requests = 0
        self.completed_requests = 0
        self.preempted_requests = 0
    
    def add_request(self, request: InferenceRequest) -> bool:
        """
        Add a new inference request.
        
        Returns:
            True if request was added successfully
        """
        if request.status != RequestStatus.PENDING:
            return False
        
        success = self.waiting_queue.add(request)
        if success:
            self.total_requests += 1
        return success
    
    def _can_allocate(self, request: InferenceRequest) -> bool:
        """Check if we can allocate resources for a request."""
        # Check batch size
        if len(self.running) >= self.config.max_batch_size:
            return False
        
        # Check total tokens (rough estimate)
        current_tokens = sum(
            sg.request.total_len for sg in self.running.values()
        )
        if current_tokens + request.prompt_len > self.config.max_total_tokens:
            return False
        
        # Check KV cache memory
        mem_stats = self.kv_manager.memory_stats()
        if "utilization" in mem_stats and mem_stats["utilization"] > 0.95:
            return False
        
        return True
    
    def _allocate_request(self, request: InferenceRequest) -> bool:
        """Allocate KV cache and add to running batch."""
        try:
            seq_id = self.kv_manager.create_cache(
                batch_size=1,
                prompt_len=request.prompt_len,
            )
            request.seq_id = seq_id
            request.status = RequestStatus.RUNNING
            request.started_at = time.time()
            
            seq_group = SequenceGroup(request)
            self.running[request.request_id] = seq_group
            
            return True
        except RuntimeError:
            return False
    
    def _free_request(self, request: InferenceRequest):
        """Free resources for a completed request."""
        if request.seq_id is not None:
            self.kv_manager.free_cache(request.seq_id)
        
        if request.request_id in self.running:
            del self.running[request.request_id]
    
    def schedule(self) -> Tuple[List[SequenceGroup], List[SequenceGroup]]:
        """
        Schedule the next batch of sequences.
        
        Returns:
            (prefill_groups, decode_groups) to process
        """
        prefill_groups = []
        decode_groups = []
        
        # 1. Check finished sequences
        finished_ids = []
        for req_id, seq_group in self.running.items():
            if seq_group.is_finished:
                finished_ids.append(req_id)
        
        # Remove finished
        for req_id in finished_ids:
            seq_group = self.running[req_id]
            request = seq_group.request
            request.completed_at = time.time()
            self._free_request(request)
            self.completed[req_id] = request
            self.completed_requests += 1
        
        # 2. Categorize running sequences
        for seq_group in self.running.values():
            if seq_group.is_prefill:
                prefill_groups.append(seq_group)
            else:
                decode_groups.append(seq_group)
        
        # 3. Try to admit new requests
        while not self.waiting_queue.is_empty():
            request = self.waiting_queue.peek()
            if request is None:
                break
            
            if not self._can_allocate(request):
                break
            
            # Pop and allocate
            request = self.waiting_queue.pop()
            if self._allocate_request(request):
                seq_group = self.running[request.request_id]
                prefill_groups.append(seq_group)
            else:
                # Failed to allocate, re-queue
                request.status = RequestStatus.PENDING
                self.waiting_queue.add(request)
                break
        
        return prefill_groups, decode_groups
    
    def update_after_forward(
        self,
        seq_groups: List[SequenceGroup],
        new_tokens: torch.Tensor,
        eos_token_id: Optional[int] = None,
    ):
        """
        Update sequences after forward pass.
        
        Args:
            seq_groups: Sequence groups that were processed
            new_tokens: Generated tokens [batch_size] or [batch_size, 1]
            eos_token_id: End of sequence token
        """
        if new_tokens.ndim == 2:
            new_tokens = new_tokens.squeeze(-1)
        
        for i, seq_group in enumerate(seq_groups):
            request = seq_group.request
            
            if seq_group.is_prefill:
                # Update prefill progress
                chunk_processed = min(
                    self.config.chunk_size,
                    seq_group.prefill_remaining,
                )
                seq_group.num_computed_tokens += chunk_processed
                seq_group.prefill_remaining -= chunk_processed
                
                if seq_group.prefill_remaining <= 0:
                    seq_group.is_prefill = False
            
            # For decode phase, add generated token
            if not seq_group.is_prefill:
                token = new_tokens[i].item()
                request.generated_tokens.append(token)
                seq_group.num_computed_tokens += 1
                
                # Check stopping conditions
                gen_config = request.generation_config
                
                # Max tokens
                if request.output_len >= gen_config.max_new_tokens:
                    request.status = RequestStatus.COMPLETED
                
                # EOS token
                if eos_token_id is not None and token == eos_token_id:
                    request.status = RequestStatus.COMPLETED
                
                # Custom stop tokens
                if token in gen_config.stop_token_ids:
                    request.status = RequestStatus.COMPLETED
    
    def get_batch_inputs(
        self,
        seq_groups: List[SequenceGroup],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Prepare batched inputs for forward pass.
        
        Returns:
            (input_ids, position_ids, seq_lens) for the batch
        """
        all_tokens = []
        all_positions = []
        seq_lens = []
        
        for seq_group in seq_groups:
            if seq_group.is_prefill:
                # Prefill: process prompt chunk
                tokens = seq_group.get_next_tokens(self.config.chunk_size)
                start_pos = seq_group.num_computed_tokens
                positions = list(range(start_pos, start_pos + len(tokens)))
            else:
                # Decode: single token
                tokens = [seq_group.request.generated_tokens[-1]] if seq_group.request.generated_tokens else seq_group.request.prompt_tokens[-1:]
                positions = [seq_group.request.total_len - 1]
            
            all_tokens.extend(tokens)
            all_positions.extend(positions)
            seq_lens.append(len(tokens))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return (
            torch.tensor(all_tokens, dtype=torch.long, device=device),
            torch.tensor(all_positions, dtype=torch.long, device=device),
            seq_lens,
        )
    
    def num_running(self) -> int:
        """Number of currently running sequences."""
        return len(self.running)
    
    def num_waiting(self) -> int:
        """Number of waiting requests."""
        return self.waiting_queue.size()
    
    def stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "running": len(self.running),
            "waiting": self.waiting_queue.size(),
            "preempted": self.preempted_requests,
            "kv_cache": self.kv_manager.memory_stats(),
        }


# =============================================================================
# INFERENCE ENGINE WITH CONTINUOUS BATCHING
# =============================================================================

class InferenceEngine:
    """
    High-level inference engine with continuous batching.
    
    Usage:
        engine = InferenceEngine(model, tokenizer)
        
        # Add requests
        req_id = engine.add_request("Hello, how are you?")
        
        # Run generation loop
        while engine.has_pending():
            engine.step()
        
        # Get results
        output = engine.get_output(req_id)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        scheduler_config: Optional[SchedulerConfig] = None,
        kv_config: Optional[KVCacheConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        
        # Extract model config
        if hasattr(model, 'config'):
            model_config = model.config
            num_layers = getattr(model_config, 'num_hidden_layers', 32)
            num_heads = getattr(model_config, 'num_attention_heads', 32)
            head_dim = getattr(model_config, 'hidden_size', 4096) // num_heads
        else:
            num_layers = 32
            num_heads = 32
            head_dim = 128
        
        # Initialize KV cache manager
        if kv_config is None:
            kv_config = KVCacheConfig(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kv_manager = KVCacheManager(
            num_layers=kv_config.num_layers,
            num_heads=kv_config.num_heads,
            head_dim=kv_config.head_dim,
            max_seq_len=kv_config.max_seq_len,
            dtype=kv_config.dtype,
            device=device,
        )
        
        # Initialize scheduler
        if scheduler_config is None:
            scheduler_config = SchedulerConfig()
        
        self.scheduler = ContinuousBatchingScheduler(
            config=scheduler_config,
            kv_cache_manager=self.kv_manager,
        )
        
        self._request_counter = 0
    
    def add_request(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Add a new generation request.
        
        Args:
            prompt: Input text
            generation_config: Generation parameters
            
        Returns:
            Request ID for tracking
        """
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = self.tokenizer(prompt)['input_ids']
        
        # Create request
        request_id = f"req_{self._request_counter}"
        self._request_counter += 1
        
        request = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            generation_config=generation_config or GenerationConfig(),
        )
        
        self.scheduler.add_request(request)
        
        return request_id
    
    def step(self) -> int:
        """
        Run one step of generation.
        
        Returns:
            Number of tokens generated
        """
        # Schedule batch
        prefill_groups, decode_groups = self.scheduler.schedule()
        
        if not prefill_groups and not decode_groups:
            return 0
        
        # Combine for batched execution
        all_groups = prefill_groups + decode_groups
        
        # Get batch inputs
        input_ids, position_ids, seq_lens = self.scheduler.get_batch_inputs(all_groups)
        
        if len(input_ids) == 0:
            return 0
        
        # Run model forward
        with torch.no_grad():
            # Simple forward - in production, use attention with KV cache
            logits = self.model(input_ids.unsqueeze(0))
            
            # Get next tokens (greedy for simplicity)
            # In production: apply temperature, top-k, top-p
            next_tokens = logits[:, -1, :].argmax(dim=-1)
        
        # Update scheduler
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        self.scheduler.update_after_forward(all_groups, next_tokens, eos_token_id)
        
        return len(all_groups)
    
    def has_pending(self) -> bool:
        """Check if there are pending requests."""
        return self.scheduler.num_running() > 0 or self.scheduler.num_waiting() > 0
    
    def get_output(self, request_id: str) -> Optional[str]:
        """Get generated output for a request."""
        if request_id in self.scheduler.completed:
            request = self.scheduler.completed[request_id]
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode(request.generated_tokens)
            return request.generated_tokens
        return None
    
    def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get status of a request."""
        if request_id in self.scheduler.running:
            return self.scheduler.running[request_id].request.status
        if request_id in self.scheduler.completed:
            return self.scheduler.completed[request_id].status
        return None
    
    def stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.scheduler.stats()
