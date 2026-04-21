"""ZStreamer Engine — Main continuous batching loop.

The public API for multi-request serving. Replaces ZSEEngine's single-request
_generate_impl with a continuous batching loop that handles concurrent requests.

Usage:
    # Initialize with a loaded model
    streamer = ZStreamerEngine(model_path="model.zse")

    # Submit requests (non-blocking)
    req_id = streamer.add_request("Tell me a story", max_tokens=100)

    # Or with streaming callback
    req_id = streamer.add_request("Hello", on_token=lambda t: print(t))

    # Run the loop (blocking — call from a dedicated thread)
    streamer.run()

    # Or step manually
    while True:
        result = streamer.step()
        for output in result.finished:
            print(output)
"""

import time
import ctypes
import threading
from collections import deque
from typing import Optional, List, Callable, Dict
from dataclasses import dataclass

from zse_compiler.runtime.device import detect_backend, get_devices
from zse_compiler.runtime.memory import GPUMemory

from zse_engine.format.loader import ZSELoader
from zse_engine.format.config import ModelConfig
from zse_engine.cache.cache_manager import KVCacheManager

from zse_engine.orchestrator.vram_allocator import VRAMAllocator
from zse_engine.orchestrator.weight_loader import WeightLoader
from zse_engine.orchestrator.kernels import InferenceKernels
from zse_engine.orchestrator.model_runner import ModelRunner
from zse_engine.orchestrator.sampler import Sampler

# Optional LoRA support
try:
    from zse_engine.orchestrator.lora_manager import LoRAManager
except ImportError:
    LoRAManager = None

from zse_engine.zstreamer.request import (
    InferenceRequest, GenerationParams, RequestOutput, FinishReason,
)
from zse_engine.zstreamer.scheduler import Scheduler, SchedulerConfig
from zse_engine.zstreamer.memory_budget import MemoryBudget
from zse_engine.zstreamer.batch_runner import BatchRunner, StepResult


@dataclass
class StreamerStats:
    """ZStreamer runtime statistics."""
    # Throughput
    total_requests: int
    total_tokens_generated: int
    total_steps: int
    uptime_s: float
    requests_per_sec: float
    tokens_per_sec: float
    avg_batch_size: float

    # Latency
    avg_ttft_ms: float
    avg_tpot_ms: float
    p99_ttft_ms: float

    # Queue
    queue_depth: int
    active_requests: int

    # Memory
    memory_utilization: float


class ZStreamerEngine:
    """Continuous batching inference engine.

    Handles concurrent requests with SLO-aware scheduling, predictive
    memory management, and anti-burst controls.

    Args:
        model_path: Path to .zse model file
        scheduler_config: Scheduler tuning parameters
        device_index: GPU device index
        max_seq_len: Max sequence length
        headroom_ratio: Memory headroom for decode (0.0-1.0)
        quiet: Suppress progress output
    """

    def __init__(
        self,
        model_path: str,
        scheduler_config: Optional[SchedulerConfig] = None,
        device_index: int = 0,
        max_seq_len: int = 0,
        headroom_ratio: float = 0.5,
        quiet: bool = False,
    ):
        self._quiet = quiet
        self._running = False
        self._start_time = None
        self._request_counter = 0
        self._lock = threading.Lock()

        # Stats accumulators (bounded ring buffer for latency samples)
        self._total_requests = 0
        self._total_tokens = 0
        self._total_steps = 0
        self._ttft_samples: deque = deque(maxlen=10000)
        self._tpot_samples: deque = deque(maxlen=10000)

        # Completed request outputs (for polling)
        self._completed: Dict[str, RequestOutput] = {}
        self._completed_maxlen = 10000  # Prevent unbounded growth

        if scheduler_config is None:
            scheduler_config = SchedulerConfig()

        init_start = time.monotonic()

        # --- GPU Init (same as ZSEEngine) ---
        if not quiet:
            print("[ZStreamer] Detecting GPU...")
        backend = detect_backend()
        devices = get_devices(backend)
        if not devices:
            raise RuntimeError("No GPU detected")

        self._device = devices[min(device_index, len(devices) - 1)]
        self._backend = backend
        if not quiet:
            print(f"[ZStreamer] GPU: {self._device.name} "
                  f"({self._device.vram_total_gb:.1f}GB)")

        t0 = time.monotonic()

        # --- Start kernel PTX compilation early (CPU-only, no GPU context needed) ---
        # This overlaps with file loading below
        if not quiet:
            print(f"[ZStreamer] Loading model: {model_path}")
        self._kernels = InferenceKernels(backend=backend)
        # Start PTX compilation in background — it's pure CPU work
        import concurrent.futures
        ptx_future = None
        if backend == "cuda":
            ptx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            ptx_future = ptx_executor.submit(
                self._kernels._compile_ptx_only, "int4"  # assume INT4, most common
            )

        # --- Load model (mmap + parse config/weight_index) ---
        t_load0 = time.monotonic()
        self._loader = ZSELoader(model_path)
        self._config = self._loader.config
        # Defer tokenizer deserialization — it's expensive and not needed until after init
        t_load = time.monotonic() - t_load0

        # Start pre-faulting weight data from NFS into page cache ASAP
        # This runs during VRAM planning + kernel compilation (pure CPU/NFS I/O)
        def _prefault_mmap():
            try:
                import os as _os
                wd_offset = self._loader._weight_data_offset
                file_size = _os.path.getsize(self._loader._path)
                if wd_offset > 0:
                    fd = _os.open(self._loader._path, _os.O_RDONLY)
                    try:
                        _os.lseek(fd, wd_offset, _os.SEEK_SET)
                        remaining = file_size - wd_offset
                        CHUNK = 8 * 1024 * 1024
                        while remaining > 0:
                            n = _os.read(fd, min(CHUNK, remaining))
                            if not n:
                                break
                            remaining -= len(n)
                    finally:
                        _os.close(fd)
            except Exception:
                pass

        prefault_thread = threading.Thread(target=_prefault_mmap, daemon=True)
        prefault_thread.start()

        if max_seq_len <= 0:
            max_seq_len = min(self._config.max_seq_len, 2048)
        self._max_seq_len = max_seq_len

        # --- GPU Memory ---
        self._gpu_mem = GPUMemory(backend=backend)

        # --- VRAM Allocation ---
        self._allocator = VRAMAllocator(self._gpu_mem, self._device)
        model_size = self._config.estimate_model_size_bytes()
        self._vram_plan = self._allocator.plan_allocation(model_size, self._config, max_seq_len)
        if not quiet:
            print(self._vram_plan.summary())

        t_plan = time.monotonic() - t0

        # --- Compile kernels + upload weights (prefault already running) ---
        if not quiet:
            print("[ZStreamer] Uploading weights + compiling kernels (parallel)...")
        t0 = time.monotonic()

        wl = WeightLoader(self._loader, self._gpu_mem)

        # Finish kernel compilation on main thread (while mmap pre-faults in parallel)
        t_comp0 = time.monotonic()
        if ptx_future is not None:
            # Wait for PTX compilation, then load modules on main thread
            ptx_results = ptx_future.result()
            ptx_executor.shutdown(wait=False)
            self._kernels._load_ptx_modules(ptx_results)
        else:
            # Non-CUDA or fallback: compile everything now
            quant_type = "int4" if self._config.quant.method == 1 else ("int8" if self._config.quant.method == 2 else "fp16")
            self._kernels.compile_all(quant_type=quant_type)
        t_compile = time.monotonic() - t_comp0

        # Upload weights — prefault thread continues in background pulling pages
        # from NFS ahead of the HtoD copies. Don't wait for it.
        weight_error = [None]
        try:
            self._weights = wl.load_all()
        except Exception as e:
            weight_error[0] = e

        # Clean up prefault thread (should be done by now)
        prefault_thread.join(timeout=1.0)

        if weight_error[0]:
            raise RuntimeError(f"Weight loading failed: {weight_error[0]}")

        t_parallel = time.monotonic() - t0
        t_weights = t_parallel  # approximate (overlapped)

        # Deserialize tokenizer now (deferred from load phase)
        self._tokenizer = self._loader.tokenizer

        # --- Scratch Buffers ---
        self._scratch = self._allocator.allocate_scratch(
            self._config, max_seq_len=max_seq_len,
        )

        # --- KV Cache ---
        kv_budget = self._vram_plan.kv_cache_bytes
        self._kv_cache = KVCacheManager(
            config=self._config,
            gpu_mem=self._gpu_mem,
            budget_bytes=kv_budget,
        )

        # --- Capture CUDA context for cross-thread use ---
        # CUDA contexts are thread-local. We capture the context from init thread
        # so run() on a background thread can push it before GPU operations.
        self._cuda_ctx = None
        if backend == "cuda":
            self._cuda_ctx = ctypes.c_void_p()
            driver = self._gpu_mem._driver
            driver.cuCtxGetCurrent(ctypes.byref(self._cuda_ctx))

        # --- Model Runner ---
        self._lora_manager = None
        if LoRAManager is not None:
            self._lora_manager = LoRAManager(self._gpu_mem, self._kernels)

        self._model_runner = ModelRunner(
            config=self._config,
            weights=self._weights,
            kv_cache=self._kv_cache,
            scratch=self._scratch,
            gpu_mem=self._gpu_mem,
            kernels=self._kernels,
            lora_manager=self._lora_manager,
        )

        # --- Sampler ---
        self._sampler = Sampler()

        # --- Memory Budget ---
        cache_stats = self._kv_cache.stats()
        total_kv_blocks = cache_stats.total_blocks

        self._memory_budget = MemoryBudget(
            total_blocks=total_kv_blocks,
            block_size_tokens=self._kv_cache.block_size,
            headroom_ratio=headroom_ratio,
        )

        # --- Scheduler ---
        self._scheduler = Scheduler(
            config=scheduler_config,
            memory_budget=self._memory_budget,
        )

        # --- Batch Runner ---
        self._batch_runner = BatchRunner(
            model_runner=self._model_runner,
            sampler=self._sampler,
            kv_cache=self._kv_cache,
            scheduler=self._scheduler,
            vocab_size=self._config.vocab_size,
            lora_manager=self._lora_manager,
        )

        init_time = time.monotonic() - init_start
        if not quiet:
            print(f"[ZStreamer] Ready in {init_time:.2f}s")
            print(f"  Init breakdown: load={t_load:.1f}s, plan={t_plan:.1f}s, "
                  f"weights={t_weights:.1f}s, compile={t_compile:.1f}s")
            print(f"  KV blocks: {total_kv_blocks} "
                  f"(block_size={self._kv_cache.block_size})")
            print(f"  Max batch tokens: {scheduler_config.max_batch_tokens}")
            print(f"  Max batch seqs: {scheduler_config.max_batch_seqs}")
            print()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        priority: int = 0,
        deadline_ms: Optional[float] = None,
        on_token: Optional[Callable[[int], None]] = None,
        on_finish: Optional[Callable[[RequestOutput], None]] = None,
        seed: Optional[int] = None,
        lora_id: Optional[str] = None,
    ) -> Optional[str]:
        """Submit an inference request.

        Non-blocking. Returns request_id, or None if queue is full.

        Args:
            prompt: Input text
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling
            top_k: Top-k filtering
            repetition_penalty: Penalize repeated tokens
            stop_tokens: Stop token IDs
            priority: Higher = more important
            deadline_ms: TTFT SLO in milliseconds
            on_token: Streaming callback (called with each token)
            on_finish: Completion callback
            seed: Random seed

        Returns:
            request_id string, or None if rejected (queue full / backpressure)
        """
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded")

        prompt_tokens = self._tokenizer.encode(prompt)

        # Validate prompt length
        if len(prompt_tokens) > self._max_seq_len:
            raise ValueError(
                f"Prompt length {len(prompt_tokens)} exceeds max_seq_len {self._max_seq_len}"
            )
        if len(prompt_tokens) == 0:
            raise ValueError("Empty prompt")

        # Default stop tokens
        if stop_tokens is None:
            stop_tokens = []
            eos_id = getattr(self._tokenizer, 'eos_id', None)
            if eos_id is None and hasattr(self._tokenizer, 'special_tokens'):
                eos_id = self._tokenizer.special_tokens.eos_id
            if eos_id is not None:
                # Sanity check: for large-vocab models (e.g. Qwen2 vocab=151936),
                # eos_id should be a high token ID. A suspiciously low eos_id
                # (like default 2) means the tokenizer didn't parse EOS correctly.
                vocab_size = self._config.vocab_size
                if vocab_size > 50000 and eos_id < 100:
                    pass  # Skip — likely wrong eos_id from bad tokenizer parse
                else:
                    stop_tokens.append(eos_id)

            # Add chat turn markers as stop tokens (e.g. <|im_end|> for Qwen2/ChatML)
            for marker in ["<|im_end|>", "<|im_start|>", "<|eot_id|>"]:
                try:
                    marker_tokens = self._tokenizer.encode(marker, add_bos=False)
                    if len(marker_tokens) == 1 and marker_tokens[0] not in stop_tokens:
                        stop_tokens.append(marker_tokens[0])
                except Exception:
                    pass

        with self._lock:
            self._request_counter += 1
            request_id = f"req-{self._request_counter}"

        request = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            params=GenerationParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stop_tokens=stop_tokens,
                seed=seed,
                lora_id=lora_id,
            ),
            priority=priority,
            deadline_ms=deadline_ms,
            on_token=on_token,
            on_finish=on_finish,
            past_tokens={t: 1 for t in prompt_tokens},
        )

        success = self._scheduler.add_request(request)
        if not success:
            return None

        with self._lock:
            self._total_requests += 1

        return request_id

    def add_request_tokens(
        self,
        token_ids: List[int],
        request_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """Submit a request with pre-tokenized input.

        Same as add_request but skips tokenization.
        """
        with self._lock:
            if request_id is None:
                self._request_counter += 1
                request_id = f"req-{self._request_counter}"

        stop_tokens = kwargs.get("stop_tokens", [])
        request = InferenceRequest(
            request_id=request_id,
            prompt_tokens=token_ids,
            params=GenerationParams(
                max_tokens=kwargs.get("max_tokens", 128),
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                stop_tokens=stop_tokens,
                seed=kwargs.get("seed"),
            ),
            priority=kwargs.get("priority", 0),
            deadline_ms=kwargs.get("deadline_ms"),
            on_token=kwargs.get("on_token"),
            on_finish=kwargs.get("on_finish"),
            past_tokens=set(token_ids),
        )

        success = self._scheduler.add_request(request)
        if not success:
            return None

        with self._lock:
            self._total_requests += 1

        return request_id

    def cancel_request(self, request_id: str):
        """Cancel an in-flight request."""
        self._scheduler.cancel_request(request_id)

    @property
    def lora_manager(self):
        """Access the LoRA manager for loading/unloading adapters."""
        return self._lora_manager

    def step(self) -> StepResult:
        """Execute one scheduling iteration.

        Call this in a loop for manual control, or use run() for automatic.

        Returns:
            StepResult with new tokens and finished requests.
        """
        if self._start_time is None:
            self._start_time = time.monotonic()

        output = self._scheduler.schedule_step()

        if output.is_idle:
            return StepResult()

        result = self._batch_runner.execute(output)

        # Track stats (lightweight)
        self._total_steps += 1
        self._total_tokens += result.num_tokens

        if result.finished:
            with self._lock:
                for finished in result.finished:
                    self._completed[finished.request_id] = finished
                    # Evict oldest if too many unpolled results
                    if len(self._completed) > self._completed_maxlen:
                        oldest = next(iter(self._completed))
                        del self._completed[oldest]
                    self._ttft_samples.append(finished.ttft_ms)
                    self._tpot_samples.append(finished.tpot_ms)

        return result

    def run(self, idle_sleep_ms: float = 1.0):
        """Run the continuous batching loop (blocking).

        Call from a dedicated thread. Loops until stop() is called.
        Thread-safe: pushes the CUDA context captured during __init__
        so GPU operations work from any thread.

        Args:
            idle_sleep_ms: Sleep time when no work available (prevents busy-wait)
        """
        # Push CUDA context onto this thread (CUDA contexts are thread-local)
        if self._cuda_ctx is not None and self._backend == "cuda":
            driver = self._gpu_mem._driver
            driver.cuCtxSetCurrent(self._cuda_ctx)

        self._running = True
        self._run_error = None
        if self._start_time is None:
            self._start_time = time.monotonic()

        try:
            while self._running:
                result = self.step()
                if result.num_tokens == 0 and not result.finished:
                    time.sleep(idle_sleep_ms / 1000)
        except Exception as e:
            self._run_error = e
            self._running = False
            if not self._quiet:
                import traceback
                print(f"[ZStreamer] Engine loop error: {e}")
                traceback.print_exc()

    def stop(self):
        """Stop the continuous batching loop."""
        self._running = False

    def get_result(self, request_id: str) -> Optional[RequestOutput]:
        """Poll for a completed request result."""
        with self._lock:
            return self._completed.pop(request_id, None)

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> StreamerStats:
        with self._lock:
            uptime = (time.monotonic() - self._start_time
                      if self._start_time else 0)
            rps = self._total_requests / uptime if uptime > 0 else 0
            tps = self._total_tokens / uptime if uptime > 0 else 0
            avg_batch = (self._total_tokens / self._total_steps
                         if self._total_steps > 0 else 0)

            avg_ttft = (sum(self._ttft_samples) / len(self._ttft_samples)
                        if self._ttft_samples else 0)
            avg_tpot = (sum(self._tpot_samples) / len(self._tpot_samples)
                        if self._tpot_samples else 0)
            # p99 via heapq.nlargest — O(n) vs O(n log n) for sorted()
            import heapq
            n_samples = len(self._ttft_samples)
            if n_samples > 0:
                top_count = max(1, n_samples - int(n_samples * 0.99))
                p99_ttft = min(heapq.nlargest(top_count, self._ttft_samples))
            else:
                p99_ttft = 0

            sched_stats = self._scheduler.stats()

            return StreamerStats(
                total_requests=self._total_requests,
                total_tokens_generated=self._total_tokens,
                total_steps=self._total_steps,
                uptime_s=uptime,
                requests_per_sec=rps,
                tokens_per_sec=tps,
                avg_batch_size=avg_batch,
                avg_ttft_ms=avg_ttft,
                avg_tpot_ms=avg_tpot,
                p99_ttft_ms=p99_ttft,
                queue_depth=sched_stats["num_waiting"],
                active_requests=sched_stats["num_active"],
                memory_utilization=sched_stats["memory"]["utilization"],
            )

    def summary(self) -> str:
        s = self.stats()
        return (
            f"ZStreamer Engine:\n"
            f"  Uptime: {s.uptime_s:.1f}s\n"
            f"  Requests: {s.total_requests} ({s.requests_per_sec:.1f} req/s)\n"
            f"  Tokens: {s.total_tokens_generated} ({s.tokens_per_sec:.1f} tok/s)\n"
            f"  Batch size (avg): {s.avg_batch_size:.1f}\n"
            f"  TTFT: {s.avg_ttft_ms:.1f}ms avg, {s.p99_ttft_ms:.1f}ms p99\n"
            f"  TPOT: {s.avg_tpot_ms:.1f}ms avg\n"
            f"  Queue: {s.queue_depth} waiting, {s.active_requests} active\n"
            f"  Memory: {s.memory_utilization:.1%}"
        )

    def scheduler_summary(self) -> str:
        return self._scheduler.summary()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self):
        """Release all GPU resources."""
        self.stop()
        if hasattr(self, '_lora_manager') and self._lora_manager is not None:
            self._lora_manager.destroy()
        if hasattr(self, '_kv_cache'):
            self._kv_cache.destroy()
        if hasattr(self, '_scratch'):
            self._scratch.destroy(self._gpu_mem)
        if hasattr(self, '_weights'):
            self._weights.destroy(self._gpu_mem)
        if hasattr(self, '_loader'):
            self._loader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.destroy()

    def __del__(self):
        self.destroy()
