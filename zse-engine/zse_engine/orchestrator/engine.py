"""ZSE Engine — The main inference engine API.

Ties everything together:
    ZSELoader (model) + GPUMemory + VRAMAllocator + WeightLoader +
    KVCacheManager + InferenceKernels + ModelRunner + Sampler → generate()

Usage:
    engine = ZSEEngine("model.zse")
    text = engine.generate("Once upon a time", max_tokens=100)
    print(text)

    # Streaming
    for token in engine.stream_generate("Hello world"):
        print(token, end="", flush=True)

    engine.destroy()
"""

import time
from typing import List, Optional, Iterator, Set
from dataclasses import dataclass

from zse_compiler.runtime.device import detect_backend, get_devices, DeviceInfo
from zse_compiler.runtime.memory import GPUMemory

from zse_engine.format.loader import ZSELoader
from zse_engine.format.config import ModelConfig
from zse_engine.cache.cache_manager import KVCacheManager

from zse_engine.orchestrator.vram_allocator import VRAMAllocator, VRAMPlan, ScratchBuffers
from zse_engine.orchestrator.weight_loader import WeightLoader, WeightStore
from zse_engine.orchestrator.kernels import InferenceKernels
from zse_engine.orchestrator.model_runner import ModelRunner
from zse_engine.orchestrator.sampler import Sampler


@dataclass
class GenerateConfig:
    """Generation configuration."""
    max_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_tokens: Optional[List[int]] = None
    seed: Optional[int] = None


@dataclass
class EngineStats:
    """Engine performance statistics."""
    model_path: str
    model_arch: str
    device_name: str
    vram_total_gb: float
    weight_load_time_s: float
    kernel_compile_time_s: float
    total_init_time_s: float
    # Generation
    total_tokens_generated: int
    total_prefill_tokens: int
    total_generate_time_s: float
    avg_tokens_per_sec: float
    # Memory
    weight_bytes: int
    kv_cache_bytes: int
    scratch_bytes: int


class ZSEEngine:
    """ZSE Inference Engine — zero-dependency LLM serving.

    Handles the full pipeline: load model → allocate GPU → compile kernels
    → run inference → produce tokens.

    Args:
        model_path: Path to .zse model file
        device_index: GPU device index (default: 0)
        max_seq_len: Max scratch buffer size (default: from model config)
        quiet: Suppress progress output
    """

    def __init__(
        self,
        model_path: str,
        device_index: int = 0,
        max_seq_len: int = 0,
        quiet: bool = False,
    ):
        self._model_path = model_path
        self._quiet = quiet
        self._seq_counter = 0
        self._total_tokens = 0
        self._total_prefill = 0
        self._total_gen_time = 0.0

        init_start = time.monotonic()

        # Step 1: Detect GPU
        if not quiet:
            print("[ZSE] Detecting GPU...")
        backend = detect_backend()
        devices = get_devices(backend)

        if not devices:
            raise RuntimeError(
                "No GPU detected. ZSE requires a CUDA or ROCm GPU."
            )

        self._device = devices[min(device_index, len(devices) - 1)]
        self._backend = backend
        if not quiet:
            print(f"[ZSE] GPU: {self._device.name} "
                  f"({self._device.vram_total_gb:.1f}GB VRAM)")

        # Step 2: Open model file
        if not quiet:
            print(f"[ZSE] Loading model: {model_path}")
        self._loader = ZSELoader(model_path)
        self._config = self._loader.config
        # Defer tokenizer — expensive deserialization not needed for GPU init

        if max_seq_len <= 0:
            max_seq_len = min(self._config.max_seq_len, 2048)  # Cap for scratch
        self._max_seq_len = max_seq_len

        # Step 3: Initialize GPU memory
        self._gpu_mem = GPUMemory(backend=backend)

        # Step 4: Plan VRAM allocation
        self._allocator = VRAMAllocator(self._gpu_mem, self._device)
        model_size = self._config.estimate_model_size_bytes()
        self._vram_plan = self._allocator.plan_allocation(model_size, self._config)
        if not quiet:
            print(self._vram_plan.summary())

        # Step 5+8: Pre-fault mmap pages AND compile kernels in parallel,
        # then upload weights sequentially (pages already in OS cache).
        import threading

        if not quiet:
            print("[ZSE] Uploading weights + compiling kernels (parallel)...")

        parallel_start = time.monotonic()

        wl = WeightLoader(self._loader, self._gpu_mem)

        # Pre-fault mmap pages in background using os.read for fast sequential NFS I/O
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

        # Kernel compilation on main thread (hipModuleLoadData needs GPU context)
        kernel_start = time.monotonic()
        self._kernels = InferenceKernels(backend=backend)
        quant_type = "int4" if self._config.quant.method == 1 else ("int8" if self._config.quant.method == 2 else "fp16")
        self._kernels.compile_all(quant_type=quant_type)
        self._kernel_compile_time = time.monotonic() - kernel_start

        # Upload weights — prefault thread continues pulling NFS pages ahead of HtoD
        weight_start = time.monotonic()
        weight_error = [None]
        try:
            def _progress(name, idx, total):
                if not quiet and idx % 50 == 0:
                    print(f"  [{idx}/{total}] {name}")
            self._weights = wl.load_all(progress_fn=_progress)
        except Exception as e:
            weight_error[0] = e
        self._weight_load_time = time.monotonic() - weight_start

        prefault_thread.join(timeout=1.0)

        if weight_error[0]:
            raise RuntimeError(f"Weight loading failed: {weight_error[0]}")

        parallel_time = time.monotonic() - parallel_start

        # Deserialize tokenizer now (deferred from load phase)
        self._tokenizer = self._loader.tokenizer

        if not quiet:
            print(f"[ZSE] Weights loaded: {self._weights.total_bytes / 1024**2:.1f}MB "
                  f"in {self._weight_load_time:.2f}s")
            print(f"[ZSE] {self._kernels.num_compiled} kernels compiled "
                  f"in {self._kernel_compile_time:.2f}s")
            print(f"[ZSE] Parallel init total: {parallel_time:.2f}s "
                  f"(saved {self._weight_load_time + self._kernel_compile_time - parallel_time:.1f}s)")

        # Step 6: Allocate scratch buffers
        self._scratch = self._allocator.allocate_scratch(
            self._config, max_seq_len=max_seq_len,
        )

        # Step 7: Create KV cache with remaining VRAM
        kv_budget = self._vram_plan.kv_cache_bytes
        self._kv_cache = KVCacheManager(
            config=self._config,
            gpu_mem=self._gpu_mem,
            budget_bytes=kv_budget,
        )
        if not quiet:
            print(f"[ZSE] KV Cache: {kv_budget / 1024**2:.1f}MB, "
                  f"max {self._vram_plan.max_batch_tokens:,} tokens")

        # Step 9: Create model runner
        self._runner = ModelRunner(
            config=self._config,
            weights=self._weights,
            kv_cache=self._kv_cache,
            scratch=self._scratch,
            gpu_mem=self._gpu_mem,
            kernels=self._kernels,
        )

        # Step 10: Create sampler
        self._sampler = Sampler()

        # Step 11: Initialize GPU graph for decode acceleration
        try:
            self._runner.init_graph(max_seq_len=max_seq_len)
            if not quiet:
                print("[ZSE] GPU Graph initialized for decode acceleration")
        except Exception as e:
            if not quiet:
                print(f"[ZSE] GPU Graph init skipped: {e}")

        self._total_init_time = time.monotonic() - init_start
        if not quiet:
            print(f"[ZSE] Engine ready in {self._total_init_time:.2f}s")
            print(f"  Weight load: {self._weight_load_time:.2f}s")
            print(f"  Kernel compile: {self._kernel_compile_time:.2f}s")
            print()

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def device(self) -> DeviceInfo:
        return self._device

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k filtering (0 = disabled)
            repetition_penalty: Penalize repeated tokens
            stop_tokens: Token IDs that stop generation
            seed: Random seed for reproducibility

        Returns:
            Generated text (prompt + completion)
        """
        tokens = list(self.stream_generate_tokens(
            prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens, seed=seed,
        ))

        if self._tokenizer:
            return self._tokenizer.decode(tokens)
        return f"[{len(tokens)} tokens generated]"

    def generate_tokens(
        self,
        token_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> List[int]:
        """Generate tokens from token IDs.

        Returns:
            List of all tokens (prompt + generated)
        """
        return list(self._generate_impl(
            token_ids, max_tokens, temperature, top_p, top_k,
            repetition_penalty, stop_tokens, seed,
        ))

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        """Stream generated text token by token.

        Yields decoded text for each new token.
        """
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded — cannot stream text")

        for token_id in self.stream_generate_tokens(
            prompt, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens, seed=seed,
        ):
            yield self._tokenizer.decode([token_id])

    def stream_generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_tokens: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> Iterator[int]:
        """Stream generated token IDs.

        Yields token IDs one at a time.
        """
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded — cannot encode prompt")

        token_ids = self._tokenizer.encode(prompt)
        yield from self._generate_impl(
            token_ids, max_tokens, temperature, top_p, top_k,
            repetition_penalty, stop_tokens, seed,
        )

    def _generate_impl(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        stop_tokens: Optional[List[int]],
        seed: Optional[int],
    ) -> Iterator[int]:
        """Core generation loop."""
        gen_start = time.monotonic()

        seq_id = self._seq_counter
        self._seq_counter += 1

        if seed is not None:
            self._sampler = Sampler(seed=seed)

        # Default stop tokens: EOS
        if stop_tokens is None:
            stop_tokens = []
            if self._tokenizer and hasattr(self._tokenizer, 'special_tokens'):
                eos = self._tokenizer.special_tokens.eos_id
                if eos is not None:
                    stop_tokens.append(eos)

        past_tokens: Set[int] = set(prompt_tokens)

        try:
            # Prefill
            logits = self._runner.prefill(prompt_tokens, seq_id)
            self._total_prefill += len(prompt_tokens)

            # Use GPU argmax for greedy decoding (avoids 300KB DtoH + Python argmax)
            use_gpu_argmax = (temperature <= 0 or temperature < 1e-6) and repetition_penalty == 1.0

            # Sample first token
            if use_gpu_argmax:
                token = self._runner.gpu_argmax(0)
            else:
                token = self._sampler.sample(
                    logits, self._config.vocab_size,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    past_tokens=past_tokens,
                )
            yield token
            past_tokens.add(token)

            if token in stop_tokens:
                return

            # Decode loop
            # For greedy decoding, use GPU graph after warmup (step 0 = warmup, step 1+ = graph)
            use_graph = use_gpu_argmax and self._runner._graph_runner is not None
            graph_ready = False

            for step in range(1, max_tokens):
                position = len(prompt_tokens) + step - 1

                if use_graph and step >= 2:
                    # Graph path: capture on step 2, replay on step 3+
                    token = self._runner.decode_step_graph(token, seq_id, position)
                else:
                    # Normal path (warmup for graph on step 1, or non-greedy)
                    logits = self._runner.decode_step(
                        token, seq_id, position,
                        skip_logits_download=use_gpu_argmax,
                    )

                    if use_gpu_argmax:
                        token = self._runner.gpu_argmax(0)
                    else:
                        token = self._sampler.sample(
                            logits, self._config.vocab_size,
                            temperature=temperature, top_p=top_p, top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            past_tokens=past_tokens,
                    )
                yield token
                past_tokens.add(token)
                self._total_tokens += 1

                if token in stop_tokens:
                    return

        finally:
            # Always cleanup the sequence
            self._kv_cache.mark_idle(seq_id)
            self._kv_cache.free_sequence(seq_id)
            gen_time = time.monotonic() - gen_start
            self._total_gen_time += gen_time

    def stats(self) -> EngineStats:
        """Get engine performance statistics."""
        tps = (self._total_tokens / self._total_gen_time
               if self._total_gen_time > 0 else 0)

        return EngineStats(
            model_path=self._model_path,
            model_arch=self._config.arch,
            device_name=self._device.name,
            vram_total_gb=self._device.vram_total_gb,
            weight_load_time_s=self._weight_load_time,
            kernel_compile_time_s=self._kernel_compile_time,
            total_init_time_s=self._total_init_time,
            total_tokens_generated=self._total_tokens,
            total_prefill_tokens=self._total_prefill,
            total_generate_time_s=self._total_gen_time,
            avg_tokens_per_sec=tps,
            weight_bytes=self._weights.total_bytes,
            kv_cache_bytes=self._vram_plan.kv_cache_bytes,
            scratch_bytes=self._scratch.total_bytes,
        )

    def summary(self) -> str:
        """Human-readable engine summary."""
        s = self.stats()
        return (
            f"ZSE Engine Summary:\n"
            f"  Model: {s.model_arch} ({s.model_path})\n"
            f"  Device: {s.device_name} ({s.vram_total_gb:.1f}GB)\n"
            f"  Init: {s.total_init_time_s:.2f}s "
            f"(weights: {s.weight_load_time_s:.2f}s, "
            f"kernels: {s.kernel_compile_time_s:.2f}s)\n"
            f"  Memory: weights={s.weight_bytes / 1024**2:.1f}MB, "
            f"kv={s.kv_cache_bytes / 1024**2:.1f}MB, "
            f"scratch={s.scratch_bytes / 1024**2:.1f}MB\n"
            f"  Generated: {s.total_tokens_generated} tokens "
            f"({s.avg_tokens_per_sec:.1f} tok/s)\n"
            f"  Prefill: {s.total_prefill_tokens} tokens"
        )

    def destroy(self):
        """Release all GPU resources."""
        if hasattr(self, '_runner'):
            self._runner.destroy_graph()
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
