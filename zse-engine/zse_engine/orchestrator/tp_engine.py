"""ZSE Tensor Parallel Engine — Multi-GPU inference orchestrator.

Spawns tp_size worker processes, each managing one GPU. All processes
run the forward pass in lockstep, synchronized by NCCL all-reduce.

Architecture:
    Main process (rank 0):
        - Owns tokenizer, sampler, API
        - Broadcasts token IDs to all ranks
        - Gathers logits and samples next token
    All ranks:
        - Own their GPU context, sharded weights, local KV cache
        - Run forward pass with local weight shards
        - NCCL all-reduce at O proj and Down proj boundaries

Usage:
    engine = TPEngine("model.zse", tp_size=2)
    text = engine.generate("Hello", max_tokens=100)
    engine.destroy()
"""

import os
import time
import struct
import multiprocessing
from contextlib import contextmanager
from multiprocessing import Process, Queue, Value, Array
from typing import Optional, List, Iterator
from dataclasses import dataclass

from zse_compiler.runtime.device import detect_backend, get_devices
from zse_compiler.runtime.memory import GPUMemory
from zse_compiler.runtime.nccl import get_unique_id, is_nccl_available, NCCL_UNIQUE_ID_BYTES

from zse_engine.format.loader import ZSELoader
from zse_engine.format.config import ModelConfig
from zse_engine.cache.cache_manager import KVCacheManager

from zse_engine.orchestrator.vram_allocator import VRAMAllocator, ScratchBuffers
from zse_engine.orchestrator.kernels import InferenceKernels
from zse_engine.orchestrator.model_runner import TPModelRunner
from zse_engine.orchestrator.sampler import Sampler
from zse_engine.orchestrator.tensor_parallel import TensorParallelGroup, TPConfig
from zse_engine.orchestrator.tp_weight_loader import TPWeightLoader


# Commands sent from main process to workers
CMD_PREFILL = 1
CMD_DECODE = 2
CMD_STOP = 3
CMD_DESTROY = 4


@dataclass
class TPEngineStats:
    """Stats for tensor parallel engine."""
    tp_size: int
    backend: str
    device_names: List[str]
    total_vram_gb: float
    weight_load_time_s: float
    kernel_compile_time_s: float
    total_init_time_s: float


def _worker_process(
    rank: int,
    tp_size: int,
    model_path: str,
    backend: str,
    nccl_uid_bytes: bytes,
    cmd_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    quiet: bool,
    local_rank: Optional[int] = None,
):
    """Worker process for one GPU rank.

    Runs in its own process with its own GPU context.
    Receives commands from main process, runs forward pass, returns results.
    """
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Let main handle Ctrl+C

        # Step 1: Init GPU for this rank
        device_index = rank if local_rank is None else local_rank
        gpu_mem = GPUMemory(backend=backend, device_index=device_index)
        gpu_mem.ensure_context()

        devices = get_devices(backend)
        device = devices[device_index]
        if not quiet:
            print(f"[TP rank {rank}] GPU: {device.name} ({device.vram_total_gb:.1f}GB)")

        # Step 2: Create NCCL communicator
        tp_group = TensorParallelGroup(
            tp_size=tp_size,
            rank=rank,
            backend=backend,
            unique_id=nccl_uid_bytes,
        )
        if not quiet:
            print(f"[TP rank {rank}] NCCL communicator initialized")

        # Step 3: Load model (each rank opens the same file, loads its shard)
        loader = ZSELoader(model_path)
        config = loader.config

        # Validate TP compatibility
        tp_config = TPConfig(tp_size=tp_size, backend=backend)
        tp_config.validate(config.num_heads, config.num_kv_heads, config.intermediate_size)

        # Step 4: Plan VRAM
        allocator = VRAMAllocator(gpu_mem, device)
        # Estimate shard size (roughly total / tp_size for parallel weights)
        full_model_size = config.estimate_model_size_bytes()
        # Column+row parallel weights are ~95% of total; they split by tp_size
        # Replicated weights (norms, embed) are ~5% — full copy
        shard_model_size = int(full_model_size * 0.05 + full_model_size * 0.95 / tp_size)
        vram_plan = allocator.plan_allocation(shard_model_size, config)

        # Step 5: Compile kernels
        kernel_start = time.monotonic()
        kernels = InferenceKernels(backend=backend)
        quant_type = "int4" if config.quant.method == 1 else ("int8" if config.quant.method == 2 else "fp16")
        kernels.compile_all(quant_type=quant_type)
        kernel_time = time.monotonic() - kernel_start
        if not quiet:
            print(f"[TP rank {rank}] Kernels compiled in {kernel_time:.2f}s")

        # Step 6: Load weight shards
        weight_start = time.monotonic()
        tp_loader = TPWeightLoader(loader, gpu_mem, tp_group)
        weights = tp_loader.load_all()
        weight_time = time.monotonic() - weight_start
        if not quiet:
            print(f"[TP rank {rank}] Weights loaded: {weights.total_bytes / 1024**2:.1f}MB in {weight_time:.2f}s")

        # Step 7: Allocate scratch buffers
        # Local config for scratch sizing
        from copy import copy
        local_config = copy(config)
        local_config.num_heads = config.num_heads // tp_size
        local_config.num_kv_heads = config.num_kv_heads // tp_size
        local_config.intermediate_size = config.intermediate_size // tp_size

        max_seq_len = min(config.max_seq_len, 2048)
        scratch = allocator.allocate_scratch(local_config, max_seq_len=max_seq_len)

        # Step 8: KV cache (local heads only)
        kv_budget = vram_plan.kv_cache_bytes
        kv_cache = KVCacheManager(
            config=local_config,
            gpu_mem=gpu_mem,
            budget_bytes=kv_budget,
        )

        # Step 9: Create TP model runner
        runner = TPModelRunner(
            config=config,  # Full config — TPModelRunner adjusts internally
            weights=weights,
            kv_cache=kv_cache,
            scratch=scratch,
            gpu_mem=gpu_mem,
            kernels=kernels,
            tp_group=tp_group,
        )

        sampler = Sampler()

        # Signal ready
        result_queue.put(("ready", rank, {
            "rank": rank,
            "local_rank": device_index,
            "pid": os.getpid(),
            "device": device.name,
            "vram_gb": device.vram_total_gb,
            "weight_mb": weights.total_bytes / 1024**2,
            "kernel_time": kernel_time,
            "weight_time": weight_time,
        }))

        # Step 10: Command loop
        while True:
            cmd = cmd_queue.get()
            if cmd is None or cmd[0] == CMD_DESTROY:
                break

            cmd_type = cmd[0]

            if cmd_type == CMD_PREFILL:
                _, token_ids, seq_id = cmd
                logits = runner.prefill(token_ids, seq_id)
                if rank == 0:
                    result_queue.put(("logits", logits))
                # Other ranks don't send logits — rank 0 has the full result
                # after all-reduce

            elif cmd_type == CMD_DECODE:
                _, token_id, seq_id, position, skip_logits = cmd
                logits = runner.decode_step(
                    token_id, seq_id, position,
                    skip_logits_download=skip_logits,
                )
                if rank == 0:
                    if skip_logits:
                        # GPU argmax
                        token = runner.gpu_argmax()
                        result_queue.put(("token", token))
                    else:
                        result_queue.put(("logits", logits))

            elif cmd_type == CMD_STOP:
                _, seq_id = cmd
                kv_cache.free_sequence(seq_id)
                cache_stats = kv_cache.stats()
                result_queue.put(("released", rank, seq_id, {
                    "num_sequences": cache_stats.num_sequences,
                    "allocated_blocks": cache_stats.allocated_blocks,
                    "free_blocks": cache_stats.free_blocks,
                    "total_blocks": cache_stats.total_blocks,
                }))

        # Cleanup
        tp_group.destroy()
        weights.destroy(gpu_mem)
        scratch.destroy(gpu_mem)

    except Exception as e:
        result_queue.put(("error", rank, str(e)))
        import traceback
        traceback.print_exc()


class TPEngine:
    """Tensor Parallel Engine — multi-GPU inference.

    Spawns tp_size worker processes. Rank 0 handles tokenization and sampling.
    All ranks run forward pass in lockstep via NCCL.

    Args:
        model_path: Path to .zse model file
        tp_size: Number of GPUs to use
        quiet: Suppress output
    """

    def __init__(
        self,
        model_path: str,
        tp_size: int = 2,
        quiet: bool = False,
        remote_endpoints=None,
    ):
        remote_endpoints = list(remote_endpoints or [])
        if remote_endpoints and len(remote_endpoints) != tp_size - 1:
            raise ValueError("Cross-host TP requires one local rank and tp_size - 1 remote endpoints")
        if len({(endpoint.host, endpoint.port) for endpoint in remote_endpoints}) != len(remote_endpoints):
            raise ValueError("Remote TP endpoints must be unique")
        self._model_path = model_path
        self._tp_size = tp_size
        self._quiet = quiet
        self._seq_counter = 0
        self._total_tokens = 0
        self._total_gen_time = 0.0

        init_start = time.monotonic()

        # Detect backend
        backend = detect_backend()
        self._backend = backend
        devices = get_devices(backend)

        local_size = 1 if remote_endpoints else tp_size
        if len(devices) < local_size:
            raise RuntimeError(
                f"Requested {local_size} local GPUs but only {len(devices)} GPUs detected"
            )

        if not is_nccl_available(backend):
            lib = "RCCL" if backend == "rocm" else "NCCL"
            raise RuntimeError(f"{lib} not found. Required for multi-GPU tensor parallelism.")

        if not quiet:
            print(f"[ZSE-TP] Initializing {tp_size}-way tensor parallelism on {backend}")
            for i in range(local_size):
                print(f"  GPU {i}: {devices[i].name} ({devices[i].vram_total_gb:.1f}GB)")

        # Generate NCCL unique ID
        nccl_uid = get_unique_id(backend)

        # Load tokenizer on main process
        loader = ZSELoader(model_path)
        self._config = loader.config
        self._tokenizer = loader.tokenizer
        self._loader = loader

        # Validate TP compatibility
        tp_config = TPConfig(tp_size=tp_size, backend=backend)
        tp_config.validate(
            self._config.num_heads,
            self._config.num_kv_heads,
            self._config.intermediate_size,
        )

        # Spawn worker processes
        self._cmd_queues = []
        worker_context = multiprocessing.get_context("spawn")
        self._result_queue = worker_context.Queue()
        self._workers = []

        ready_info = {}
        try:
            for rank in range(tp_size):
                if remote_endpoints and rank > 0:
                    from zse_engine.orchestrator.tp_transport import RemoteTPWorker
                    remote_worker = RemoteTPWorker(
                        remote_endpoints[rank - 1], rank, tp_size, backend,
                        nccl_uid, self._result_queue,
                    )
                    self._cmd_queues.append(remote_worker)
                    self._workers.append(remote_worker)
                    continue
                cmd_q = worker_context.Queue()
                self._cmd_queues.append(cmd_q)
                worker = worker_context.Process(
                    target=_worker_process,
                    args=(rank, tp_size, model_path, backend, nccl_uid,
                          cmd_q, self._result_queue, quiet),
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)

            for _ in range(tp_size):
                msg = self._result_queue.get(timeout=300)
                if (len(msg) != 3 or msg[0] != "ready"
                        or msg[1] not in range(tp_size) or msg[1] in ready_info):
                    raise RuntimeError(f"Worker initialization failed: {msg}")
                ready_info[msg[1]] = msg[2]
        except BaseException:
            self.destroy()
            raise
        self._ready_info = ready_info

        self._total_init_time = time.monotonic() - init_start
        self._sampler = Sampler()

        if not quiet:
            total_weight_mb = sum(info["weight_mb"] for info in ready_info.values())
            max_kernel_time = max(info["kernel_time"] for info in ready_info.values())
            max_weight_time = max(info["weight_time"] for info in ready_info.values())
            print(f"[ZSE-TP] All {tp_size} ranks ready in {self._total_init_time:.2f}s")
            print(f"  Total weight shards: {total_weight_mb:.1f}MB across {tp_size} GPUs")
            print(f"  Kernel compile: {max_kernel_time:.2f}s, Weight load: {max_weight_time:.2f}s")

    def _broadcast_cmd(self, cmd):
        """Send command to all worker ranks."""
        for q in self._cmd_queues:
            q.put(cmd)

    def _default_stop_tokens(self):
        stop_tokens = set()
        eos_id = self._tokenizer.special_tokens.eos_id
        if eos_id is not None:
            stop_tokens.add(eos_id)
        for marker in ("<|im_end|>", "<|im_start|>", "<|eot_id|>"):
            token_ids = self._tokenizer.encode(marker, add_bos=False)
            if len(token_ids) == 1:
                stop_tokens.add(token_ids[0])
        return stop_tokens

    def _release_sequence(self, seq_id):
        self._broadcast_cmd((CMD_STOP, seq_id))
        released = {}
        for _ in range(self._tp_size):
            message = self._result_queue.get(timeout=30)
            if (message[0] != "released" or message[2] != seq_id
                    or message[1] not in range(self._tp_size) or message[1] in released):
                raise RuntimeError(f"Sequence release failed: {message}")
            released[message[1]] = message[3]
        self._last_release_stats = released

    @contextmanager
    def _request_sequence(self):
        seq_id = self._seq_counter
        self._seq_counter += 1
        try:
            yield seq_id
        except BaseException:
            try:
                self._release_sequence(seq_id)
            except Exception:
                pass
            raise
        else:
            self._release_sequence(seq_id)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_tokens: Optional[List[int]] = None,
    ) -> str:
        """Generate text from prompt using tensor parallelism."""
        with self._request_sequence() as seq_id:
            return self._generate(
                prompt, max_tokens, temperature, top_k, top_p, stop_tokens, seq_id,
            )

    def _generate(self, prompt, max_tokens, temperature, top_k, top_p, stop_tokens, seq_id):
        # Tokenize
        token_ids = self._tokenizer.encode(prompt)

        gen_start = time.monotonic()

        # Prefill — broadcast to all ranks
        self._broadcast_cmd((CMD_PREFILL, token_ids, seq_id))

        # Get logits from rank 0
        msg = self._result_queue.get(timeout=120)
        if msg[0] == "error":
            raise RuntimeError(f"Prefill failed: {msg}")
        logits_data = msg[1]

        # Sample first token
        next_token = self._sampler.sample(
            logits_data, self._config.vocab_size,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

        stop_ids = self._default_stop_tokens() | set(stop_tokens or [])
        generated = [next_token]
        position = len(token_ids)

        # Decode loop
        for step in range(max_tokens - 1):
            if next_token in stop_ids:
                break

            # Use GPU argmax for greedy (temperature ~0)
            use_gpu_argmax = (temperature < 0.01)

            self._broadcast_cmd((CMD_DECODE, next_token, seq_id, position, use_gpu_argmax))

            msg = self._result_queue.get(timeout=30)
            if msg[0] == "error":
                raise RuntimeError(f"Decode step {step} failed: {msg}")

            if use_gpu_argmax:
                next_token = msg[1]
            else:
                logits_data = msg[1]
                next_token = self._sampler.sample(
                    logits_data, self._config.vocab_size,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                )

            generated.append(next_token)
            position += 1

        gen_time = time.monotonic() - gen_start
        self._total_tokens += len(generated)
        self._total_gen_time += gen_time

        # Decode tokens to text
        return self._tokenizer.decode([token for token in generated if token not in stop_ids])

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Iterator[str]:
        """Stream-generate tokens one at a time."""
        with self._request_sequence() as seq_id:
            yield from self._stream_generate(prompt, max_tokens, temperature, top_k, seq_id)

    def _stream_generate(self, prompt, max_tokens, temperature, top_k, seq_id):
        token_ids = self._tokenizer.encode(prompt)

        # Prefill
        self._broadcast_cmd((CMD_PREFILL, token_ids, seq_id))
        msg = self._result_queue.get(timeout=120)
        if msg[0] == "error":
            raise RuntimeError(f"Prefill failed: {msg}")

        next_token = self._sampler.sample(
            msg[1], self._config.vocab_size,
            temperature=temperature, top_k=top_k,
        )
        stop_ids = self._default_stop_tokens()
        if next_token in stop_ids:
            return
        yield self._tokenizer.decode([next_token])

        position = len(token_ids)

        for _ in range(max_tokens - 1):
            self._broadcast_cmd((CMD_DECODE, next_token, seq_id, position, False))
            msg = self._result_queue.get(timeout=30)
            if msg[0] == "error":
                raise RuntimeError(f"Stream decode failed: {msg}")

            next_token = self._sampler.sample(
                msg[1], self._config.vocab_size,
                temperature=temperature, top_k=top_k,
            )
            if next_token in stop_ids:
                break
            yield self._tokenizer.decode([next_token])
            position += 1

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def tp_size(self) -> int:
        return self._tp_size

    def destroy(self):
        """Shutdown all worker processes."""
        for q in self._cmd_queues:
            try:
                q.put((CMD_DESTROY,))
            except Exception:
                pass

        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        self._workers.clear()
        self._cmd_queues.clear()

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def summary(self) -> str:
        tok_s = self._total_tokens / self._total_gen_time if self._total_gen_time > 0 else 0
        return (
            f"TPEngine: {self._tp_size}-way TP on {self._backend}\n"
            f"  Model: {self._model_path}\n"
            f"  Init time: {self._total_init_time:.2f}s\n"
            f"  Tokens generated: {self._total_tokens}\n"
            f"  Avg throughput: {tok_s:.1f} tok/s"
        )
