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
):
    """Worker process for one GPU rank.

    Runs in its own process with its own GPU context.
    Receives commands from main process, runs forward pass, returns results.
    """
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # Let main handle Ctrl+C

        # Step 1: Init GPU for this rank
        gpu_mem = GPUMemory(backend=backend, device_index=rank)
        gpu_mem.ensure_context()

        devices = get_devices(backend)
        device = devices[rank] if rank < len(devices) else devices[0]
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
                # Cancel current generation
                pass

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
    ):
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

        if len(devices) < tp_size:
            raise RuntimeError(
                f"Requested tp_size={tp_size} but only {len(devices)} GPUs detected"
            )

        if not is_nccl_available(backend):
            lib = "RCCL" if backend == "rocm" else "NCCL"
            raise RuntimeError(f"{lib} not found. Required for multi-GPU tensor parallelism.")

        if not quiet:
            print(f"[ZSE-TP] Initializing {tp_size}-way tensor parallelism on {backend}")
            for i in range(tp_size):
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
        self._result_queue = multiprocessing.Queue()
        self._workers = []

        for rank in range(tp_size):
            cmd_q = multiprocessing.Queue()
            self._cmd_queues.append(cmd_q)

            p = Process(
                target=_worker_process,
                args=(rank, tp_size, model_path, backend, nccl_uid,
                      cmd_q, self._result_queue, quiet),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        # Wait for all workers to be ready
        ready_info = {}
        for _ in range(tp_size):
            msg = self._result_queue.get(timeout=300)  # 5 min timeout
            if msg[0] == "error":
                raise RuntimeError(f"Worker rank {msg[1]} failed: {msg[2]}")
            assert msg[0] == "ready"
            ready_info[msg[1]] = msg[2]

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
        # Tokenize
        token_ids = self._tokenizer.encode(prompt)
        seq_id = self._seq_counter
        self._seq_counter += 1

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

        generated = [next_token]
        position = len(token_ids)

        # Decode loop
        if stop_tokens is None:
            stop_tokens = []
        eos = self._config.eos_token_id if hasattr(self._config, 'eos_token_id') else 2

        for step in range(max_tokens - 1):
            if next_token in stop_tokens or next_token == eos:
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
        return self._tokenizer.decode(generated)

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Iterator[str]:
        """Stream-generate tokens one at a time."""
        token_ids = self._tokenizer.encode(prompt)
        seq_id = self._seq_counter
        self._seq_counter += 1

        # Prefill
        self._broadcast_cmd((CMD_PREFILL, token_ids, seq_id))
        msg = self._result_queue.get(timeout=120)
        if msg[0] == "error":
            raise RuntimeError(f"Prefill failed: {msg}")

        next_token = self._sampler.sample(
            msg[1], self._config.vocab_size,
            temperature=temperature, top_k=top_k,
        )
        yield self._tokenizer.decode([next_token])

        position = len(token_ids)
        eos = self._config.eos_token_id if hasattr(self._config, 'eos_token_id') else 2

        for _ in range(max_tokens - 1):
            if next_token == eos:
                break

            self._broadcast_cmd((CMD_DECODE, next_token, seq_id, position, False))
            msg = self._result_queue.get(timeout=30)
            if msg[0] == "error":
                break

            next_token = self._sampler.sample(
                msg[1], self._config.vocab_size,
                temperature=temperature, top_k=top_k,
            )
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
