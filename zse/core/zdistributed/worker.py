"""
Tensor Parallel Worker

Each worker process owns one GPU and holds one shard of the model.
Workers communicate via NCCL all-reduce during forward passes.

Architecture:
    Rank 0 (coordinator): Receives inputs, broadcasts, collects logits
    Rank 1..N-1 (workers): Receive inputs, run forward, participate in all-reduce

The coordinator also runs the generation loop / serves the API.
Workers just execute forward passes in lockstep.

Usage (internal — launched by TPCoordinator):
    torchrun --nproc_per_node=2 -m zse.core.zdistributed.worker --model path.zse
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import time


def _worker_main(
    rank: int,
    world_size: int,
    model_path: str,
    backend: str,
    master_addr: str,
    master_port: str,
    model_loaded_barrier,
    input_queue,
    output_queue,
):
    """
    Main function for each TP worker process.
    
    Args:
        rank: This worker's GPU index
        world_size: Total number of GPUs
        model_path: Path to model (HF ID, local dir, or .zse file)
        backend: NCCL backend string
        master_addr: Address for rendezvous
        master_port: Port for rendezvous
        model_loaded_barrier: mp.Barrier to sync after loading
        input_queue: Queue to receive (input_ids, kwargs) from coordinator
        output_queue: Queue to send logits back (rank 0 only)
    """
    import traceback
    
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        # Initialize process group
        torch.cuda.set_device(rank)
        print(f"[Worker {rank}] Initializing process group (backend={backend})", flush=True)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(f"[Worker {rank}] Process group initialized", flush=True)
        
        # Create TP group
        tp_group = dist.new_group(list(range(world_size)))
        
        # Load model on this rank's GPU
        print(f"[Worker {rank}] Loading model shard...", flush=True)
        model, tokenizer = _load_model_shard(
            model_path, rank, world_size, tp_group
        )
        
        vram_mb = torch.cuda.memory_allocated(rank) / 1024**2
        print(f"[Worker {rank}] Model loaded. VRAM: {vram_mb:.0f} MB", flush=True)
        
        # Signal that model is loaded
        model_loaded_barrier.wait()
        
        if rank == 0:
            # Coordinator: put tokenizer info in output queue
            output_queue.put({"status": "ready", "rank": rank})
        
        # Worker loop: receive inputs, run forward, send outputs
        while True:
            try:
                cmd = input_queue.get()
            except Exception:
                break
            
            if cmd is None or cmd.get("action") == "shutdown":
                break
            
            try:
                if cmd["action"] == "forward":
                    input_ids = cmd["input_ids"].to(f"cuda:{rank}")
                    kwargs = cmd.get("kwargs", {})
                    
                    # Move past_key_values to this device if present
                    past = kwargs.get("past_key_values")
                    if past is not None:
                        kwargs["past_key_values"] = _move_cache_to_device(past, rank)
                    
                    with torch.no_grad():
                        output = model(input_ids=input_ids, **kwargs)
                    
                    # Only rank 0 sends output back
                    if rank == 0:
                        # Move output to CPU to avoid GPU memory buildup in queue
                        result = {
                            "logits": output.logits.cpu(),
                            "past_key_values": output.past_key_values,
                        }
                        output_queue.put(result)
                
                elif cmd["action"] == "generate":
                    input_ids = cmd["input_ids"].to(f"cuda:{rank}")
                    gen_kwargs = cmd.get("kwargs", {})
                    print(f"[Worker {rank}] generate() start, input shape: {input_ids.shape}", flush=True)
                    
                    with torch.no_grad():
                        output_ids = model.generate(input_ids=input_ids, **gen_kwargs)
                    
                    print(f"[Worker {rank}] generate() done, output shape: {output_ids.shape}", flush=True)
                    
                    if rank == 0:
                        output_queue.put({"output_ids": output_ids.cpu()})
            
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Worker {rank}] ERROR in {cmd.get('action', '?')}: {e}\n{tb}", flush=True)
                if rank == 0:
                    output_queue.put({"error": str(e), "traceback": tb})
                # Don't break — let other workers continue for barrier sync
                # But for generate, all ranks must participate, so we need to break
                break
    
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Worker {rank}] FATAL ERROR during init: {e}\n{tb}", flush=True)
        # Try to unblock the coordinator
        try:
            output_queue.put({"status": "error", "rank": rank, "error": str(e), "traceback": tb})
        except Exception:
            pass
        # Try to unblock the barrier
        try:
            model_loaded_barrier.wait()
        except Exception:
            pass
        return
    
    # Cleanup
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _load_model_shard(
    model_path: str,
    rank: int,
    world_size: int,
    tp_group,
) -> Tuple[nn.Module, Any]:
    """
    Load model and apply TP sharding for this rank.
    
    Each rank loads the full model on its GPU, then applies TP to keep
    only its shard. This is memory-inefficient during loading but simple.
    For production, we'd stream only the relevant shard from disk.
    """
    device = f"cuda:{rank}"
    model_p = Path(model_path)
    
    # Load full model
    if model_p.exists() and (model_p.suffix == ".zse" or
        (model_p.is_dir() and (model_p / "model.zse").exists())):
        from zse.format.reader_v2 import load_zse_model
        zse_file = model_p if model_p.suffix == ".zse" else model_p / "model.zse"
        model, tokenizer, _ = load_zse_model(str(zse_file), device=device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    # Apply tensor parallelism for this rank
    from zse.core.zdistributed.tensor_parallel import TensorParallel
    tp = TensorParallel(tp_size=world_size, tp_rank=rank, tp_group=tp_group)
    model = tp.apply(model)
    
    # Ensure all parameters are on the correct device
    # TP replaces layers with new ones whose Parameters may be on CPU
    # Skip meta tensors (uninitialized placeholders from model skeleton)
    target_device = torch.device(device)
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            continue
        if param.device != target_device:
            param.data = param.data.to(device)
    for name, buf in model.named_buffers():
        if buf is None or buf.device == torch.device("meta"):
            continue
        if buf.device != target_device:
            buf.data = buf.data.to(device)
    
    model.eval()
    
    # Free memory from non-shard weights
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, tokenizer


def _move_cache_to_device(past_key_values, rank: int):
    """Move KV cache tensors to the specified GPU."""
    device = f"cuda:{rank}"
    if past_key_values is None:
        return None
    
    # DynamicCache (transformers 5.x)
    if hasattr(past_key_values, 'layers'):
        for layer in past_key_values.layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
        return past_key_values
    
    # DynamicCache (transformers 4.x)
    if hasattr(past_key_values, 'key_cache'):
        for i in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[i] = past_key_values.key_cache[i].to(device)
            past_key_values.value_cache[i] = past_key_values.value_cache[i].to(device)
        return past_key_values
    
    # Legacy tuple format
    return tuple(
        (k.to(device), v.to(device)) for k, v in past_key_values
    )


class TPCoordinator:
    """
    Coordinates tensor parallel workers from a single main process.
    
    Spawns N worker processes (one per GPU), each holding a model shard.
    Provides a simple interface: forward(input_ids) → logits.
    
    Usage:
        coord = TPCoordinator(model_path="model.zse", tp_size=4)
        coord.start()
        
        # Use like a normal model
        logits = coord.forward(input_ids)
        output_ids = coord.generate(input_ids, max_new_tokens=128)
        
        coord.shutdown()
    """
    
    def __init__(
        self,
        model_path: str,
        tp_size: int,
        backend: str = "nccl",
        master_addr: str = "127.0.0.1",
        master_port: str = "29500",
    ):
        self.model_path = model_path
        self.tp_size = tp_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        
        self._processes = []
        self._input_queues = []
        self._output_queue = None
        self._started = False
    
    def start(self, verbose: bool = True) -> None:
        """Spawn worker processes and wait for model loading."""
        if self._started:
            return
        
        mp.set_start_method("spawn", force=True)
        
        barrier = mp.Barrier(self.tp_size)
        self._output_queue = mp.Queue()
        
        if verbose:
            print(f"🔀 Launching {self.tp_size} TP workers...")
        
        for rank in range(self.tp_size):
            q = mp.Queue()
            self._input_queues.append(q)
            
            p = mp.Process(
                target=_worker_main,
                args=(
                    rank,
                    self.tp_size,
                    self.model_path,
                    self.backend,
                    self.master_addr,
                    self.master_port,
                    barrier,
                    q,
                    self._output_queue,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
        
        if verbose:
            print(f"   Waiting for models to load on {self.tp_size} GPUs...")
        
        # Wait for rank 0 ready signal
        result = self._output_queue.get(timeout=600)  # 10 min timeout
        if result.get("status") == "error":
            error_msg = result.get("error", "Unknown error")
            tb = result.get("traceback", "")
            self.shutdown()
            raise RuntimeError(
                f"TP worker {result.get('rank', '?')} failed during init:\n{error_msg}\n{tb}"
            )
        assert result["status"] == "ready", f"Unexpected status: {result}"
        
        self._started = True
        if verbose:
            print(f"   ✅ All {self.tp_size} workers ready")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run forward pass across all TP workers.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            **kwargs: Additional model kwargs (use_cache, past_key_values, etc.)
        
        Returns:
            Dict with 'logits' and 'past_key_values'
        """
        assert self._started, "Call start() first"
        
        cmd = {
            "action": "forward",
            "input_ids": input_ids.cpu(),
            "kwargs": {k: v for k, v in kwargs.items() if k != "past_key_values"},
        }
        
        # Handle past_key_values separately (keep on CPU for transfer)
        if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
            cmd["kwargs"]["past_key_values"] = kwargs["past_key_values"]
        
        # Send to all workers
        for q in self._input_queues:
            q.put(cmd)
        
        # Get result from rank 0
        result = self._output_queue.get(timeout=120)
        if "error" in result:
            raise RuntimeError(
                f"TP worker error during forward:\n{result['error']}\n{result.get('traceback', '')}"
            )
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run model.generate() across all TP workers.
        
        Leverages HuggingFace's built-in generation loop which
        handles KV caching internally. All-reduce happens in the
        forward pass of each TP layer.
        
        Returns:
            Output token IDs
        """
        assert self._started, "Call start() first"
        
        cmd = {
            "action": "generate",
            "input_ids": input_ids.cpu(),
            "kwargs": kwargs,
        }
        
        for q in self._input_queues:
            q.put(cmd)
        
        result = self._output_queue.get(timeout=300)
        if "error" in result:
            raise RuntimeError(
                f"TP worker error during generate:\n{result['error']}\n{result.get('traceback', '')}"
            )
        return result["output_ids"]
    
    def shutdown(self) -> None:
        """Stop all worker processes."""
        for q in self._input_queues:
            try:
                q.put({"action": "shutdown"})
            except Exception:
                pass
        
        for p in self._processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        
        self._processes.clear()
        self._input_queues.clear()
        self._started = False
    
    def __del__(self):
        if self._started:
            self.shutdown()
