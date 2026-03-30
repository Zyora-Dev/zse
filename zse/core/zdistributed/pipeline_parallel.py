"""
Pipeline Parallelism for ZSE

Splits transformer layers sequentially across GPUs.
Each GPU holds a contiguous range of layers and passes
activations to the next GPU via point-to-point communication.

Architecture (4 GPUs, 28-layer model):
    GPU 0: embed_tokens + layers[0:7]
    GPU 1: layers[7:14]
    GPU 2: layers[14:21]
    GPU 3: layers[21:28] + norm + lm_head

Communication: NCCL point-to-point send/recv between adjacent stages.
Micro-batching: Splits batch into micro-batches to pipeline and keep GPUs busy.

Usage:
    coord = PPCoordinator(model_path="model.zse", pp_size=4)
    coord.start()
    output_ids = coord.generate(input_ids, max_new_tokens=128)
    coord.shutdown()
"""

import os
import sys
import gc
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class StageInfo:
    """Describes what a pipeline stage holds."""
    rank: int
    layer_start: int   # inclusive
    layer_end: int      # exclusive
    has_embed: bool     # stage 0 holds embedding
    has_head: bool      # last stage holds norm + lm_head
    device: str


def compute_stage_assignments(
    num_layers: int,
    pp_size: int,
) -> List[StageInfo]:
    """
    Compute which layers go to which stage.
    
    Distributes layers as evenly as possible.
    Stage 0 gets embedding, last stage gets norm + lm_head.
    """
    layers_per_stage = num_layers // pp_size
    remainder = num_layers % pp_size
    
    stages = []
    current = 0
    for rank in range(pp_size):
        # Distribute remainder evenly among first stages
        count = layers_per_stage + (1 if rank < remainder else 0)
        stages.append(StageInfo(
            rank=rank,
            layer_start=current,
            layer_end=current + count,
            has_embed=(rank == 0),
            has_head=(rank == pp_size - 1),
            device=f"cuda:{rank}",
        ))
        current += count
    
    return stages


def _pp_worker_main(
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
    Main function for each PP worker process.
    
    Each worker holds a range of layers and processes activations
    received from the previous stage, then passes them to the next.
    """
    import traceback
    
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        torch.cuda.set_device(rank)
        print(f"[PP Worker {rank}] Initializing process group (backend={backend})", flush=True)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(f"[PP Worker {rank}] Process group initialized", flush=True)
        
        # Load this stage's model shard
        print(f"[PP Worker {rank}] Loading model stage...", flush=True)
        stage_model, tokenizer, stage_info = _load_pp_stage(
            model_path, rank, world_size
        )
        
        vram_mb = torch.cuda.memory_allocated(rank) / 1024**2
        print(
            f"[PP Worker {rank}] Stage loaded: layers[{stage_info.layer_start}:{stage_info.layer_end}]"
            f" embed={stage_info.has_embed} head={stage_info.has_head}"
            f" VRAM: {vram_mb:.0f} MB",
            flush=True,
        )
        
        model_loaded_barrier.wait()
        
        if rank == 0:
            output_queue.put({
                "status": "ready",
                "rank": rank,
                "num_layers": stage_info.layer_end - stage_info.layer_start,
            })
        
        # Worker loop
        while True:
            try:
                cmd = input_queue.get()
            except Exception:
                break
            
            if cmd is None or cmd.get("action") == "shutdown":
                break
            
            try:
                if cmd["action"] == "generate":
                    _pp_generate_loop(
                        rank, world_size, stage_model, tokenizer,
                        stage_info, cmd, output_queue,
                    )
                
                elif cmd["action"] == "forward":
                    _pp_forward_step(
                        rank, world_size, stage_model, stage_info, cmd, output_queue,
                    )
            
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[PP Worker {rank}] ERROR: {e}\n{tb}", flush=True)
                if rank == world_size - 1:
                    output_queue.put({"error": str(e), "traceback": tb})
                break
    
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[PP Worker {rank}] FATAL ERROR: {e}\n{tb}", flush=True)
        try:
            output_queue.put({"status": "error", "rank": rank, "error": str(e), "traceback": tb})
        except Exception:
            pass
        try:
            model_loaded_barrier.wait()
        except Exception:
            pass
        return
    
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _load_pp_stage(
    model_path: str,
    rank: int,
    world_size: int,
) -> Tuple[Dict[str, Any], Any, StageInfo]:
    """
    Load a pipeline stage's portion of the model.
    
    Returns a dict with the relevant model components rather than
    the whole model, to save VRAM.
    """
    device = f"cuda:{rank}"
    model_p = Path(model_path)
    
    # Load the full model first (then prune)
    if model_p.exists() and (model_p.suffix == ".zse" or
        (model_p.is_dir() and (model_p / "model.zse").exists())):
        from zse.format.reader_v2 import load_zse_model
        zse_file = model_p if model_p.suffix == ".zse" else model_p / "model.zse"
        full_model, tokenizer, _ = load_zse_model(str(zse_file), device=device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        full_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    # Detect layer structure
    inner_model = _get_inner_model(full_model)
    layers = _get_layers(inner_model)
    num_layers = len(layers)
    
    # Compute stage assignment
    stages = compute_stage_assignments(num_layers, world_size)
    stage_info = stages[rank]
    
    # Extract this stage's components
    stage_model = {}
    
    if stage_info.has_embed:
        stage_model["embed_tokens"] = _get_embedding(inner_model).to(device)
    
    # Always keep the rotary embedding module for position embeddings
    # Recreate on correct device (the original may have meta tensors)
    rotary_emb = _get_rotary_emb(inner_model)
    if rotary_emb is not None:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        try:
            stage_model["rotary_emb"] = Qwen2RotaryEmbedding(full_model.config, device=device)
        except Exception:
            # Fallback: try generic copy
            try:
                stage_model["rotary_emb"] = rotary_emb.to_empty(device=device)
                # Re-register inv_freq buffer
                rope_config = full_model.config
                head_dim = getattr(rope_config, "head_dim",
                                   rope_config.hidden_size // rope_config.num_attention_heads)
                base = getattr(rope_config, "rope_theta", 10000.0)
                inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
                stage_model["rotary_emb"].inv_freq = inv_freq
            except Exception:
                pass  # Will use fallback in _compute_position_embeddings
    
    # Extract assigned layers
    stage_layers = nn.ModuleList()
    for i in range(stage_info.layer_start, stage_info.layer_end):
        stage_layers.append(layers[i].to(device))
    stage_model["layers"] = stage_layers
    
    if stage_info.has_head:
        stage_model["norm"] = _get_final_norm(inner_model).to(device)
        stage_model["lm_head"] = full_model.lm_head.to(device)
    
    # Store config for shape info
    stage_model["config"] = full_model.config
    
    # Free the full model
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return stage_model, tokenizer, stage_info


def _get_inner_model(model: nn.Module) -> nn.Module:
    """Get the inner transformer model (model.model for CausalLM wrappers)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer
    return model


def _get_layers(inner_model: nn.Module) -> nn.ModuleList:
    """Get the decoder layers ModuleList."""
    if hasattr(inner_model, "layers"):
        return inner_model.layers
    if hasattr(inner_model, "h"):
        return inner_model.h
    raise ValueError("Cannot find decoder layers in model")


def _get_embedding(inner_model: nn.Module) -> nn.Module:
    """Get the input embedding layer."""
    if hasattr(inner_model, "embed_tokens"):
        return inner_model.embed_tokens
    if hasattr(inner_model, "wte"):
        return inner_model.wte
    raise ValueError("Cannot find embedding layer in model")


def _get_final_norm(inner_model: nn.Module) -> nn.Module:
    """Get the final layer norm."""
    if hasattr(inner_model, "norm"):
        return inner_model.norm
    if hasattr(inner_model, "ln_f"):
        return inner_model.ln_f
    raise ValueError("Cannot find final norm in model")


def _get_rotary_emb(inner_model: nn.Module) -> Optional[nn.Module]:
    """Get the rotary embedding module (shared across layers)."""
    if hasattr(inner_model, "rotary_emb"):
        return inner_model.rotary_emb
    return None


def _pp_forward_step(
    rank: int,
    world_size: int,
    stage_model: Dict[str, Any],
    stage_info: StageInfo,
    cmd: Dict,
    output_queue,
):
    """
    Execute one forward pass through this pipeline stage.
    
    Stage 0: embed input_ids → run layers → send to stage 1
    Middle stages: recv from prev → run layers → send to next
    Last stage: recv from prev → run layers → norm → lm_head → output
    """
    device = f"cuda:{rank}"
    config = stage_model["config"]
    
    with torch.no_grad():
        if stage_info.has_embed:
            # Stage 0: embed the input tokens
            input_ids = cmd["input_ids"].to(device)
            hidden = stage_model["embed_tokens"](input_ids)
        else:
            # Receive hidden states from previous stage
            hidden_shape = cmd["hidden_shape"]
            hidden = torch.empty(hidden_shape, dtype=torch.float16, device=device)
            dist.recv(hidden, src=rank - 1)
        
        # Get position info for RoPE
        seq_len = hidden.shape[1]
        cache_position = cmd.get("cache_position")
        if cache_position is not None:
            cache_position = cache_position.to(device)
        else:
            cache_position = torch.arange(seq_len, device=device)
        
        position_ids = cache_position.unsqueeze(0).expand(hidden.shape[0], -1)
        
        # Compute position embeddings (cos, sin for RoPE)
        # We need rotary_emb from the model config
        position_embeddings = _compute_position_embeddings(
            stage_model, hidden, position_ids, device,
        )
        
        # Run through this stage's layers
        past_key_values = cmd.get("past_key_values")
        use_cache = cmd.get("use_cache", True)
        new_past = []
        
        for i, layer in enumerate(stage_model["layers"]):
            global_idx = stage_info.layer_start + i
            
            # Get layer's past KV cache
            layer_past = None
            if past_key_values is not None and global_idx < len(past_key_values):
                layer_past = past_key_values[global_idx]
            
            layer_output = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=cmd.get("attention_mask"),
                past_key_values=layer_past,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            
            # Handle different output formats
            if isinstance(layer_output, tuple):
                hidden = layer_output[0]
                if use_cache and len(layer_output) > 1:
                    new_past.append(layer_output[1])
            else:
                hidden = layer_output
        
        if stage_info.has_head:
            # Last stage: apply final norm and lm_head
            hidden = stage_model["norm"](hidden)
            logits = stage_model["lm_head"](hidden)
            
            # Send result back to coordinator (rank 0 gets it via queue)
            output_queue.put({
                "logits": logits.cpu(),
                "past_key_values": new_past if use_cache else None,
            })
        else:
            # Send hidden states to next stage
            dist.send(hidden.contiguous(), dst=rank + 1)


def _compute_position_embeddings(
    stage_model: Dict[str, Any],
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RoPE position embeddings (cos, sin).
    
    Uses the model's own rotary_emb module when available to ensure
    identical results to the single-GPU path (handles attention_scaling,
    rope_type, etc.).
    """
    # Use the actual model rotary embedding if we have it
    rotary_emb = stage_model.get("rotary_emb")
    if rotary_emb is not None:
        return rotary_emb(hidden, position_ids)
    
    # Fallback: compute from config (less reliable)
    config = stage_model["config"]
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    base = getattr(config, "rope_theta", 10000.0)
    
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    
    with torch.autocast(device_type="cuda", enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(hidden.dtype)
        sin = emb.sin().to(hidden.dtype)
    
    return (cos, sin)


def _pp_generate_loop(
    rank: int,
    world_size: int,
    stage_model: Dict[str, Any],
    tokenizer: Any,
    stage_info: StageInfo,
    cmd: Dict,
    output_queue,
):
    """
    Run autoregressive generation across the pipeline.
    
    For each token:
    1. Stage 0 embeds the current token(s) and runs its layers
    2. Activation passes through intermediate stages
    3. Last stage produces logits, selects next token
    4. Next token is broadcast to all stages for KV cache consistency
    
    This is coordinated via NCCL send/recv between adjacent stages
    and broadcast for the selected token.
    """
    device = f"cuda:{rank}"
    gen_kwargs = cmd.get("kwargs", {})
    max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
    do_sample = gen_kwargs.get("do_sample", False)
    temperature = gen_kwargs.get("temperature", 1.0)
    top_p = gen_kwargs.get("top_p", 1.0)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    
    config = stage_model["config"]
    
    if stage_info.has_embed:
        input_ids = cmd["input_ids"].to(device)
        all_token_ids = input_ids.clone()
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
    else:
        # Non-embed stages get shape info via broadcast
        shape_tensor = torch.zeros(2, dtype=torch.long, device=device)
        dist.broadcast(shape_tensor, src=0)
        batch_size = shape_tensor[0].item()
        prompt_len = shape_tensor[1].item()
        all_token_ids = torch.zeros(batch_size, prompt_len, dtype=torch.long, device=device)
    
    # Stage 0 broadcasts shape info
    if stage_info.has_embed:
        shape_tensor = torch.tensor([batch_size, prompt_len], dtype=torch.long, device=device)
        dist.broadcast(shape_tensor, src=0)
    
    hidden_size = config.hidden_size
    
    # ─── Prefill phase: process all prompt tokens ────────────
    # Create a DynamicCache for this stage's layers
    from transformers import DynamicCache
    kv_cache = DynamicCache(config=config)
    
    # Stage 0: embed
    if stage_info.has_embed:
        hidden = stage_model["embed_tokens"](input_ids)
    else:
        hidden = torch.empty(batch_size, prompt_len, hidden_size, dtype=torch.float16, device=device)
        dist.recv(hidden, src=rank - 1)
    
    # Position embeddings for prefill
    cache_position = torch.arange(prompt_len, device=device)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    position_embeddings = _compute_position_embeddings(
        stage_model, hidden, position_ids, device,
    )
    
    # Run through this stage's layers (prefill)
    for i, layer in enumerate(stage_model["layers"]):
        hidden = layer(
            hidden,
            position_embeddings=position_embeddings,
            past_key_values=kv_cache,
            use_cache=True,
            cache_position=cache_position,
        )
    
    if stage_info.has_head:
        # Last stage: get logits for last position
        normed = stage_model["norm"](hidden)
        logits = stage_model["lm_head"](normed)
        next_logits = logits[:, -1, :]  # [batch, vocab]
    else:
        # Pass hidden to next stage
        dist.send(hidden.contiguous(), dst=rank + 1)
    
    # ─── Decode phase: generate tokens one by one ────────────
    generated_tokens = []
    
    for step in range(max_new_tokens):
        # Last stage selects next token and broadcasts
        if stage_info.has_head:
            if do_sample and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, 1))
                else:
                    next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
            
            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                # Broadcast stop signal (-1)
                stop_token = torch.tensor([[-1]], dtype=torch.long, device=device).expand(batch_size, 1)
                dist.broadcast(stop_token, src=world_size - 1)
                generated_tokens.append(next_token)
                break
            
            dist.broadcast(next_token, src=world_size - 1)
            generated_tokens.append(next_token)
        else:
            # Receive broadcasted next token from last stage
            next_token = torch.empty(batch_size, 1, dtype=torch.long, device=device)
            dist.broadcast(next_token, src=world_size - 1)
            
            # Check stop signal
            if next_token[0, 0].item() == -1:
                break
        
        # ─── Forward pass for next token ─────────────────────
        cur_pos = prompt_len + step
        cache_pos_step = torch.tensor([cur_pos], device=device)
        position_ids_step = cache_pos_step.unsqueeze(0).expand(batch_size, -1)
        
        if stage_info.has_embed:
            hidden = stage_model["embed_tokens"](next_token)
        else:
            hidden = torch.empty(batch_size, 1, hidden_size, dtype=torch.float16, device=device)
            dist.recv(hidden, src=rank - 1)
        
        pos_emb = _compute_position_embeddings(
            stage_model, hidden, position_ids_step, device,
        )
        
        for i, layer in enumerate(stage_model["layers"]):
            hidden = layer(
                hidden,
                position_embeddings=pos_emb,
                past_key_values=kv_cache,
                use_cache=True,
                cache_position=cache_pos_step,
            )
        
        if stage_info.has_head:
            normed = stage_model["norm"](hidden)
            logits = stage_model["lm_head"](normed)
            next_logits = logits[:, -1, :]
        else:
            dist.send(hidden.contiguous(), dst=rank + 1)
    
    # ─── Return result from last stage ───────────────────────
    if stage_info.has_head:
        gen_ids = torch.cat(generated_tokens, dim=-1)  # [batch, num_generated]
        # Also need original input_ids — broadcast from stage 0
        full_input_ids = torch.empty(batch_size, prompt_len, dtype=torch.long, device=device)
        dist.broadcast(full_input_ids, src=0)
        
        output_ids = torch.cat([full_input_ids, gen_ids], dim=-1)
        output_queue.put({"output_ids": output_ids.cpu()})
    
    if stage_info.has_embed:
        # Stage 0 broadcasts input_ids for final concatenation
        dist.broadcast(all_token_ids, src=0)


class PPCoordinator:
    """
    Coordinates pipeline parallel workers.
    
    Spawns N worker processes (one per GPU), each holding a stage
    of the model (contiguous range of layers).
    
    Stage 0: embedding + first layers
    Stage 1..N-2: middle layers
    Stage N-1: last layers + norm + lm_head
    
    Provides: forward(input_ids) → logits, generate(input_ids) → output_ids
    
    Usage:
        coord = PPCoordinator(model_path="model.zse", pp_size=2)
        coord.start()
        output_ids = coord.generate(input_ids, max_new_tokens=128)
        coord.shutdown()
    """
    
    def __init__(
        self,
        model_path: str,
        pp_size: int,
        backend: str = "nccl",
        master_addr: str = "127.0.0.1",
        master_port: str = "29600",
    ):
        self.model_path = model_path
        self.pp_size = pp_size
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
        
        barrier = mp.Barrier(self.pp_size)
        self._output_queue = mp.Queue()
        
        if verbose:
            print(f"🔗 Launching {self.pp_size} PP stages...", flush=True)
        
        for rank in range(self.pp_size):
            q = mp.Queue()
            self._input_queues.append(q)
            
            p = mp.Process(
                target=_pp_worker_main,
                args=(
                    rank,
                    self.pp_size,
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
            print(f"   Waiting for {self.pp_size} stages to load...", flush=True)
        
        # Wait for readiness from stage 0
        result = self._output_queue.get(timeout=600)
        if result.get("status") == "error":
            self.shutdown()
            raise RuntimeError(
                f"PP worker {result.get('rank', '?')} failed:\n"
                f"{result.get('error', 'Unknown')}\n{result.get('traceback', '')}"
            )
        assert result["status"] == "ready", f"Unexpected: {result}"
        
        self._started = True
        if verbose:
            print(f"   ✅ All {self.pp_size} stages ready", flush=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run one forward pass through the pipeline."""
        assert self._started, "Call start() first"
        
        cmd = {
            "action": "forward",
            "input_ids": input_ids.cpu(),
            "hidden_shape": list(input_ids.shape) + [kwargs.get("hidden_size", 0)],
            "use_cache": kwargs.get("use_cache", True),
        }
        
        for q in self._input_queues:
            q.put(cmd)
        
        result = self._output_queue.get(timeout=120)
        if "error" in result:
            raise RuntimeError(
                f"PP error during forward:\n{result['error']}\n{result.get('traceback', '')}"
            )
        return result
    
    def generate(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run autoregressive generation across the pipeline.
        
        Activations flow through stages sequentially for each token.
        Last stage selects the next token and broadcasts it.
        """
        assert self._started, "Call start() first"
        
        cmd = {
            "action": "generate",
            "input_ids": input_ids.cpu(),
            "kwargs": kwargs,
        }
        
        for q in self._input_queues:
            q.put(cmd)
        
        result = self._output_queue.get(timeout=600)
        if "error" in result:
            raise RuntimeError(
                f"PP error during generate:\n{result['error']}\n{result.get('traceback', '')}"
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
