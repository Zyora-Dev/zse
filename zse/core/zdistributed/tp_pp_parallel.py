"""
Combined Tensor + Pipeline Parallelism (TP-PP)

Arranges GPUs in a 2-D grid:  pp_size × tp_size  =  total GPUs.

Example — 4 GPUs, pp_size=2, tp_size=2:

    PP stage 0 (layers 0-13):   GPU 0  GPU 1   ← TP group [0,1]
    PP stage 1 (layers 14-27):  GPU 2  GPU 3   ← TP group [2,3]

Within a stage:
  • TensorParallel.apply() shards each layer's weights across the
    TP group.  All-reduce happens inside every forward() call.

Between stages:
  • TP-rank 0 of stage N  sends  hidden_states  to  TP-rank 0 of stage N+1
    via NCCL point-to-point send/recv.
  • TP-rank 0 of the receiving stage broadcasts the tensor to the rest of
    its TP group so every rank has the full activation.

Generation loop:
  • Last stage's TP-rank 0 selects the next token and broadcasts to ALL
    ranks so everyone can update KV caches.

Usage:
    coord = TPPPCoordinator(model_path="model.zse", tp_size=2, pp_size=2)
    coord.start()
    output_ids = coord.generate(input_ids, max_new_tokens=128)
    coord.shutdown()
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from zse.core.zdistributed.pipeline_parallel import (
    StageInfo,
    compute_stage_assignments,
    _compute_position_embeddings,
    _get_inner_model,
    _get_layers,
    _get_embedding,
    _get_final_norm,
    _get_rotary_emb,
)


# -----------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------


@dataclass
class GridPosition:
    """A worker's position in the 2-D grid."""

    global_rank: int  # 0 … (tp_size*pp_size - 1)
    pp_rank: int  # pipeline stage index
    tp_rank: int  # tensor-parallel rank within the stage
    pp_size: int
    tp_size: int
    device: str  # e.g. "cuda:2"

    @property
    def is_first_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @property
    def is_tp_root(self) -> bool:
        """TP-rank 0 handles inter-stage send/recv."""
        return self.tp_rank == 0

    @property
    def prev_stage_tp_root(self) -> int:
        """Global rank of TP-rank 0 in the previous PP stage."""
        return (self.pp_rank - 1) * self.tp_size

    @property
    def next_stage_tp_root(self) -> int:
        """Global rank of TP-rank 0 in the next PP stage."""
        return (self.pp_rank + 1) * self.tp_size


def _grid_pos(global_rank: int, tp_size: int, pp_size: int) -> GridPosition:
    pp_rank = global_rank // tp_size
    tp_rank = global_rank % tp_size
    return GridPosition(
        global_rank=global_rank,
        pp_rank=pp_rank,
        tp_rank=tp_rank,
        pp_size=pp_size,
        tp_size=tp_size,
        device=f"cuda:{global_rank}",
    )


# -----------------------------------------------------------------------
# Worker entry point
# -----------------------------------------------------------------------


def _tppp_worker_main(
    rank: int,
    world_size: int,
    tp_size: int,
    pp_size: int,
    model_path: str,
    backend: str,
    master_addr: str,
    master_port: str,
    model_loaded_barrier,
    input_queue,
    output_queue,
):
    """
    Entry point for each GPU worker in the TP-PP grid.
    """
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        torch.cuda.set_device(rank)
        pos = _grid_pos(rank, tp_size, pp_size)

        # ---------- init global process group ----------
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(
            f"[TP-PP {rank}] pp={pos.pp_rank} tp={pos.tp_rank} device={pos.device}  group_init=OK",
            flush=True,
        )

        # ---------- create TP sub-groups (one per stage) ----------
        tp_groups: Dict[int, dist.ProcessGroup] = {}
        for stage in range(pp_size):
            ranks_in_stage = list(range(stage * tp_size, (stage + 1) * tp_size))
            tp_groups[stage] = dist.new_group(ranks_in_stage)
        my_tp_group = tp_groups[pos.pp_rank]

        # ---------- load model stage + apply TP sharding ----------
        stage_model, tokenizer, stage_info = _load_tppp_stage(
            model_path,
            pos,
            my_tp_group,
        )
        vram_mb = torch.cuda.memory_allocated(rank) / 1024**2
        print(
            f"[TP-PP {rank}] Stage loaded: layers[{stage_info.layer_start}:{stage_info.layer_end}] "
            f"VRAM={vram_mb:.0f}MB",
            flush=True,
        )

        model_loaded_barrier.wait()

        if rank == 0:
            output_queue.put({"status": "ready", "rank": rank})

        # ---------- command loop ----------
        while True:
            try:
                cmd = input_queue.get()
            except Exception:
                break
            if cmd is None or cmd.get("action") == "shutdown":
                break
            try:
                if cmd["action"] == "generate":
                    _tppp_generate_loop(
                        pos,
                        stage_model,
                        tokenizer,
                        stage_info,
                        my_tp_group,
                        cmd,
                        output_queue,
                    )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[TP-PP {rank}] ERROR: {e}\n{tb}", flush=True)
                if pos.is_last_stage and pos.is_tp_root:
                    output_queue.put({"error": str(e), "traceback": tb})
                break

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[TP-PP {rank}] FATAL: {e}\n{tb}", flush=True)
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


# -----------------------------------------------------------------------
# Model loading  (PP stage extraction + TP sharding)
# -----------------------------------------------------------------------


def _load_tppp_stage(
    model_path: str,
    pos: GridPosition,
    tp_group: dist.ProcessGroup,
) -> Tuple[Dict[str, Any], Any, StageInfo]:
    """
    1. Load full model on this GPU.
    2. Compute PP stage assignment, extract relevant layers.
    3. Apply TP sharding to the extracted layers.
    4. Free everything else.
    """
    device = pos.device
    model_p = Path(model_path)

    # --- load full model ---
    if model_p.exists() and (
        model_p.suffix == ".zse" or (model_p.is_dir() and (model_p / "model.zse").exists())
    ):
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

    inner = _get_inner_model(full_model)
    layers = _get_layers(inner)
    num_layers = len(layers)

    stages = compute_stage_assignments(num_layers, pos.pp_size)
    stage_info = stages[pos.pp_rank]

    # --- extract PP stage components ---
    stage_model: Dict[str, Any] = {}
    stage_model["config"] = full_model.config

    if stage_info.has_embed:
        stage_model["embed_tokens"] = _get_embedding(inner).to(device)

    # Rotary embedding
    rotary_emb = _get_rotary_emb(inner)
    if rotary_emb is not None:
        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

            stage_model["rotary_emb"] = Qwen2RotaryEmbedding(
                full_model.config,
                device=device,
            )
        except Exception:
            try:
                stage_model["rotary_emb"] = rotary_emb.to(device)
            except Exception:
                pass

    # Extract stage layers into a temporary nn.Module so TP can find them
    stage_layers = nn.ModuleList()
    for i in range(stage_info.layer_start, stage_info.layer_end):
        stage_layers.append(layers[i].to(device))

    if stage_info.has_head:
        stage_model["norm"] = _get_final_norm(inner).to(device)
        stage_model["lm_head"] = full_model.lm_head.to(device)

    # --- apply TP sharding to the extracted layers ---
    from zse.core.zdistributed.tensor_parallel import TensorParallel

    tp = TensorParallel(
        tp_size=pos.tp_size,
        tp_rank=pos.tp_rank,
        tp_group=tp_group,
    )

    # Build a thin wrapper so TP can detect the architecture
    class _StageModule(nn.Module):
        def __init__(self, stage_layers, config, lm_head=None):
            super().__init__()
            # Mimic Qwen2/Llama structure: model.layers + lm_head
            self.model = nn.Module()
            self.model.layers = stage_layers
            self.config = config
            if lm_head is not None:
                self.lm_head = lm_head

    lm_head = stage_model.pop("lm_head", None)
    temp_module = _StageModule(stage_layers, full_model.config, lm_head)
    temp_module = tp.apply(temp_module)

    # Extract TP-sharded layers back
    stage_model["layers"] = temp_module.model.layers
    if lm_head is not None:
        stage_model["lm_head"] = temp_module.lm_head

    # Ensure all params on correct device (layers + lm_head + norm + embed)
    target = torch.device(device)
    meta = torch.device("meta")

    def _ensure_device(module: nn.Module) -> None:
        for p in module.parameters():
            if p.device != target and p.device != meta:
                p.data = p.data.to(device)
        for _, buf in module.named_buffers():
            if buf is not None and buf.device != target and buf.device != meta:
                buf.data = buf.data.to(device)

    for layer in stage_model["layers"]:
        _ensure_device(layer)
    for key in ("lm_head", "norm", "embed_tokens"):
        if key in stage_model and isinstance(stage_model[key], nn.Module):
            _ensure_device(stage_model[key])

    # Free full model
    del full_model, temp_module
    gc.collect()
    torch.cuda.empty_cache()

    return stage_model, tokenizer, stage_info


# -----------------------------------------------------------------------
# Generation loop  (TP-PP aware)
# -----------------------------------------------------------------------


def _tppp_generate_loop(
    pos: GridPosition,
    stage_model: Dict[str, Any],
    tokenizer: Any,
    stage_info: StageInfo,
    tp_group: dist.ProcessGroup,
    cmd: Dict,
    output_queue,
):
    """
    Autoregressive generation across the TP-PP grid.

    Within a stage: TP all-reduce happens inside each layer's forward().
    Between stages:
        • TP-rank 0 of sender stage → send → TP-rank 0 of receiver stage
        • TP-rank 0 of receiver stage → broadcast within its TP group

    Token selection: last-stage TP-rank 0 picks the token, then
    broadcasts to ALL ranks globally.
    """
    device = pos.device
    gen_kwargs = cmd.get("kwargs", {})
    max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
    do_sample = gen_kwargs.get("do_sample", False)
    temperature = gen_kwargs.get("temperature", 1.0)
    top_p = gen_kwargs.get("top_p", 1.0)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    config = stage_model["config"]
    hidden_size = config.hidden_size
    world_size = pos.pp_size * pos.tp_size

    # The rank that selects next tokens: last stage's TP-root
    last_stage_tp0 = (pos.pp_size - 1) * pos.tp_size

    with torch.no_grad():
        # ---- shape info ----
        if pos.is_first_stage and pos.is_tp_root:
            input_ids = cmd["input_ids"].to(device)
            batch_size = input_ids.shape[0]
            prompt_len = input_ids.shape[1]
            shape_t = torch.tensor([batch_size, prompt_len], dtype=torch.long, device=device)
        else:
            shape_t = torch.zeros(2, dtype=torch.long, device=device)

        dist.broadcast(shape_t, src=0)
        batch_size = shape_t[0].item()
        prompt_len = shape_t[1].item()

        # All ranks need input_ids for final output assembly
        if not (pos.is_first_stage and pos.is_tp_root):
            input_ids = torch.zeros(batch_size, prompt_len, dtype=torch.long, device=device)
        # Broadcast input_ids from rank 0 to ALL ranks (world group)
        dist.broadcast(input_ids, src=0)

        # KV cache
        from transformers import DynamicCache

        kv_cache = DynamicCache(config=config)

        # ===== PREFILL =====
        if pos.is_first_stage:
            hidden = stage_model["embed_tokens"](input_ids)
        else:
            hidden = torch.empty(
                batch_size, prompt_len, hidden_size, dtype=torch.float16, device=device
            )
            # TP-rank 0 receives from previous stage, then broadcasts within TP group
            if pos.is_tp_root:
                dist.recv(hidden, src=pos.prev_stage_tp_root)
            dist.broadcast(hidden, src=pos.pp_rank * pos.tp_size, group=tp_group)

        if pos.is_tp_root:
            print(
                f"[TP-PP {pos.global_rank}] prefill: embed/recv done, running {len(stage_model['layers'])} layers",
                flush=True,
            )

        cache_position = torch.arange(prompt_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        pos_embs = _compute_position_embeddings(stage_model, hidden, position_ids, device)

        for layer in stage_model["layers"]:
            layer_out = layer(
                hidden,
                position_embeddings=pos_embs,
                past_key_values=kv_cache,
                use_cache=True,
                cache_position=cache_position,
            )
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        if pos.is_last_stage:
            normed = stage_model["norm"](hidden)
            logits = stage_model["lm_head"](normed)
            next_logits = logits[:, -1, :]
            if pos.is_tp_root:
                print(
                    f"[TP-PP {pos.global_rank}] prefill done, first token logits ready", flush=True
                )
        else:
            # TP-rank 0 sends hidden to next stage
            if pos.is_tp_root:
                dist.send(hidden.contiguous(), dst=pos.next_stage_tp_root)
                print(
                    f"[TP-PP {pos.global_rank}] prefill done, hidden sent to stage {pos.pp_rank + 1}",
                    flush=True,
                )

        # ===== DECODE =====
        generated_tokens: List[torch.Tensor] = []

        for step in range(max_new_tokens):
            # --- token selection (last stage, tp-rank 0) ---
            if pos.is_last_stage and pos.is_tp_root:
                if do_sample and temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_p, sorted_i = torch.sort(probs, descending=True)
                        cum = torch.cumsum(sorted_p, dim=-1)
                        mask = cum - sorted_p > top_p
                        sorted_p[mask] = 0.0
                        sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
                        next_token = sorted_i.gather(-1, torch.multinomial(sorted_p, 1))
                    else:
                        next_token = torch.multinomial(probs, 1)
                else:
                    next_token = next_logits.argmax(-1, keepdim=True)

                # Check EOS
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    stop = torch.tensor([[-1]], dtype=torch.long, device=device).expand(
                        batch_size, 1
                    )
                    dist.broadcast(stop, src=last_stage_tp0)
                    generated_tokens.append(next_token)
                    break

                dist.broadcast(next_token, src=last_stage_tp0)
                generated_tokens.append(next_token)
                if step % 10 == 0:
                    print(
                        f"[TP-PP {pos.global_rank}] decode step {step}/{max_new_tokens}", flush=True
                    )
            else:
                next_token = torch.empty(batch_size, 1, dtype=torch.long, device=device)
                dist.broadcast(next_token, src=last_stage_tp0)
                if next_token[0, 0].item() == -1:
                    break

            # --- one decode step through the pipeline ---
            cur_pos = prompt_len + step
            cache_pos_step = torch.tensor([cur_pos], device=device)
            pos_ids_step = cache_pos_step.unsqueeze(0).expand(batch_size, -1)

            if pos.is_first_stage:
                hidden = stage_model["embed_tokens"](next_token)
            else:
                hidden = torch.empty(batch_size, 1, hidden_size, dtype=torch.float16, device=device)
                if pos.is_tp_root:
                    dist.recv(hidden, src=pos.prev_stage_tp_root)
                dist.broadcast(hidden, src=pos.pp_rank * pos.tp_size, group=tp_group)

            pos_emb = _compute_position_embeddings(stage_model, hidden, pos_ids_step, device)

            for layer in stage_model["layers"]:
                layer_out = layer(
                    hidden,
                    position_embeddings=pos_emb,
                    past_key_values=kv_cache,
                    use_cache=True,
                    cache_position=cache_pos_step,
                )
                hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            if pos.is_last_stage:
                normed = stage_model["norm"](hidden)
                logits = stage_model["lm_head"](normed)
                next_logits = logits[:, -1, :]
            else:
                if pos.is_tp_root:
                    dist.send(hidden.contiguous(), dst=pos.next_stage_tp_root)

        # ===== OUTPUT =====
        # Only last-stage TP-root assembles and returns
        if pos.is_last_stage and pos.is_tp_root:
            gen_ids = torch.cat(generated_tokens, dim=-1)
            output_ids = torch.cat([input_ids, gen_ids], dim=-1)
            output_queue.put({"output_ids": output_ids.cpu()})


# -----------------------------------------------------------------------
# Coordinator  (main process)
# -----------------------------------------------------------------------


class TPPPCoordinator:
    """
    Coordinator for combined TP + PP parallelism.

    Spawns  tp_size × pp_size  worker processes in a 2-D grid.
    Provides the same interface as TPCoordinator / PPCoordinator.
    """

    def __init__(
        self,
        model_path: str,
        tp_size: int = 2,
        pp_size: int = 2,
        backend: str = "nccl",
        master_addr: str = "127.0.0.1",
        master_port: str = "29700",
    ):
        self.model_path = model_path
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.world_size = tp_size * pp_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port

        self._processes: List[mp.Process] = []
        self._input_queues: List[mp.Queue] = []
        self._output_queue: Optional[mp.Queue] = None
        self._started = False

    def start(self, verbose: bool = True) -> None:
        if self._started:
            return

        mp.set_start_method("spawn", force=True)
        barrier = mp.Barrier(self.world_size)
        self._output_queue = mp.Queue()

        if verbose:
            print(
                f"🔀🔗 TP-PP: spawning {self.world_size} workers "
                f"({self.pp_size} stages × {self.tp_size} TP)...",
                flush=True,
            )

        for rank in range(self.world_size):
            q = mp.Queue()
            self._input_queues.append(q)
            p = mp.Process(
                target=_tppp_worker_main,
                args=(
                    rank,
                    self.world_size,
                    self.tp_size,
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
            print(f"   Waiting for {self.world_size} workers to load...", flush=True)

        result = self._output_queue.get(timeout=900)
        if result.get("status") == "error":
            self.shutdown()
            raise RuntimeError(
                f"TP-PP worker {result.get('rank')} failed:\n"
                f"{result.get('error')}\n{result.get('traceback', '')}"
            )
        assert result["status"] == "ready"
        self._started = True
        if verbose:
            print(f"   ✅ All {self.world_size} workers ready", flush=True)

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
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
                f"TP-PP generate error:\n{result['error']}\n{result.get('traceback', '')}"
            )
        return result["output_ids"]

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """Single forward pass (returns logits). Not used for generation."""
        assert self._started
        cmd = {
            "action": "forward",
            "input_ids": input_ids.cpu(),
            "kwargs": kwargs,
        }
        for q in self._input_queues:
            q.put(cmd)
        result = self._output_queue.get(timeout=120)
        if "error" in result:
            raise RuntimeError(f"TP-PP forward error:\n{result['error']}")
        return result

    def shutdown(self) -> None:
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
