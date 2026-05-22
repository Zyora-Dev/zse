"""ZSE VRAM Allocator — Unified GPU memory budget manager.

Unlike vLLM's two-tier uncoordinated model (KV slab + PyTorch arena),
we manage a single VRAM budget across:
1. Model weights (loaded from .zse, static)
2. KV cache (dynamic, grows/shrinks with sequences)
3. Scratch buffers (activations, reused every forward pass)

This ensures zero memory fragmentation and accurate capacity planning.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List

from zse_compiler.types.tensor import Tensor
from zse_compiler.types.dtypes import float16, int32


@dataclass
class VRAMPlan:
    """VRAM allocation plan computed before loading."""
    total_vram: int          # Total GPU VRAM in bytes
    available_vram: int      # After CUDA context overhead
    weight_bytes: int        # Model weights (static)
    kv_cache_bytes: int      # KV cache budget
    scratch_bytes: int       # Activation scratch buffers
    reserved_bytes: int      # Safety margin (5%)
    max_batch_tokens: int    # Max tokens KV cache can hold
    utilization_pct: float   # % of VRAM planned to use

    def summary(self) -> str:
        def _mb(b): return f"{b / 1024**2:.1f}MB"
        def _gb(b): return f"{b / 1024**3:.2f}GB"
        return (
            f"VRAM Plan:\n"
            f"  Total VRAM:    {_gb(self.total_vram)}\n"
            f"  Available:     {_gb(self.available_vram)}\n"
            f"  Weights:       {_mb(self.weight_bytes)}\n"
            f"  KV Cache:      {_mb(self.kv_cache_bytes)}\n"
            f"  Scratch:       {_mb(self.scratch_bytes)}\n"
            f"  Reserved:      {_mb(self.reserved_bytes)}\n"
            f"  Max KV tokens: {self.max_batch_tokens:,}\n"
            f"  Utilization:   {self.utilization_pct:.1f}%"
        )


@dataclass
class ScratchBuffers:
    """Pre-allocated GPU buffers for intermediate activations.

    Allocated once at startup, reused every forward pass.
    Zero allocation during inference — unlike PyTorch.
    """
    # Main hidden state buffer: [max_seq_len, hidden_size] fp16
    hidden: Optional[Tensor] = None
    # Residual connection buffer
    residual: Optional[Tensor] = None
    # QKV projection output: [max_seq_len, (num_heads + 2*num_kv_heads) * head_dim] fp16
    qkv: Optional[Tensor] = None
    # Attention output: [max_seq_len, hidden_size] fp16
    attn_out: Optional[Tensor] = None
    # MLP intermediate: [max_seq_len, intermediate_size] fp16
    mlp_gate: Optional[Tensor] = None
    mlp_up: Optional[Tensor] = None
    mlp_out: Optional[Tensor] = None
    # Logits: [max_seq_len, vocab_size] fp16
    logits: Optional[Tensor] = None
    # RMSNorm scratch: [max_seq_len, hidden_size] fp16
    norm_out: Optional[Tensor] = None
    # FP32 residual stream buffers (prevents fp16 overflow across layers)
    hidden_f32: Optional[Tensor] = None
    residual_f32: Optional[Tensor] = None
    # Total bytes used
    total_bytes: int = 0

    def destroy(self, gpu_mem):
        """Free all scratch buffers."""
        for name in ['hidden', 'residual', 'qkv', 'attn_out',
                     'mlp_gate', 'mlp_up', 'mlp_out', 'logits', 'norm_out',
                     'hidden_f32', 'residual_f32']:
            tensor = getattr(self, name)
            if tensor is not None and tensor.data_ptr != 0:
                gpu_mem.free(tensor)
                setattr(self, name, None)


class VRAMAllocator:
    """Unified VRAM budget manager.

    Usage:
        alloc = VRAMAllocator(gpu_mem, device_info)
        plan = alloc.plan_allocation(model_size, config)
        scratch = alloc.allocate_scratch(config, max_seq_len=2048)
    """

    # CUDA context overhead (conservative estimate)
    CUDA_CONTEXT_BYTES = 400 * 1024 * 1024  # 400MB
    # Safety margin
    SAFETY_MARGIN_PCT = 0.05

    def __init__(self, gpu_mem=None, device_info=None):
        self._gpu_mem = gpu_mem
        self._device = device_info
        self._allocated_bytes = 0
        self._weight_bytes = 0

    def plan_allocation(
        self,
        model_size_bytes: int,
        config,
        max_seq_len: int = 2048,
        max_batch_seqs: int = 8,
        kv_headroom_ratio: float = 1.5,
    ) -> VRAMPlan:
        """Plan VRAM allocation before loading anything.

        Memory budget strategy:
        - Weights: fixed, known from model size
        - Scratch: fixed, computed from config + max_seq_len
        - KV cache: sized to actual workload (max_batch_seqs * max_seq_len) with
          a small headroom multiplier — NOT all remaining VRAM
        - Reserved: 5% safety margin + CUDA context + whatever remains free

        Args:
            model_size_bytes: Estimated model weight size in GPU
            config: ModelConfig for computing scratch/KV sizes
            max_seq_len: Max sequence length for scratch buffer sizing
            max_batch_seqs: Max concurrent sequences (drives KV demand)
            kv_headroom_ratio: Multiplier on demand-based KV size (default 1.5x)
        """
        if self._gpu_mem is not None:
            total_vram = self._gpu_mem.get_total_memory()
        elif self._device is not None:
            total_vram = self._device.vram_total_bytes
        else:
            total_vram = 16 * 1024**3

        # Subtract CUDA context
        available = total_vram - self.CUDA_CONTEXT_BYTES

        # Safety margin
        reserved = int(available * self.SAFETY_MARGIN_PCT)
        usable = available - reserved

        # Weight budget (known, static)
        weight_bytes = model_size_bytes

        # Scratch budget
        scratch_bytes = self._estimate_scratch_bytes(config, max_seq_len)

        # KV cache: size from ACTUAL workload, not "all remaining VRAM"
        # demand = max_batch_seqs * max_seq_len * bytes_per_token * headroom
        remaining = usable - weight_bytes - scratch_bytes
        remaining = max(remaining, 0)

        bytes_per_token = max(1, config.total_kv_cache_bytes_per_token)
        demand_tokens = max(1, max_batch_seqs) * max(1, max_seq_len)
        demand_bytes = int(demand_tokens * bytes_per_token * max(1.0, kv_headroom_ratio))

        # Floor: at least 256 MB so tiny configs still work
        # Ceiling: never exceed what's remaining after weights+scratch
        kv_bytes = max(256 * 1024 * 1024, demand_bytes)
        kv_bytes = min(kv_bytes, remaining)

        # If we have a GPU memory handle, double-check against actual free memory
        if self._gpu_mem is not None:
            try:
                actual_free = self._gpu_mem.get_free_memory()
                max_from_free = max(0, actual_free - weight_bytes - scratch_bytes - 1 * 1024**3)
                kv_bytes = min(kv_bytes, max_from_free)
            except Exception:
                pass

        # Update reserved to include the unused remainder
        actual_used = weight_bytes + scratch_bytes + kv_bytes
        reserved = usable - actual_used + reserved

        # Max tokens the KV cache can hold
        if config.total_kv_cache_bytes_per_token > 0:
            max_tokens = kv_bytes // config.total_kv_cache_bytes_per_token
        else:
            max_tokens = 0

        utilization = actual_used / total_vram * 100

        return VRAMPlan(
            total_vram=total_vram,
            available_vram=available,
            weight_bytes=weight_bytes,
            kv_cache_bytes=kv_bytes,
            scratch_bytes=scratch_bytes,
            reserved_bytes=reserved,
            max_batch_tokens=max_tokens,
            utilization_pct=utilization,
        )

    def _estimate_scratch_bytes(self, config, max_seq_len: int = 2048) -> int:
        """Estimate scratch buffer sizes."""
        h = config.hidden_size
        inter = config.intermediate_size
        vocab = config.vocab_size
        n_heads = config.num_heads
        n_kv = config.num_kv_heads
        d = config.head_dim
        S = max_seq_len

        # All in fp16 (2 bytes)
        total = 0
        total += S * h * 2         # hidden
        total += S * h * 2         # residual
        total += S * (n_heads + 2 * n_kv) * d * 2  # qkv
        total += S * h * 2         # attn_out
        total += S * inter * 2     # mlp_gate
        total += S * inter * 2     # mlp_up
        total += S * h * 2         # mlp_out
        # Logits: during decode we only sample 1 row per sequence.
        # During prefill we only need the LAST token's logits.
        # Size for max_batch_seqs rows, not full seq_len.
        logits_rows = min(S, 64)   # 64 = typical max_batch_seqs
        total += logits_rows * vocab * 2  # logits
        total += S * h * 2         # norm_out
        total += S * h * 4         # hidden_f32
        total += S * h * 4         # residual_f32
        return total

    def allocate_scratch(self, config, max_seq_len: int = 2048) -> ScratchBuffers:
        """Allocate all scratch buffers on GPU.

        Args:
            config: ModelConfig
            max_seq_len: Max tokens in a single forward pass (prefill batch size)
        """
        h = config.hidden_size
        inter = config.intermediate_size
        vocab = config.vocab_size
        n_heads = config.num_heads
        n_kv = config.num_kv_heads
        d = config.head_dim
        S = max_seq_len

        # Logits only need rows for decode batch, not full seq_len.
        # During prefill we only need the last token's logits row.
        logits_rows = min(S, 64)

        scratch = ScratchBuffers()

        if self._gpu_mem is not None:
            scratch.hidden = self._gpu_mem.allocate((S, h), float16)
            scratch.residual = self._gpu_mem.allocate((S, h), float16)
            scratch.qkv = self._gpu_mem.allocate((S, (n_heads + 2 * n_kv) * d), float16)
            scratch.attn_out = self._gpu_mem.allocate((S, h), float16)
            scratch.mlp_gate = self._gpu_mem.allocate((S, inter), float16)
            scratch.mlp_up = self._gpu_mem.allocate((S, inter), float16)
            scratch.mlp_out = self._gpu_mem.allocate((S, h), float16)
            scratch.logits = self._gpu_mem.allocate((logits_rows, vocab), float16)
            scratch.norm_out = self._gpu_mem.allocate((S, h), float16)
            # FP32 residual stream (prevents fp16 overflow across transformer layers)
            from zse_compiler.types.dtypes import float32
            scratch.hidden_f32 = self._gpu_mem.allocate((S, h), float32)
            scratch.residual_f32 = self._gpu_mem.allocate((S, h), float32)

        scratch.total_bytes = self._estimate_scratch_bytes(config, max_seq_len)
        return scratch

    def track_weight_upload(self, nbytes: int):
        """Track weight bytes uploaded to GPU."""
        self._weight_bytes += nbytes
        self._allocated_bytes += nbytes

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    @property
    def weight_bytes(self) -> int:
        return self._weight_bytes

    def utilization(self) -> dict:
        """Current VRAM utilization stats."""
        if self._gpu_mem is not None:
            total = self._gpu_mem.get_total_memory()
            free = self._gpu_mem.get_free_memory()
            used = total - free
        else:
            total = 0
            used = self._allocated_bytes
            free = 0

        return {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "weight_bytes": self._weight_bytes,
            "utilization_pct": (used / total * 100) if total > 0 else 0,
        }
