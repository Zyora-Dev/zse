"""ZSE LoRA Manager — Adapter lifecycle and GPU execution.

Manages loading, unloading, and applying LoRA adapters during inference.

LoRA math:
    out = W @ x + (α/r) * B @ (A @ x)

Where:
    W: base weight [N, K] (quantized or fp16)
    A: LoRA down-projection [r, K] (fp16)
    B: LoRA up-projection [N, r] (fp16)
    α: scaling factor, r: rank
    The two small matmuls add negligible latency (~1-3% overhead)

Usage:
    manager = LoRAManager(gpu_mem, kernels)
    manager.load_adapter("customer-a", adapter_data)
    # During inference:
    manager.apply_lora(out, inp, adapter, layer=0, module="q_proj", M=1, N=4096, K=4096)
"""

import struct
import threading
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

from zse_compiler.types.tensor import Tensor
from zse_compiler.types.dtypes import float16

from zse_engine.orchestrator.lora_weights import (
    LoRAWeight, LoRAAdapter, LoRAWeightStore,
)
from zse_engine.orchestrator.kernels import InferenceKernels
from zse_engine.format.lora_format import load_lora, estimate_lora_size


class LoRAManager:
    """Manages LoRA adapter lifecycle and applies adapters during inference.

    Thread-safe for concurrent adapter lookups during batched inference.
    Load/unload operations should be serialized externally.

    Args:
        gpu_mem: GPUMemory instance for allocating adapter weights
        kernels: InferenceKernels for launching matmul kernels
    """

    def __init__(self, gpu_mem, kernels: InferenceKernels):
        self._gpu_mem = gpu_mem
        self._kernels = kernels
        self._store = LoRAWeightStore()
        self._lock = threading.Lock()  # Protects store mutations and scratch buffers

        # Scratch buffer for LoRA intermediate: A @ x → [rank, M]
        # Allocated lazily, reused across calls
        self._lora_scratch_ptr: int = 0
        self._lora_scratch_bytes: int = 0

        # Scratch buffer for LoRA output: B @ scratch → [M, N]
        # Allocated lazily, reused across calls (avoids per-call alloc/free)
        self._lora_out_ptr: int = 0
        self._lora_out_bytes: int = 0

    def load_adapter_from_file(
        self,
        adapter_id: str,
        path: str,
        weight_shapes: Dict[str, Tuple[int, int]],
    ) -> LoRAAdapter:
        """Load a LoRA adapter from a .zse-lora file and upload to GPU.

        Args:
            adapter_id: Unique identifier for this adapter
            path: Path to .zse-lora file
            weight_shapes: module_name → (in_features, out_features)

        Returns:
            Loaded LoRAAdapter with weights on GPU
        """
        loaded_adapter, weight_data = load_lora(path, weight_shapes)

        # Build weight_shapes dict keyed by (layer, module) for load_adapter_from_dict
        per_layer_shapes = {}
        for (layer_idx, module_name) in weight_data:
            if module_name in weight_shapes:
                per_layer_shapes[(layer_idx, module_name)] = weight_shapes[module_name]

        return self.load_adapter_from_dict(
            adapter_id=adapter_id,
            rank=loaded_adapter.rank,
            alpha=loaded_adapter.alpha,
            num_layers=loaded_adapter.num_layers,
            target_modules=loaded_adapter.target_modules,
            weights_data=weight_data,
            weight_shapes=per_layer_shapes,
        )

    def load_adapter_from_dict(
        self,
        adapter_id: str,
        rank: int,
        alpha: float,
        num_layers: int,
        target_modules: List[str],
        weights_data: Dict[Tuple[int, str], Tuple[bytes, bytes]],
        weight_shapes: Dict[Tuple[int, str], Tuple[int, int]],
    ) -> LoRAAdapter:
        """Load a LoRA adapter from in-memory weight data.

        Args:
            adapter_id: Unique identifier for this adapter
            rank: LoRA rank (r)
            alpha: LoRA scaling factor
            num_layers: Number of transformer layers
            target_modules: Which modules have LoRA (e.g., ["q_proj", "v_proj"])
            weights_data: (layer_idx, module_name) → (a_bytes, b_bytes) in fp16
            weight_shapes: (layer_idx, module_name) → (in_features, out_features)

        Returns:
            Loaded LoRAAdapter with weights on GPU
        """
        adapter = LoRAAdapter(
            adapter_id=adapter_id,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            num_layers=num_layers,
        )

        for (layer_idx, module_name), (a_bytes, b_bytes) in weights_data.items():
            in_features, out_features = weight_shapes[(layer_idx, module_name)]

            # Upload A [rank, in_features] to GPU
            a_tensor = self._gpu_mem.allocate((rank, in_features), float16)
            self._gpu_mem.copy_host_to_device(a_bytes, a_tensor)

            # Upload B [out_features, rank] to GPU
            b_tensor = self._gpu_mem.allocate((out_features, rank), float16)
            self._gpu_mem.copy_host_to_device(b_bytes, b_tensor)

            weight = LoRAWeight(
                layer_name=f"model.layers.{layer_idx}.{module_name}",
                rank=rank,
                in_features=in_features,
                out_features=out_features,
                a_ptr=a_tensor.data_ptr,
                a_nbytes=rank * in_features * 2,
                b_ptr=b_tensor.data_ptr,
                b_nbytes=out_features * rank * 2,
            )
            adapter.add_weight(layer_idx, module_name, weight)

        with self._lock:
            self._store.add(adapter)
        return adapter

    def load_adapter_random(
        self,
        adapter_id: str,
        rank: int,
        alpha: float,
        num_layers: int,
        target_modules: List[str],
        hidden_size: int,
        intermediate_size: int = 0,
        num_heads: int = 32,
        num_kv_heads: int = 32,
        head_dim: int = 128,
    ) -> LoRAAdapter:
        """Load a LoRA adapter with random weights (for testing).

        Creates appropriately-shaped A and B matrices for each target module.
        """
        adapter = LoRAAdapter(
            adapter_id=adapter_id,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            num_layers=num_layers,
        )

        # Determine shapes for each target module
        module_shapes = {
            "q_proj": (hidden_size, num_heads * head_dim),
            "k_proj": (hidden_size, num_kv_heads * head_dim),
            "v_proj": (hidden_size, num_kv_heads * head_dim),
            "o_proj": (num_heads * head_dim, hidden_size),
        }
        if intermediate_size > 0:
            module_shapes.update({
                "gate_proj": (hidden_size, intermediate_size),
                "up_proj": (hidden_size, intermediate_size),
                "down_proj": (intermediate_size, hidden_size),
            })

        for layer_idx in range(num_layers):
            for module_name in target_modules:
                if module_name not in module_shapes:
                    continue
                in_features, out_features = module_shapes[module_name]

                # Create zero-initialized A and B (fp16)
                a_bytes = b'\x00' * (rank * in_features * 2)
                b_bytes = b'\x00' * (out_features * rank * 2)

                # For testing: just store the shape info without GPU alloc
                weight = LoRAWeight(
                    layer_name=f"model.layers.{layer_idx}.{module_name}",
                    rank=rank,
                    in_features=in_features,
                    out_features=out_features,
                    a_ptr=0,  # No GPU in test mode
                    a_nbytes=rank * in_features * 2,
                    b_ptr=0,
                    b_nbytes=out_features * rank * 2,
                )
                adapter.add_weight(layer_idx, module_name, weight)

        with self._lock:
            self._store.add(adapter)
        return adapter

    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload a LoRA adapter and free its GPU memory.

        Returns True if the adapter was found and unloaded.
        """
        with self._lock:
            adapter = self._store.remove(adapter_id)
        if adapter is None:
            return False

        # Free GPU memory for all weight pairs
        for weight in adapter.weights.values():
            if weight.a_ptr:
                t = Tensor(shape=(1,), dtype=float16)
                t._data_ptr = weight.a_ptr
                t._nbytes = weight.a_nbytes
                self._gpu_mem.free(t)
            if weight.b_ptr:
                t = Tensor(shape=(1,), dtype=float16)
                t._data_ptr = weight.b_ptr
                t._nbytes = weight.b_nbytes
                self._gpu_mem.free(t)

        return True

    def get_adapter(self, adapter_id: str) -> Optional[LoRAAdapter]:
        """Get a loaded adapter by ID."""
        return self._store.get(adapter_id)

    def has_adapter(self, adapter_id: str) -> bool:
        return self._store.has(adapter_id)

    def apply_lora(
        self,
        out: Tensor,
        inp: Tensor,
        adapter: LoRAAdapter,
        layer_idx: int,
        module_name: str,
        M: int,
        N: int,
        K: int,
    ):
        """Apply LoRA delta to an existing matmul output.

        Computes: out += (α/r) * B @ (A @ x)

        Where:
            out: [M, N] — already contains W @ x (base result)
            inp: [M, K] — input activations
            A: [r, K], B: [N, r] — LoRA weights

        Uses two small fp16 matmuls:
            1. tmp = A @ x^T → [r, M] (or equivalently x @ A^T → [M, r])
            2. out += scaling * B @ tmp → adds [M, N] to out

        Args:
            out: Output tensor [M, N] — modified in-place (base + LoRA delta)
            inp: Input tensor [M, K]
            adapter: LoRA adapter
            layer_idx: Transformer layer index
            module_name: Target module (e.g., "q_proj")
            M: Batch dimension
            N: Output dimension
            K: Input dimension
        """
        weight = adapter.get_weight(layer_idx, module_name)
        if weight is None:
            return  # No LoRA for this module

        r = weight.rank
        scaling = adapter.scaling

        # Ensure scratch buffer for intermediate result [M, r]
        scratch_bytes = M * r * 2  # fp16
        if scratch_bytes > self._lora_scratch_bytes:
            if self._lora_scratch_ptr:
                old = Tensor(shape=(1,), dtype=float16)
                old._data_ptr = self._lora_scratch_ptr
                old._nbytes = self._lora_scratch_bytes
                self._gpu_mem.free(old)
            scratch = self._gpu_mem.allocate((M * r,), float16)
            self._lora_scratch_ptr = scratch.data_ptr
            self._lora_scratch_bytes = scratch_bytes

        # Create tensor wrappers
        a_tensor = weight.make_a_tensor()    # [r, K]
        b_tensor = weight.make_b_tensor()    # [N, r]
        scratch_tensor = Tensor(shape=(M, r), dtype=float16)
        scratch_tensor._data_ptr = self._lora_scratch_ptr
        scratch_tensor._nbytes = scratch_bytes

        # Step 1: scratch = inp @ A^T → [M, r]
        # A is [r, K], A^T is [K, r]. We want inp [M, K] @ A^T [K, r] = [M, r]
        # Our tiled_fp16_matmul does: out = A @ B^T where B is [N, K] row-major
        # So: scratch = inp @ A^T ↔ tiled_fp16_matmul(scratch, inp, A, M, r, K)
        # because A is [r, K] (row-major), and the kernel does A[m,k] * B[n,k]
        self._kernels.launch(
            "tiled_fp16_matmul",
            ((r + 31) // 32, (M + 31) // 32),
            (32, 32),
            scratch_tensor, inp, a_tensor, M, r, K,
        )

        # Step 2: lora_out = scratch @ B^T → [M, N]
        # B is [N, r], scratch is [M, r]
        # Ensure reusable lora_out buffer for [M, N]
        lora_out_bytes = M * N * 2  # fp16
        if lora_out_bytes > self._lora_out_bytes:
            if self._lora_out_ptr:
                old = Tensor(shape=(1,), dtype=float16)
                old._data_ptr = self._lora_out_ptr
                old._nbytes = self._lora_out_bytes
                self._gpu_mem.free(old)
            lora_out_tensor = self._gpu_mem.allocate((M * N,), float16)
            self._lora_out_ptr = lora_out_tensor.data_ptr
            self._lora_out_bytes = lora_out_bytes

        lora_out = Tensor(shape=(M, N), dtype=float16)
        lora_out._data_ptr = self._lora_out_ptr
        lora_out._nbytes = lora_out_bytes

        self._kernels.launch(
            "tiled_fp16_matmul",
            ((N + 31) // 32, (M + 31) // 32),
            (32, 32),
            lora_out, scratch_tensor, b_tensor, M, N, r,
        )

        # Step 3: out += scaling * lora_out
        # Use scaled_add kernel (or fuse into residual_add with scaling)
        # For now: launch the lora_scaled_add kernel
        total = M * N
        self._kernels.launch(
            "lora_scaled_add",
            ((total + 255) // 256,),
            (256,),
            out, lora_out, total, scaling,
        )

    @property
    def num_adapters(self) -> int:
        return self._store.num_adapters

    def list_adapters(self) -> List[str]:
        return self._store.list_adapters()

    def stats(self) -> dict:
        return {
            "num_adapters": self._store.num_adapters,
            "total_gpu_bytes": self._store.total_gpu_bytes,
            "adapter_ids": self._store.list_adapters(),
        }

    def summary(self) -> str:
        return self._store.summary()

    def destroy(self):
        """Free all GPU memory (adapters + scratch buffers)."""
        for adapter_id in list(self._store.list_adapters()):
            self.unload_adapter(adapter_id)
        for attr, bytes_attr in [
            ('_lora_scratch_ptr', '_lora_scratch_bytes'),
            ('_lora_out_ptr', '_lora_out_bytes'),
        ]:
            ptr = getattr(self, attr, 0)
            nbytes = getattr(self, bytes_attr, 0)
            if ptr:
                t = Tensor(shape=(1,), dtype=float16)
                t._data_ptr = ptr
                t._nbytes = nbytes
                self._gpu_mem.free(t)
                setattr(self, attr, 0)
                setattr(self, bytes_attr, 0)
