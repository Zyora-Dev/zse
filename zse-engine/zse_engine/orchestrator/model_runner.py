"""ZSE Model Runner — Transformer forward pass wiring.

Connects inference kernels + weights + KV cache into the complete
Llama/Mistral/Qwen transformer architecture:

    Embedding → [RMSNorm → Attention → Residual → RMSNorm → MLP → Residual] × L → RMSNorm → LM Head

Supports both:
- Prefill: Process full prompt at once (batched matmul, write KV to cache)
- Decode: Process single token (read KV from paged cache via PagedAttention)
"""

import struct
import math
from typing import List, Optional, Tuple

from zse_compiler.types.tensor import Tensor
from zse_compiler.types.dtypes import float16, int32

from zse_engine.format.config import ModelConfig
from zse_engine.cache.cache_manager import KVCacheManager
from zse_engine.orchestrator.kernels import InferenceKernels
from zse_engine.orchestrator.weight_loader import WeightStore, GPUWeight
from zse_engine.orchestrator.vram_allocator import ScratchBuffers


# ---------------------------------------------------------------------------
# fp16 <-> Python float conversion (zero-dep, used by embed_pooled)
# ---------------------------------------------------------------------------

def _fp16_bytes_to_floats(data: bytes, n: int) -> List[float]:
    """Decode n fp16 elements from bytes to Python floats."""
    # struct lacks native fp16 — bit-unpack via 'e' format (added in Py 3.6)
    return list(struct.unpack(f'<{n}e', data[:n * 2]))


def _floats_to_fp16_bytes(floats: List[float]) -> bytes:
    """Encode Python floats as fp16 bytes."""
    return struct.pack(f'<{len(floats)}e', *floats)


# Avoid circular import — LoRAManager is optional
try:
    from zse_engine.orchestrator.lora_manager import LoRAManager
    from zse_engine.orchestrator.lora_weights import LoRAAdapter
except ImportError:
    LoRAManager = None
    LoRAAdapter = None


class ModelRunner:
    """Runs the transformer forward pass on GPU.

    Usage:
        runner = ModelRunner(config, weights, kv_cache, scratch, gpu_mem, kernels)
        logits = runner.prefill(token_ids=[1, 2, 3], seq_id=0)
        logits = runner.decode_step(token_id=4, seq_id=0, position=3)
    """

    def __init__(
        self,
        config: ModelConfig,
        weights: WeightStore,
        kv_cache: KVCacheManager,
        scratch: ScratchBuffers,
        gpu_mem,
        kernels: InferenceKernels,
        lora_manager=None,  # Optional LoRAManager for adapter serving
    ):
        self._config = config
        self._weights = weights
        self._kv_cache = kv_cache
        self._scratch = scratch
        self._lora_manager = lora_manager
        self._gpu_mem = gpu_mem
        self._kernels = kernels
        self._backend = getattr(kernels, '_backend', 'cuda')

        # Precompute constants
        self._scale = 1.0 / math.sqrt(config.head_dim)
        self._num_heads = config.num_heads
        self._num_kv_heads = config.num_kv_heads
        self._head_dim = config.head_dim
        self._hidden_size = config.hidden_size
        self._intermediate_size = config.intermediate_size
        self._num_layers = config.num_layers
        self._rope_theta = config.rope_theta
        self._rms_eps = config.rms_norm_eps

        # Pre-build per-layer weight index to avoid f-string + dict lookup per call
        self._layer_weights = self._build_weight_index()

        # Pre-cache decode tensor wrappers (eliminates 2880 allocs/token)
        self._decode_tensor_cache = self._build_decode_tensor_cache()

        # Pre-allocate decode-step buffers to avoid cuMemAlloc/Free per step
        from zse_compiler.types.dtypes import int32 as dt_int32
        max_batch = 64
        self._decode_token_buf = gpu_mem.allocate((max_batch,), dt_int32)
        max_blocks = 256
        self._decode_bt_buf = gpu_mem.allocate((max_blocks * max_batch,), dt_int32)
        self._decode_seqlens_buf = gpu_mem.allocate((max_batch,), dt_int32)
        self._decode_pos_buf = gpu_mem.allocate((max_batch,), dt_int32)
        # Prefill always uses position_offset=0, pre-upload it
        self._prefill_pos_zero_buf = gpu_mem.allocate((1,), dt_int32)
        gpu_mem.copy_host_to_device(struct.pack('<1i', 0), self._prefill_pos_zero_buf)
        self._kv_slab = self._get_kv_slab_tensor()

        # GPU Graph state for graph-captured decode
        self._graph_runner = None
        self._graph_captured = False
        self._graph_stream = None
        self._graph_max_shared_mem = 2048 * 4  # max_seq_len * sizeof(float) for paged_attn
        self._graph_argmax_buf = None  # Separate argmax output for graph (can't reuse token_buf)

        # Fast C decoder — DISABLED (kernel signatures changed for GPU Graph support:
        # rotary_embedding and kv_cache_write now take position as GPU buffer pointer)
        self._fast_decoder = None

        # WMMA tiled weight cache — repacked for tensor core coalesced access
        self._wmma_tiled_weights = {}  # weight_name → (tiled_ptr, num_k_tiles)
        self._wmma_tiled_by_ptr = {}   # weight.data_ptr → (tiled_ptr, num_k_tiles)
        self._wmma_available = False  # WMMA needs store optimization — GEMV is faster for now
        # if self._wmma_available:
        #     self._repack_weights_for_wmma()

    def _repack_weights_for_wmma(self):
        """Repack all INT4 weight matrices into tiled format for WMMA coalesced access.

        Layout: [N, K/2] row-major → [num_n_tiles, num_k_tiles, 64, 8]
        Each tile is 64 rows × 8 bytes = 512 bytes, loaded by 128 threads coalesced.
        Run once at model init time.
        """
        import time
        t0 = time.monotonic()
        count = 0

        # Find all INT4 weight matrices
        proj_names = []
        for layer in range(self._num_layers):
            for proj in ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                         "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                         "mlp.gate_proj.weight", "mlp.up_proj.weight",
                         "mlp.down_proj.weight"]:
                full_name = f"model.layers.{layer}.{proj}"
                proj_names.append((full_name, layer, proj))

        # Also repack lm_head
        proj_names.append(("lm_head.weight", -1, "lm_head.weight"))

        for name, layer_idx, short_name in proj_names:
            # Use _get_layer_weight for layer weights, direct get for others
            try:
                if layer_idx >= 0:
                    weight = self._get_layer_weight(name)
                else:
                    weight = self._weights.get(name)
            except (KeyError, IndexError):
                continue
            if weight is None or weight.dtype != "int4":
                continue

            N = weight.shape[0]  # output dim
            K = weight.shape[1]  # input dim (full, not packed)
            half_K = K // 2
            num_k_tiles = (K + 15) // 16
            num_n_tiles = (N + 63) // 64
            tiled_size = num_n_tiles * num_k_tiles * 512

            # Allocate tiled buffer
            from zse_compiler.types.dtypes import int32 as dt_int32
            # Allocate as bytes (use int32 and round up)
            tiled_buf = self._gpu_mem.allocate(((tiled_size + 3) // 4,), dt_int32)

            # Run repack kernel
            src_tensor = self._make_tensor_from_ptr(weight.data_ptr, (N, half_K))
            self._kernels.launch(
                "repack_int4_tiled",
                (num_k_tiles, num_n_tiles),
                (256,),
                tiled_buf, src_tensor,
                N, half_K, num_k_tiles,
            )

            self._wmma_tiled_weights[name] = (tiled_buf, num_k_tiles)
            # Also index by data_ptr for fast lookup in _launch_matmul
            self._wmma_tiled_by_ptr[weight.data_ptr] = (tiled_buf, num_k_tiles)
            count += 1

        dt = time.monotonic() - t0
        if count > 0:
            print(f"[ZSE] Repacked {count} INT4 weights for WMMA tensor cores ({dt:.2f}s)")


    def _build_weight_index(self) -> list:
        """Pre-build per-layer weight dict for O(1) lookup without f-string formatting."""
        layer_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        index = []
        for layer in range(self._num_layers):
            d = {}
            for name in layer_names:
                full = f"model.layers.{layer}.{name}"
                w = self._weights.find(full)
                if w is None:
                    w = self._weights.find(f"layers.{layer}.{name}")
                if w is not None:
                    d[name] = w
            index.append(d)
        return index

    def _build_decode_tensor_cache(self):
        """Pre-build all Tensor wrappers needed during decode.

        Eliminates ~2880 Tensor() allocations per token by caching them.
        Tensors are lightweight wrappers around fixed GPU pointers, so they're
        safe to reuse across decode steps.
        """
        from zse_compiler.types.dtypes import float16 as dt_f16

        cache = []
        for layer in range(self._num_layers):
            lw = self._layer_weights[layer] if self._layer_weights else {}
            lc = {}

            # Weight tensors (pointer wrappers)
            for proj_name in ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                              "mlp.gate_proj.weight", "mlp.up_proj.weight",
                              "mlp.down_proj.weight"]:
                w = lw.get(proj_name)
                if w and w.dtype == "int4":
                    N = w.shape[0] if w.shape else 0
                    K = w.num_elements * 2 // N if N > 0 else 0
                    lc[f"{proj_name}.data"] = self._make_tensor_from_ptr(w.data_ptr, (N, K // 2))
                    lc[f"{proj_name}.scales"] = self._make_tensor_from_ptr(
                        w.scales_ptr, (N, K // w.group_size))
                    lc[f"{proj_name}.zeros"] = self._make_tensor_from_ptr(
                        w.zeros_ptr, (N, K // w.group_size))

            cache.append(lc)
        return cache

    def prefill(self, token_ids: List[int], seq_id: int, lora_adapter=None) -> bytes:
        """Run prefill: process full prompt, return logits for last token.

        Args:
            token_ids: Prompt token IDs
            seq_id: Sequence ID in the KV cache

        Returns:
            Raw bytes of logits (fp16, vocab_size elements) for the LAST token
        """
        seq_len = len(token_ids)
        if seq_len == 0:
            return b''

        # Allocate KV cache for this sequence
        self._kv_cache.allocate_sequence(seq_id, prompt_tokens=token_ids)
        self._kv_cache.mark_active(seq_id)

        # Upload token IDs to GPU
        token_tensor = self._upload_int32(token_ids)

        # Embedding lookup → fp32 hidden_f32
        embed_weight = self._weights.get("embed_tokens.weight")
        self._kernels.launch(
            "embedding_lookup_f32out",
            ((seq_len * self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._make_tensor_from_ptr(
                embed_weight.data_ptr, (self._config.vocab_size, self._hidden_size)),
            token_tensor, self._hidden_size, seq_len,
        )

        # Hoist attention metadata upload (shared across all layers)
        meta = self._kv_cache.get_attention_metadata([seq_id])
        block_table_tensor = self._upload_block_table(meta, seq_id=0)
        seq_lens_tensor = self._upload_int32(meta.seq_lengths)
        kv_slab = self._get_kv_slab_tensor()

        # Process each transformer layer
        for layer in range(self._num_layers):
            self._transformer_block_prefill(
                layer, seq_id, seq_len,
                kv_slab, block_table_tensor, seq_lens_tensor, meta,
                lora_adapter=lora_adapter,
            )

        # Cleanup hoisted tensors
        self._gpu_mem.free(block_table_tensor)
        self._gpu_mem.free(seq_lens_tensor)

        # Final RMSNorm — fp32 input → fp16 output
        final_norm_w = self._get_layer_weight("model.norm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            final_norm_w, seq_len,
        )

        # LM head: [seq_len, hidden] @ [vocab, hidden]^T → [seq_len, vocab]
        lm_head = self._weights.get("lm_head.weight")
        self._launch_matmul(
            self._scratch.logits, self._scratch.norm_out, lm_head,
            seq_len, self._config.vocab_size, self._hidden_size,
        )

        # Download logits for last token only
        # Offset to last row: (seq_len - 1) * vocab_size * 2 bytes
        last_logits_bytes = self._config.vocab_size * 2  # fp16
        logits_data = self._download_fp16(
            self._scratch.logits, self._config.vocab_size,
            row_offset=seq_len - 1,
        )

        self._gpu_mem.free(token_tensor)
        return logits_data

    # ------------------------------------------------------------------
    # Embedding extraction (for RAG dense retrieval / reranking)
    # ------------------------------------------------------------------

    def embed_pooled(self, token_ids: List[int], seq_id: int) -> bytes:
        """Run forward pass and return MEAN-POOLED, L2-normalized hidden state.

        Used by RAG for dense semantic embeddings — uses the loaded LLM as the
        encoder (no separate embedding model needed). Skips the LM-head matmul
        for ~10% speedup vs prefill.

        Args:
            token_ids: Input token IDs
            seq_id: Ephemeral KV slot — caller is responsible for freeing via
                    self._kv_cache.free_sequence(seq_id) afterward.

        Returns:
            Raw fp16 bytes of the pooled hidden state (hidden_size elements,
            L2-normalized so dot product == cosine similarity).
        """
        seq_len = len(token_ids)
        if seq_len == 0:
            return b''

        self._kv_cache.allocate_sequence(seq_id, prompt_tokens=token_ids)
        self._kv_cache.mark_active(seq_id)
        token_tensor = self._upload_int32(token_ids)

        embed_weight = self._weights.get("embed_tokens.weight")
        self._kernels.launch(
            "embedding_lookup_f32out",
            ((seq_len * self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._make_tensor_from_ptr(
                embed_weight.data_ptr, (self._config.vocab_size, self._hidden_size)),
            token_tensor, self._hidden_size, seq_len,
        )

        meta = self._kv_cache.get_attention_metadata([seq_id])
        block_table_tensor = self._upload_block_table(meta, seq_id=0)
        seq_lens_tensor = self._upload_int32(meta.seq_lengths)
        kv_slab = self._get_kv_slab_tensor()

        for layer in range(self._num_layers):
            self._transformer_block_prefill(
                layer, seq_id, seq_len,
                kv_slab, block_table_tensor, seq_lens_tensor, meta,
                lora_adapter=None,
            )

        self._gpu_mem.free(block_table_tensor)
        self._gpu_mem.free(seq_lens_tensor)

        # Final RMSNorm → fp16 [seq_len, hidden] in self._scratch.norm_out
        final_norm_w = self._get_layer_weight("model.norm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            final_norm_w, seq_len,
        )

        # Download all seq_len rows of fp16 hidden states
        all_hidden = self._download_fp16(
            self._scratch.norm_out, seq_len * self._hidden_size,
        )

        self._gpu_mem.free(token_tensor)

        # Mean-pool + L2-normalize in Python (cheap: hidden_size ~ 4K-5K)
        import struct, math
        H = self._hidden_size
        # Convert fp16 bytes → list[float]
        floats = _fp16_bytes_to_floats(all_hidden, seq_len * H)
        pooled = [0.0] * H
        for t in range(seq_len):
            base = t * H
            for i in range(H):
                pooled[i] += floats[base + i]
        inv = 1.0 / max(seq_len, 1)
        for i in range(H):
            pooled[i] *= inv

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in pooled))
        if norm > 1e-8:
            inv_n = 1.0 / norm
            for i in range(H):
                pooled[i] *= inv_n

        # Pack back to fp16 bytes for storage
        return _floats_to_fp16_bytes(pooled)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size

    def decode_step(self, token_id: int, seq_id: int, position: int,
                    lora_adapter=None, skip_logits_download: bool = False) -> bytes:
        """Run single decode step, return logits.

        Uses C fast decoder when available (10-20x faster).
        Falls back to Python path for LoRA or when C extension unavailable.

        Args:
            skip_logits_download: If True, don't download logits to CPU.
                Use with gpu_argmax() for greedy decoding (saves ~5ms for large vocabs).
        """
        # Extend KV cache by 1 token
        self._kv_cache.extend_sequence(seq_id, num_new_tokens=1)

        # Upload token + position to pre-allocated buffers
        data = struct.pack('<1i', token_id)
        self._gpu_mem.copy_host_to_device(data, self._decode_token_buf)
        pos_data = struct.pack('<1i', position)
        self._gpu_mem.copy_host_to_device(pos_data, self._decode_pos_buf)

        # Build attention metadata + upload block table / seq lens
        meta = self._kv_cache.get_attention_metadata([seq_id])
        bt = meta.block_tables[0]
        padded = bt + [-1] * (meta.max_blocks_per_seq - len(bt))
        bt_data = struct.pack(f'<{len(padded)}i', *padded)
        self._gpu_mem.copy_host_to_device(bt_data, self._decode_bt_buf)
        sl_data = struct.pack(f'<{len(meta.seq_lengths)}i', *meta.seq_lengths)
        self._gpu_mem.copy_host_to_device(sl_data, self._decode_seqlens_buf)

        # Fast C path (no LoRA support yet)
        if self._fast_decoder is not None and lora_adapter is None:
            total_tokens = meta.seq_lengths[0]
            result = self._fast_decoder.decode_step(
                self._decode_token_buf.data_ptr,
                self._decode_pos_buf.data_ptr,
                self._decode_bt_buf.data_ptr,
                self._decode_seqlens_buf.data_ptr,
                position,
                total_tokens,
            )
            if result != 0:
                raise RuntimeError(f"Fast decode failed with code {result}")
            return self._download_fp16(self._scratch.logits, self._config.vocab_size)

        # Python fallback path — fp32 residual stream
        embed_weight = self._weights.get("embed_tokens.weight")
        self._kernels.launch(
            "embedding_lookup_f32out",
            ((self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32,
            self._make_tensor_from_ptr(
                embed_weight.data_ptr, (self._config.vocab_size, self._hidden_size)),
            self._decode_token_buf, self._hidden_size, 1,
        )

        for layer in range(self._num_layers):
            self._transformer_block_decode(
                layer, seq_id, position,
                self._kv_slab, self._decode_bt_buf, self._decode_seqlens_buf, meta,
                lora_adapter=lora_adapter,
            )

        final_norm_w = self._get_layer_weight("model.norm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            final_norm_w, 1,
        )

        lm_head = self._weights.get("lm_head.weight")
        self._launch_matmul(
            self._scratch.logits, self._scratch.norm_out, lm_head,
            1, self._config.vocab_size, self._hidden_size,
        )

        if skip_logits_download:
            return b''  # Caller will use gpu_argmax() instead
        return self._download_fp16(self._scratch.logits, self._config.vocab_size)

    def verify_tokens(
        self,
        token_ids: List[int],
        seq_id: int,
        start_position: int,
        lora_adapter=None,
    ) -> List[bytes]:
        """Verify K+1 tokens for speculative decoding, return logits for EACH position.

        Like prefill but:
        - Returns ALL rows of logits (not just last)
        - Doesn't allocate new KV cache (sequence already exists)
        - Extends KV by len(token_ids) tokens

        Args:
            token_ids: [K+1] tokens to verify (last_accepted + K drafts)
            seq_id: Existing sequence ID in KV cache
            start_position: Position of the first token

        Returns:
            List of K+1 logits byte arrays (fp16, vocab_size each)
        """
        seq_len = len(token_ids)
        if seq_len == 0:
            return []

        # Extend KV cache for the verification tokens
        self._kv_cache.extend_sequence(seq_id, num_new_tokens=seq_len)

        # Upload token IDs
        token_tensor = self._upload_int32(token_ids)

        # Embedding lookup → fp32 hidden
        embed_weight = self._weights.get("embed_tokens.weight")
        self._kernels.launch(
            "embedding_lookup_f32out",
            ((seq_len * self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32,
            self._make_tensor_from_ptr(
                embed_weight.data_ptr, (self._config.vocab_size, self._hidden_size)),
            token_tensor, self._hidden_size, seq_len,
        )

        # Hoist attention metadata (shared across all layers)
        meta = self._kv_cache.get_attention_metadata([seq_id])
        block_table_tensor = self._upload_block_table(meta, seq_id=0)
        seq_lens_tensor = self._upload_int32(meta.seq_lengths)
        kv_slab = self._get_kv_slab_tensor()

        # Process transformer layers (reuse prefill path — it handles seq_len > 1)
        for layer in range(self._num_layers):
            self._transformer_block_prefill(
                layer, seq_id, seq_len,
                kv_slab, block_table_tensor, seq_lens_tensor, meta,
                lora_adapter=lora_adapter,
            )

        # Cleanup hoisted tensors
        self._gpu_mem.free(block_table_tensor)
        self._gpu_mem.free(seq_lens_tensor)

        # Final RMSNorm (fp32 → fp16)
        final_norm_w = self._get_layer_weight("model.norm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            final_norm_w, seq_len,
        )

        # LM head [seq_len, hidden] → [seq_len, vocab]
        lm_head = self._weights.get("lm_head.weight")
        self._launch_matmul(
            self._scratch.logits, self._scratch.norm_out, lm_head,
            seq_len, self._config.vocab_size, self._hidden_size,
        )

        # Bulk download ALL rows of logits
        all_logits = self._bulk_download_logits(seq_len)

        self._gpu_mem.free(token_tensor)
        return all_logits

    def batched_decode(
        self,
        token_ids: List[int],
        seq_ids: List[int],
        positions: List[int],
        lora_adapter=None,
        skip_logits_download: bool = False,
    ) -> Optional[List[bytes]]:
        """Run batched decode: process one token per sequence for M sequences.

        Single set of kernel launches for all M sequences — ~27x fewer launches
        than calling decode_step() M times in a loop.

        Args:
            token_ids: [M] — one token per sequence (last generated or last prompt)
            seq_ids: [M] — sequence IDs in KV cache
            positions: [M] — token position for each sequence

        Returns:
            List of M logits byte arrays (fp16, vocab_size elements each)
        """
        M = len(token_ids)
        if M == 0:
            return []

        # Step 0: Extend KV cache for each sequence
        for sid in seq_ids:
            self._kv_cache.extend_sequence(sid, num_new_tokens=1)

        # Upload tokens, positions, block table, seq lens to pre-allocated GPU buffers
        token_data = struct.pack(f'<{M}i', *token_ids)
        self._gpu_mem.copy_host_to_device(token_data, self._decode_token_buf)

        pos_data = struct.pack(f'<{M}i', *positions)
        self._gpu_mem.copy_host_to_device(pos_data, self._decode_pos_buf)

        meta = self._kv_cache.get_attention_metadata(seq_ids)
        block_table_data = meta.pack_block_tables()
        self._gpu_mem.copy_host_to_device(block_table_data, self._decode_bt_buf)

        sl_data = struct.pack(f'<{len(meta.seq_lengths)}i', *meta.seq_lengths)
        self._gpu_mem.copy_host_to_device(sl_data, self._decode_seqlens_buf)

        # === FAST C PATH (M=1, no LoRA) — one C call for entire forward pass ===
        if M == 1 and self._fast_decoder is not None and lora_adapter is None:
            total_tokens = meta.seq_lengths[0]
            result = self._fast_decoder.decode_step(
                self._decode_token_buf.data_ptr,
                self._decode_pos_buf.data_ptr,
                self._decode_bt_buf.data_ptr,
                self._decode_seqlens_buf.data_ptr,
                positions[0],
                total_tokens,
            )
            if result != 0:
                # Fall through to Python path on error
                self._fast_fail_count = getattr(self, '_fast_fail_count', 0) + 1
                if self._fast_fail_count <= 3:
                    print(f"[WARN] Fast decode failed (code {result}), using Python path ({self._fast_fail_count}x)")
                if self._fast_fail_count >= 5:
                    print("[WARN] Disabling fast decoder after 5 failures")
                    self._fast_decoder = None
            else:
                if skip_logits_download:
                    return None
                return self._bulk_download_logits(M)

        # === Python fallback path (M>=1, LoRA, or C extension unavailable) ===

        # Batched embedding lookup → fp32 hidden_f32
        embed_weight = self._weights.get("embed_tokens.weight")
        self._kernels.launch(
            "embedding_lookup_f32out",
            ((M * self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32,
            self._make_tensor_from_ptr(
                embed_weight.data_ptr, (self._config.vocab_size, self._hidden_size)),
            self._decode_token_buf, self._hidden_size, M,
        )

        kv_slab = self._kv_slab

        # Process all transformer layers (fp32 residual stream)
        for layer in range(self._num_layers):
            self._batched_transformer_block_decode(
                layer, M, positions, kv_slab,
                self._decode_bt_buf, self._decode_seqlens_buf, meta,
                lora_adapter=lora_adapter,
            )

        # Final RMSNorm [M, hidden_size] — fp32 input → fp16 output
        final_norm_w = self._get_layer_weight("model.norm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            final_norm_w, M,
        )

        # LM head [M, hidden] @ [vocab, hidden]^T → [M, vocab]
        lm_head = self._weights.get("lm_head.weight")
        self._launch_matmul(
            self._scratch.logits, self._scratch.norm_out, lm_head,
            M, self._config.vocab_size, self._hidden_size,
        )

        # Bulk download all M rows of logits in one transfer
        if skip_logits_download:
            return None
        all_logits = self._bulk_download_logits(M)

        return all_logits

    def _batched_transformer_block_decode(
        self,
        layer: int,
        M: int,
        positions: List[int],
        kv_slab: Tensor,
        block_table_tensor: Tensor,
        seq_lens_tensor: Tensor,
        meta,
        lora_adapter=None,
    ):
        """One transformer layer for batched decode (M sequences, 1 token each).

        Batched operations (single kernel launch for M sequences):
            - RMSNorm, matmuls, SiLU, residual: just pass M as batch dim
            - Paged attention: grid=(M, num_heads), each blockIdx.x handles one sequence

        Per-sequence operations (M small launches):
            - RoPE: different position per sequence (M launches, each tiny)
            - KV cache write: different block table per sequence (M launches)
        """
        n = M * self._hidden_size

        # Save residual [M, hidden] — fp32 copy
        self._copy_tensor_f32(self._scratch.residual_f32, self._scratch.hidden_f32, n)

        # Pre-attention RMSNorm [M rows] — fp32 input → fp16 output
        attn_norm_w = self._get_layer_weight(f"model.layers.{layer}.input_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            attn_norm_w, M,
        )

        # QKV projection (batched: M rows at once) — fp16 in/out
        q_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.q_proj.weight")
        k_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.k_proj.weight")
        v_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.v_proj.weight")

        q_dim = self._num_heads * self._head_dim
        k_dim = self._num_kv_heads * self._head_dim

        q_buf = self._scratch.qkv
        k_buf = self._scratch.attn_out
        v_buf = self._scratch.mlp_out

        self._launch_matmul_with_lora(q_buf, self._scratch.norm_out, q_proj, M, q_dim, self._hidden_size,
                                      lora_adapter, layer, "q_proj")
        self._launch_matmul_with_lora(k_buf, self._scratch.norm_out, k_proj, M, k_dim, self._hidden_size,
                                      lora_adapter, layer, "k_proj")
        self._launch_matmul_with_lora(v_buf, self._scratch.norm_out, v_proj, M, k_dim, self._hidden_size,
                                      lora_adapter, layer, "v_proj")

        # Apply QKV biases (Qwen2 has biases on QKV projections)
        self._apply_bias_if_present(q_buf, layer, "self_attn.q_proj.bias", M, q_dim)
        self._apply_bias_if_present(k_buf, layer, "self_attn.k_proj.bias", M, k_dim)
        self._apply_bias_if_present(v_buf, layer, "self_attn.v_proj.bias", M, k_dim)

        # RoPE + KV cache write — on ROCm, use the fused kernel (1 launch instead of 2,
        # K-buffer round-trip eliminated). CUDA falls back to the original two-kernel path.
        if self._backend == "rocm" and "fused_rope_kv_write" in self._kernels._kernel_sources:
            half_dim = self._head_dim // 2
            max_threads = max(self._num_heads * half_dim, self._num_kv_heads * self._head_dim)
            self._kernels.launch(
                "fused_rope_kv_write",
                ((max_threads + 255) // 256, M),
                (256,),
                q_buf, k_buf, v_buf, kv_slab,
                block_table_tensor, self._decode_pos_buf,
                M, self._num_heads, self._num_kv_heads, self._head_dim,
                self._kv_cache.block_size, meta.max_blocks_per_seq,
                self._num_layers, layer, self._rope_theta,
            )
        else:
            # RoPE — BATCHED (single kernel launch for all M sequences)
            # Uses pre-uploaded positions buffer (no per-layer alloc/free)
            half_dim = self._head_dim // 2
            total_rope = M * max(self._num_heads, self._num_kv_heads) * half_dim
            self._kernels.launch(
                "batched_rotary_embedding",
                ((total_rope + 255) // 256,),
                (256,),
                q_buf, k_buf, self._decode_pos_buf,
                M, self._num_heads, self._num_kv_heads,
                self._head_dim, self._rope_theta,
            )

            # KV cache write — BATCHED (single launch for all M sequences)
            total_kv = self._num_kv_heads * self._head_dim
            self._kernels.launch(
                "batched_kv_cache_write",
                ((total_kv + 255) // 256, M),
                (256,),
                kv_slab, k_buf, v_buf,
                block_table_tensor, self._decode_pos_buf,
                M, self._num_kv_heads, self._head_dim,
                self._kv_cache.block_size,
                meta.max_blocks_per_seq, self._num_layers, layer,
            )

        # Paged attention — FULLY BATCHED (this is the big win)
        # Grid: (M, num_heads) — each blockIdx.x handles one sequence
        max_seq_len = max(meta.seq_lengths) if meta.seq_lengths else 1
        shared_mem = max_seq_len * 4  # float per token for attention scores

        self._kernels.launch(
            "paged_attention",
            (M, self._num_heads),
            (min(256, self._head_dim),),
            self._scratch.attn_out, q_buf, kv_slab,
            block_table_tensor, seq_lens_tensor,
            self._num_heads, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer,
            self._scale,
            shared_mem_bytes=shared_mem,
        )

        # O projection (batched: M rows, LoRA-aware) → fp16 hidden
        o_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.o_proj.weight")
        self._launch_matmul_with_lora(self._scratch.hidden, self._scratch.attn_out, o_proj,
                            M, self._hidden_size, q_dim,
                            lora_adapter, layer, "o_proj")

        # Fused: residual_f32 = fp16_hidden + fp32_residual_f32; norm_out = RMSNorm(residual_f32)
        mlp_norm_w = self._get_layer_weight(f"model.layers.{layer}.post_attention_layernorm.weight")
        self._launch_fused_residual_rmsnorm_f32(
            self._scratch.norm_out, self._scratch.residual_f32,
            self._scratch.hidden, self._scratch.residual_f32, mlp_norm_w, M,
        )
        # Copy updated residual_f32 → hidden_f32
        self._copy_tensor_f32(self._scratch.hidden_f32, self._scratch.residual_f32, n)

        # MLP (batched: M rows, LoRA-aware) — all fp16
        gate_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.gate_proj.weight")
        up_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.up_proj.weight")
        down_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.down_proj.weight")

        self._launch_matmul_with_lora(self._scratch.mlp_gate, self._scratch.norm_out, gate_proj,
                            M, self._intermediate_size, self._hidden_size,
                            lora_adapter, layer, "gate_proj")
        self._launch_matmul_with_lora(self._scratch.mlp_up, self._scratch.norm_out, up_proj,
                            M, self._intermediate_size, self._hidden_size,
                            lora_adapter, layer, "up_proj")

        n_mlp = M * self._intermediate_size
        self._kernels.launch(
            "silu_mul",
            ((n_mlp + 255) // 256,),
            (256,),
            self._scratch.mlp_gate, self._scratch.mlp_gate, self._scratch.mlp_up, n_mlp,
        )

        self._launch_matmul_with_lora(self._scratch.hidden, self._scratch.mlp_gate, down_proj,
                            M, self._hidden_size, self._intermediate_size,
                            lora_adapter, layer, "down_proj")

        # Final residual: hidden_f32 += fp16 hidden (MLP output)
        self._kernels.launch(
            "residual_add_f32",
            ((n + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.hidden_f32, self._scratch.hidden, n,
        )

    # ================================================================
    # Transformer block implementations (single-sequence)
    # ================================================================

    def _transformer_block_prefill(self, layer: int, seq_id: int, seq_len: int,
                                    kv_slab: Tensor, block_table_tensor: Tensor,
                                    seq_lens_tensor: Tensor, meta,
                                    lora_adapter=None):
        """One transformer layer during prefill."""
        n = seq_len * self._hidden_size

        # Save residual (fp32)
        self._copy_tensor_f32(self._scratch.residual_f32, self._scratch.hidden_f32, n)

        # Pre-attention RMSNorm — fp32 input → fp16 output
        attn_norm_w = self._get_layer_weight(f"model.layers.{layer}.input_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            attn_norm_w, seq_len,
        )

        # QKV projection: [seq_len, hidden] @ [qkv_dim, hidden]^T
        q_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.q_proj.weight")
        k_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.k_proj.weight")
        v_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.v_proj.weight")

        q_dim = self._num_heads * self._head_dim
        k_dim = self._num_kv_heads * self._head_dim
        v_dim = k_dim

        # Q projection (LoRA-aware)
        q_buf = self._scratch.qkv  # Reuse QKV buffer for Q
        self._launch_matmul_with_lora(q_buf, self._scratch.norm_out, q_proj,
                            seq_len, q_dim, self._hidden_size,
                            lora_adapter, layer, "q_proj")

        # K projection (LoRA-aware)
        k_buf = self._scratch.attn_out  # Temporarily reuse attn_out for K
        self._launch_matmul_with_lora(k_buf, self._scratch.norm_out, k_proj,
                            seq_len, k_dim, self._hidden_size,
                            lora_adapter, layer, "k_proj")

        # V projection (LoRA-aware)
        v_buf = self._scratch.mlp_out  # Temporarily reuse mlp_out for V
        self._launch_matmul_with_lora(v_buf, self._scratch.norm_out, v_proj,
                            seq_len, v_dim, self._hidden_size,
                            lora_adapter, layer, "v_proj")

        # Apply QKV biases (Qwen2 has biases on QKV projections)
        self._apply_bias_if_present(q_buf, layer, "self_attn.q_proj.bias", seq_len, q_dim)
        self._apply_bias_if_present(k_buf, layer, "self_attn.k_proj.bias", seq_len, k_dim)
        self._apply_bias_if_present(v_buf, layer, "self_attn.v_proj.bias", seq_len, v_dim)

        # RoPE on Q and K
        half_dim = self._head_dim // 2
        total_rope = seq_len * max(self._num_heads, self._num_kv_heads) * half_dim
        self._kernels.launch(
            "rotary_embedding",
            ((total_rope + 255) // 256,),
            (256,),
            q_buf, k_buf,
            seq_len, self._num_heads, self._num_kv_heads,
            self._head_dim, self._prefill_pos_zero_buf, self._rope_theta,
        )

        # Write KV to paged cache BEFORE attention (attention overwrites k_buf=attn_out)
        total_kv = seq_len * self._num_kv_heads * self._head_dim
        self._kernels.launch(
            "kv_cache_write",
            ((total_kv + 255) // 256,),
            (256,),
            kv_slab, k_buf, v_buf, block_table_tensor,
            seq_len, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer, self._prefill_pos_zero_buf,
        )

        # Self-attention (prefill: causal attention on freshly-computed Q/K/V)
        # Use prefill_attention kernel with causal masking, not paged_attention
        # NOTE: Write output to norm_out (not attn_out) because k_buf aliases attn_out.
        # Writing to attn_out would corrupt K data mid-computation.
        attn_output = self._scratch.norm_out  # norm_out is free (consumed by QKV projections)
        shared_mem = seq_len * 4  # float per token for scores
        if shared_mem > 48 * 1024:
            # Exceeds default 48KB smem limit — chunk the prefill
            # For now, cap and warn. FlashAttention tiling needed for seq_len > 12K.
            raise ValueError(
                f"Prefill seq_len={seq_len} requires {shared_mem//1024}KB shared memory "
                f"(max 48KB). Reduce max_seq_len or implement FlashAttention."
            )
        self._kernels.launch(
            "prefill_attention",
            (seq_len, self._num_heads),
            (min(256, self._head_dim),),
            attn_output, q_buf, k_buf, v_buf,
            seq_len, self._num_heads, self._num_kv_heads, self._head_dim,
            self._scale,
            shared_mem_bytes=shared_mem,
        )

        # O projection: [seq_len, num_heads * head_dim] → [seq_len, hidden] (LoRA-aware) — fp16
        o_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.o_proj.weight")
        self._launch_matmul_with_lora(self._scratch.hidden, attn_output, o_proj,
                            seq_len, self._hidden_size, q_dim,
                            lora_adapter, layer, "o_proj")

        # Residual: hidden_f32 = residual_f32 + fp16 hidden (attention output)
        self._kernels.launch(
            "residual_add_f32",
            ((n + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.residual_f32, self._scratch.hidden, n,
        )

        # Fused: residual_f32 = fp16_hidden(from hidden buffer reused as temp) + fp32_hidden_f32
        # Actually simpler: save hidden_f32 as new residual, then RMSNorm
        # Use fused_residual_rmsnorm_f32 which does: residual_f32 = hidden(fp16) + residual_f32; norm = RMSNorm
        # But we already did the residual add above. So just do RMSNorm on hidden_f32.
        mlp_norm_w = self._get_layer_weight(f"model.layers.{layer}.post_attention_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            mlp_norm_w, seq_len,
        )

        # MLP: gate_proj + up_proj → SiLU*up → down_proj
        gate_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.gate_proj.weight")
        up_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.up_proj.weight")
        down_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.down_proj.weight")

        # Fused gate+up GEMV for M=1 decode (saves kernel launch + input re-read)
        if seq_len == 1 and lora_adapter is None and gate_proj.dtype == "int4":
            N = self._intermediate_size
            K = self._hidden_size
            gs = gate_proj.group_size
            gate_w = self._make_tensor_from_ptr(gate_proj.data_ptr, (N, K // 2))
            gate_s = self._make_tensor_from_ptr(gate_proj.scales_ptr, (N, K // gs))
            gate_z = self._make_tensor_from_ptr(gate_proj.zeros_ptr, (N, K // gs))
            up_w = self._make_tensor_from_ptr(up_proj.data_ptr, (N, K // 2))
            up_s = self._make_tensor_from_ptr(up_proj.scales_ptr, (N, K // gs))
            up_z = self._make_tensor_from_ptr(up_proj.zeros_ptr, (N, K // gs))
            fgu_rpb = 2 if self._backend == "rocm" else 4  # rows per block for fused gate+up
            self._kernels.launch(
                "fused_gate_up_gemv_int4",
                ((N + fgu_rpb - 1) // fgu_rpb,),
                (256,),
                self._scratch.mlp_gate, self._scratch.mlp_up,
                gate_w, gate_s, gate_z,
                up_w, up_s, up_z,
                self._scratch.norm_out, N, K, gs,
            )
        else:
            self._launch_matmul_with_lora(self._scratch.mlp_gate, self._scratch.norm_out, gate_proj,
                                seq_len, self._intermediate_size, self._hidden_size,
                                lora_adapter, layer, "gate_proj")
            self._launch_matmul_with_lora(self._scratch.mlp_up, self._scratch.norm_out, up_proj,
                                seq_len, self._intermediate_size, self._hidden_size,
                                lora_adapter, layer, "up_proj")

        # SiLU(gate) * up
        n_mlp = seq_len * self._intermediate_size
        self._kernels.launch(
            "silu_mul",
            ((n_mlp + 255) // 256,),
            (256,),
            self._scratch.mlp_gate, self._scratch.mlp_gate, self._scratch.mlp_up, n_mlp,
        )

        # Down projection (LoRA-aware)
        self._launch_matmul_with_lora(self._scratch.hidden, self._scratch.mlp_gate, down_proj,
                            seq_len, self._hidden_size, self._intermediate_size,
                            lora_adapter, layer, "down_proj")

        # Final residual: hidden_f32 += fp16 hidden (MLP output)
        self._kernels.launch(
            "residual_add_f32",
            ((n + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.hidden_f32, self._scratch.hidden, n,
        )

    def _transformer_block_decode(self, layer: int, seq_id: int, position: int,
                                   kv_slab: Tensor, block_table_tensor: Tensor,
                                   seq_lens_tensor: Tensor, meta,
                                   lora_adapter=None):
        """One transformer layer during decode (single token)."""
        seq_len = 1

        # Save residual (fp32)
        self._copy_tensor_f32(self._scratch.residual_f32, self._scratch.hidden_f32, self._hidden_size)

        # Pre-attention RMSNorm — fp32→fp16
        attn_norm_w = self._get_layer_weight(f"model.layers.{layer}.input_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            attn_norm_w, seq_len,
        )

        # QKV projection (use cached tensors for fast path when no LoRA)
        q_dim = self._num_heads * self._head_dim
        k_dim = self._num_kv_heads * self._head_dim

        q_buf = self._scratch.qkv
        k_buf = self._scratch.attn_out
        v_buf = self._scratch.mlp_out

        if lora_adapter is None:
            self._launch_gemv_cached(q_buf, self._scratch.norm_out, layer,
                                     "self_attn.q_proj.weight", q_dim, self._hidden_size)
            self._launch_gemv_cached(k_buf, self._scratch.norm_out, layer,
                                     "self_attn.k_proj.weight", k_dim, self._hidden_size)
            self._launch_gemv_cached(v_buf, self._scratch.norm_out, layer,
                                     "self_attn.v_proj.weight", k_dim, self._hidden_size)
        else:
            q_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.q_proj.weight")
            k_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.k_proj.weight")
            v_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.v_proj.weight")
            self._launch_matmul_with_lora(q_buf, self._scratch.norm_out, q_proj, 1, q_dim, self._hidden_size,
                                lora_adapter, layer, "q_proj")
            self._launch_matmul_with_lora(k_buf, self._scratch.norm_out, k_proj, 1, k_dim, self._hidden_size,
                                lora_adapter, layer, "k_proj")
            self._launch_matmul_with_lora(v_buf, self._scratch.norm_out, v_proj, 1, k_dim, self._hidden_size,
                                lora_adapter, layer, "v_proj")

        # Apply QKV biases (Qwen2 has biases on QKV projections)
        self._apply_bias_if_present(q_buf, layer, "self_attn.q_proj.bias", 1, q_dim)
        self._apply_bias_if_present(k_buf, layer, "self_attn.k_proj.bias", 1, k_dim)
        self._apply_bias_if_present(v_buf, layer, "self_attn.v_proj.bias", 1, k_dim)

        # RoPE
        half_dim = self._head_dim // 2
        total_rope = max(self._num_heads, self._num_kv_heads) * half_dim
        self._kernels.launch(
            "rotary_embedding",
            ((total_rope + 255) // 256,),
            (256,),
            q_buf, k_buf,
            1, self._num_heads, self._num_kv_heads,
            self._head_dim, self._decode_pos_buf, self._rope_theta,
        )

        # Write new K, V to cache (use hoisted tensors)
        total_kv = self._num_kv_heads * self._head_dim
        self._kernels.launch(
            "kv_cache_write",
            ((total_kv + 255) // 256,),
            (256,),
            kv_slab, k_buf, v_buf, block_table_tensor,
            1, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer, self._decode_pos_buf,
        )

        # Paged attention (use hoisted seq_lens_tensor)
        total_tokens = meta.seq_lengths[0]
        shared_mem = total_tokens * 4

        self._kernels.launch(
            "paged_attention",
            (1, self._num_heads),
            (min(256, self._head_dim),),
            self._scratch.attn_out, q_buf, kv_slab,
            block_table_tensor, seq_lens_tensor,
            self._num_heads, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer,
            self._scale,
            shared_mem_bytes=shared_mem,
        )

        # O projection (use cached path for non-LoRA)
        if lora_adapter is None:
            self._launch_gemv_cached(self._scratch.hidden, self._scratch.attn_out, layer,
                                     "self_attn.o_proj.weight", self._hidden_size, q_dim)
        else:
            o_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.o_proj.weight")
            self._launch_matmul_with_lora(self._scratch.hidden, self._scratch.attn_out, o_proj,
                                1, self._hidden_size, q_dim,
                                lora_adapter, layer, "o_proj")

        # Residual: hidden_f32 = residual_f32 + fp16 hidden
        self._kernels.launch(
            "residual_add_f32",
            ((self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.residual_f32, self._scratch.hidden,
            self._hidden_size,
        )

        # Post-attention RMSNorm — fp32→fp16
        mlp_norm_w = self._get_layer_weight(f"model.layers.{layer}.post_attention_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            mlp_norm_w, 1,
        )

        gate_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.gate_proj.weight")
        up_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.up_proj.weight")
        down_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.down_proj.weight")

        # Use fused gate+up GEMV for INT4 decode (single kernel reads input once)
        use_fused_gemv = (gate_proj.dtype == "int4" and lora_adapter is None)
        if use_fused_gemv:
            N = self._intermediate_size
            K = self._hidden_size
            gate_wt = self._make_tensor_from_ptr(gate_proj.data_ptr, (N, K // 2))
            gate_sc = self._make_tensor_from_ptr(gate_proj.scales_ptr, (N, K // gate_proj.group_size))
            gate_zr = self._make_tensor_from_ptr(gate_proj.zeros_ptr, (N, K // gate_proj.group_size))
            up_wt = self._make_tensor_from_ptr(up_proj.data_ptr, (N, K // 2))
            up_sc = self._make_tensor_from_ptr(up_proj.scales_ptr, (N, K // up_proj.group_size))
            up_zr = self._make_tensor_from_ptr(up_proj.zeros_ptr, (N, K // up_proj.group_size))
            fgu_rpb = 2 if self._backend == "rocm" else 4
            self._kernels.launch(
                "fused_gate_up_gemv_int4",
                ((N + fgu_rpb - 1) // fgu_rpb,),
                (256,),
                self._scratch.mlp_gate, self._scratch.mlp_up,
                gate_wt, gate_sc, gate_zr,
                up_wt, up_sc, up_zr,
                self._scratch.norm_out,
                N, K, gate_proj.group_size,
            )
        else:
            self._launch_matmul_with_lora(self._scratch.mlp_gate, self._scratch.norm_out, gate_proj,
                                1, self._intermediate_size, self._hidden_size,
                                lora_adapter, layer, "gate_proj")
            self._launch_matmul_with_lora(self._scratch.mlp_up, self._scratch.norm_out, up_proj,
                                1, self._intermediate_size, self._hidden_size,
                                lora_adapter, layer, "up_proj")

        n_mlp = self._intermediate_size
        self._kernels.launch(
            "silu_mul",
            ((n_mlp + 255) // 256,),
            (256,),
            self._scratch.mlp_gate, self._scratch.mlp_gate, self._scratch.mlp_up, n_mlp,
        )

        if lora_adapter is None:
            self._launch_gemv_cached(self._scratch.hidden, self._scratch.mlp_gate, layer,
                                     "mlp.down_proj.weight", self._hidden_size, self._intermediate_size)
        else:
            self._launch_matmul_with_lora(self._scratch.hidden, self._scratch.mlp_gate, down_proj,
                                1, self._hidden_size, self._intermediate_size,
                                lora_adapter, layer, "down_proj")

        # MLP residual: hidden_f32 += fp16 hidden (MLP output)
        # IMPORTANT: use hidden_f32 (which already includes attention residual),
        # NOT residual_f32 (which is pre-attention)
        n_res = self._hidden_size
        self._kernels.launch(
            "residual_add_f32",
            ((n_res + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.hidden_f32, self._scratch.hidden,
            n_res,
        )

    # ================================================================
    # Helper methods
    # ================================================================

    def _apply_bias_if_present(self, buf: Tensor, layer: int, bias_name: str, M: int, N: int,
                               stream=None):
        """Add bias to buf [M, N] if bias weight exists (e.g. Qwen2 QKV biases)."""
        if self._layer_weights and layer < len(self._layer_weights):
            bias_w = self._layer_weights[layer].get(bias_name)
            if bias_w is not None:
                bias_t = self._make_tensor_from_ptr(bias_w.data_ptr, (N,))
                total = M * N
                self._kernels.launch(
                    "bias_add",
                    ((total + 255) // 256,),
                    (256,),
                    buf, bias_t, M, N,
                    stream=stream,
                )

    def _get_layer_weight(self, name: str) -> GPUWeight:
        """Get a weight, using pre-built index for layer weights."""
        # Fast path: layer weight from pre-built index
        if name.startswith("model.layers.") and self._layer_weights:
            parts = name.split(".", 3)  # ['model', 'layers', '0', 'self_attn.q_proj.weight']
            if len(parts) == 4:
                try:
                    layer_idx = int(parts[2])
                    short_name = parts[3]
                    w = self._layer_weights[layer_idx].get(short_name)
                    if w is not None:
                        return w
                except (ValueError, IndexError):
                    pass
        # Slow fallback
        w = self._weights.find(name)
        if w is not None:
            return w
        # Try without 'model.' prefix
        if name.startswith("model."):
            w = self._weights.find(name[6:])
            if w is not None:
                return w
        raise KeyError(f"Weight not found: {name}")

    def _bulk_download_logits(self, num_rows: int) -> list:
        """Download all rows of logits in one bulk transfer, then slice."""
        vocab = self._config.vocab_size
        row_bytes = vocab * 2  # fp16
        total_bytes = num_rows * row_bytes
        # Create a view spanning all rows
        view = Tensor(shape=(num_rows * vocab,), dtype=float16)
        view._data_ptr = self._scratch.logits.data_ptr
        view._nbytes = total_bytes
        bulk_data = self._gpu_mem.copy_device_to_host(view)
        # Slice into per-row byte arrays
        return [bulk_data[i * row_bytes:(i + 1) * row_bytes] for i in range(num_rows)]

    def gpu_argmax(self, row_idx: int = 0) -> int:
        """Run argmax on GPU logits without downloading them.

        For greedy decoding: avoids 300KB DtoH transfer + CPU argmax.
        Returns token ID directly. Only downloads 4 bytes (1 int).
        """
        vocab = self._config.vocab_size
        # Point to the right row in logits buffer
        logits_row = self._make_tensor_from_ptr(
            self._scratch.logits.data_ptr + row_idx * vocab * 2,
            (vocab,))

        # Use decode_token_buf[0] as output (reuse pre-allocated buffer)
        out_buf = self._decode_token_buf

        self._kernels.launch(
            "argmax_fp16",
            (1,),
            (256,),
            out_buf, logits_row, vocab,
        )

        # Download just 4 bytes (the argmax index)
        view = Tensor(shape=(1,), dtype=int32)
        view._data_ptr = out_buf.data_ptr
        view._nbytes = 4
        result_bytes = self._gpu_mem.copy_device_to_host(view)
        return struct.unpack('<i', result_bytes[:4])[0]

    # ================================================================
    # GPU Graph — capture & replay decode forward pass
    # ================================================================

    def init_graph(self, max_seq_len: int = 2048):
        """Initialize graph runner. Call once before using decode_step_graph()."""
        from zse_engine.orchestrator.hip_graph import HIPGraphRunner, CUDAGraphRunner

        if self._backend == "rocm":
            driver = self._gpu_mem._driver
            self._graph_runner = HIPGraphRunner(driver)
        else:
            driver = self._gpu_mem._driver
            self._graph_runner = CUDAGraphRunner(driver)

        self._graph_stream = self._graph_runner.create_stream()
        self._graph_max_shared_mem = max_seq_len * 4
        self._graph_max_blocks = (max_seq_len + self._kv_cache.block_size - 1) // self._kv_cache.block_size
        self._graph_captured = False

        # Allocate a separate argmax output buffer for graph (max 64 sequences)
        from zse_compiler.types.dtypes import int32 as dt_int32
        self._graph_argmax_buf = self._gpu_mem.allocate((64,), dt_int32)
        self._batched_graph_runners = {}

    def _run_decode_kernels(self, seq_id: int, meta, stream=None):
        """Run all decode kernels (embedding → layers → norm → lm_head → argmax).

        Extracted so it can be called normally OR during graph capture.
        All kernel launches go to `stream` if provided.
        """
        # Embedding lookup (token already in _decode_token_buf)
        embed_weight = self._weights.get("embed_tokens.weight")
        self._kernels.launch(
            "embedding_lookup_f32out",
            ((self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32,
            self._make_tensor_from_ptr(
                embed_weight.data_ptr, (self._config.vocab_size, self._hidden_size)),
            self._decode_token_buf, self._hidden_size, 1,
            stream=stream,
        )

        # Transformer blocks
        for layer in range(self._num_layers):
            self._transformer_block_decode_graph(
                layer, seq_id, meta, stream=stream,
            )

        # Final RMSNorm
        final_norm_w = self._get_layer_weight("model.norm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            final_norm_w, 1, stream=stream,
        )

        # LM Head
        lm_head = self._weights.get("lm_head.weight")
        self._launch_matmul(
            self._scratch.logits, self._scratch.norm_out, lm_head,
            1, self._config.vocab_size, self._hidden_size,
            stream=stream,
        )

        # GPU argmax into graph_argmax_buf
        vocab = self._config.vocab_size
        logits_row = self._make_tensor_from_ptr(
            self._scratch.logits.data_ptr, (vocab,))
        self._kernels.launch(
            "argmax_fp16",
            (1,),
            (256,),
            self._graph_argmax_buf, logits_row, vocab,
            stream=stream,
        )

    def _transformer_block_decode_graph(self, layer: int, seq_id: int, meta,
                                         stream=None):
        """One transformer layer during graph-captured decode.

        Same as _transformer_block_decode but:
        - No LoRA support (graph mode is greedy-only optimization)
        - All launches go to stream
        - Uses max shared_mem for paged_attention
        """
        seq_len = 1

        # Save residual
        self._copy_tensor_f32(self._scratch.residual_f32, self._scratch.hidden_f32,
                              self._hidden_size, stream=stream)

        # Pre-attention RMSNorm
        attn_norm_w = self._get_layer_weight(f"model.layers.{layer}.input_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            attn_norm_w, seq_len, stream=stream,
        )

        # QKV projection
        q_dim = self._num_heads * self._head_dim
        k_dim = self._num_kv_heads * self._head_dim
        q_buf = self._scratch.qkv
        k_buf = self._scratch.attn_out
        v_buf = self._scratch.mlp_out

        self._launch_gemv_cached(q_buf, self._scratch.norm_out, layer,
                                 "self_attn.q_proj.weight", q_dim, self._hidden_size,
                                 stream=stream)
        self._launch_gemv_cached(k_buf, self._scratch.norm_out, layer,
                                 "self_attn.k_proj.weight", k_dim, self._hidden_size,
                                 stream=stream)
        self._launch_gemv_cached(v_buf, self._scratch.norm_out, layer,
                                 "self_attn.v_proj.weight", k_dim, self._hidden_size,
                                 stream=stream)

        # QKV biases
        self._apply_bias_if_present(q_buf, layer, "self_attn.q_proj.bias", 1, q_dim,
                                    stream=stream)
        self._apply_bias_if_present(k_buf, layer, "self_attn.k_proj.bias", 1, k_dim,
                                    stream=stream)
        self._apply_bias_if_present(v_buf, layer, "self_attn.v_proj.bias", 1, k_dim,
                                    stream=stream)

        # RoPE (reads position from self._decode_pos_buf)
        half_dim = self._head_dim // 2
        total_rope = max(self._num_heads, self._num_kv_heads) * half_dim
        self._kernels.launch(
            "rotary_embedding",
            ((total_rope + 255) // 256,),
            (256,),
            q_buf, k_buf,
            1, self._num_heads, self._num_kv_heads,
            self._head_dim, self._decode_pos_buf, self._rope_theta,
            stream=stream,
        )

        # KV cache write (reads position from self._decode_pos_buf)
        total_kv = self._num_kv_heads * self._head_dim
        self._kernels.launch(
            "kv_cache_write",
            ((total_kv + 255) // 256,),
            (256,),
            self._kv_slab, k_buf, v_buf, self._decode_bt_buf,
            1, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer, self._decode_pos_buf,
            stream=stream,
        )

        # Paged attention — use max shared mem for graph compatibility
        self._kernels.launch(
            "paged_attention",
            (1, self._num_heads),
            (min(256, self._head_dim),),
            self._scratch.attn_out, q_buf, self._kv_slab,
            self._decode_bt_buf, self._decode_seqlens_buf,
            self._num_heads, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer,
            self._scale,
            shared_mem_bytes=self._graph_max_shared_mem,
            stream=stream,
        )

        # O projection
        self._launch_gemv_cached(self._scratch.hidden, self._scratch.attn_out, layer,
                                 "self_attn.o_proj.weight", self._hidden_size, q_dim,
                                 stream=stream)

        # Attention residual
        self._kernels.launch(
            "residual_add_f32",
            ((self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.residual_f32, self._scratch.hidden,
            self._hidden_size,
            stream=stream,
        )

        # Post-attention RMSNorm
        mlp_norm_w = self._get_layer_weight(f"model.layers.{layer}.post_attention_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            mlp_norm_w, 1, stream=stream,
        )

        # MLP — fused gate+up for INT4
        gate_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.gate_proj.weight")
        up_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.up_proj.weight")

        if gate_proj.dtype == "int4":
            N = self._intermediate_size
            K = self._hidden_size
            gate_wt = self._make_tensor_from_ptr(gate_proj.data_ptr, (N, K // 2))
            gate_sc = self._make_tensor_from_ptr(gate_proj.scales_ptr, (N, K // gate_proj.group_size))
            gate_zr = self._make_tensor_from_ptr(gate_proj.zeros_ptr, (N, K // gate_proj.group_size))
            up_wt = self._make_tensor_from_ptr(up_proj.data_ptr, (N, K // 2))
            up_sc = self._make_tensor_from_ptr(up_proj.scales_ptr, (N, K // up_proj.group_size))
            up_zr = self._make_tensor_from_ptr(up_proj.zeros_ptr, (N, K // up_proj.group_size))
            fgu_rpb = 2 if self._backend == "rocm" else 4
            self._kernels.launch(
                "fused_gate_up_gemv_int4",
                ((N + fgu_rpb - 1) // fgu_rpb,),
                (256,),
                self._scratch.mlp_gate, self._scratch.mlp_up,
                gate_wt, gate_sc, gate_zr,
                up_wt, up_sc, up_zr,
                self._scratch.norm_out,
                N, K, gate_proj.group_size,
                stream=stream,
            )
        else:
            self._launch_matmul(self._scratch.mlp_gate, self._scratch.norm_out, gate_proj,
                                1, self._intermediate_size, self._hidden_size, stream=stream)
            self._launch_matmul(self._scratch.mlp_up, self._scratch.norm_out, up_proj,
                                1, self._intermediate_size, self._hidden_size, stream=stream)

        # SiLU * up
        n_mlp = self._intermediate_size
        self._kernels.launch(
            "silu_mul",
            ((n_mlp + 255) // 256,),
            (256,),
            self._scratch.mlp_gate, self._scratch.mlp_gate, self._scratch.mlp_up, n_mlp,
            stream=stream,
        )

        # Down projection
        self._launch_gemv_cached(self._scratch.hidden, self._scratch.mlp_gate, layer,
                                 "mlp.down_proj.weight", self._hidden_size, self._intermediate_size,
                                 stream=stream)

        # MLP residual
        self._kernels.launch(
            "residual_add_f32",
            ((self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.hidden_f32, self._scratch.hidden,
            self._hidden_size,
            stream=stream,
        )

    def decode_step_graph(self, token_id: int, seq_id: int, position: int) -> int:
        """Graph-captured decode step. Returns token_id directly (greedy only).

        First call: captures the graph (warmup must have been done via normal decode_step).
        Subsequent calls: replay the captured graph.
        """
        # CPU-side work (always before graph replay)
        self._kv_cache.extend_sequence(seq_id, num_new_tokens=1)

        # Upload changing values to GPU buffers
        self._gpu_mem.copy_host_to_device(struct.pack('<1i', token_id), self._decode_token_buf)
        self._gpu_mem.copy_host_to_device(struct.pack('<1i', position), self._decode_pos_buf)

        # Build attention metadata + upload block table / seq lens
        # Use FIXED max_blocks for graph compatibility (graph captures buffer layout)
        meta = self._kv_cache.get_attention_metadata([seq_id])
        bt = meta.block_tables[0]
        max_blocks = self._graph_max_blocks
        padded = bt + [-1] * (max_blocks - len(bt))
        bt_data = struct.pack(f'<{max_blocks}i', *padded)
        self._gpu_mem.copy_host_to_device(bt_data, self._decode_bt_buf)
        sl_data = struct.pack(f'<{len(meta.seq_lengths)}i', *meta.seq_lengths)
        self._gpu_mem.copy_host_to_device(sl_data, self._decode_seqlens_buf)

        # Override meta.max_blocks_per_seq with fixed value for graph
        meta.max_blocks_per_seq = max_blocks

        if not self._graph_captured:
            # Capture phase — run all kernels on capture stream
            self._graph_runner.begin_capture()
            self._run_decode_kernels(seq_id, meta, stream=self._graph_stream)
            self._graph_runner.end_capture()
            self._graph_captured = True

            # Sync to get the capture result
            self._graph_runner.sync()
        else:
            # Replay phase — one call replays all ~960 kernels
            self._graph_runner.replay()
            self._graph_runner.sync()

        # Download argmax result (4 bytes)
        view = Tensor(shape=(1,), dtype=int32)
        view._data_ptr = self._graph_argmax_buf.data_ptr
        view._nbytes = 4
        result_bytes = self._gpu_mem.copy_device_to_host(view)
        return struct.unpack('<i', result_bytes[:4])[0]

    def destroy_graph(self):
        """Release graph resources."""
        if self._graph_runner is not None:
            self._graph_runner.destroy()
            self._graph_runner = None
            self._graph_captured = False
        if self._graph_argmax_buf is not None:
            self._gpu_mem.free(self._graph_argmax_buf)
            self._graph_argmax_buf = None
        self._batched_graph_runners = {}

    def batched_decode_graph(
        self,
        token_ids: List[int],
        seq_ids: List[int],
        positions: List[int],
    ) -> List[int]:
        """Graph-captured batched decode. Returns M token IDs (greedy only).

        Captures a graph for batch size M on first call, replays on subsequent calls.
        Different batch sizes get separate captured graphs.
        """
        M = len(token_ids)
        if M == 0:
            return []

        # CPU-side: extend KV cache
        for sid in seq_ids:
            self._kv_cache.extend_sequence(sid, num_new_tokens=1)

        # Upload changing values to GPU buffers
        token_data = struct.pack(f'<{M}i', *token_ids)
        self._gpu_mem.copy_host_to_device(token_data, self._decode_token_buf)
        pos_data = struct.pack(f'<{M}i', *positions)
        self._gpu_mem.copy_host_to_device(pos_data, self._decode_pos_buf)

        # Build attention metadata with FIXED max_blocks for graph
        meta = self._kv_cache.get_attention_metadata(seq_ids)
        max_blocks = self._graph_max_blocks
        # Pack block tables with fixed padding
        bt_parts = []
        for bt in meta.block_tables:
            padded = bt + [-1] * (max_blocks - len(bt))
            bt_parts.extend(padded)
        bt_data = struct.pack(f'<{len(bt_parts)}i', *bt_parts)
        self._gpu_mem.copy_host_to_device(bt_data, self._decode_bt_buf)
        sl_data = struct.pack(f'<{M}i', *meta.seq_lengths)
        self._gpu_mem.copy_host_to_device(sl_data, self._decode_seqlens_buf)
        meta.max_blocks_per_seq = max_blocks

        # Check if we have a graph for this batch size
        if not hasattr(self, '_batched_graph_runners'):
            self._batched_graph_runners = {}

        if M not in self._batched_graph_runners:
            # Capture a new graph for batch size M
            self._capture_batched_graph(M, meta)
            # Opportunistically pre-capture all smaller batch sizes too, using
            # the first M' seq_ids from this batch. This eliminates the recapture
            # stall when sequences finish mid-stream and M shrinks.
            for sub_m in range(M - 1, 0, -1):
                if sub_m in self._batched_graph_runners:
                    continue
                sub_meta = self._kv_cache.get_attention_metadata(seq_ids[:sub_m])
                sub_meta.max_blocks_per_seq = max_blocks
                self._capture_batched_graph(sub_m, sub_meta)

        # Replay (always — captured fresh above or hit cache)
        gr, stream = self._batched_graph_runners[M]
        gr.replay()
        gr.sync()

        # Bulk download all M argmax results in ONE DtoH transfer (4*M bytes)
        # Previously did M separate 4-byte copies = M * ~50us ctypes overhead.
        view = Tensor(shape=(M,), dtype=int32)
        view._data_ptr = self._graph_argmax_buf.data_ptr
        view._nbytes = M * 4
        result_bytes = self._gpu_mem.copy_device_to_host(view)
        return list(struct.unpack(f'<{M}i', result_bytes[:M * 4]))

    def _capture_batched_graph(self, M: int, meta):
        """Capture a graph for batch size M. Called lazily or via precapture_batched_graphs()."""
        from zse_engine.orchestrator.hip_graph import HIPGraphRunner, CUDAGraphRunner
        if self._backend == "rocm":
            gr = HIPGraphRunner(self._gpu_mem._driver)
        else:
            gr = CUDAGraphRunner(self._gpu_mem._driver)
        stream = gr.create_stream()

        gr.begin_capture()
        self._run_batched_decode_kernels(M, meta, stream=stream)
        gr.end_capture()
        gr.sync()

        self._batched_graph_runners[M] = (gr, stream)

    def precapture_batched_graphs(self, batch_sizes=(1, 2, 4, 8), warmup_seq_ids=None):
        """Pre-capture graphs for common batch sizes during warmup.

        Without this, the first time we see a new batch size mid-serving we stall
        ~50-150ms capturing the graph. Pre-capturing during init makes transitions
        between batch sizes free (e.g. when one of 4 concurrent requests finishes
        and the next decode step needs an M=3 graph).

        Requires `warmup_seq_ids` (list of real allocated seq IDs) to build a valid
        attention metadata for capture. The captured graph is reusable for any
        future M with that batch size.
        """
        if not warmup_seq_ids:
            return
        max_bs = len(warmup_seq_ids)
        for bs in batch_sizes:
            if bs > max_bs or bs in self._batched_graph_runners:
                continue
            ids = warmup_seq_ids[:bs]
            meta = self._kv_cache.get_attention_metadata(ids)
            meta.max_blocks_per_seq = self._graph_max_blocks
            self._capture_batched_graph(bs, meta)

    def _run_batched_decode_kernels(self, M: int, meta, stream=None):
        """Run all batched decode kernels on stream (for graph capture)."""
        # Embedding lookup
        embed_weight = self._weights.get("embed_tokens.weight")
        self._kernels.launch(
            "embedding_lookup_f32out",
            ((M * self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32,
            self._make_tensor_from_ptr(
                embed_weight.data_ptr, (self._config.vocab_size, self._hidden_size)),
            self._decode_token_buf, self._hidden_size, M,
            stream=stream,
        )

        kv_slab = self._kv_slab

        # Transformer layers
        for layer in range(self._num_layers):
            self._batched_transformer_block_decode_graph(
                layer, M, kv_slab, meta, stream=stream,
            )

        # Final RMSNorm
        final_norm_w = self._get_layer_weight("model.norm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            final_norm_w, M, stream=stream,
        )

        # LM head
        lm_head = self._weights.get("lm_head.weight")
        self._launch_matmul(
            self._scratch.logits, self._scratch.norm_out, lm_head,
            M, self._config.vocab_size, self._hidden_size, stream=stream,
        )

        # GPU argmax — single kernel launch for all M rows
        vocab = self._config.vocab_size
        self._kernels.launch(
            "batched_argmax_fp16",
            (M,),
            (256,),
            self._graph_argmax_buf, self._scratch.logits, M, vocab,
            stream=stream,
        )

    def _batched_transformer_block_decode_graph(
        self, layer: int, M: int, kv_slab: Tensor, meta, stream=None,
    ):
        """One transformer layer for batched decode with stream support."""
        n = M * self._hidden_size
        q_dim = self._num_heads * self._head_dim
        k_dim = self._num_kv_heads * self._head_dim

        # Save residual
        self._copy_tensor_f32(self._scratch.residual_f32, self._scratch.hidden_f32, n,
                              stream=stream)

        # Pre-attention RMSNorm
        attn_norm_w = self._get_layer_weight(f"model.layers.{layer}.input_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            attn_norm_w, M, stream=stream,
        )

        # QKV projection
        q_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.q_proj.weight")
        k_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.k_proj.weight")
        v_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.v_proj.weight")

        q_buf = self._scratch.qkv
        k_buf = self._scratch.attn_out
        v_buf = self._scratch.mlp_out

        self._launch_matmul_with_lora(q_buf, self._scratch.norm_out, q_proj,
                                      M, q_dim, self._hidden_size, stream=stream)
        self._launch_matmul_with_lora(k_buf, self._scratch.norm_out, k_proj,
                                      M, k_dim, self._hidden_size, stream=stream)
        self._launch_matmul_with_lora(v_buf, self._scratch.norm_out, v_proj,
                                      M, k_dim, self._hidden_size, stream=stream)

        # QKV biases
        self._apply_bias_if_present(q_buf, layer, "self_attn.q_proj.bias", M, q_dim,
                                    stream=stream)
        self._apply_bias_if_present(k_buf, layer, "self_attn.k_proj.bias", M, k_dim,
                                    stream=stream)
        self._apply_bias_if_present(v_buf, layer, "self_attn.v_proj.bias", M, k_dim,
                                    stream=stream)

        # Batched RoPE (reads positions from _decode_pos_buf)
        half_dim = self._head_dim // 2
        total_rope = M * max(self._num_heads, self._num_kv_heads) * half_dim
        self._kernels.launch(
            "batched_rotary_embedding",
            ((total_rope + 255) // 256,),
            (256,),
            q_buf, k_buf, self._decode_pos_buf,
            M, self._num_heads, self._num_kv_heads,
            self._head_dim, self._rope_theta,
            stream=stream,
        )

        # KV cache write — BATCHED (single launch for all M sequences)
        total_kv = self._num_kv_heads * self._head_dim
        self._kernels.launch(
            "batched_kv_cache_write",
            ((total_kv + 255) // 256, M),
            (256,),
            kv_slab, k_buf, v_buf,
            self._decode_bt_buf, self._decode_pos_buf,
            M, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            self._graph_max_blocks, self._num_layers, layer,
            stream=stream,
        )

        # Paged attention — batched, max shared mem for graph
        self._kernels.launch(
            "paged_attention",
            (M, self._num_heads),
            (min(256, self._head_dim),),
            self._scratch.attn_out, q_buf, kv_slab,
            self._decode_bt_buf, self._decode_seqlens_buf,
            self._num_heads, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            self._graph_max_blocks, self._num_layers, layer,
            self._scale,
            shared_mem_bytes=self._graph_max_shared_mem,
            stream=stream,
        )

        # O projection
        o_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.o_proj.weight")
        self._launch_matmul_with_lora(self._scratch.hidden, self._scratch.attn_out, o_proj,
                                      M, self._hidden_size, q_dim, stream=stream)

        # Fused residual + RMSNorm
        mlp_norm_w = self._get_layer_weight(f"model.layers.{layer}.post_attention_layernorm.weight")
        self._launch_fused_residual_rmsnorm_f32(
            self._scratch.norm_out, self._scratch.residual_f32,
            self._scratch.hidden, self._scratch.residual_f32, mlp_norm_w, M,
            stream=stream,
        )
        self._copy_tensor_f32(self._scratch.hidden_f32, self._scratch.residual_f32, n,
                              stream=stream)

        # MLP
        gate_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.gate_proj.weight")
        up_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.up_proj.weight")
        down_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.down_proj.weight")

        self._launch_matmul_with_lora(self._scratch.mlp_gate, self._scratch.norm_out, gate_proj,
                                      M, self._intermediate_size, self._hidden_size, stream=stream)
        self._launch_matmul_with_lora(self._scratch.mlp_up, self._scratch.norm_out, up_proj,
                                      M, self._intermediate_size, self._hidden_size, stream=stream)

        n_mlp = M * self._intermediate_size
        self._kernels.launch(
            "silu_mul",
            ((n_mlp + 255) // 256,),
            (256,),
            self._scratch.mlp_gate, self._scratch.mlp_gate, self._scratch.mlp_up, n_mlp,
            stream=stream,
        )

        self._launch_matmul_with_lora(self._scratch.hidden, self._scratch.mlp_gate, down_proj,
                                      M, self._hidden_size, self._intermediate_size, stream=stream)

        # MLP residual
        self._kernels.launch(
            "residual_add_f32",
            ((n + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.hidden_f32, self._scratch.hidden, n,
            stream=stream,
        )

    def _launch_rmsnorm(self, out: Tensor, inp: Tensor, weight: GPUWeight,
                        num_rows: int):
        """Launch RMSNorm kernel (fp16 input)."""
        block_size = min(256, self._hidden_size)
        if block_size < 32:
            block_size = 32

        weight_tensor = self._make_tensor_from_ptr(
            weight.data_ptr, (self._hidden_size,))
        self._kernels.launch(
            "rmsnorm",
            (num_rows,),
            (block_size,),
            out, inp, weight_tensor, self._hidden_size, self._rms_eps,
        )

    def _launch_rmsnorm_f32in(self, out: Tensor, inp_f32: Tensor, weight: GPUWeight,
                               num_rows: int, stream=None):
        """Launch RMSNorm with fp32 input → fp16 output."""
        block_size = min(256, self._hidden_size)
        if block_size < 32:
            block_size = 32

        weight_tensor = self._make_tensor_from_ptr(
            weight.data_ptr, (self._hidden_size,))
        self._kernels.launch(
            "rmsnorm_f32in",
            (num_rows,),
            (block_size,),
            out, inp_f32, weight_tensor, self._hidden_size, self._rms_eps,
            stream=stream,
        )

    def _launch_fused_residual_rmsnorm(self, norm_out: Tensor, residual: Tensor,
                                        a: Tensor, b: Tensor, weight: GPUWeight,
                                        num_rows: int):
        """Launch fused residual_add + RMSNorm (fp16 version, legacy)."""
        block_size = min(256, self._hidden_size)
        if block_size < 32:
            block_size = 32

        weight_tensor = self._make_tensor_from_ptr(
            weight.data_ptr, (self._hidden_size,))
        self._kernels.launch(
            "fused_residual_rmsnorm",
            (num_rows,),
            (block_size,),
            norm_out, residual, a, b, weight_tensor,
            self._hidden_size, self._rms_eps,
        )

    def _launch_fused_residual_rmsnorm_f32(self, norm_out: Tensor, residual_f32: Tensor,
                                            a_fp16: Tensor, b_f32: Tensor, weight: GPUWeight,
                                            num_rows: int, stream=None):
        """Fused: residual_f32 = fp16_a + fp32_b; norm_out = RMSNorm(residual_f32) → fp16."""
        block_size = min(256, self._hidden_size)
        if block_size < 32:
            block_size = 32

        weight_tensor = self._make_tensor_from_ptr(
            weight.data_ptr, (self._hidden_size,))
        self._kernels.launch(
            "fused_residual_rmsnorm_f32",
            (num_rows,),
            (block_size,),
            norm_out, residual_f32, a_fp16, b_f32, weight_tensor,
            self._hidden_size, self._rms_eps,
            stream=stream,
        )

    def _launch_matmul(self, out: Tensor, inp: Tensor, weight: GPUWeight,
                       M: int, N: int, K: int, stream=None):
        """Launch appropriate matmul kernel based on weight dtype.

        For M=1 (single-token decode), uses GEMV kernels optimized for
        memory-bandwidth-bound matrix-vector multiply (~15x faster than tiled GEMM).
        For M>1 (prefill/batched), uses tiled GEMM kernels.
        """
        use_gemv = (M == 1)
        # For small batch (M<=8), use GEMV per row (faster than tiled GEMM for small M)
        use_multi_gemv = (M > 1 and M <= 8)
        # GEMV: 8 rows per block. CUDA = 8 warps of 32 = 256 threads.
        # ROCm = 8 wavefronts of 64 = 512 threads.
        gemv_rpb = 8  # 8 rows per block on both CUDA and ROCm
        gemv_block = 512 if self._backend == "rocm" else 256

        if weight.dtype == "int4":
            weight_t = self._make_tensor_from_ptr(weight.data_ptr, (N, K // 2))
            scales_t = self._make_tensor_from_ptr(weight.scales_ptr, (N, K // weight.group_size))
            zeros_t = self._make_tensor_from_ptr(weight.zeros_ptr, (N, K // weight.group_size))
            if use_gemv:
                self._kernels.launch(
                    "dequant_gemv_int4",
                    ((N + gemv_rpb - 1) // gemv_rpb,),
                    (gemv_block,),
                    out, weight_t, scales_t, zeros_t, inp,
                    N, K, weight.group_size,
                    stream=stream,
                )
            elif use_multi_gemv:
                # ROCm: use portable wave-64 kernel (2.13x faster than C-string bgemv at M=4).
                # Constraints: K%8==0 (u32-aligned), group_size>=8 (lane K-window fits a group).
                use_wave64_bgemv = (
                    self._backend == "rocm"
                    and (K % 8 == 0)
                    and (weight.group_size >= 8)
                )
                # 128-bit (uint4) variant: tighter constraints, ~1.10x avg over wave-64.
                use_wave64_v2 = (
                    use_wave64_bgemv
                    and (K % 32 == 0)
                    and (weight.group_size % 32 == 0)
                )
                if use_wave64_v2:
                    self._kernels.launch(
                        "bgemv_int4_wave64_v2",
                        ((N + 7) // 8, 1, 1),
                        (512, 1, 1),
                        out, weight_t, scales_t, zeros_t, inp,
                        M, N, K, weight.group_size,
                        stream=stream,
                    )
                elif use_wave64_bgemv:
                    self._kernels.launch(
                        "bgemv_int4_wave64",
                        ((N + 7) // 8, 1, 1),
                        (512, 1, 1),
                        out, weight_t, scales_t, zeros_t, inp,
                        M, N, K, weight.group_size,
                        stream=stream,
                    )
                else:
                    # NVIDIA path. The hand-written C-string bgemv remains the
                    # production default — the portable warp-32 GEMV is parity-
                    # proven but bandwidth-bound (≈parity, slight regression on
                    # wide-N shapes), so it is NOT activated here. Kept in the
                    # kernel registry as an IP-pure scaffold for a future WMMA
                    # (tensor-core) small-M variant. Flip `use_wave32_bgemv` to
                    # True only once a faster portable kernel exists.
                    use_wave32_bgemv = False
                    if use_wave32_bgemv:
                        self._kernels.launch(
                            "bgemv_int4_wave32",
                            ((N + 7) // 8,),  # 8 N-rows per block
                            (256,),           # 8 warps × 32 lanes
                            out, weight_t, scales_t, zeros_t, inp,
                            M, N, K, weight.group_size,
                            stream=stream,
                        )
                    else:
                        # Batched GEMV: reads weight ONCE, computes M dot products.
                        self._kernels.launch(
                            "batched_dequant_gemv_int4",
                            ((N + 8 - 1) // 8,),  # BGEMV_RPB=8
                            (gemv_block,),
                            out, weight_t, scales_t, zeros_t, inp,
                            M, N, K, weight.group_size,
                            stream=stream,
                        )
            else:
                # ROCm/CDNA: use MFMA-accelerated v3 kernel when shapes are compatible.
                # Constraints: K%64==0 (CHUNK_K=64) and group_size%16==0 (lane-window assumption).
                # All real .zse models satisfy this (Qwen2.5-32B: K=5120/27648, gs=128).
                use_mfma_v3 = (
                    self._backend == "rocm"
                    and (K % 64 == 0)
                    and (weight.group_size % 16 == 0)
                )
                if use_mfma_v3:
                    self._kernels.launch(
                        "mfma_dequant_matmul_int4_v3",
                        ((N + 15) // 16, (M + 15) // 16),
                        (64, 1, 1),
                        out, weight_t, scales_t, zeros_t, inp,
                        M, N, K, weight.group_size,
                        stream=stream,
                    )
                else:
                    self._kernels.launch(
                        "tiled_dequant_matmul_int4",
                        ((N + 31) // 32, (M + 31) // 32),
                        (32, 32),
                        out, weight_t, scales_t, zeros_t, inp,
                        M, N, K, weight.group_size,
                        stream=stream,
                    )
        elif weight.dtype == "int8":
            weight_t = self._make_tensor_from_ptr(weight.data_ptr, (N, K))
            scales_t = self._make_tensor_from_ptr(weight.scales_ptr, (N, K // weight.group_size))
            self._kernels.launch(
                "tiled_dequant_matmul_int8",
                ((N + 31) // 32, (M + 31) // 32),
                (32, 32),
                out, weight_t, scales_t, inp,
                M, N, K, weight.group_size,
                stream=stream,
            )
        elif weight.dtype == "float16":
            weight_t = self._make_tensor_from_ptr(weight.data_ptr, (N, K))
            if use_gemv:
                self._kernels.launch(
                    "fp16_gemv",
                    ((N + gemv_rpb - 1) // gemv_rpb,),
                    (gemv_block,),
                    out, weight_t, inp,
                    N, K,
                    stream=stream,
                )
            else:
                self._kernels.launch(
                    "tiled_fp16_matmul",
                    ((N + 31) // 32, (M + 31) // 32),
                    (32, 32),
                    out, inp, weight_t, M, N, K,
                    stream=stream,
                )
        else:
            raise ValueError(f"Unsupported weight dtype: {weight.dtype}")

    def _launch_gemv_cached(self, out: Tensor, inp: Tensor, layer: int,
                            proj_name: str, N: int, K: int, stream=None):
        """Fast GEMV using pre-cached tensor wrappers. Zero allocation per call.

        Only for INT4 decode (M=1). Falls back to _launch_matmul for other types.
        """
        cache_key_data = f"{proj_name}.data"
        lc = self._decode_tensor_cache[layer]

        if cache_key_data not in lc:
            # Fallback to regular path
            weight = self._get_layer_weight(f"model.layers.{layer}.{proj_name}")
            self._launch_matmul(out, inp, weight, 1, N, K, stream=stream)
            return

        weight_t = lc[cache_key_data]
        scales_t = lc[f"{proj_name}.scales"]
        zeros_t = lc[f"{proj_name}.zeros"]
        w = self._layer_weights[layer][proj_name]
        gemv_rpb = 8  # 8 rows per block on both CUDA and ROCm
        gemv_block = 512 if self._backend == "rocm" else 256

        self._kernels.launch(
            "dequant_gemv_int4",
            ((N + gemv_rpb - 1) // gemv_rpb,),
            (gemv_block,),
            out, weight_t, scales_t, zeros_t, inp,
            N, K, w.group_size,
            stream=stream,
        )

    def _launch_matmul_with_lora(
        self, out: Tensor, inp: Tensor, weight: GPUWeight,
        M: int, N: int, K: int,
        lora_adapter=None, layer_idx: int = 0, module_name: str = "",
        stream=None,
    ):
        """Launch matmul with optional LoRA delta.

        Computes: out = W @ x + (α/r) * B @ (A @ x)
        If no LoRA adapter or no weight for this module, same as _launch_matmul.
        """
        # Base matmul
        self._launch_matmul(out, inp, weight, M, N, K, stream=stream)

        # Apply LoRA delta if applicable
        if (lora_adapter is not None and self._lora_manager is not None
                and lora_adapter.has_weight(layer_idx, module_name)):
            self._lora_manager.apply_lora(
                out, inp, lora_adapter, layer_idx, module_name, M, N, K,
            )

    def _upload_int32(self, values: List[int]) -> Tensor:
        """Upload int32 values to GPU."""
        data = struct.pack(f'<{len(values)}i', *values)
        tensor = self._gpu_mem.allocate((len(values),), int32)
        self._gpu_mem.copy_host_to_device(data, tensor)
        return tensor

    def _upload_block_table(self, meta, seq_id: int = 0) -> Tensor:
        """Upload block table for a sequence to GPU."""
        bt = meta.block_tables[seq_id]
        padded = bt + [-1] * (meta.max_blocks_per_seq - len(bt))
        data = struct.pack(f'<{len(padded)}i', *padded)
        tensor = self._gpu_mem.allocate((len(padded),), int32)
        self._gpu_mem.copy_host_to_device(data, tensor)
        return tensor

    def _download_fp16(self, tensor: Tensor, num_elements: int,
                       row_offset: int = 0) -> bytes:
        """Download fp16 data from GPU."""
        # Create a view tensor at the right offset
        byte_offset = row_offset * num_elements * 2
        nbytes = num_elements * 2
        view = Tensor(shape=(num_elements,), dtype=float16)
        view._data_ptr = tensor.data_ptr + byte_offset
        view._nbytes = nbytes
        return self._gpu_mem.copy_device_to_host(view)

    def _make_tensor_from_ptr(self, ptr: int, shape: tuple) -> Tensor:
        """Create a Tensor wrapper around an existing GPU pointer."""
        t = Tensor(shape=shape, dtype=float16)
        t._data_ptr = ptr
        return t

    def _copy_tensor(self, dst: Tensor, src: Tensor, num_elements: int):
        """Copy fp16 data between GPU tensors."""
        nbytes = num_elements * 2  # fp16
        self._gpu_mem.copy_device_to_device(src.data_ptr, dst.data_ptr, nbytes)

    def _copy_tensor_f32(self, dst: Tensor, src: Tensor, num_elements: int, stream=None):
        """Copy fp32 data between GPU tensors."""
        nbytes = num_elements * 4  # fp32
        if stream is not None:
            self._gpu_mem.copy_device_to_device_async(src.data_ptr, dst.data_ptr, nbytes, stream)
        else:
            self._gpu_mem.copy_device_to_device(src.data_ptr, dst.data_ptr, nbytes)

    def _get_kv_slab_tensor(self) -> Tensor:
        """Get a Tensor wrapper for the KV cache GPU slab."""
        pool = self._kv_cache._pool
        t = Tensor(shape=(pool.total_bytes,))
        t._data_ptr = pool._gpu_base_ptr
        return t


class TPModelRunner(ModelRunner):
    """Tensor-parallel model runner — sharded forward pass with NCCL all-reduce.

    Subclasses ModelRunner and overrides the transformer block methods to insert
    all-reduce operations at the correct points:
    - After O projection (row parallel) → all-reduce before residual add
    - After Down projection (row parallel) → all-reduce before residual add

    Each GPU processes its local shard of Q/K/V/Gate/Up heads, then combines
    partial sums via all-reduce for O and Down projections.

    The local dimensions are already adjusted by the config overrides:
    - num_heads → num_heads // tp_size
    - num_kv_heads → num_kv_heads // tp_size
    - intermediate_size → intermediate_size // tp_size
    """

    def __init__(
        self,
        config,
        weights,
        kv_cache,
        scratch,
        gpu_mem,
        kernels,
        tp_group,
        lora_manager=None,
    ):
        # Adjust config dimensions for this TP rank before parent init
        # The weights are already sharded by TPWeightLoader, but the ModelRunner
        # needs to know the local dimensions for kernel launches
        self._tp_group = tp_group
        self._tp_size = tp_group.tp_size
        self._tp_rank = tp_group.rank

        # Store original full-model dimensions for all-reduce sizing
        self._full_hidden_size = config.hidden_size
        self._full_num_heads = config.num_heads
        self._full_num_kv_heads = config.num_kv_heads
        self._full_intermediate_size = config.intermediate_size

        # Create a modified config with local dimensions
        # (Don't mutate the original — other components may need full dims)
        from copy import copy
        local_config = copy(config)
        local_config.num_heads = config.num_heads // tp_group.tp_size
        local_config.num_kv_heads = config.num_kv_heads // tp_group.tp_size
        local_config.intermediate_size = config.intermediate_size // tp_group.tp_size
        # hidden_size stays the same — it's the full residual stream width

        super().__init__(
            config=local_config,
            weights=weights,
            kv_cache=kv_cache,
            scratch=scratch,
            gpu_mem=gpu_mem,
            kernels=kernels,
            lora_manager=lora_manager,
        )

        # Allocate scratch buffer for all-reduce (reuse hidden buffer if possible)
        # All-reduce operates in-place on hidden_size fp16 elements
        self._allreduce_count = self._full_hidden_size  # Elements, not bytes

    def _transformer_block_decode(self, layer, seq_id, position,
                                   kv_slab, block_table_tensor,
                                   seq_lens_tensor, meta,
                                   lora_adapter=None):
        """One transformer layer during decode with TP all-reduce.

        Same as parent, but inserts:
        1. all-reduce after O projection (before attention residual add)
        2. all-reduce after Down projection (before MLP residual add)
        """
        seq_len = 1

        # Save residual (fp32)
        self._copy_tensor_f32(self._scratch.residual_f32, self._scratch.hidden_f32, self._hidden_size)

        # Pre-attention RMSNorm — fp32→fp16 (replicated, same on all ranks)
        attn_norm_w = self._get_layer_weight(f"model.layers.{layer}.input_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            attn_norm_w, seq_len,
        )

        # QKV projection — column parallel (local heads only)
        q_dim = self._num_heads * self._head_dim  # local q_dim
        k_dim = self._num_kv_heads * self._head_dim  # local k_dim

        q_buf = self._scratch.qkv
        k_buf = self._scratch.attn_out
        v_buf = self._scratch.mlp_out

        self._launch_gemv_cached(q_buf, self._scratch.norm_out, layer,
                                 "self_attn.q_proj.weight", q_dim, self._hidden_size)
        self._launch_gemv_cached(k_buf, self._scratch.norm_out, layer,
                                 "self_attn.k_proj.weight", k_dim, self._hidden_size)
        self._launch_gemv_cached(v_buf, self._scratch.norm_out, layer,
                                 "self_attn.v_proj.weight", k_dim, self._hidden_size)

        # Biases (also sharded for column parallel QKV)
        self._apply_bias_if_present(q_buf, layer, "self_attn.q_proj.bias", 1, q_dim)
        self._apply_bias_if_present(k_buf, layer, "self_attn.k_proj.bias", 1, k_dim)
        self._apply_bias_if_present(v_buf, layer, "self_attn.v_proj.bias", 1, k_dim)

        # RoPE (local heads)
        half_dim = self._head_dim // 2
        total_rope = max(self._num_heads, self._num_kv_heads) * half_dim
        self._kernels.launch(
            "rotary_embedding",
            ((total_rope + 255) // 256,),
            (256,),
            q_buf, k_buf,
            1, self._num_heads, self._num_kv_heads,
            self._head_dim, self._decode_pos_buf, self._rope_theta,
        )

        # KV cache write (local heads)
        total_kv = self._num_kv_heads * self._head_dim
        self._kernels.launch(
            "kv_cache_write",
            ((total_kv + 255) // 256,),
            (256,),
            kv_slab, k_buf, v_buf, block_table_tensor,
            1, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer, self._decode_pos_buf,
        )

        # Paged attention (local heads)
        total_tokens = meta.seq_lengths[0]
        shared_mem = total_tokens * 4
        self._kernels.launch(
            "paged_attention",
            (1, self._num_heads),
            (min(256, self._head_dim),),
            self._scratch.attn_out, q_buf, kv_slab,
            block_table_tensor, seq_lens_tensor,
            self._num_heads, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer,
            self._scale,
            shared_mem_bytes=shared_mem,
        )

        # O projection — row parallel (input is local attn output, output is partial hidden)
        o_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.o_proj.weight")
        self._launch_matmul(
            self._scratch.hidden, self._scratch.attn_out, o_proj,
            1, self._hidden_size, q_dim,  # K = local q_dim (sharded input)
        )

        # *** ALL-REDUCE after O projection ***
        # Each rank has a partial sum of hidden_size elements. Sum across ranks.
        self._tp_group.all_reduce_inplace(
            self._scratch.hidden.data_ptr,
            self._hidden_size,
            dtype="float16",
        )

        # Residual: hidden_f32 = residual_f32 + fp16 hidden (now complete after all-reduce)
        self._kernels.launch(
            "residual_add_f32",
            ((self._hidden_size + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.residual_f32, self._scratch.hidden,
            self._hidden_size,
        )

        # Post-attention RMSNorm — fp32→fp16 (replicated)
        mlp_norm_w = self._get_layer_weight(f"model.layers.{layer}.post_attention_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            mlp_norm_w, 1,
        )

        # Gate + Up — column parallel (local intermediate shard)
        gate_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.gate_proj.weight")
        up_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.up_proj.weight")
        down_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.down_proj.weight")

        use_fused_gemv = (gate_proj.dtype == "int4" and lora_adapter is None)
        if use_fused_gemv:
            N = self._intermediate_size  # local shard
            K = self._hidden_size
            gate_wt = self._make_tensor_from_ptr(gate_proj.data_ptr, (N, K // 2))
            gate_sc = self._make_tensor_from_ptr(gate_proj.scales_ptr, (N, K // gate_proj.group_size))
            gate_zr = self._make_tensor_from_ptr(gate_proj.zeros_ptr, (N, K // gate_proj.group_size))
            up_wt = self._make_tensor_from_ptr(up_proj.data_ptr, (N, K // 2))
            up_sc = self._make_tensor_from_ptr(up_proj.scales_ptr, (N, K // up_proj.group_size))
            up_zr = self._make_tensor_from_ptr(up_proj.zeros_ptr, (N, K // up_proj.group_size))
            fgu_rpb = 2 if self._backend == "rocm" else 4
            self._kernels.launch(
                "fused_gate_up_gemv_int4",
                ((N + fgu_rpb - 1) // fgu_rpb,),
                (256,),
                self._scratch.mlp_gate, self._scratch.mlp_up,
                gate_wt, gate_sc, gate_zr,
                up_wt, up_sc, up_zr,
                self._scratch.norm_out,
                N, K, gate_proj.group_size,
            )
        else:
            self._launch_matmul(self._scratch.mlp_gate, self._scratch.norm_out, gate_proj,
                                1, self._intermediate_size, self._hidden_size)
            self._launch_matmul(self._scratch.mlp_up, self._scratch.norm_out, up_proj,
                                1, self._intermediate_size, self._hidden_size)

        # SiLU (local intermediate shard)
        n_mlp = self._intermediate_size
        self._kernels.launch(
            "silu_mul",
            ((n_mlp + 255) // 256,),
            (256,),
            self._scratch.mlp_gate, self._scratch.mlp_gate, self._scratch.mlp_up, n_mlp,
        )

        # Down projection — row parallel (input is local intermediate, output is partial hidden)
        self._launch_matmul(
            self._scratch.hidden, self._scratch.mlp_gate, down_proj,
            1, self._hidden_size, self._intermediate_size,  # K = local intermediate (sharded)
        )

        # *** ALL-REDUCE after Down projection ***
        self._tp_group.all_reduce_inplace(
            self._scratch.hidden.data_ptr,
            self._hidden_size,
            dtype="float16",
        )

        # MLP residual: hidden_f32 += fp16 hidden (now complete after all-reduce)
        n_res = self._hidden_size
        self._kernels.launch(
            "residual_add_f32",
            ((n_res + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.hidden_f32, self._scratch.hidden,
            n_res,
        )

    def _transformer_block_prefill(self, layer, seq_id, seq_len,
                                    kv_slab, block_table_tensor,
                                    seq_lens_tensor, meta,
                                    lora_adapter=None):
        """Prefill transformer block with TP all-reduce.

        Same structure as decode but with seq_len > 1 and matmul instead of GEMV.
        All-reduce points are identical: after O proj and after Down proj.
        """
        # For prefill, delegate to parent but intercept at the right points.
        # Since the parent method is complex, we call it and rely on the fact
        # that the sharded weights produce correct partial results.
        # However, we need to insert all-reduce, so we must override fully.

        # Save residual
        self._copy_tensor_f32(self._scratch.residual_f32, self._scratch.hidden_f32,
                             self._hidden_size * seq_len)

        # Pre-attention RMSNorm
        attn_norm_w = self._get_layer_weight(f"model.layers.{layer}.input_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            attn_norm_w, seq_len,
        )

        # QKV projections — column parallel
        q_dim = self._num_heads * self._head_dim
        k_dim = self._num_kv_heads * self._head_dim

        q_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.q_proj.weight")
        k_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.k_proj.weight")
        v_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.v_proj.weight")

        self._launch_matmul(self._scratch.qkv, self._scratch.norm_out, q_proj,
                           seq_len, q_dim, self._hidden_size)
        self._launch_matmul(self._scratch.attn_out, self._scratch.norm_out, k_proj,
                           seq_len, k_dim, self._hidden_size)
        self._launch_matmul(self._scratch.mlp_out, self._scratch.norm_out, v_proj,
                           seq_len, k_dim, self._hidden_size)

        # Biases
        self._apply_bias_if_present(self._scratch.qkv, layer, "self_attn.q_proj.bias", seq_len, q_dim)
        self._apply_bias_if_present(self._scratch.attn_out, layer, "self_attn.k_proj.bias", seq_len, k_dim)
        self._apply_bias_if_present(self._scratch.mlp_out, layer, "self_attn.v_proj.bias", seq_len, k_dim)

        # RoPE
        half_dim = self._head_dim // 2
        total_rope = seq_len * max(self._num_heads, self._num_kv_heads) * half_dim
        self._kernels.launch(
            "rotary_embedding",
            ((total_rope + 255) // 256,),
            (256,),
            self._scratch.qkv, self._scratch.attn_out,
            seq_len, self._num_heads, self._num_kv_heads,
            self._head_dim, self._prefill_pos_zero_buf, self._rope_theta,
        )

        # KV cache write
        total_kv = seq_len * self._num_kv_heads * self._head_dim
        self._kernels.launch(
            "kv_cache_write",
            ((total_kv + 255) // 256,),
            (256,),
            kv_slab, self._scratch.attn_out, self._scratch.mlp_out,
            block_table_tensor,
            seq_len, self._num_kv_heads, self._head_dim,
            self._kv_cache.block_size,
            meta.max_blocks_per_seq, self._num_layers, layer, self._prefill_pos_zero_buf,
        )

        # Prefill attention (local heads)
        if self._kernels.has_kernel("prefill_attention"):
            self._kernels.launch(
                "prefill_attention",
                (seq_len, self._num_heads),
                (min(256, self._head_dim),),
                self._scratch.attn_out, self._scratch.qkv, self._scratch.attn_out,
                self._scratch.mlp_out,
                seq_len, self._num_heads, self._num_kv_heads, self._head_dim,
                self._scale,
            )
        else:
            # Fallback to paged attention
            total_tokens = meta.seq_lengths[0] if meta.seq_lengths else seq_len
            shared_mem = total_tokens * 4
            self._kernels.launch(
                "paged_attention",
                (seq_len, self._num_heads),
                (min(256, self._head_dim),),
                self._scratch.attn_out, self._scratch.qkv, kv_slab,
                block_table_tensor, seq_lens_tensor,
                self._num_heads, self._num_kv_heads, self._head_dim,
                self._kv_cache.block_size,
                meta.max_blocks_per_seq, self._num_layers, layer,
                self._scale,
                shared_mem_bytes=shared_mem,
            )

        # O projection — row parallel
        o_proj = self._get_layer_weight(f"model.layers.{layer}.self_attn.o_proj.weight")
        self._launch_matmul(
            self._scratch.hidden, self._scratch.attn_out, o_proj,
            seq_len, self._hidden_size, q_dim,
        )

        # *** ALL-REDUCE after O projection ***
        self._tp_group.all_reduce_inplace(
            self._scratch.hidden.data_ptr,
            self._hidden_size * seq_len,
            dtype="float16",
        )

        # Attention residual
        total_res = self._hidden_size * seq_len
        self._kernels.launch(
            "residual_add_f32",
            ((total_res + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.residual_f32, self._scratch.hidden,
            total_res,
        )

        # Post-attention RMSNorm
        mlp_norm_w = self._get_layer_weight(f"model.layers.{layer}.post_attention_layernorm.weight")
        self._launch_rmsnorm_f32in(
            self._scratch.norm_out, self._scratch.hidden_f32,
            mlp_norm_w, seq_len,
        )

        # Gate + Up — column parallel
        gate_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.gate_proj.weight")
        up_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.up_proj.weight")
        down_proj = self._get_layer_weight(f"model.layers.{layer}.mlp.down_proj.weight")

        self._launch_matmul(self._scratch.mlp_gate, self._scratch.norm_out, gate_proj,
                           seq_len, self._intermediate_size, self._hidden_size)
        self._launch_matmul(self._scratch.mlp_up, self._scratch.norm_out, up_proj,
                           seq_len, self._intermediate_size, self._hidden_size)

        # SiLU
        n_mlp = self._intermediate_size * seq_len
        self._kernels.launch(
            "silu_mul",
            ((n_mlp + 255) // 256,),
            (256,),
            self._scratch.mlp_gate, self._scratch.mlp_gate, self._scratch.mlp_up, n_mlp,
        )

        # Down projection — row parallel
        self._launch_matmul(
            self._scratch.hidden, self._scratch.mlp_gate, down_proj,
            seq_len, self._hidden_size, self._intermediate_size,
        )

        # *** ALL-REDUCE after Down projection ***
        self._tp_group.all_reduce_inplace(
            self._scratch.hidden.data_ptr,
            self._hidden_size * seq_len,
            dtype="float16",
        )

        # MLP residual
        self._kernels.launch(
            "residual_add_f32",
            ((total_res + 255) // 256,),
            (256,),
            self._scratch.hidden_f32, self._scratch.hidden_f32, self._scratch.hidden,
            total_res,
        )
