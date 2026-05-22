# Changelog

All notable changes to ZSE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-05-22

### The "Own Everything" Release

A complete ground-up rewrite. ZSE no longer depends on PyTorch, Triton, transformers,
or bitsandbytes. The entire stack — from the kernel compiler up to the OpenAI-compatible
HTTP server — is pure Python + ctypes. Install size dropped from ~3 GB to ~5 MB.

### ⚠️ Breaking Changes

- **Package rename**: `zllm-zse` → `zse-engine` on PyPI. `pip install zllm-zse` no longer works.
- **Zero third-party ML dependencies**. PyTorch, Triton, transformers, bitsandbytes, and accelerate are all removed. If you had code that imported `zse` and relied on PyTorch tensors leaking through, it will break.
- **Python module rename**: `zse` → `zse_engine`. Update imports:
  ```python
  # Old (1.x)
  from zse.format import load_zse_model
  from zse.format.writer import convert_model

  # New (2.0)
  from zse_engine.format.loader import load_model
  from zse_engine.format.convert import convert_hf_model
  ```
- **`.zse` v2 format**. Old `.zse` files from 1.x will not load. Re-convert your HuggingFace models with `zse convert` — it's ~600× faster than the 1.x converter.
- **`bnb` backend removed**. The dual-backend "ZSE custom kernel + bitsandbytes" model is gone. ZSE's INT4 kernels now serve both inference and training.
- **CLI flag changes**: `-tp 2` is now `--tp 2`. `--multi-gpu` removed (auto-detected). `cache_weights="auto"` removed (always optimal).

### Added

#### Kernel compiler (`zse-compiler`) — new standalone package
- Pure-Python DSL (`@zse.kernel`) compiles to CUDA C, HIP C, and Metal Shading Language.
- Targets NVRTC, HIPRTC, and Apple's Metal compiler — no Triton, no Xcode required.
- Warp primitives: shuffle (down/up/xor/idx), reduce, ballot, vote, lane/warp id.
- Vectorized memory: `float4`, `half2` load/store.
- Block reductions (two-stage warp + shared mem) and tiling DSL (`tile_load` / `tile_store` with optional bounds predication).
- Kernel fusion (`zse.fuse()`) — eliminates intermediate global-mem trips.
- Tensor Core / WMMA intrinsics (`wmma_load_a/b`, `wmma_mma`, `wmma_store`) on NVIDIA.
- AMD CDNA matrix-core intrinsics: `mfma_f32_16x16x16_f16`, `mfma_f32_32x32x8_f16`.
- INT4 unpacking intrinsics: `unpack_int4`, `unpack_uint4`.
- Local stack arrays (`zse.local_array`) and typed pointer reinterpret (`zse.reinterpret`).
- Auto-tuner with CUDA / HIP event profiling.
- Kernel disk cache — compile once, load instantly forever.

#### Model format (`.zse` v2)
- 64-byte header + CRC32 + section table + page-aligned weights.
- Pre-quantized INT4 / INT8 / FP16 — no runtime quantization tax.
- mmap-friendly with direct GPU pointer API.
- Resumable conversion (crash recovery), C-accelerated quantization (~600× faster than pure Python).
- Architecture adapters: Llama / Mistral, Qwen2, Gemma2, Phi3.

#### KV cache manager
- Adaptive block sizing (not fixed at 16 like vLLM).
- Token-level fine-grained eviction (vLLM is sequence-level only).
- Smart eviction policy: frequency × recency + recompute cost (configurable LRU / LFU).
- Block deduplication via FNV-1a hash + token equality verification.
- COW forking for beam search.
- Sliding-window attention support.

#### Orchestrator
- 29 GPU kernels (18 base + 7 optimized + 4 portable @zse.kernel rewrites).
- Tiled matmul (shared-mem TILE=32) ~10–20× over naive 1-thread-per-output.
- Fused residual + RMSNorm (3 launches → 1).
- Batched RoPE / KV-write / decode (27× fewer kernel launches).
- Hand-tuned AMD wave-64 INT4 GEMV (`bgemv_int4_wave64_v2`) — 2.14× over scalar baseline, 128-bit weight loads on MI300X.
- CUDA Graphs + HIP Graphs decode acceleration (opportunistic pre-capture, bulk DtoH argmax).
- Unified VRAM allocator — KV cache + activations in one pool.
- Demand-based KV sizing (was static 12 GB cap).

#### ZStreamer (continuous batching)
- Disaggregated prefill / decode scheduling.
- SLO-aware request ordering (urgency > priority > arrival time).
- Predictive memory budgeting + admission control.
- Chunked prefill for long prompts.
- Anti-burst control (token budget per iteration, gradual admission).
- Preemption under memory pressure.
- **Speculative decoding** — lossless accept/reject (Leviathan / Chen et al.), n-gram + self-draft. 2–4× throughput boost.

#### Server
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) — pure asyncio, zero web framework.
- SSE streaming.
- API key auth + per-key RPM/TPM rate limiting (SQLite-backed).
- Admin API, LoRA management API.
- Web dashboard for chat + session management.
- Built-in RAG API.

#### RAG system (modern)
- Hybrid retrieval: BM25 + TF-IDF + **dense embeddings** (mean-pooled LLM hidden states — zero extra VRAM, zero extra deps).
- Reciprocal Rank Fusion (k=60).
- **LLM cross-encoder reranker** via Yes/No logprob scoring.
- ZPF compressed document format — 25% fewer LLM tokens at 100% retrieval accuracy.
- Full PDF parser:
  - ToUnicode CMap support (handles subset fonts from Word, LaTeX, modern PDF writers).
  - Stream filters: ASCIIHexDecode, ASCII85Decode, RunLengthDecode, LZWDecode, FlateDecode.
  - **Encryption**: RC4-40, RC4-128, AES-128, AES-256 (Standard Security Handler V=1/2/4/5, R=2/3/4/5/6). Empty password handled by default; user passwords via `pdf_password` metadata.
  - Multi-column layout reflow with text-matrix tracking.
  - /ObjStm compressed object streams (PDF 1.5+).
  - OCR fallback hook for scanned PDFs.

#### LoRA serving
- Per-adapter overhead < 1% VRAM (e.g., rank=16 on 7B ≈ 16 MB).
- Hot-swap without restart, 100s of adapters per GPU.
- Per-request routing via `lora_id`.
- `.zse-lora` binary format.

#### Tensor parallelism
- Pure-ctypes NCCL / RCCL wrapper (AllReduce, AllGather, Broadcast, ReduceScatter, Barrier).
- Multi-process, weight sharding (column-parallel for QKV/Gate/Up, row-parallel for O/Down).
- `zse serve model.zse --tp 4` Just Works.

#### Cold start optimization
- Pinned-host alloc + async copy + CUDA streams (pure ctypes).
- Double-buffered weight streaming, 64 MB chunks, posix_fadvise(WILLNEED).
- Parallel section reads + skip CRC on WEIGHT_INDEX.
- Bulk GPU allocation (one `cuMemAlloc` instead of 771).
- Kernel disk cache (`~/.cache/zse/kernels/`).
- `zse warm <model.zse>` to pre-fault into page cache.

#### Hardware support
- **NVIDIA**: T4 (sm_75), A10G (sm_86), L4 (sm_89), A100 (sm_80), H100 / H200 (sm_90) — arch auto-detected at runtime.
- **AMD**: MI300X (gfx942, CDNA3) — full ROCm validation.
- **Apple**: M1 (kernel-level validated).

### Changed
- Cold start (Qwen2.5-14B INT4, A100-80GB): **6.29s** (was 12.89s in 1.x, was 30–120s for vLLM).
- VRAM (Qwen2.5-14B INT4, A100-80GB): **12.28 GB** (was 21.96 GB in 1.x, 71.45 GB for vLLM).
- Single-seq throughput (Qwen2.5-14B INT4, A100-80GB): **37 tok/s** (1.40× vs vLLM AWQ INT4).
- Conversion speed: ~600× faster (C-accelerated quantization).
- Pip install size: ~3 GB → **~5 MB**.

### Removed
- `bitsandbytes` dependency and `bnb` backend.
- `PyTorch` dependency.
- `transformers` dependency.
- `accelerate` dependency.
- `triton` dependency.
- Runtime model quantization (replaced by ahead-of-time `.zse` conversion).
- `zllm-zse` PyPI package name (renamed `zse-engine`).

### Fixed
- NVRTC arch hardcode (was `compute_80` — broke on T4 / Turing GPUs). Now auto-detects per device via `cuDeviceGetAttribute`.
- KV cache double-free on request finalization.
- Request leak when admission control rejected mid-batch.
- ZStreamer decode progress double-counting.
- Stats unbounded list memory leak (now bounded deque).
- p99 TTFT off-by-one.

### Benchmark Summary (vs vLLM AWQ INT4)

| GPU | Model | Cold start | VRAM | Single tok/s |
|---|---|---:|---:|---:|
| NVIDIA T4 | Qwen2.5-7B | **30.2× faster** | ~2.5× less | 0.53× |
| NVIDIA L4 | Qwen2.5-7B | **26.0× faster** | ~3× less | 0.77× |
| NVIDIA A10G | Qwen2.5-7B | **32.1× faster** | ~3× less | 0.95× |
| NVIDIA A100-80GB | Qwen2.5-14B | **20.2× faster** | **5.82× less** | **1.40× faster** |
| AMD MI300X | Qwen2.5-32B | **13.6× faster** | **7.33× less** | 0.68× |

### Migration from 1.4.x

```bash
pip uninstall zllm-zse
pip install zse-engine

# Re-convert your models (one-time, ~600× faster now)
zse convert Qwen/Qwen2.5-7B-Instruct qwen-7b.zse --quant int4

# Serve as before
zse serve qwen-7b.zse --port 8000
```

Python imports:
```python
# Old
from zse.format import load_zse_model
model, tokenizer, info = load_zse_model("model.zse")

# New
from zse_engine.format.loader import load_model
model = load_model("model.zse")
```

---

## [1.4.1] — 2026-03-02

Last release of the bitsandbytes-backed era. See [git history](https://github.com/Zyora-Dev/zse/commits/v1.4.1) for details.
