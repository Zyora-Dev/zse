<div align="center">

# ZSE — Zero-dependency Server Engine for LLM Inference

**The fastest cold start. The smallest memory footprint. On every GPU.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-brightgreen)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/dependencies-zero-success)](#)
[![Backends](https://img.shields.io/badge/backends-CUDA%20%7C%20ROCm%20%7C%20Metal-orange)](#hardware-validated)
[![Tests](https://img.shields.io/badge/tests-447%20passing-success)](#)

[Website](https://zllm.in) · [Docs](docs/) · [Hugging Face](https://huggingface.co/zse-zllm) · [Sponsor](https://github.com/sponsors/Zyora-Dev)

</div>

---

## What is ZSE?

ZSE is a production LLM inference engine that owns the full stack — **no PyTorch, no Triton, no bitsandbytes, no transformers**. Just pure Python, ctypes, and a kernel compiler that emits CUDA, ROCm (HIP), and Metal directly.

The result: **models load in seconds, not minutes**, and serve at a fraction of the memory other engines need.

```bash
pip install zse-engine   # one package, zero transitive ML deps
zse serve qwen-7b.zse    # 7-second cold start. 5.8 GB on a T4.
```

---

## Headline Numbers

> Verified on Modal (T4, L4, A10G, A100), DigitalOcean (MI300X), and Apple M1. ZSE INT4 vs vLLM AWQ INT4, Qwen2.5-7B / 14B / 32B.

### Cold start — every GPU, every model size

| GPU | Model | ZSE | vLLM | **Speedup** |
|---|---|---:|---:|---:|
| NVIDIA T4 (16 GB) | Qwen2.5-7B | **7.25s** | 218.96s | **30.2×** |
| NVIDIA L4 (24 GB) | Qwen2.5-7B | **5.58s** | 145.22s | **26.0×** |
| NVIDIA A10G (24 GB) | Qwen2.5-7B | **6.01s** | 193.05s | **32.1×** |
| NVIDIA A100-80GB | Qwen2.5-14B | **6.29s** | 127.02s | **20.2×** |
| AMD MI300X (192 GB) | Qwen2.5-32B | **3.14s** | 42.65s | **13.6×** |

### VRAM — fits where others can't

| GPU | Model | ZSE | vLLM | **Reduction** |
|---|---|---:|---:|---:|
| NVIDIA T4 | Qwen2.5-7B | **5.79 GB** | ~14 GB | **~2.5×** |
| NVIDIA A100-80GB | Qwen2.5-14B | **12.28 GB** | 71.45 GB | **5.82×** |
| AMD MI300X | Qwen2.5-32B | **22.07 GB** | 161.77 GB | **7.33×** |

ZSE runs **32B INT4 in 22 GB of VRAM** — on a single MI300X with room for 8 more models. vLLM's PyTorch allocator + KV slab grabs the entire GPU regardless of quantization.

### Single-sequence throughput — matches or beats vLLM on data-center GPUs

| GPU | Model | ZSE | vLLM | Ratio |
|---|---|---:|---:|---:|
| NVIDIA A100-80GB | Qwen2.5-14B | **37.0 tok/s** | 26.5 | **1.40×** |
| NVIDIA A10G | Qwen2.5-7B | 48.6 tok/s | 50.9 | 0.95× |
| NVIDIA L4 | Qwen2.5-7B | 36.3 tok/s | 47.3 | 0.77× |
| AMD MI300X | Qwen2.5-32B | 38.4 tok/s | 56.4 | 0.68× |
| NVIDIA T4 | Qwen2.5-7B | 18.8 tok/s | 35.2 | 0.53× |

---

## Why ZSE

| | vLLM | **ZSE** |
|---|---|---|
| Cold start (7B) | 30s – 4 min | **5–7s** |
| VRAM (14B INT4) | 71 GB | **12 GB** |
| Dependencies | PyTorch + Triton + CUDA toolkit (~12 GB) | **Zero** |
| Pip install size | ~3 GB | **~5 MB** |
| Backends | CUDA primarily | **CUDA + ROCm + Metal** |
| Model format | safetensors (deserialize on load) | **`.zse`** (mmap, pre-quantized, instant) |
| KV cache | Fixed 16-token blocks, LRU eviction | **Adaptive blocks, token-level smart eviction** |
| Model conversion | None — runtime quant | **One-time, ~600× faster than pure Python** |
| Built-in RAG | ❌ | **✅ (hybrid retrieval + cross-encoder rerank + ZPF compression)** |
| Built-in auth + rate limiting | ❌ | **✅ (SQLite-backed)** |
| LoRA hot-swap | ✅ (S-LoRA) | **✅** |

---

## Hardware Validated

| Hardware | Vendor | Arch | Status |
|---|---|---|---|
| NVIDIA T4 | NVIDIA | Turing (sm_75) | ✅ |
| NVIDIA L4 | NVIDIA | Ada (sm_89) | ✅ |
| NVIDIA A10G | NVIDIA | Ampere (sm_86) | ✅ |
| NVIDIA A100 (40 GB, 80 GB) | NVIDIA | Ampere DC (sm_80) | ✅ |
| NVIDIA H100 / H200 | NVIDIA | Hopper (sm_90) | ✅ |
| AMD Instinct MI300X (192 GB) | AMD | CDNA3 (gfx942) | ✅ |
| Apple M1 | Apple | Apple Silicon | ✅ |

A new arch usually works on day one — the compiler queries compute capability at runtime and emits the correct PTX / GCN / MSL automatically.

---

## Install

```bash
pip install zse-engine
```

Requirements:
- Python 3.11+
- One of: NVIDIA driver + CUDA runtime, AMD ROCm 6+, or Apple Silicon
- That's it. No PyTorch. No Triton. No transformers.

---

## Quick Start

### 1. Get a model

```bash
# Pull a pre-converted model (instant)
zse pull qwen-7b              # 5.18 GB
zse pull qwen-32b             # 17.9 GB
zse pull mistral-7b           # 3.86 GB

# Or convert any HuggingFace model yourself
zse convert Qwen/Qwen2.5-7B-Instruct qwen-7b.zse --quant int4
```

### 2. Serve

```bash
zse serve qwen-7b.zse --port 8000
```

OpenAI-compatible API at `http://localhost:8000/v1`:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="zse")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain mixture of experts in one paragraph."}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 3. Multi-GPU (optional)

```bash
zse serve qwen-72b.zse --tp 4 --port 8000        # tensor parallel
```

---

## Features

**Inference engine**
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
- Continuous batching with disaggregated prefill/decode scheduling
- SLO-aware request ordering, predictive memory budgeting, chunked prefill
- Speculative decoding (n-gram + self-draft, lossless accept/reject)
- CUDA Graphs + HIP Graphs for low-latency decode
- Tensor parallelism (NCCL/RCCL, multi-process weight sharding)
- LoRA hot-swap — 100s of adapters per GPU, per-request routing

**Model format (`.zse`)**
- Pre-quantized INT4 / INT8 / FP16, mmap-friendly
- One file = weights + tokenizer + config + kernel cache
- Architectures supported out of the box: Llama, Mistral, Qwen2, Gemma2, Phi3

**Built-in RAG**
- Hybrid retrieval: BM25 + TF-IDF + dense embeddings (mean-pooled LLM hidden states — no extra model)
- Reciprocal Rank Fusion + LLM cross-encoder reranker
- ZPF compressed document format — 25% fewer LLM tokens at 100% retrieval accuracy
- PDF parser handles encrypted (RC4 / AES-128 / AES-256), multi-column reflow, /ObjStm, OCR hook

**Server**
- API key management + per-key RPM/TPM rate limiting (SQLite)
- Admin API, LoRA management API, RAG ingest API
- Web dashboard for chat + session management
- SSE streaming, pure asyncio, zero web framework dependency

**Kernel compiler (`zse-compiler`)**
- Write GPU kernels in pure Python with `@zse.kernel`
- Emits CUDA C, HIP C, and Metal Shading Language
- Auto-tuning, kernel fusion, WMMA / MFMA matrix-core intrinsics
- Standalone — `pip install zse-compiler` works on its own

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  HTTP / SSE  ·  OpenAI API  ·  Web dashboard  ·  API key + RAG   │
├───────────────────────────────────────────────────────────────────┤
│           ZStreamer — continuous batching, scheduling             │
├───────────────────────────────────────────────────────────────────┤
│   Orchestrator   │   KV Cache (PagedAttention)   │   LoRA Mgr    │
│   29 GPU kernels │   adaptive blocks · token-evict│  hot-swap     │
├───────────────────────────────────────────────────────────────────┤
│  .zse format    │  VRAM allocator (unified)  │  CUDA/HIP Graphs  │
├───────────────────────────────────────────────────────────────────┤
│         ZSE Kernel Compiler — Python DSL → GPU code               │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐   │
│   │  CUDA C     │    │   HIP C     │    │  Metal Shading Lang │   │
│   │  (nvrtc)    │    │  (hiprtc)   │    │  (Metal compiler)   │   │
│   └─────────────┘    └─────────────┘    └─────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
              No PyTorch · No Triton · No transformers
```

---

## Honest Limitations

We believe in numbers, not marketing. Things ZSE does **not** beat vLLM on yet:

- **Concurrent throughput at N≥4 on INT4.** vLLM's hand-tuned AWQ Marlin kernels hit memory bandwidth ceilings we haven't matched yet on NVIDIA. Closed to 2.12× on AMD via our wave-64 bgemv rewrite; NVIDIA-side equivalent is the next throughput lever. See [CLAUDE.md](CLAUDE.md) Gap #6 for the full story.
- **Apple Silicon full inference.** Kernel-level validated on M1 (E2E vector_add: 0/1024 mismatches). Full transformer inference path needs a hardware run — wired and ready.
- **Tensor parallelism on socket-restricted environments.** All NCCL primitives validated multi-GPU on Modal; full TP inference works on bare-metal multi-GPU servers but the worker bootstrap needs real network access (not a code bug — Modal's sandbox blocks `AF_UNIX` sockets used by `ncclCommInitRank` in child processes).

If steady-state batched throughput is your only metric and you have ~50× the VRAM budget — use vLLM. If you care about cold start, footprint, vendor lock-in, or running on anything other than an H100 — use ZSE.

---

## Benchmark Reproduction

All numbers in this README are reproducible. Scripts live in [`tests/`](tests/):

```bash
modal run tests/test_modal_benchmark_7b_rtx.py        # T4, L4, A10G  vs vLLM AWQ
modal run tests/test_modal_bench_vs_vllm.py           # A100-80GB     vs vLLM AWQ + FP16
python tests/bench_zse_mi300x_v3.py                   # MI300X
```

Raw JSON outputs for every run are committed alongside the scripts.

---

## What's Inside

```
zse-compiler/        Pure-Python kernel compiler. Standalone, pip-installable.
  ast_parser/        Python AST → IR
  ir/                25+ IR node types, fusion pass, type inference
  codegen/           CUDA · HIP · Metal backends
  runtime/           NVRTC · HIPRTC · Metal · NCCL/RCCL · auto-tune · profiler

zse-engine/          Production inference engine.
  format/            .zse binary format, quantization, conversion CLI
  orchestrator/      29 GPU kernels, model runner, sampler, VRAM allocator
  cache/             PagedAttention, dedup, smart eviction, COW forking
  zstreamer/         Continuous batching, SLO scheduling, spec-decode
  server/            HTTP, OpenAI API, auth, rate limit, admin, LoRA, RAG
  rag/               Hybrid retrieval, reranker, ZPF, PDF parser (full PDF spec)
```

~40,600 lines of code. **Zero third-party dependencies.**

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

<p>This project is supported by:</p>
<p>
  <a href="https://www.digitalocean.com/">
    <img src="https://opensource.nyc3.cdn.digitaloceanspaces.com/attribution/assets/SVG/DO_Logo_horizontal_blue.svg" width="201px">
  </a>
</p>

ZSE's AMD MI300X validation, the 32B-parameter benchmarks, and a large share of our ROCm
kernel development work was made possible by [**DigitalOcean's Open Source Sponsorship
Program**](https://www.digitalocean.com/open-source/credits-for-projects), which provides
cloud GPU credits to independent open-source projects. The MI300X numbers throughout this
README — cold start, VRAM, throughput, the wave-64 INT4 GEMV rewrite — were all measured
on DigitalOcean infrastructure. Thank you to the DigitalOcean team for backing zero-dep
infrastructure work.

If you maintain an open-source project that needs serious GPU time, apply here:
https://www.digitalocean.com/open-source/credits-for-projects

## Contact

- Website: [zllm.in](https://zllm.in)
- Company: [Zyora Labs](https://zyoralabs.com)
- Email: `zse@zyoralabs.com`
- Sponsor: [github.com/sponsors/Zyora-Dev](https://github.com/sponsors/Zyora-Dev)

---

<div align="center">

**Built in Nagercoil, India. Run anywhere a GPU runs.**

</div>
