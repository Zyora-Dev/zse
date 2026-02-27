# ZSE Architecture Pipeline

> **Last Updated:** February 27, 2026  
> **Version:** 1.2.0

This document describes the actual code flow and architecture of ZSE (Z Server Engine).

---

## 📊 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER ENTRY POINTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  CLI (zse chat/serve)  │  Python API  │  HTTP Server (/v1/chat/completions) │
└────────────┬───────────┴──────┬───────┴──────────────────┬──────────────────┘
             │                  │                          │
             ▼                  ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          .ZSE FORMAT LOADER                                  │
│                     zse/format/reader_v2.py                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ load_zse_model(path, cache_weights="auto")                          │    │
│  │ • Memory-mapped file access (fast cold start)                       │    │
│  │ • Embedded tokenizer + config (no network calls)                    │    │
│  │ • INT4 weights stored in packed format                              │    │
│  │ • Auto-converts to bnb format for CUDA inference                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      QUANTIZED LINEAR LAYER                                  │
│                     QuantizedLinearZSE (reader_v2.py)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Storage: INT4 packed (0.5 bytes/param)                              │    │
│  │ Inference: bitsandbytes.matmul_4bit CUDA kernel                     │    │
│  │                                                                     │    │
│  │ On first forward():                                                 │    │
│  │   1. Dequantize ZSE INT4 → FP16                                     │    │
│  │   2. Re-quantize with bnb (nf4 format)                              │    │
│  │   3. Use bnb.matmul_4bit for fast CUDA inference                    │    │
│  │                                                                     │    │
│  │ Result: ~60 tok/s for 7B, ~27 tok/s for 32B                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GENERATION ENGINE                                     │
│                     model.generate() with KV cache                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 .zse Format Data Flow

### 1. Conversion (One-Time)

```
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen7b.zse

writer.py: convert_model()
    ├── Load model from HuggingFace
    ├── Quantize weights to INT4 (group_size=128)
    ├── Pack INT4 pairs into uint8
    ├── Serialize tokenizer files (base64)
    ├── Serialize HF config JSON
    └── Write single .zse file with header
    
Output: qwen7b.zse (5.57 GB for 7B model)
```

### 2. Loading (.zse → GPU)

```python
model, tokenizer, info = load_zse_model("qwen7b.zse")

reader_v2.py: load_zse_model()
    ├── Memory-map .zse file
    ├── Parse header (architecture, quantization, offsets)
    ├── Load tokenizer from embedded data (no network)
    ├── Load config from embedded JSON (no network)
    ├── Create model skeleton on meta device
    ├── Replace Linear layers with QuantizedLinearZSE
    ├── Load INT4 packed weights directly to GPU
    └── Convert to bnb format (first forward or explicit)

VRAM: 5.9 GB for 7B, 20.9 GB for 32B
Load time: 9s for 7B, 24s for 32B
```

### 3. Inference (bnb.matmul_4bit)

```python
output = model.generate(input_ids, max_new_tokens=100)

QuantizedLinearZSE.forward(x):
    # First call: convert ZSE INT4 → bnb nf4
    if self._bnb_weight is None:
        weight_fp16 = dequantize_int4_zse(self.weight_packed, ...)
        self._bnb_weight, self._bnb_quant_state = quantize_4bit(weight_fp16)
    
    # Fast CUDA kernel (0.018ms per 1024x1024 matmul)
    return bnb.matmul_4bit(x, self._bnb_weight.t(), quant_state=...)

Speed: 58.7 tok/s for 7B, 26.9 tok/s for 32B
```

---

## 📁 Key Files

```
zse/format/
├── writer.py           # convert_model() - HF → .zse
│   ├── quantize_to_int4()     # Group-wise INT4 quantization
│   └── ZSEWriter              # Serializes model + tokenizer + config
│
├── reader_v2.py        # load_zse_model() - .zse → GPU
│   ├── ZSEReaderV2            # Memory-mapped file access
│   ├── QuantizedLinearZSE     # INT4 layer with bnb inference
│   ├── convert_model_to_bnb() # Pre-convert all layers
│   └── cache_model_weights()  # Optional FP16 caching
│
└── spec.py             # ZSEHeader, TensorInfo, dtype enums
```

---

## 🎯 Performance Summary

| Operation | 7B Model | 32B Model |
|-----------|----------|-----------|
| File size | 5.57 GB | 19.23 GB |
| Load time | 9.1s | 24.1s |
| VRAM usage | 5.9 GB | 20.9 GB |
| Inference speed | 58.7 tok/s | 26.9 tok/s |

### Why bnb.matmul_4bit?

Before v1.2.0, we had two options:
1. **Python dequantization**: 2.2 tok/s (32B) - unusable
2. **cache_weights (FP16)**: 32.7 tok/s but 82GB VRAM - too much

`bnb.matmul_4bit` gives us:
- **Fast CUDA kernel**: 26.9 tok/s for 32B
- **Low VRAM**: 20.9 GB (fits on 24GB GPUs)
- **Best of both worlds**: Speed + Memory efficiency

### 2. Server API Flow (HTTP Request)

```
HTTP Request:
    POST /v1/chat/completions
    {"model": "...", "messages": [...], "stream": true}

Code Path:
    app.py: chat_completions()                    # server/app.py:480
        ├── get_batching_state()
        │   └── If batching.enabled:
        │       └── batched_chat_completion()     # server/batching.py
        │   └── Else:
        │       └── _stream_chat_completion()     # Direct generation
        │
        └── _stream_chat_completion()             # server/app.py:630
            ├── Apply chat template
            ├── orch.generate(prompt, stream=True)
            └── StreamingResponse(event_generator)
```

### 3. Batched Server Flow (High Throughput)

```
Enable Batching:
    POST /api/batching/enable

Batched Request Flow:
    POST /v1/chat/completions → batching.py
    
    BatchingEngine                                # batching.py:92
        ├── start() → Creates processing_loop task
        ├── generate() / generate_stream()
        │   ├── Create BatchRequest
        │   └── Put in _request_queue
        │
        └── _processing_loop()                    # batching.py:282
            ├── _collect_batch()                  # Wait 50ms, collect requests
            └── _process_batch()
                ├── _run_prefill()                # Process prompts with KV cache
                │   ├── model(input_ids, use_cache=True)
                │   └── Store past_key_values in _kv_cache[request_id]
                └── _run_decode()                 # Generate tokens
                    ├── model(input_ids, past_key_values=kv, use_cache=True)
                    ├── Update _kv_cache[request_id]
                    └── Clean up cache when finished
```

---

## 📁 File Structure & Responsibilities

```
zse/
├── format/                      # ★ .ZSE FORMAT (Main Feature)
│   ├── writer.py                # convert_model() - HF → .zse
│   │   ├── quantize_to_int4()       # Group-wise INT4 quantization
│   │   └── ZSEWriter                # Serializes model + tokenizer + config
│   │
│   ├── reader_v2.py             # ★ load_zse_model() - .zse → GPU
│   │   ├── ZSEReaderV2              # Memory-mapped file access
│   │   ├── QuantizedLinearZSE       # INT4 layer with bnb.matmul_4bit
│   │   ├── convert_model_to_bnb()   # Pre-convert layers
│   │   └── cache_model_weights()    # Optional FP16 caching
│   │
│   └── spec.py                  # ZSEHeader, TensorInfo, dtype enums
│
├── engine/
│   ├── orchestrator/
│   │   ├── __init__.py          # Exports IntelligenceOrchestrator
│   │   └── core.py              # Orchestrator (uses .zse format)
│   │
│   ├── generation.py            # TextGenerator - Token-by-token generation
│   ├── batching.py              # BatchingEngine - Async batching for server
│   └── kv_cache.py              # KV cache implementations
│
├── api/
│   ├── cli/
│   │   └── main.py              # CLI: zse serve, zse convert, zse chat
│   │
│   └── server/
│       ├── app.py               # FastAPI server
│       └── batching.py          # Batched endpoints
│
└── core/
    └── zattention/              # Custom attention (future)
```

---

## ✅ What's Working (v1.2.0)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| **.zse Writer** | format/writer.py | ✅ Active | INT4 quantization + embed tokenizer/config |
| **.zse Reader** | format/reader_v2.py | ✅ Active | Memory-mapped, direct GPU loading |
| **QuantizedLinearZSE** | format/reader_v2.py | ✅ Active | bnb.matmul_4bit inference |
| **IntelligenceOrchestrator** | orchestrator/core.py | ✅ Active | VRAM detection, auto-optimize |
| **TextGenerator** | generation.py | ✅ Active | KV cache generation |
| **FastAPI Server** | server/app.py | ✅ Active | OpenAI-compatible API |

## 🎯 v1.2.0 Key Innovation

**Problem:** Python INT4 dequantization = 2.2 tok/s (unusable)

**Solution:** Use `bitsandbytes.matmul_4bit` CUDA kernel

```python
# QuantizedLinearZSE.forward() - v1.2.0
def forward(self, x):
    # Convert ZSE INT4 → bnb format (first call only)
    if self._bnb_weight is None:
        self.convert_to_bnb()
    
    # Fast CUDA kernel
    return bnb.matmul_4bit(x, self._bnb_weight.t(), quant_state=self._bnb_quant_state)
```

**Result:**
- 32B: 2.2 → **26.9 tok/s** (12x speedup)
- 7B: ~10 → **58.7 tok/s** (6x speedup)
- VRAM unchanged: 5.9 GB (7B), 20.9 GB (32B)

---

## 🔧 Configuration Options

### IntelligenceOrchestrator

```python
# Quantization modes
IntelligenceOrchestrator.min_memory(model)   # INT4 - lowest VRAM
IntelligenceOrchestrator.balanced(model)     # INT8 - balanced
IntelligenceOrchestrator.max_speed(model)    # FP16 - fastest
IntelligenceOrchestrator.auto(model)         # Auto-detect

# Multi-GPU
IntelligenceOrchestrator.multi_gpu(model, gpu_ids=[0,1])
```

### BatchingEngine

```python
BatchConfig(
    max_batch_size=32,          # Max concurrent requests
    max_tokens_per_batch=4096,  # Token limit
    batch_wait_timeout_ms=50,   # Wait time to form batches
    enable_cuda_graphs=False,   # CUDA graph optimization
)
```

### Server

```bash
# Start server
zse serve "Qwen/Qwen2.5-14B-Instruct" --host 0.0.0.0 --port 8000

# Enable batching (runtime)
curl -X POST http://localhost:8000/api/batching/enable
```

---

## 📈 Request Lifecycle Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REQUEST LIFECYCLE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

[User Request]
      │
      ▼
┌─────────────┐
│ Tokenize    │  prompt → input_ids
│ Prompt      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ PREFILL     │  Process full prompt, generate first token
│ Phase       │  Returns: logits + KV cache
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ DECODE      │  Loop until stop condition:
│ Loop        │    1. Pass only NEW token + KV cache
│             │    2. Get logits for next token
│  ┌────────────────────────────────────────┐
│  │ model(new_token, past_key_values=kv)   │
│  │ → logits, updated_kv_cache             │
│  │ → sample(logits) → next_token          │
│  │ → yield StreamChunk(next_token)        │
│  └────────────────────────────────────────┘
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ STOP        │  EOS token / max_tokens / stop_sequence
│ Condition   │
└──────┬──────┘
       │
       ▼
[Complete Response]
```

---

## 🧪 Testing Code Paths

```python
# Test 1: Python API (TextGenerator path)
from zse.engine.orchestrator import IntelligenceOrchestrator

orch = IntelligenceOrchestrator.auto("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
orch.load()
for chunk in orch.generate("Say hello", max_tokens=20, stream=True):
    print(chunk, end="", flush=True)

# Test 2: Server (start then curl)
# Terminal 1:
# zse serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --port 8000

# Terminal 2:
# curl http://localhost:8000/v1/chat/completions \
#   -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'

# Test 3: Enable batching
# curl -X POST http://localhost:8000/api/batching/enable
# Then send concurrent requests
```

---

## 📝 Version History

| Version | Change |
|---------|--------|
| 0.1.4 | Fixed KV cache in TextGenerator and BatchingEngine |
| 0.1.3 | Added multi-GPU support |
| 0.1.2 | Added server batching endpoints |
| 0.1.1 | Initial release with INT4/INT8/FP16 |
