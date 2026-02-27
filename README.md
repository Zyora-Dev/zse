# ZSE - Z Server Engine

[![PyPI](https://img.shields.io/pypi/v/zllm-zse.svg)](https://pypi.org/project/zllm-zse/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template?repo=https://github.com/Zyora-Dev/zse)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Zyora-Dev/zse)

**Ultra memory-efficient LLM inference engine with native INT4 CUDA kernels.**

Run 32B models on 24GB GPUs. Run 7B models on 8GB GPUs. Fast cold starts, single-file deployment.

## 🚀 Benchmarks (Verified, February 2026)

| Model | File Size | VRAM | Speed | Load Time | GPU |
|-------|-----------|------|-------|-----------|-----|
| **Qwen 7B** | 5.57 GB | **5.9 GB** | **58.7 tok/s** | 9.1s | H200 |
| **Qwen 32B** | 19.23 GB | **20.9 GB** | **26.9 tok/s** | 24.1s | H200 |

### GPU Compatibility

| GPU | VRAM | Max Model | Expected Speed |
|-----|------|-----------|----------------|
| RTX 3070/4070 | 8GB | 7B | ~50-60 tok/s |
| RTX 3080/4080 | 12-16GB | 7B | ~50-60 tok/s |
| RTX 3090/4090 | 24GB | **32B** | ~25-30 tok/s |
| A100-40GB | 40GB | 32B | ~25-30 tok/s |
| A100-80GB / H200 | 80-141GB | 72B | TBD |

## Key Features

- 📦 **Single .zse File**: Model + tokenizer + config in one file
- 🚫 **No Network Calls**: Everything embedded, works offline
- ⚡ **Native INT4 CUDA**: bitsandbytes.matmul_4bit for fast inference
- 🧠 **Memory Efficient**: 32B model in 21GB VRAM
- 🏃 **Fast Cold Start**: 9s for 7B, 24s for 32B
- 🎯 **Auto-Optimize**: Detects VRAM and picks optimal strategy

## Installation

```bash
pip install zllm-zse
```

**Requirements:**
- Python 3.11+
- CUDA GPU (8GB+ VRAM recommended)
- bitsandbytes (auto-installed)

## Quick Start

### 1. Convert Model to .zse Format (One-Time)

```bash
# Convert any HuggingFace model
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen7b.zse
zse convert Qwen/Qwen2.5-32B-Instruct -o qwen32b.zse

# Or in Python
from zse.format.writer import convert_model
convert_model("Qwen/Qwen2.5-7B-Instruct", "qwen7b.zse", quantization="int4")
```

### 2. Load and Run

```python
from zse.format.reader_v2 import load_zse_model

# Load model (auto-detects optimal settings)
model, tokenizer, info = load_zse_model("qwen7b.zse")

# Generate
inputs = tokenizer("Write a poem about AI:", return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 3. Start Server (OpenAI-Compatible)

```bash
zse serve qwen7b.zse --port 8000
```

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="zse")
response = client.chat.completions.create(
    model="qwen7b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## .zse Format Benefits

| Feature | HuggingFace | .zse Format |
|---------|-------------|-------------|
| Cold start (7B) | 45s | **9s** |
| Cold start (32B) | 120s | **24s** |
| Network calls on load | Yes | **No** |
| Files to manage | Many | **One** |
| Quantization time | Runtime | **Pre-done** |

## Advanced Usage

### Control Caching Strategy

```python
# Auto (default): Detect VRAM, pick optimal strategy
model, tok, info = load_zse_model("qwen7b.zse", cache_weights="auto")

# Force bnb mode (low VRAM, fast inference)
model, tok, info = load_zse_model("qwen7b.zse", cache_weights=False)

# Force FP16 cache (max speed, high VRAM)
model, tok, info = load_zse_model("qwen7b.zse", cache_weights=True)
```

### Benchmark Your Setup

```bash
# Full benchmark
python3 -c "
import time, torch
from zse.format.reader_v2 import load_zse_model

t0 = time.time()
model, tokenizer, info = load_zse_model('qwen7b.zse')
print(f'Load: {time.time()-t0:.1f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB')

inputs = tokenizer('Hello', return_tensors='pt').to('cuda')
model.generate(**inputs, max_new_tokens=10)  # Warmup

prompt = 'Write a detailed essay about AI.'
inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
torch.cuda.synchronize()
t0 = time.time()
out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
torch.cuda.synchronize()
tokens = out.shape[1] - inputs['input_ids'].shape[1]
print(f'{tokens} tokens in {time.time()-t0:.2f}s = {tokens/(time.time()-t0):.1f} tok/s')
"
```

## CLI Commands

```bash
# Convert model
zse convert <model_id> -o output.zse

# Start server
zse serve <model.zse> --port 8000

# Interactive chat
zse chat <model.zse>

# Show model info
zse info <model.zse>

# Check hardware
zse hardware
```

## How It Works

1. **Conversion**: Quantize HF model to INT4, pack weights, embed tokenizer + config
2. **Loading**: Memory-map .zse file, load INT4 weights directly to GPU
3. **Inference**: Convert to bnb format on first forward, use CUDA kernel for matmul

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  HuggingFace    │────▶│   .zse File     │────▶│   GPU Model     │
│  Model (FP16)   │     │   (INT4 + tok)  │     │   (bnb matmul)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
    One-time             Single file             Fast inference
    conversion           ~0.5 bytes/param        58 tok/s (7B)
```

## Docker Deployment

```bash
# CPU
docker run -p 8000:8000 ghcr.io/zyora-dev/zse:latest

# GPU (NVIDIA)
docker run --gpus all -p 8000:8000 ghcr.io/zyora-dev/zse:gpu

# With model pre-loaded
docker run -p 8000:8000 -e ZSE_MODEL=Qwen/Qwen2.5-0.5B-Instruct ghcr.io/zyora-dev/zse:latest
```

See [deploy/DEPLOY.md](deploy/DEPLOY.md) for full deployment guide including Runpod, Vast.ai, Railway, Render, and Kubernetes.

## License

Apache 2.0
