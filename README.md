<p align="center">
  <a href="https://zllm.in">
    <img src="https://zllm.in/_next/image?url=%2Fimages%2Fzllm-logo.png&w=256&q=75" alt="ZLLM Logo" width="128">
  </a>
</p>

<h1 align="center">ZSE - Z Server Engine</h1>

<p align="center">
  <a href="https://pypi.org/project/zllm-zse/"><img src="https://img.shields.io/pypi/v/zllm-zse.svg" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://zllm.in"><img src="https://img.shields.io/badge/website-zllm.in-blue" alt="Website"></a>
</p>

<p align="center">
  <a href="https://railway.app/template?repo=https://github.com/Zyora-Dev/zse"><img src="https://railway.app/button.svg" alt="Deploy on Railway"></a>
  <a href="https://render.com/deploy?repo=https://github.com/Zyora-Dev/zse"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
</p>

**Ultra memory-efficient LLM inference engine with native INT4 CUDA kernels.**

Run 32B models on 24GB GPUs. Run 7B models on 8GB GPUs. Fast cold starts, single-file deployment.

## 🚀 Benchmarks (Verified, March 2026)

### Custom Triton Kernel (Default - Max VRAM Efficiency)

| Model | File Size | VRAM | Speed | Cold Start | GPU |
|-------|-----------|------|-------|------------|-----|
| **Qwen 7B** | 5.57 GB | **5.67 GB** | 37.2 tok/s | 5.7s | H200 |
| **Qwen 14B** | 9.95 GB | **10.08 GB** | 20.8 tok/s | 10.5s | H200 |
| **Qwen 32B** | 19.23 GB | **19.47 GB** | 10.9 tok/s | 20.4s | H200 |
| **Qwen 72B** | 41.21 GB | **41.54 GB** | 6.3 tok/s | 51.8s | H200 |

### bitsandbytes Backend (Optional - Max Speed)

| Model | VRAM | Speed | Cold Start |
|-------|------|-------|------------|
| **Qwen 7B** | 6.57 GB | 45.6 tok/s | 6.0s |
| **Qwen 14B** | 11.39 GB | 27.6 tok/s | 7.1s |
| **Qwen 32B** | 22.27 GB | 20.4 tok/s | 20.8s |
| **Qwen 72B** | 47.05 GB | 16.4 tok/s | 53.0s |

### VRAM Savings (Triton vs bnb)

| Model | Triton VRAM | bnb VRAM | Savings |
|-------|-------------|----------|----------|
| 7B | 5.67 GB | 6.57 GB | **-0.90 GB (14%)** |
| 14B | 10.08 GB | 11.39 GB | **-1.31 GB (12%)** |
| 32B | 19.47 GB | 22.27 GB | **-2.80 GB (13%)** |
| 72B | 41.54 GB | 47.05 GB | **-5.51 GB (12%)** |

### GPU Compatibility

| GPU | VRAM | Max Model (Triton) | Max Model (bnb) |
|-----|------|--------------------|-----------------|
| RTX 3070/4070 | 8GB | **7B** | 7B |
| RTX 3080 | 12GB | **14B** | 7B |
| RTX 3090/4090 | 24GB | **32B** | 32B |
| A100-40GB | 40GB | **32B** | 32B |
| A100-80GB / H200 | 80-141GB | **72B** | 72B |

## Key Features

- 📦 **Single .zse File**: Model + tokenizer + config in one file
- 🚫 **No Network Calls**: Everything embedded, works offline
- ⚡ **Custom Triton Kernel**: Native INT4 inference, no bitsandbytes required
- 🧠 **Memory Efficient**: 72B in 41GB, 32B in 19GB, 7B in 5.7GB VRAM
- 🏃 **Fast Cold Start**: 5.7s for 7B, 20s for 32B, 52s for 72B
- 🎯 **Auto Backend**: Triton (VRAM efficient) or bnb (max speed)

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
3. **Inference**: Custom Triton kernel (default) or bnb for matmul

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  HuggingFace    │────▶│   .zse File     │────▶│   GPU Model     │
│  Model (FP16)   │     │   (INT4 + tok)  │     │ (Triton/bnb)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
    One-time             Single file             12% less VRAM
    conversion           ~0.5 bytes/param        vs bitsandbytes
```

## OpenClaw Integration

Run local models with [OpenClaw](https://openclaw.ai) - the 24/7 AI assistant by @steipete.

```bash
# Start ZSE server
zse serve <model-name> --port 8000

# Configure OpenClaw to use local ZSE
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=zse
```

Or in OpenClaw's `config.yaml`:

```yaml
llm:
  provider: openai-compatible
  api_base: http://localhost:8000/v1
  api_key: zse
  model: default
```

**Benefits:** 100% private, zero API costs, works offline, run ANY model.

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

## Contact

- **Website:** [zllm.in](https://zllm.in)
- **Company:** [Zyora Labs](https://zyoralabs.com)
- **Email:** [zse@zyoralabs.com](mailto:zse@zyoralabs.com)

---

<p align="center">
  Made with ❤️ by <a href="https://zyoralabs.com">Zyora Labs</a>
</p>
