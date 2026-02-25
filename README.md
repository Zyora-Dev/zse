# ZSE - Z Server Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template?repo=https://github.com/Zyora-Dev/zse)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Zyora-Dev/zse)

**Ultra memory-efficient LLM inference engine.**

ZSE is designed to run large language models with minimal memory footprint while maintaining high performance. Our key innovation is the **Intelligence Orchestrator** that provides smart recommendations based on your available (not total) memory.

## Key Features

- 🧠 **zAttention**: Custom CUDA kernels for paged, flash, and sparse attention
- 🗜️ **zQuantize**: Per-tensor INT2-8 mixed precision quantization
- 💾 **zKV**: Quantized KV cache with sliding precision (4x memory savings)
- 🌊 **zStream**: Layer streaming with async prefetch (run 70B on 24GB GPU)
- 🎯 **zOrchestrator**: Smart recommendations based on FREE memory
- 📊 **Efficiency Modes**: speed / balanced / memory / ultra

## ⚡ Cold Start Benchmark

**3.9s (7B)** and **21.4s (32B)** to first token with `.zse` format — verified on A100-80GB.

| Model | bitsandbytes | ZSE (.zse) | Speedup |
|-------|--------------|------------|----------|
| **Qwen 7B** | 45.4s | **3.9s** | **11.6×** |
| **Qwen 32B** | 120.0s | **21.4s** | **5.6×** |

```bash
# One-time conversion (~20s)
zse convert Qwen/Qwen2.5-Coder-7B-Instruct -o qwen-7b.zse

# Every subsequent start: 3.9s
zse serve qwen-7b.zse
```

> **Note:** Results measured on A100-80GB with NVMe storage (Feb 2026). On consumer SSDs expect 5-10s; HDDs may be slower. Any modern SSD achieves sub-10s cold starts.

## Memory Benchmarks (Verified, A100-80GB)

| Model | FP16 | INT4/NF4 | Reduction | Throughput |
|-------|------|----------|----------|------------|
| Qwen 7B | 14.2 GB | **5.2 GB** | 63% ✅ | 12-15 tok/s |
| Qwen 32B | ~64 GB | **19.3 GB** (NF4) / ~35 GB (.zse) | 70% ✅ | 7.9 tok/s |
| 14B | ~28 GB | *~7 GB* | ⏳ est | - |
| 70B | ~140 GB | *~24 GB* | ⏳ est | - |

> **32B note:** Use NF4 (19.3 GB) on GPUs with <36 GB VRAM. Use `.zse` (35 GB, 5.6× faster start) on 40 GB+ GPUs.

## Installation

```bash
pip install zllm-zse
```

With CUDA support (recommended):
```bash
pip install zllm-zse[cuda]
```

From source:
```bash
git clone https://github.com/Zyora-Dev/zse.git
cd zse
pip install -e ".[dev]"
```

## Quick Start

### Start Server

```bash
# Any HuggingFace model works!
zse serve Qwen/Qwen2.5-7B-Instruct
zse serve meta-llama/Llama-3.1-8B-Instruct
zse serve mistralai/Mistral-7B-Instruct-v0.3
zse serve microsoft/Phi-3-mini-4k-instruct
zse serve google/gemma-2-9b-it

# With memory optimization
zse serve Qwen/Qwen2.5-32B-Instruct --max-memory 24GB

# With recommendations
zse serve meta-llama/Llama-3.1-70B-Instruct --recommend

# Ultra memory efficiency
zse serve deepseek-ai/DeepSeek-V2-Lite --efficiency ultra

# GGUF models (via llama.cpp)
zse serve ./model-Q4_K_M.gguf
```

> **💡 Supported Models:** Any HuggingFace transformers model, safetensors, GGUF, or .zse format. Popular choices: Qwen, Llama, Mistral, Phi, Gemma, DeepSeek, Yi, and more.

### Interactive Chat

```bash
zse chat Qwen/Qwen2.5-7B-Instruct
```

### Convert to ZSE Format

```bash
zse convert Qwen/Qwen2.5-32B-Instruct -o qwen-32b.zse --target-memory 24GB
```

### Check Hardware

```bash
zse hardware
```

## API Server

ZSE provides an OpenAI-compatible API:

```bash
zse serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="zse")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Efficiency Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `speed` | Maximum throughput | Production with ample GPU memory |
| `balanced` | Good throughput, moderate memory | Standard deployment (default) |
| `memory` | Low memory, reduced throughput | Consumer GPUs |
| `ultra` | Extreme memory savings | 4GB GPUs, laptops |

```bash
zse serve model --efficiency memory
```

## Deployment

### Developer Mode

```bash
zse serve model --mode dev
```

- No authentication required
- SQLite database
- Hot reload enabled
- Debug logging

### Enterprise Mode

```bash
zse serve model --config configs/enterprise.yaml
```

- API key authentication
- PostgreSQL + Redis
- Prometheus metrics
- Rate limiting
- Multi-tenancy

## Architecture

```
zse/
├── core/                   # ZSE Native Engine (100% custom)
│   ├── zattention/         # Custom attention kernels
│   ├── zquantize/          # Quantization (GPTQ, HQQ, INT2-8)
│   ├── zkv/                # Paged + quantized KV cache
│   ├── zstream/            # Layer streaming + prefetch
│   ├── zscheduler/         # Continuous batching
│   └── zdistributed/       # Tensor/pipeline parallelism
├── models/                 # Model loaders + architectures
├── engine/                 # Executor + Orchestrator
├── api/                    # CLI, FastAPI server, Web UI
└── enterprise/             # Auth, monitoring, scaling
```

## GGUF Support

GGUF models are supported via llama.cpp backend:

```bash
pip install zse[gguf]
zse serve ./model.gguf
```

Note: GGUF uses llama.cpp for inference. Native ZSE engine handles HuggingFace, safetensors, and .zse formats.

## Docker Deployment

```bash
# CPU
docker run -p 8000:8000 ghcr.io/zyora-dev/zse:latest

# GPU (NVIDIA)
docker run --gpus all -p 8000:8000 ghcr.io/zyora-dev/zse:gpu

# With model pre-loaded
docker run -p 8000:8000 -e ZSE_MODEL=Qwen/Qwen2.5-0.5B-Instruct ghcr.io/zyora-dev/zse:latest
```

**Docker Compose:**
```bash
docker-compose up -d                    # CPU
docker-compose --profile gpu up -d      # GPU
```

See [deploy/DEPLOY.md](deploy/DEPLOY.md) for full deployment guide including Runpod, Vast.ai, Railway, Render, and Kubernetes.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=zse

# Type checking
mypy zse

# Linting
ruff check zse
```

## License

Apache 2.0

## Acknowledgments

- PagedAttention concept from vLLM (UC Berkeley)
- Flash Attention from Tri Dao
- GPTQ, HQQ, and other quantization research
