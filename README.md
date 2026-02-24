# ZSE - Z Server Engine

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

**Ultra memory-efficient LLM inference engine.**

ZSE is designed to run large language models with minimal memory footprint while maintaining high performance. Our key innovation is the **Intelligence Orchestrator** that provides smart recommendations based on your available (not total) memory.

## Key Features

- 🧠 **zAttention**: Custom CUDA kernels for paged, flash, and sparse attention
- 🗜️ **zQuantize**: Per-tensor INT2-8 mixed precision quantization
- 💾 **zKV**: Quantized KV cache with sliding precision (4x memory savings)
- 🌊 **zStream**: Layer streaming with async prefetch (run 70B on 24GB GPU)
- 🎯 **zOrchestrator**: Smart recommendations based on FREE memory
- 📊 **Efficiency Modes**: speed / balanced / memory / ultra

## Memory Targets

| Model Size | Standard FP16 | ZSE Target | Reduction |
|------------|---------------|------------|-----------|
| 7B | 14+ GB | **3 - 3.5 GB** | ~75% |
| 14B | 28+ GB | **6 GB** | ~78% |
| 32B | 64+ GB | **16 - 20 GB** | ~70% |
| 70B | 140+ GB | **24 - 32 GB** | ~77% |

## Installation

```bash
pip install zse
```

With CUDA support (recommended):
```bash
pip install zse[cuda]
```

For development:
```bash
git clone https://github.com/zse-team/zse.git
cd zse
pip install -e ".[dev]"
```

## Quick Start

### Start Server

```bash
# Basic usage
zse serve meta-llama/Llama-3-8B

# With memory target
zse serve meta-llama/Llama-3-70B --max-memory 24GB

# With recommendations
zse serve meta-llama/Llama-3-70B --recommend

# Ultra memory efficiency
zse serve meta-llama/Llama-3-70B --efficiency ultra
```

### Interactive Chat

```bash
zse chat meta-llama/Llama-3-8B
```

### Convert to ZSE Format

```bash
zse convert meta-llama/Llama-3-70B -o llama-70b.zse --target-memory 24GB
```

### Check Hardware

```bash
zse hardware
```

## API Server

ZSE provides an OpenAI-compatible API:

```bash
zse serve meta-llama/Llama-3-8B --port 8000
```

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="zse")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B",
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

- Authentication (API key, OAuth2, SSO)
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
