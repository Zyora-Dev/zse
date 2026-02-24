# ZSE Model Guide

**Complete guide to finding, downloading, converting, and serving models with ZSE.**

## Quick Start

```bash
# 1. Browse available models
zse models list -r                    # Show recommended models
zse models list -v 8                  # Models fitting in 8GB VRAM

# 2. Download a model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# 3. Serve it
zse serve Qwen/Qwen2.5-7B-Instruct
```

## Table of Contents

1. [Finding Models](#finding-models)
2. [Downloading Models](#downloading-models)
3. [Converting to .zse Format](#converting-to-zse-format)
4. [Serving Models](#serving-models)
5. [Model Registry Reference](#model-registry-reference)

---

## Finding Models

### CLI Commands

```bash
# List all curated models
zse models list

# Filter by category
zse models list -c chat        # Chat models
zse models list -c code        # Code models
zse models list -c instruct    # Instruction-tuned models
zse models list -c reasoning   # Reasoning-focused models

# Filter by size tier
zse models list -s tiny        # < 1B params, < 2GB VRAM
zse models list -s small       # 1-3B params, 2-6GB VRAM
zse models list -s medium      # 3-8B params, 6-16GB VRAM
zse models list -s large       # 8-13B params, 16-26GB VRAM
zse models list -s xlarge      # 13-34B params, 26-70GB VRAM

# Filter by VRAM requirement
zse models list -v 8           # Fits in 8GB VRAM (INT8)
zse models list -v 24          # Fits in 24GB VRAM

# Show only recommended models
zse models list -r
zse models list -r -v 16       # Recommended, fits in 16GB

# Search HuggingFace for more models
zse models search llama
zse models search "code python"
zse models search mistral --limit 30

# Check if a specific model is compatible
zse models check meta-llama/Llama-3.1-8B-Instruct

# Get detailed info about a model
zse models info Qwen/Qwen2.5-7B-Instruct
```

### API Endpoints

```bash
# List curated registry
curl http://localhost:8000/api/models/registry

# Filter by category
curl http://localhost:8000/api/models/registry?category=code

# Filter by VRAM
curl http://localhost:8000/api/models/registry?max_vram=16

# Get recommended models
curl http://localhost:8000/api/models/registry?recommended=true

# Search HuggingFace
curl "http://localhost:8000/api/models/search?q=llama&limit=20"

# Check model compatibility
curl http://localhost:8000/api/models/check/meta-llama/Llama-3.1-8B-Instruct
```

---

## Downloading Models

### Using HuggingFace CLI (Recommended)

```bash
# Install huggingface-cli if needed
pip install huggingface-hub

# Download a model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Download to specific directory
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/qwen-7b

# Download with authentication (for gated models like Llama)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

### Default Download Location

Models are cached in: `~/.cache/huggingface/hub/`

Full path pattern:
```
~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<hash>/
```

Example:
```
~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/abcd1234/
```

### For Gated Models (Llama, etc.)

1. Create account at https://huggingface.co
2. Accept model license at model page (e.g., https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
3. Generate access token at https://huggingface.co/settings/tokens
4. Login: `huggingface-cli login`
5. Download as normal

---

## Converting to .zse Format

The `.zse` format provides optimized model storage for fast streaming loading.

### Why Convert to .zse?

| Benefit | Description |
|---------|-------------|
| **Faster Loading** | Stream from disk/cloud, no decompression |
| **Lower Memory** | Memory-mapped, load only what's needed |
| **Better Caching** | Layer-by-layer streaming |
| **Quantization Built-in** | INT8/INT4 quantized in format |

### Convert Command

```bash
# Convert a downloaded model
zse convert ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>/ \
    -f zse \
    -o ./models/qwen-7b.zse

# Convert with quantization
zse convert <model_path> -f zse -q int8 -o model-int8.zse
zse convert <model_path> -f zse -q int4 -o model-int4.zse

# Convert directly from HuggingFace ID (downloads + converts)
zse convert Qwen/Qwen2.5-7B-Instruct -f zse -o qwen-7b.zse
```

### Conversion Options

```bash
zse convert <source> -f zse [options]

Options:
  -o, --output PATH       Output file path
  -q, --quantization      Quantization: fp16, int8, int4
  --chunk-size SIZE       Chunk size in MB (default: 64)
  --compress              Enable compression
```

### Example Complete Workflow

```bash
# 1. Find a model
zse models list -c code -v 8
# Shows: Qwen/Qwen2.5-Coder-7B-Instruct

# 2. Download it
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct

# 3. Find the downloaded path
ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/

# 4. Convert to .zse
zse convert ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/abc123/ \
    -f zse \
    -q int8 \
    -o ./models/qwen-coder-7b.zse

# 5. Serve the .zse model
zse serve ./models/qwen-coder-7b.zse
```

---

## Serving Models

### Basic Serving

```bash
# Serve by HuggingFace ID (auto-downloads if needed)
zse serve Qwen/Qwen2.5-7B-Instruct

# Serve a local .zse file
zse serve ./models/qwen-7b.zse

# Serve with specific port
zse serve Qwen/Qwen2.5-7B-Instruct -p 8080

# Serve with quantization
zse serve Qwen/Qwen2.5-7B-Instruct -q int8
```

### Server Options

```bash
zse serve <model> [options]

Options:
  -p, --port INT          Port number (default: 8000)
  -h, --host TEXT         Host address (default: 0.0.0.0)
  -q, --quantization      Quantization: auto, fp16, int8, int4
  --max-batch-size INT    Max batch size for batching
  --context-length INT    Override context length
  --trust-remote-code     Trust remote code in model
```

### API Usage

Once serving, use the OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"  # or "sk-" if auth disabled
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

```bash
# Or with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Model Registry Reference

### Recommended Models by Use Case

#### Chat & General Purpose
| Model | Size | VRAM (INT8) | Notes |
|-------|------|-------------|-------|
| Qwen/Qwen2.5-7B-Instruct | 7B | 8 GB | Best overall balance |
| meta-llama/Llama-3.1-8B-Instruct | 8B | 9 GB | Long context (128K) |
| mistralai/Mistral-7B-Instruct-v0.3 | 7B | 8 GB | Excellent quality |

#### Code Generation
| Model | Size | VRAM (INT8) | Notes |
|-------|------|-------------|-------|
| Qwen/Qwen2.5-Coder-7B-Instruct | 7B | 8 GB | Best code model |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.5B | 1.8 GB | Fast completions |
| deepseek-ai/deepseek-coder-6.7b-instruct | 6.7B | 7.5 GB | Fill-in-middle |

#### Small/Edge Devices
| Model | Size | VRAM (INT8) | Notes |
|-------|------|-------------|-------|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | 1.2 GB | Testing & edge |
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | 0.6 GB | Ultra compact |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | 1.8 GB | Good quality |

#### Large/High Quality
| Model | Size | VRAM (INT8) | Notes |
|-------|------|-------------|-------|
| Qwen/Qwen2.5-32B-Instruct | 32B | 36 GB | Near-frontier |
| meta-llama/Llama-3.1-70B-Instruct | 70B | 78 GB | Frontier quality |
| Qwen/Qwen2.5-72B-Instruct | 72B | 80 GB | Best multilingual |

### VRAM Requirements Guide

| GPU | VRAM | Recommended Models |
|-----|------|-------------------|
| RTX 3060/4060 | 8-12 GB | 7B models with INT8 |
| RTX 3080/4080 | 10-16 GB | 7-14B models with INT8 |
| RTX 3090/4090 | 24 GB | 14B+ models, 7B with FP16 |
| A100 40GB | 40 GB | 32B models |
| A100 80GB | 80 GB | 70B models |

### Supported Architectures

ZSE supports these HuggingFace architectures:

- LlamaForCausalLM (Llama 1/2/3, TinyLlama, etc.)
- MistralForCausalLM (Mistral 7B)
- MixtralForCausalLM (Mixtral 8x7B MoE)
- Qwen2ForCausalLM (Qwen 2/2.5)
- Phi3ForCausalLM (Phi-3)
- Gemma2ForCausalLM (Gemma 2)
- GemmaForCausalLM (Gemma 1)
- FalconForCausalLM (Falcon)
- StarCoder2ForCausalLM (StarCoder 2)

---

## Troubleshooting

### Model Not Compatible

```bash
zse models check <model_id>
# Shows compatibility issues and recommendations
```

Common issues:
- Architecture not supported → Check supported architectures above
- No safetensors files → Model will load slower from .bin files
- Gated model → Login with `huggingface-cli login`

### Out of Memory

1. Use INT8 quantization: `zse serve <model> -q int8`
2. Use INT4 for larger models: `zse serve <model> -q int4`
3. Choose a smaller model: `zse models list -v <your_vram>`

### Slow Loading

1. Convert to .zse format for streaming
2. Use SSD storage
3. Pre-download models before serving

---

## Next Steps

- [API Documentation](/docs) - Available at http://localhost:8000/docs
- [Batching Guide](./BATCHING.md) - Enable request batching for throughput
- [Progress Tracker](./progress.md) - Project status and roadmap
