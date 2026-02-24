# ZSE Deployment Guide

One-click deployment options for ZSE - Ultra memory-efficient LLM inference engine.

## Quick Deploy Buttons

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template?repo=https://github.com/Zyora-Dev/zse)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Zyora-Dev/zse)

## Platform Guides

### Docker (Local/Any Server)

```bash
# CPU
docker run -p 8000:8000 ghcr.io/zyora-dev/zse:latest

# GPU (NVIDIA)
docker run --gpus all -p 8000:8000 ghcr.io/zyora-dev/zse:gpu

# With model pre-loaded
docker run -p 8000:8000 -e ZSE_MODEL=Qwen/Qwen2.5-0.5B-Instruct ghcr.io/zyora-dev/zse:latest
```

### Docker Compose

```bash
# CPU
docker-compose up -d

# GPU
docker-compose --profile gpu up -d

# With specific model
ZSE_MODEL=Qwen/Qwen2.5-7B-Instruct docker-compose --profile gpu up -d
```

---

## GPU Cloud Platforms

### Runpod

1. Go to [Runpod Templates](https://www.runpod.io/console/templates)
2. Click "New Template"
3. Configure:
   - **Name**: ZSE - LLM Inference Engine
   - **Container Image**: `ghcr.io/zyora-dev/zse:gpu`
   - **Docker Command**: `zse serve --host 0.0.0.0 --port 8000`
   - **Container Disk**: 50 GB (for model cache)
   - **Expose HTTP Ports**: 8000
   - **Environment Variables**:
     ```
     ZSE_MODEL=Qwen/Qwen2.5-7B-Instruct
     ZSE_QUANTIZATION=int4
     ```

4. Deploy a Pod with your template

**Recommended GPU Types:**
| Model Size | GPU | VRAM | Quantization |
|------------|-----|------|--------------|
| 0.5B-3B | RTX 3060 | 12 GB | FP16 |
| 7B | RTX 3090/4090 | 24 GB | INT8 |
| 13B | RTX 3090/4090 | 24 GB | INT4 |
| 32B | A100 40GB | 40 GB | INT4 |

### Vast.ai

1. Go to [Vast.ai Console](https://vast.ai/console/create/)
2. Select a GPU instance
3. Use Docker image: `ghcr.io/zyora-dev/zse:gpu`
4. Set on-start script:
   ```bash
   zse serve --host 0.0.0.0 --port 8000
   ```
5. Expose port 8000
6. Launch instance

### Modal

Already integrated! See `deploy/modal_app.py`:

```python
modal run deploy/modal_app.py
```

Or deploy as a persistent endpoint:

```python
modal deploy deploy/modal_app.py
```

---

## PaaS Platforms

### Railway

1. Click the deploy button above, or:
2. Fork repo → Connect to Railway → Deploy

**Environment Variables:**
- `ZSE_MODEL`: Model to pre-load (optional)
- `ZSE_QUANTIZATION`: `auto`, `int4`, `int8`, `fp16`

### Render

1. Click the deploy button above, or:
2. Connect repo → Use `render.yaml` blueprint

**Note:** Render free tier has limited resources. Use Standard+ for LLM inference.

### DigitalOcean App Platform

1. Create new App
2. Connect GitHub repo
3. Select Dockerfile deployment
4. Configure resources (minimum 2 GB RAM)
5. Deploy

---

## Kubernetes (Helm)

Coming soon! For now, use the Docker image directly:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zse
spec:
  replicas: 1
  selector:
    matchLabels:
      app: zse
  template:
    metadata:
      labels:
        app: zse
    spec:
      containers:
      - name: zse
        image: ghcr.io/zyora-dev/zse:gpu
        ports:
        - containerPort: 8000
        env:
        - name: ZSE_MODEL
          value: "Qwen/Qwen2.5-7B-Instruct"
        resources:
          limits:
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: zse
spec:
  selector:
    app: zse
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ZSE_MODEL` | (none) | Model to pre-load on startup |
| `ZSE_QUANTIZATION` | `auto` | `auto`, `int4`, `int8`, `nf4`, `fp16` |
| `ZSE_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `ZSE_HOST` | `0.0.0.0` | Server bind address |
| `ZSE_PORT` | `8000` | Server port |
| `HF_TOKEN` | (none) | HuggingFace token for gated models |

---

## After Deployment

1. **Access Dashboard**: `http://your-server:8000/dashboard`
2. **Load Model**: Use dashboard or API
3. **Test API**: 
   ```bash
   curl http://your-server:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'
   ```

4. **Enable Auth** (optional):
   ```bash
   # Create API key
   docker exec zse-server zse api-key create my-app
   
   # Use in requests
   curl -H "X-API-Key: zse-xxx..." ...
   ```
