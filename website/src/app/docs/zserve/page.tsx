'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { Steps, FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'basic-usage', title: 'Basic Usage', level: 2 },
  { id: 'configuration', title: 'Configuration', level: 2 },
  { id: 'multi-model', title: 'Multi-Model Serving', level: 2 },
  { id: 'scaling', title: 'Scaling', level: 2 },
  { id: 'monitoring', title: 'Monitoring', level: 2 },
]

export default function ZServePage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="zServe"
          description="Production-ready inference server with OpenAI-compatible API and sub-4-second cold start."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            <InlineCode>zServe</InlineCode> is ZSE&apos;s high-performance inference server, 
            designed for production workloads with minimal latency and maximum throughput.
          </p>

          <CardGrid columns={3}>
            <Card
              title="3.9s Cold Start"
              description="Fastest model loading in the industry"
            />
            <Card
              title="OpenAI Compatible"
              description="Drop-in replacement for OpenAI API"
            />
            <Card
              title="Multi-GPU"
              description="Automatic model parallelism"
            />
          </CardGrid>

          <FeatureList features={[
            "OpenAI-compatible /v1/chat/completions endpoint",
            "Streaming responses with SSE",
            "Request batching and queuing",
            "Built-in rate limiting",
            "Prometheus metrics endpoint",
            "Health checks and graceful shutdown",
          ]} />
        </DocSection>

        <DocSection id="basic-usage" title="Basic Usage">
          <Steps steps={[
            {
              title: "Start the server",
              description: "Launch with a model",
              code: "zse serve qwen-7b.zse"
            },
            {
              title: "Test the endpoint",
              description: "Send a request with curl",
              code: `curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "Hello!"}]}'`
            },
            {
              title: "Use with OpenAI client",
              description: "Point your existing code to the server",
              code: `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Hello!"}]
)`
            },
          ]} />

          <Callout type="info">
            The server starts on port 8000 by default. Use <InlineCode>--port</InlineCode> to change.
          </Callout>
        </DocSection>

        <DocSection id="configuration" title="Configuration">
          <DocSubSection id="cli-options" title="CLI Options">
            <CodeBlock
              language="bash"
              code={`zse serve model.zse \\
  --port 8000           # Server port
  --host 0.0.0.0        # Bind address
  --workers 4           # Worker processes
  --max-batch 32        # Max batch size
  --max-concurrent 100  # Max concurrent requests
  --timeout 60          # Request timeout (seconds)
  --api-key "sk-xxx"    # Require API key`}
            />
          </DocSubSection>

          <DocSubSection id="config-file" title="Config File">
            <p className="mb-2">
              Use a YAML config file for complex setups:
            </p>

            <CodeBlock
              language="yaml"
              filename="zse.yaml"
              code={`server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
model:
  path: ./qwen-7b.zse
  max_batch_size: 32
  max_sequence_length: 4096
  
inference:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2048
  
limits:
  max_concurrent_requests: 100
  requests_per_minute: 1000
  timeout_seconds: 60
  
auth:
  api_keys:
    - sk-key-1
    - sk-key-2`}
            />

            <CodeBlock
              language="bash"
              code={`zse serve --config zse.yaml`}
            />
          </DocSubSection>

          <DocSubSection id="environment" title="Environment Variables">
            <CodeBlock
              language="bash"
              code={`# Model configuration
export ZSE_MODEL_PATH="./qwen-7b.zse"
export ZSE_MAX_BATCH_SIZE=32

# Server configuration
export ZSE_PORT=8000
export ZSE_HOST=0.0.0.0

# Auth
export ZSE_API_KEY="sk-xxx"

# Start server
zse serve`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="multi-model" title="Multi-Model Serving">
          <p className="mb-4">
            Serve multiple models from a single server instance:
          </p>

          <CodeBlock
            language="bash"
            code={`# Serve multiple models from a directory
zse serve ./models/

# Or specify individually
zse serve model1.zse model2.zse model3.zse`}
          />

          <p className="mt-4">
            Models are loaded on-demand with LRU caching:
          </p>

          <CodeBlock
            language="yaml"
            filename="zse.yaml"
            code={`models:
  - name: qwen-7b
    path: ./qwen-7b.zse
    max_loaded: true      # Keep loaded
    
  - name: llama-8b
    path: ./llama-8b.zse
    max_loaded: false     # Load on demand
    
  - name: codellama-34b
    path: ./codellama-34b.zse
    gpu: [0, 1]           # Specific GPUs

cache:
  max_models: 3           # Max models in memory
  eviction: lru           # Eviction policy`}
          />

          <Callout type="tip">
            Pin frequently-used models with <InlineCode>max_loaded: true</InlineCode> to 
            avoid cold starts.
          </Callout>
        </DocSection>

        <DocSection id="scaling" title="Scaling">
          <DocSubSection id="multi-gpu" title="Multi-GPU">
            <p className="mb-2">
              Automatically shard large models across GPUs:
            </p>

            <CodeBlock
              language="bash"
              code={`# Auto-detect and use all GPUs
zse serve model.zse --tensor-parallel auto

# Specify GPUs
zse serve model.zse --tensor-parallel 4 --gpus 0,1,2,3

# Pipeline parallelism for very large models
zse serve model.zse --pipeline-parallel 2`}
            />
          </DocSubSection>

          <DocSubSection id="load-balancing" title="Load Balancing">
            <p className="mb-2">
              Run multiple instances behind a load balancer:
            </p>

            <CodeBlock
              language="nginx"
              filename="nginx.conf"
              code={`upstream zse {
    least_conn;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    
    location /v1/ {
        proxy_pass http://zse;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # For streaming
        proxy_buffering off;
        proxy_cache off;
    }
}`}
            />
          </DocSubSection>

          <DocSubSection id="kubernetes" title="Kubernetes">
            <CodeBlock
              language="yaml"
              filename="deployment.yaml"
              code={`apiVersion: apps/v1
kind: Deployment
metadata:
  name: zse
spec:
  replicas: 3
  selector:
    matchLabels:
      app: zse
  template:
    spec:
      containers:
      - name: zse
        image: zllm/zse:latest
        command: ["zse", "serve", "/models/qwen-7b.zse"]
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models
          mountPath: /models
        readinessProbe:
          httpGet:
            path: /health
            port: 8000`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="monitoring" title="Monitoring">
          <DocSubSection id="health-endpoint" title="Health Endpoint">
            <CodeBlock
              language="bash"
              code={`curl http://localhost:8000/health`}
            />

            <CodeBlock
              language="json"
              code={`{
  "status": "healthy",
  "model": "qwen-7b",
  "uptime": 3600,
  "requests_processed": 15420,
  "gpu_memory_used": "4.2 GB",
  "gpu_memory_total": "24 GB"
}`}
            />
          </DocSubSection>

          <DocSubSection id="metrics" title="Prometheus Metrics">
            <CodeBlock
              language="bash"
              code={`curl http://localhost:8000/metrics`}
            />

            <CodeBlock
              language="text"
              code={`# TYPE zse_requests_total counter
zse_requests_total{model="qwen-7b",status="success"} 15420
zse_requests_total{model="qwen-7b",status="error"} 12

# TYPE zse_request_duration_seconds histogram
zse_request_duration_seconds_bucket{le="0.1"} 1000
zse_request_duration_seconds_bucket{le="0.5"} 12000
zse_request_duration_seconds_bucket{le="1.0"} 15000

# TYPE zse_tokens_generated_total counter
zse_tokens_generated_total{model="qwen-7b"} 2456789

# TYPE zse_gpu_memory_bytes gauge
zse_gpu_memory_bytes{gpu="0"} 4500000000`}
            />
          </DocSubSection>

          <DocSubSection id="logging" title="Logging">
            <CodeBlock
              language="bash"
              code={`# Verbose logging
zse serve model.zse --log-level debug

# JSON logs for production
zse serve model.zse --log-format json

# Log to file
zse serve model.zse --log-file /var/log/zse/server.log`}
            />
          </DocSubSection>
        </DocSection>

        <DocNav
          prev={{ title: 'zQuantize', href: '/docs/zquantize' }}
          next={{ title: 'zInfer', href: '/docs/zinfer' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
