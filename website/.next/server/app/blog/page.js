(()=>{var e={};e.id=404,e.ids=[404],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},1559:(e,t,n)=>{"use strict";n.r(t),n.d(t,{GlobalError:()=>r.a,__next_app__:()=>u,originalPathname:()=>m,pages:()=>c,routeModule:()=>p,tree:()=>d});var s=n(482),a=n(9108),o=n(2563),r=n.n(o),i=n(8300),l={};for(let e in i)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(l[e]=()=>i[e]);n.d(t,l);let d=["",{children:["blog",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(n.bind(n,1808)),"/Users/redfoxhotels/zse/website/src/app/blog/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(n.bind(n,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(n.t.bind(n,9361,23)),"next/dist/client/components/not-found-error"]}],c=["/Users/redfoxhotels/zse/website/src/app/blog/page.tsx"],m="/blog/page",u={require:n,loadChunk:()=>Promise.resolve()},p=new s.AppPageRouteModule({definition:{kind:a.x.APP_PAGE,page:"/blog/page",pathname:"/blog",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:d}})},5233:(e,t,n)=>{Promise.resolve().then(n.bind(n,805))},805:(e,t,n)=>{"use strict";n.r(t),n.d(t,{default:()=>d});var s=n(5344),a=n(6506),o=n(1912),r=n(771),i=n(1453);function l({post:e,index:t}){return s.jsx(o.E.article,{initial:{opacity:0,y:20},animate:{opacity:1,y:0},transition:{duration:.5,delay:.1*t},className:"group",children:s.jsx(a.default,{href:`/blog/${e.slug}`,children:(0,s.jsxs)("div",{className:"border border-white/10 rounded-xl p-6 hover:border-lime/50 transition-all duration-300 bg-white/[0.02] hover:bg-white/[0.04]",children:[e.featured&&s.jsx("span",{className:"inline-block px-3 py-1 text-xs font-medium bg-lime/20 text-lime rounded-full mb-4",children:"Featured"}),s.jsx("h2",{className:"text-xl font-semibold text-white mb-3 group-hover:text-lime transition-colors",children:e.title}),s.jsx("p",{className:"text-white/60 mb-4 line-clamp-2",children:e.excerpt}),(0,s.jsxs)("div",{className:"flex items-center justify-between text-sm text-white/40",children:[(0,s.jsxs)("div",{className:"flex items-center gap-2",children:[s.jsx("span",{children:e.author}),s.jsx("span",{children:"•"}),s.jsx("span",{children:(0,i.p)(e.date)})]}),s.jsx("span",{children:e.readTime})]}),s.jsx("div",{className:"flex gap-2 mt-4",children:e.tags.map(e=>s.jsx("span",{className:"px-2 py-1 text-xs bg-white/5 text-white/50 rounded",children:e},e))})]})})})}function d(){let e=r.nd.filter(e=>e.featured),t=r.nd.filter(e=>!e.featured);return s.jsx("div",{className:"min-h-screen bg-black pt-24 pb-16",children:(0,s.jsxs)("div",{className:"max-w-5xl mx-auto px-6",children:[(0,s.jsxs)(o.E.div,{initial:{opacity:0,y:20},animate:{opacity:1,y:0},className:"mb-12",children:[s.jsx("h1",{className:"text-4xl md:text-5xl font-bold text-white mb-4",children:"Blog"}),s.jsx("p",{className:"text-xl text-white/60",children:"Updates, tutorials, and insights from the ZSE team"})]}),e.length>0&&(0,s.jsxs)("section",{className:"mb-12",children:[s.jsx("h2",{className:"text-lg font-semibold text-lime mb-6",children:"Featured"}),s.jsx("div",{className:"space-y-6",children:e.map((e,t)=>s.jsx(l,{post:e,index:t},e.id))})]}),(0,s.jsxs)("section",{children:[s.jsx("h2",{className:"text-lg font-semibold text-white/80 mb-6",children:"All Posts"}),s.jsx("div",{className:"space-y-6",children:t.map((t,n)=>s.jsx(l,{post:t,index:n+e.length},t.id))})]}),0===r.nd.length&&s.jsx("div",{className:"text-center py-12",children:s.jsx("p",{className:"text-white/60",children:"No blog posts yet. Check back soon!"})})]})})}},771:(e,t,n)=>{"use strict";n.d(t,{nd:()=>s,zl:()=>a});let s=[{id:"1",title:"Introducing ZSE: 3.9s Cold Starts for LLM Inference",slug:"introducing-zse",excerpt:"We're excited to announce ZSE, a new inference engine that loads 7B models in under 4 seconds with the .zse format.",content:`
# Introducing ZSE

Today we're releasing ZSE (Z Server Engine), an ultra memory-efficient LLM inference engine that achieves **3.9 second cold starts** for 7B models.

## The Problem

Loading large language models is slow. A typical 7B model with bitsandbytes takes 45+ seconds to load. This makes serverless deployments expensive and development iteration slow.

## Our Solution

ZSE introduces the \`.zse\` format - pre-quantized model files that skip runtime quantization entirely. The result:

- **7B models**: 3.9s cold start (11.6\xd7 faster)
- **32B models**: 21.4s cold start (5.6\xd7 faster)
- **63-72% memory savings** compared to FP16

## How It Works

1. **Pre-quantization**: Convert once, load fast forever
2. **Memory mapping**: Direct tensor loading from disk
3. **Lazy initialization**: Only load what's needed
4. **OpenAI-compatible API**: Drop-in replacement

## Try It Now

\`\`\`bash
pip install zllm-zse
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse
zse serve qwen-7b.zse
\`\`\`

We're just getting started. Follow us for updates on zStream, zKV, and more features.
    `,author:"ZSE Team",authorImage:"/images/zllm-logo.png",date:"2026-02-25",readTime:"5 min read",tags:["announcement","performance"],featured:!0},{id:"2",title:"Complete Guide: Running Your First Model with ZSE",slug:"getting-started-tutorial",excerpt:"Step-by-step tutorial to install ZSE, convert a model, and start generating text in under 5 minutes.",content:`
# Complete Guide: Running Your First Model with ZSE

This guide walks you through installing ZSE and running your first LLM inference in under 5 minutes.

## Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU) or CPU-only support
- 8GB+ VRAM for 7B models

## Step 1: Install ZSE

\`\`\`bash
pip install zllm-zse
\`\`\`

Verify installation:
\`\`\`bash
zse --version
zse hardware  # Check GPU detection
\`\`\`

## Step 2: Convert a Model

Convert a HuggingFace model to .zse format:

\`\`\`bash
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse
\`\`\`

This downloads the model (~14GB), quantizes it to NF4, and saves a 4.2GB .zse file.

## Step 3: Start the Server

\`\`\`bash
zse serve qwen-7b.zse --port 8000
\`\`\`

## Step 4: Send a Request

Using curl:
\`\`\`bash
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "Hello!"}]}'
\`\`\`

Or with Python:
\`\`\`python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
print(response.choices[0].message.content)
\`\`\`

## Next Steps

- Enable streaming with \`stream=True\`
- Try different quantization: \`--quant int4\` or \`--quant int8\`
- Read the [Documentation](/docs) for advanced features
    `,author:"ZSE Team",authorImage:"/images/zllm-logo.png",date:"2026-02-24",readTime:"6 min read",tags:["tutorial","getting-started"],featured:!0},{id:"3",title:"Running 70B Models on a 24GB GPU with ZSE",slug:"running-70b-on-24gb",excerpt:"How to run Llama 70B and other large models on consumer GPUs using ZSE's memory optimization features.",content:`
# Running 70B Models on a 24GB GPU

Yes, you can run 70B parameter models on a single RTX 4090. Here's how.

## The Challenge

A 70B model in FP16 needs ~140GB VRAM. Even with 4-bit quantization, that's ~35GB.

## ZSE's Solution

Combine multiple techniques:

### 1. NF4 Quantization
\`\`\`bash
zse convert meta-llama/Llama-3.1-70B-Instruct -o llama-70b.zse --quant nf4
\`\`\`
Brings weights down to ~35GB.

### 2. CPU Offloading
\`\`\`bash
zse serve llama-70b.zse --offload-layers 20
\`\`\`
Keep attention-heavy layers on GPU, offload FFN layers to CPU RAM.

### 3. 4-bit KV Cache
\`\`\`bash
zse serve llama-70b.zse --kv-quant int4 --max-context 4096
\`\`\`
Reduces KV cache memory by 4\xd7.

## Full Command

\`\`\`bash
zse serve llama-70b.zse \\
  --offload-layers 20 \\
  --kv-quant int4 \\
  --max-context 4096 \\
  --max-batch 4
\`\`\`

## Performance Expectations

| GPU | Throughput | Latency |
|-----|------------|---------|
| RTX 4090 24GB | ~15 tok/s | ~200ms TTFT |
| RTX 3090 24GB | ~10 tok/s | ~300ms TTFT |
| A100 80GB | ~45 tok/s | ~80ms TTFT |

## Tips for Best Results

1. **Use SSD storage** - NVMe makes offloading faster
2. **Allocate enough RAM** - 64GB system RAM recommended
3. **Reduce batch size** - Trade throughput for memory
4. **Limit context length** - Shorter contexts use less KV cache

Now you can run frontier models locally!
    `,author:"ZSE Team",authorImage:"/images/zllm-logo.png",date:"2026-02-23",readTime:"7 min read",tags:["tutorial","memory","advanced"],featured:!1},{id:"4",title:"ZSE Quantization Guide: NF4 vs INT4 vs INT8",slug:"quantization-guide",excerpt:"Understanding the tradeoffs between different quantization types and when to use each one.",content:`
# ZSE Quantization Guide

Choosing the right quantization is key to balancing quality, speed, and memory.

## Available Quantization Types

### NF4 (NormalFloat4) - Default
\`\`\`bash
zse convert model -o model.zse --quant nf4
\`\`\`
- **Bits**: 4
- **Quality**: ★★★★☆ (best 4-bit)
- **Size**: ~0.56GB per billion params
- **Use case**: Most models, production deployments

NF4 uses an asymmetric quantization grid optimized for the weight distribution of neural networks.

### INT4
\`\`\`bash
zse convert model -o model.zse --quant int4
\`\`\`
- **Bits**: 4
- **Quality**: ★★★☆☆
- **Size**: ~0.53GB per billion params
- **Use case**: Maximum compression, less sensitive tasks

### INT8
\`\`\`bash
zse convert model -o model.zse --quant int8
\`\`\`
- **Bits**: 8
- **Quality**: ★★★★★ (near FP16)
- **Size**: ~1.1GB per billion params
- **Use case**: When quality is critical

### FP16 (No Quantization)
\`\`\`bash
zse convert model -o model.zse --quant fp16
\`\`\`
- **Bits**: 16
- **Quality**: ★★★★★ (original)
- **Size**: ~2GB per billion params
- **Use case**: Fine-tuning, debugging

## Quality Comparison (Qwen 7B)

| Quant | Perplexity | MMLU | Size |
|-------|------------|------|------|
| FP16 | 5.38 | 64.8% | 14GB |
| INT8 | 5.39 | 64.7% | 7.5GB |
| NF4 | 5.42 | 64.2% | 4.2GB |
| INT4 | 5.51 | 63.5% | 4.0GB |

## Recommendations

- **General use**: NF4 (best quality/size ratio)
- **Code generation**: INT8 (higher precision helps)
- **Embeddings**: INT8 or FP16
- **Chat/creative**: NF4 is plenty
    `,author:"ZSE Team",authorImage:"/images/zllm-logo.png",date:"2026-02-22",readTime:"8 min read",tags:["technical","quantization","guide"],featured:!1},{id:"5",title:"Building a Local RAG Chatbot with ZSE",slug:"building-rag-chatbot",excerpt:"Create a retrieval-augmented generation chatbot that answers questions about your documents.",content:`
# Building a Local RAG Chatbot with ZSE

Build a chatbot that can answer questions about your documents using ZSE's built-in RAG features.

## What We're Building

A chatbot that:
1. Indexes your PDF/text documents
2. Retrieves relevant context for questions
3. Generates accurate answers using an LLM

## Step 1: Prepare Your Model

\`\`\`bash
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse
\`\`\`

## Step 2: Index Documents

\`\`\`python
from zllm_zse import ZSE, RAGIndex

# Load model
model = ZSE("qwen-7b.zse")

# Create index
index = RAGIndex(embedding_model="sentence-transformers/all-MiniLM-L6-v2")

# Add documents
index.add_documents([
    "docs/manual.pdf",
    "docs/faq.txt",
    "docs/api-reference.md"
])

# Save index
index.save("my_knowledge_base")
\`\`\`

## Step 3: Query with Context

\`\`\`python
# Load index
index = RAGIndex.load("my_knowledge_base")

# Ask a question
question = "How do I reset my password?"
context = index.search(question, top_k=3)

# Generate answer with context
response = model.chat([
    {"role": "system", "content": f"Answer based on this context:\\n{context}"},
    {"role": "user", "content": question}
])
print(response)
\`\`\`

## Step 4: Run as API Server

\`\`\`bash
zse serve qwen-7b.zse --rag-index my_knowledge_base --port 8000
\`\`\`

Now your API automatically retrieves context for each query!

## Tips for Better RAG

1. **Chunk size matters** - Try 512-1024 tokens per chunk
2. **Use hybrid search** - Combine semantic + keyword search
3. **Add metadata** - Filter by document type/date
4. **Tune retrieval** - More context isn't always better (3-5 chunks)

Your documents stay local - nothing leaves your machine.
    `,author:"ZSE Team",authorImage:"/images/zllm-logo.png",date:"2026-02-21",readTime:"10 min read",tags:["tutorial","rag","chatbot"],featured:!1},{id:"6",title:"Deploying ZSE to Production: Docker & Kubernetes",slug:"production-deployment",excerpt:"Best practices for deploying ZSE in production environments with Docker, Kubernetes, and monitoring.",content:`
# Deploying ZSE to Production

A complete guide to deploying ZSE in production with Docker, Kubernetes, and proper monitoring.

## Docker Deployment

### Dockerfile
\`\`\`dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN pip install zllm-zse

COPY models/qwen-7b.zse /models/
EXPOSE 8000

CMD ["zse", "serve", "/models/qwen-7b.zse", "--host", "0.0.0.0"]
\`\`\`

### Docker Compose
\`\`\`yaml
version: '3.8'
services:
  zse:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - ./models:/models
    environment:
      - ZSE_MAX_BATCH_SIZE=32
      - ZSE_LOG_LEVEL=info
\`\`\`

## Kubernetes Deployment

\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zse-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: zse
  template:
    spec:
      containers:
      - name: zse
        image: your-registry/zse:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 30
\`\`\`

## Health Checks & Monitoring

### Prometheus Metrics
ZSE exposes metrics at \`/metrics\`:
- \`zse_requests_total\`
- \`zse_request_duration_seconds\`
- \`zse_tokens_generated_total\`
- \`zse_gpu_memory_bytes\`

### Alerting Rules
\`\`\`yaml
groups:
- name: zse
  rules:
  - alert: HighLatency
    expr: zse_request_duration_seconds{quantile="0.99"} > 5
    for: 5m
  - alert: GPUMemoryHigh
    expr: zse_gpu_memory_bytes / zse_gpu_memory_total > 0.95
    for: 1m
\`\`\`

## Production Checklist

- [ ] Set appropriate \`--max-batch\` and \`--max-concurrent\`
- [ ] Enable request logging: \`--log-format json\`
- [ ] Configure rate limiting: \`--rate-limit 100\`
- [ ] Set up Prometheus scraping
- [ ] Test graceful shutdown handling
- [ ] Configure horizontal pod autoscaling
    `,author:"ZSE Team",authorImage:"/images/zllm-logo.png",date:"2026-02-20",readTime:"12 min read",tags:["deployment","docker","kubernetes","production"],featured:!1},{id:"7",title:"Streaming Responses with ZSE: Real-time Token Generation",slug:"streaming-responses",excerpt:"Implement real-time streaming for chat applications with minimal time-to-first-token.",content:`
# Streaming Responses with ZSE

Enable real-time token streaming for responsive chat applications.

## Why Streaming?

Without streaming, users wait for the entire response. With streaming:
- **~50ms** time-to-first-token
- Immediate visual feedback
- Better perceived performance

## Enable Streaming

### REST API
\`\`\`bash
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen-7b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
\`\`\`

### Python (OpenAI SDK)
\`\`\`python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

stream = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
\`\`\`

### JavaScript/React
\`\`\`javascript
async function streamChat(message) {
  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen-7b',
      messages: [{ role: 'user', content: message }],
      stream: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    // Parse SSE and update UI
    console.log(text);
  }
}
\`\`\`

## Server-Sent Events Format

Each chunk is prefixed with \`data: \`:
\`\`\`
data: {"choices":[{"delta":{"content":"Once"}}]}
data: {"choices":[{"delta":{"content":" upon"}}]}
data: {"choices":[{"delta":{"content":" a"}}]}
data: [DONE]
\`\`\`

## Best Practices

1. **Handle backpressure** - Don't overwhelm slow clients
2. **Implement cancellation** - Let users stop generation
3. **Show typing indicator** - While waiting for first token
4. **Buffer intelligently** - Consider word-level chunks for smoother UX
    `,author:"ZSE Team",authorImage:"/images/zllm-logo.png",date:"2026-02-19",readTime:"7 min read",tags:["tutorial","streaming","api"],featured:!1},{id:"8",title:"Benchmarking Your ZSE Setup: Measuring Real Performance",slug:"benchmarking-guide",excerpt:"How to accurately measure cold start time, throughput, and latency for your specific hardware.",content:`
# Benchmarking Your ZSE Setup

Learn how to measure real performance metrics for your hardware.

## Built-in Benchmark Command

\`\`\`bash
zse benchmark qwen-7b.zse
\`\`\`

Output:
\`\`\`
┌────────────────────────────────────────────────┐
│ Benchmark Results: qwen-7b.zse                 │
├────────────────────────────────────────────────┤
│ Cold Start:         3.9s                       │
│ Throughput:         87.3 tok/s                 │
│ Time-to-First:      52ms                       │
│ Latency (p50):      11.4ms/tok                 │
│ Latency (p99):      18.2ms/tok                 │
│ GPU Memory:         5.2 GB                     │
└────────────────────────────────────────────────┘
\`\`\`

## Specific Benchmarks

### Cold Start Only
\`\`\`bash
zse benchmark model.zse --metric cold-start --runs 5
\`\`\`

### Throughput Test
\`\`\`bash
zse benchmark model.zse --metric throughput \\
  --prompt-length 512 \\
  --output-length 256 \\
  --batch-sizes 1,4,8,16
\`\`\`

### Memory Profiling
\`\`\`bash
zse benchmark model.zse --metric memory \\
  --context-lengths 1024,4096,8192,16384
\`\`\`

## Compare Configurations

\`\`\`bash
# Compare quantization types
zse benchmark model-nf4.zse model-int4.zse model-int8.zse

# Compare context lengths
zse benchmark model.zse --sweep max-context 1024:16384:2x
\`\`\`

## Python Benchmarking

\`\`\`python
from zllm_zse import ZSE, benchmark

model = ZSE("qwen-7b.zse")

results = benchmark(
    model,
    prompts=["Explain quantum computing" * 10 for _ in range(100)],
    max_tokens=256
)

print(f"Mean throughput: {results.throughput_mean:.1f} tok/s")
print(f"p99 latency: {results.latency_p99:.1f} ms/tok")
\`\`\`

## Hardware-Specific Expectations

| GPU | 7B Throughput | 7B Cold Start |
|-----|---------------|---------------|
| RTX 3060 12GB | ~45 tok/s | ~4.5s |
| RTX 4070 12GB | ~80 tok/s | ~4.0s |
| RTX 4090 24GB | ~120 tok/s | ~3.5s |
| A100 80GB | ~180 tok/s | ~3.2s |

Your mileage may vary based on PCIe bandwidth, CPU, and storage speed.
    `,author:"ZSE Team",authorImage:"/images/zllm-logo.png",date:"2026-02-18",readTime:"6 min read",tags:["benchmarks","performance","guide"],featured:!1}];function a(e){return s.find(t=>t.slug===e)}},1453:(e,t,n)=>{"use strict";n.d(t,{cn:()=>o,p:()=>r});var s=n(6815),a=n(9377);function o(...e){return(0,a.m6)((0,s.W)(e))}function r(e){return new Date(e).toLocaleDateString("en-US",{year:"numeric",month:"long",day:"numeric"})}},1808:(e,t,n)=>{"use strict";n.r(t),n.d(t,{$$typeof:()=>o,__esModule:()=>a,default:()=>r});let s=(0,n(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/blog/page.tsx`),{__esModule:a,$$typeof:o}=s,r=s.default}};var t=require("../../webpack-runtime.js");t.C(e);var n=e=>t(t.s=e),s=t.X(0,[638,498,697,782],()=>n(1559));module.exports=s})();