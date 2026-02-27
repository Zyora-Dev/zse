'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'system-requirements', title: 'System Requirements', level: 2 },
  { id: 'configuration', title: 'Configuration', level: 2 },
  { id: 'security', title: 'Security', level: 2 },
  { id: 'monitoring', title: 'Monitoring', level: 2 },
  { id: 'scaling', title: 'Scaling', level: 2 },
]

export default function ProductionPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Production Setup"
          description="Deploy ZSE in production with best practices for security, performance, and reliability."
          badge="Deployment"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            Running ZSE in production requires careful attention to security, monitoring, 
            and resource management. This guide covers the essential configurations and 
            best practices for a production deployment.
          </p>

          <CardGrid columns={3}>
            <Card
              title="Security"
              description="API keys, rate limiting, HTTPS"
            />
            <Card
              title="Performance"
              description="GPU optimization, caching, batching"
            />
            <Card
              title="Reliability"
              description="Health checks, logging, alerts"
            />
          </CardGrid>
        </DocSection>

        <DocSection id="system-requirements" title="System Requirements">
          <p className="mb-4">
            Recommended specifications for production deployments:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Component</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Minimum</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Recommended</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">GPU (7B models)</td>
                  <td className="py-3 px-4 text-gray-400">8GB VRAM</td>
                  <td className="py-3 px-4 text-lime">24GB+ VRAM</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">GPU (32B models)</td>
                  <td className="py-3 px-4 text-gray-400">24GB VRAM</td>
                  <td className="py-3 px-4 text-lime">80GB+ VRAM</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">System RAM</td>
                  <td className="py-3 px-4 text-gray-400">16GB</td>
                  <td className="py-3 px-4 text-lime">64GB+</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">Storage</td>
                  <td className="py-3 px-4 text-gray-400">50GB SSD</td>
                  <td className="py-3 px-4 text-lime">500GB+ NVMe</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="info">
            Use INT4 quantization to fit larger models on smaller GPUs. A 32B model 
            can run on 24GB VRAM with INT4.
          </Callout>
        </DocSection>

        <DocSection id="configuration" title="Configuration">
          <DocSubSection id="env-vars" title="Environment Variables">
            <CodeBlock
              language="bash"
              code={`# Production environment variables
export ZSE_ENV=production
export ZSE_HOST=0.0.0.0
export ZSE_PORT=8000
export ZSE_WORKERS=4
export ZSE_LOG_LEVEL=info
export ZSE_MAX_BATCH_SIZE=32
export ZSE_MAX_CONTEXT=8192`}
            />
          </DocSubSection>

          <DocSubSection id="config-file" title="Configuration File">
            <CodeBlock
              language="yaml"
              filename="config.yaml"
              code={`# ZSE Production Configuration
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

model:
  path: /models/qwen7b.zse
  max_context: 8192
  batch_size: 32

memory:
  kv_cache: 4GB
  max_memory_per_gpu: 22GB

logging:
  level: info
  format: json
  output: /var/log/zse/server.log`}
            />
          </DocSubSection>

          <DocSubSection id="start-production" title="Starting the Server">
            <CodeBlock
              language="bash"
              code={`# Start with config file
zse serve model.zse --config config.yaml

# Or with environment variables
ZSE_ENV=production zse serve model.zse \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --workers 4`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="security" title="Security">
          <DocSubSection id="api-keys" title="API Key Authentication">
            <p className="mb-4">
              Always require API keys in production:
            </p>

            <CodeBlock
              language="bash"
              code={`# Create an API key with rate limiting
zse api-key create production-app --rate-limit 100

# Start server with authentication
zse serve model.zse --require-auth

# Clients must include the key
curl -H "Authorization: Bearer sk-xxx" \\
  http://localhost:8000/v1/chat/completions`}
            />
          </DocSubSection>

          <DocSubSection id="rate-limiting" title="Rate Limiting">
            <CodeBlock
              language="bash"
              code={`# Per-key rate limits
zse api-key create app1 --rate-limit 60   # 60 req/min
zse api-key create app2 --rate-limit 1000 # 1000 req/min

# Check rate limit status
zse api-key status app1`}
            />

            <Callout type="warning">
              Rate limits help prevent abuse and ensure fair resource allocation. 
              Clients receive 429 responses when limits are exceeded.
            </Callout>
          </DocSubSection>

          <DocSubSection id="https" title="HTTPS with Reverse Proxy">
            <p className="mb-4">
              Use a reverse proxy like Nginx for HTTPS termination:
            </p>

            <CodeBlock
              language="nginx"
              filename="nginx.conf"
              code={`upstream zse {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass http://zse;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="monitoring" title="Monitoring">
          <DocSubSection id="health-checks" title="Health Checks">
            <CodeBlock
              language="bash"
              code={`# Liveness probe
curl http://localhost:8000/health

# Readiness probe (model loaded)
curl http://localhost:8000/ready

# Detailed status
curl http://localhost:8000/v1/status`}
            />
          </DocSubSection>

          <DocSubSection id="metrics" title="Prometheus Metrics">
            <CodeBlock
              language="bash"
              code={`# Enable Prometheus metrics
zse serve model.zse --metrics

# Metrics available at
curl http://localhost:8000/metrics

# Example metrics:
# zse_requests_total
# zse_tokens_generated_total
# zse_inference_latency_seconds
# zse_gpu_memory_used_bytes`}
            />
          </DocSubSection>

          <DocSubSection id="logging" title="Structured Logging">
            <CodeBlock
              language="bash"
              code={`# JSON logging for production
zse serve model.zse --log-format json

# Log output example:
# {"timestamp":"2026-02-26T10:00:00Z","level":"info","msg":"Request completed","request_id":"abc123","tokens":150,"latency_ms":1234}`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="scaling" title="Scaling">
          <DocSubSection id="horizontal" title="Horizontal Scaling">
            <p className="mb-4">
              Run multiple ZSE instances behind a load balancer:
            </p>

            <CodeBlock
              language="bash"
              code={`# Instance 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 zse serve model.zse --port 8001

# Instance 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 zse serve model.zse --port 8002

# Load balance with Nginx
upstream zse {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}`}
            />
          </DocSubSection>

          <DocSubSection id="multi-model" title="Multi-Model Deployment">
            <CodeBlock
              language="bash"
              code={`# Serve different models on different ports
zse serve qwen7b.zse --port 8001   # Fast, general purpose
zse serve qwen32b.zse --port 8002  # High quality, slower

# Route by endpoint or header`}
            />
          </DocSubSection>

          <Callout type="tip">
            For high-availability deployments, use Kubernetes with multiple replicas. 
            See the <a href="/docs/deployment/kubernetes" className="text-lime hover:underline">Kubernetes guide</a>.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ href: '/docs/api/config', title: 'Configuration' }}
          next={{ href: '/docs/deployment/docker', title: 'Docker' }}
        />
      </article>
      
      <TableOfContents items={tocItems} />
    </div>
  )
}
