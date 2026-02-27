'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'quick-start', title: 'Quick Start', level: 2 },
  { id: 'dockerfile', title: 'Dockerfile', level: 2 },
  { id: 'docker-compose', title: 'Docker Compose', level: 2 },
  { id: 'gpu-support', title: 'GPU Support', level: 2 },
  { id: 'volumes', title: 'Volumes & Persistence', level: 2 },
]

export default function DockerPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Docker Deployment"
          description="Run ZSE in Docker containers with GPU acceleration for consistent, reproducible deployments."
          badge="Deployment"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            Docker provides isolation and reproducibility for ZSE deployments. 
            Our official images include all dependencies and support NVIDIA GPU acceleration.
          </p>

          <CardGrid columns={3}>
            <Card
              title="Official Images"
              description="Pre-built images on Docker Hub"
            />
            <Card
              title="GPU Ready"
              description="NVIDIA Container Toolkit support"
            />
            <Card
              title="Compose"
              description="Multi-container orchestration"
            />
          </CardGrid>
        </DocSection>

        <DocSection id="quick-start" title="Quick Start">
          <p className="mb-4">
            Run ZSE with a single command using our official Docker image:
          </p>

          <CodeBlock
            language="bash"
            code={`# Pull the latest image
docker pull zyora/zse:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 \\
  zyora/zse:latest \\
  zse serve Qwen/Qwen2.5-7B-Instruct`}
          />

          <Callout type="info">
            The first run will download the model from HuggingFace. Use a volume 
            to persist models between container restarts.
          </Callout>
        </DocSection>

        <DocSection id="dockerfile" title="Dockerfile">
          <DocSubSection id="official-image" title="Using Official Image">
            <CodeBlock
              language="dockerfile"
              filename="Dockerfile"
              code={`FROM zyora/zse:latest

# Copy your pre-converted model
COPY ./models/qwen7b.zse /models/

# Set environment variables
ENV ZSE_HOST=0.0.0.0
ENV ZSE_PORT=8000

# Expose the port
EXPOSE 8000

# Start the server
CMD ["zse", "serve", "/models/qwen7b.zse"]`}
            />
          </DocSubSection>

          <DocSubSection id="custom-image" title="Building Custom Image">
            <CodeBlock
              language="dockerfile"
              filename="Dockerfile"
              code={`FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install ZSE
RUN pip install zllm-zse

# Create model directory
RUN mkdir -p /models

# Set working directory
WORKDIR /app

# Environment
ENV ZSE_HOST=0.0.0.0
ENV ZSE_PORT=8000

EXPOSE 8000

CMD ["zse", "serve", "/models/model.zse"]`}
            />
          </DocSubSection>

          <DocSubSection id="build" title="Building the Image">
            <CodeBlock
              language="bash"
              code={`# Build the image
docker build -t my-zse-app .

# Run with GPU
docker run --gpus all -p 8000:8000 my-zse-app`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="docker-compose" title="Docker Compose">
          <DocSubSection id="basic-compose" title="Basic Setup">
            <CodeBlock
              language="yaml"
              filename="docker-compose.yml"
              code={`version: '3.8'

services:
  zse:
    image: zyora/zse:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    environment:
      - ZSE_HOST=0.0.0.0
      - ZSE_PORT=8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: zse serve /models/qwen7b.zse`}
            />
          </DocSubSection>

          <DocSubSection id="full-stack" title="Full Stack with Monitoring">
            <CodeBlock
              language="yaml"
              filename="docker-compose.yml"
              code={`version: '3.8'

services:
  zse:
    image: zyora/zse:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
      - ./logs:/var/log/zse
    environment:
      - ZSE_HOST=0.0.0.0
      - ZSE_PORT=8000
      - ZSE_LOG_FORMAT=json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: zse serve /models/qwen7b.zse --metrics
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/ssl/certs
    depends_on:
      - zse

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - zse`}
            />
          </DocSubSection>

          <DocSubSection id="compose-commands" title="Commands">
            <CodeBlock
              language="bash"
              code={`# Start services
docker-compose up -d

# View logs
docker-compose logs -f zse

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="gpu-support" title="GPU Support">
          <DocSubSection id="nvidia-toolkit" title="NVIDIA Container Toolkit">
            <p className="mb-4">
              Install the NVIDIA Container Toolkit to enable GPU access in Docker:
            </p>

            <CodeBlock
              language="bash"
              code={`# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker`}
            />
          </DocSubSection>

          <DocSubSection id="gpu-flags" title="GPU Flags">
            <CodeBlock
              language="bash"
              code={`# Use all GPUs
docker run --gpus all ...

# Use specific GPUs
docker run --gpus '"device=0,1"' ...

# Use N GPUs
docker run --gpus 2 ...`}
            />
          </DocSubSection>

          <DocSubSection id="verify-gpu" title="Verify GPU Access">
            <CodeBlock
              language="bash"
              code={`docker run --gpus all zyora/zse:latest zse hardware
# Should show detected GPUs`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="volumes" title="Volumes & Persistence">
          <DocSubSection id="model-volume" title="Model Storage">
            <CodeBlock
              language="bash"
              code={`# Create a named volume for models
docker volume create zse-models

# Run with volume
docker run --gpus all -p 8000:8000 \\
  -v zse-models:/models \\
  zyora/zse:latest \\
  zse serve /models/qwen7b.zse`}
            />
          </DocSubSection>

          <DocSubSection id="download-models" title="Downloading Models">
            <CodeBlock
              language="bash"
              code={`# Download and convert a model into the volume
docker run --gpus all \\
  -v zse-models:/models \\
  zyora/zse:latest \\
  zse convert Qwen/Qwen2.5-7B-Instruct -o /models/qwen7b.zse

# Now serve it
docker run --gpus all -p 8000:8000 \\
  -v zse-models:/models \\
  zyora/zse:latest \\
  zse serve /models/qwen7b.zse`}
            />
          </DocSubSection>

          <Callout type="tip">
            Pre-convert models to .zse format and include them in your image for 
            faster cold starts in production.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ href: '/docs/deployment/production', title: 'Production Setup' }}
          next={{ href: '/docs/deployment/kubernetes', title: 'Kubernetes' }}
        />
      </article>
      
      <TableOfContents items={tocItems} />
    </div>
  )
}
