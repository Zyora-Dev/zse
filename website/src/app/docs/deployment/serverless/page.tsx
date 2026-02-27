'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'modal', title: 'Modal', level: 2 },
  { id: 'runpod', title: 'RunPod', level: 2 },
  { id: 'replicate', title: 'Replicate', level: 2 },
  { id: 'aws-lambda', title: 'AWS Lambda', level: 2 },
  { id: 'cold-start', title: 'Cold Start Optimization', level: 2 },
]

export default function ServerlessPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Serverless Deployment"
          description="Deploy ZSE on serverless GPU platforms for pay-per-use inference with automatic scaling."
          badge="Deployment"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            Serverless GPU platforms let you run ZSE without managing infrastructure. 
            Pay only for compute time used, with automatic scaling from zero to thousands 
            of concurrent requests.
          </p>

          <CardGrid columns={2}>
            <Card
              title="Pay-per-use"
              description="No idle GPU costs, pay only for inference time"
            />
            <Card
              title="Auto-scaling"
              description="Scale to zero or thousands automatically"
            />
          </CardGrid>

          <Callout type="info">
            ZSE&apos;s fast cold start (3.9s for 7B models) makes it ideal for serverless 
            where containers may be created on-demand.
          </Callout>
        </DocSection>

        <DocSection id="modal" title="Modal">
          <p className="mb-4">
            Modal is recommended for ZSE serverless deployments due to its excellent 
            GPU support and fast container start times.
          </p>

          <DocSubSection id="modal-setup" title="Setup">
            <CodeBlock
              language="bash"
              code={`# Install Modal CLI
pip install modal

# Authenticate
modal token new`}
            />
          </DocSubSection>

          <DocSubSection id="modal-app" title="Modal Application">
            <CodeBlock
              language="python"
              filename="app.py"
              code={`import modal

app = modal.App("zse-inference")

# Define the image with ZSE
image = modal.Image.debian_slim().pip_install("zllm-zse")

# Create a volume for models
volume = modal.Volume.from_name("zse-models", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A10G",  # or "A100", "T4", "H100"
    volumes={"/models": volume},
    container_idle_timeout=300,  # Keep warm for 5 mins
)
class ZSEServer:
    @modal.enter()
    def load_model(self):
        from zse.engine.orchestrator import IntelligenceOrchestrator
        self.orch = IntelligenceOrchestrator("/models/qwen7b.zse")
        self.orch.load()

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 256):
        return self.orch.generate(prompt, max_tokens=max_tokens)

    @modal.method()
    def chat(self, messages: list):
        return self.orch.chat(messages)

# Web endpoint
@app.function(image=image, gpu="A10G", volumes={"/models": volume})
@modal.web_endpoint(method="POST")
def chat_endpoint(request: dict):
    from zse.engine.orchestrator import IntelligenceOrchestrator
    orch = IntelligenceOrchestrator("/models/qwen7b.zse")
    orch.load()
    return {"response": orch.chat(request["messages"])}`}
            />
          </DocSubSection>

          <DocSubSection id="modal-deploy" title="Deploy">
            <CodeBlock
              language="bash"
              code={`# Upload model to volume first
modal volume put zse-models ./qwen7b.zse /qwen7b.zse

# Deploy the app
modal deploy app.py

# Test the endpoint
curl -X POST https://your-app--chat-endpoint.modal.run \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="runpod" title="RunPod">
          <p className="mb-4">
            RunPod offers serverless GPU endpoints with competitive pricing.
          </p>

          <DocSubSection id="runpod-handler" title="Handler">
            <CodeBlock
              language="python"
              filename="handler.py"
              code={`import runpod
from zse.engine.orchestrator import IntelligenceOrchestrator

# Load model globally (persists across requests)
orch = None

def load_model():
    global orch
    if orch is None:
        orch = IntelligenceOrchestrator("/models/qwen7b.zse")
        orch.load()
    return orch

def handler(job):
    """Handle inference requests."""
    job_input = job["input"]
    
    orch = load_model()
    
    if "messages" in job_input:
        # Chat completion
        response = orch.chat(job_input["messages"])
    else:
        # Text generation
        response = orch.generate(
            job_input.get("prompt", ""),
            max_tokens=job_input.get("max_tokens", 256)
        )
    
    return {"response": response}

runpod.serverless.start({"handler": handler})`}
            />
          </DocSubSection>

          <DocSubSection id="runpod-dockerfile" title="Dockerfile">
            <CodeBlock
              language="dockerfile"
              filename="Dockerfile"
              code={`FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Install ZSE
RUN pip install zllm-zse

# Copy model
COPY ./qwen7b.zse /models/

# Copy handler
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="replicate" title="Replicate">
          <p className="mb-4">
            Deploy ZSE models on Replicate with Cog for easy API creation.
          </p>

          <DocSubSection id="replicate-cog" title="Cog Configuration">
            <CodeBlock
              language="yaml"
              filename="cog.yaml"
              code={`build:
  gpu: true
  python_version: "3.10"
  python_packages:
    - "zllm-zse"

predict: "predict.py:Predictor"`}
            />
          </DocSubSection>

          <DocSubSection id="replicate-predict" title="Predictor">
            <CodeBlock
              language="python"
              filename="predict.py"
              code={`from cog import BasePredictor, Input
from zse.engine.orchestrator import IntelligenceOrchestrator

class Predictor(BasePredictor):
    def setup(self):
        """Load the model."""
        self.orch = IntelligenceOrchestrator("./qwen7b.zse")
        self.orch.load()

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        max_tokens: int = Input(description="Max tokens", default=256),
        temperature: float = Input(description="Temperature", default=0.7),
    ) -> str:
        """Run inference."""
        return self.orch.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )`}
            />
          </DocSubSection>

          <DocSubSection id="replicate-deploy" title="Deploy">
            <CodeBlock
              language="bash"
              code={`# Login to Replicate
cog login

# Push to Replicate
cog push r8.im/username/zse-qwen7b`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="aws-lambda" title="AWS Lambda">
          <p className="mb-4">
            For AWS deployments, use Lambda with container images and Provisioned Concurrency.
          </p>

          <Callout type="warning">
            AWS Lambda has a 10GB container size limit and 15-minute timeout. 
            For larger models, use SageMaker or EC2 with auto-scaling instead.
          </Callout>

          <DocSubSection id="lambda-container" title="Container Setup">
            <CodeBlock
              language="dockerfile"
              filename="Dockerfile"
              code={`FROM public.ecr.aws/lambda/python:3.10

# Install ZSE (CPU-only for Lambda)
RUN pip install zllm-zse

# Copy model (must be < 10GB total)
COPY ./tinyllama.zse /models/

# Copy handler
COPY app.py \${LAMBDA_TASK_ROOT}

CMD ["app.handler"]`}
            />
          </DocSubSection>

          <DocSubSection id="lambda-handler" title="Handler">
            <CodeBlock
              language="python"
              filename="app.py"
              code={`from zse.engine.orchestrator import IntelligenceOrchestrator

# Load model outside handler for warm starts
orch = IntelligenceOrchestrator("/models/tinyllama.zse")
orch.load()

def handler(event, context):
    prompt = event.get("prompt", "")
    max_tokens = event.get("max_tokens", 128)
    
    response = orch.generate(prompt, max_tokens=max_tokens)
    
    return {
        "statusCode": 200,
        "body": {"response": response}
    }`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="cold-start" title="Cold Start Optimization">
          <p className="mb-4">
            Cold starts are critical for serverless. ZSE&apos;s .zse format provides 
            significantly faster cold starts than alternatives.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Format</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">7B Cold Start</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Serverless Cost Impact</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">.zse format</td>
                  <td className="py-3 px-4 text-right text-lime font-mono">3.9s</td>
                  <td className="py-3 px-4 text-right text-lime">Minimal startup overhead</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">bitsandbytes</td>
                  <td className="py-3 px-4 text-right text-gray-400">45.4s</td>
                  <td className="py-3 px-4 text-right text-gray-400">~40s billed startup</td>
                </tr>
              </tbody>
            </table>
          </div>

          <FeatureList features={[
            "Always use pre-converted .zse models",
            "Use container idle timeout to keep warm",
            "Consider provisioned concurrency for consistent latency",
            "Store models in fast storage (NVMe volumes)",
          ]} />

          <CodeBlock
            language="bash"
            code={`# Convert model before deployment (one-time)
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen7b.zse

# Include .zse file in container (NOT HuggingFace weights)`}
          />
        </DocSection>

        <DocNav
          prev={{ href: '/docs/deployment/kubernetes', title: 'Kubernetes' }}
          next={{ href: '/docs/advanced/custom-models', title: 'Custom Models' }}
        />
      </article>
      
      <TableOfContents items={tocItems} />
    </div>
  )
}
