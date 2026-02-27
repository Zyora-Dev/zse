'use client'

import { DocHeader, DocSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { Steps, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'prerequisites', title: 'Prerequisites', level: 2 },
  { id: 'installation', title: 'Installation', level: 2 },
  { id: 'start-server', title: 'Start the Server', level: 2 },
  { id: 'api-calls', title: 'Make API Calls', level: 2 },
  { id: 'convert-format', title: 'Convert to .zse', level: 2 },
  { id: 'next-steps', title: 'Next Steps', level: 2 },
]

export default function QuickStartPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Quick Start"
          description="Get up and running with ZSE in under 5 minutes. This guide covers installation, running your first model, and converting to the fast .zse format."
          badge="Getting Started"
        />

        <DocSection id="prerequisites" title="Prerequisites">
          <p className="mb-4">
            Before installing ZSE, ensure you have:
          </p>
          <ul className="space-y-2 text-sm">
            <li className="flex items-start gap-2">
              <span className="text-lime">•</span>
              <span><strong className="text-white">Python 3.8+</strong> — We recommend Python 3.10 or newer</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-lime">•</span>
              <span><strong className="text-white">CUDA 11.8+</strong> — For GPU acceleration (optional but recommended)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-lime">•</span>
              <span><strong className="text-white">8GB+ VRAM</strong> — For 7B models; 24GB+ for 32B models</span>
            </li>
          </ul>

          <Callout type="tip" title="CPU-Only Mode">
            ZSE can run on CPU-only machines, but performance will be significantly slower. 
            GPU acceleration is recommended for production use.
          </Callout>
        </DocSection>

        <DocSection id="installation" title="Installation">
          <p className="mb-4">
            Install ZSE from PyPI with pip:
          </p>

          <CodeBlock
            language="bash"
            code="pip install zllm-zse"
          />

          <p className="mt-4 mb-4">
            For GGUF model support (Ollama/llama.cpp models):
          </p>

          <CodeBlock
            language="bash"
            code="pip install zllm-zse[gguf]"
          />

          <Callout type="info">
            ZSE will automatically detect and use your GPU if available. No additional 
            configuration is required for CUDA.
          </Callout>

          <p className="mt-4 mb-2">
            Verify the installation:
          </p>

          <CodeBlock
            language="bash"
            code={`zse --version
# Output: zse 1.2.0

zse hardware
# Shows detected GPUs and available memory`}
          />
        </DocSection>

        <DocSection id="start-server" title="Start the Server">
          <p className="mb-4">
            Serve any HuggingFace model with a single command:
          </p>

          <CodeBlock
            language="bash"
            code="zse serve Qwen/Qwen2.5-7B-Instruct"
          />

          <p className="mt-4 mb-4">
            The server will:
          </p>
          <ul className="space-y-1 text-sm mb-4">
            <li className="flex items-start gap-2">
              <span className="text-white/40">1.</span>
              <span>Download the model from HuggingFace (cached for future use)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-white/40">2.</span>
              <span>Quantize to INT4 format (first run only)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-white/40">3.</span>
              <span>Start an OpenAI-compatible API at <InlineCode>http://localhost:8000</InlineCode></span>
            </li>
          </ul>

          <Callout type="success">
            Once started, you'll see: <InlineCode>Server running at http://localhost:8000</InlineCode>
          </Callout>

          <p className="mt-4 mb-2">
            Common server options:
          </p>

          <CodeBlock
            language="bash"
            code={`# Custom port and host
zse serve Qwen/Qwen2.5-7B-Instruct --port 8080 --host 0.0.0.0

# With API key authentication
zse serve Qwen/Qwen2.5-7B-Instruct --api-key your-secret-key

# Specify quantization type
zse serve Qwen/Qwen2.5-7B-Instruct --quant nf4`}
          />
        </DocSection>

        <DocSection id="api-calls" title="Make API Calls">
          <p className="mb-4">
            ZSE provides an OpenAI-compatible API. Use any OpenAI SDK or make direct HTTP requests:
          </p>

          <CodeBlock
            language="bash"
            filename="curl"
            code={`curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Write hello world in Python"}
    ]
  }'`}
          />

          <p className="mt-6 mb-4">
            Using the Python OpenAI SDK:
          </p>

          <CodeBlock
            language="python"
            filename="client.py"
            showLineNumbers
            code={`from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Unless you set --api-key
)

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "Write hello world in Python"}
    ],
    stream=True  # Enable streaming
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")`}
          />
        </DocSection>

        <DocSection id="convert-format" title="Convert to .zse Format">
          <p className="mb-4">
            For maximum performance, pre-convert models to the <InlineCode>.zse</InlineCode> format. 
            This eliminates runtime quantization and achieves 11× faster cold starts.
          </p>

          <Steps steps={[
            {
              title: 'Convert the model',
              description: 'One-time conversion, takes ~20 seconds',
              content: (
                <CodeBlock
                  language="bash"
                  code="zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse"
                />
              ),
            },
            {
              title: 'Serve the converted model',
              description: 'Now cold starts take only 3.9 seconds!',
              content: (
                <CodeBlock
                  language="bash"
                  code="zse serve qwen-7b.zse"
                />
              ),
            },
          ]} />

          <Callout type="warning">
            The <InlineCode>.zse</InlineCode> file is ~4GB for 7B models. Ensure you have sufficient disk space.
          </Callout>
        </DocSection>

        <DocSection id="next-steps" title="Next Steps">
          <CardGrid columns={2}>
            <Card
              title="Installation Details"
              description="System requirements, GPU setup, and troubleshooting"
              href="/docs/installation"
            />
            <Card
              title="Model Formats"
              description="Learn about .zse, GGUF, and safetensors formats"
              href="/docs/model-formats"
            />
            <Card
              title="API Reference"
              description="Full documentation of all CLI commands"
              href="/docs/api/cli"
            />
            <Card
              title="REST API"
              description="OpenAI-compatible endpoint documentation"
              href="/docs/api/rest"
            />
          </CardGrid>
        </DocSection>

        <DocNav
          prev={{ title: 'Introduction', href: '/docs' }}
          next={{ title: 'Installation', href: '/docs/installation' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
