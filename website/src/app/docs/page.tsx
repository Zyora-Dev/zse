'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { DocHeader, DocSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { Card, CardGrid, FeatureList } from '@/components/docs/steps'
import { Zap, Server, Code, Layers, Cpu, Terminal } from 'lucide-react'

const tocItems = [
  { id: 'what-is-zse', title: 'What is ZSE?', level: 2 },
  { id: 'key-features', title: 'Key Features', level: 2 },
  { id: 'benchmarks', title: 'Benchmarks', level: 2 },
  { id: 'quick-install', title: 'Quick Install', level: 2 },
  { id: 'next-steps', title: 'Next Steps', level: 2 },
]

export default function DocsPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Introduction"
          description="ZSE (Z Server Engine) is an ultra memory-efficient LLM inference engine designed for fast cold starts and low memory usage."
          badge="Getting Started"
        />

        <DocSection id="what-is-zse" title="What is ZSE?">
          <p className="mb-4">
            ZSE is an inference engine that loads large language models in seconds, not minutes. 
            It achieves this through pre-quantized model formats that skip runtime quantization entirely.
          </p>
          
          <Callout type="tip" title="Why ZSE?">
            Traditional inference engines like vLLM and transformers spend 30-60 seconds quantizing 
            models at startup. ZSE's <InlineCode>.zse</InlineCode> format eliminates this overhead, 
            enabling sub-4-second cold starts for 7B models.
          </Callout>

          <p className="mt-4">
            Whether you're building serverless AI endpoints, developing locally, or deploying to production, 
            ZSE helps you iterate faster and reduce costs.
          </p>
        </DocSection>

        <DocSection id="key-features" title="Key Features">
          <CardGrid columns={2}>
            <Card
              icon={Zap}
              title="zQuantize"
              description="Pre-quantize models to INT4/NF4 format for instant loading"
            />
            <Card
              icon={Server}
              title="zServe"
              description="OpenAI-compatible API server with streaming support"
            />
            <Card
              icon={Terminal}
              title="zInfer"
              description="CLI tool for quick model testing and inference"
            />
            <Card
              icon={Layers}
              title="zStream"
              description="Layer streaming for running large models on limited VRAM"
            />
            <Card
              icon={Cpu}
              title="zKV"
              description="Quantized KV cache for 4× memory savings"
            />
            <Card
              icon={Code}
              title="OpenAI API"
              description="Drop-in replacement for OpenAI's chat completions API"
            />
          </CardGrid>

          <FeatureList features={[
            "9.1s cold start for 7B models with embedded config/tokenizer",
            "24.1s cold start for 32B models",
            "58.7 tok/s (7B) and 26.9 tok/s (32B) with bnb.matmul_4bit",
            "32B fits on 24GB consumer GPUs (RTX 3090/4090)",
            "Single .zse file format - no network calls needed",
            "Streaming token generation",
          ]} />
        </DocSection>

        <DocSection id="benchmarks" title="Benchmarks">
          <p className="mb-4">
            Performance benchmarks on H200 with Qwen 2.5 Instruct models (v1.2.0):
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Model</th>
                  <th className="text-center py-2 px-4 text-white/50 font-medium">File Size</th>
                  <th className="text-center py-2 px-4 text-white/50 font-medium">VRAM</th>
                  <th className="text-center py-2 px-4 text-lime font-medium">Speed</th>
                  <th className="text-center py-2 pl-4 text-white/50 font-medium">Load Time</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">7B</td>
                  <td className="text-center py-2 px-4">5.57 GB</td>
                  <td className="text-center py-2 px-4">5.9 GB</td>
                  <td className="text-center py-2 px-4 text-lime">58.7 tok/s</td>
                  <td className="text-center py-2 pl-4">9.1s</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">32B</td>
                  <td className="text-center py-2 px-4">19.23 GB</td>
                  <td className="text-center py-2 px-4">20.9 GB</td>
                  <td className="text-center py-2 px-4 text-lime">26.9 tok/s</td>
                  <td className="text-center py-2 pl-4">24.1s</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="info">
            32B models now fit on 24GB consumer GPUs (RTX 3090/4090) thanks to bnb.matmul_4bit integration.
          </Callout>
        </DocSection>

        <DocSection id="quick-install" title="Quick Install">
          <p className="mb-4">
            Install ZSE from PyPI:
          </p>

          <CodeBlock
            language="bash"
            code="pip install zllm-zse"
          />

          <p className="mt-4 mb-4">
            For GGUF model support, install with the optional dependency:
          </p>

          <CodeBlock
            language="bash"
            code="pip install zllm-zse[gguf]"
          />

          <p className="mt-4 mb-4">
            Start a server with a pre-trained model:
          </p>

          <CodeBlock
            language="bash"
            code={`# Start the server
zse serve Qwen/Qwen2.5-7B-Instruct

# Or with custom settings
zse serve Qwen/Qwen2.5-7B-Instruct --port 8080 --host 0.0.0.0`}
          />
        </DocSection>

        <DocSection id="next-steps" title="Next Steps">
          <CardGrid columns={3}>
            <Card
              title="Quick Start"
              description="Run your first model in 5 minutes"
              href="/docs/quickstart"
            />
            <Card
              title="Installation"
              description="Detailed setup and requirements"
              href="/docs/installation"
            />
            <Card
              title="API Reference"
              description="Full CLI and Python API docs"
              href="/docs/api/cli"
            />
          </CardGrid>
        </DocSection>

        <DocNav
          next={{ title: 'Quick Start', href: '/docs/quickstart' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
