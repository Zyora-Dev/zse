'use client'

import { DocHeader, DocSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { Card, CardGrid, FeatureList } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'zse-format', title: '.zse Format', level: 2 },
  { id: 'gguf-format', title: 'GGUF Format', level: 2 },
  { id: 'safetensors', title: 'Safetensors', level: 2 },
  { id: 'comparison', title: 'Comparison', level: 2 },
]

export default function ModelFormatsPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Model Formats"
          description="Understanding different model formats supported by ZSE: .zse, GGUF, and safetensors."
          badge="Core Concepts"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            ZSE supports multiple model formats, each with different tradeoffs for loading speed, 
            memory usage, and compatibility.
          </p>

          <CardGrid columns={3}>
            <Card
              title=".zse"
              description="Native format — fastest loading, best optimization"
            />
            <Card
              title="GGUF"
              description="Ollama/llama.cpp compatible — good portability"
            />
            <Card
              title="Safetensors"
              description="HuggingFace format — universal compatibility"
            />
          </CardGrid>
        </DocSection>

        <DocSection id="zse-format" title=".zse Format">
          <p className="mb-4">
            The native ZSE format offers the fastest loading times and best memory efficiency.
          </p>

          <FeatureList features={[
            "Pre-quantized weights — no runtime quantization",
            "Memory-mapped loading — instant access",
            "Optimized tensor layout — sequential reads",
            "Built-in tokenizer and config",
          ]} />

          <p className="mt-4 mb-2">
            <strong className="text-white">Structure:</strong>
          </p>

          <CodeBlock
            language="text"
            code={`model.zse
├── header.json          # Model metadata
├── config.json          # Model configuration
├── tokenizer/           # Tokenizer files
│   ├── vocab.json
│   └── merges.txt
└── tensors/             # Quantized weights
    ├── embed.bin
    ├── layer_0.bin
    ├── layer_1.bin
    └── ...`}
          />

          <p className="mt-4 mb-2">
            <strong className="text-white">Creating .zse files:</strong>
          </p>

          <CodeBlock
            language="bash"
            code={`# From HuggingFace model
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse

# From local safetensors
zse convert ./my-model/ -o my-model.zse

# With specific quantization
zse convert model -o model.zse --quant nf4`}
          />
        </DocSection>

        <DocSection id="gguf-format" title="GGUF Format">
          <p className="mb-4">
            GGUF (GPT-Generated Unified Format) is used by llama.cpp and Ollama. ZSE can import 
            GGUF files directly.
          </p>

          <Callout type="info">
            Install GGUF support: <InlineCode>pip install zllm-zse[gguf]</InlineCode>
          </Callout>

          <p className="mt-4 mb-2">
            <strong className="text-white">Loading GGUF models:</strong>
          </p>

          <CodeBlock
            language="bash"
            code={`# Serve GGUF directly
zse serve ./qwen-7b-q4_k_m.gguf

# Convert GGUF to .zse for faster loading
zse convert ./model.gguf -o model.zse`}
          />

          <p className="mt-4 mb-4">
            Supported GGUF quantization types:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Type</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Bits</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>Q4_0</InlineCode></td>
                  <td className="py-2 px-4">4</td>
                  <td className="py-2 pl-4">Basic 4-bit quantization</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>Q4_K_M</InlineCode></td>
                  <td className="py-2 px-4">4</td>
                  <td className="py-2 pl-4">K-quants medium (recommended)</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>Q5_K_M</InlineCode></td>
                  <td className="py-2 px-4">5</td>
                  <td className="py-2 pl-4">Higher quality</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>Q8_0</InlineCode></td>
                  <td className="py-2 px-4">8</td>
                  <td className="py-2 pl-4">Best quality</td>
                </tr>
              </tbody>
            </table>
          </div>
        </DocSection>

        <DocSection id="safetensors" title="Safetensors">
          <p className="mb-4">
            Safetensors is the standard format for HuggingFace models. ZSE can load safetensors 
            directly with runtime quantization.
          </p>

          <CodeBlock
            language="bash"
            code={`# Load from HuggingFace Hub
zse serve Qwen/Qwen2.5-7B-Instruct

# Load local safetensors
zse serve ./my-local-model/`}
          />

          <Callout type="warning">
            Loading safetensors requires runtime quantization, which adds 30-60 seconds to 
            cold start time. For production, convert to <InlineCode>.zse</InlineCode> format.
          </Callout>
        </DocSection>

        <DocSection id="comparison" title="Comparison">
          <p className="mb-4">
            Choose the right format for your use case:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Feature</th>
                  <th className="text-center py-2 px-4 text-lime font-medium">.zse</th>
                  <th className="text-center py-2 px-4 text-white/50 font-medium">GGUF</th>
                  <th className="text-center py-2 pl-4 text-white/50 font-medium">Safetensors</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">Cold Start (7B)</td>
                  <td className="text-center py-2 px-4 text-lime">3.9s</td>
                  <td className="text-center py-2 px-4">~15s</td>
                  <td className="text-center py-2 pl-4">~45s</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">Pre-quantized</td>
                  <td className="text-center py-2 px-4 text-lime">✓</td>
                  <td className="text-center py-2 px-4">✓</td>
                  <td className="text-center py-2 pl-4">✗</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">Memory-mapped</td>
                  <td className="text-center py-2 px-4 text-lime">✓</td>
                  <td className="text-center py-2 px-4">✓</td>
                  <td className="text-center py-2 pl-4">✓</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">Portability</td>
                  <td className="text-center py-2 px-4">ZSE only</td>
                  <td className="text-center py-2 px-4">Ollama, llama.cpp</td>
                  <td className="text-center py-2 pl-4 text-lime">Universal</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">Best For</td>
                  <td className="text-center py-2 px-4 text-lime">Production</td>
                  <td className="text-center py-2 px-4">Cross-platform</td>
                  <td className="text-center py-2 pl-4">Development</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="tip" title="Recommendation">
            Use <InlineCode>.zse</InlineCode> for production deployments where cold start time matters. 
            Use GGUF if you need compatibility with Ollama. Use safetensors for quick experimentation.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ title: 'Architecture', href: '/docs/architecture' }}
          next={{ title: 'Quantization', href: '/docs/quantization' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
