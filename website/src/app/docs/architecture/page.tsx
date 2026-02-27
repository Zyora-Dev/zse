'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'components', title: 'Core Components', level: 2 },
  { id: 'inference-flow', title: 'Inference Flow', level: 2 },
  { id: 'memory-architecture', title: 'Memory Architecture', level: 2 },
]

export default function ArchitecturePage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Architecture"
          description="Deep dive into ZSE's internal architecture, component design, and how the pieces fit together."
          badge="Core Concepts"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            ZSE is designed with one goal: minimize time-to-first-token while maintaining high throughput. 
            The architecture consists of five core components that work together to achieve this.
          </p>

          <div className="my-6 p-6 rounded-lg border border-white/[0.06] bg-white/[0.02]">
            <div className="text-center text-sm font-mono">
              <div className="flex items-center justify-center gap-2 mb-4">
                <div className="px-3 py-2 rounded bg-lime/20 text-lime border border-lime/30">zQuantize</div>
                <span className="text-white/30">→</span>
                <div className="px-3 py-2 rounded bg-white/10 text-white/70 border border-white/10">.zse File</div>
              </div>
              <div className="flex items-center justify-center gap-2 mb-4">
                <div className="px-3 py-2 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">zServe</div>
                <span className="text-white/30">↔</span>
                <div className="px-3 py-2 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">zInfer</div>
              </div>
              <div className="flex items-center justify-center gap-2">
                <div className="px-3 py-2 rounded bg-orange-500/20 text-orange-400 border border-orange-500/30">zStream</div>
                <span className="text-white/30">+</span>
                <div className="px-3 py-2 rounded bg-pink-500/20 text-pink-400 border border-pink-500/30">zKV</div>
              </div>
            </div>
          </div>
        </DocSection>

        <DocSection id="components" title="Core Components">
          <DocSubSection id="zquantize-component" title="zQuantize">
            <p className="mb-4">
              The quantization engine that converts models to the optimized <InlineCode>.zse</InlineCode> format.
            </p>
            <FeatureList features={[
              "INT4, INT8, NF4 quantization schemes",
              "Per-tensor scaling for minimal quality loss",
              "Optimized memory layout for sequential loading",
              "Preserves tokenizer and model config",
            ]} />
          </DocSubSection>

          <DocSubSection id="zserve-component" title="zServe">
            <p className="mb-4">
              OpenAI-compatible HTTP server for production deployments.
            </p>
            <FeatureList features={[
              "Full OpenAI API compatibility",
              "Server-Sent Events streaming",
              "Batch processing support",
              "API key authentication",
            ]} />
          </DocSubSection>

          <DocSubSection id="zinfer-component" title="zInfer">
            <p className="mb-4">
              Low-level inference engine powering both CLI and API.
            </p>
            <FeatureList features={[
              "Direct tensor operations",
              "Async token generation",
              "Multi-GPU support",
              "Dynamic batching",
            ]} />
          </DocSubSection>

          <DocSubSection id="zstream-component" title="zStream">
            <p className="mb-4">
              Layer streaming for running models larger than available VRAM.
            </p>
            <FeatureList features={[
              "GPU ↔ CPU layer offloading",
              "Automatic memory management",
              "Run 70B models on 24GB GPUs",
              "Minimal latency overhead",
            ]} />
          </DocSubSection>

          <DocSubSection id="zkv-component" title="zKV">
            <p className="mb-4">
              Quantized KV cache for 4× memory savings during generation.
            </p>
            <FeatureList features={[
              "8-bit KV cache quantization",
              "Paged attention support",
              "Longer context windows",
              "Sub-1% quality impact",
            ]} />
          </DocSubSection>
        </DocSection>

        <DocSection id="inference-flow" title="Inference Flow">
          <p className="mb-4">
            When a request comes in, here's what happens:
          </p>

          <div className="space-y-4 my-6">
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-bold text-lime">1</span>
              </div>
              <div>
                <p className="text-sm text-white font-medium">Request Received</p>
                <p className="text-sm text-white/50">zServe validates the request and adds it to the batch queue</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-bold text-lime">2</span>
              </div>
              <div>
                <p className="text-sm text-white font-medium">Tokenization</p>
                <p className="text-sm text-white/50">Input text is converted to token IDs using the model's tokenizer</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-bold text-lime">3</span>
              </div>
              <div>
                <p className="text-sm text-white font-medium">Forward Pass</p>
                <p className="text-sm text-white/50">zInfer runs the model layers, storing KV cache with zKV</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-bold text-lime">4</span>
              </div>
              <div>
                <p className="text-sm text-white font-medium">Token Sampling</p>
                <p className="text-sm text-white/50">Apply temperature, top-p, and sample next token</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center flex-shrink-0">
                <span className="text-xs font-bold text-lime">5</span>
              </div>
              <div>
                <p className="text-sm text-white font-medium">Streaming Response</p>
                <p className="text-sm text-white/50">Tokens are decoded and streamed back via SSE</p>
              </div>
            </div>
          </div>
        </DocSection>

        <DocSection id="memory-architecture" title="Memory Architecture">
          <p className="mb-4">
            ZSE uses a three-tier memory hierarchy:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Tier</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Storage</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Content</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Access Speed</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4 text-lime">Hot</td>
                  <td className="py-2 px-4">GPU VRAM</td>
                  <td className="py-2 px-4">Active layers, KV cache</td>
                  <td className="py-2 pl-4">~1 TB/s</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4 text-yellow-400">Warm</td>
                  <td className="py-2 px-4">System RAM</td>
                  <td className="py-2 px-4">Offloaded layers (zStream)</td>
                  <td className="py-2 pl-4">~50 GB/s</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 text-blue-400">Cold</td>
                  <td className="py-2 px-4">Disk (.zse)</td>
                  <td className="py-2 px-4">Model weights</td>
                  <td className="py-2 pl-4">~3 GB/s (NVMe)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="tip">
            The <InlineCode>.zse</InlineCode> format is designed for memory-mapped loading, 
            which means the OS kernel efficiently pages weights from disk as needed.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ title: 'First Model', href: '/docs/first-model' }}
          next={{ title: 'Model Formats', href: '/docs/model-formats' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
