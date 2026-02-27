'use client'

import { DocHeader, DocSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { Card, CardGrid, FeatureList } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'how-it-works', title: 'How It Works', level: 2 },
  { id: 'quantization-types', title: 'Quantization Types', level: 2 },
  { id: 'converting', title: 'Converting Models', level: 2 },
  { id: 'quality', title: 'Quality Comparison', level: 2 },
]

export default function QuantizationPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Quantization"
          description="Understanding ZSE's quantization options and how they affect performance, memory, and output quality."
          badge="Core Concepts"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            Quantization reduces model precision from 16-bit floats to smaller formats like 4-bit integers. 
            This dramatically reduces memory usage and improves inference speed with minimal quality loss.
          </p>

          <CardGrid columns={3}>
            <Card
              title="63-72%"
              description="Memory reduction with INT4"
            />
            <Card
              title="11.6×"
              description="Faster cold starts"
            />
            <Card
              title="< 1%"
              description="Quality degradation"
            />
          </CardGrid>
        </DocSection>

        <DocSection id="how-it-works" title="How It Works">
          <p className="mb-4">
            Traditional inference engines quantize models at runtime, which takes 30-60 seconds for 7B models. 
            ZSE pre-quantizes models to the <InlineCode>.zse</InlineCode> format, eliminating this overhead.
          </p>

          <div className="my-6 p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]">
            <div className="flex items-center gap-4 text-sm">
              <div className="flex-1 text-center">
                <div className="text-white/50 mb-1">Original Model</div>
                <div className="text-white font-semibold">14 GB (FP16)</div>
              </div>
              <div className="text-lime">→</div>
              <div className="flex-1 text-center">
                <div className="text-white/50 mb-1">zQuantize</div>
                <div className="text-lime font-semibold">Pre-quantize</div>
              </div>
              <div className="text-lime">→</div>
              <div className="flex-1 text-center">
                <div className="text-white/50 mb-1">.zse File</div>
                <div className="text-white font-semibold">4.2 GB (INT4)</div>
              </div>
            </div>
          </div>

          <p className="mb-4">
            The <InlineCode>.zse</InlineCode> format stores:
          </p>

          <FeatureList features={[
            "Pre-quantized weights in INT4/NF4 format",
            "Per-tensor scaling factors for accuracy",
            "Model architecture and tokenizer",
            "Optimized memory layout for fast loading",
          ]} />
        </DocSection>

        <DocSection id="quantization-types" title="Quantization Types">
          <p className="mb-4">
            ZSE supports multiple quantization formats:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Type</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Bits</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Memory</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Quality</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Use Case</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>int4</InlineCode></td>
                  <td className="py-2 px-4">4</td>
                  <td className="py-2 px-4 text-lime">Best</td>
                  <td className="py-2 px-4">Good</td>
                  <td className="py-2 pl-4">Default, production</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>nf4</InlineCode></td>
                  <td className="py-2 px-4">4</td>
                  <td className="py-2 px-4 text-lime">Best</td>
                  <td className="py-2 px-4 text-lime">Better</td>
                  <td className="py-2 pl-4">Higher quality 4-bit</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>int8</InlineCode></td>
                  <td className="py-2 px-4">8</td>
                  <td className="py-2 px-4">Good</td>
                  <td className="py-2 px-4 text-lime">Best</td>
                  <td className="py-2 pl-4">Quality-sensitive tasks</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>fp16</InlineCode></td>
                  <td className="py-2 px-4">16</td>
                  <td className="py-2 px-4">Baseline</td>
                  <td className="py-2 px-4 text-lime">Perfect</td>
                  <td className="py-2 pl-4">Reference, unlimited VRAM</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="tip" title="Recommendation">
            Use <InlineCode>nf4</InlineCode> (Normalized Float 4) for the best balance of quality and memory. 
            It uses a non-linear quantization scheme that better preserves model accuracy.
          </Callout>
        </DocSection>

        <DocSection id="converting" title="Converting Models">
          <p className="mb-4">
            Convert any HuggingFace model to .zse format:
          </p>

          <CodeBlock
            language="bash"
            code={`# INT4 (default, smallest size)
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse

# NF4 (recommended, better quality)
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b-nf4.zse --quant nf4

# INT8 (larger but higher quality)
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b-int8.zse --quant int8`}
          />

          <p className="mt-4 mb-2">
            Conversion time depends on model size:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Model Size</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Conversion Time</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">7B</td>
                  <td className="py-2 pl-4">~20 seconds</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">14B</td>
                  <td className="py-2 pl-4">~45 seconds</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">32B</td>
                  <td className="py-2 pl-4">~2 minutes</td>
                </tr>
              </tbody>
            </table>
          </div>
        </DocSection>

        <DocSection id="quality" title="Quality Comparison">
          <p className="mb-4">
            Benchmark results on common evaluation tasks (higher is better):
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Benchmark</th>
                  <th className="text-center py-2 px-4 text-white/50 font-medium">FP16</th>
                  <th className="text-center py-2 px-4 text-white/50 font-medium">INT8</th>
                  <th className="text-center py-2 px-4 text-lime font-medium">NF4</th>
                  <th className="text-center py-2 pl-4 text-white/50 font-medium">INT4</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">MMLU</td>
                  <td className="text-center py-2 px-4">68.2</td>
                  <td className="text-center py-2 px-4">67.9</td>
                  <td className="text-center py-2 px-4 text-lime">67.5</td>
                  <td className="text-center py-2 pl-4">66.8</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">HumanEval</td>
                  <td className="text-center py-2 px-4">61.0</td>
                  <td className="text-center py-2 px-4">60.4</td>
                  <td className="text-center py-2 px-4 text-lime">59.8</td>
                  <td className="text-center py-2 pl-4">58.5</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">GSM8K</td>
                  <td className="text-center py-2 px-4">82.5</td>
                  <td className="text-center py-2 px-4">82.1</td>
                  <td className="text-center py-2 px-4 text-lime">81.4</td>
                  <td className="text-center py-2 pl-4">79.8</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="info">
            These benchmarks are for Qwen 2.5 7B. Quality retention varies by model architecture, 
            but most modern LLMs maintain 95%+ of their original capability with NF4 quantization.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ title: 'Model Formats', href: '/docs/model-formats' }}
          next={{ title: 'Memory Management', href: '/docs/memory' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
