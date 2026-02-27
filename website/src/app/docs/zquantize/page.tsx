'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { Steps, FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'quick-start', title: 'Quick Start', level: 2 },
  { id: 'quantization-types', title: 'Quantization Types', level: 2 },
  { id: 'advanced-options', title: 'Advanced Options', level: 2 },
  { id: 'batch-conversion', title: 'Batch Conversion', level: 2 },
  { id: 'quality-validation', title: 'Quality Validation', level: 2 },
]

export default function ZQuantizePage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="zQuantize"
          description="Convert models to optimized .zse format with extreme compression and minimal quality loss."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            <InlineCode>zQuantize</InlineCode> converts transformer models from HuggingFace, 
            safetensors, or GGUF to ZSE&apos;s native format with configurable quantization.
          </p>

          <FeatureList features={[
            "4-bit and 8-bit quantization with calibration",
            "NF4 (NormalFloat4) for best quality",
            "Group quantization for accuracy preservation",
            "Mixed precision for critical layers",
            "CUDA-accelerated conversion",
          ]} />
        </DocSection>

        <DocSection id="quick-start" title="Quick Start">
          <Steps steps={[
            {
              title: "Convert a model",
              description: "Run the convert command with default settings (NF4)",
              code: "zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse"
            },
            {
              title: "Verify the conversion",
              description: "Check model info and size",
              code: "zse info qwen-7b.zse"
            },
            {
              title: "Test the model",
              description: "Run a quick inference test",
              code: 'zse chat qwen-7b.zse -p "Hello, how are you?"'
            },
          ]} />

          <Callout type="info">
            Conversion time depends on model size and GPU availability. A 7B model takes 
            ~2 minutes on GPU or ~15 minutes on CPU.
          </Callout>
        </DocSection>

        <DocSection id="quantization-types" title="Quantization Types">
          <p className="mb-4">
            Choose the right quantization type for your memory/quality tradeoff:
          </p>

          <CardGrid columns={2}>
            <Card
              title="NF4 (Default)"
              description="Best quality 4-bit — normalized float distribution"
            />
            <Card
              title="INT4"
              description="Standard 4-bit — fastest conversion speed"
            />
            <Card
              title="INT8"
              description="8-bit integer — higher quality, larger files"
            />
            <Card
              title="FP16"
              description="Half precision — maximum quality, largest files"
            />
          </CardGrid>

          <DocSubSection id="nf4" title="NF4 Quantization">
            <p className="mb-2">
              NormalFloat4 is optimized for the weight distribution of neural networks:
            </p>

            <CodeBlock
              language="bash"
              code={`zse convert model-id -o model.zse --quant nf4`}
            />

            <FeatureList features={[
              "Asymmetric quantization grid",
              "Optimal for normally-distributed weights",
              "Industry-leading 4-bit quality",
              "Default for most models",
            ]} />
          </DocSubSection>

          <DocSubSection id="int4" title="INT4 Quantization">
            <p className="mb-2">
              Standard symmetric 4-bit integer quantization:
            </p>

            <CodeBlock
              language="bash"
              code={`zse convert model-id -o model.zse --quant int4`}
            />

            <FeatureList features={[
              "Symmetric quantization grid",
              "Faster conversion than NF4",
              "Compatible with more hardware",
              "Slightly lower quality than NF4",
            ]} />
          </DocSubSection>

          <DocSubSection id="int8" title="INT8 Quantization">
            <p className="mb-2">
              8-bit quantization for higher quality:
            </p>

            <CodeBlock
              language="bash"
              code={`zse convert model-id -o model.zse --quant int8`}
            />
          </DocSubSection>

          <DocSubSection id="fp16" title="FP16 (No Quantization)">
            <p className="mb-2">
              Convert without quantization — useful for fine-tuning or maximum quality:
            </p>

            <CodeBlock
              language="bash"
              code={`zse convert model-id -o model.zse --quant fp16`}
            />
          </DocSubSection>

          <div className="overflow-x-auto my-6">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Type</th>
                  <th className="text-right py-2 px-4 text-white/50 font-medium">7B Size</th>
                  <th className="text-right py-2 px-4 text-white/50 font-medium">Quality</th>
                  <th className="text-right py-2 pl-4 text-white/50 font-medium">Speed</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>nf4</InlineCode></td>
                  <td className="text-right py-2 px-4">4.2 GB</td>
                  <td className="text-right py-2 px-4 text-lime">★★★★☆</td>
                  <td className="text-right py-2 pl-4">★★★★☆</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>int4</InlineCode></td>
                  <td className="text-right py-2 px-4">4.0 GB</td>
                  <td className="text-right py-2 px-4">★★★☆☆</td>
                  <td className="text-right py-2 pl-4 text-lime">★★★★★</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>int8</InlineCode></td>
                  <td className="text-right py-2 px-4">7.5 GB</td>
                  <td className="text-right py-2 px-4 text-lime">★★★★★</td>
                  <td className="text-right py-2 pl-4">★★★☆☆</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>fp16</InlineCode></td>
                  <td className="text-right py-2 px-4">14 GB</td>
                  <td className="text-right py-2 px-4 text-lime">★★★★★</td>
                  <td className="text-right py-2 pl-4">★★☆☆☆</td>
                </tr>
              </tbody>
            </table>
          </div>
        </DocSection>

        <DocSection id="advanced-options" title="Advanced Options">
          <DocSubSection id="group-size" title="Group Size">
            <p className="mb-2">
              Control quantization granularity. Smaller groups = better quality, larger files:
            </p>

            <CodeBlock
              language="bash"
              code={`# Default group size (128)
zse convert model -o model.zse --quant nf4

# Smaller groups for better quality
zse convert model -o model.zse --quant nf4 --group-size 64

# Larger groups for smaller files
zse convert model -o model.zse --quant nf4 --group-size 256`}
            />
          </DocSubSection>

          <DocSubSection id="calibration" title="Calibration Dataset">
            <p className="mb-2">
              Use calibration data for optimal quantization ranges:
            </p>

            <CodeBlock
              language="bash"
              code={`# Use built-in calibration (default)
zse convert model -o model.zse --calibrate

# Use custom calibration data
zse convert model -o model.zse --calibrate-data ./prompts.txt

# Skip calibration (faster, slightly lower quality)
zse convert model -o model.zse --no-calibrate`}
            />

            <Callout type="tip">
              Custom calibration with domain-specific data can improve quality for specialized tasks.
            </Callout>
          </DocSubSection>

          <DocSubSection id="mixed-precision" title="Mixed Precision">
            <p className="mb-2">
              Keep critical layers at higher precision:
            </p>

            <CodeBlock
              language="bash"
              code={`# Keep embedding and output layers at FP16
zse convert model -o model.zse --quant nf4 --mixed-precision

# Keep specific layers at higher precision
zse convert model -o model.zse --quant nf4 --keep-fp16 "embed,lm_head"`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="batch-conversion" title="Batch Conversion">
          <p className="mb-4">
            Convert multiple models in a script:
          </p>

          <CodeBlock
            language="bash"
            code={`#!/bin/bash
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

for model in "\${MODELS[@]}"; do
  name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
  zse convert "$model" -o "./models/$name.zse" --quant nf4
done`}
          />

          <p className="mt-4 mb-2">
            Python API for programmatic conversion:
          </p>

          <CodeBlock
            language="python"
            code={`from zllm_zse import convert_model

# Convert with options
convert_model(
    source="Qwen/Qwen2.5-7B-Instruct",
    output="qwen-7b.zse",
    quant="nf4",
    group_size=128,
    calibrate=True
)

# Convert multiple models
models = ["model-a", "model-b", "model-c"]
for model in models:
    convert_model(model, f"{model}.zse")`}
          />
        </DocSection>

        <DocSection id="quality-validation" title="Quality Validation">
          <p className="mb-4">
            Verify quantized model quality with built-in benchmarks:
          </p>

          <CodeBlock
            language="bash"
            code={`# Run perplexity benchmark
zse benchmark qwen-7b.zse --metric perplexity

# Compare with original
zse benchmark qwen-7b.zse --compare Qwen/Qwen2.5-7B-Instruct

# Full evaluation suite
zse benchmark qwen-7b.zse --eval mmlu,hellaswag,arc`}
          />

          <p className="mt-4">
            Example output:
          </p>

          <CodeBlock
            language="text"
            code={`┌─────────────────────────────────────────────────────────┐
│ Model: qwen-7b.zse (NF4, 4.2 GB)                        │
├─────────────────────────────────────────────────────────┤
│ Perplexity:     5.42 (original: 5.38, Δ +0.7%)         │
│ MMLU:           64.2% (original: 64.8%, Δ -0.6%)       │
│ HellaSwag:      78.1% (original: 78.9%, Δ -0.8%)       │
│ ARC-Challenge:  52.3% (original: 53.1%, Δ -0.8%)       │
└─────────────────────────────────────────────────────────┘`}
          />

          <Callout type="success">
            Less than 1% quality loss is typical with NF4 quantization.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ title: 'Memory Management', href: '/docs/memory' }}
          next={{ title: 'zServe', href: '/docs/zserve' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
