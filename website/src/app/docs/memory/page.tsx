'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'gpu-memory', title: 'GPU Memory', level: 2 },
  { id: 'model-sizing', title: 'Model Sizing', level: 2 },
  { id: 'memory-tiers', title: 'Memory Tiers', level: 2 },
  { id: 'optimization', title: 'Optimization Tips', level: 2 },
]

export default function MemoryPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Memory Management"
          description="Understanding and optimizing memory usage for model inference."
          badge="Core Concepts"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            Memory management is critical for running large language models. ZSE provides 
            multiple strategies to maximize model size while minimizing resource usage.
          </p>

          <CardGrid columns={3}>
            <Card
              title="4-bit Weights"
              description="4x reduction in model memory"
            />
            <Card
              title="4-bit KV Cache"
              description="4x longer context windows"
            />
            <Card
              title="Memory Tiers"
              description="GPU → CPU → Disk overflow"
            />
          </CardGrid>
        </DocSection>

        <DocSection id="gpu-memory" title="GPU Memory">
          <p className="mb-4">
            GPU memory is used for three main components:
          </p>

          <CodeBlock
            language="text"
            code={`Total GPU Memory
├── Model Weights (~60-80%)
│   └── Quantized weights + embedding layers
├── KV Cache (~15-30%)
│   └── Attention key-value pairs for context
├── Activations (~5-10%)
│   └── Intermediate computations during inference
└── CUDA Overhead (~2-5%)
    └── CUDA context, kernels, buffers`}
          />

          <DocSubSection id="memory-formula" title="Memory Estimation">
            <p className="mb-2">
              Estimate memory requirements:
            </p>

            <CodeBlock
              language="text"
              code={`Model Memory = (Parameters × Bits per Param) / 8

Examples (4-bit quantized):
  7B params × 4 bits / 8 = 3.5 GB
 14B params × 4 bits / 8 = 7.0 GB
 70B params × 4 bits / 8 = 35 GB

KV Cache Memory = 2 × Layers × Hidden × Context × Bytes per KV

Example (7B model, 8K context, FP16 KV):
  2 × 32 × 4096 × 8192 × 2 = 4.3 GB`}
            />
          </DocSubSection>

          <DocSubSection id="check-memory" title="Checking Memory">
            <CodeBlock
              language="bash"
              code={`# Check available GPU memory
zse hardware

# Check model memory requirements
zse info model.zse

# Monitor during inference
watch -n1 nvidia-smi`}
            />

            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# Get memory info
mem = model.memory_info()
print(f"Model: {mem['model_size'] / 1e9:.1f} GB")
print(f"KV Cache: {mem['kv_cache_size'] / 1e9:.1f} GB")
print(f"Free GPU: {mem['gpu_free'] / 1e9:.1f} GB")`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="model-sizing" title="Model Sizing">
          <p className="mb-4">
            Choose the right model size for your hardware:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">GPU VRAM</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Max Model (NF4)</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Context (FP16 KV)</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">8 GB</td>
                  <td className="py-2 px-4">7B</td>
                  <td className="py-2 pl-4">4K</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">12 GB</td>
                  <td className="py-2 px-4">14B</td>
                  <td className="py-2 pl-4">8K</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">16 GB</td>
                  <td className="py-2 px-4">14B</td>
                  <td className="py-2 pl-4">16K</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">24 GB</td>
                  <td className="py-2 px-4">34B</td>
                  <td className="py-2 pl-4">32K</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">48 GB</td>
                  <td className="py-2 px-4">70B</td>
                  <td className="py-2 pl-4">32K</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">80 GB</td>
                  <td className="py-2 px-4">70B</td>
                  <td className="py-2 pl-4">128K</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="tip">
            With 4-bit KV cache, you can double the context length in the table above.
          </Callout>
        </DocSection>

        <DocSection id="memory-tiers" title="Memory Tiers">
          <p className="mb-4">
            ZSE supports tiered memory for running models larger than GPU memory:
          </p>

          <CodeBlock
            language="text"
            code={`┌─────────────────────────────────────────────────────────┐
│ Tier 1: GPU Memory (fastest)                            │
│   - Model weights (always)                              │
│   - Active KV cache                                     │
│   - Current activations                                 │
├─────────────────────────────────────────────────────────┤
│ Tier 2: CPU Memory (slower)                             │
│   - Overflow KV cache                                   │
│   - Inactive model layers (offloading)                  │
├─────────────────────────────────────────────────────────┤
│ Tier 3: Disk (slowest)                                  │
│   - Cold KV cache                                       │
│   - Very long context overflow                          │
└─────────────────────────────────────────────────────────┘`}
          />

          <DocSubSection id="cpu-offloading" title="CPU Offloading">
            <CodeBlock
              language="bash"
              code={`# Offload some layers to CPU
zse serve model.zse --offload-layers 8

# Auto-detect optimal offload
zse serve model.zse --offload auto`}
            />

            <Callout type="warning">
              CPU offloading reduces throughput by 2-5x. Use only when GPU memory is insufficient.
            </Callout>
          </DocSubSection>

          <DocSubSection id="disk-cache" title="Disk-Based KV Cache">
            <CodeBlock
              language="yaml"
              filename="zse.yaml"
              code={`kv_cache:
  tiers:
    - type: gpu
      size: auto        # Use available GPU memory
      
    - type: cpu
      size: 32GB        # Overflow to system RAM
      
    - type: disk
      path: /tmp/zse_cache
      size: 100GB       # For very long contexts`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="optimization" title="Optimization Tips">
          <FeatureList features={[
            "Use NF4 quantization for best quality/size ratio",
            "Enable 4-bit KV cache for long contexts",
            "Set max_context to your actual needs",
            "Use prompt caching for repeated prefixes",
            "Monitor with nvidia-smi during development",
          ]} />

          <DocSubSection id="oom-errors" title="Handling OOM Errors">
            <CodeBlock
              language="bash"
              code={`# If you get CUDA OOM errors:

# 1. Reduce context length
zse serve model.zse --max-context 2048

# 2. Enable KV cache compression
zse serve model.zse --kv-quant int4

# 3. Reduce batch size
zse serve model.zse --max-batch 8

# 4. Try a smaller model
zse convert Qwen/Qwen2.5-3B-Instruct -o qwen-3b.zse`}
            />
          </DocSubSection>

          <DocSubSection id="benchmarking" title="Memory Benchmarking">
            <CodeBlock
              language="bash"
              code={`# Benchmark memory usage
zse benchmark model.zse --metric memory

# Profile memory over time
zse serve model.zse --profile-memory --profile-output memory.json`}
            />
          </DocSubSection>
        </DocSection>

        <DocNav
          prev={{ title: 'Quantization', href: '/docs/quantization' }}
          next={{ title: 'zQuantize', href: '/docs/zquantize' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
