'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'how-it-works', title: 'How It Works', level: 2 },
  { id: 'usage', title: 'Usage', level: 2 },
  { id: 'vram-distribution', title: 'VRAM Distribution', level: 2 },
  { id: 'benchmarks', title: 'Benchmarks', level: 2 },
]

export default function MultiGPUPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Multi-GPU Support"
          description="Run larger models by distributing weights across multiple GPUs."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            ZSE supports multi-GPU inference, allowing you to run models that don&apos;t fit on
            a single GPU by automatically sharding layers across available devices.
          </p>

          <CardGrid columns={3}>
            <Card
              title="Auto Detection"
              description="Automatically detects available GPUs and VRAM"
            />
            <Card
              title="Model Sharding"
              description="Distributes layers evenly across GPUs"
            />
            <Card
              title="VRAM Balancing"
              description="Configurable memory limits per device"
            />
          </CardGrid>

          <FeatureList features={[
            "Automatic GPU count and memory detection",
            "HuggingFace accelerate integration",
            "Even layer distribution across GPUs",
            "Configurable max_memory per GPU",
            "Support for mixed GPU configurations",
          ]} />
        </DocSection>

        <DocSection id="how-it-works" title="How It Works">
          <p className="mb-4">
            Multi-GPU inference works by distributing model layers across available GPUs.
            ZSE uses HuggingFace&apos;s <InlineCode>device_map=&quot;auto&quot;</InlineCode> under the hood
            for automatic layer distribution.
          </p>

          <CodeBlock
            language="text"
            code={`Model: Qwen 32B (64 layers)
Available: 2x A10G (24GB each)

Layer Distribution:
┌─────────────┐    ┌─────────────┐
│   GPU 0     │    │   GPU 1     │
│             │    │             │
│ Layers 0-31 │    │ Layers 32-63│
│             │    │             │
│ ~12GB VRAM  │    │ ~12GB VRAM  │
└─────────────┘    └─────────────┘

Forward Pass:
Input → GPU 0 → Transfer → GPU 1 → Output`}
          />

          <Callout type="info">
            Inter-GPU communication adds some latency, but enables running models that 
            wouldn&apos;t fit on a single GPU. The overhead is typically 5-15%.
          </Callout>
        </DocSection>

        <DocSection id="usage" title="Usage">
          <DocSubSection id="python-api" title="Python API">
            <CodeBlock
              language="python"
              code={`from zse.engine.orchestrator import IntelligenceOrchestrator

# Auto-detect and use all available GPUs
orch = IntelligenceOrchestrator.multi_gpu("Qwen/Qwen2.5-Coder-7B-Instruct")
orch.load()  # Model automatically sharded across GPUs

# Or specify which GPUs to use
orch = IntelligenceOrchestrator.multi_gpu("model_name", gpu_ids=[0, 1])

# Check GPU info
info = IntelligenceOrchestrator.get_gpu_info()
print(f"GPUs: {info['count']}, Total VRAM: {info['total_memory'] / 1e9:.1f} GB")`}
            />
          </DocSubSection>

          <DocSubSection id="cli" title="CLI">
            <CodeBlock
              language="bash"
              code={`# Auto-detect GPUs and distribute model
zse serve model.zse --multi-gpu

# Specify GPU IDs
zse serve model.zse --gpus 0,1,2

# Set memory limits per GPU
zse serve model.zse --multi-gpu --max-memory "0:20GB,1:20GB"`}
            />
          </DocSubSection>

          <DocSubSection id="gpu-info" title="Checking GPU Info">
            <CodeBlock
              language="python"
              code={`from zse.engine.orchestrator import IntelligenceOrchestrator

info = IntelligenceOrchestrator.get_gpu_info()
# Returns:
# {
#   "count": 2,
#   "devices": [
#     {"id": 0, "name": "NVIDIA A10G", "memory": 24576000000},
#     {"id": 1, "name": "NVIDIA A10G", "memory": 24576000000}
#   ],
#   "total_memory": 49152000000
# }`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="vram-distribution" title="VRAM Distribution">
          <p className="mb-4">
            ZSE automatically balances VRAM usage across GPUs. You can also configure 
            maximum memory per GPU for fine-grained control.
          </p>

          <CodeBlock
            language="python"
            code={`# Set max memory per GPU (useful for reserving memory for KV cache)
orch = IntelligenceOrchestrator.multi_gpu(
    "model_name",
    max_memory={
        0: "20GB",  # Leave 4GB for KV cache on GPU 0
        1: "22GB",  # Leave 2GB for KV cache on GPU 1
    }
)`}
          />

          <Callout type="warning">
            Reserve some VRAM for the KV cache and CUDA kernels. Using 100% of available 
            memory will cause out-of-memory errors during inference.
          </Callout>
        </DocSection>

        <DocSection id="benchmarks" title="Benchmarks">
          <p className="mb-4">
            Verified on Modal with 2x A10G GPUs:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Test</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Result</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">GPU Detection</td>
                  <td className="py-3 px-4 text-lime">2 GPUs detected ✓</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">Model Load (Qwen 7B FP16)</td>
                  <td className="py-3 px-4 text-lime">80s ✓</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">VRAM Distribution</td>
                  <td className="py-3 px-4 text-lime">GPU 0: 6.22 GB, GPU 1: 7.96 GB ✓</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">Generation Speed</td>
                  <td className="py-3 px-4 text-lime">100 tokens @ 15.0 tok/s ✓</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="info">
            Multi-GPU is most beneficial for models larger than 30B parameters, where 
            the memory savings outweigh the inter-GPU communication overhead.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ href: '/docs/zkv', title: 'zKV' }}
          next={{ href: '/docs/gguf', title: 'GGUF Compatibility' }}
        />
      </article>
      
      <TableOfContents items={tocItems} />
    </div>
  )
}
