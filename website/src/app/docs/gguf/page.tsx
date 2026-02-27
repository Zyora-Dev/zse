'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'supported-formats', title: 'Supported Formats', level: 2 },
  { id: 'installation', title: 'Installation', level: 2 },
  { id: 'usage', title: 'Usage', level: 2 },
  { id: 'gpu-offloading', title: 'GPU Offloading', level: 2 },
  { id: 'zse-vs-gguf', title: '.zse vs GGUF', level: 2 },
]

export default function GGUFPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="GGUF Compatibility"
          description="Run GGUF models via llama.cpp backend for compatibility with existing model files."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            ZSE includes full support for GGUF models via the llama-cpp-python backend.
            This lets you run existing GGUF models from Hugging Face or other sources
            while maintaining API compatibility with the rest of ZSE.
          </p>

          <CardGrid columns={3}>
            <Card
              title="GGUF v2/v3"
              description="Full support for modern GGUF format versions"
            />
            <Card
              title="All Quant Types"
              description="Q4_K_M, Q5_K_M, Q8_0, and more"
            />
            <Card
              title="GPU Offloading"
              description="Configurable layer offloading to GPU"
            />
          </CardGrid>

          <FeatureList features={[
            "Parse GGUF v2/v3 format metadata",
            "Support all GGML quantization types",
            "Streaming and non-streaming generation",
            "Chat completion support",
            "GPU layer offloading configuration",
            "Seamless integration with ZSE server",
          ]} />
        </DocSection>

        <DocSection id="supported-formats" title="Supported Formats">
          <p className="mb-4">
            ZSE supports all standard GGML quantization types found in GGUF files:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Format</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Bits</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Use Case</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">Q4_K_M</td>
                  <td className="py-3 px-4 text-white">4-bit</td>
                  <td className="py-3 px-4 text-gray-400">Best balance of size/quality</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">Q5_K_M</td>
                  <td className="py-3 px-4 text-white">5-bit</td>
                  <td className="py-3 px-4 text-gray-400">Higher quality, slightly larger</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">Q8_0</td>
                  <td className="py-3 px-4 text-white">8-bit</td>
                  <td className="py-3 px-4 text-gray-400">Near-lossless quality</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">Q2_K</td>
                  <td className="py-3 px-4 text-white">2-bit</td>
                  <td className="py-3 px-4 text-gray-400">Maximum compression</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">Q6_K</td>
                  <td className="py-3 px-4 text-white">6-bit</td>
                  <td className="py-3 px-4 text-gray-400">High quality</td>
                </tr>
              </tbody>
            </table>
          </div>
        </DocSection>

        <DocSection id="installation" title="Installation">
          <DocSubSection id="cpu-only" title="CPU Only">
            <CodeBlock
              language="bash"
              code={`pip install llama-cpp-python`}
            />
          </DocSubSection>

          <DocSubSection id="with-cuda" title="With CUDA (Recommended)">
            <CodeBlock
              language="bash"
              code={`CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python`}
            />
          </DocSubSection>

          <DocSubSection id="with-metal" title="With Metal (macOS)">
            <CodeBlock
              language="bash"
              code={`CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python`}
            />
          </DocSubSection>

          <Callout type="info">
            GPU acceleration is highly recommended. CPU-only inference is 10-50x slower.
          </Callout>
        </DocSection>

        <DocSection id="usage" title="Usage">
          <DocSubSection id="python-api" title="Python API">
            <CodeBlock
              language="python"
              code={`from zse.gguf import GGUFWrapper, is_gguf_file

# Check if file is GGUF format
if is_gguf_file("model-Q4_K_M.gguf"):
    # Create wrapper (matches IntelligenceOrchestrator API)
    wrapper = GGUFWrapper("model-Q4_K_M.gguf")
    wrapper.load()
    
    # Streaming generation
    for text in wrapper.generate("Hello, how are you?"):
        print(text, end="")

    # Chat completion
    response = wrapper.chat([
        {"role": "user", "content": "Write a haiku about coding"}
    ])
    print(response)`}
            />
          </DocSubSection>

          <DocSubSection id="cli" title="CLI">
            <CodeBlock
              language="bash"
              code={`# Auto-detect GGUF format and serve
zse serve model-Q4_K_M.gguf

# Show GGUF metadata
zse info model-Q4_K_M.gguf

# Run inference directly
zse infer model-Q4_K_M.gguf --prompt "Hello, world!"`}
            />
          </DocSubSection>

          <DocSubSection id="reading-metadata" title="Reading Metadata">
            <CodeBlock
              language="python"
              code={`from zse.gguf import GGUFReader

reader = GGUFReader("model-Q4_K_M.gguf")
metadata = reader.read_metadata()

print(f"Architecture: {metadata['architecture']}")
print(f"Context Length: {metadata['context_length']}")
print(f"Layers: {metadata['num_layers']}")
print(f"Quantization: {metadata['quantization_type']}")`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="gpu-offloading" title="GPU Offloading">
          <p className="mb-4">
            Configure how many layers to offload to GPU for faster inference:
          </p>

          <CodeBlock
            language="python"
            code={`from zse.gguf import GGUFWrapper

# Offload all layers to GPU (fastest, requires most VRAM)
wrapper = GGUFWrapper("model.gguf", n_gpu_layers=-1)

# Offload first 20 layers to GPU
wrapper = GGUFWrapper("model.gguf", n_gpu_layers=20)

# CPU only (no GPU offloading)
wrapper = GGUFWrapper("model.gguf", n_gpu_layers=0)

# Auto-detect optimal layers based on available VRAM
wrapper = GGUFWrapper("model.gguf")  # Default behavior`}
          />

          <Callout type="warning">
            The more layers offloaded to GPU, the faster inference will be. However, 
            you need sufficient VRAM. If you run out of memory, reduce n_gpu_layers.
          </Callout>
        </DocSection>

        <DocSection id="zse-vs-gguf" title=".zse vs GGUF">
          <p className="mb-4">
            While GGUF is widely supported, native <InlineCode>.zse</InlineCode> format 
            offers significant advantages:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Feature</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">.zse</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">GGUF</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">Memory Allocation</td>
                  <td className="py-3 px-4 text-lime">Streaming (on-demand)</td>
                  <td className="py-3 px-4 text-gray-400">Static (all at once)</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">Cold Start</td>
                  <td className="py-3 px-4 text-lime">3.9s (7B model)</td>
                  <td className="py-3 px-4 text-gray-400">10-30s typical</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">Memory Efficiency</td>
                  <td className="py-3 px-4 text-lime">Load only needed layers</td>
                  <td className="py-3 px-4 text-gray-400">Full model in RAM+VRAM</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">Quantization</td>
                  <td className="py-3 px-4 text-lime">INT4 @ full precision</td>
                  <td className="py-3 px-4 text-gray-400">Various (Q4_K_M, etc.)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="info">
            Use GGUF for compatibility with existing model files. Convert to .zse for 
            optimal performance with ZSE&apos;s streaming inference engine.
          </Callout>

          <CodeBlock
            language="bash"
            code={`# Convert GGUF to .zse for better performance
zse convert model-Q4_K_M.gguf -o model.zse`}
          />
        </DocSection>

        <DocNav
          prev={{ href: '/docs/multi-gpu', title: 'Multi-GPU' }}
          next={{ href: '/docs/api/cli', title: 'CLI Commands' }}
        />
      </article>
      
      <TableOfContents items={tocItems} />
    </div>
  )
}
