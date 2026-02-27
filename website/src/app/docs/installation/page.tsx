'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList } from '@/components/docs/steps'

const tocItems = [
  { id: 'requirements', title: 'System Requirements', level: 2 },
  { id: 'gpu-requirements', title: 'GPU Requirements', level: 3 },
  { id: 'install-pip', title: 'Install with pip', level: 2 },
  { id: 'install-source', title: 'Install from Source', level: 2 },
  { id: 'verify', title: 'Verify Installation', level: 2 },
  { id: 'troubleshooting', title: 'Troubleshooting', level: 2 },
]

export default function InstallationPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Installation"
          description="Detailed installation instructions for ZSE, including system requirements, GPU setup, and troubleshooting common issues."
          badge="Getting Started"
        />

        <DocSection id="requirements" title="System Requirements">
          <p className="mb-4">
            ZSE is designed to run on a variety of hardware configurations. Here are the minimum 
            and recommended requirements:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Component</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Minimum</th>
                  <th className="text-left py-2 pl-4 text-lime font-medium">Recommended</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4 text-white">Python</td>
                  <td className="py-2 px-4">3.8</td>
                  <td className="py-2 pl-4 text-lime">3.10+</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4 text-white">RAM</td>
                  <td className="py-2 px-4">16 GB</td>
                  <td className="py-2 pl-4 text-lime">32 GB+</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4 text-white">Disk Space</td>
                  <td className="py-2 px-4">20 GB</td>
                  <td className="py-2 pl-4 text-lime">100 GB+</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4 text-white">OS</td>
                  <td className="py-2 px-4">Linux, macOS, Windows</td>
                  <td className="py-2 pl-4 text-lime">Linux (Ubuntu 20.04+)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <DocSubSection id="gpu-requirements" title="GPU Requirements">
            <p className="mb-4">
              GPU acceleration significantly improves performance. Here are the VRAM requirements by model size:
            </p>

            <div className="overflow-x-auto my-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-2 pr-4 text-white/50 font-medium">Model Size</th>
                    <th className="text-left py-2 px-4 text-white/50 font-medium">FP16</th>
                    <th className="text-left py-2 px-4 text-white/50 font-medium">INT8</th>
                    <th className="text-left py-2 pl-4 text-lime font-medium">INT4 (ZSE)</th>
                  </tr>
                </thead>
                <tbody className="text-white/70">
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4 text-white">7B</td>
                    <td className="py-2 px-4">14 GB</td>
                    <td className="py-2 px-4">8 GB</td>
                    <td className="py-2 pl-4 text-lime">5 GB</td>
                  </tr>
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4 text-white">14B</td>
                    <td className="py-2 px-4">28 GB</td>
                    <td className="py-2 px-4">16 GB</td>
                    <td className="py-2 pl-4 text-lime">10 GB</td>
                  </tr>
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4 text-white">32B</td>
                    <td className="py-2 px-4">64 GB</td>
                    <td className="py-2 px-4">32 GB</td>
                    <td className="py-2 pl-4 text-lime">20 GB</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 text-white">70B</td>
                    <td className="py-2 px-4">140 GB</td>
                    <td className="py-2 px-4">70 GB</td>
                    <td className="py-2 pl-4 text-lime">40 GB</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <Callout type="tip" title="Limited VRAM?">
              Use <InlineCode>zStream</InlineCode> layer offloading to run larger models on 
              GPUs with limited memory. See the <a href="/docs/zstream" className="text-lime hover:underline">zStream documentation</a>.
            </Callout>
          </DocSubSection>
        </DocSection>

        <DocSection id="install-pip" title="Install with pip">
          <p className="mb-4">
            The simplest way to install ZSE is via pip:
          </p>

          <CodeBlock
            language="bash"
            code="pip install zllm-zse"
          />

          <p className="mt-4 mb-2">
            Optional dependencies:
          </p>

          <CodeBlock
            language="bash"
            code={`# GGUF model support
pip install zllm-zse[gguf]

# All optional features
pip install zllm-zse[all]`}
          />

          <Callout type="info">
            We recommend using a virtual environment to avoid dependency conflicts.
          </Callout>

          <CodeBlock
            language="bash"
            code={`# Create virtual environment
python -m venv zse-env
source zse-env/bin/activate  # Linux/macOS
# or: zse-env\\Scripts\\activate  # Windows

# Install ZSE
pip install zllm-zse`}
          />
        </DocSection>

        <DocSection id="install-source" title="Install from Source">
          <p className="mb-4">
            For development or the latest features, install from source:
          </p>

          <CodeBlock
            language="bash"
            code={`git clone https://github.com/Zyora-Dev/zse.git
cd zse
pip install -e .`}
          />

          <p className="mt-4 mb-2">
            For development with testing dependencies:
          </p>

          <CodeBlock
            language="bash"
            code="pip install -e .[dev]"
          />
        </DocSection>

        <DocSection id="verify" title="Verify Installation">
          <p className="mb-4">
            Verify that ZSE is installed correctly:
          </p>

          <CodeBlock
            language="bash"
            code={`# Check version
zse --version

# Check hardware detection
zse hardware

# Run a quick test (downloads a small model)
zse chat --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Hello!"`}
          />

          <p className="mt-4">
            Expected <InlineCode>zse hardware</InlineCode> output:
          </p>

          <CodeBlock
            language="text"
            code={`ZSE Hardware Detection
━━━━━━━━━━━━━━━━━━━━━━
GPU 0: NVIDIA A100-SXM4-80GB
  Memory: 80 GB
  CUDA: 12.1
  Compute: 8.0

Total VRAM: 80 GB
Recommended models: Up to 70B (INT4)`}
          />
        </DocSection>

        <DocSection id="troubleshooting" title="Troubleshooting">
          <DocSubSection id="cuda-not-found" title="CUDA Not Found">
            <p className="mb-4">
              If ZSE doesn't detect your GPU, ensure CUDA is properly installed:
            </p>

            <CodeBlock
              language="bash"
              code={`# Check CUDA version
nvidia-smi

# If not found, install CUDA toolkit
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit

# Or install PyTorch with CUDA:
pip install torch --index-url https://download.pytorch.org/whl/cu121`}
            />
          </DocSubSection>

          <DocSubSection id="memory-errors" title="Out of Memory Errors">
            <p className="mb-4">
              If you encounter OOM errors:
            </p>

            <FeatureList features={[
              "Use a smaller model (7B instead of 32B)",
              "Enable zStream layer offloading: zse serve model --offload",
              "Reduce batch size: zse serve model --max-batch 1",
              "Use INT4 quantization (default) instead of INT8",
            ]} />
          </DocSubSection>

          <DocSubSection id="import-errors" title="Import Errors">
            <p className="mb-4">
              Common import errors and fixes:
            </p>

            <CodeBlock
              language="bash"
              code={`# Missing dependencies
pip install zllm-zse --upgrade

# Torch version conflicts
pip install torch>=2.0.0 --upgrade

# Clean reinstall
pip uninstall zllm-zse
pip cache purge
pip install zllm-zse`}
            />
          </DocSubSection>
        </DocSection>

        <DocNav
          prev={{ title: 'Quick Start', href: '/docs/quickstart' }}
          next={{ title: 'First Model', href: '/docs/first-model' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
