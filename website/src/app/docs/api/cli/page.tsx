'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'zse-serve', title: 'zse serve', level: 2 },
  { id: 'zse-convert', title: 'zse convert', level: 2 },
  { id: 'zse-chat', title: 'zse chat', level: 2 },
  { id: 'zse-info', title: 'zse info', level: 2 },
  { id: 'zse-benchmark', title: 'zse benchmark', level: 2 },
  { id: 'zse-hardware', title: 'zse hardware', level: 2 },
  { id: 'global-options', title: 'Global Options', level: 2 },
]

export default function CLIPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="CLI Commands"
          description="Complete reference for all ZSE command-line interface commands and options."
          badge="API Reference"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            ZSE provides a powerful CLI for model management, serving, and inference. All commands 
            follow the pattern:
          </p>

          <CodeBlock
            language="bash"
            code="zse <command> [options] [arguments]"
          />

          <p className="mt-4 mb-2">
            Get help for any command:
          </p>

          <CodeBlock
            language="bash"
            code={`zse --help           # List all commands
zse serve --help     # Help for specific command`}
          />
        </DocSection>

        <DocSection id="zse-serve" title="zse serve">
          <p className="mb-4">
            Start an OpenAI-compatible API server.
          </p>

          <CodeBlock
            language="bash"
            code="zse serve <model> [options]"
          />

          <p className="mt-4 mb-2">
            <strong className="text-white">Arguments:</strong>
          </p>
          <ul className="space-y-2 text-sm mb-4">
            <li className="flex gap-3">
              <InlineCode>model</InlineCode>
              <span>Model name (HuggingFace ID) or path to .zse/.gguf file</span>
            </li>
          </ul>

          <p className="mb-2">
            <strong className="text-white">Options:</strong>
          </p>
          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Option</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Default</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--port</InlineCode></td>
                  <td className="py-2 px-4">8000</td>
                  <td className="py-2 pl-4">Server port</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--host</InlineCode></td>
                  <td className="py-2 px-4">127.0.0.1</td>
                  <td className="py-2 pl-4">Bind address</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--api-key</InlineCode></td>
                  <td className="py-2 px-4">None</td>
                  <td className="py-2 pl-4">Require API key authentication</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--quant</InlineCode></td>
                  <td className="py-2 px-4">int4</td>
                  <td className="py-2 pl-4">Quantization type: int4, int8, nf4, fp16</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--max-batch</InlineCode></td>
                  <td className="py-2 px-4">8</td>
                  <td className="py-2 pl-4">Maximum batch size</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--max-tokens</InlineCode></td>
                  <td className="py-2 px-4">4096</td>
                  <td className="py-2 pl-4">Maximum output tokens</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--offload</InlineCode></td>
                  <td className="py-2 px-4">False</td>
                  <td className="py-2 pl-4">Enable layer offloading (zStream)</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>--gpu</InlineCode></td>
                  <td className="py-2 px-4">auto</td>
                  <td className="py-2 pl-4">GPU device ID or "auto"</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="mb-2">
            <strong className="text-white">Examples:</strong>
          </p>

          <CodeBlock
            language="bash"
            code={`# Basic usage
zse serve Qwen/Qwen2.5-7B-Instruct

# Production server
zse serve qwen-7b.zse --host 0.0.0.0 --port 8080 --api-key sk-xxx

# With layer offloading for large models
zse serve Qwen/Qwen2.5-32B-Instruct --offload

# Specific GPU
zse serve model.zse --gpu 1`}
          />
        </DocSection>

        <DocSection id="zse-convert" title="zse convert">
          <p className="mb-4">
            Convert models to the optimized <InlineCode>.zse</InlineCode> format.
          </p>

          <CodeBlock
            language="bash"
            code="zse convert <model> [options]"
          />

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Option</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Default</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>-o, --output</InlineCode></td>
                  <td className="py-2 px-4">model.zse</td>
                  <td className="py-2 pl-4">Output file path</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--quant</InlineCode></td>
                  <td className="py-2 px-4">int4</td>
                  <td className="py-2 pl-4">Quantization type</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>--format</InlineCode></td>
                  <td className="py-2 px-4">zse</td>
                  <td className="py-2 pl-4">Output format: zse, gguf</td>
                </tr>
              </tbody>
            </table>
          </div>

          <CodeBlock
            language="bash"
            code={`# Convert to .zse
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse

# Convert with NF4 quantization
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b-nf4.zse --quant nf4

# Convert to GGUF
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.gguf --format gguf`}
          />
        </DocSection>

        <DocSection id="zse-chat" title="zse chat">
          <p className="mb-4">
            Interactive chat mode for quick testing.
          </p>

          <CodeBlock
            language="bash"
            code="zse chat <model> [options]"
          />

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Option</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Default</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--prompt</InlineCode></td>
                  <td className="py-2 px-4">None</td>
                  <td className="py-2 pl-4">Single prompt (non-interactive)</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--system</InlineCode></td>
                  <td className="py-2 px-4">None</td>
                  <td className="py-2 pl-4">System prompt</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>--temperature</InlineCode></td>
                  <td className="py-2 px-4">0.7</td>
                  <td className="py-2 pl-4">Sampling temperature</td>
                </tr>
              </tbody>
            </table>
          </div>

          <CodeBlock
            language="bash"
            code={`# Interactive chat
zse chat Qwen/Qwen2.5-7B-Instruct

# Single prompt
zse chat Qwen/Qwen2.5-7B-Instruct --prompt "Explain quantum computing"

# With system prompt
zse chat model.zse --system "You are a helpful coding assistant"`}
          />
        </DocSection>

        <DocSection id="zse-info" title="zse info">
          <p className="mb-4">
            Display information about a model.
          </p>

          <CodeBlock
            language="bash"
            code="zse info <model>"
          />

          <CodeBlock
            language="bash"
            code={`$ zse info qwen-7b.zse

Model Information
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name:          Qwen/Qwen2.5-7B-Instruct
Format:        .zse
Quantization:  INT4
Parameters:    7.6B
Size:          4.2 GB
Vocab Size:    152064
Context:       32768
Created:       2024-02-25`}
          />
        </DocSection>

        <DocSection id="zse-benchmark" title="zse benchmark">
          <p className="mb-4">
            Run performance benchmarks.
          </p>

          <CodeBlock
            language="bash"
            code="zse benchmark <model> [options]"
          />

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Option</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Default</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--iterations</InlineCode></td>
                  <td className="py-2 px-4">10</td>
                  <td className="py-2 pl-4">Number of iterations</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--warmup</InlineCode></td>
                  <td className="py-2 px-4">2</td>
                  <td className="py-2 pl-4">Warmup iterations</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>--output</InlineCode></td>
                  <td className="py-2 px-4">None</td>
                  <td className="py-2 pl-4">Save results to JSON file</td>
                </tr>
              </tbody>
            </table>
          </div>

          <CodeBlock
            language="bash"
            code={`$ zse benchmark qwen-7b.zse

ZSE Benchmark Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model:              qwen-7b.zse
Cold Start:         3.9s
First Token:        45ms
Tokens/sec:         82.4
Memory (Peak):      5.2 GB`}
          />
        </DocSection>

        <DocSection id="zse-hardware" title="zse hardware">
          <p className="mb-4">
            Detect and display hardware information.
          </p>

          <CodeBlock
            language="bash"
            code="zse hardware"
          />

          <Callout type="info">
            This command helps you understand what models your hardware can support and 
            diagnose GPU detection issues.
          </Callout>
        </DocSection>

        <DocSection id="global-options" title="Global Options">
          <p className="mb-4">
            These options are available for all commands:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Option</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--verbose, -v</InlineCode></td>
                  <td className="py-2 pl-4">Enable verbose output</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--quiet, -q</InlineCode></td>
                  <td className="py-2 pl-4">Suppress non-essential output</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>--version</InlineCode></td>
                  <td className="py-2 pl-4">Show version and exit</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>--help</InlineCode></td>
                  <td className="py-2 pl-4">Show help message</td>
                </tr>
              </tbody>
            </table>
          </div>
        </DocSection>

        <DocNav
          prev={{ title: 'zKV', href: '/docs/zkv' }}
          next={{ title: 'Python API', href: '/docs/api/python' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
