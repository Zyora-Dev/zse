'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'cli-usage', title: 'CLI Usage', level: 2 },
  { id: 'python-api', title: 'Python API', level: 2 },
  { id: 'parameters', title: 'Parameters', level: 2 },
  { id: 'chat-templates', title: 'Chat Templates', level: 2 },
  { id: 'batch-inference', title: 'Batch Inference', level: 2 },
]

export default function ZInferPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="zInfer"
          description="High-performance local inference for transformer models with optimized sampling."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            <InlineCode>zInfer</InlineCode> provides direct inference capabilities for 
            both interactive chat and programmatic text generation.
          </p>

          <CardGrid columns={3}>
            <Card
              title="~100 tok/s"
              description="High throughput on consumer GPUs"
            />
            <Card
              title="Flash Attention"
              description="Memory-efficient attention"
            />
            <Card
              title="Speculative"
              description="2-3x faster with draft models"
            />
          </CardGrid>

          <FeatureList features={[
            "Optimized CUDA kernels for inference",
            "Flash Attention 2 support",
            "Speculative decoding with draft models",
            "Continuous batching for throughput",
            "Custom sampling strategies",
          ]} />
        </DocSection>

        <DocSection id="cli-usage" title="CLI Usage">
          <DocSubSection id="chat" title="Interactive Chat">
            <CodeBlock
              language="bash"
              code={`# Start interactive chat
zse chat qwen-7b.zse

# With system prompt
zse chat qwen-7b.zse --system "You are a helpful coding assistant"

# With initial prompt
zse chat qwen-7b.zse -p "Explain quantum computing"`}
            />

            <p className="mt-4 mb-2">
              Chat commands:
            </p>

            <div className="overflow-x-auto my-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-2 pr-4 text-white/50 font-medium">Command</th>
                    <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                  </tr>
                </thead>
                <tbody className="text-white/70">
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4"><InlineCode>/clear</InlineCode></td>
                    <td className="py-2 pl-4">Clear conversation history</td>
                  </tr>
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4"><InlineCode>/system &lt;prompt&gt;</InlineCode></td>
                    <td className="py-2 pl-4">Set system prompt</td>
                  </tr>
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4"><InlineCode>/temp &lt;value&gt;</InlineCode></td>
                    <td className="py-2 pl-4">Set temperature (0.0-2.0)</td>
                  </tr>
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4"><InlineCode>/save &lt;file&gt;</InlineCode></td>
                    <td className="py-2 pl-4">Save conversation to file</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4"><InlineCode>/quit</InlineCode></td>
                    <td className="py-2 pl-4">Exit chat</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </DocSubSection>

          <DocSubSection id="completion" title="Text Completion">
            <CodeBlock
              language="bash"
              code={`# Single completion
zse complete qwen-7b.zse -p "The quick brown fox"

# With parameters
zse complete qwen-7b.zse \\
  -p "Write a poem about AI" \\
  --max-tokens 200 \\
  --temperature 0.8`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="python-api" title="Python API">
          <DocSubSection id="quick-inference" title="Quick Inference">
            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE

# Load model
model = ZSE("qwen-7b.zse")

# Chat completion
response = model.chat([
    {"role": "user", "content": "Hello!"}
])
print(response)

# Text completion
text = model.complete("The meaning of life is")
print(text)`}
            />
          </DocSubSection>

          <DocSubSection id="streaming" title="Streaming">
            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# Stream chat response
for chunk in model.chat_stream([
    {"role": "user", "content": "Tell me a story"}
]):
    print(chunk, end="", flush=True)

# Stream completion
for token in model.complete_stream("Once upon a time"):
    print(token, end="", flush=True)`}
            />
          </DocSubSection>

          <DocSubSection id="async" title="Async API">
            <CodeBlock
              language="python"
              code={`import asyncio
from zllm_zse import AsyncZSE

async def main():
    model = AsyncZSE("qwen-7b.zse")
    
    # Async chat
    response = await model.chat([
        {"role": "user", "content": "Hello!"}
    ])
    
    # Async streaming
    async for chunk in model.chat_stream([
        {"role": "user", "content": "Tell me a story"}
    ]):
        print(chunk, end="")

asyncio.run(main())`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="parameters" title="Parameters">
          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Parameter</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Default</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>temperature</InlineCode></td>
                  <td className="py-2 px-4">0.7</td>
                  <td className="py-2 pl-4">Sampling randomness (0.0-2.0)</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>top_p</InlineCode></td>
                  <td className="py-2 px-4">0.9</td>
                  <td className="py-2 pl-4">Nucleus sampling threshold</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>top_k</InlineCode></td>
                  <td className="py-2 px-4">50</td>
                  <td className="py-2 pl-4">Top-k sampling (0 = disabled)</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>max_tokens</InlineCode></td>
                  <td className="py-2 px-4">2048</td>
                  <td className="py-2 pl-4">Maximum tokens to generate</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>repetition_penalty</InlineCode></td>
                  <td className="py-2 px-4">1.0</td>
                  <td className="py-2 pl-4">Penalty for repeated tokens</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>stop</InlineCode></td>
                  <td className="py-2 px-4">[]</td>
                  <td className="py-2 pl-4">Stop sequences</td>
                </tr>
              </tbody>
            </table>
          </div>

          <CodeBlock
            language="python"
            code={`# Custom parameters
response = model.chat(
    messages=[{"role": "user", "content": "Write code"}],
    temperature=0.2,      # Lower for code
    top_p=0.95,
    max_tokens=1000,
    stop=["\`\`\`"]       # Stop at code block end
)`}
          />
        </DocSection>

        <DocSection id="chat-templates" title="Chat Templates">
          <p className="mb-4">
            ZSE automatically detects and applies the correct chat template for each model.
            Override if needed:
          </p>

          <CodeBlock
            language="python"
            code={`from zllm_zse import ZSE

# Use built-in template
model = ZSE("qwen-7b.zse")  # Auto-detects Qwen template

# Override template
model = ZSE("custom-model.zse", chat_template="chatml")

# Custom template string
model = ZSE("model.zse", chat_template="""
{%- for message in messages %}
{%- if message['role'] == 'user' %}
User: {{ message['content'] }}
{%- elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{%- endif %}
{%- endfor %}
Assistant:""")`}
          />

          <p className="mt-4 mb-2">
            Built-in templates:
          </p>

          <div className="flex flex-wrap gap-2 mb-4">
            {['chatml', 'llama', 'mistral', 'vicuna', 'alpaca', 'zephyr'].map(t => (
              <span key={t} className="px-2 py-0.5 bg-white/5 rounded text-sm text-white/60">
                {t}
              </span>
            ))}
          </div>
        </DocSection>

        <DocSection id="batch-inference" title="Batch Inference">
          <p className="mb-4">
            Process multiple prompts efficiently:
          </p>

          <CodeBlock
            language="python"
            code={`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# Batch completion
prompts = [
    "Translate to French: Hello",
    "Translate to French: Goodbye", 
    "Translate to French: Thank you",
]

results = model.complete_batch(prompts)
for prompt, result in zip(prompts, results):
    print(f"{prompt} -> {result}")

# Batch chat
conversations = [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is 3+3?"}],
    [{"role": "user", "content": "What is 4+4?"}],
]

responses = model.chat_batch(conversations)
for response in responses:
    print(response)`}
          />

          <Callout type="tip">
            Batch inference can be 2-4x faster than sequential inference due to GPU parallelism.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ title: 'zServe', href: '/docs/zserve' }}
          next={{ title: 'zStream', href: '/docs/zstream' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
