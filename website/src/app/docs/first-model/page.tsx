'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { Steps, FeatureList } from '@/components/docs/steps'

const tocItems = [
  { id: 'choosing-model', title: 'Choosing a Model', level: 2 },
  { id: 'download-convert', title: 'Download & Convert', level: 2 },
  { id: 'run-model', title: 'Running Your Model', level: 2 },
  { id: 'verify', title: 'Verify Everything Works', level: 2 },
  { id: 'next-steps', title: 'Next Steps', level: 2 },
]

export default function FirstModelPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="Your First Model"
          description="Download, convert, and run your first LLM with ZSE in under 5 minutes."
          badge="Getting Started"
        />

        <DocSection id="choosing-model" title="Choosing a Model">
          <p className="mb-4">
            Choose a model based on your GPU memory:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">GPU VRAM</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Recommended Model</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">HuggingFace ID</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">6-8 GB</td>
                  <td className="py-2 px-4">Qwen 2.5 3B</td>
                  <td className="py-2 pl-4"><InlineCode>Qwen/Qwen2.5-3B-Instruct</InlineCode></td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">8-12 GB</td>
                  <td className="py-2 px-4 text-lime">Qwen 2.5 7B ✓</td>
                  <td className="py-2 pl-4"><InlineCode>Qwen/Qwen2.5-7B-Instruct</InlineCode></td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">12-16 GB</td>
                  <td className="py-2 px-4">Llama 3.1 8B</td>
                  <td className="py-2 pl-4"><InlineCode>meta-llama/Llama-3.1-8B-Instruct</InlineCode></td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">24+ GB</td>
                  <td className="py-2 px-4">Qwen 2.5 14B</td>
                  <td className="py-2 pl-4"><InlineCode>Qwen/Qwen2.5-14B-Instruct</InlineCode></td>
                </tr>
              </tbody>
            </table>
          </div>

          <Callout type="tip">
            Not sure? Check your GPU with <InlineCode>zse hardware</InlineCode>
          </Callout>
        </DocSection>

        <DocSection id="download-convert" title="Download & Convert">
          <Steps steps={[
            {
              title: "Check your hardware",
              description: "Verify GPU is detected",
              code: "zse hardware"
            },
            {
              title: "Convert the model",
              description: "Download from HuggingFace and convert to .zse format",
              code: "zse convert Qwen/Qwen2.5-7B-Instruct -o qwen-7b.zse"
            },
            {
              title: "Verify the conversion",
              description: "Check model info",
              code: "zse info qwen-7b.zse"
            },
          ]} />

          <Callout type="info">
            First conversion downloads ~7GB and takes ~2 minutes on GPU (longer on CPU). 
            Files are cached for future conversions.
          </Callout>
        </DocSection>

        <DocSection id="run-model" title="Running Your Model">
          <DocSubSection id="interactive-chat" title="Interactive Chat">
            <CodeBlock
              language="bash"
              code={`zse chat qwen-7b.zse`}
            />
          </DocSubSection>

          <DocSubSection id="api-server" title="API Server">
            <CodeBlock
              language="bash"
              code={`# Start the server
zse serve qwen-7b.zse

# Test with curl
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "Hello!"}]}'`}
            />
          </DocSubSection>

          <DocSubSection id="python-code" title="Python Code">
            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")
response = model.chat([
    {"role": "user", "content": "Hello!"}
])
print(response)`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="verify" title="Verify Everything Works">
          <p className="mb-4">
            Run a quick benchmark to verify your setup:
          </p>

          <CodeBlock
            language="bash"
            code={`zse benchmark qwen-7b.zse`}
          />

          <p className="mt-4 mb-2">
            Expected results on consumer GPUs:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">GPU</th>
                  <th className="text-right py-2 px-4 text-white/50 font-medium">Cold Start</th>
                  <th className="text-right py-2 pl-4 text-white/50 font-medium">Throughput</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">RTX 3060 12GB</td>
                  <td className="text-right py-2 px-4">4.2s</td>
                  <td className="text-right py-2 pl-4">~45 tok/s</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">RTX 4070 12GB</td>
                  <td className="text-right py-2 px-4">3.9s</td>
                  <td className="text-right py-2 pl-4">~80 tok/s</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">RTX 4090 24GB</td>
                  <td className="text-right py-2 px-4">3.5s</td>
                  <td className="text-right py-2 pl-4">~120 tok/s</td>
                </tr>
              </tbody>
            </table>
          </div>
        </DocSection>

        <DocSection id="next-steps" title="Next Steps">
          <FeatureList features={[
            "Learn about model formats and quantization",
            "Set up a production API server with zServe",
            "Explore streaming responses with zStream",
            "Optimize memory with KV cache compression",
          ]} />

          <Callout type="success">
            You are ready to use ZSE! Check out the features documentation to learn more.
          </Callout>
        </DocSection>

        <DocNav
          prev={{ title: 'Installation', href: '/docs/installation' }}
          next={{ title: 'Architecture', href: '/docs/architecture' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
