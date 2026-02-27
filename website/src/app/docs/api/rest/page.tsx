'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'authentication', title: 'Authentication', level: 2 },
  { id: 'chat-completions', title: 'Chat Completions', level: 2 },
  { id: 'streaming', title: 'Streaming', level: 3 },
  { id: 'models', title: 'Models', level: 2 },
  { id: 'health', title: 'Health Check', level: 2 },
  { id: 'errors', title: 'Error Handling', level: 2 },
]

export default function RESTAPIPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="REST API"
          description="OpenAI-compatible REST API reference. Use any OpenAI SDK or make direct HTTP requests."
          badge="API Reference"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            ZSE provides an OpenAI-compatible REST API. This means you can use any OpenAI SDK 
            or existing code with minimal changes.
          </p>

          <CodeBlock
            language="text"
            code={`Base URL: http://localhost:8000/v1

Endpoints:
  POST /v1/chat/completions  - Generate chat completions
  GET  /v1/models            - List available models
  GET  /health               - Health check`}
          />

          <Callout type="tip">
            If you're using the OpenAI Python SDK, just change the <InlineCode>base_url</InlineCode> parameter 
            to point to your ZSE server.
          </Callout>
        </DocSection>

        <DocSection id="authentication" title="Authentication">
          <p className="mb-4">
            By default, ZSE does not require authentication. If you start the server with 
            <InlineCode>--api-key</InlineCode>, all requests must include the key.
          </p>

          <CodeBlock
            language="bash"
            code={`# Start server with API key
zse serve model.zse --api-key sk-your-secret-key`}
          />

          <p className="mt-4 mb-2">
            Include the API key in the <InlineCode>Authorization</InlineCode> header:
          </p>

          <CodeBlock
            language="bash"
            code={`curl http://localhost:8000/v1/chat/completions \\
  -H "Authorization: Bearer sk-your-secret-key" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}]}'`}
          />
        </DocSection>

        <DocSection id="chat-completions" title="Chat Completions">
          <p className="mb-4">
            Generate chat completions with the <InlineCode>/v1/chat/completions</InlineCode> endpoint.
          </p>

          <CodeBlock
            language="bash"
            filename="POST /v1/chat/completions"
            code={`curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'`}
          />

          <p className="mt-4 mb-2">
            <strong className="text-white">Request Parameters:</strong>
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Parameter</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Type</th>
                  <th className="text-left py-2 px-4 text-white/50 font-medium">Required</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>model</InlineCode></td>
                  <td className="py-2 px-4">string</td>
                  <td className="py-2 px-4">Yes</td>
                  <td className="py-2 pl-4">Model ID (use "default" for single model)</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>messages</InlineCode></td>
                  <td className="py-2 px-4">array</td>
                  <td className="py-2 px-4">Yes</td>
                  <td className="py-2 pl-4">Array of message objects</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>temperature</InlineCode></td>
                  <td className="py-2 px-4">float</td>
                  <td className="py-2 px-4">No</td>
                  <td className="py-2 pl-4">Sampling temperature (0-2, default: 1.0)</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>max_tokens</InlineCode></td>
                  <td className="py-2 px-4">int</td>
                  <td className="py-2 px-4">No</td>
                  <td className="py-2 pl-4">Maximum tokens to generate</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>stream</InlineCode></td>
                  <td className="py-2 px-4">bool</td>
                  <td className="py-2 px-4">No</td>
                  <td className="py-2 pl-4">Enable streaming response</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4"><InlineCode>top_p</InlineCode></td>
                  <td className="py-2 px-4">float</td>
                  <td className="py-2 px-4">No</td>
                  <td className="py-2 pl-4">Nucleus sampling (0-1, default: 1.0)</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4"><InlineCode>stop</InlineCode></td>
                  <td className="py-2 px-4">array</td>
                  <td className="py-2 px-4">No</td>
                  <td className="py-2 pl-4">Stop sequences</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="mt-4 mb-2">
            <strong className="text-white">Response:</strong>
          </p>

          <CodeBlock
            language="json"
            filename="Response"
            code={`{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1709147520,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}`}
          />

          <DocSubSection id="streaming" title="Streaming">
            <p className="mb-4">
              Enable streaming to receive tokens as they're generated:
            </p>

            <CodeBlock
              language="bash"
              code={`curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'`}
            />

            <p className="mt-4 mb-2">
              Streaming response (Server-Sent Events):
            </p>

            <CodeBlock
              language="text"
              code={`data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"1"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":", "},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"2"},"index":0}]}

...

data: [DONE]`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="models" title="Models">
          <p className="mb-4">
            List available models with the <InlineCode>/v1/models</InlineCode> endpoint.
          </p>

          <CodeBlock
            language="bash"
            filename="GET /v1/models"
            code="curl http://localhost:8000/v1/models"
          />

          <CodeBlock
            language="json"
            filename="Response"
            code={`{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-7B-Instruct",
      "object": "model",
      "created": 1709147520,
      "owned_by": "zse"
    }
  ]
}`}
          />
        </DocSection>

        <DocSection id="health" title="Health Check">
          <p className="mb-4">
            Check server health with the <InlineCode>/health</InlineCode> endpoint.
          </p>

          <CodeBlock
            language="bash"
            filename="GET /health"
            code="curl http://localhost:8000/health"
          />

          <CodeBlock
            language="json"
            filename="Response"
            code={`{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "gpu_memory_used": "5.2 GB",
  "gpu_memory_total": "80 GB",
  "uptime": "2h 15m"
}`}
          />
        </DocSection>

        <DocSection id="errors" title="Error Handling">
          <p className="mb-4">
            ZSE returns standard HTTP status codes and JSON error responses:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-2 pr-4 text-white/50 font-medium">Status</th>
                  <th className="text-left py-2 pl-4 text-white/50 font-medium">Description</th>
                </tr>
              </thead>
              <tbody className="text-white/70">
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">200</td>
                  <td className="py-2 pl-4">Success</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">400</td>
                  <td className="py-2 pl-4">Bad request (invalid parameters)</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">401</td>
                  <td className="py-2 pl-4">Unauthorized (invalid API key)</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">404</td>
                  <td className="py-2 pl-4">Model not found</td>
                </tr>
                <tr className="border-b border-white/[0.05]">
                  <td className="py-2 pr-4">429</td>
                  <td className="py-2 pl-4">Rate limited</td>
                </tr>
                <tr>
                  <td className="py-2 pr-4">500</td>
                  <td className="py-2 pl-4">Internal server error</td>
                </tr>
              </tbody>
            </table>
          </div>

          <CodeBlock
            language="json"
            filename="Error Response"
            code={`{
  "error": {
    "message": "Invalid request: 'messages' is required",
    "type": "invalid_request_error",
    "code": "missing_required_field"
  }
}`}
          />
        </DocSection>

        <DocNav
          prev={{ title: 'Python API', href: '/docs/api/python' }}
          next={{ title: 'Configuration', href: '/docs/api/config' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
