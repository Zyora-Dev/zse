'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'built-in-tools', title: 'Built-in Tools', level: 2 },
  { id: 'using-tools', title: 'Using Tools', level: 2 },
  { id: 'custom-tools', title: 'Custom Tools', level: 2 },
  { id: 'openai-format', title: 'OpenAI Format', level: 2 },
  { id: 'api-reference', title: 'API Reference', level: 2 },
]

export default function MCPPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="MCP Tools"
          description="Model Context Protocol for extending LLM capabilities with function calling and tool execution."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            MCP (Model Context Protocol) enables LLMs to use external tools and functions. 
            ZSE includes built-in tools and supports custom tool definitions compatible 
            with OpenAI&apos;s function calling format.
          </p>

          <CardGrid columns={3}>
            <Card
              title="Function Calling"
              description="Parse and execute tool calls from LLM output"
            />
            <Card
              title="Built-in Tools"
              description="Calculator, datetime, JSON parser, string ops"
            />
            <Card
              title="OpenAI Compatible"
              description="Same format as OpenAI function calling"
            />
          </CardGrid>

          <FeatureList features={[
            "JSON schema tool definitions",
            "Automatic tool call parsing from LLM output",
            "Built-in utility tools",
            "Custom tool registration",
            "OpenAI-compatible function calling format",
          ]} />
        </DocSection>

        <DocSection id="built-in-tools" title="Built-in Tools">
          <p className="mb-4">
            ZSE includes several utility tools out of the box:
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Tool</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Description</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Example Input</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">calculator</td>
                  <td className="py-3 px-4 text-gray-400">Math expressions (sqrt, sin, cos, log, etc.)</td>
                  <td className="py-3 px-4 font-mono text-white text-xs">sqrt(16) + 2**3</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">datetime</td>
                  <td className="py-3 px-4 text-gray-400">Current date/time with timezone support</td>
                  <td className="py-3 px-4 font-mono text-white text-xs">America/New_York</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">parse_json</td>
                  <td className="py-3 px-4 text-gray-400">Parse JSON and extract data</td>
                  <td className="py-3 px-4 font-mono text-white text-xs">{'{\"key\": \"value\"}'}</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">string_ops</td>
                  <td className="py-3 px-4 text-gray-400">String operations (upper, lower, split, etc.)</td>
                  <td className="py-3 px-4 font-mono text-white text-xs">upper: hello world</td>
                </tr>
              </tbody>
            </table>
          </div>

          <DocSubSection id="calculator-tool" title="Calculator">
            <CodeBlock
              language="bash"
              code={`# Execute calculator tool
curl -X POST http://localhost:8000/api/tools/execute \\
  -H "Content-Type: application/json" \\
  -d '{
    "tool": "calculator",
    "input": "sqrt(144) + sin(3.14159/2)"
  }'

# Response:
# {"result": 13.0, "success": true}`}
            />

            <p className="mt-4 mb-2 text-sm text-gray-400">
              Supported functions: sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil
            </p>
          </DocSubSection>

          <DocSubSection id="datetime-tool" title="Datetime">
            <CodeBlock
              language="bash"
              code={`# Get current time
curl -X POST http://localhost:8000/api/tools/execute \\
  -H "Content-Type: application/json" \\
  -d '{
    "tool": "datetime",
    "input": "UTC"
  }'

# Response:
# {
#   "result": {
#     "datetime": "2026-02-26T10:30:00+00:00",
#     "timezone": "UTC",
#     "unix_timestamp": 1772092200
#   },
#   "success": true
# }`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="using-tools" title="Using Tools">
          <DocSubSection id="list-tools" title="List Available Tools">
            <CodeBlock
              language="bash"
              code={`# List all registered tools
curl http://localhost:8000/api/tools/

# Response:
# {
#   "tools": [
#     {
#       "name": "calculator",
#       "description": "Evaluate mathematical expressions",
#       "parameters": {...}
#     },
#     ...
#   ]
# }`}
            />
          </DocSubSection>

          <DocSubSection id="execute-tool" title="Execute Tool">
            <CodeBlock
              language="python"
              code={`import requests

# Execute a tool directly
response = requests.post(
    "http://localhost:8000/api/tools/execute",
    json={
        "tool": "calculator",
        "input": "2 ** 10"
    }
)
print(response.json())
# {"result": 1024, "success": true}`}
            />
          </DocSubSection>

          <DocSubSection id="parse-calls" title="Parse Tool Calls from Text">
            <CodeBlock
              language="bash"
              code={`# Parse tool calls from LLM output
curl -X POST http://localhost:8000/api/tools/parse \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "I need to calculate sqrt(256). Let me use the calculator tool."
  }'

# Response:
# {
#   "tool_calls": [
#     {"tool": "calculator", "input": "sqrt(256)"}
#   ]
# }`}
            />
          </DocSubSection>

          <DocSubSection id="process-calls" title="Parse and Execute">
            <CodeBlock
              language="bash"
              code={`# Parse and execute tool calls in one step
curl -X POST http://localhost:8000/api/tools/process \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "What time is it in Tokyo? Use datetime tool with Asia/Tokyo timezone."
  }'

# Response:
# {
#   "results": [
#     {
#       "tool": "datetime",
#       "input": "Asia/Tokyo",
#       "result": {"datetime": "2026-02-26T19:30:00+09:00", ...},
#       "success": true
#     }
#   ]
# }`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="custom-tools" title="Custom Tools">
          <p className="mb-4">
            Register custom tools with JSON schema definitions:
          </p>

          <CodeBlock
            language="python"
            code={`from zse.api.server.mcp import MCPRegistry

# Get the registry
registry = MCPRegistry()

# Define a custom tool
@registry.tool(
    name="weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or coordinates"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["location"]
    }
)
def get_weather(location: str, units: str = "celsius"):
    # Your implementation here
    return {"temperature": 22, "condition": "sunny", "units": units}

# Now the tool is available via API
# POST /api/tools/execute {"tool": "weather", "input": {"location": "Tokyo"}}`}
          />

          <Callout type="info">
            Custom tools are persisted in the server session. For permanent tools, 
            add them to your server startup script.
          </Callout>
        </DocSection>

        <DocSection id="openai-format" title="OpenAI Format">
          <p className="mb-4">
            Get tools in OpenAI-compatible function calling format:
          </p>

          <DocSubSection id="get-functions" title="Get Functions">
            <CodeBlock
              language="bash"
              code={`# Get tools in OpenAI format
curl http://localhost:8000/api/tools/openai/functions

# Response:
# {
#   "functions": [
#     {
#       "name": "calculator",
#       "description": "Evaluate mathematical expressions",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "expression": {
#             "type": "string",
#             "description": "Math expression to evaluate"
#           }
#         },
#         "required": ["expression"]
#       }
#     }
#   ]
# }`}
            />
          </DocSubSection>

          <DocSubSection id="chat-with-tools" title="Chat with Tools">
            <CodeBlock
              language="python"
              code={`import requests

# Get available tools
tools_response = requests.get(
    "http://localhost:8000/api/tools/openai/functions"
)
tools = tools_response.json()["functions"]

# Chat with tool support
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen-7b",
        "messages": [
            {"role": "user", "content": "What is 15% of 280?"}
        ],
        "tools": tools,
        "tool_choice": "auto"
    }
)

result = response.json()
# The model may return a tool call which you can execute
if result["choices"][0].get("tool_calls"):
    tool_call = result["choices"][0]["tool_calls"][0]
    # Execute the tool call...`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="api-reference" title="API Reference">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Endpoint</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Method</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Description</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/tools/</td>
                  <td className="py-3 px-4 text-white">GET</td>
                  <td className="py-3 px-4 text-gray-400">List all available tools</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/tools/execute</td>
                  <td className="py-3 px-4 text-white">POST</td>
                  <td className="py-3 px-4 text-gray-400">Execute a specific tool</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/tools/parse</td>
                  <td className="py-3 px-4 text-white">POST</td>
                  <td className="py-3 px-4 text-gray-400">Parse tool calls from text</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/tools/process</td>
                  <td className="py-3 px-4 text-white">POST</td>
                  <td className="py-3 px-4 text-gray-400">Parse and execute tool calls</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/tools/openai/functions</td>
                  <td className="py-3 px-4 text-white">GET</td>
                  <td className="py-3 px-4 text-gray-400">Get OpenAI-compatible format</td>
                </tr>
              </tbody>
            </table>
          </div>
        </DocSection>

        <DocNav
          prev={{ href: '/docs/rag', title: 'RAG Module' }}
          next={{ href: '/docs/api/cli', title: 'CLI Commands' }}
        />
      </article>
      
      <TableOfContents items={tocItems} />
    </div>
  )
}
