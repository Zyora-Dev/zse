'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'server-sent-events', title: 'Server-Sent Events', level: 2 },
  { id: 'python-streaming', title: 'Python Streaming', level: 2 },
  { id: 'client-integration', title: 'Client Integration', level: 2 },
  { id: 'chunked-responses', title: 'Chunked Responses', level: 2 },
]

export default function ZStreamPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="zStream"
          description="Real-time token streaming for responsive AI applications with minimal time-to-first-token."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            <InlineCode>zStream</InlineCode> provides real-time token-by-token streaming 
            for chat and completion endpoints, enabling responsive user experiences.
          </p>

          <CardGrid columns={3}>
            <Card
              title="~50ms TTFT"
              description="Time to first token"
            />
            <Card
              title="SSE"
              description="Standard Server-Sent Events"
            />
            <Card
              title="OpenAI Compatible"
              description="Same streaming format"
            />
          </CardGrid>

          <FeatureList features={[
            "Sub-100ms time-to-first-token",
            "OpenAI-compatible SSE format",
            "Backpressure handling for slow clients",
            "Graceful stream cancellation",
            "Token usage stats in final chunk",
          ]} />
        </DocSection>

        <DocSection id="server-sent-events" title="Server-Sent Events">
          <p className="mb-4">
            Enable streaming with <InlineCode>stream: true</InlineCode> in your request:
          </p>

          <CodeBlock
            language="bash"
            code={`curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen-7b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'`}
          />

          <p className="mt-4 mb-2">
            Response format (each line prefixed with <InlineCode>data: </InlineCode>):
          </p>

          <CodeBlock
            language="json"
            code={`data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":42,"total_tokens":57}}

data: [DONE]`}
          />

          <Callout type="info">
            The final chunk includes <InlineCode>finish_reason</InlineCode> and 
            <InlineCode>usage</InlineCode> stats.
          </Callout>
        </DocSection>

        <DocSection id="python-streaming" title="Python Streaming">
          <DocSubSection id="openai-client" title="OpenAI Client">
            <CodeBlock
              language="python"
              code={`from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Streaming chat completion
stream = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)`}
            />
          </DocSubSection>

          <DocSubSection id="native-api" title="Native API">
            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# Generator-based streaming
for token in model.chat_stream([
    {"role": "user", "content": "Tell me a story"}
]):
    print(token, end="", flush=True)

# With callback
def on_token(token: str):
    print(token, end="", flush=True)

model.chat(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream_callback=on_token
)`}
            />
          </DocSubSection>

          <DocSubSection id="async-streaming" title="Async Streaming">
            <CodeBlock
              language="python"
              code={`import asyncio
from openai import AsyncOpenAI

async def stream_chat():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"
    )
    
    stream = await client.chat.completions.create(
        model="qwen-7b",
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(stream_chat())`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="client-integration" title="Client Integration">
          <DocSubSection id="javascript" title="JavaScript/TypeScript">
            <CodeBlock
              language="typescript"
              code={`// Using OpenAI SDK
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'not-needed',
});

async function streamChat() {
  const stream = await openai.chat.completions.create({
    model: 'qwen-7b',
    messages: [{ role: 'user', content: 'Tell me a story' }],
    stream: true,
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content || '';
    process.stdout.write(content);
  }
}`}
            />

            <CodeBlock
              language="typescript"
              code={`// Using fetch with EventSource
async function streamWithFetch() {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'qwen-7b',
      messages: [{ role: 'user', content: 'Tell me a story' }],
      stream: true,
    }),
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const text = decoder.decode(value);
    const lines = text.split('\\n').filter(line => line.startsWith('data: '));
    
    for (const line of lines) {
      const data = line.slice(6);
      if (data === '[DONE]') return;
      
      const chunk = JSON.parse(data);
      const content = chunk.choices[0]?.delta?.content || '';
      process.stdout.write(content);
    }
  }
}`}
            />
          </DocSubSection>

          <DocSubSection id="react" title="React Integration">
            <CodeBlock
              language="tsx"
              code={`import { useState, useCallback } from 'react';

function ChatComponent() {
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = useCallback(async (message: string) => {
    setLoading(true);
    setResponse('');

    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'qwen-7b',
        messages: [{ role: 'user', content: message }],
        stream: true,
      }),
    });

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const text = decoder.decode(value);
      // Parse SSE and update state
      const matches = text.matchAll(/data: ({.*})/g);
      for (const match of matches) {
        const chunk = JSON.parse(match[1]);
        const content = chunk.choices[0]?.delta?.content || '';
        setResponse(prev => prev + content);
      }
    }

    setLoading(false);
  }, []);

  return (
    <div>
      <div>{response}</div>
      <button onClick={() => sendMessage('Hello!')}>
        {loading ? 'Generating...' : 'Send'}
      </button>
    </div>
  );
}`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="chunked-responses" title="Chunked Responses">
          <p className="mb-4">
            Control how tokens are grouped in streaming responses:
          </p>

          <CodeBlock
            language="bash"
            code={`# Stream every token (default)
curl ... -d '{"stream": true}'

# Stream every N tokens
curl ... -d '{"stream": true, "stream_options": {"chunk_size": 5}}'

# Stream by words (space-delimited)
curl ... -d '{"stream": true, "stream_options": {"chunk_by": "word"}}'

# Stream by sentences
curl ... -d '{"stream": true, "stream_options": {"chunk_by": "sentence"}}'`}
          />

          <Callout type="tip">
            Larger chunk sizes reduce network overhead but increase perceived latency. 
            Token-by-token streaming provides the best UX for chat applications.
          </Callout>

          <p className="mt-4 mb-2">
            <strong className="text-white">Python configuration:</strong>
          </p>

          <CodeBlock
            language="python"
            code={`# Stream configuration
response = model.chat_stream(
    messages=[{"role": "user", "content": "Hello"}],
    chunk_size=1,        # Tokens per chunk
    include_usage=True,  # Include token counts
)`}
          />
        </DocSection>

        <DocNav
          prev={{ title: 'zInfer', href: '/docs/zinfer' }}
          next={{ title: 'zKV', href: '/docs/zkv' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
