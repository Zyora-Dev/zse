'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'how-it-works', title: 'How It Works', level: 2 },
  { id: 'configuration', title: 'Configuration', level: 2 },
  { id: 'persistent-cache', title: 'Persistent Cache', level: 2 },
  { id: 'memory-optimization', title: 'Memory Optimization', level: 2 },
]

export default function ZKVPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="zKV"
          description="Intelligent KV cache management for optimal memory usage and prompt caching."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            <InlineCode>zKV</InlineCode> manages the key-value cache used during inference, 
            enabling prompt caching, memory optimization, and long-context conversations.
          </p>

          <CardGrid columns={3}>
            <Card
              title="Prompt Caching"
              description="Reuse computations for repeated prompts"
            />
            <Card
              title="4-bit KV"
              description="Compressed cache for long contexts"
            />
            <Card
              title="Persistence"
              description="Save/load KV state to disk"
            />
          </CardGrid>

          <FeatureList features={[
            "Automatic prompt prefix caching",
            "4-bit KV cache compression",
            "Paged attention for dynamic memory",
            "Disk-backed KV for very long contexts",
            "Multi-user cache isolation",
          ]} />
        </DocSection>

        <DocSection id="how-it-works" title="How It Works">
          <p className="mb-4">
            The KV (key-value) cache stores intermediate computations from the attention 
            mechanism. Without caching, these must be recomputed for every token generated.
          </p>

          <CodeBlock
            language="text"
            code={`Without KV Cache:
┌──────────────────────────────────────────────────────┐
│ Prompt: "The quick brown fox"                        │
│ → Compute attention for all tokens (4 forward passes)│
│ → Generate "jumps" (recompute all + new token)       │
│ → Generate "over"  (recompute all + new tokens)      │
│ → Total: O(n²) computations                          │
└──────────────────────────────────────────────────────┘

With KV Cache:
┌──────────────────────────────────────────────────────┐
│ Prompt: "The quick brown fox"                        │
│ → Compute attention, STORE in KV cache               │
│ → Generate "jumps" (reuse cache + 1 new computation) │
│ → Generate "over"  (reuse cache + 1 new computation) │
│ → Total: O(n) computations                           │
└──────────────────────────────────────────────────────┘`}
          />

          <Callout type="info">
            KV caching typically provides 10-100x speedup for generation after the initial 
            prompt processing.
          </Callout>
        </DocSection>

        <DocSection id="configuration" title="Configuration">
          <DocSubSection id="cache-size" title="Cache Size">
            <CodeBlock
              language="bash"
              code={`# Set maximum context length (determines cache size)
zse serve model.zse --max-context 8192

# Set maximum cache memory
zse serve model.zse --kv-cache-memory 4GB

# Dynamic cache sizing
zse serve model.zse --kv-cache dynamic`}
            />

            <p className="mt-4 mb-2">
              Memory requirements per context length (7B model):
            </p>

            <div className="overflow-x-auto my-4">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="text-left py-2 pr-4 text-white/50 font-medium">Context</th>
                    <th className="text-right py-2 px-4 text-white/50 font-medium">FP16 KV</th>
                    <th className="text-right py-2 pl-4 text-white/50 font-medium">4-bit KV</th>
                  </tr>
                </thead>
                <tbody className="text-white/70">
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4">4,096</td>
                    <td className="text-right py-2 px-4">1.0 GB</td>
                    <td className="text-right py-2 pl-4 text-lime">0.3 GB</td>
                  </tr>
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4">8,192</td>
                    <td className="text-right py-2 px-4">2.0 GB</td>
                    <td className="text-right py-2 pl-4 text-lime">0.5 GB</td>
                  </tr>
                  <tr className="border-b border-white/[0.05]">
                    <td className="py-2 pr-4">32,768</td>
                    <td className="text-right py-2 px-4">8.0 GB</td>
                    <td className="text-right py-2 pl-4 text-lime">2.0 GB</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4">131,072</td>
                    <td className="text-right py-2 px-4">32 GB</td>
                    <td className="text-right py-2 pl-4 text-lime">8.0 GB</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </DocSubSection>

          <DocSubSection id="prompt-caching" title="Prompt Caching">
            <p className="mb-2">
              ZSE automatically caches common prompt prefixes:
            </p>

            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# First request - full computation
response1 = model.chat([
    {"role": "system", "content": "You are a helpful assistant..."},  # Cached
    {"role": "user", "content": "Hello!"}
])  # ~500ms

# Second request - reuses system prompt cache
response2 = model.chat([
    {"role": "system", "content": "You are a helpful assistant..."},  # From cache!
    {"role": "user", "content": "How are you?"}
])  # ~50ms (10x faster)`}
            />

            <p className="mt-4 mb-2">
              Configure prompt caching:
            </p>

            <CodeBlock
              language="yaml"
              filename="zse.yaml"
              code={`kv_cache:
  prompt_cache: true
  prompt_cache_size: 1GB
  prompt_cache_ttl: 3600  # seconds
  
  # Cache specific prefixes
  prefixes:
    - name: "default_system"
      content: "You are a helpful assistant..."
      preload: true`}
            />
          </DocSubSection>

          <DocSubSection id="compression" title="KV Cache Compression">
            <p className="mb-2">
              Enable 4-bit KV cache for longer contexts:
            </p>

            <CodeBlock
              language="bash"
              code={`# Enable 4-bit KV cache
zse serve model.zse --kv-quant int4

# 8-bit KV cache (better quality)
zse serve model.zse --kv-quant int8`}
            />

            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE

# Enable quantized KV cache
model = ZSE("qwen-7b.zse", kv_quant="int4")

# Now supports 4x longer contexts!
response = model.chat(
    messages=[...],
    max_context=131072  # 128K context
)`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="persistent-cache" title="Persistent Cache">
          <p className="mb-4">
            Save and restore KV cache for long-running conversations:
          </p>

          <CodeBlock
            language="python"
            code={`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# Start conversation
session = model.create_session()
session.chat([{"role": "user", "content": "My name is Alice"}])
session.chat([{"role": "user", "content": "I live in New York"}])

# Save session state (includes KV cache)
session.save("alice_session.zse")

# Later: restore session
session = model.load_session("alice_session.zse")
response = session.chat([{"role": "user", "content": "What's my name?"}])
# → "Your name is Alice"`}
          />

          <DocSubSection id="session-api" title="Session Management API">
            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE, Session

model = ZSE("qwen-7b.zse")

# Create named session
session = model.create_session(name="user_123")

# List sessions
sessions = model.list_sessions()
print(sessions)  # ['user_123', 'user_456', ...]

# Get session info
info = session.info()
print(info)
# {
#   'name': 'user_123',
#   'context_length': 1024,
#   'cache_size_bytes': 52428800,
#   'created_at': '2024-01-15T10:30:00Z'
# }

# Clear session
session.clear()

# Delete session
model.delete_session("user_123")`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="memory-optimization" title="Memory Optimization">
          <DocSubSection id="paged-attention" title="Paged Attention">
            <p className="mb-2">
              Paged attention allocates KV cache in fixed-size blocks, reducing memory 
              fragmentation:
            </p>

            <CodeBlock
              language="bash"
              code={`# Enable paged attention
zse serve model.zse --paged-attention

# Configure block size
zse serve model.zse --paged-attention --block-size 16`}
            />

            <FeatureList features={[
              "Reduces memory fragmentation",
              "Enables dynamic batch sizes",
              "More concurrent requests with same memory",
            ]} />
          </DocSubSection>

          <DocSubSection id="memory-tiers" title="Memory Tiers">
            <p className="mb-2">
              ZSE can use CPU memory and disk as overflow for KV cache:
            </p>

            <CodeBlock
              language="yaml"
              filename="zse.yaml"
              code={`kv_cache:
  tiers:
    - type: gpu
      size: 8GB
      priority: 1
      
    - type: cpu
      size: 32GB
      priority: 2
      
    - type: disk
      path: /tmp/zse_kv_cache
      size: 100GB
      priority: 3
      
  eviction: lru  # Evict least-recently-used`}
            />

            <Callout type="warning">
              CPU and disk tiers add latency. Use GPU memory for latency-sensitive workloads.
            </Callout>
          </DocSubSection>

          <DocSubSection id="cache-stats" title="Cache Statistics">
            <CodeBlock
              language="python"
              code={`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# Get cache statistics
stats = model.kv_cache_stats()
print(stats)
# {
#   'total_size': 8589934592,      # 8 GB
#   'used_size': 2147483648,       # 2 GB
#   'prompt_cache_hits': 1542,
#   'prompt_cache_misses': 89,
#   'hit_rate': 0.945,
#   'evictions': 23
# }

# Clear cache
model.clear_kv_cache()

# Clear specific session
model.clear_kv_cache(session="user_123")`}
            />
          </DocSubSection>
        </DocSection>

        <DocNav
          prev={{ title: 'zStream', href: '/docs/zstream' }}
          next={{ title: 'CLI Reference', href: '/docs/api/cli' }}
        />
      </article>

      <TableOfContents items={tocItems} />
    </div>
  )
}
