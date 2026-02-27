'use client'

import { DocHeader, DocSection, DocSubSection, DocNav, TableOfContents } from '@/components/docs/toc'
import { CodeBlock, InlineCode } from '@/components/docs/code-block'
import { Callout } from '@/components/docs/callout'
import { FeatureList, Card, CardGrid } from '@/components/docs/steps'

const tocItems = [
  { id: 'overview', title: 'Overview', level: 2 },
  { id: 'how-it-works', title: 'How It Works', level: 2 },
  { id: 'document-upload', title: 'Document Upload', level: 2 },
  { id: 'searching', title: 'Searching Documents', level: 2 },
  { id: 'chat-integration', title: 'Chat Integration', level: 2 },
  { id: 'api-reference', title: 'API Reference', level: 2 },
]

export default function RAGPage() {
  return (
    <div className="flex">
      <article className="flex-1 min-w-0 py-8 px-6 lg:px-10">
        <DocHeader
          title="RAG Module"
          description="Retrieval-Augmented Generation for grounding LLM responses in your documents."
          badge="Feature"
        />

        <DocSection id="overview" title="Overview">
          <p className="mb-4">
            The RAG (Retrieval-Augmented Generation) module allows you to upload documents 
            and automatically inject relevant context into your LLM conversations. This 
            grounds the model&apos;s responses in your data, reducing hallucinations and 
            enabling domain-specific knowledge.
          </p>

          <CardGrid columns={3}>
            <Card
              title="Document Upload"
              description="PDF, TXT, MD file support"
            />
            <Card
              title="Smart Chunking"
              description="Intelligent text splitting with overlap"
            />
            <Card
              title="Semantic Search"
              description="Find relevant context automatically"
            />
          </CardGrid>

          <FeatureList features={[
            "Support for PDF, TXT, and Markdown files",
            "Smart text chunking with configurable overlap",
            "TF-IDF or sentence-transformers embeddings",
            "SQLite + NumPy vector storage",
            "Semantic search with top-k retrieval",
            "Source citations in results",
          ]} />
        </DocSection>

        <DocSection id="how-it-works" title="How It Works">
          <CodeBlock
            language="text"
            code={`RAG Pipeline:

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Document   │ ──▶ │   Chunker   │ ──▶ │  Embedder   │
│   Upload    │     │  (split)    │     │ (vectorize) │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Response  │ ◀── │     LLM     │ ◀── │   Vector    │
│   + Source  │     │  (generate) │     │   Store     │
└─────────────┘     └─────────────┘     └─────────────┘

At query time:
1. Query is embedded
2. Similar chunks retrieved from vector store
3. Context injected into LLM prompt
4. Response includes source citations`}
          />

          <Callout type="info">
            The chunker uses smart text splitting that respects paragraph boundaries 
            and includes overlap to maintain context across chunks.
          </Callout>
        </DocSection>

        <DocSection id="document-upload" title="Document Upload">
          <DocSubSection id="upload-file" title="Upload File">
            <CodeBlock
              language="bash"
              code={`# Upload a PDF document
curl -X POST http://localhost:8000/api/rag/documents/upload \\
  -F "file=@knowledge_base.pdf"

# Upload a text file
curl -X POST http://localhost:8000/api/rag/documents/upload \\
  -F "file=@documentation.txt"

# Upload markdown
curl -X POST http://localhost:8000/api/rag/documents/upload \\
  -F "file=@readme.md"`}
            />
          </DocSubSection>

          <DocSubSection id="upload-content" title="Upload Raw Content">
            <CodeBlock
              language="bash"
              code={`# Add document by content
curl -X POST http://localhost:8000/api/rag/documents \\
  -H "Content-Type: application/json" \\
  -d '{
    "content": "Your document content here...",
    "metadata": {
      "title": "Company Policies",
      "source": "HR Department"
    }
  }'`}
            />
          </DocSubSection>

          <DocSubSection id="python-upload" title="Python API">
            <CodeBlock
              language="python"
              code={`import requests

# Upload a file
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/rag/documents/upload",
        files={"file": f}
    )
    doc_id = response.json()["id"]
    print(f"Document uploaded: {doc_id}")

# Add content directly
response = requests.post(
    "http://localhost:8000/api/rag/documents",
    json={
        "content": "ZSE is an ultra memory-efficient LLM inference engine...",
        "metadata": {"title": "ZSE Overview"}
    }
)`}
            />
          </DocSubSection>

          <DocSubSection id="list-documents" title="List Documents">
            <CodeBlock
              language="bash"
              code={`# List all documents
curl http://localhost:8000/api/rag/documents

# Response:
# {
#   "documents": [
#     {"id": "abc123", "title": "Company Policies", "chunks": 15},
#     {"id": "def456", "title": "Product Manual", "chunks": 42}
#   ]
# }`}
            />
          </DocSubSection>

          <DocSubSection id="delete-document" title="Delete Document">
            <CodeBlock
              language="bash"
              code={`# Delete a document
curl -X DELETE http://localhost:8000/api/rag/documents/abc123`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="searching" title="Searching Documents">
          <DocSubSection id="semantic-search" title="Semantic Search">
            <CodeBlock
              language="bash"
              code={`# Search for relevant content
curl -X POST http://localhost:8000/api/rag/search \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "What is the return policy?",
    "top_k": 5
  }'

# Response:
# {
#   "results": [
#     {
#       "content": "Returns are accepted within 30 days...",
#       "score": 0.89,
#       "document_id": "abc123",
#       "metadata": {"title": "Company Policies", "chunk": 3}
#     },
#     ...
#   ]
# }`}
            />
          </DocSubSection>

          <DocSubSection id="get-context" title="Get Context for Chat">
            <CodeBlock
              language="bash"
              code={`# Get context formatted for chat injection
curl -X POST http://localhost:8000/api/rag/context \\
  -H "Content-Type: application/json" \\
  -d '{
    "query": "How do I configure the server?",
    "top_k": 3,
    "include_sources": true
  }'

# Response:
# {
#   "context": "Based on the documentation:\\n\\n1. Edit config.yaml...",
#   "sources": [
#     {"title": "Configuration Guide", "relevance": 0.92},
#     {"title": "Quick Start", "relevance": 0.78}
#   ]
# }`}
            />
          </DocSubSection>
        </DocSection>

        <DocSection id="chat-integration" title="Chat Integration">
          <p className="mb-4">
            The RAG module integrates seamlessly with the chat API to automatically 
            inject relevant context:
          </p>

          <DocSubSection id="rag-chat" title="RAG-Enhanced Chat">
            <CodeBlock
              language="python"
              code={`import requests

# Chat with RAG context
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen-7b",
        "messages": [
            {"role": "user", "content": "What's our refund policy?"}
        ],
        "rag": {
            "enabled": True,
            "top_k": 3
        }
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
# The response will be grounded in your uploaded documents`}
            />
          </DocSubSection>

          <DocSubSection id="playground" title="Playground Integration">
            <p className="mb-4">
              The ZSE playground at <InlineCode>/chat</InlineCode> includes RAG controls 
              in the sidebar. Upload documents and toggle RAG to see context-aware responses.
            </p>
          </DocSubSection>

          <Callout type="tip">
            For best results, upload documents that are specific to your use case. 
            The more relevant your documents, the better the RAG context.
          </Callout>
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
                  <td className="py-3 px-4 font-mono text-lime">/api/rag/documents</td>
                  <td className="py-3 px-4 text-white">POST</td>
                  <td className="py-3 px-4 text-gray-400">Add document by content</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/rag/documents/upload</td>
                  <td className="py-3 px-4 text-white">POST</td>
                  <td className="py-3 px-4 text-gray-400">Upload document file</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/rag/documents</td>
                  <td className="py-3 px-4 text-white">GET</td>
                  <td className="py-3 px-4 text-gray-400">List all documents</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/rag/documents/{'{id}'}</td>
                  <td className="py-3 px-4 text-white">DELETE</td>
                  <td className="py-3 px-4 text-gray-400">Delete document</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/rag/search</td>
                  <td className="py-3 px-4 text-white">POST</td>
                  <td className="py-3 px-4 text-gray-400">Search documents</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 font-mono text-lime">/api/rag/context</td>
                  <td className="py-3 px-4 text-white">POST</td>
                  <td className="py-3 px-4 text-gray-400">Get context for chat</td>
                </tr>
              </tbody>
            </table>
          </div>
        </DocSection>

        <DocNav
          prev={{ href: '/docs/gguf', title: 'GGUF Compatibility' }}
          next={{ href: '/docs/mcp', title: 'MCP Tools' }}
        />
      </article>
      
      <TableOfContents items={tocItems} />
    </div>
  )
}
