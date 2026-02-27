(()=>{var e={};e.id=656,e.ids=[656],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},2013:(e,t,s)=>{"use strict";s.r(t),s.d(t,{GlobalError:()=>r.a,__next_app__:()=>h,originalPathname:()=>p,pages:()=>d,routeModule:()=>m,tree:()=>c});var n=s(482),a=s(9108),i=s(2563),r=s.n(i),o=s(8300),l={};for(let e in o)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(l[e]=()=>o[e]);s.d(t,l);let c=["",{children:["docs",{children:["rag",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(s.bind(s,8146)),"/Users/redfoxhotels/zse/website/src/app/docs/rag/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(s.bind(s,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(s.bind(s,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(s.t.bind(s,9361,23)),"next/dist/client/components/not-found-error"]}],d=["/Users/redfoxhotels/zse/website/src/app/docs/rag/page.tsx"],p="/docs/rag/page",h={require:s,loadChunk:()=>Promise.resolve()},m=new n.AppPageRouteModule({definition:{kind:a.x.APP_PAGE,page:"/docs/rag/page",pathname:"/docs/rag",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:c}})},191:(e,t,s)=>{Promise.resolve().then(s.bind(s,8956))},8956:(e,t,s)=>{"use strict";s.r(t),s.d(t,{default:()=>c});var n=s(5344),a=s(1499),i=s(196),r=s(1812),o=s(9039);let l=[{id:"overview",title:"Overview",level:2},{id:"how-it-works",title:"How It Works",level:2},{id:"document-upload",title:"Document Upload",level:2},{id:"searching",title:"Searching Documents",level:2},{id:"chat-integration",title:"Chat Integration",level:2},{id:"api-reference",title:"API Reference",level:2}];function c(){return(0,n.jsxs)("div",{className:"flex",children:[(0,n.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[n.jsx(a.lv,{title:"RAG Module",description:"Retrieval-Augmented Generation for grounding LLM responses in your documents.",badge:"Feature"}),(0,n.jsxs)(a.Je,{id:"overview",title:"Overview",children:[n.jsx("p",{className:"mb-4",children:"The RAG (Retrieval-Augmented Generation) module allows you to upload documents and automatically inject relevant context into your LLM conversations. This grounds the model's responses in your data, reducing hallucinations and enabling domain-specific knowledge."}),(0,n.jsxs)(o.gy,{columns:3,children:[n.jsx(o.Zb,{title:"Document Upload",description:"PDF, TXT, MD file support"}),n.jsx(o.Zb,{title:"Smart Chunking",description:"Intelligent text splitting with overlap"}),n.jsx(o.Zb,{title:"Semantic Search",description:"Find relevant context automatically"})]}),n.jsx(o.VS,{features:["Support for PDF, TXT, and Markdown files","Smart text chunking with configurable overlap","TF-IDF or sentence-transformers embeddings","SQLite + NumPy vector storage","Semantic search with top-k retrieval","Source citations in results"]})]}),(0,n.jsxs)(a.Je,{id:"how-it-works",title:"How It Works",children:[n.jsx(i.d,{language:"text",code:`RAG Pipeline:

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
4. Response includes source citations`}),n.jsx(r.U,{type:"info",children:"The chunker uses smart text splitting that respects paragraph boundaries and includes overlap to maintain context across chunks."})]}),(0,n.jsxs)(a.Je,{id:"document-upload",title:"Document Upload",children:[n.jsx(a.KU,{id:"upload-file",title:"Upload File",children:n.jsx(i.d,{language:"bash",code:`# Upload a PDF document
curl -X POST http://localhost:8000/api/rag/documents/upload \\
  -F "file=@knowledge_base.pdf"

# Upload a text file
curl -X POST http://localhost:8000/api/rag/documents/upload \\
  -F "file=@documentation.txt"

# Upload markdown
curl -X POST http://localhost:8000/api/rag/documents/upload \\
  -F "file=@readme.md"`})}),n.jsx(a.KU,{id:"upload-content",title:"Upload Raw Content",children:n.jsx(i.d,{language:"bash",code:`# Add document by content
curl -X POST http://localhost:8000/api/rag/documents \\
  -H "Content-Type: application/json" \\
  -d '{
    "content": "Your document content here...",
    "metadata": {
      "title": "Company Policies",
      "source": "HR Department"
    }
  }'`})}),n.jsx(a.KU,{id:"python-upload",title:"Python API",children:n.jsx(i.d,{language:"python",code:`import requests

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
)`})}),n.jsx(a.KU,{id:"list-documents",title:"List Documents",children:n.jsx(i.d,{language:"bash",code:`# List all documents
curl http://localhost:8000/api/rag/documents

# Response:
# {
#   "documents": [
#     {"id": "abc123", "title": "Company Policies", "chunks": 15},
#     {"id": "def456", "title": "Product Manual", "chunks": 42}
#   ]
# }`})}),n.jsx(a.KU,{id:"delete-document",title:"Delete Document",children:n.jsx(i.d,{language:"bash",code:`# Delete a document
curl -X DELETE http://localhost:8000/api/rag/documents/abc123`})})]}),(0,n.jsxs)(a.Je,{id:"searching",title:"Searching Documents",children:[n.jsx(a.KU,{id:"semantic-search",title:"Semantic Search",children:n.jsx(i.d,{language:"bash",code:`# Search for relevant content
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
# }`})}),n.jsx(a.KU,{id:"get-context",title:"Get Context for Chat",children:n.jsx(i.d,{language:"bash",code:`# Get context formatted for chat injection
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
# }`})})]}),(0,n.jsxs)(a.Je,{id:"chat-integration",title:"Chat Integration",children:[n.jsx("p",{className:"mb-4",children:"The RAG module integrates seamlessly with the chat API to automatically inject relevant context:"}),n.jsx(a.KU,{id:"rag-chat",title:"RAG-Enhanced Chat",children:n.jsx(i.d,{language:"python",code:`import requests

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
# The response will be grounded in your uploaded documents`})}),n.jsx(a.KU,{id:"playground",title:"Playground Integration",children:(0,n.jsxs)("p",{className:"mb-4",children:["The ZSE playground at ",n.jsx(i.Z,{children:"/chat"})," includes RAG controls in the sidebar. Upload documents and toggle RAG to see context-aware responses."]})}),n.jsx(r.U,{type:"tip",children:"For best results, upload documents that are specific to your use case. The more relevant your documents, the better the RAG context."})]}),n.jsx(a.Je,{id:"api-reference",title:"API Reference",children:n.jsx("div",{className:"overflow-x-auto",children:(0,n.jsxs)("table",{className:"w-full text-sm",children:[n.jsx("thead",{children:(0,n.jsxs)("tr",{className:"border-b border-white/10",children:[n.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Endpoint"}),n.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Method"}),n.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Description"})]})}),(0,n.jsxs)("tbody",{children:[(0,n.jsxs)("tr",{className:"border-b border-white/5",children:[n.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/rag/documents"}),n.jsx("td",{className:"py-3 px-4 text-white",children:"POST"}),n.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Add document by content"})]}),(0,n.jsxs)("tr",{className:"border-b border-white/5",children:[n.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/rag/documents/upload"}),n.jsx("td",{className:"py-3 px-4 text-white",children:"POST"}),n.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Upload document file"})]}),(0,n.jsxs)("tr",{className:"border-b border-white/5",children:[n.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/rag/documents"}),n.jsx("td",{className:"py-3 px-4 text-white",children:"GET"}),n.jsx("td",{className:"py-3 px-4 text-gray-400",children:"List all documents"})]}),(0,n.jsxs)("tr",{className:"border-b border-white/5",children:[(0,n.jsxs)("td",{className:"py-3 px-4 font-mono text-lime",children:["/api/rag/documents/","{id}"]}),n.jsx("td",{className:"py-3 px-4 text-white",children:"DELETE"}),n.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Delete document"})]}),(0,n.jsxs)("tr",{className:"border-b border-white/5",children:[n.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/rag/search"}),n.jsx("td",{className:"py-3 px-4 text-white",children:"POST"}),n.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Search documents"})]}),(0,n.jsxs)("tr",{className:"border-b border-white/5",children:[n.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/rag/context"}),n.jsx("td",{className:"py-3 px-4 text-white",children:"POST"}),n.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Get context for chat"})]})]})]})})}),n.jsx(a.KO,{prev:{href:"/docs/gguf",title:"GGUF Compatibility"},next:{href:"/docs/mcp",title:"MCP Tools"}})]}),n.jsx(a.o5,{items:l})]})}},9039:(e,t,s)=>{"use strict";s.d(t,{Rg:()=>o,VS:()=>l,Zb:()=>c,gy:()=>d});var n=s(5344),a=s(1912),i=s(2312),r=s(1453);function o({steps:e}){return n.jsx("div",{className:"my-6 space-y-0",children:e.map((t,s)=>(0,n.jsxs)(a.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*s},className:"relative pl-8 pb-8 last:pb-0",children:[s<e.length-1&&n.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),n.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:n.jsx("span",{className:"text-xs font-bold text-lime",children:s+1})}),(0,n.jsxs)("div",{children:[n.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:t.title}),t.description&&n.jsx("p",{className:"text-sm text-white/50 mb-3",children:t.description}),t.code&&n.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:n.jsx("code",{className:"text-sm text-lime/90 font-mono",children:t.code})}),t.content&&n.jsx("div",{className:"text-sm text-white/70",children:t.content})]})]},s))})}function l({features:e}){return n.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,t)=>(0,n.jsxs)(a.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*t},className:"flex items-start gap-2",children:[n.jsx(i.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),n.jsx("span",{className:"text-sm text-white/70",children:e})]},t))})}function c({title:e,description:t,icon:s,href:i,children:o}){let l=i?"a":"div";return n.jsx(a.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:i?{y:-2}:void 0,children:(0,n.jsxs)(l,{...i?{href:i,className:"block"}:{},className:(0,r.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",i&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[s&&n.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:n.jsx(s,{className:"w-4 h-4 text-lime"})}),n.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),t&&n.jsx("p",{className:"text-sm text-white/50",children:t}),o]})})}function d({children:e,columns:t=2}){return n.jsx("div",{className:(0,r.cn)("grid gap-4 my-6",2===t&&"md:grid-cols-2",3===t&&"md:grid-cols-3"),children:e})}},8146:(e,t,s)=>{"use strict";s.r(t),s.d(t,{$$typeof:()=>i,__esModule:()=>a,default:()=>r});let n=(0,s(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/rag/page.tsx`),{__esModule:a,$$typeof:i}=n,r=n.default}};var t=require("../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),n=t.X(0,[638,498,697,224,782,883],()=>s(2013));module.exports=n})();