(()=>{var e={};e.id=437,e.ids=[437],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},9854:(e,t,s)=>{"use strict";s.r(t),s.d(t,{GlobalError:()=>r.a,__next_app__:()=>h,originalPathname:()=>m,pages:()=>d,routeModule:()=>p,tree:()=>l});var n=s(482),a=s(9108),o=s(2563),r=s.n(o),i=s(8300),c={};for(let e in i)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(c[e]=()=>i[e]);s.d(t,c);let l=["",{children:["docs",{children:["zstream",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(s.bind(s,1666)),"/Users/redfoxhotels/zse/website/src/app/docs/zstream/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(s.bind(s,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(s.bind(s,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(s.t.bind(s,9361,23)),"next/dist/client/components/not-found-error"]}],d=["/Users/redfoxhotels/zse/website/src/app/docs/zstream/page.tsx"],m="/docs/zstream/page",h={require:s,loadChunk:()=>Promise.resolve()},p=new n.AppPageRouteModule({definition:{kind:a.x.APP_PAGE,page:"/docs/zstream/page",pathname:"/docs/zstream",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:l}})},2087:(e,t,s)=>{Promise.resolve().then(s.bind(s,7502))},7502:(e,t,s)=>{"use strict";s.r(t),s.d(t,{default:()=>l});var n=s(5344),a=s(1499),o=s(196),r=s(1812),i=s(9039);let c=[{id:"overview",title:"Overview",level:2},{id:"server-sent-events",title:"Server-Sent Events",level:2},{id:"python-streaming",title:"Python Streaming",level:2},{id:"client-integration",title:"Client Integration",level:2},{id:"chunked-responses",title:"Chunked Responses",level:2}];function l(){return(0,n.jsxs)("div",{className:"flex",children:[(0,n.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[n.jsx(a.lv,{title:"zStream",description:"Real-time token streaming for responsive AI applications with minimal time-to-first-token.",badge:"Feature"}),(0,n.jsxs)(a.Je,{id:"overview",title:"Overview",children:[(0,n.jsxs)("p",{className:"mb-4",children:[n.jsx(o.Z,{children:"zStream"})," provides real-time token-by-token streaming for chat and completion endpoints, enabling responsive user experiences."]}),(0,n.jsxs)(i.gy,{columns:3,children:[n.jsx(i.Zb,{title:"~50ms TTFT",description:"Time to first token"}),n.jsx(i.Zb,{title:"SSE",description:"Standard Server-Sent Events"}),n.jsx(i.Zb,{title:"OpenAI Compatible",description:"Same streaming format"})]}),n.jsx(i.VS,{features:["Sub-100ms time-to-first-token","OpenAI-compatible SSE format","Backpressure handling for slow clients","Graceful stream cancellation","Token usage stats in final chunk"]})]}),(0,n.jsxs)(a.Je,{id:"server-sent-events",title:"Server-Sent Events",children:[(0,n.jsxs)("p",{className:"mb-4",children:["Enable streaming with ",n.jsx(o.Z,{children:"stream: true"})," in your request:"]}),n.jsx(o.d,{language:"bash",code:`curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen-7b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'`}),(0,n.jsxs)("p",{className:"mt-4 mb-2",children:["Response format (each line prefixed with ",n.jsx(o.Z,{children:"data: "}),"):"]}),n.jsx(o.d,{language:"json",code:`data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{"content":" upon"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}

data: {"id":"chat-1","object":"chat.completion.chunk","created":1234567890,"model":"qwen-7b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":42,"total_tokens":57}}

data: [DONE]`}),(0,n.jsxs)(r.U,{type:"info",children:["The final chunk includes ",n.jsx(o.Z,{children:"finish_reason"})," and",n.jsx(o.Z,{children:"usage"})," stats."]})]}),(0,n.jsxs)(a.Je,{id:"python-streaming",title:"Python Streaming",children:[n.jsx(a.KU,{id:"openai-client",title:"OpenAI Client",children:n.jsx(o.d,{language:"python",code:`from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Streaming chat completion
stream = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)`})}),n.jsx(a.KU,{id:"native-api",title:"Native API",children:n.jsx(o.d,{language:"python",code:`from zllm_zse import ZSE

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
)`})}),n.jsx(a.KU,{id:"async-streaming",title:"Async Streaming",children:n.jsx(o.d,{language:"python",code:`import asyncio
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

asyncio.run(stream_chat())`})})]}),(0,n.jsxs)(a.Je,{id:"client-integration",title:"Client Integration",children:[(0,n.jsxs)(a.KU,{id:"javascript",title:"JavaScript/TypeScript",children:[n.jsx(o.d,{language:"typescript",code:`// Using OpenAI SDK
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
}`}),n.jsx(o.d,{language:"typescript",code:`// Using fetch with EventSource
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
}`})]}),n.jsx(a.KU,{id:"react",title:"React Integration",children:n.jsx(o.d,{language:"tsx",code:`import { useState, useCallback } from 'react';

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
}`})})]}),(0,n.jsxs)(a.Je,{id:"chunked-responses",title:"Chunked Responses",children:[n.jsx("p",{className:"mb-4",children:"Control how tokens are grouped in streaming responses:"}),n.jsx(o.d,{language:"bash",code:`# Stream every token (default)
curl ... -d '{"stream": true}'

# Stream every N tokens
curl ... -d '{"stream": true, "stream_options": {"chunk_size": 5}}'

# Stream by words (space-delimited)
curl ... -d '{"stream": true, "stream_options": {"chunk_by": "word"}}'

# Stream by sentences
curl ... -d '{"stream": true, "stream_options": {"chunk_by": "sentence"}}'`}),n.jsx(r.U,{type:"tip",children:"Larger chunk sizes reduce network overhead but increase perceived latency. Token-by-token streaming provides the best UX for chat applications."}),n.jsx("p",{className:"mt-4 mb-2",children:n.jsx("strong",{className:"text-white",children:"Python configuration:"})}),n.jsx(o.d,{language:"python",code:`# Stream configuration
response = model.chat_stream(
    messages=[{"role": "user", "content": "Hello"}],
    chunk_size=1,        # Tokens per chunk
    include_usage=True,  # Include token counts
)`})]}),n.jsx(a.KO,{prev:{title:"zInfer",href:"/docs/zinfer"},next:{title:"zKV",href:"/docs/zkv"}})]}),n.jsx(a.o5,{items:c})]})}},9039:(e,t,s)=>{"use strict";s.d(t,{Rg:()=>i,VS:()=>c,Zb:()=>l,gy:()=>d});var n=s(5344),a=s(1912),o=s(2312),r=s(1453);function i({steps:e}){return n.jsx("div",{className:"my-6 space-y-0",children:e.map((t,s)=>(0,n.jsxs)(a.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*s},className:"relative pl-8 pb-8 last:pb-0",children:[s<e.length-1&&n.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),n.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:n.jsx("span",{className:"text-xs font-bold text-lime",children:s+1})}),(0,n.jsxs)("div",{children:[n.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:t.title}),t.description&&n.jsx("p",{className:"text-sm text-white/50 mb-3",children:t.description}),t.code&&n.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:n.jsx("code",{className:"text-sm text-lime/90 font-mono",children:t.code})}),t.content&&n.jsx("div",{className:"text-sm text-white/70",children:t.content})]})]},s))})}function c({features:e}){return n.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,t)=>(0,n.jsxs)(a.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*t},className:"flex items-start gap-2",children:[n.jsx(o.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),n.jsx("span",{className:"text-sm text-white/70",children:e})]},t))})}function l({title:e,description:t,icon:s,href:o,children:i}){let c=o?"a":"div";return n.jsx(a.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:o?{y:-2}:void 0,children:(0,n.jsxs)(c,{...o?{href:o,className:"block"}:{},className:(0,r.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",o&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[s&&n.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:n.jsx(s,{className:"w-4 h-4 text-lime"})}),n.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),t&&n.jsx("p",{className:"text-sm text-white/50",children:t}),i]})})}function d({children:e,columns:t=2}){return n.jsx("div",{className:(0,r.cn)("grid gap-4 my-6",2===t&&"md:grid-cols-2",3===t&&"md:grid-cols-3"),children:e})}},1666:(e,t,s)=>{"use strict";s.r(t),s.d(t,{$$typeof:()=>o,__esModule:()=>a,default:()=>r});let n=(0,s(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/zstream/page.tsx`),{__esModule:a,$$typeof:o}=n,r=n.default}};var t=require("../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),n=t.X(0,[638,498,697,224,782,883],()=>s(9854));module.exports=n})();