(()=>{var e={};e.id=362,e.ids=[362],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},2853:(e,t,s)=>{"use strict";s.r(t),s.d(t,{GlobalError:()=>l.a,__next_app__:()=>p,originalPathname:()=>m,pages:()=>d,routeModule:()=>h,tree:()=>o});var r=s(482),a=s(9108),i=s(2563),l=s.n(i),n=s(8300),c={};for(let e in n)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(c[e]=()=>n[e]);s.d(t,c);let o=["",{children:["docs",{children:["zinfer",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(s.bind(s,3165)),"/Users/redfoxhotels/zse/website/src/app/docs/zinfer/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(s.bind(s,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(s.bind(s,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(s.t.bind(s,9361,23)),"next/dist/client/components/not-found-error"]}],d=["/Users/redfoxhotels/zse/website/src/app/docs/zinfer/page.tsx"],m="/docs/zinfer/page",p={require:s,loadChunk:()=>Promise.resolve()},h=new r.AppPageRouteModule({definition:{kind:a.x.APP_PAGE,page:"/docs/zinfer/page",pathname:"/docs/zinfer",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:o}})},3561:(e,t,s)=>{Promise.resolve().then(s.bind(s,8488))},8488:(e,t,s)=>{"use strict";s.r(t),s.d(t,{default:()=>o});var r=s(5344),a=s(1499),i=s(196),l=s(1812),n=s(9039);let c=[{id:"overview",title:"Overview",level:2},{id:"cli-usage",title:"CLI Usage",level:2},{id:"python-api",title:"Python API",level:2},{id:"parameters",title:"Parameters",level:2},{id:"chat-templates",title:"Chat Templates",level:2},{id:"batch-inference",title:"Batch Inference",level:2}];function o(){return(0,r.jsxs)("div",{className:"flex",children:[(0,r.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[r.jsx(a.lv,{title:"zInfer",description:"High-performance local inference for transformer models with optimized sampling.",badge:"Feature"}),(0,r.jsxs)(a.Je,{id:"overview",title:"Overview",children:[(0,r.jsxs)("p",{className:"mb-4",children:[r.jsx(i.Z,{children:"zInfer"})," provides direct inference capabilities for both interactive chat and programmatic text generation."]}),(0,r.jsxs)(n.gy,{columns:3,children:[r.jsx(n.Zb,{title:"~100 tok/s",description:"High throughput on consumer GPUs"}),r.jsx(n.Zb,{title:"Flash Attention",description:"Memory-efficient attention"}),r.jsx(n.Zb,{title:"Speculative",description:"2-3x faster with draft models"})]}),r.jsx(n.VS,{features:["Optimized CUDA kernels for inference","Flash Attention 2 support","Speculative decoding with draft models","Continuous batching for throughput","Custom sampling strategies"]})]}),(0,r.jsxs)(a.Je,{id:"cli-usage",title:"CLI Usage",children:[(0,r.jsxs)(a.KU,{id:"chat",title:"Interactive Chat",children:[r.jsx(i.d,{language:"bash",code:`# Start interactive chat
zse chat qwen-7b.zse

# With system prompt
zse chat qwen-7b.zse --system "You are a helpful coding assistant"

# With initial prompt
zse chat qwen-7b.zse -p "Explain quantum computing"`}),r.jsx("p",{className:"mt-4 mb-2",children:"Chat commands:"}),r.jsx("div",{className:"overflow-x-auto my-4",children:(0,r.jsxs)("table",{className:"w-full text-sm",children:[r.jsx("thead",{children:(0,r.jsxs)("tr",{className:"border-b border-white/10",children:[r.jsx("th",{className:"text-left py-2 pr-4 text-white/50 font-medium",children:"Command"}),r.jsx("th",{className:"text-left py-2 pl-4 text-white/50 font-medium",children:"Description"})]})}),(0,r.jsxs)("tbody",{className:"text-white/70",children:[(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"/clear"})}),r.jsx("td",{className:"py-2 pl-4",children:"Clear conversation history"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"/system <prompt>"})}),r.jsx("td",{className:"py-2 pl-4",children:"Set system prompt"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"/temp <value>"})}),r.jsx("td",{className:"py-2 pl-4",children:"Set temperature (0.0-2.0)"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"/save <file>"})}),r.jsx("td",{className:"py-2 pl-4",children:"Save conversation to file"})]}),(0,r.jsxs)("tr",{children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"/quit"})}),r.jsx("td",{className:"py-2 pl-4",children:"Exit chat"})]})]})]})})]}),r.jsx(a.KU,{id:"completion",title:"Text Completion",children:r.jsx(i.d,{language:"bash",code:`# Single completion
zse complete qwen-7b.zse -p "The quick brown fox"

# With parameters
zse complete qwen-7b.zse \\
  -p "Write a poem about AI" \\
  --max-tokens 200 \\
  --temperature 0.8`})})]}),(0,r.jsxs)(a.Je,{id:"python-api",title:"Python API",children:[r.jsx(a.KU,{id:"quick-inference",title:"Quick Inference",children:r.jsx(i.d,{language:"python",code:`from zllm_zse import ZSE

# Load model
model = ZSE("qwen-7b.zse")

# Chat completion
response = model.chat([
    {"role": "user", "content": "Hello!"}
])
print(response)

# Text completion
text = model.complete("The meaning of life is")
print(text)`})}),r.jsx(a.KU,{id:"streaming",title:"Streaming",children:r.jsx(i.d,{language:"python",code:`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# Stream chat response
for chunk in model.chat_stream([
    {"role": "user", "content": "Tell me a story"}
]):
    print(chunk, end="", flush=True)

# Stream completion
for token in model.complete_stream("Once upon a time"):
    print(token, end="", flush=True)`})}),r.jsx(a.KU,{id:"async",title:"Async API",children:r.jsx(i.d,{language:"python",code:`import asyncio
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

asyncio.run(main())`})})]}),(0,r.jsxs)(a.Je,{id:"parameters",title:"Parameters",children:[r.jsx("div",{className:"overflow-x-auto my-4",children:(0,r.jsxs)("table",{className:"w-full text-sm",children:[r.jsx("thead",{children:(0,r.jsxs)("tr",{className:"border-b border-white/10",children:[r.jsx("th",{className:"text-left py-2 pr-4 text-white/50 font-medium",children:"Parameter"}),r.jsx("th",{className:"text-left py-2 px-4 text-white/50 font-medium",children:"Default"}),r.jsx("th",{className:"text-left py-2 pl-4 text-white/50 font-medium",children:"Description"})]})}),(0,r.jsxs)("tbody",{className:"text-white/70",children:[(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"temperature"})}),r.jsx("td",{className:"py-2 px-4",children:"0.7"}),r.jsx("td",{className:"py-2 pl-4",children:"Sampling randomness (0.0-2.0)"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"top_p"})}),r.jsx("td",{className:"py-2 px-4",children:"0.9"}),r.jsx("td",{className:"py-2 pl-4",children:"Nucleus sampling threshold"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"top_k"})}),r.jsx("td",{className:"py-2 px-4",children:"50"}),r.jsx("td",{className:"py-2 pl-4",children:"Top-k sampling (0 = disabled)"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"max_tokens"})}),r.jsx("td",{className:"py-2 px-4",children:"2048"}),r.jsx("td",{className:"py-2 pl-4",children:"Maximum tokens to generate"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"repetition_penalty"})}),r.jsx("td",{className:"py-2 px-4",children:"1.0"}),r.jsx("td",{className:"py-2 pl-4",children:"Penalty for repeated tokens"})]}),(0,r.jsxs)("tr",{children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(i.Z,{children:"stop"})}),r.jsx("td",{className:"py-2 px-4",children:"[]"}),r.jsx("td",{className:"py-2 pl-4",children:"Stop sequences"})]})]})]})}),r.jsx(i.d,{language:"python",code:`# Custom parameters
response = model.chat(
    messages=[{"role": "user", "content": "Write code"}],
    temperature=0.2,      # Lower for code
    top_p=0.95,
    max_tokens=1000,
    stop=["\`\`\`"]       # Stop at code block end
)`})]}),(0,r.jsxs)(a.Je,{id:"chat-templates",title:"Chat Templates",children:[r.jsx("p",{className:"mb-4",children:"ZSE automatically detects and applies the correct chat template for each model. Override if needed:"}),r.jsx(i.d,{language:"python",code:`from zllm_zse import ZSE

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
Assistant:""")`}),r.jsx("p",{className:"mt-4 mb-2",children:"Built-in templates:"}),r.jsx("div",{className:"flex flex-wrap gap-2 mb-4",children:["chatml","llama","mistral","vicuna","alpaca","zephyr"].map(e=>r.jsx("span",{className:"px-2 py-0.5 bg-white/5 rounded text-sm text-white/60",children:e},e))})]}),(0,r.jsxs)(a.Je,{id:"batch-inference",title:"Batch Inference",children:[r.jsx("p",{className:"mb-4",children:"Process multiple prompts efficiently:"}),r.jsx(i.d,{language:"python",code:`from zllm_zse import ZSE

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
    print(response)`}),r.jsx(l.U,{type:"tip",children:"Batch inference can be 2-4x faster than sequential inference due to GPU parallelism."})]}),r.jsx(a.KO,{prev:{title:"zServe",href:"/docs/zserve"},next:{title:"zStream",href:"/docs/zstream"}})]}),r.jsx(a.o5,{items:c})]})}},9039:(e,t,s)=>{"use strict";s.d(t,{Rg:()=>n,VS:()=>c,Zb:()=>o,gy:()=>d});var r=s(5344),a=s(1912),i=s(2312),l=s(1453);function n({steps:e}){return r.jsx("div",{className:"my-6 space-y-0",children:e.map((t,s)=>(0,r.jsxs)(a.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*s},className:"relative pl-8 pb-8 last:pb-0",children:[s<e.length-1&&r.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),r.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:r.jsx("span",{className:"text-xs font-bold text-lime",children:s+1})}),(0,r.jsxs)("div",{children:[r.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:t.title}),t.description&&r.jsx("p",{className:"text-sm text-white/50 mb-3",children:t.description}),t.code&&r.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:r.jsx("code",{className:"text-sm text-lime/90 font-mono",children:t.code})}),t.content&&r.jsx("div",{className:"text-sm text-white/70",children:t.content})]})]},s))})}function c({features:e}){return r.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,t)=>(0,r.jsxs)(a.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*t},className:"flex items-start gap-2",children:[r.jsx(i.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),r.jsx("span",{className:"text-sm text-white/70",children:e})]},t))})}function o({title:e,description:t,icon:s,href:i,children:n}){let c=i?"a":"div";return r.jsx(a.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:i?{y:-2}:void 0,children:(0,r.jsxs)(c,{...i?{href:i,className:"block"}:{},className:(0,l.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",i&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[s&&r.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:r.jsx(s,{className:"w-4 h-4 text-lime"})}),r.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),t&&r.jsx("p",{className:"text-sm text-white/50",children:t}),n]})})}function d({children:e,columns:t=2}){return r.jsx("div",{className:(0,l.cn)("grid gap-4 my-6",2===t&&"md:grid-cols-2",3===t&&"md:grid-cols-3"),children:e})}},3165:(e,t,s)=>{"use strict";s.r(t),s.d(t,{$$typeof:()=>i,__esModule:()=>a,default:()=>l});let r=(0,s(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/zinfer/page.tsx`),{__esModule:a,$$typeof:i}=r,l=r.default}};var t=require("../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),r=t.X(0,[638,498,697,224,782,883],()=>s(2853));module.exports=r})();