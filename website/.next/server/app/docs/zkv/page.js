(()=>{var e={};e.id=399,e.ids=[399],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},9927:(e,s,t)=>{"use strict";t.r(s),t.d(s,{GlobalError:()=>n.a,__next_app__:()=>p,originalPathname:()=>m,pages:()=>d,routeModule:()=>h,tree:()=>l});var i=t(482),r=t(9108),a=t(2563),n=t.n(a),o=t(8300),c={};for(let e in o)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(c[e]=()=>o[e]);t.d(s,c);let l=["",{children:["docs",{children:["zkv",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(t.bind(t,1386)),"/Users/redfoxhotels/zse/website/src/app/docs/zkv/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(t.bind(t,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(t.bind(t,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(t.t.bind(t,9361,23)),"next/dist/client/components/not-found-error"]}],d=["/Users/redfoxhotels/zse/website/src/app/docs/zkv/page.tsx"],m="/docs/zkv/page",p={require:t,loadChunk:()=>Promise.resolve()},h=new i.AppPageRouteModule({definition:{kind:r.x.APP_PAGE,page:"/docs/zkv/page",pathname:"/docs/zkv",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:l}})},5358:(e,s,t)=>{Promise.resolve().then(t.bind(t,4450))},4450:(e,s,t)=>{"use strict";t.r(s),t.d(s,{default:()=>l});var i=t(5344),r=t(1499),a=t(196),n=t(1812),o=t(9039);let c=[{id:"overview",title:"Overview",level:2},{id:"how-it-works",title:"How It Works",level:2},{id:"configuration",title:"Configuration",level:2},{id:"persistent-cache",title:"Persistent Cache",level:2},{id:"memory-optimization",title:"Memory Optimization",level:2}];function l(){return(0,i.jsxs)("div",{className:"flex",children:[(0,i.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[i.jsx(r.lv,{title:"zKV",description:"Intelligent KV cache management for optimal memory usage and prompt caching.",badge:"Feature"}),(0,i.jsxs)(r.Je,{id:"overview",title:"Overview",children:[(0,i.jsxs)("p",{className:"mb-4",children:[i.jsx(a.Z,{children:"zKV"})," manages the key-value cache used during inference, enabling prompt caching, memory optimization, and long-context conversations."]}),(0,i.jsxs)(o.gy,{columns:3,children:[i.jsx(o.Zb,{title:"Prompt Caching",description:"Reuse computations for repeated prompts"}),i.jsx(o.Zb,{title:"4-bit KV",description:"Compressed cache for long contexts"}),i.jsx(o.Zb,{title:"Persistence",description:"Save/load KV state to disk"})]}),i.jsx(o.VS,{features:["Automatic prompt prefix caching","4-bit KV cache compression","Paged attention for dynamic memory","Disk-backed KV for very long contexts","Multi-user cache isolation"]})]}),(0,i.jsxs)(r.Je,{id:"how-it-works",title:"How It Works",children:[i.jsx("p",{className:"mb-4",children:"The KV (key-value) cache stores intermediate computations from the attention mechanism. Without caching, these must be recomputed for every token generated."}),i.jsx(a.d,{language:"text",code:`Without KV Cache:
┌──────────────────────────────────────────────────────┐
│ Prompt: "The quick brown fox"                        │
│ → Compute attention for all tokens (4 forward passes)│
│ → Generate "jumps" (recompute all + new token)       │
│ → Generate "over"  (recompute all + new tokens)      │
│ → Total: O(n\xb2) computations                          │
└──────────────────────────────────────────────────────┘

With KV Cache:
┌──────────────────────────────────────────────────────┐
│ Prompt: "The quick brown fox"                        │
│ → Compute attention, STORE in KV cache               │
│ → Generate "jumps" (reuse cache + 1 new computation) │
│ → Generate "over"  (reuse cache + 1 new computation) │
│ → Total: O(n) computations                           │
└──────────────────────────────────────────────────────┘`}),i.jsx(n.U,{type:"info",children:"KV caching typically provides 10-100x speedup for generation after the initial prompt processing."})]}),(0,i.jsxs)(r.Je,{id:"configuration",title:"Configuration",children:[(0,i.jsxs)(r.KU,{id:"cache-size",title:"Cache Size",children:[i.jsx(a.d,{language:"bash",code:`# Set maximum context length (determines cache size)
zse serve model.zse --max-context 8192

# Set maximum cache memory
zse serve model.zse --kv-cache-memory 4GB

# Dynamic cache sizing
zse serve model.zse --kv-cache dynamic`}),i.jsx("p",{className:"mt-4 mb-2",children:"Memory requirements per context length (7B model):"}),i.jsx("div",{className:"overflow-x-auto my-4",children:(0,i.jsxs)("table",{className:"w-full text-sm",children:[i.jsx("thead",{children:(0,i.jsxs)("tr",{className:"border-b border-white/10",children:[i.jsx("th",{className:"text-left py-2 pr-4 text-white/50 font-medium",children:"Context"}),i.jsx("th",{className:"text-right py-2 px-4 text-white/50 font-medium",children:"FP16 KV"}),i.jsx("th",{className:"text-right py-2 pl-4 text-white/50 font-medium",children:"4-bit KV"})]})}),(0,i.jsxs)("tbody",{className:"text-white/70",children:[(0,i.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[i.jsx("td",{className:"py-2 pr-4",children:"4,096"}),i.jsx("td",{className:"text-right py-2 px-4",children:"1.0 GB"}),i.jsx("td",{className:"text-right py-2 pl-4 text-lime",children:"0.3 GB"})]}),(0,i.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[i.jsx("td",{className:"py-2 pr-4",children:"8,192"}),i.jsx("td",{className:"text-right py-2 px-4",children:"2.0 GB"}),i.jsx("td",{className:"text-right py-2 pl-4 text-lime",children:"0.5 GB"})]}),(0,i.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[i.jsx("td",{className:"py-2 pr-4",children:"32,768"}),i.jsx("td",{className:"text-right py-2 px-4",children:"8.0 GB"}),i.jsx("td",{className:"text-right py-2 pl-4 text-lime",children:"2.0 GB"})]}),(0,i.jsxs)("tr",{children:[i.jsx("td",{className:"py-2 pr-4",children:"131,072"}),i.jsx("td",{className:"text-right py-2 px-4",children:"32 GB"}),i.jsx("td",{className:"text-right py-2 pl-4 text-lime",children:"8.0 GB"})]})]})]})})]}),(0,i.jsxs)(r.KU,{id:"prompt-caching",title:"Prompt Caching",children:[i.jsx("p",{className:"mb-2",children:"ZSE automatically caches common prompt prefixes:"}),i.jsx(a.d,{language:"python",code:`from zllm_zse import ZSE

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
])  # ~50ms (10x faster)`}),i.jsx("p",{className:"mt-4 mb-2",children:"Configure prompt caching:"}),i.jsx(a.d,{language:"yaml",filename:"zse.yaml",code:`kv_cache:
  prompt_cache: true
  prompt_cache_size: 1GB
  prompt_cache_ttl: 3600  # seconds
  
  # Cache specific prefixes
  prefixes:
    - name: "default_system"
      content: "You are a helpful assistant..."
      preload: true`})]}),(0,i.jsxs)(r.KU,{id:"compression",title:"KV Cache Compression",children:[i.jsx("p",{className:"mb-2",children:"Enable 4-bit KV cache for longer contexts:"}),i.jsx(a.d,{language:"bash",code:`# Enable 4-bit KV cache
zse serve model.zse --kv-quant int4

# 8-bit KV cache (better quality)
zse serve model.zse --kv-quant int8`}),i.jsx(a.d,{language:"python",code:`from zllm_zse import ZSE

# Enable quantized KV cache
model = ZSE("qwen-7b.zse", kv_quant="int4")

# Now supports 4x longer contexts!
response = model.chat(
    messages=[...],
    max_context=131072  # 128K context
)`})]})]}),(0,i.jsxs)(r.Je,{id:"persistent-cache",title:"Persistent Cache",children:[i.jsx("p",{className:"mb-4",children:"Save and restore KV cache for long-running conversations:"}),i.jsx(a.d,{language:"python",code:`from zllm_zse import ZSE

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
# → "Your name is Alice"`}),i.jsx(r.KU,{id:"session-api",title:"Session Management API",children:i.jsx(a.d,{language:"python",code:`from zllm_zse import ZSE, Session

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
model.delete_session("user_123")`})})]}),(0,i.jsxs)(r.Je,{id:"memory-optimization",title:"Memory Optimization",children:[(0,i.jsxs)(r.KU,{id:"paged-attention",title:"Paged Attention",children:[i.jsx("p",{className:"mb-2",children:"Paged attention allocates KV cache in fixed-size blocks, reducing memory fragmentation:"}),i.jsx(a.d,{language:"bash",code:`# Enable paged attention
zse serve model.zse --paged-attention

# Configure block size
zse serve model.zse --paged-attention --block-size 16`}),i.jsx(o.VS,{features:["Reduces memory fragmentation","Enables dynamic batch sizes","More concurrent requests with same memory"]})]}),(0,i.jsxs)(r.KU,{id:"memory-tiers",title:"Memory Tiers",children:[i.jsx("p",{className:"mb-2",children:"ZSE can use CPU memory and disk as overflow for KV cache:"}),i.jsx(a.d,{language:"yaml",filename:"zse.yaml",code:`kv_cache:
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
      
  eviction: lru  # Evict least-recently-used`}),i.jsx(n.U,{type:"warning",children:"CPU and disk tiers add latency. Use GPU memory for latency-sensitive workloads."})]}),i.jsx(r.KU,{id:"cache-stats",title:"Cache Statistics",children:i.jsx(a.d,{language:"python",code:`from zllm_zse import ZSE

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
model.clear_kv_cache(session="user_123")`})})]}),i.jsx(r.KO,{prev:{title:"zStream",href:"/docs/zstream"},next:{title:"CLI Reference",href:"/docs/api/cli"}})]}),i.jsx(r.o5,{items:c})]})}},9039:(e,s,t)=>{"use strict";t.d(s,{Rg:()=>o,VS:()=>c,Zb:()=>l,gy:()=>d});var i=t(5344),r=t(1912),a=t(2312),n=t(1453);function o({steps:e}){return i.jsx("div",{className:"my-6 space-y-0",children:e.map((s,t)=>(0,i.jsxs)(r.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*t},className:"relative pl-8 pb-8 last:pb-0",children:[t<e.length-1&&i.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),i.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:i.jsx("span",{className:"text-xs font-bold text-lime",children:t+1})}),(0,i.jsxs)("div",{children:[i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:s.title}),s.description&&i.jsx("p",{className:"text-sm text-white/50 mb-3",children:s.description}),s.code&&i.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:i.jsx("code",{className:"text-sm text-lime/90 font-mono",children:s.code})}),s.content&&i.jsx("div",{className:"text-sm text-white/70",children:s.content})]})]},t))})}function c({features:e}){return i.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,s)=>(0,i.jsxs)(r.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*s},className:"flex items-start gap-2",children:[i.jsx(a.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),i.jsx("span",{className:"text-sm text-white/70",children:e})]},s))})}function l({title:e,description:s,icon:t,href:a,children:o}){let c=a?"a":"div";return i.jsx(r.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:a?{y:-2}:void 0,children:(0,i.jsxs)(c,{...a?{href:a,className:"block"}:{},className:(0,n.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",a&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[t&&i.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:i.jsx(t,{className:"w-4 h-4 text-lime"})}),i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),s&&i.jsx("p",{className:"text-sm text-white/50",children:s}),o]})})}function d({children:e,columns:s=2}){return i.jsx("div",{className:(0,n.cn)("grid gap-4 my-6",2===s&&"md:grid-cols-2",3===s&&"md:grid-cols-3"),children:e})}},1386:(e,s,t)=>{"use strict";t.r(s),t.d(s,{$$typeof:()=>a,__esModule:()=>r,default:()=>n});let i=(0,t(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/zkv/page.tsx`),{__esModule:r,$$typeof:a}=i,n=i.default}};var s=require("../../../webpack-runtime.js");s.C(e);var t=e=>s(s.s=e),i=s.X(0,[638,498,697,224,782,883],()=>t(9927));module.exports=i})();