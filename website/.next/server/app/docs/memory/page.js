(()=>{var e={};e.id=524,e.ids=[524],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},1717:(e,s,t)=>{"use strict";t.r(s),t.d(s,{GlobalError:()=>o.a,__next_app__:()=>x,originalPathname:()=>m,pages:()=>c,routeModule:()=>p,tree:()=>d});var r=t(482),i=t(9108),a=t(2563),o=t.n(a),l=t(8300),n={};for(let e in l)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(n[e]=()=>l[e]);t.d(s,n);let d=["",{children:["docs",{children:["memory",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(t.bind(t,3213)),"/Users/redfoxhotels/zse/website/src/app/docs/memory/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(t.bind(t,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(t.bind(t,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(t.t.bind(t,9361,23)),"next/dist/client/components/not-found-error"]}],c=["/Users/redfoxhotels/zse/website/src/app/docs/memory/page.tsx"],m="/docs/memory/page",x={require:t,loadChunk:()=>Promise.resolve()},p=new r.AppPageRouteModule({definition:{kind:i.x.APP_PAGE,page:"/docs/memory/page",pathname:"/docs/memory",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:d}})},4148:(e,s,t)=>{Promise.resolve().then(t.bind(t,528))},528:(e,s,t)=>{"use strict";t.r(s),t.d(s,{default:()=>d});var r=t(5344),i=t(1499),a=t(196),o=t(1812),l=t(9039);let n=[{id:"overview",title:"Overview",level:2},{id:"gpu-memory",title:"GPU Memory",level:2},{id:"model-sizing",title:"Model Sizing",level:2},{id:"memory-tiers",title:"Memory Tiers",level:2},{id:"optimization",title:"Optimization Tips",level:2}];function d(){return(0,r.jsxs)("div",{className:"flex",children:[(0,r.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[r.jsx(i.lv,{title:"Memory Management",description:"Understanding and optimizing memory usage for model inference.",badge:"Core Concepts"}),(0,r.jsxs)(i.Je,{id:"overview",title:"Overview",children:[r.jsx("p",{className:"mb-4",children:"Memory management is critical for running large language models. ZSE provides multiple strategies to maximize model size while minimizing resource usage."}),(0,r.jsxs)(l.gy,{columns:3,children:[r.jsx(l.Zb,{title:"4-bit Weights",description:"4x reduction in model memory"}),r.jsx(l.Zb,{title:"4-bit KV Cache",description:"4x longer context windows"}),r.jsx(l.Zb,{title:"Memory Tiers",description:"GPU → CPU → Disk overflow"})]})]}),(0,r.jsxs)(i.Je,{id:"gpu-memory",title:"GPU Memory",children:[r.jsx("p",{className:"mb-4",children:"GPU memory is used for three main components:"}),r.jsx(a.d,{language:"text",code:`Total GPU Memory
├── Model Weights (~60-80%)
│   └── Quantized weights + embedding layers
├── KV Cache (~15-30%)
│   └── Attention key-value pairs for context
├── Activations (~5-10%)
│   └── Intermediate computations during inference
└── CUDA Overhead (~2-5%)
    └── CUDA context, kernels, buffers`}),(0,r.jsxs)(i.KU,{id:"memory-formula",title:"Memory Estimation",children:[r.jsx("p",{className:"mb-2",children:"Estimate memory requirements:"}),r.jsx(a.d,{language:"text",code:`Model Memory = (Parameters \xd7 Bits per Param) / 8

Examples (4-bit quantized):
  7B params \xd7 4 bits / 8 = 3.5 GB
 14B params \xd7 4 bits / 8 = 7.0 GB
 70B params \xd7 4 bits / 8 = 35 GB

KV Cache Memory = 2 \xd7 Layers \xd7 Hidden \xd7 Context \xd7 Bytes per KV

Example (7B model, 8K context, FP16 KV):
  2 \xd7 32 \xd7 4096 \xd7 8192 \xd7 2 = 4.3 GB`})]}),(0,r.jsxs)(i.KU,{id:"check-memory",title:"Checking Memory",children:[r.jsx(a.d,{language:"bash",code:`# Check available GPU memory
zse hardware

# Check model memory requirements
zse info model.zse

# Monitor during inference
watch -n1 nvidia-smi`}),r.jsx(a.d,{language:"python",code:`from zllm_zse import ZSE

model = ZSE("qwen-7b.zse")

# Get memory info
mem = model.memory_info()
print(f"Model: {mem['model_size'] / 1e9:.1f} GB")
print(f"KV Cache: {mem['kv_cache_size'] / 1e9:.1f} GB")
print(f"Free GPU: {mem['gpu_free'] / 1e9:.1f} GB")`})]})]}),(0,r.jsxs)(i.Je,{id:"model-sizing",title:"Model Sizing",children:[r.jsx("p",{className:"mb-4",children:"Choose the right model size for your hardware:"}),r.jsx("div",{className:"overflow-x-auto my-4",children:(0,r.jsxs)("table",{className:"w-full text-sm",children:[r.jsx("thead",{children:(0,r.jsxs)("tr",{className:"border-b border-white/10",children:[r.jsx("th",{className:"text-left py-2 pr-4 text-white/50 font-medium",children:"GPU VRAM"}),r.jsx("th",{className:"text-left py-2 px-4 text-white/50 font-medium",children:"Max Model (NF4)"}),r.jsx("th",{className:"text-left py-2 pl-4 text-white/50 font-medium",children:"Context (FP16 KV)"})]})}),(0,r.jsxs)("tbody",{className:"text-white/70",children:[(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"8 GB"}),r.jsx("td",{className:"py-2 px-4",children:"7B"}),r.jsx("td",{className:"py-2 pl-4",children:"4K"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"12 GB"}),r.jsx("td",{className:"py-2 px-4",children:"14B"}),r.jsx("td",{className:"py-2 pl-4",children:"8K"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"16 GB"}),r.jsx("td",{className:"py-2 px-4",children:"14B"}),r.jsx("td",{className:"py-2 pl-4",children:"16K"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"24 GB"}),r.jsx("td",{className:"py-2 px-4",children:"34B"}),r.jsx("td",{className:"py-2 pl-4",children:"32K"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"48 GB"}),r.jsx("td",{className:"py-2 px-4",children:"70B"}),r.jsx("td",{className:"py-2 pl-4",children:"32K"})]}),(0,r.jsxs)("tr",{children:[r.jsx("td",{className:"py-2 pr-4",children:"80 GB"}),r.jsx("td",{className:"py-2 px-4",children:"70B"}),r.jsx("td",{className:"py-2 pl-4",children:"128K"})]})]})]})}),r.jsx(o.U,{type:"tip",children:"With 4-bit KV cache, you can double the context length in the table above."})]}),(0,r.jsxs)(i.Je,{id:"memory-tiers",title:"Memory Tiers",children:[r.jsx("p",{className:"mb-4",children:"ZSE supports tiered memory for running models larger than GPU memory:"}),r.jsx(a.d,{language:"text",code:`┌─────────────────────────────────────────────────────────┐
│ Tier 1: GPU Memory (fastest)                            │
│   - Model weights (always)                              │
│   - Active KV cache                                     │
│   - Current activations                                 │
├─────────────────────────────────────────────────────────┤
│ Tier 2: CPU Memory (slower)                             │
│   - Overflow KV cache                                   │
│   - Inactive model layers (offloading)                  │
├─────────────────────────────────────────────────────────┤
│ Tier 3: Disk (slowest)                                  │
│   - Cold KV cache                                       │
│   - Very long context overflow                          │
└─────────────────────────────────────────────────────────┘`}),(0,r.jsxs)(i.KU,{id:"cpu-offloading",title:"CPU Offloading",children:[r.jsx(a.d,{language:"bash",code:`# Offload some layers to CPU
zse serve model.zse --offload-layers 8

# Auto-detect optimal offload
zse serve model.zse --offload auto`}),r.jsx(o.U,{type:"warning",children:"CPU offloading reduces throughput by 2-5x. Use only when GPU memory is insufficient."})]}),r.jsx(i.KU,{id:"disk-cache",title:"Disk-Based KV Cache",children:r.jsx(a.d,{language:"yaml",filename:"zse.yaml",code:`kv_cache:
  tiers:
    - type: gpu
      size: auto        # Use available GPU memory
      
    - type: cpu
      size: 32GB        # Overflow to system RAM
      
    - type: disk
      path: /tmp/zse_cache
      size: 100GB       # For very long contexts`})})]}),(0,r.jsxs)(i.Je,{id:"optimization",title:"Optimization Tips",children:[r.jsx(l.VS,{features:["Use NF4 quantization for best quality/size ratio","Enable 4-bit KV cache for long contexts","Set max_context to your actual needs","Use prompt caching for repeated prefixes","Monitor with nvidia-smi during development"]}),r.jsx(i.KU,{id:"oom-errors",title:"Handling OOM Errors",children:r.jsx(a.d,{language:"bash",code:`# If you get CUDA OOM errors:

# 1. Reduce context length
zse serve model.zse --max-context 2048

# 2. Enable KV cache compression
zse serve model.zse --kv-quant int4

# 3. Reduce batch size
zse serve model.zse --max-batch 8

# 4. Try a smaller model
zse convert Qwen/Qwen2.5-3B-Instruct -o qwen-3b.zse`})}),r.jsx(i.KU,{id:"benchmarking",title:"Memory Benchmarking",children:r.jsx(a.d,{language:"bash",code:`# Benchmark memory usage
zse benchmark model.zse --metric memory

# Profile memory over time
zse serve model.zse --profile-memory --profile-output memory.json`})})]}),r.jsx(i.KO,{prev:{title:"Quantization",href:"/docs/quantization"},next:{title:"zQuantize",href:"/docs/zquantize"}})]}),r.jsx(i.o5,{items:n})]})}},9039:(e,s,t)=>{"use strict";t.d(s,{Rg:()=>l,VS:()=>n,Zb:()=>d,gy:()=>c});var r=t(5344),i=t(1912),a=t(2312),o=t(1453);function l({steps:e}){return r.jsx("div",{className:"my-6 space-y-0",children:e.map((s,t)=>(0,r.jsxs)(i.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*t},className:"relative pl-8 pb-8 last:pb-0",children:[t<e.length-1&&r.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),r.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:r.jsx("span",{className:"text-xs font-bold text-lime",children:t+1})}),(0,r.jsxs)("div",{children:[r.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:s.title}),s.description&&r.jsx("p",{className:"text-sm text-white/50 mb-3",children:s.description}),s.code&&r.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:r.jsx("code",{className:"text-sm text-lime/90 font-mono",children:s.code})}),s.content&&r.jsx("div",{className:"text-sm text-white/70",children:s.content})]})]},t))})}function n({features:e}){return r.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,s)=>(0,r.jsxs)(i.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*s},className:"flex items-start gap-2",children:[r.jsx(a.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),r.jsx("span",{className:"text-sm text-white/70",children:e})]},s))})}function d({title:e,description:s,icon:t,href:a,children:l}){let n=a?"a":"div";return r.jsx(i.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:a?{y:-2}:void 0,children:(0,r.jsxs)(n,{...a?{href:a,className:"block"}:{},className:(0,o.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",a&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[t&&r.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:r.jsx(t,{className:"w-4 h-4 text-lime"})}),r.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),s&&r.jsx("p",{className:"text-sm text-white/50",children:s}),l]})})}function c({children:e,columns:s=2}){return r.jsx("div",{className:(0,o.cn)("grid gap-4 my-6",2===s&&"md:grid-cols-2",3===s&&"md:grid-cols-3"),children:e})}},3213:(e,s,t)=>{"use strict";t.r(s),t.d(s,{$$typeof:()=>a,__esModule:()=>i,default:()=>o});let r=(0,t(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/memory/page.tsx`),{__esModule:i,$$typeof:a}=r,o=r.default}};var s=require("../../../webpack-runtime.js");s.C(e);var t=e=>s(s.s=e),r=s.X(0,[638,498,697,224,782,883],()=>t(1717));module.exports=r})();