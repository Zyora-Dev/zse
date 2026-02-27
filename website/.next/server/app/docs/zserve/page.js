(()=>{var e={};e.id=138,e.ids=[138],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},9443:(e,s,t)=>{"use strict";t.r(s),t.d(s,{GlobalError:()=>l.a,__next_app__:()=>p,originalPathname:()=>m,pages:()=>c,routeModule:()=>u,tree:()=>d});var i=t(482),r=t(9108),o=t(2563),l=t.n(o),n=t(8300),a={};for(let e in n)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(a[e]=()=>n[e]);t.d(s,a);let d=["",{children:["docs",{children:["zserve",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(t.bind(t,1869)),"/Users/redfoxhotels/zse/website/src/app/docs/zserve/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(t.bind(t,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(t.bind(t,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(t.t.bind(t,9361,23)),"next/dist/client/components/not-found-error"]}],c=["/Users/redfoxhotels/zse/website/src/app/docs/zserve/page.tsx"],m="/docs/zserve/page",p={require:t,loadChunk:()=>Promise.resolve()},u=new i.AppPageRouteModule({definition:{kind:r.x.APP_PAGE,page:"/docs/zserve/page",pathname:"/docs/zserve",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:d}})},6035:(e,s,t)=>{Promise.resolve().then(t.bind(t,7155))},7155:(e,s,t)=>{"use strict";t.r(s),t.d(s,{default:()=>d});var i=t(5344),r=t(1499),o=t(196),l=t(1812),n=t(9039);let a=[{id:"overview",title:"Overview",level:2},{id:"basic-usage",title:"Basic Usage",level:2},{id:"configuration",title:"Configuration",level:2},{id:"multi-model",title:"Multi-Model Serving",level:2},{id:"scaling",title:"Scaling",level:2},{id:"monitoring",title:"Monitoring",level:2}];function d(){return(0,i.jsxs)("div",{className:"flex",children:[(0,i.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[i.jsx(r.lv,{title:"zServe",description:"Production-ready inference server with OpenAI-compatible API and sub-4-second cold start.",badge:"Feature"}),(0,i.jsxs)(r.Je,{id:"overview",title:"Overview",children:[(0,i.jsxs)("p",{className:"mb-4",children:[i.jsx(o.Z,{children:"zServe"})," is ZSE's high-performance inference server, designed for production workloads with minimal latency and maximum throughput."]}),(0,i.jsxs)(n.gy,{columns:3,children:[i.jsx(n.Zb,{title:"3.9s Cold Start",description:"Fastest model loading in the industry"}),i.jsx(n.Zb,{title:"OpenAI Compatible",description:"Drop-in replacement for OpenAI API"}),i.jsx(n.Zb,{title:"Multi-GPU",description:"Automatic model parallelism"})]}),i.jsx(n.VS,{features:["OpenAI-compatible /v1/chat/completions endpoint","Streaming responses with SSE","Request batching and queuing","Built-in rate limiting","Prometheus metrics endpoint","Health checks and graceful shutdown"]})]}),(0,i.jsxs)(r.Je,{id:"basic-usage",title:"Basic Usage",children:[i.jsx(n.Rg,{steps:[{title:"Start the server",description:"Launch with a model",code:"zse serve qwen-7b.zse"},{title:"Test the endpoint",description:"Send a request with curl",code:`curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "Hello!"}]}'`},{title:"Use with OpenAI client",description:"Point your existing code to the server",code:`from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="qwen-7b",
    messages=[{"role": "user", "content": "Hello!"}]
)`}]}),(0,i.jsxs)(l.U,{type:"info",children:["The server starts on port 8000 by default. Use ",i.jsx(o.Z,{children:"--port"})," to change."]})]}),(0,i.jsxs)(r.Je,{id:"configuration",title:"Configuration",children:[i.jsx(r.KU,{id:"cli-options",title:"CLI Options",children:i.jsx(o.d,{language:"bash",code:`zse serve model.zse \\
  --port 8000           # Server port
  --host 0.0.0.0        # Bind address
  --workers 4           # Worker processes
  --max-batch 32        # Max batch size
  --max-concurrent 100  # Max concurrent requests
  --timeout 60          # Request timeout (seconds)
  --api-key "sk-xxx"    # Require API key`})}),(0,i.jsxs)(r.KU,{id:"config-file",title:"Config File",children:[i.jsx("p",{className:"mb-2",children:"Use a YAML config file for complex setups:"}),i.jsx(o.d,{language:"yaml",filename:"zse.yaml",code:`server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
model:
  path: ./qwen-7b.zse
  max_batch_size: 32
  max_sequence_length: 4096
  
inference:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 2048
  
limits:
  max_concurrent_requests: 100
  requests_per_minute: 1000
  timeout_seconds: 60
  
auth:
  api_keys:
    - sk-key-1
    - sk-key-2`}),i.jsx(o.d,{language:"bash",code:"zse serve --config zse.yaml"})]}),i.jsx(r.KU,{id:"environment",title:"Environment Variables",children:i.jsx(o.d,{language:"bash",code:`# Model configuration
export ZSE_MODEL_PATH="./qwen-7b.zse"
export ZSE_MAX_BATCH_SIZE=32

# Server configuration
export ZSE_PORT=8000
export ZSE_HOST=0.0.0.0

# Auth
export ZSE_API_KEY="sk-xxx"

# Start server
zse serve`})})]}),(0,i.jsxs)(r.Je,{id:"multi-model",title:"Multi-Model Serving",children:[i.jsx("p",{className:"mb-4",children:"Serve multiple models from a single server instance:"}),i.jsx(o.d,{language:"bash",code:`# Serve multiple models from a directory
zse serve ./models/

# Or specify individually
zse serve model1.zse model2.zse model3.zse`}),i.jsx("p",{className:"mt-4",children:"Models are loaded on-demand with LRU caching:"}),i.jsx(o.d,{language:"yaml",filename:"zse.yaml",code:`models:
  - name: qwen-7b
    path: ./qwen-7b.zse
    max_loaded: true      # Keep loaded
    
  - name: llama-8b
    path: ./llama-8b.zse
    max_loaded: false     # Load on demand
    
  - name: codellama-34b
    path: ./codellama-34b.zse
    gpu: [0, 1]           # Specific GPUs

cache:
  max_models: 3           # Max models in memory
  eviction: lru           # Eviction policy`}),(0,i.jsxs)(l.U,{type:"tip",children:["Pin frequently-used models with ",i.jsx(o.Z,{children:"max_loaded: true"})," to avoid cold starts."]})]}),(0,i.jsxs)(r.Je,{id:"scaling",title:"Scaling",children:[(0,i.jsxs)(r.KU,{id:"multi-gpu",title:"Multi-GPU",children:[i.jsx("p",{className:"mb-2",children:"Automatically shard large models across GPUs:"}),i.jsx(o.d,{language:"bash",code:`# Auto-detect and use all GPUs
zse serve model.zse --tensor-parallel auto

# Specify GPUs
zse serve model.zse --tensor-parallel 4 --gpus 0,1,2,3

# Pipeline parallelism for very large models
zse serve model.zse --pipeline-parallel 2`})]}),(0,i.jsxs)(r.KU,{id:"load-balancing",title:"Load Balancing",children:[i.jsx("p",{className:"mb-2",children:"Run multiple instances behind a load balancer:"}),i.jsx(o.d,{language:"nginx",filename:"nginx.conf",code:`upstream zse {
    least_conn;
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    
    location /v1/ {
        proxy_pass http://zse;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # For streaming
        proxy_buffering off;
        proxy_cache off;
    }
}`})]}),i.jsx(r.KU,{id:"kubernetes",title:"Kubernetes",children:i.jsx(o.d,{language:"yaml",filename:"deployment.yaml",code:`apiVersion: apps/v1
kind: Deployment
metadata:
  name: zse
spec:
  replicas: 3
  selector:
    matchLabels:
      app: zse
  template:
    spec:
      containers:
      - name: zse
        image: zllm/zse:latest
        command: ["zse", "serve", "/models/qwen-7b.zse"]
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models
          mountPath: /models
        readinessProbe:
          httpGet:
            path: /health
            port: 8000`})})]}),(0,i.jsxs)(r.Je,{id:"monitoring",title:"Monitoring",children:[(0,i.jsxs)(r.KU,{id:"health-endpoint",title:"Health Endpoint",children:[i.jsx(o.d,{language:"bash",code:"curl http://localhost:8000/health"}),i.jsx(o.d,{language:"json",code:`{
  "status": "healthy",
  "model": "qwen-7b",
  "uptime": 3600,
  "requests_processed": 15420,
  "gpu_memory_used": "4.2 GB",
  "gpu_memory_total": "24 GB"
}`})]}),(0,i.jsxs)(r.KU,{id:"metrics",title:"Prometheus Metrics",children:[i.jsx(o.d,{language:"bash",code:"curl http://localhost:8000/metrics"}),i.jsx(o.d,{language:"text",code:`# TYPE zse_requests_total counter
zse_requests_total{model="qwen-7b",status="success"} 15420
zse_requests_total{model="qwen-7b",status="error"} 12

# TYPE zse_request_duration_seconds histogram
zse_request_duration_seconds_bucket{le="0.1"} 1000
zse_request_duration_seconds_bucket{le="0.5"} 12000
zse_request_duration_seconds_bucket{le="1.0"} 15000

# TYPE zse_tokens_generated_total counter
zse_tokens_generated_total{model="qwen-7b"} 2456789

# TYPE zse_gpu_memory_bytes gauge
zse_gpu_memory_bytes{gpu="0"} 4500000000`})]}),i.jsx(r.KU,{id:"logging",title:"Logging",children:i.jsx(o.d,{language:"bash",code:`# Verbose logging
zse serve model.zse --log-level debug

# JSON logs for production
zse serve model.zse --log-format json

# Log to file
zse serve model.zse --log-file /var/log/zse/server.log`})})]}),i.jsx(r.KO,{prev:{title:"zQuantize",href:"/docs/zquantize"},next:{title:"zInfer",href:"/docs/zinfer"}})]}),i.jsx(r.o5,{items:a})]})}},9039:(e,s,t)=>{"use strict";t.d(s,{Rg:()=>n,VS:()=>a,Zb:()=>d,gy:()=>c});var i=t(5344),r=t(1912),o=t(2312),l=t(1453);function n({steps:e}){return i.jsx("div",{className:"my-6 space-y-0",children:e.map((s,t)=>(0,i.jsxs)(r.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*t},className:"relative pl-8 pb-8 last:pb-0",children:[t<e.length-1&&i.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),i.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:i.jsx("span",{className:"text-xs font-bold text-lime",children:t+1})}),(0,i.jsxs)("div",{children:[i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:s.title}),s.description&&i.jsx("p",{className:"text-sm text-white/50 mb-3",children:s.description}),s.code&&i.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:i.jsx("code",{className:"text-sm text-lime/90 font-mono",children:s.code})}),s.content&&i.jsx("div",{className:"text-sm text-white/70",children:s.content})]})]},t))})}function a({features:e}){return i.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,s)=>(0,i.jsxs)(r.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*s},className:"flex items-start gap-2",children:[i.jsx(o.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),i.jsx("span",{className:"text-sm text-white/70",children:e})]},s))})}function d({title:e,description:s,icon:t,href:o,children:n}){let a=o?"a":"div";return i.jsx(r.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:o?{y:-2}:void 0,children:(0,i.jsxs)(a,{...o?{href:o,className:"block"}:{},className:(0,l.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",o&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[t&&i.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:i.jsx(t,{className:"w-4 h-4 text-lime"})}),i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),s&&i.jsx("p",{className:"text-sm text-white/50",children:s}),n]})})}function c({children:e,columns:s=2}){return i.jsx("div",{className:(0,l.cn)("grid gap-4 my-6",2===s&&"md:grid-cols-2",3===s&&"md:grid-cols-3"),children:e})}},1869:(e,s,t)=>{"use strict";t.r(s),t.d(s,{$$typeof:()=>o,__esModule:()=>r,default:()=>l});let i=(0,t(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/zserve/page.tsx`),{__esModule:r,$$typeof:o}=i,l=i.default}};var s=require("../../../webpack-runtime.js");s.C(e);var t=e=>s(s.s=e),i=s.X(0,[638,498,697,224,782,883],()=>t(9443));module.exports=i})();