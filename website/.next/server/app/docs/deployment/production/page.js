(()=>{var e={};e.id=617,e.ids=[617],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},3378:(e,t,s)=>{"use strict";s.r(t),s.d(t,{GlobalError:()=>n.a,__next_app__:()=>m,originalPathname:()=>p,pages:()=>d,routeModule:()=>x,tree:()=>c});var i=s(482),r=s(9108),l=s(2563),n=s.n(l),a=s(8300),o={};for(let e in a)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(o[e]=()=>a[e]);s.d(t,o);let c=["",{children:["docs",{children:["deployment",{children:["production",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(s.bind(s,2856)),"/Users/redfoxhotels/zse/website/src/app/docs/deployment/production/page.tsx"]}]},{}]},{}]},{layout:[()=>Promise.resolve().then(s.bind(s,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(s.bind(s,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(s.t.bind(s,9361,23)),"next/dist/client/components/not-found-error"]}],d=["/Users/redfoxhotels/zse/website/src/app/docs/deployment/production/page.tsx"],p="/docs/deployment/production/page",m={require:s,loadChunk:()=>Promise.resolve()},x=new i.AppPageRouteModule({definition:{kind:r.x.APP_PAGE,page:"/docs/deployment/production/page",pathname:"/docs/deployment/production",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:c}})},5568:(e,t,s)=>{Promise.resolve().then(s.bind(s,6056))},6056:(e,t,s)=>{"use strict";s.r(t),s.d(t,{default:()=>c});var i=s(5344),r=s(1499),l=s(196),n=s(1812),a=s(9039);let o=[{id:"overview",title:"Overview",level:2},{id:"system-requirements",title:"System Requirements",level:2},{id:"configuration",title:"Configuration",level:2},{id:"security",title:"Security",level:2},{id:"monitoring",title:"Monitoring",level:2},{id:"scaling",title:"Scaling",level:2}];function c(){return(0,i.jsxs)("div",{className:"flex",children:[(0,i.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[i.jsx(r.lv,{title:"Production Setup",description:"Deploy ZSE in production with best practices for security, performance, and reliability.",badge:"Deployment"}),(0,i.jsxs)(r.Je,{id:"overview",title:"Overview",children:[i.jsx("p",{className:"mb-4",children:"Running ZSE in production requires careful attention to security, monitoring, and resource management. This guide covers the essential configurations and best practices for a production deployment."}),(0,i.jsxs)(a.gy,{columns:3,children:[i.jsx(a.Zb,{title:"Security",description:"API keys, rate limiting, HTTPS"}),i.jsx(a.Zb,{title:"Performance",description:"GPU optimization, caching, batching"}),i.jsx(a.Zb,{title:"Reliability",description:"Health checks, logging, alerts"})]})]}),(0,i.jsxs)(r.Je,{id:"system-requirements",title:"System Requirements",children:[i.jsx("p",{className:"mb-4",children:"Recommended specifications for production deployments:"}),i.jsx("div",{className:"overflow-x-auto",children:(0,i.jsxs)("table",{className:"w-full text-sm",children:[i.jsx("thead",{children:(0,i.jsxs)("tr",{className:"border-b border-white/10",children:[i.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Component"}),i.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Minimum"}),i.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Recommended"})]})}),(0,i.jsxs)("tbody",{children:[(0,i.jsxs)("tr",{className:"border-b border-white/5",children:[i.jsx("td",{className:"py-3 px-4 text-white",children:"GPU (7B models)"}),i.jsx("td",{className:"py-3 px-4 text-gray-400",children:"8GB VRAM"}),i.jsx("td",{className:"py-3 px-4 text-lime",children:"24GB+ VRAM"})]}),(0,i.jsxs)("tr",{className:"border-b border-white/5",children:[i.jsx("td",{className:"py-3 px-4 text-white",children:"GPU (32B models)"}),i.jsx("td",{className:"py-3 px-4 text-gray-400",children:"24GB VRAM"}),i.jsx("td",{className:"py-3 px-4 text-lime",children:"80GB+ VRAM"})]}),(0,i.jsxs)("tr",{className:"border-b border-white/5",children:[i.jsx("td",{className:"py-3 px-4 text-white",children:"System RAM"}),i.jsx("td",{className:"py-3 px-4 text-gray-400",children:"16GB"}),i.jsx("td",{className:"py-3 px-4 text-lime",children:"64GB+"})]}),(0,i.jsxs)("tr",{className:"border-b border-white/5",children:[i.jsx("td",{className:"py-3 px-4 text-white",children:"Storage"}),i.jsx("td",{className:"py-3 px-4 text-gray-400",children:"50GB SSD"}),i.jsx("td",{className:"py-3 px-4 text-lime",children:"500GB+ NVMe"})]})]})]})}),i.jsx(n.U,{type:"info",children:"Use INT4 quantization to fit larger models on smaller GPUs. A 32B model can run on 24GB VRAM with INT4."})]}),(0,i.jsxs)(r.Je,{id:"configuration",title:"Configuration",children:[i.jsx(r.KU,{id:"env-vars",title:"Environment Variables",children:i.jsx(l.d,{language:"bash",code:`# Production environment variables
export ZSE_ENV=production
export ZSE_HOST=0.0.0.0
export ZSE_PORT=8000
export ZSE_WORKERS=4
export ZSE_LOG_LEVEL=info
export ZSE_MAX_BATCH_SIZE=32
export ZSE_MAX_CONTEXT=8192`})}),i.jsx(r.KU,{id:"config-file",title:"Configuration File",children:i.jsx(l.d,{language:"yaml",filename:"config.yaml",code:`# ZSE Production Configuration
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

model:
  path: /models/qwen7b.zse
  max_context: 8192
  batch_size: 32

memory:
  kv_cache: 4GB
  max_memory_per_gpu: 22GB

logging:
  level: info
  format: json
  output: /var/log/zse/server.log`})}),i.jsx(r.KU,{id:"start-production",title:"Starting the Server",children:i.jsx(l.d,{language:"bash",code:`# Start with config file
zse serve model.zse --config config.yaml

# Or with environment variables
ZSE_ENV=production zse serve model.zse \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --workers 4`})})]}),(0,i.jsxs)(r.Je,{id:"security",title:"Security",children:[(0,i.jsxs)(r.KU,{id:"api-keys",title:"API Key Authentication",children:[i.jsx("p",{className:"mb-4",children:"Always require API keys in production:"}),i.jsx(l.d,{language:"bash",code:`# Create an API key with rate limiting
zse api-key create production-app --rate-limit 100

# Start server with authentication
zse serve model.zse --require-auth

# Clients must include the key
curl -H "Authorization: Bearer sk-xxx" \\
  http://localhost:8000/v1/chat/completions`})]}),(0,i.jsxs)(r.KU,{id:"rate-limiting",title:"Rate Limiting",children:[i.jsx(l.d,{language:"bash",code:`# Per-key rate limits
zse api-key create app1 --rate-limit 60   # 60 req/min
zse api-key create app2 --rate-limit 1000 # 1000 req/min

# Check rate limit status
zse api-key status app1`}),i.jsx(n.U,{type:"warning",children:"Rate limits help prevent abuse and ensure fair resource allocation. Clients receive 429 responses when limits are exceeded."})]}),(0,i.jsxs)(r.KU,{id:"https",title:"HTTPS with Reverse Proxy",children:[i.jsx("p",{className:"mb-4",children:"Use a reverse proxy like Nginx for HTTPS termination:"}),i.jsx(l.d,{language:"nginx",filename:"nginx.conf",code:`upstream zse {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass http://zse;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}`})]})]}),(0,i.jsxs)(r.Je,{id:"monitoring",title:"Monitoring",children:[i.jsx(r.KU,{id:"health-checks",title:"Health Checks",children:i.jsx(l.d,{language:"bash",code:`# Liveness probe
curl http://localhost:8000/health

# Readiness probe (model loaded)
curl http://localhost:8000/ready

# Detailed status
curl http://localhost:8000/v1/status`})}),i.jsx(r.KU,{id:"metrics",title:"Prometheus Metrics",children:i.jsx(l.d,{language:"bash",code:`# Enable Prometheus metrics
zse serve model.zse --metrics

# Metrics available at
curl http://localhost:8000/metrics

# Example metrics:
# zse_requests_total
# zse_tokens_generated_total
# zse_inference_latency_seconds
# zse_gpu_memory_used_bytes`})}),i.jsx(r.KU,{id:"logging",title:"Structured Logging",children:i.jsx(l.d,{language:"bash",code:`# JSON logging for production
zse serve model.zse --log-format json

# Log output example:
# {"timestamp":"2026-02-26T10:00:00Z","level":"info","msg":"Request completed","request_id":"abc123","tokens":150,"latency_ms":1234}`})})]}),(0,i.jsxs)(r.Je,{id:"scaling",title:"Scaling",children:[(0,i.jsxs)(r.KU,{id:"horizontal",title:"Horizontal Scaling",children:[i.jsx("p",{className:"mb-4",children:"Run multiple ZSE instances behind a load balancer:"}),i.jsx(l.d,{language:"bash",code:`# Instance 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 zse serve model.zse --port 8001

# Instance 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 zse serve model.zse --port 8002

# Load balance with Nginx
upstream zse {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}`})]}),i.jsx(r.KU,{id:"multi-model",title:"Multi-Model Deployment",children:i.jsx(l.d,{language:"bash",code:`# Serve different models on different ports
zse serve qwen7b.zse --port 8001   # Fast, general purpose
zse serve qwen32b.zse --port 8002  # High quality, slower

# Route by endpoint or header`})}),(0,i.jsxs)(n.U,{type:"tip",children:["For high-availability deployments, use Kubernetes with multiple replicas. See the ",i.jsx("a",{href:"/docs/deployment/kubernetes",className:"text-lime hover:underline",children:"Kubernetes guide"}),"."]})]}),i.jsx(r.KO,{prev:{href:"/docs/api/config",title:"Configuration"},next:{href:"/docs/deployment/docker",title:"Docker"}})]}),i.jsx(r.o5,{items:o})]})}},9039:(e,t,s)=>{"use strict";s.d(t,{Rg:()=>a,VS:()=>o,Zb:()=>c,gy:()=>d});var i=s(5344),r=s(1912),l=s(2312),n=s(1453);function a({steps:e}){return i.jsx("div",{className:"my-6 space-y-0",children:e.map((t,s)=>(0,i.jsxs)(r.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*s},className:"relative pl-8 pb-8 last:pb-0",children:[s<e.length-1&&i.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),i.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:i.jsx("span",{className:"text-xs font-bold text-lime",children:s+1})}),(0,i.jsxs)("div",{children:[i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:t.title}),t.description&&i.jsx("p",{className:"text-sm text-white/50 mb-3",children:t.description}),t.code&&i.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:i.jsx("code",{className:"text-sm text-lime/90 font-mono",children:t.code})}),t.content&&i.jsx("div",{className:"text-sm text-white/70",children:t.content})]})]},s))})}function o({features:e}){return i.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,t)=>(0,i.jsxs)(r.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*t},className:"flex items-start gap-2",children:[i.jsx(l.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),i.jsx("span",{className:"text-sm text-white/70",children:e})]},t))})}function c({title:e,description:t,icon:s,href:l,children:a}){let o=l?"a":"div";return i.jsx(r.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:l?{y:-2}:void 0,children:(0,i.jsxs)(o,{...l?{href:l,className:"block"}:{},className:(0,n.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",l&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[s&&i.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:i.jsx(s,{className:"w-4 h-4 text-lime"})}),i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),t&&i.jsx("p",{className:"text-sm text-white/50",children:t}),a]})})}function d({children:e,columns:t=2}){return i.jsx("div",{className:(0,n.cn)("grid gap-4 my-6",2===t&&"md:grid-cols-2",3===t&&"md:grid-cols-3"),children:e})}},2856:(e,t,s)=>{"use strict";s.r(t),s.d(t,{$$typeof:()=>l,__esModule:()=>r,default:()=>n});let i=(0,s(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/deployment/production/page.tsx`),{__esModule:r,$$typeof:l}=i,n=i.default}};var t=require("../../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),i=t.X(0,[638,498,697,224,782,883],()=>s(3378));module.exports=i})();