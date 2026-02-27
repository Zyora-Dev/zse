(()=>{var e={};e.id=899,e.ids=[899],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},2034:(e,s,t)=>{"use strict";t.r(s),t.d(s,{GlobalError:()=>l.a,__next_app__:()=>p,originalPathname:()=>m,pages:()=>c,routeModule:()=>u,tree:()=>d});var i=t(482),o=t(9108),r=t(2563),l=t.n(r),n=t(8300),a={};for(let e in n)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(a[e]=()=>n[e]);t.d(s,a);let d=["",{children:["docs",{children:["deployment",{children:["docker",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(t.bind(t,7947)),"/Users/redfoxhotels/zse/website/src/app/docs/deployment/docker/page.tsx"]}]},{}]},{}]},{layout:[()=>Promise.resolve().then(t.bind(t,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(t.bind(t,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(t.t.bind(t,9361,23)),"next/dist/client/components/not-found-error"]}],c=["/Users/redfoxhotels/zse/website/src/app/docs/deployment/docker/page.tsx"],m="/docs/deployment/docker/page",p={require:t,loadChunk:()=>Promise.resolve()},u=new i.AppPageRouteModule({definition:{kind:o.x.APP_PAGE,page:"/docs/deployment/docker/page",pathname:"/docs/deployment/docker",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:d}})},2173:(e,s,t)=>{Promise.resolve().then(t.bind(t,5354))},5354:(e,s,t)=>{"use strict";t.r(s),t.d(s,{default:()=>d});var i=t(5344),o=t(1499),r=t(196),l=t(1812),n=t(9039);let a=[{id:"overview",title:"Overview",level:2},{id:"quick-start",title:"Quick Start",level:2},{id:"dockerfile",title:"Dockerfile",level:2},{id:"docker-compose",title:"Docker Compose",level:2},{id:"gpu-support",title:"GPU Support",level:2},{id:"volumes",title:"Volumes & Persistence",level:2}];function d(){return(0,i.jsxs)("div",{className:"flex",children:[(0,i.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[i.jsx(o.lv,{title:"Docker Deployment",description:"Run ZSE in Docker containers with GPU acceleration for consistent, reproducible deployments.",badge:"Deployment"}),(0,i.jsxs)(o.Je,{id:"overview",title:"Overview",children:[i.jsx("p",{className:"mb-4",children:"Docker provides isolation and reproducibility for ZSE deployments. Our official images include all dependencies and support NVIDIA GPU acceleration."}),(0,i.jsxs)(n.gy,{columns:3,children:[i.jsx(n.Zb,{title:"Official Images",description:"Pre-built images on Docker Hub"}),i.jsx(n.Zb,{title:"GPU Ready",description:"NVIDIA Container Toolkit support"}),i.jsx(n.Zb,{title:"Compose",description:"Multi-container orchestration"})]})]}),(0,i.jsxs)(o.Je,{id:"quick-start",title:"Quick Start",children:[i.jsx("p",{className:"mb-4",children:"Run ZSE with a single command using our official Docker image:"}),i.jsx(r.d,{language:"bash",code:`# Pull the latest image
docker pull zyora/zse:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 \\
  zyora/zse:latest \\
  zse serve Qwen/Qwen2.5-7B-Instruct`}),i.jsx(l.U,{type:"info",children:"The first run will download the model from HuggingFace. Use a volume to persist models between container restarts."})]}),(0,i.jsxs)(o.Je,{id:"dockerfile",title:"Dockerfile",children:[i.jsx(o.KU,{id:"official-image",title:"Using Official Image",children:i.jsx(r.d,{language:"dockerfile",filename:"Dockerfile",code:`FROM zyora/zse:latest

# Copy your pre-converted model
COPY ./models/qwen7b.zse /models/

# Set environment variables
ENV ZSE_HOST=0.0.0.0
ENV ZSE_PORT=8000

# Expose the port
EXPOSE 8000

# Start the server
CMD ["zse", "serve", "/models/qwen7b.zse"]`})}),i.jsx(o.KU,{id:"custom-image",title:"Building Custom Image",children:i.jsx(r.d,{language:"dockerfile",filename:"Dockerfile",code:`FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

# Install ZSE
RUN pip install zllm-zse

# Create model directory
RUN mkdir -p /models

# Set working directory
WORKDIR /app

# Environment
ENV ZSE_HOST=0.0.0.0
ENV ZSE_PORT=8000

EXPOSE 8000

CMD ["zse", "serve", "/models/model.zse"]`})}),i.jsx(o.KU,{id:"build",title:"Building the Image",children:i.jsx(r.d,{language:"bash",code:`# Build the image
docker build -t my-zse-app .

# Run with GPU
docker run --gpus all -p 8000:8000 my-zse-app`})})]}),(0,i.jsxs)(o.Je,{id:"docker-compose",title:"Docker Compose",children:[i.jsx(o.KU,{id:"basic-compose",title:"Basic Setup",children:i.jsx(r.d,{language:"yaml",filename:"docker-compose.yml",code:`version: '3.8'

services:
  zse:
    image: zyora/zse:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
    environment:
      - ZSE_HOST=0.0.0.0
      - ZSE_PORT=8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: zse serve /models/qwen7b.zse`})}),i.jsx(o.KU,{id:"full-stack",title:"Full Stack with Monitoring",children:i.jsx(r.d,{language:"yaml",filename:"docker-compose.yml",code:`version: '3.8'

services:
  zse:
    image: zyora/zse:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models
      - ./logs:/var/log/zse
    environment:
      - ZSE_HOST=0.0.0.0
      - ZSE_PORT=8000
      - ZSE_LOG_FORMAT=json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: zse serve /models/qwen7b.zse --metrics
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/ssl/certs
    depends_on:
      - zse

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - zse`})}),i.jsx(o.KU,{id:"compose-commands",title:"Commands",children:i.jsx(r.d,{language:"bash",code:`# Start services
docker-compose up -d

# View logs
docker-compose logs -f zse

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build`})})]}),(0,i.jsxs)(o.Je,{id:"gpu-support",title:"GPU Support",children:[(0,i.jsxs)(o.KU,{id:"nvidia-toolkit",title:"NVIDIA Container Toolkit",children:[i.jsx("p",{className:"mb-4",children:"Install the NVIDIA Container Toolkit to enable GPU access in Docker:"}),i.jsx(r.d,{language:"bash",code:`# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker`})]}),i.jsx(o.KU,{id:"gpu-flags",title:"GPU Flags",children:i.jsx(r.d,{language:"bash",code:`# Use all GPUs
docker run --gpus all ...

# Use specific GPUs
docker run --gpus '"device=0,1"' ...

# Use N GPUs
docker run --gpus 2 ...`})}),i.jsx(o.KU,{id:"verify-gpu",title:"Verify GPU Access",children:i.jsx(r.d,{language:"bash",code:`docker run --gpus all zyora/zse:latest zse hardware
# Should show detected GPUs`})})]}),(0,i.jsxs)(o.Je,{id:"volumes",title:"Volumes & Persistence",children:[i.jsx(o.KU,{id:"model-volume",title:"Model Storage",children:i.jsx(r.d,{language:"bash",code:`# Create a named volume for models
docker volume create zse-models

# Run with volume
docker run --gpus all -p 8000:8000 \\
  -v zse-models:/models \\
  zyora/zse:latest \\
  zse serve /models/qwen7b.zse`})}),i.jsx(o.KU,{id:"download-models",title:"Downloading Models",children:i.jsx(r.d,{language:"bash",code:`# Download and convert a model into the volume
docker run --gpus all \\
  -v zse-models:/models \\
  zyora/zse:latest \\
  zse convert Qwen/Qwen2.5-7B-Instruct -o /models/qwen7b.zse

# Now serve it
docker run --gpus all -p 8000:8000 \\
  -v zse-models:/models \\
  zyora/zse:latest \\
  zse serve /models/qwen7b.zse`})}),i.jsx(l.U,{type:"tip",children:"Pre-convert models to .zse format and include them in your image for faster cold starts in production."})]}),i.jsx(o.KO,{prev:{href:"/docs/deployment/production",title:"Production Setup"},next:{href:"/docs/deployment/kubernetes",title:"Kubernetes"}})]}),i.jsx(o.o5,{items:a})]})}},9039:(e,s,t)=>{"use strict";t.d(s,{Rg:()=>n,VS:()=>a,Zb:()=>d,gy:()=>c});var i=t(5344),o=t(1912),r=t(2312),l=t(1453);function n({steps:e}){return i.jsx("div",{className:"my-6 space-y-0",children:e.map((s,t)=>(0,i.jsxs)(o.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*t},className:"relative pl-8 pb-8 last:pb-0",children:[t<e.length-1&&i.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),i.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:i.jsx("span",{className:"text-xs font-bold text-lime",children:t+1})}),(0,i.jsxs)("div",{children:[i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:s.title}),s.description&&i.jsx("p",{className:"text-sm text-white/50 mb-3",children:s.description}),s.code&&i.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:i.jsx("code",{className:"text-sm text-lime/90 font-mono",children:s.code})}),s.content&&i.jsx("div",{className:"text-sm text-white/70",children:s.content})]})]},t))})}function a({features:e}){return i.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,s)=>(0,i.jsxs)(o.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*s},className:"flex items-start gap-2",children:[i.jsx(r.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),i.jsx("span",{className:"text-sm text-white/70",children:e})]},s))})}function d({title:e,description:s,icon:t,href:r,children:n}){let a=r?"a":"div";return i.jsx(o.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:r?{y:-2}:void 0,children:(0,i.jsxs)(a,{...r?{href:r,className:"block"}:{},className:(0,l.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",r&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[t&&i.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:i.jsx(t,{className:"w-4 h-4 text-lime"})}),i.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),s&&i.jsx("p",{className:"text-sm text-white/50",children:s}),n]})})}function c({children:e,columns:s=2}){return i.jsx("div",{className:(0,l.cn)("grid gap-4 my-6",2===s&&"md:grid-cols-2",3===s&&"md:grid-cols-3"),children:e})}},7947:(e,s,t)=>{"use strict";t.r(s),t.d(s,{$$typeof:()=>r,__esModule:()=>o,default:()=>l});let i=(0,t(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/deployment/docker/page.tsx`),{__esModule:o,$$typeof:r}=i,l=i.default}};var s=require("../../../../webpack-runtime.js");s.C(e);var t=e=>s(s.s=e),i=s.X(0,[638,498,697,224,782,883],()=>t(2034));module.exports=i})();