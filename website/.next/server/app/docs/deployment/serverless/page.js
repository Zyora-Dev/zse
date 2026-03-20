(()=>{var e={};e.id=360,e.ids=[360],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},4425:(e,t,s)=>{"use strict";s.r(t),s.d(t,{GlobalError:()=>a.a,__next_app__:()=>p,originalPathname:()=>m,pages:()=>c,routeModule:()=>h,tree:()=>d});var r=s(482),o=s(9108),l=s(2563),a=s.n(l),n=s(8300),i={};for(let e in n)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(i[e]=()=>n[e]);s.d(t,i);let d=["",{children:["docs",{children:["deployment",{children:["serverless",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(s.bind(s,6009)),"/Users/redfoxhotels/zse/website/src/app/docs/deployment/serverless/page.tsx"]}]},{}]},{}]},{layout:[()=>Promise.resolve().then(s.bind(s,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(s.bind(s,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(s.t.bind(s,9361,23)),"next/dist/client/components/not-found-error"]}],c=["/Users/redfoxhotels/zse/website/src/app/docs/deployment/serverless/page.tsx"],m="/docs/deployment/serverless/page",p={require:s,loadChunk:()=>Promise.resolve()},h=new r.AppPageRouteModule({definition:{kind:o.x.APP_PAGE,page:"/docs/deployment/serverless/page",pathname:"/docs/deployment/serverless",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:d}})},2345:(e,t,s)=>{Promise.resolve().then(s.bind(s,6640))},6640:(e,t,s)=>{"use strict";s.r(t),s.d(t,{default:()=>d});var r=s(5344),o=s(1499),l=s(196),a=s(1812),n=s(9039);let i=[{id:"overview",title:"Overview",level:2},{id:"modal",title:"Modal",level:2},{id:"runpod",title:"RunPod",level:2},{id:"replicate",title:"Replicate",level:2},{id:"aws-lambda",title:"AWS Lambda",level:2},{id:"cold-start",title:"Cold Start Optimization",level:2}];function d(){return(0,r.jsxs)("div",{className:"flex",children:[(0,r.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[r.jsx(o.lv,{title:"Serverless Deployment",description:"Deploy ZSE on serverless GPU platforms for pay-per-use inference with automatic scaling.",badge:"Deployment"}),(0,r.jsxs)(o.Je,{id:"overview",title:"Overview",children:[r.jsx("p",{className:"mb-4",children:"Serverless GPU platforms let you run ZSE without managing infrastructure. Pay only for compute time used, with automatic scaling from zero to thousands of concurrent requests."}),(0,r.jsxs)(n.gy,{columns:2,children:[r.jsx(n.Zb,{title:"Pay-per-use",description:"No idle GPU costs, pay only for inference time"}),r.jsx(n.Zb,{title:"Auto-scaling",description:"Scale to zero or thousands automatically"})]}),r.jsx(a.U,{type:"info",children:"ZSE's fast cold start (3.9s for 7B models) makes it ideal for serverless where containers may be created on-demand."})]}),(0,r.jsxs)(o.Je,{id:"modal",title:"Modal",children:[r.jsx("p",{className:"mb-4",children:"Modal is recommended for ZSE serverless deployments due to its excellent GPU support and fast container start times."}),r.jsx(o.KU,{id:"modal-setup",title:"Setup",children:r.jsx(l.d,{language:"bash",code:`# Install Modal CLI
pip install modal

# Authenticate
modal token new`})}),r.jsx(o.KU,{id:"modal-app",title:"Modal Application",children:r.jsx(l.d,{language:"python",filename:"app.py",code:`import modal

app = modal.App("zse-inference")

# Define the image with ZSE
image = modal.Image.debian_slim().pip_install("zllm-zse")

# Create a volume for models
volume = modal.Volume.from_name("zse-models", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A10G",  # or "A100", "T4", "H100"
    volumes={"/models": volume},
    container_idle_timeout=300,  # Keep warm for 5 mins
)
class ZSEServer:
    @modal.enter()
    def load_model(self):
        from zse.engine.orchestrator import IntelligenceOrchestrator
        self.orch = IntelligenceOrchestrator("/models/qwen7b.zse")
        self.orch.load()

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 256):
        return self.orch.generate(prompt, max_tokens=max_tokens)

    @modal.method()
    def chat(self, messages: list):
        return self.orch.chat(messages)

# Web endpoint
@app.function(image=image, gpu="A10G", volumes={"/models": volume})
@modal.web_endpoint(method="POST")
def chat_endpoint(request: dict):
    from zse.engine.orchestrator import IntelligenceOrchestrator
    orch = IntelligenceOrchestrator("/models/qwen7b.zse")
    orch.load()
    return {"response": orch.chat(request["messages"])}`})}),r.jsx(o.KU,{id:"modal-deploy",title:"Deploy",children:r.jsx(l.d,{language:"bash",code:`# Upload model to volume first
modal volume put zse-models ./qwen7b.zse /qwen7b.zse

# Deploy the app
modal deploy app.py

# Test the endpoint
curl -X POST https://your-app--chat-endpoint.modal.run \\
  -H "Content-Type: application/json" \\
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'`})})]}),(0,r.jsxs)(o.Je,{id:"runpod",title:"RunPod",children:[r.jsx("p",{className:"mb-4",children:"RunPod offers serverless GPU endpoints with competitive pricing."}),r.jsx(o.KU,{id:"runpod-handler",title:"Handler",children:r.jsx(l.d,{language:"python",filename:"handler.py",code:`import runpod
from zse.engine.orchestrator import IntelligenceOrchestrator

# Load model globally (persists across requests)
orch = None

def load_model():
    global orch
    if orch is None:
        orch = IntelligenceOrchestrator("/models/qwen7b.zse")
        orch.load()
    return orch

def handler(job):
    """Handle inference requests."""
    job_input = job["input"]
    
    orch = load_model()
    
    if "messages" in job_input:
        # Chat completion
        response = orch.chat(job_input["messages"])
    else:
        # Text generation
        response = orch.generate(
            job_input.get("prompt", ""),
            max_tokens=job_input.get("max_tokens", 256)
        )
    
    return {"response": response}

runpod.serverless.start({"handler": handler})`})}),r.jsx(o.KU,{id:"runpod-dockerfile",title:"Dockerfile",children:r.jsx(l.d,{language:"dockerfile",filename:"Dockerfile",code:`FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.0-devel-ubuntu22.04

# Install ZSE
RUN pip install zllm-zse

# Copy model
COPY ./qwen7b.zse /models/

# Copy handler
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]`})})]}),(0,r.jsxs)(o.Je,{id:"replicate",title:"Replicate",children:[r.jsx("p",{className:"mb-4",children:"Deploy ZSE models on Replicate with Cog for easy API creation."}),r.jsx(o.KU,{id:"replicate-cog",title:"Cog Configuration",children:r.jsx(l.d,{language:"yaml",filename:"cog.yaml",code:`build:
  gpu: true
  python_version: "3.10"
  python_packages:
    - "zllm-zse"

predict: "predict.py:Predictor"`})}),r.jsx(o.KU,{id:"replicate-predict",title:"Predictor",children:r.jsx(l.d,{language:"python",filename:"predict.py",code:`from cog import BasePredictor, Input
from zse.engine.orchestrator import IntelligenceOrchestrator

class Predictor(BasePredictor):
    def setup(self):
        """Load the model."""
        self.orch = IntelligenceOrchestrator("./qwen7b.zse")
        self.orch.load()

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        max_tokens: int = Input(description="Max tokens", default=256),
        temperature: float = Input(description="Temperature", default=0.7),
    ) -> str:
        """Run inference."""
        return self.orch.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )`})}),r.jsx(o.KU,{id:"replicate-deploy",title:"Deploy",children:r.jsx(l.d,{language:"bash",code:`# Login to Replicate
cog login

# Push to Replicate
cog push r8.im/username/zse-qwen7b`})})]}),(0,r.jsxs)(o.Je,{id:"aws-lambda",title:"AWS Lambda",children:[r.jsx("p",{className:"mb-4",children:"For AWS deployments, use Lambda with container images and Provisioned Concurrency."}),r.jsx(a.U,{type:"warning",children:"AWS Lambda has a 10GB container size limit and 15-minute timeout. For larger models, use SageMaker or EC2 with auto-scaling instead."}),r.jsx(o.KU,{id:"lambda-container",title:"Container Setup",children:r.jsx(l.d,{language:"dockerfile",filename:"Dockerfile",code:`FROM public.ecr.aws/lambda/python:3.10

# Install ZSE (CPU-only for Lambda)
RUN pip install zllm-zse

# Copy model (must be < 10GB total)
COPY ./tinyllama.zse /models/

# Copy handler
COPY app.py \${LAMBDA_TASK_ROOT}

CMD ["app.handler"]`})}),r.jsx(o.KU,{id:"lambda-handler",title:"Handler",children:r.jsx(l.d,{language:"python",filename:"app.py",code:`from zse.engine.orchestrator import IntelligenceOrchestrator

# Load model outside handler for warm starts
orch = IntelligenceOrchestrator("/models/tinyllama.zse")
orch.load()

def handler(event, context):
    prompt = event.get("prompt", "")
    max_tokens = event.get("max_tokens", 128)
    
    response = orch.generate(prompt, max_tokens=max_tokens)
    
    return {
        "statusCode": 200,
        "body": {"response": response}
    }`})})]}),(0,r.jsxs)(o.Je,{id:"cold-start",title:"Cold Start Optimization",children:[r.jsx("p",{className:"mb-4",children:"Cold starts are critical for serverless. ZSE's .zse format provides significantly faster cold starts than alternatives."}),r.jsx("div",{className:"overflow-x-auto",children:(0,r.jsxs)("table",{className:"w-full text-sm",children:[r.jsx("thead",{children:(0,r.jsxs)("tr",{className:"border-b border-white/10",children:[r.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Format"}),r.jsx("th",{className:"text-right py-3 px-4 font-medium text-gray-400",children:"7B Cold Start"}),r.jsx("th",{className:"text-right py-3 px-4 font-medium text-gray-400",children:"Serverless Cost Impact"})]})}),(0,r.jsxs)("tbody",{children:[(0,r.jsxs)("tr",{className:"border-b border-white/5 bg-lime/5",children:[r.jsx("td",{className:"py-3 px-4 text-lime font-medium",children:".zse format"}),r.jsx("td",{className:"py-3 px-4 text-right text-lime font-mono",children:"3.9s"}),r.jsx("td",{className:"py-3 px-4 text-right text-lime",children:"Minimal startup overhead"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/5",children:[r.jsx("td",{className:"py-3 px-4 text-white",children:"bitsandbytes"}),r.jsx("td",{className:"py-3 px-4 text-right text-gray-400",children:"45.4s"}),r.jsx("td",{className:"py-3 px-4 text-right text-gray-400",children:"~40s billed startup"})]})]})]})}),r.jsx(n.VS,{features:["Always use pre-converted .zse models","Use container idle timeout to keep warm","Consider provisioned concurrency for consistent latency","Store models in fast storage (NVMe volumes)"]}),r.jsx(l.d,{language:"bash",code:`# Convert model before deployment (one-time)
zse convert Qwen/Qwen2.5-7B-Instruct -o qwen7b.zse

# Include .zse file in container (NOT HuggingFace weights)`})]}),r.jsx(o.KO,{prev:{href:"/docs/deployment/kubernetes",title:"Kubernetes"},next:{href:"/docs/advanced/custom-models",title:"Custom Models"}})]}),r.jsx(o.o5,{items:i})]})}},9039:(e,t,s)=>{"use strict";s.d(t,{Rg:()=>n,VS:()=>i,Zb:()=>d,gy:()=>c});var r=s(5344),o=s(1912),l=s(2312),a=s(1453);function n({steps:e}){return r.jsx("div",{className:"my-6 space-y-0",children:e.map((t,s)=>(0,r.jsxs)(o.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*s},className:"relative pl-8 pb-8 last:pb-0",children:[s<e.length-1&&r.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),r.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:r.jsx("span",{className:"text-xs font-bold text-lime",children:s+1})}),(0,r.jsxs)("div",{children:[r.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:t.title}),t.description&&r.jsx("p",{className:"text-sm text-white/50 mb-3",children:t.description}),t.code&&r.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:r.jsx("code",{className:"text-sm text-lime/90 font-mono",children:t.code})}),t.content&&r.jsx("div",{className:"text-sm text-white/70",children:t.content})]})]},s))})}function i({features:e}){return r.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,t)=>(0,r.jsxs)(o.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*t},className:"flex items-start gap-2",children:[r.jsx(l.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),r.jsx("span",{className:"text-sm text-white/70",children:e})]},t))})}function d({title:e,description:t,icon:s,href:l,children:n}){let i=l?"a":"div";return r.jsx(o.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:l?{y:-2}:void 0,children:(0,r.jsxs)(i,{...l?{href:l,className:"block"}:{},className:(0,a.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",l&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[s&&r.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:r.jsx(s,{className:"w-4 h-4 text-lime"})}),r.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),t&&r.jsx("p",{className:"text-sm text-white/50",children:t}),n]})})}function c({children:e,columns:t=2}){return r.jsx("div",{className:(0,a.cn)("grid gap-4 my-6",2===t&&"md:grid-cols-2",3===t&&"md:grid-cols-3"),children:e})}},6009:(e,t,s)=>{"use strict";s.r(t),s.d(t,{$$typeof:()=>l,__esModule:()=>o,default:()=>a});let r=(0,s(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/deployment/serverless/page.tsx`),{__esModule:o,$$typeof:l}=r,a=r.default}};var t=require("../../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),r=t.X(0,[638,498,697,224,782,883],()=>s(4425));module.exports=r})();