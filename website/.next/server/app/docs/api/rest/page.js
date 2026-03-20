(()=>{var e={};e.id=55,e.ids=[55],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},6319:(e,s,t)=>{"use strict";t.r(s),t.d(s,{GlobalError:()=>i.a,__next_app__:()=>p,originalPathname:()=>h,pages:()=>o,routeModule:()=>m,tree:()=>d});var r=t(482),a=t(9108),l=t(2563),i=t.n(l),n=t(8300),c={};for(let e in n)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(c[e]=()=>n[e]);t.d(s,c);let d=["",{children:["docs",{children:["api",{children:["rest",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(t.bind(t,3184)),"/Users/redfoxhotels/zse/website/src/app/docs/api/rest/page.tsx"]}]},{}]},{}]},{layout:[()=>Promise.resolve().then(t.bind(t,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(t.bind(t,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(t.t.bind(t,9361,23)),"next/dist/client/components/not-found-error"]}],o=["/Users/redfoxhotels/zse/website/src/app/docs/api/rest/page.tsx"],h="/docs/api/rest/page",p={require:t,loadChunk:()=>Promise.resolve()},m=new r.AppPageRouteModule({definition:{kind:a.x.APP_PAGE,page:"/docs/api/rest/page",pathname:"/docs/api/rest",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:d}})},2495:(e,s,t)=>{Promise.resolve().then(t.bind(t,8126))},8126:(e,s,t)=>{"use strict";t.r(s),t.d(s,{default:()=>c});var r=t(5344),a=t(1499),l=t(196),i=t(1812);let n=[{id:"overview",title:"Overview",level:2},{id:"authentication",title:"Authentication",level:2},{id:"chat-completions",title:"Chat Completions",level:2},{id:"streaming",title:"Streaming",level:3},{id:"models",title:"Models",level:2},{id:"health",title:"Health Check",level:2},{id:"errors",title:"Error Handling",level:2}];function c(){return(0,r.jsxs)("div",{className:"flex",children:[(0,r.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[r.jsx(a.lv,{title:"REST API",description:"OpenAI-compatible REST API reference. Use any OpenAI SDK or make direct HTTP requests.",badge:"API Reference"}),(0,r.jsxs)(a.Je,{id:"overview",title:"Overview",children:[r.jsx("p",{className:"mb-4",children:"ZSE provides an OpenAI-compatible REST API. This means you can use any OpenAI SDK or existing code with minimal changes."}),r.jsx(l.d,{language:"text",code:`Base URL: http://localhost:8000/v1

Endpoints:
  POST /v1/chat/completions  - Generate chat completions
  GET  /v1/models            - List available models
  GET  /health               - Health check`}),(0,r.jsxs)(i.U,{type:"tip",children:["If you're using the OpenAI Python SDK, just change the ",r.jsx(l.Z,{children:"base_url"})," parameter to point to your ZSE server."]})]}),(0,r.jsxs)(a.Je,{id:"authentication",title:"Authentication",children:[(0,r.jsxs)("p",{className:"mb-4",children:["By default, ZSE does not require authentication. If you start the server with",r.jsx(l.Z,{children:"--api-key"}),", all requests must include the key."]}),r.jsx(l.d,{language:"bash",code:`# Start server with API key
zse serve model.zse --api-key sk-your-secret-key`}),(0,r.jsxs)("p",{className:"mt-4 mb-2",children:["Include the API key in the ",r.jsx(l.Z,{children:"Authorization"})," header:"]}),r.jsx(l.d,{language:"bash",code:`curl http://localhost:8000/v1/chat/completions \\
  -H "Authorization: Bearer sk-your-secret-key" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello"}]}'`})]}),(0,r.jsxs)(a.Je,{id:"chat-completions",title:"Chat Completions",children:[(0,r.jsxs)("p",{className:"mb-4",children:["Generate chat completions with the ",r.jsx(l.Z,{children:"/v1/chat/completions"})," endpoint."]}),r.jsx(l.d,{language:"bash",filename:"POST /v1/chat/completions",code:`curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'`}),r.jsx("p",{className:"mt-4 mb-2",children:r.jsx("strong",{className:"text-white",children:"Request Parameters:"})}),r.jsx("div",{className:"overflow-x-auto my-4",children:(0,r.jsxs)("table",{className:"w-full text-sm",children:[r.jsx("thead",{children:(0,r.jsxs)("tr",{className:"border-b border-white/10",children:[r.jsx("th",{className:"text-left py-2 pr-4 text-white/50 font-medium",children:"Parameter"}),r.jsx("th",{className:"text-left py-2 px-4 text-white/50 font-medium",children:"Type"}),r.jsx("th",{className:"text-left py-2 px-4 text-white/50 font-medium",children:"Required"}),r.jsx("th",{className:"text-left py-2 pl-4 text-white/50 font-medium",children:"Description"})]})}),(0,r.jsxs)("tbody",{className:"text-white/70",children:[(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(l.Z,{children:"model"})}),r.jsx("td",{className:"py-2 px-4",children:"string"}),r.jsx("td",{className:"py-2 px-4",children:"Yes"}),r.jsx("td",{className:"py-2 pl-4",children:'Model ID (use "default" for single model)'})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(l.Z,{children:"messages"})}),r.jsx("td",{className:"py-2 px-4",children:"array"}),r.jsx("td",{className:"py-2 px-4",children:"Yes"}),r.jsx("td",{className:"py-2 pl-4",children:"Array of message objects"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(l.Z,{children:"temperature"})}),r.jsx("td",{className:"py-2 px-4",children:"float"}),r.jsx("td",{className:"py-2 px-4",children:"No"}),r.jsx("td",{className:"py-2 pl-4",children:"Sampling temperature (0-2, default: 1.0)"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(l.Z,{children:"max_tokens"})}),r.jsx("td",{className:"py-2 px-4",children:"int"}),r.jsx("td",{className:"py-2 px-4",children:"No"}),r.jsx("td",{className:"py-2 pl-4",children:"Maximum tokens to generate"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(l.Z,{children:"stream"})}),r.jsx("td",{className:"py-2 px-4",children:"bool"}),r.jsx("td",{className:"py-2 px-4",children:"No"}),r.jsx("td",{className:"py-2 pl-4",children:"Enable streaming response"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(l.Z,{children:"top_p"})}),r.jsx("td",{className:"py-2 px-4",children:"float"}),r.jsx("td",{className:"py-2 px-4",children:"No"}),r.jsx("td",{className:"py-2 pl-4",children:"Nucleus sampling (0-1, default: 1.0)"})]}),(0,r.jsxs)("tr",{children:[r.jsx("td",{className:"py-2 pr-4",children:r.jsx(l.Z,{children:"stop"})}),r.jsx("td",{className:"py-2 px-4",children:"array"}),r.jsx("td",{className:"py-2 px-4",children:"No"}),r.jsx("td",{className:"py-2 pl-4",children:"Stop sequences"})]})]})]})}),r.jsx("p",{className:"mt-4 mb-2",children:r.jsx("strong",{className:"text-white",children:"Response:"})}),r.jsx(l.d,{language:"json",filename:"Response",code:`{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1709147520,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}`}),(0,r.jsxs)(a.KU,{id:"streaming",title:"Streaming",children:[r.jsx("p",{className:"mb-4",children:"Enable streaming to receive tokens as they're generated:"}),r.jsx(l.d,{language:"bash",code:`curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'`}),r.jsx("p",{className:"mt-4 mb-2",children:"Streaming response (Server-Sent Events):"}),r.jsx(l.d,{language:"text",code:`data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"1"},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":", "},"index":0}]}

data: {"id":"chatcmpl-abc123","choices":[{"delta":{"content":"2"},"index":0}]}

...

data: [DONE]`})]})]}),(0,r.jsxs)(a.Je,{id:"models",title:"Models",children:[(0,r.jsxs)("p",{className:"mb-4",children:["List available models with the ",r.jsx(l.Z,{children:"/v1/models"})," endpoint."]}),r.jsx(l.d,{language:"bash",filename:"GET /v1/models",code:"curl http://localhost:8000/v1/models"}),r.jsx(l.d,{language:"json",filename:"Response",code:`{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-7B-Instruct",
      "object": "model",
      "created": 1709147520,
      "owned_by": "zse"
    }
  ]
}`})]}),(0,r.jsxs)(a.Je,{id:"health",title:"Health Check",children:[(0,r.jsxs)("p",{className:"mb-4",children:["Check server health with the ",r.jsx(l.Z,{children:"/health"})," endpoint."]}),r.jsx(l.d,{language:"bash",filename:"GET /health",code:"curl http://localhost:8000/health"}),r.jsx(l.d,{language:"json",filename:"Response",code:`{
  "status": "healthy",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "gpu_memory_used": "5.2 GB",
  "gpu_memory_total": "80 GB",
  "uptime": "2h 15m"
}`})]}),(0,r.jsxs)(a.Je,{id:"errors",title:"Error Handling",children:[r.jsx("p",{className:"mb-4",children:"ZSE returns standard HTTP status codes and JSON error responses:"}),r.jsx("div",{className:"overflow-x-auto my-4",children:(0,r.jsxs)("table",{className:"w-full text-sm",children:[r.jsx("thead",{children:(0,r.jsxs)("tr",{className:"border-b border-white/10",children:[r.jsx("th",{className:"text-left py-2 pr-4 text-white/50 font-medium",children:"Status"}),r.jsx("th",{className:"text-left py-2 pl-4 text-white/50 font-medium",children:"Description"})]})}),(0,r.jsxs)("tbody",{className:"text-white/70",children:[(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"200"}),r.jsx("td",{className:"py-2 pl-4",children:"Success"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"400"}),r.jsx("td",{className:"py-2 pl-4",children:"Bad request (invalid parameters)"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"401"}),r.jsx("td",{className:"py-2 pl-4",children:"Unauthorized (invalid API key)"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"404"}),r.jsx("td",{className:"py-2 pl-4",children:"Model not found"})]}),(0,r.jsxs)("tr",{className:"border-b border-white/[0.05]",children:[r.jsx("td",{className:"py-2 pr-4",children:"429"}),r.jsx("td",{className:"py-2 pl-4",children:"Rate limited"})]}),(0,r.jsxs)("tr",{children:[r.jsx("td",{className:"py-2 pr-4",children:"500"}),r.jsx("td",{className:"py-2 pl-4",children:"Internal server error"})]})]})]})}),r.jsx(l.d,{language:"json",filename:"Error Response",code:`{
  "error": {
    "message": "Invalid request: 'messages' is required",
    "type": "invalid_request_error",
    "code": "missing_required_field"
  }
}`})]}),r.jsx(a.KO,{prev:{title:"Python API",href:"/docs/api/python"},next:{title:"Configuration",href:"/docs/api/config"}})]}),r.jsx(a.o5,{items:n})]})}},3184:(e,s,t)=>{"use strict";t.r(s),t.d(s,{$$typeof:()=>l,__esModule:()=>a,default:()=>i});let r=(0,t(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/api/rest/page.tsx`),{__esModule:a,$$typeof:l}=r,i=r.default}};var s=require("../../../../webpack-runtime.js");s.C(e);var t=e=>s(s.s=e),r=s.X(0,[638,498,697,224,782,883],()=>t(6319));module.exports=r})();