(()=>{var e={};e.id=905,e.ids=[905],e.modules={7849:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external")},2934:e=>{"use strict";e.exports=require("next/dist/client/components/action-async-storage.external.js")},5403:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external")},4580:e=>{"use strict";e.exports=require("next/dist/client/components/request-async-storage.external.js")},4749:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external")},5869:e=>{"use strict";e.exports=require("next/dist/client/components/static-generation-async-storage.external.js")},399:e=>{"use strict";e.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},7184:(e,t,s)=>{"use strict";s.r(t),s.d(t,{GlobalError:()=>r.a,__next_app__:()=>x,originalPathname:()=>p,pages:()=>d,routeModule:()=>m,tree:()=>c});var o=s(482),l=s(9108),i=s(2563),r=s.n(i),a=s(8300),n={};for(let e in a)0>["default","tree","pages","GlobalError","originalPathname","__next_app__","routeModule"].indexOf(e)&&(n[e]=()=>a[e]);s.d(t,n);let c=["",{children:["docs",{children:["mcp",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(s.bind(s,8626)),"/Users/redfoxhotels/zse/website/src/app/docs/mcp/page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(s.bind(s,9231)),"/Users/redfoxhotels/zse/website/src/app/docs/layout.tsx"]}]},{layout:[()=>Promise.resolve().then(s.bind(s,7633)),"/Users/redfoxhotels/zse/website/src/app/layout.tsx"],"not-found":[()=>Promise.resolve().then(s.t.bind(s,9361,23)),"next/dist/client/components/not-found-error"]}],d=["/Users/redfoxhotels/zse/website/src/app/docs/mcp/page.tsx"],p="/docs/mcp/page",x={require:s,loadChunk:()=>Promise.resolve()},m=new o.AppPageRouteModule({definition:{kind:l.x.APP_PAGE,page:"/docs/mcp/page",pathname:"/docs/mcp",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:c}})},257:(e,t,s)=>{Promise.resolve().then(s.bind(s,6163))},6163:(e,t,s)=>{"use strict";s.r(t),s.d(t,{default:()=>c});var o=s(5344),l=s(1499),i=s(196),r=s(1812),a=s(9039);let n=[{id:"overview",title:"Overview",level:2},{id:"built-in-tools",title:"Built-in Tools",level:2},{id:"using-tools",title:"Using Tools",level:2},{id:"custom-tools",title:"Custom Tools",level:2},{id:"openai-format",title:"OpenAI Format",level:2},{id:"api-reference",title:"API Reference",level:2}];function c(){return(0,o.jsxs)("div",{className:"flex",children:[(0,o.jsxs)("article",{className:"flex-1 min-w-0 py-8 px-6 lg:px-10",children:[o.jsx(l.lv,{title:"MCP Tools",description:"Model Context Protocol for extending LLM capabilities with function calling and tool execution.",badge:"Feature"}),(0,o.jsxs)(l.Je,{id:"overview",title:"Overview",children:[o.jsx("p",{className:"mb-4",children:"MCP (Model Context Protocol) enables LLMs to use external tools and functions. ZSE includes built-in tools and supports custom tool definitions compatible with OpenAI's function calling format."}),(0,o.jsxs)(a.gy,{columns:3,children:[o.jsx(a.Zb,{title:"Function Calling",description:"Parse and execute tool calls from LLM output"}),o.jsx(a.Zb,{title:"Built-in Tools",description:"Calculator, datetime, JSON parser, string ops"}),o.jsx(a.Zb,{title:"OpenAI Compatible",description:"Same format as OpenAI function calling"})]}),o.jsx(a.VS,{features:["JSON schema tool definitions","Automatic tool call parsing from LLM output","Built-in utility tools","Custom tool registration","OpenAI-compatible function calling format"]})]}),(0,o.jsxs)(l.Je,{id:"built-in-tools",title:"Built-in Tools",children:[o.jsx("p",{className:"mb-4",children:"ZSE includes several utility tools out of the box:"}),o.jsx("div",{className:"overflow-x-auto",children:(0,o.jsxs)("table",{className:"w-full text-sm",children:[o.jsx("thead",{children:(0,o.jsxs)("tr",{className:"border-b border-white/10",children:[o.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Tool"}),o.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Description"}),o.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Example Input"})]})}),(0,o.jsxs)("tbody",{children:[(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"calculator"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Math expressions (sqrt, sin, cos, log, etc.)"}),o.jsx("td",{className:"py-3 px-4 font-mono text-white text-xs",children:"sqrt(16) + 2**3"})]}),(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"datetime"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Current date/time with timezone support"}),o.jsx("td",{className:"py-3 px-4 font-mono text-white text-xs",children:"America/New_York"})]}),(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"parse_json"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Parse JSON and extract data"}),o.jsx("td",{className:"py-3 px-4 font-mono text-white text-xs",children:'{"key": "value"}'})]}),(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"string_ops"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"String operations (upper, lower, split, etc.)"}),o.jsx("td",{className:"py-3 px-4 font-mono text-white text-xs",children:"upper: hello world"})]})]})]})}),(0,o.jsxs)(l.KU,{id:"calculator-tool",title:"Calculator",children:[o.jsx(i.d,{language:"bash",code:`# Execute calculator tool
curl -X POST http://localhost:8000/api/tools/execute \\
  -H "Content-Type: application/json" \\
  -d '{
    "tool": "calculator",
    "input": "sqrt(144) + sin(3.14159/2)"
  }'

# Response:
# {"result": 13.0, "success": true}`}),o.jsx("p",{className:"mt-4 mb-2 text-sm text-gray-400",children:"Supported functions: sqrt, sin, cos, tan, log, log10, exp, abs, round, floor, ceil"})]}),o.jsx(l.KU,{id:"datetime-tool",title:"Datetime",children:o.jsx(i.d,{language:"bash",code:`# Get current time
curl -X POST http://localhost:8000/api/tools/execute \\
  -H "Content-Type: application/json" \\
  -d '{
    "tool": "datetime",
    "input": "UTC"
  }'

# Response:
# {
#   "result": {
#     "datetime": "2026-02-26T10:30:00+00:00",
#     "timezone": "UTC",
#     "unix_timestamp": 1772092200
#   },
#   "success": true
# }`})})]}),(0,o.jsxs)(l.Je,{id:"using-tools",title:"Using Tools",children:[o.jsx(l.KU,{id:"list-tools",title:"List Available Tools",children:o.jsx(i.d,{language:"bash",code:`# List all registered tools
curl http://localhost:8000/api/tools/

# Response:
# {
#   "tools": [
#     {
#       "name": "calculator",
#       "description": "Evaluate mathematical expressions",
#       "parameters": {...}
#     },
#     ...
#   ]
# }`})}),o.jsx(l.KU,{id:"execute-tool",title:"Execute Tool",children:o.jsx(i.d,{language:"python",code:`import requests

# Execute a tool directly
response = requests.post(
    "http://localhost:8000/api/tools/execute",
    json={
        "tool": "calculator",
        "input": "2 ** 10"
    }
)
print(response.json())
# {"result": 1024, "success": true}`})}),o.jsx(l.KU,{id:"parse-calls",title:"Parse Tool Calls from Text",children:o.jsx(i.d,{language:"bash",code:`# Parse tool calls from LLM output
curl -X POST http://localhost:8000/api/tools/parse \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "I need to calculate sqrt(256). Let me use the calculator tool."
  }'

# Response:
# {
#   "tool_calls": [
#     {"tool": "calculator", "input": "sqrt(256)"}
#   ]
# }`})}),o.jsx(l.KU,{id:"process-calls",title:"Parse and Execute",children:o.jsx(i.d,{language:"bash",code:`# Parse and execute tool calls in one step
curl -X POST http://localhost:8000/api/tools/process \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "What time is it in Tokyo? Use datetime tool with Asia/Tokyo timezone."
  }'

# Response:
# {
#   "results": [
#     {
#       "tool": "datetime",
#       "input": "Asia/Tokyo",
#       "result": {"datetime": "2026-02-26T19:30:00+09:00", ...},
#       "success": true
#     }
#   ]
# }`})})]}),(0,o.jsxs)(l.Je,{id:"custom-tools",title:"Custom Tools",children:[o.jsx("p",{className:"mb-4",children:"Register custom tools with JSON schema definitions:"}),o.jsx(i.d,{language:"python",code:`from zse.api.server.mcp import MCPRegistry

# Get the registry
registry = MCPRegistry()

# Define a custom tool
@registry.tool(
    name="weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or coordinates"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["location"]
    }
)
def get_weather(location: str, units: str = "celsius"):
    # Your implementation here
    return {"temperature": 22, "condition": "sunny", "units": units}

# Now the tool is available via API
# POST /api/tools/execute {"tool": "weather", "input": {"location": "Tokyo"}}`}),o.jsx(r.U,{type:"info",children:"Custom tools are persisted in the server session. For permanent tools, add them to your server startup script."})]}),(0,o.jsxs)(l.Je,{id:"openai-format",title:"OpenAI Format",children:[o.jsx("p",{className:"mb-4",children:"Get tools in OpenAI-compatible function calling format:"}),o.jsx(l.KU,{id:"get-functions",title:"Get Functions",children:o.jsx(i.d,{language:"bash",code:`# Get tools in OpenAI format
curl http://localhost:8000/api/tools/openai/functions

# Response:
# {
#   "functions": [
#     {
#       "name": "calculator",
#       "description": "Evaluate mathematical expressions",
#       "parameters": {
#         "type": "object",
#         "properties": {
#           "expression": {
#             "type": "string",
#             "description": "Math expression to evaluate"
#           }
#         },
#         "required": ["expression"]
#       }
#     }
#   ]
# }`})}),o.jsx(l.KU,{id:"chat-with-tools",title:"Chat with Tools",children:o.jsx(i.d,{language:"python",code:`import requests

# Get available tools
tools_response = requests.get(
    "http://localhost:8000/api/tools/openai/functions"
)
tools = tools_response.json()["functions"]

# Chat with tool support
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen-7b",
        "messages": [
            {"role": "user", "content": "What is 15% of 280?"}
        ],
        "tools": tools,
        "tool_choice": "auto"
    }
)

result = response.json()
# The model may return a tool call which you can execute
if result["choices"][0].get("tool_calls"):
    tool_call = result["choices"][0]["tool_calls"][0]
    # Execute the tool call...`})})]}),o.jsx(l.Je,{id:"api-reference",title:"API Reference",children:o.jsx("div",{className:"overflow-x-auto",children:(0,o.jsxs)("table",{className:"w-full text-sm",children:[o.jsx("thead",{children:(0,o.jsxs)("tr",{className:"border-b border-white/10",children:[o.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Endpoint"}),o.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Method"}),o.jsx("th",{className:"text-left py-3 px-4 font-medium text-gray-400",children:"Description"})]})}),(0,o.jsxs)("tbody",{children:[(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/tools/"}),o.jsx("td",{className:"py-3 px-4 text-white",children:"GET"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"List all available tools"})]}),(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/tools/execute"}),o.jsx("td",{className:"py-3 px-4 text-white",children:"POST"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Execute a specific tool"})]}),(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/tools/parse"}),o.jsx("td",{className:"py-3 px-4 text-white",children:"POST"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Parse tool calls from text"})]}),(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/tools/process"}),o.jsx("td",{className:"py-3 px-4 text-white",children:"POST"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Parse and execute tool calls"})]}),(0,o.jsxs)("tr",{className:"border-b border-white/5",children:[o.jsx("td",{className:"py-3 px-4 font-mono text-lime",children:"/api/tools/openai/functions"}),o.jsx("td",{className:"py-3 px-4 text-white",children:"GET"}),o.jsx("td",{className:"py-3 px-4 text-gray-400",children:"Get OpenAI-compatible format"})]})]})]})})}),o.jsx(l.KO,{prev:{href:"/docs/rag",title:"RAG Module"},next:{href:"/docs/api/cli",title:"CLI Commands"}})]}),o.jsx(l.o5,{items:n})]})}},9039:(e,t,s)=>{"use strict";s.d(t,{Rg:()=>a,VS:()=>n,Zb:()=>c,gy:()=>d});var o=s(5344),l=s(1912),i=s(2312),r=s(1453);function a({steps:e}){return o.jsx("div",{className:"my-6 space-y-0",children:e.map((t,s)=>(0,o.jsxs)(l.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},transition:{delay:.1*s},className:"relative pl-8 pb-8 last:pb-0",children:[s<e.length-1&&o.jsx("div",{className:"absolute left-[11px] top-6 bottom-0 w-px bg-white/10"}),o.jsx("div",{className:"absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center",children:o.jsx("span",{className:"text-xs font-bold text-lime",children:s+1})}),(0,o.jsxs)("div",{children:[o.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:t.title}),t.description&&o.jsx("p",{className:"text-sm text-white/50 mb-3",children:t.description}),t.code&&o.jsx("pre",{className:"bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2",children:o.jsx("code",{className:"text-sm text-lime/90 font-mono",children:t.code})}),t.content&&o.jsx("div",{className:"text-sm text-white/70",children:t.content})]})]},s))})}function n({features:e}){return o.jsx("ul",{className:"my-4 space-y-2",children:e.map((e,t)=>(0,o.jsxs)(l.E.li,{initial:{opacity:0,x:-10},animate:{opacity:1,x:0},transition:{delay:.05*t},className:"flex items-start gap-2",children:[o.jsx(i.Z,{className:"w-4 h-4 text-lime mt-0.5 flex-shrink-0"}),o.jsx("span",{className:"text-sm text-white/70",children:e})]},t))})}function c({title:e,description:t,icon:s,href:i,children:a}){let n=i?"a":"div";return o.jsx(l.E.div,{initial:{opacity:0,y:10},animate:{opacity:1,y:0},whileHover:i?{y:-2}:void 0,children:(0,o.jsxs)(n,{...i?{href:i,className:"block"}:{},className:(0,r.cn)("p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",i&&"hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"),children:[s&&o.jsx("div",{className:"w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3",children:o.jsx(s,{className:"w-4 h-4 text-lime"})}),o.jsx("h4",{className:"text-base font-semibold text-white mb-1",children:e}),t&&o.jsx("p",{className:"text-sm text-white/50",children:t}),a]})})}function d({children:e,columns:t=2}){return o.jsx("div",{className:(0,r.cn)("grid gap-4 my-6",2===t&&"md:grid-cols-2",3===t&&"md:grid-cols-3"),children:e})}},8626:(e,t,s)=>{"use strict";s.r(t),s.d(t,{$$typeof:()=>i,__esModule:()=>l,default:()=>r});let o=(0,s(6843).createProxy)(String.raw`/Users/redfoxhotels/zse/website/src/app/docs/mcp/page.tsx`),{__esModule:l,$$typeof:i}=o,r=o.default}};var t=require("../../../webpack-runtime.js");t.C(e);var s=e=>t(t.s=e),o=t.X(0,[638,498,697,224,782,883],()=>s(7184));module.exports=o})();