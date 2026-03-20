'use client'

import { motion } from 'framer-motion'
import { Terminal, Zap, Shield, DollarSign, Wifi, Server } from 'lucide-react'

export default function OpenClawIntegration() {
  return (
    <div className="max-w-4xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-4xl font-bold mb-4">OpenClaw Integration</h1>
        <p className="text-xl text-gray-400 mb-8">
          Run any local model with OpenClaw using ZSE as your inference engine.
        </p>

        {/* Overview */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4">Why ZSE + OpenClaw?</h2>
          <p className="text-gray-300 mb-6">
            <a href="https://openclaw.ai" className="text-blue-400 hover:underline">OpenClaw</a> is 
            a powerful 24/7 AI assistant by @steipete. By default it uses Claude API, but you can 
            run it with <strong>any local model</strong> using ZSE - completely private, zero API costs.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h3 className="font-semibold text-red-400 mb-2">❌ Claude API</h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• $15-20/month API costs</li>
                <li>• Data sent to external servers</li>
                <li>• Rate limits apply</li>
                <li>• Requires internet</li>
                <li>• Locked to Claude models</li>
              </ul>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 border border-green-700">
              <h3 className="font-semibold text-green-400 mb-2">✅ ZSE Local</h3>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Zero ongoing costs</li>
                <li>• 100% private, runs locally</li>
                <li>• No rate limits</li>
                <li>• Works offline</li>
                <li>• Run ANY model you want</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Quick Start */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <Zap className="w-6 h-6 text-yellow-400" />
            Quick Start
          </h2>
          
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-700 mb-6">
            <p className="text-gray-400 mb-3">One command to set everything up:</p>
            <pre className="bg-black rounded p-4 overflow-x-auto">
              <code className="text-green-400">curl -fsSL https://zllm.in/openclaw.sh | bash</code>
            </pre>
          </div>
        </section>

        {/* Manual Setup */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <Terminal className="w-6 h-6 text-blue-400" />
            Manual Setup
          </h2>

          <div className="space-y-6">
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h3 className="font-semibold mb-2">Step 1: Install ZSE</h3>
              <pre className="bg-black rounded p-3 overflow-x-auto">
                <code className="text-gray-300">pip install zllm-zse</code>
              </pre>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h3 className="font-semibold mb-2">Step 2: Start ZSE Server</h3>
              <pre className="bg-black rounded p-3 overflow-x-auto">
                <code className="text-gray-300">zse serve &lt;model-name&gt; --port 8000</code>
              </pre>
              <p className="text-sm text-gray-500 mt-2">
                Replace &lt;model-name&gt; with any HuggingFace model (e.g., Qwen/Qwen2.5-7B-Instruct)
              </p>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h3 className="font-semibold mb-2">Step 3: Configure OpenClaw</h3>
              <p className="text-gray-400 mb-3">Add to your OpenClaw config.yaml:</p>
              <pre className="bg-black rounded p-3 overflow-x-auto">
                <code className="text-gray-300">{`llm:
  provider: openai-compatible
  api_base: http://localhost:8000/v1
  api_key: zse
  model: default`}</code>
              </pre>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h3 className="font-semibold mb-2">Step 4: Or Use Environment Variables</h3>
              <pre className="bg-black rounded p-3 overflow-x-auto">
                <code className="text-gray-300">{`export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=zse`}</code>
              </pre>
            </div>
          </div>
        </section>

        {/* Recommended Models */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
            <Server className="w-6 h-6 text-purple-400" />
            Recommended Models
          </h2>
          
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4">GPU VRAM</th>
                  <th className="text-left py-3 px-4">Model</th>
                  <th className="text-left py-3 px-4">Speed</th>
                </tr>
              </thead>
              <tbody className="text-gray-400">
                <tr className="border-b border-gray-800">
                  <td className="py-3 px-4">8GB</td>
                  <td className="py-3 px-4 font-mono text-xs">Qwen/Qwen2.5-7B-Instruct</td>
                  <td className="py-3 px-4">~50-60 tok/s</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-3 px-4">12-16GB</td>
                  <td className="py-3 px-4 font-mono text-xs">Qwen/Qwen2.5-14B-Instruct</td>
                  <td className="py-3 px-4">~35-45 tok/s</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-3 px-4">24GB</td>
                  <td className="py-3 px-4 font-mono text-xs">Qwen/Qwen2.5-32B-Instruct</td>
                  <td className="py-3 px-4">~25-30 tok/s</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-3 px-4">48GB+</td>
                  <td className="py-3 px-4 font-mono text-xs">Qwen/Qwen2.5-72B-Instruct</td>
                  <td className="py-3 px-4">~15-20 tok/s</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Benefits */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4">Benefits</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 text-center">
              <Shield className="w-8 h-8 text-green-400 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">100% Private</h3>
              <p className="text-sm text-gray-400">All data stays on your machine</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 text-center">
              <DollarSign className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">Zero API Costs</h3>
              <p className="text-sm text-gray-400">No monthly fees or usage limits</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 text-center">
              <Wifi className="w-8 h-8 text-blue-400 mx-auto mb-2" />
              <h3 className="font-semibold mb-1">Works Offline</h3>
              <p className="text-sm text-gray-400">No internet required after setup</p>
            </div>
          </div>
        </section>

        {/* Troubleshooting */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold mb-4">Troubleshooting</h2>
          
          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h3 className="font-semibold text-yellow-400 mb-2">OpenClaw can't connect?</h3>
              <p className="text-gray-400 text-sm">
                Make sure ZSE server is running on port 8000. Check with: <code className="bg-black px-2 py-1 rounded">curl http://localhost:8000/health</code>
              </p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h3 className="font-semibold text-yellow-400 mb-2">Out of memory?</h3>
              <p className="text-gray-400 text-sm">
                Try a smaller model. 7B models fit in 8GB VRAM, 32B needs 24GB.
              </p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <h3 className="font-semibold text-yellow-400 mb-2">Slow responses?</h3>
              <p className="text-gray-400 text-sm">
                Make sure you have a CUDA GPU. CPU inference is much slower.
              </p>
            </div>
          </div>
        </section>

      </motion.div>
    </div>
  )
}
