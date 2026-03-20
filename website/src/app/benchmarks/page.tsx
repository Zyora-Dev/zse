'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import { useState } from 'react'

type TabType = 'cold-start' | 'memory' | 'throughput' | 'comparison'

export default function BenchmarksPage() {
  const [activeTab, setActiveTab] = useState<TabType>('cold-start')

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <section className="relative border-b border-white/10 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-lime/5 via-transparent to-transparent" />
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-16 relative">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center"
          >
            <h1 className="text-4xl sm:text-5xl font-bold text-white">
              Benchmarks
            </h1>
            <p className="mt-4 text-xl text-gray-400 max-w-2xl mx-auto">
              Real measurements on NVIDIA H200. No inflated claims, just verified data.
            </p>
            <p className="mt-2 text-sm text-gray-500">
              All tests conducted February 2026 on io.net H200 infrastructure. Results verified on v1.2.0.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Tab Navigation */}
      <section className="border-b border-white/10 sticky top-[64px] bg-black/90 backdrop-blur-sm z-10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex gap-1 overflow-x-auto py-4">
            {[
              { id: 'cold-start', label: 'Cold Start' },
              { id: 'memory', label: 'Memory Usage' },
              { id: 'throughput', label: 'Throughput' },
              { id: 'comparison', label: 'vs Competitors' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as TabType)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'bg-lime/10 text-lime'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Content */}
      <section className="py-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          {activeTab === 'cold-start' && <ColdStartBenchmarks />}
          {activeTab === 'memory' && <MemoryBenchmarks />}
          {activeTab === 'throughput' && <ThroughputBenchmarks />}
          {activeTab === 'comparison' && <ComparisonBenchmarks />}
        </div>
      </section>

      {/* Methodology */}
      <section className="py-16 border-t border-white/10 bg-white/[0.01]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-2xl font-bold text-white mb-8">Methodology</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="p-6 rounded-xl bg-white/[0.02] border border-white/5">
              <h3 className="font-semibold text-white mb-3">Hardware</h3>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>• NVIDIA H200 141GB (Primary benchmark GPU)</li>
                <li>• NVIDIA A100-80GB (Secondary)</li>
                <li>• CPU: AMD EPYC / Intel Xeon</li>
                <li>• CUDA 12.1+, PyTorch 2.1+</li>
              </ul>
            </div>
            <div className="p-6 rounded-xl bg-white/[0.02] border border-white/5">
              <h3 className="font-semibold text-white mb-3">Test Conditions</h3>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>• Cold start: Fresh process, no cached weights</li>
                <li>• Warm cache: Model weights on disk, GPU free</li>
                <li>• Memory: PyTorch memory profiler</li>
                <li>• Throughput: Average of 5 runs, 256 output tokens</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 border-t border-white/10">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Run Your Own Benchmarks</h2>
          <p className="text-gray-400 mb-8 max-w-xl mx-auto">
            Reproduce these results on your hardware with our benchmark suite.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              href="/docs/advanced/benchmarking"
              className="px-6 py-3 bg-lime text-black font-medium rounded-lg hover:bg-lime/90 transition-colors"
            >
              Benchmarking Guide
            </Link>
            <Link
              href="/docs/quickstart"
              className="px-6 py-3 bg-white/5 text-white font-medium rounded-lg hover:bg-white/10 transition-colors"
            >
              Get Started
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}

function ColdStartBenchmarks() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Cold Start Performance</h2>
        <p className="text-gray-400 mb-8">
          Time from process start to first token generation. Critical for serverless and auto-scaling.
        </p>

        {/* 7B Model */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-white mb-4">Qwen 2.5 7B Instruct (H200)</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Method</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Load Time</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">VRAM</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Throughput</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">.zse v1.2.0 (bnb.matmul_4bit)</td>
                  <td className="py-3 px-4 text-right text-lime font-mono font-bold">9.1s</td>
                  <td className="py-3 px-4 text-right text-lime font-mono">5.9 GB</td>
                  <td className="py-3 px-4 text-right text-lime font-bold">58.7 tok/s</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Key Stats */}
          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="p-4 rounded-lg bg-lime/5 border border-lime/20 text-center">
              <div className="text-2xl font-bold text-lime">5.57 GB</div>
              <div className="text-xs text-gray-400 mt-1">File Size</div>
            </div>
            <div className="p-4 rounded-lg bg-lime/5 border border-lime/20 text-center">
              <div className="text-2xl font-bold text-lime">5.9 GB</div>
              <div className="text-xs text-gray-400 mt-1">VRAM Usage</div>
            </div>
            <div className="p-4 rounded-lg bg-lime/5 border border-lime/20 text-center">
              <div className="text-2xl font-bold text-lime">58.7</div>
              <div className="text-xs text-gray-400 mt-1">tok/s</div>
            </div>
          </div>
        </div>

        {/* 32B Model */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">Qwen 2.5 32B Instruct (H200)</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Method</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Load Time</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">VRAM</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Throughput</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">.zse v1.2.0 (bnb.matmul_4bit)</td>
                  <td className="py-3 px-4 text-right text-lime font-mono font-bold">24.1s</td>
                  <td className="py-3 px-4 text-right text-lime font-mono">20.9 GB</td>
                  <td className="py-3 px-4 text-right text-lime font-bold">26.9 tok/s</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Key Stats */}
          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="p-4 rounded-lg bg-lime/5 border border-lime/20 text-center">
              <div className="text-2xl font-bold text-lime">19.23 GB</div>
              <div className="text-xs text-gray-400 mt-1">File Size</div>
            </div>
            <div className="p-4 rounded-lg bg-lime/5 border border-lime/20 text-center">
              <div className="text-2xl font-bold text-lime">20.9 GB</div>
              <div className="text-xs text-gray-400 mt-1">VRAM Usage</div>
            </div>
            <div className="p-4 rounded-lg bg-lime/5 border border-lime/20 text-center">
              <div className="text-2xl font-bold text-lime">26.9</div>
              <div className="text-xs text-gray-400 mt-1">tok/s</div>
            </div>
          </div>

          <div className="mt-6 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
            <p className="text-sm text-green-200">
              <strong>✓ Fits 24GB GPUs!</strong> Run 32B models on consumer RTX 3090/4090 cards.
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function MemoryBenchmarks() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Memory Efficiency</h2>
        <p className="text-gray-400 mb-8">
          VRAM usage with .zse INT4 format + bnb.matmul_4bit. Lower is better for constrained hardware.
        </p>

        {/* 7B Memory */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-white mb-4">Qwen 2.5 7B Instruct</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Metric</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Value</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">File Size</td>
                  <td className="py-3 px-4 text-right text-lime font-mono">5.57 GB</td>
                  <td className="py-3 px-4 text-gray-500 text-xs">Single .zse file</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">VRAM Usage</td>
                  <td className="py-3 px-4 text-right text-lime font-mono">5.9 GB</td>
                  <td className="py-3 px-4 text-gray-500 text-xs">Runtime with bnb.matmul_4bit</td>
                </tr>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">Throughput</td>
                  <td className="py-3 px-4 text-right text-lime font-mono font-bold">58.7 tok/s</td>
                  <td className="py-3 px-4 text-gray-400 text-xs">CUDA kernel inference</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* 32B Memory */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-white mb-4">Qwen 2.5 32B Instruct</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Metric</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Value</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">File Size</td>
                  <td className="py-3 px-4 text-right text-lime font-mono">19.23 GB</td>
                  <td className="py-3 px-4 text-gray-500 text-xs">Single .zse file</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">VRAM Usage</td>
                  <td className="py-3 px-4 text-right text-lime font-mono">20.9 GB</td>
                  <td className="py-3 px-4 text-gray-500 text-xs">Fits on 24GB GPUs!</td>
                </tr>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">Throughput</td>
                  <td className="py-3 px-4 text-right text-lime font-mono font-bold">26.9 tok/s</td>
                  <td className="py-3 px-4 text-gray-400 text-xs">CUDA kernel inference</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* GPU Recommendations */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">GPU Compatibility</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Your GPU</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Model Size</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">VRAM</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Speed</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">8GB (RTX 3070/4070)</td>
                  <td className="py-3 px-4 text-lime">7B INT4</td>
                  <td className="py-3 px-4 text-right text-gray-400">5.9 GB</td>
                  <td className="py-3 px-4 text-right text-gray-400">~50 tok/s</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">12GB (RTX 3080/4070Ti)</td>
                  <td className="py-3 px-4 text-lime">7B INT4</td>
                  <td className="py-3 px-4 text-right text-gray-400">5.9 GB</td>
                  <td className="py-3 px-4 text-right text-gray-400">~55 tok/s</td>
                </tr>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">24GB (RTX 3090/4090)</td>
                  <td className="py-3 px-4 text-lime font-medium">32B INT4</td>
                  <td className="py-3 px-4 text-right text-lime">20.9 GB</td>
                  <td className="py-3 px-4 text-right text-lime">~25 tok/s</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function ThroughputBenchmarks() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">Throughput</h2>
        <p className="text-gray-400 mb-8">
          Tokens per second using bnb.matmul_4bit CUDA kernels. Verified on H200.
        </p>

        {/* Main Throughput Table */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-white mb-4">ZSE v1.2.0 Performance</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Model</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Throughput</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">VRAM</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Load Time</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">Qwen 2.5 7B</td>
                  <td className="py-3 px-4 text-right text-lime font-mono font-bold">58.7 tok/s</td>
                  <td className="py-3 px-4 text-right text-gray-400">5.9 GB</td>
                  <td className="py-3 px-4 text-right text-gray-400">9.1s</td>
                </tr>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">Qwen 2.5 32B</td>
                  <td className="py-3 px-4 text-right text-lime font-mono font-bold">26.9 tok/s</td>
                  <td className="py-3 px-4 text-right text-gray-400">20.9 GB</td>
                  <td className="py-3 px-4 text-right text-gray-400">24.1s</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Visual Comparison */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-white mb-4">Throughput Visualization</h3>
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <span className="text-xs text-lime w-24 font-medium">7B</span>
              <div className="flex-1 h-8 bg-white/5 rounded overflow-hidden">
                <div className="h-full bg-lime rounded flex items-center justify-end pr-3" style={{ width: '100%' }}>
                  <span className="text-xs text-black font-bold">58.7 tok/s</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-xs text-lime w-24 font-medium">32B</span>
              <div className="flex-1 h-8 bg-white/5 rounded overflow-hidden">
                <div className="h-full bg-lime/70 rounded flex items-center justify-end pr-3" style={{ width: '46%' }}>
                  <span className="text-xs text-black font-bold">26.9 tok/s</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Key Innovation */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">Why is it fast?</h3>
          <div className="p-4 rounded-lg bg-lime/5 border border-lime/20">
            <p className="text-sm text-gray-300 mb-3">
              <strong className="text-lime">bnb.matmul_4bit CUDA Kernel:</strong> ZSE v1.2.0 uses 
              bitsandbytes' optimized CUDA kernel for INT4 matrix multiplication.
            </p>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• 0.018ms per 1024×1024 matmul operation</li>
              <li>• Weights stay packed in INT4 format</li>
              <li>• Dequantization happens on-the-fly in CUDA</li>
              <li>• No Python overhead during inference</li>
            </ul>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function ComparisonBenchmarks() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-12"
    >
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">vs llama.cpp (Verified)</h2>
        <p className="text-gray-400 mb-8">
          Head-to-head cold start benchmark. Both loading pre-quantized 7B model weights from disk to GPU.
        </p>

        {/* Test Methodology Box */}
        <div className="mb-8 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <h4 className="text-sm font-semibold text-blue-300 mb-2">Test Methodology</h4>
          <ul className="text-sm text-blue-200/80 space-y-1">
            <li>• Hardware: NVIDIA A100 80GB PCIe</li>
            <li>• Model: Qwen 2.5 Coder 7B equivalent structure</li>
            <li>• llama.cpp: GGUF Q4_K_M format (4.68 GB), n_gpu_layers=-1</li>
            <li>• ZSE: safetensors INT4 format (5.73 GB), direct GPU loading</li>
            <li>• Metric: Time from file open to model ready on GPU</li>
            <li>• Date: February 26, 2026</li>
          </ul>
        </div>

        {/* Main Comparison - llama.cpp */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-white mb-4">Cold Start: Pre-quantized Weight Loading</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Method</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Cold Start</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">File Size</th>
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">ZSE (safe_open→cuda)</td>
                  <td className="py-3 px-4 text-right text-lime font-mono font-bold">1.14s</td>
                  <td className="py-3 px-4 text-right text-gray-400">5.73 GB</td>
                  <td className="py-3 px-4 text-gray-400 text-xs">Direct GPU loading via safetensors</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">llama.cpp (GGUF Q4_K_M)</td>
                  <td className="py-3 px-4 text-right text-gray-400 font-mono">1.20s</td>
                  <td className="py-3 px-4 text-right text-gray-400">4.68 GB</td>
                  <td className="py-3 px-4 text-gray-400 text-xs">C++ mmap, full GPU offload</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">ZSE (load_file)</td>
                  <td className="py-3 px-4 text-right text-gray-400 font-mono">1.37s</td>
                  <td className="py-3 px-4 text-right text-gray-400">5.73 GB</td>
                  <td className="py-3 px-4 text-gray-400 text-xs">Single-call loading</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Visual Comparison */}
          <div className="mt-8 space-y-4">
            <div className="flex items-center gap-4">
              <span className="text-xs text-lime w-36 font-medium">ZSE (safe_open)</span>
              <div className="flex-1 h-8 bg-white/5 rounded overflow-hidden">
                <div className="h-full bg-lime rounded flex items-center justify-end pr-2" style={{ width: '83%' }}>
                  <span className="text-xs text-black font-bold">1.14s</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-xs text-gray-400 w-36">llama.cpp (GGUF)</span>
              <div className="flex-1 h-8 bg-white/5 rounded overflow-hidden">
                <div className="h-full bg-gray-500 rounded flex items-center justify-end pr-2" style={{ width: '88%' }}>
                  <span className="text-xs text-white">1.20s</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-xs text-gray-400 w-36">ZSE (load_file)</span>
              <div className="flex-1 h-8 bg-white/5 rounded overflow-hidden">
                <div className="h-full bg-gray-600 rounded flex items-center justify-end pr-2" style={{ width: '100%' }}>
                  <span className="text-xs text-white">1.37s</span>
                </div>
              </div>
            </div>
          </div>

          {/* Honest Assessment */}
          <div className="mt-6 p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
            <p className="text-sm text-yellow-200">
              <strong>Honest Assessment:</strong> ZSE and llama.cpp perform within 5% of each other for cold start loading 
              of pre-quantized weights. The 0.06s difference (1.14s vs 1.20s) is within measurement variance. 
              Both solutions achieve excellent cold start performance on modern GPUs.
            </p>
          </div>
        </div>

        {/* vs bitsandbytes - This IS a real advantage */}
        <div className="mb-12">
          <h3 className="text-lg font-semibold text-white mb-4">vs On-the-fly Quantization (bitsandbytes)</h3>
          <p className="text-gray-400 mb-4 text-sm">
            Pre-quantized formats (ZSE, GGUF) have a significant advantage over on-the-fly quantization.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-3 px-4 font-medium text-gray-400">Method</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">Cold Start</th>
                  <th className="text-right py-3 px-4 font-medium text-gray-400">vs ZSE</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-white/5 bg-lime/5">
                  <td className="py-3 px-4 text-lime font-medium">ZSE (.zse pre-quantized)</td>
                  <td className="py-3 px-4 text-right text-lime font-mono font-bold">1.14s</td>
                  <td className="py-3 px-4 text-right text-gray-500">—</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">bitsandbytes NF4 (warm cache)</td>
                  <td className="py-3 px-4 text-right text-gray-400">45.4s</td>
                  <td className="py-3 px-4 text-right text-lime">40× slower</td>
                </tr>
                <tr className="border-b border-white/5">
                  <td className="py-3 px-4 text-white">bitsandbytes NF4 (first run)</td>
                  <td className="py-3 px-4 text-right text-gray-400">216.7s</td>
                  <td className="py-3 px-4 text-right text-lime">190× slower</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs text-gray-500">
            bitsandbytes must download FP16 weights and quantize on every cold start. Pre-quantized formats skip this entirely.
          </p>
        </div>

        {/* When to Use What */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-4">When to Use What</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-4 rounded-lg bg-lime/5 border border-lime/20">
              <h4 className="font-semibold text-lime mb-2">Use ZSE when:</h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• You want Python-native integration</li>
                <li>• You need OpenAI-compatible API</li>
                <li>• Your workflow involves PyTorch/transformers</li>
                <li>• You want automated quantization + serving</li>
              </ul>
            </div>
            <div className="p-4 rounded-lg bg-white/[0.02] border border-white/10">
              <h4 className="font-semibold text-white mb-2">Use llama.cpp when:</h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• You need C/C++ integration</li>
                <li>• You want CPU inference fallback</li>
                <li>• You prefer GGUF ecosystem</li>
                <li>• You need minimal dependencies</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-8 p-4 rounded-lg bg-white/[0.02] border border-white/5">
          <p className="text-sm text-gray-400">
            <strong className="text-white">Transparency:</strong> We only publish benchmarks we have actually run. 
            The llama.cpp comparison above was conducted on Modal's A100-80GB infrastructure on February 26, 2026.
            Test code is available in our repository at <code className="text-lime">/tests/modal/test_coldstart_fair.py</code>.
          </p>
        </div>
      </div>
    </motion.div>
  )
}
