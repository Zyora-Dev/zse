'use client'

import Link from 'next/link'
import { motion, useScroll, useTransform, useInView, AnimatePresence } from 'framer-motion'
import { Terminal, Zap, HardDrive, Cpu, ArrowRight, Copy, Check, Github, Package, Server, Code2, Sparkles, Database, ChevronRight, Shield, Globe, Rocket, FileBox } from 'lucide-react'
import { useState, useRef, useEffect } from 'react'

const features = [
  {
    icon: Zap,
    title: '6.5s for 72B',
    description: 'Load Qwen 72B in 6.5 seconds. 79× faster than bitsandbytes. No more waiting.',
    stat: '79×',
    statLabel: 'Faster',
  },
  {
    icon: HardDrive,
    title: '63-70% Memory Savings',
    description: 'Run 32B models in 19GB VRAM. Fit larger models on your existing hardware.',
    stat: '70%',
    statLabel: 'Less VRAM',
  },
  {
    icon: Cpu,
    title: 'GPU + CPU Support',
    description: 'Auto-detect hardware. Run on CUDA GPUs, Apple Silicon, or CPU-only setups.',
    stat: '3',
    statLabel: 'Platforms',
  },
  {
    icon: Terminal,
    title: 'OpenAI Compatible',
    description: 'Drop-in replacement API. Works with LangChain, OpenAI SDK, and your existing code.',
    stat: '100%',
    statLabel: 'Compatible',
  },
]

const benchmarks = [
  { model: 'Qwen 7B', bnb: '45.4s', zse: '3.9s', speedup: '11.6×', memory: '5.2GB' },
  { model: 'Qwen 32B', bnb: '120.0s', zse: '21.4s', speedup: '5.6×', memory: '19.3GB' },
  { model: 'Qwen 72B', bnb: '512.7s', zse: '6.5s', speedup: '79×', memory: '76.6GB' },
]

const verifiedModels = [
  { name: 'Qwen 2.5 7B', provider: 'Alibaba', vram: '4.5 GB', category: 'Chat/Code', optimized: true },
  { name: 'Qwen 2.5 32B', provider: 'Alibaba', vram: '19 GB', category: 'Chat/Code', optimized: true },
  { name: 'Mistral 7B v0.3', provider: 'Mistral AI', vram: '4.5 GB', category: 'Chat', optimized: true },
  { name: 'DeepSeek Coder 6.7B', provider: 'DeepSeek', vram: '4 GB', category: 'Code', optimized: true },
  { name: 'Llama 3.2 3B', provider: 'Meta', vram: '2 GB', category: 'Chat', optimized: false },
  { name: 'Gemma 2 9B', provider: 'Google', vram: '5.5 GB', category: 'Reasoning', optimized: false },
  { name: 'Phi-3 Mini', provider: 'Microsoft', vram: '2.4 GB', category: 'Reasoning', optimized: false },
  { name: 'TinyLlama 1.1B', provider: 'TinyLlama', vram: '0.7 GB', category: 'Testing', optimized: false },
]

const howItWorks = [
  {
    step: '01',
    title: 'Install ZSE',
    description: 'One pip command to get started. No complex dependencies or configurations.',
    code: 'pip install zllm-zse',
    icon: Package,
  },
  {
    step: '02',
    title: 'Convert to .zse',
    description: 'Convert any HuggingFace model to optimized .zse format with 11× faster loading.',
    code: 'zse quantize Qwen/Qwen2.5-7B-Instruct -o ./model.zse',
    icon: FileBox,
  },
  {
    step: '03',
    title: 'Serve Your Model',
    description: 'Start the OpenAI-compatible API server with instant cold starts.',
    code: 'zse serve ./model.zse --port 8000',
    icon: Server,
  },
  {
    step: '04',
    title: 'Query the API',
    description: 'Use the OpenAI-compatible API with any client library or framework.',
    code: 'curl localhost:8000/v1/chat/completions',
    icon: Code2,
  },
]

const useCases = [
  {
    icon: Rocket,
    title: 'Serverless Inference',
    description: 'Sub-5s cold starts make ZSE perfect for serverless deployments where every millisecond of startup time costs money.',
  },
  {
    icon: Shield,
    title: 'Local AI Development',
    description: 'Run large models on your laptop. Test and iterate without cloud costs or API rate limits.',
  },
  {
    icon: Globe,
    title: 'Edge Deployment',
    description: 'Memory-efficient enough for edge devices. Deploy AI at the edge without expensive hardware.',
  },
  {
    icon: Database,
    title: 'Cost Optimization',
    description: 'Fit larger models on smaller GPUs. Cut your cloud compute bills by up to 70%.',
  },
]

const stats = [
  { value: 6.5, suffix: 's', label: '72B Cold Start' },
  { value: 45, suffix: '%', label: 'Less VRAM' },
  { value: 79, suffix: '×', label: 'Faster Loading' },
  { value: 100, suffix: '%', label: 'API Compatible' },
]

// Animated counter component
function AnimatedCounter({ value, suffix }: { value: number; suffix: string }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true })
  const [displayValue, setDisplayValue] = useState(0)

  useEffect(() => {
    if (isInView) {
      const duration = 2000
      const steps = 60
      const increment = value / steps
      let current = 0
      const timer = setInterval(() => {
        current += increment
        if (current >= value) {
          setDisplayValue(value)
          clearInterval(timer)
        } else {
          setDisplayValue(current)
        }
      }, duration / steps)
      return () => clearInterval(timer)
    }
  }, [isInView, value])

  return (
    <span ref={ref} className="tabular-nums">
      {displayValue.toFixed(value % 1 !== 0 ? 1 : 0)}{suffix}
    </span>
  )
}

// Floating particles background
function FloatingParticles() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-lime/30 rounded-full"
          initial={{
            x: `${Math.random() * 100}%`,
            y: `${Math.random() * 100}%`,
            scale: Math.random() * 0.5 + 0.5,
          }}
          animate={{
            y: [null, '-20%', '120%'],
            opacity: [0, 1, 0],
          }}
          transition={{
            duration: Math.random() * 10 + 10,
            repeat: Infinity,
            delay: Math.random() * 5,
            ease: 'linear',
          }}
        />
      ))}
    </div>
  )
}

// Animated gradient orb
function GradientOrb() {
  return (
    <motion.div
      className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full"
      style={{
        background: 'radial-gradient(circle, rgba(192,255,113,0.15) 0%, rgba(192,255,113,0.05) 40%, transparent 70%)',
      }}
      animate={{
        scale: [1, 1.2, 1],
        opacity: [0.5, 0.8, 0.5],
      }}
      transition={{
        duration: 8,
        repeat: Infinity,
        ease: 'easeInOut',
      }}
    />
  )
}

// Container animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
}

export default function HomePage() {
  const [copied, setCopied] = useState(false)
  const [activeStep, setActiveStep] = useState(0)
  const installCommand = 'pip install zllm-zse'

  const heroRef = useRef(null)
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ['start start', 'end start'],
  })
  
  const heroY = useTransform(scrollYProgress, [0, 1], [0, 200])
  const heroOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])

  const handleCopy = () => {
    navigator.clipboard.writeText(installCommand)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Auto-cycle through steps
  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % 4)
    }, 4000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="relative">
      {/* Hero Section */}
      <section ref={heroRef} className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
        {/* Background Effects */}
        <div className="absolute inset-0 grid-pattern opacity-30" />
        <FloatingParticles />
        <GradientOrb />
        
        <motion.div 
          style={{ y: heroY, opacity: heroOpacity }}
          className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center"
        >
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {/* Badge */}
            <motion.div variants={itemVariants}>
              <Link 
                href="/changelog"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-lime/10 border border-lime/20 text-lime text-sm mb-8 hover:bg-lime/20 transition-colors group"
              >
                <motion.span
                  animate={{ rotate: [0, 15, -15, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <Sparkles className="w-4 h-4" />
                </motion.span>
                <span>v0.1.3 released — 72B benchmarks!</span>
                <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
              </Link>
            </motion.div>

            {/* Main Heading */}
            <motion.h1 
              variants={itemVariants}
              className="text-5xl sm:text-6xl md:text-7xl font-bold tracking-tight"
            >
              <span className="text-white">Ultra Memory-Efficient</span>
              <br />
              <span className="gradient-text inline-block">
                LLM Inference
              </span>
            </motion.h1>

            <motion.p 
              variants={itemVariants}
              className="mt-6 text-xl text-gray-400 max-w-2xl mx-auto"
            >
              Load 72B models in <span className="text-lime font-semibold">6.5 seconds</span>. 
              <span className="text-lime font-semibold">79× faster</span> than bitsandbytes. 
              OpenAI-compatible API out of the box.
            </motion.p>

            {/* Install Command */}
            <motion.div variants={itemVariants} className="mt-10">
              <motion.div 
                className="inline-flex items-center gap-3 px-6 py-4 bg-[#0a0a0a] rounded-xl border border-white/10 font-mono text-lg"
                whileHover={{ scale: 1.02, borderColor: 'rgba(192,255,113,0.3)' }}
                transition={{ duration: 0.2 }}
              >
                <span className="text-gray-500">$</span>
                <span className="text-white">{installCommand}</span>
                <motion.button
                  onClick={handleCopy}
                  className="ml-2 p-2 text-gray-400 hover:text-lime transition-colors"
                  whileTap={{ scale: 0.9 }}
                >
                  <AnimatePresence mode="wait">
                    {copied ? (
                      <motion.div
                        key="check"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={{ scale: 0 }}
                      >
                        <Check className="w-5 h-5 text-lime" />
                      </motion.div>
                    ) : (
                      <motion.div
                        key="copy"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={{ scale: 0 }}
                      >
                        <Copy className="w-5 h-5" />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.button>
              </motion.div>
            </motion.div>

            {/* CTA Buttons */}
            <motion.div 
              variants={itemVariants}
              className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Link href="/docs/quickstart">
                <motion.div
                  className="group flex items-center gap-2 px-8 py-4 bg-lime text-black font-semibold rounded-xl"
                  whileHover={{ scale: 1.05, boxShadow: '0 0 30px rgba(192,255,113,0.4)' }}
                  whileTap={{ scale: 0.98 }}
                >
                  Get Started
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </motion.div>
              </Link>
              <Link href="https://github.com/Zyora-Dev/zse" target="_blank">
                <motion.div
                  className="flex items-center gap-2 px-8 py-4 bg-white/5 text-white font-semibold rounded-xl border border-white/10"
                  whileHover={{ scale: 1.05, backgroundColor: 'rgba(255,255,255,0.1)' }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Github className="w-5 h-5" />
                  View on GitHub
                </motion.div>
              </Link>
            </motion.div>
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="w-6 h-10 rounded-full border-2 border-white/20 flex justify-center pt-2">
            <motion.div
              className="w-1.5 h-1.5 bg-lime rounded-full"
              animate={{ y: [0, 16, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-16 border-y border-white/5 bg-[#030303]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, i) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold text-lime">
                  <AnimatedCounter value={stat.value} suffix={stat.suffix} />
                </div>
                <p className="mt-2 text-gray-500 text-sm">{stat.label}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 bg-black">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Get Running in 3 Steps
            </h2>
            <p className="mt-4 text-gray-400">
              From zero to serving models in under a minute
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Steps */}
            <div className="space-y-6">
              {howItWorks.map((item, i) => (
                <motion.div
                  key={item.step}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className={`relative p-6 rounded-xl border transition-all duration-300 cursor-pointer ${
                    activeStep === i 
                      ? 'bg-lime/5 border-lime/30' 
                      : 'bg-white/[0.02] border-white/5 hover:border-white/10'
                  }`}
                  onClick={() => setActiveStep(i)}
                >
                  <div className="flex items-start gap-4">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${
                      activeStep === i ? 'bg-lime/20' : 'bg-white/5'
                    }`}>
                      <item.icon className={`w-6 h-6 ${activeStep === i ? 'text-lime' : 'text-gray-400'}`} />
                    </div>
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`text-sm font-mono ${activeStep === i ? 'text-lime' : 'text-gray-500'}`}>
                          {item.step}
                        </span>
                        <h3 className="text-lg font-semibold text-white">{item.title}</h3>
                      </div>
                      <p className="text-gray-400 text-sm">{item.description}</p>
                    </div>
                  </div>
                  
                  {/* Progress bar for active step */}
                  {activeStep === i && (
                    <motion.div
                      className="absolute bottom-0 left-0 h-0.5 bg-lime rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: '100%' }}
                      transition={{ duration: 4, ease: 'linear' }}
                      key={`progress-${activeStep}`}
                    />
                  )}
                </motion.div>
              ))}
            </div>

            {/* Code Preview */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="relative"
            >
              <div className="rounded-2xl bg-[#0a0a0a] border border-white/10 overflow-hidden">
                <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10">
                  <div className="w-3 h-3 rounded-full bg-red-500/60" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
                  <div className="w-3 h-3 rounded-full bg-green-500/60" />
                  <span className="ml-3 text-sm text-gray-500">terminal</span>
                </div>
                <div className="p-6 min-h-[200px] flex items-center justify-center">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={activeStep}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.3 }}
                      className="font-mono text-lg"
                    >
                      <span className="text-lime">$</span>{' '}
                      <span className="text-white">{howItWorks[activeStep].code}</span>
                      <motion.span
                        className="inline-block w-2 h-5 bg-lime ml-1"
                        animate={{ opacity: [1, 0] }}
                        transition={{ duration: 0.8, repeat: Infinity }}
                      />
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
              
              {/* Decorative gradient */}
              <div className="absolute -inset-4 bg-gradient-to-r from-lime/10 via-transparent to-lime/10 rounded-3xl blur-xl -z-10 opacity-50" />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Why ZSE Section */}
      <section className="py-24 bg-[#050505]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Why ZSE?
            </h2>
            <p className="mt-4 text-gray-400 max-w-3xl mx-auto">
              ZSE solves a real problem: loading large models with bitsandbytes is slow because it quantizes on every load.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* bitsandbytes Card */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="p-8 rounded-2xl bg-[#0a0a0a] border border-white/10"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-3 h-3 rounded-full bg-gray-500" />
                <h3 className="text-xl font-semibold text-gray-400">bitsandbytes (Standard)</h3>
              </div>
              <div className="space-y-4 text-gray-400">
                <p>Every time you load a model:</p>
                <ol className="list-decimal list-inside space-y-2 text-sm">
                  <li>Download FP16 weights (14GB for 7B model)</li>
                  <li>Quantize to INT4 <span className="text-yellow-400">(takes 40+ seconds)</span></li>
                  <li>Finally ready to use</li>
                </ol>
                <div className="pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Qwen 7B Load Time:</span>
                    <span className="text-lg font-mono text-gray-300">45.4s</span>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* .zse Card */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="p-8 rounded-2xl bg-[#0a0a0a] border border-lime/20"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-3 h-3 rounded-full bg-lime" />
                <h3 className="text-xl font-semibold text-lime">.zse Format (Pre-quantized)</h3>
              </div>
              <div className="space-y-4 text-gray-400">
                <p>With ZSE, you quantize once, load instantly:</p>
                <ol className="list-decimal list-inside space-y-2 text-sm">
                  <li>One-time: <code className="text-lime bg-lime/10 px-1 rounded">zse quantize</code> → .zse file</li>
                  <li>Every load: Read pre-quantized weights <span className="text-lime">(instant)</span></li>
                  <li>Ready in seconds, not minutes</li>
                </ol>
                <div className="pt-4 border-t border-lime/20">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Qwen 7B Load Time:</span>
                    <span className="text-lg font-mono text-lime">3.9s</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Developer Note */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-12 p-6 rounded-xl bg-lime/5 border border-lime/10"
          >
            <div className="flex items-start gap-4">
              <Sparkles className="w-6 h-6 text-lime flex-shrink-0 mt-1" />
              <div>
                <h4 className="font-semibold text-white mb-2">ZSE vs llama.cpp (GGUF)</h4>
                <p className="text-gray-400 text-sm">
                  On Qwen 72B: ZSE loads in <span className="text-lime font-semibold">6.5s</span> vs llama.cpp GGUF in <span className="text-white">10.2s</span> — 
                  ZSE is <span className="text-lime font-semibold">1.6× faster</span> while using Python ecosystem.
                </p>
                <p className="text-gray-400 text-sm mt-2">
                  <strong className="text-white">When to use .zse:</strong> Production deployments, serverless functions, CI/CD pipelines, 
                  anywhere you need fast cold starts with the Python/HuggingFace ecosystem.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Benchmarks Section */}
      <section className="py-24 bg-black">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Verified Benchmarks
            </h2>
            <p className="mt-4 text-gray-400">
              .zse format vs bitsandbytes on-the-fly quantization. Tested on NVIDIA H200 (150GB VRAM).
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="overflow-hidden rounded-2xl border border-white/10 bg-[#0a0a0a]"
          >
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="px-6 py-4 text-left text-sm font-semibold text-gray-400">Model</th>
                    <th className="px-6 py-4 text-center text-sm font-semibold text-gray-400">bitsandbytes</th>
                    <th className="px-6 py-4 text-center text-sm font-semibold text-lime">ZSE (.zse)</th>
                    <th className="px-6 py-4 text-center text-sm font-semibold text-gray-400">VRAM</th>
                    <th className="px-6 py-4 text-right text-sm font-semibold text-gray-400">Speedup</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarks.map((row, i) => (
                    <motion.tr
                      key={row.model}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: i * 0.1 }}
                      className="border-b border-white/5 last:border-0 group"
                    >
                      <td className="px-6 py-5 text-white font-medium">{row.model}</td>
                      <td className="px-6 py-5 text-center text-gray-400">{row.bnb}</td>
                      <td className="px-6 py-5 text-center text-lime font-semibold">{row.zse}</td>
                      <td className="px-6 py-5 text-center text-gray-400">{row.memory}</td>
                      <td className="px-6 py-5 text-right">
                        <motion.span 
                          className="inline-flex items-center px-3 py-1 rounded-full bg-lime/10 text-lime text-sm font-semibold"
                          whileHover={{ scale: 1.1 }}
                        >
                          {row.speedup}
                        </motion.span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>

          {/* Visual comparison bar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-12 p-8 rounded-2xl bg-[#0a0a0a] border border-white/10"
          >
            <h3 className="text-lg font-semibold text-white mb-2">Load Time Comparison (Qwen 72B)</h3>
            <p className="text-sm text-gray-500 mb-6">Tested on NVIDIA H200 (150GB VRAM)</p>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">bitsandbytes</span>
                  <span className="text-gray-400">512.7s</span>
                </div>
                <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gray-500 rounded-full"
                    initial={{ width: 0 }}
                    whileInView={{ width: '100%' }}
                    viewport={{ once: true }}
                    transition={{ duration: 1.5, ease: 'easeOut' }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-blue-400">llama.cpp (GGUF)</span>
                  <span className="text-blue-400">10.2s</span>
                </div>
                <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-blue-500 rounded-full"
                    initial={{ width: 0 }}
                    whileInView={{ width: '2%' }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, ease: 'easeOut', delay: 0.3 }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-lime">ZSE (.zse)</span>
                  <span className="text-lime">6.5s</span>
                </div>
                <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-lime rounded-full"
                    initial={{ width: 0 }}
                    whileInView={{ width: '1.3%' }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, ease: 'easeOut', delay: 0.6 }}
                  />
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-black">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Built for Efficiency
            </h2>
            <p className="mt-4 text-gray-400">
              Every feature designed for memory efficiency and fast cold starts
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ 
                  y: -5,
                  transition: { duration: 0.2 }
                }}
                className="group relative p-8 rounded-2xl bg-[#0a0a0a] border border-white/5 hover:border-lime/20 transition-all overflow-hidden"
              >
                {/* Background gradient on hover */}
                <div className="absolute inset-0 bg-gradient-to-br from-lime/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                
                <div className="relative">
                  <div className="flex items-start justify-between">
                    <div className="w-12 h-12 rounded-xl bg-lime/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                      <feature.icon className="w-6 h-6 text-lime" />
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-lime">{feature.stat}</div>
                      <div className="text-xs text-gray-500">{feature.statLabel}</div>
                    </div>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                  <p className="text-gray-400">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases Section */}
      <section className="py-24 bg-[#050505]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Perfect For
            </h2>
            <p className="mt-4 text-gray-400">
              From local development to production deployments
            </p>
          </motion.div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {useCases.map((useCase, i) => (
              <motion.div
                key={useCase.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                whileHover={{ scale: 1.02 }}
                className="p-6 rounded-xl bg-white/[0.02] border border-white/5 hover:border-lime/20 transition-all"
              >
                <div className="w-10 h-10 rounded-lg bg-lime/10 flex items-center justify-center mb-4">
                  <useCase.icon className="w-5 h-5 text-lime" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">{useCase.title}</h3>
                <p className="text-sm text-gray-400">{useCase.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Verified Models Section */}
      <section className="py-24 bg-black">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Verified Models
            </h2>
            <p className="mt-4 text-gray-400">
              Tested and optimized for ZSE. VRAM shown for INT4 quantization.
            </p>
          </motion.div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="text-left py-4 px-4 text-sm font-medium text-gray-400">Model</th>
                  <th className="text-left py-4 px-4 text-sm font-medium text-gray-400">Provider</th>
                  <th className="text-left py-4 px-4 text-sm font-medium text-gray-400">Category</th>
                  <th className="text-right py-4 px-4 text-sm font-medium text-gray-400">VRAM (INT4)</th>
                  <th className="text-center py-4 px-4 text-sm font-medium text-gray-400">.zse Ready</th>
                </tr>
              </thead>
              <tbody>
                {verifiedModels.map((model, i) => (
                  <motion.tr
                    key={model.name}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.05 }}
                    className="border-b border-white/5 hover:bg-white/[0.02] transition-colors"
                  >
                    <td className="py-4 px-4">
                      <span className="font-medium text-white">{model.name}</span>
                    </td>
                    <td className="py-4 px-4 text-gray-400">{model.provider}</td>
                    <td className="py-4 px-4">
                      <span className="px-2 py-1 text-xs rounded-full bg-lime/10 text-lime">
                        {model.category}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-right font-mono text-lime">{model.vram}</td>
                    <td className="py-4 px-4 text-center">
                      {model.optimized ? (
                        <span className="text-lime">✓</span>
                      ) : (
                        <span className="text-gray-500">—</span>
                      )}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="mt-8 text-center"
          >
            <Link href="/docs/model-formats" className="text-lime hover:underline text-sm">
              See full model compatibility list →
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Code Example Section */}
      <section className="py-24 bg-black">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Simple, Powerful API
            </h2>
            <p className="mt-4 text-gray-400">
              Start serving models with just a few lines
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="rounded-2xl bg-[#0a0a0a] border border-white/10 overflow-hidden"
          >
            <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10">
              <div className="w-3 h-3 rounded-full bg-red-500/60" />
              <div className="w-3 h-3 rounded-full bg-yellow-500/60" />
              <div className="w-3 h-3 rounded-full bg-green-500/60" />
              <span className="ml-3 text-sm text-gray-500">terminal</span>
            </div>
            <pre className="p-6 overflow-x-auto">
              <code className="text-sm leading-relaxed">
                <span className="text-gray-500"># Install ZSE</span>{'\n'}
                <span className="text-lime">$</span> <span className="text-white">pip install zllm-zse</span>{'\n\n'}
                <span className="text-gray-500"># Convert to optimized .zse format (11× faster loading)</span>{'\n'}
                <span className="text-lime">$</span> <span className="text-white">zse quantize Qwen/Qwen2.5-7B-Instruct -o ./model.zse</span>{'\n\n'}
                <span className="text-gray-500"># Serve your model with instant cold starts</span>{'\n'}
                <span className="text-lime">$</span> <span className="text-white">zse serve ./model.zse --port 8000</span>{'\n\n'}
                <span className="text-gray-500"># OpenAI-compatible API is ready!</span>{'\n'}
                <span className="text-lime">$</span> <span className="text-white">curl localhost:8000/v1/chat/completions \</span>{'\n'}
                <span className="text-white">    -H </span><span className="text-yellow-400">&quot;Content-Type: application/json&quot;</span><span className="text-white"> \</span>{'\n'}
                <span className="text-white">    -d </span><span className="text-yellow-400">&apos;{`{"model":"default","messages":[{"role":"user","content":"Hello!"}]}`}&apos;</span>
              </code>
            </pre>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-[#050505] relative overflow-hidden">
        {/* Background effect */}
        <div className="absolute inset-0 bg-gradient-to-b from-lime/5 via-transparent to-transparent" />
        
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <motion.div
              animate={{ 
                scale: [1, 1.05, 1],
                opacity: [0.5, 1, 0.5]
              }}
              transition={{ duration: 3, repeat: Infinity }}
              className="w-20 h-20 mx-auto mb-8 rounded-2xl bg-lime/10 flex items-center justify-center"
            >
              <Rocket className="w-10 h-10 text-lime" />
            </motion.div>
            
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Ready to Try ZSE?
            </h2>
            <p className="mt-4 text-gray-400 max-w-2xl mx-auto">
              Get memory-efficient LLM inference with fast cold starts. Install and start serving in minutes.
            </p>
            <motion.div 
              className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Link href="/docs/quickstart">
                <motion.div
                  className="group flex items-center gap-2 px-8 py-4 bg-lime text-black font-semibold rounded-xl"
                  whileHover={{ scale: 1.05, boxShadow: '0 0 40px rgba(192,255,113,0.5)' }}
                  whileTap={{ scale: 0.98 }}
                >
                  Start Building
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </motion.div>
              </Link>
              <Link href="/community">
                <motion.div
                  className="flex items-center gap-2 px-8 py-4 bg-white/5 text-white font-semibold rounded-xl border border-white/10"
                  whileHover={{ scale: 1.05, backgroundColor: 'rgba(255,255,255,0.1)' }}
                  whileTap={{ scale: 0.98 }}
                >
                  Join Community
                </motion.div>
              </Link>
            </motion.div>
            
            {/* Trust badges */}
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3 }}
              className="mt-12 flex flex-wrap items-center justify-center gap-6 text-gray-500 text-sm"
            >
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4" />
                <span>Apache 2.0 Licensed</span>
              </div>
              <div className="flex items-center gap-2">
                <Github className="w-4 h-4" />
                <span>Open Source</span>
              </div>
              <div className="flex items-center gap-2">
                <Package className="w-4 h-4" />
                <span>PyPI Published</span>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
