'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { Zap, Target, Heart, Rocket } from 'lucide-react'

const values = [
  {
    icon: Zap,
    title: 'Speed',
    description: 'Every millisecond counts. We obsess over cold start times and inference latency.',
  },
  {
    icon: Target,
    title: 'Simplicity',
    description: 'Complex problems deserve simple solutions. Our API is clean and intuitive.',
  },
  {
    icon: Heart,
    title: 'Open Source',
    description: 'We believe in building in public. ZSE is Apache 2.0 licensed and community-driven.',
  },
  {
    icon: Rocket,
    title: 'Innovation',
    description: 'Pushing the boundaries of what\'s possible with efficient LLM inference.',
  },
]

const milestones = [
  { date: 'January 2026', event: 'ZSE research project begins' },
  { date: 'February 2026', event: 'First working prototype with 3.9s cold starts' },
  { date: 'February 2026', event: 'Public release on PyPI' },
]

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-black pt-24 pb-16">
      <div className="max-w-4xl mx-auto px-6">
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-20"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
            About ZSE
          </h1>
          <p className="text-xl text-white/60 max-w-2xl mx-auto">
            Making LLM inference fast, efficient, and accessible to everyone.
          </p>
        </motion.div>

        {/* Who We Are */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-20"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">Who We Are</h2>
          <div className="border border-lime/20 rounded-xl p-8 bg-lime/[0.03]">
            <p className="text-lg text-white/90 leading-relaxed">
              <span className="text-lime font-semibold">zLLM-ZSE</span> is developed by{' '}
              <span className="text-lime font-semibold">Zyora Labs</span>, an AI research 
              and development firm based in Tamil Nadu, India. We're dedicated to pushing 
              the boundaries of efficient AI inference and making advanced language models 
              accessible to developers worldwide.
            </p>
          </div>
        </motion.section>

        {/* Mission */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="mb-20"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">Our Mission</h2>
          <div className="border border-white/10 rounded-xl p-8 bg-white/[0.02]">
            <p className="text-lg text-white/80 leading-relaxed">
              We started ZSE because we were frustrated with slow model loading times. 
              Every time we wanted to test a model or spin up a serverless endpoint, 
              we'd wait 30-60 seconds for the model to load. That's unacceptable.
            </p>
            <p className="text-lg text-white/80 leading-relaxed mt-4">
              ZSE introduces pre-quantized model formats that load instantly. 
              No runtime quantization. No wasted compute. Just fast inference.
            </p>
            <p className="text-lg text-white/80 leading-relaxed mt-4">
              Our goal is to make LLM inference as fast and efficient as possible, 
              enabling developers to build better AI applications without breaking the bank.
            </p>
          </div>
        </motion.section>

        {/* Values */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-20"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">Our Values</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {values.map((value, index) => (
              <motion.div
                key={value.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + index * 0.1 }}
                className="border border-white/10 rounded-xl p-6 bg-white/[0.02]"
              >
                <div className="w-10 h-10 bg-lime/20 rounded-lg flex items-center justify-center mb-4">
                  <value.icon className="w-5 h-5 text-lime" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">{value.title}</h3>
                <p className="text-white/60">{value.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Timeline */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="mb-20"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">Timeline</h2>
          <div className="border border-white/10 rounded-xl p-8 bg-white/[0.02]">
            <div className="space-y-6">
              {milestones.map((milestone, index) => (
                <div key={index} className="flex gap-4">
                  <div className="w-2 h-2 bg-lime rounded-full mt-2 flex-shrink-0" />
                  <div>
                    <p className="text-lime font-medium">{milestone.date}</p>
                    <p className="text-white/80">{milestone.event}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* Tech Stack */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-20"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">Technology</h2>
          <div className="border border-white/10 rounded-xl p-8 bg-white/[0.02]">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold text-white mb-3">Core Stack</h3>
                <ul className="space-y-2 text-white/60">
                  <li>• Python 3.8+</li>
                  <li>• PyTorch 2.0+</li>
                  <li>• CUDA/ROCm acceleration</li>
                  <li>• Hugging Face Transformers</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-3">Key Techniques</h3>
                <ul className="space-y-2 text-white/60">
                  <li>• INT4/NF4 quantization</li>
                  <li>• Memory-mapped tensors</li>
                  <li>• KV cache compression</li>
                  <li>• Layer streaming</li>
                </ul>
              </div>
            </div>
          </div>
        </motion.section>

        {/* CTA */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          className="text-center"
        >
          <div className="border border-white/10 rounded-xl p-8 bg-white/[0.02]">
            <h2 className="text-2xl font-semibold text-white mb-4">
              Ready to get started?
            </h2>
            <p className="text-white/60 mb-6">
              Install ZSE and run your first model in minutes.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link
                href="/docs"
                className="px-6 py-3 bg-lime text-black font-medium rounded-lg hover:bg-lime/90 transition-colors"
              >
                Read the docs
              </Link>
              <Link
                href="/community"
                className="px-6 py-3 border border-white/20 text-white font-medium rounded-lg hover:border-lime/50 transition-colors"
              >
                Join community
              </Link>
            </div>
          </div>
        </motion.section>
      </div>
    </div>
  )
}
