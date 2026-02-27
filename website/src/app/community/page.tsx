'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { Github, Twitter, Youtube, BookOpen, Users, Code, Heart, Linkedin } from 'lucide-react'
import { DiscordIcon } from '@/components/icons/discord'

const communityLinks = [
  {
    icon: Github,
    title: 'GitHub',
    description: 'Report issues and contribute to the codebase',
    link: 'https://github.com/Zyora-Dev/zse',
    stats: 'Open Source',
    color: 'bg-white/10',
  },
  {
    icon: DiscordIcon,
    title: 'Discord',
    description: 'Join our community server for real-time discussions',
    link: 'https://discord.gg/f9JKreJA',
    stats: 'Join us',
    color: 'bg-indigo-500/20',
  },
  {
    icon: Twitter,
    title: 'X (Twitter)',
    description: 'Follow us for updates, tips, and announcements',
    link: 'https://x.com/zyoralabs',
    stats: '@zyoralabs',
    color: 'bg-blue-500/20',
  },
  {
    icon: Youtube,
    title: 'YouTube',
    description: 'Watch tutorials, demos, and deep dives',
    link: 'https://youtube.com/@zse',
    stats: 'Coming soon',
    color: 'bg-red-500/20',
  },
  {
    icon: Linkedin,
    title: 'LinkedIn',
    description: 'Connect with us professionally',
    link: 'https://www.linkedin.com/company/zyora-labs/',
    stats: 'Zyora Labs',
    color: 'bg-blue-600/20',
  },
]

const contributions = [
  {
    icon: Code,
    title: 'Code Contributions',
    description: 'Help us build new features, fix bugs, and improve performance',
    link: 'https://github.com/Zyora-Dev/zse',
  },
  {
    icon: BookOpen,
    title: 'Documentation',
    description: 'Improve our docs, write tutorials, and translate content',
    link: 'https://github.com/Zyora-Dev/zse/tree/main/docs',
  },
  {
    icon: Users,
    title: 'Community Support',
    description: 'Help others on Discord and GitHub Discussions',
    link: 'https://discord.gg/f9JKreJA',
  },
  {
    icon: Heart,
    title: 'Spread the Word',
    description: 'Share ZSE with your network, write blog posts, and give talks',
    link: 'https://x.com/intent/tweet?text=Check%20out%20ZSE%20-%20Memory-efficient%20LLM%20inference%20https://zllm.in',
  },
]

const showcaseProjects = [
  {
    title: 'ZSE Documentation',
    description: 'Official documentation and quickstart guides',
    author: 'Zyora Labs',
    link: 'https://zllm.in/docs',
  },
  {
    title: 'PyPI Package',
    description: 'Install ZSE directly from PyPI',
    author: 'Zyora Labs',
    link: 'https://pypi.org/project/zllm-zse/',
  },
  {
    title: 'GitHub Repository',
    description: 'Source code and issue tracker',
    author: 'Zyora Labs',
    link: 'https://github.com/Zyora-Dev/zse',
  },
]

export default function CommunityPage() {
  return (
    <div className="min-h-screen bg-black pt-24 pb-16">
      <div className="max-w-5xl mx-auto px-6">
        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Community
          </h1>
          <p className="text-xl text-white/60 max-w-2xl mx-auto">
            ZSE is built by and for the community. Join us in making LLM inference faster and more efficient.
          </p>
        </motion.div>

        {/* Social Links */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-16"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">Connect with us</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {communityLinks.map((item, index) => (
              <a
                key={item.title}
                href={item.link}
                target="_blank"
                rel="noopener noreferrer"
                className="border border-white/10 rounded-xl p-6 bg-white/[0.02] hover:border-lime/50 transition-all duration-300 group flex gap-4"
              >
                <div className={`w-12 h-12 ${item.color} rounded-xl flex items-center justify-center flex-shrink-0`}>
                  <item.icon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-lg font-semibold text-white group-hover:text-lime transition-colors">
                      {item.title}
                    </h3>
                    <span className="text-xs text-white/40 bg-white/5 px-2 py-0.5 rounded">
                      {item.stats}
                    </span>
                  </div>
                  <p className="text-white/60 text-sm">{item.description}</p>
                </div>
              </a>
            ))}
          </div>
        </motion.section>

        {/* Contributing */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-16"
        >
          <h2 className="text-2xl font-semibold text-white mb-6">How to contribute</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {contributions.map((item, index) => (
              <a
                key={item.title}
                href={item.link}
                target="_blank"
                rel="noopener noreferrer"
                className="border border-white/10 rounded-xl p-6 bg-white/[0.02] hover:border-lime/50 transition-all duration-300 group"
              >
                <div className="w-10 h-10 bg-lime/20 rounded-lg flex items-center justify-center mb-4">
                  <item.icon className="w-5 h-5 text-lime" />
                </div>
                <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-lime transition-colors">
                  {item.title}
                </h3>
                <p className="text-white/60 text-sm">{item.description}</p>
              </a>
            ))}
          </div>
        </motion.section>

        {/* Showcase */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-16"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-white">Community Showcase</h2>
            <a
              href="https://github.com/zse/zse/discussions"
              target="_blank"
              rel="noopener noreferrer"
              className="text-lime text-sm hover:underline"
            >
              Submit your project →
            </a>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {showcaseProjects.map((project, index) => (
              <a
                key={project.title}
                href={project.link}
                target="_blank"
                rel="noopener noreferrer"
                className="border border-white/10 rounded-xl p-6 bg-white/[0.02] hover:border-lime/50 transition-all duration-300 group"
              >
                <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-lime transition-colors">
                  {project.title}
                </h3>
                <p className="text-white/60 text-sm mb-4">{project.description}</p>
                <span className="text-xs text-white/40">by {project.author}</span>
              </a>
            ))}
          </div>
        </motion.section>

        {/* Code of Conduct */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="border border-white/10 rounded-xl p-8 bg-white/[0.02]"
        >
          <h2 className="text-2xl font-semibold text-white mb-4">Code of Conduct</h2>
          <p className="text-white/60 mb-6">
            We're committed to providing a welcoming and inclusive community for everyone. 
            All participants are expected to uphold our code of conduct:
          </p>
          <ul className="space-y-3 text-white/80">
            <li className="flex gap-3">
              <span className="text-lime">•</span>
              Be respectful and inclusive of all community members
            </li>
            <li className="flex gap-3">
              <span className="text-lime">•</span>
              Use welcoming and inclusive language
            </li>
            <li className="flex gap-3">
              <span className="text-lime">•</span>
              Be patient with newcomers and help them get started
            </li>
            <li className="flex gap-3">
              <span className="text-lime">•</span>
              Focus on constructive feedback and collaboration
            </li>
          </ul>
          <a
            href="https://github.com/zse/zse/blob/main/CODE_OF_CONDUCT.md"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block mt-6 text-lime hover:underline"
          >
            Read full Code of Conduct →
          </a>
        </motion.section>
      </div>
    </div>
  )
}
