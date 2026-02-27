'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import { ChevronRight, Book, Zap, Settings, Code, Server, Database, Cpu, Terminal, FileCode, Layers, ArrowRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DocLink {
  title: string
  href: string
  icon?: React.ElementType
}

interface DocSection {
  title: string
  icon: React.ElementType
  links: DocLink[]
}

const docSections: DocSection[] = [
  {
    title: 'Getting Started',
    icon: Zap,
    links: [
      { title: 'Introduction', href: '/docs' },
      { title: 'Quick Start', href: '/docs/quickstart' },
      { title: 'Installation', href: '/docs/installation' },
      { title: 'First Model', href: '/docs/first-model' },
    ],
  },
  {
    title: 'Core Concepts',
    icon: Book,
    links: [
      { title: 'Architecture', href: '/docs/architecture' },
      { title: 'Model Formats', href: '/docs/model-formats' },
      { title: 'Quantization', href: '/docs/quantization' },
      { title: 'Memory Management', href: '/docs/memory' },
    ],
  },
  {
    title: 'Features',
    icon: Layers,
    links: [
      { title: 'zQuantize', href: '/docs/zquantize' },
      { title: 'zServe', href: '/docs/zserve' },
      { title: 'zInfer', href: '/docs/zinfer' },
      { title: 'zStream', href: '/docs/zstream' },
      { title: 'zKV', href: '/docs/zkv' },
      { title: 'Multi-GPU', href: '/docs/multi-gpu' },
      { title: 'GGUF Compatibility', href: '/docs/gguf' },
      { title: 'RAG Module', href: '/docs/rag' },
      { title: 'MCP Tools', href: '/docs/mcp' },
    ],
  },
  {
    title: 'API Reference',
    icon: Code,
    links: [
      { title: 'CLI Commands', href: '/docs/api/cli' },
      { title: 'Python API', href: '/docs/api/python' },
      { title: 'REST API', href: '/docs/api/rest' },
      { title: 'Configuration', href: '/docs/api/config' },
    ],
  },
  {
    title: 'Deployment',
    icon: Server,
    links: [
      { title: 'Production Setup', href: '/docs/deployment/production' },
      { title: 'Docker', href: '/docs/deployment/docker' },
      { title: 'Kubernetes', href: '/docs/deployment/kubernetes' },
      { title: 'Serverless', href: '/docs/deployment/serverless' },
    ],
  },
  {
    title: 'Advanced',
    icon: Cpu,
    links: [
      { title: 'Custom Models', href: '/docs/advanced/custom-models' },
      { title: 'Performance Tuning', href: '/docs/advanced/performance' },
      { title: 'Benchmarking', href: '/docs/advanced/benchmarking' },
      { title: 'Troubleshooting', href: '/docs/advanced/troubleshooting' },
    ],
  },
]

function SidebarSection({ section, isExpanded, onToggle }: { 
  section: DocSection
  isExpanded: boolean
  onToggle: () => void 
}) {
  const pathname = usePathname()
  const isActive = section.links.some(link => pathname === link.href)
  const Icon = section.icon

  return (
    <div className="mb-1">
      <button
        onClick={onToggle}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200",
          isActive || isExpanded
            ? "text-white bg-white/5"
            : "text-white/50 hover:text-white/80 hover:bg-white/[0.02]"
        )}
      >
        <Icon className="w-4 h-4 text-lime/70" />
        <span className="flex-1 text-left">{section.title}</span>
        <ChevronRight 
          className={cn(
            "w-3.5 h-3.5 transition-transform duration-200",
            isExpanded && "rotate-90"
          )} 
        />
      </button>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="ml-3 pl-3 border-l border-white/5 mt-1 space-y-0.5">
              {section.links.map((link) => {
                const isLinkActive = pathname === link.href
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={cn(
                      "block px-3 py-1.5 text-sm rounded-md transition-all duration-200",
                      isLinkActive
                        ? "text-lime bg-lime/10"
                        : "text-white/40 hover:text-white/70 hover:bg-white/[0.02]"
                    )}
                  >
                    {link.title}
                  </Link>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function DocsSidebar() {
  const pathname = usePathname()
  const [expandedSections, setExpandedSections] = useState<string[]>(() => {
    // Auto-expand section containing current page
    const activeSection = docSections.find(s => 
      s.links.some(l => pathname === l.href)
    )
    return activeSection ? [activeSection.title] : ['Getting Started']
  })

  const toggleSection = (title: string) => {
    setExpandedSections(prev => 
      prev.includes(title) 
        ? prev.filter(t => t !== title)
        : [...prev, title]
    )
  }

  return (
    <aside className="w-56 flex-shrink-0 sticky top-20 h-[calc(100vh-5rem)] overflow-y-auto scrollbar-thin">
      <nav className="py-6 pr-4">
        <div className="mb-6">
          <Link 
            href="/docs"
            className="flex items-center gap-2 text-sm font-semibold text-white/80 hover:text-lime transition-colors"
          >
            <FileCode className="w-4 h-4" />
            Documentation
          </Link>
        </div>

        <div className="space-y-1">
          {docSections.map((section) => (
            <SidebarSection
              key={section.title}
              section={section}
              isExpanded={expandedSections.includes(section.title)}
              onToggle={() => toggleSection(section.title)}
            />
          ))}
        </div>

        <div className="mt-8 pt-6 border-t border-white/5">
          <div className="px-3 space-y-3">
            <a
              href="https://github.com/Zyora-Dev/zse"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-white/40 hover:text-white/70 transition-colors"
            >
              <Terminal className="w-3.5 h-3.5" />
              View on GitHub
              <ArrowRight className="w-3 h-3" />
            </a>
            <a
              href="https://pypi.org/project/zllm-zse/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-white/40 hover:text-white/70 transition-colors"
            >
              <Database className="w-3.5 h-3.5" />
              PyPI Package
              <ArrowRight className="w-3 h-3" />
            </a>
          </div>
        </div>
      </nav>
    </aside>
  )
}
