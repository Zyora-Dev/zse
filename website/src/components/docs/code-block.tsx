'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Copy, Check, Terminal, FileCode } from 'lucide-react'
import { cn } from '@/lib/utils'

interface CodeBlockProps {
  code: string
  language?: string
  filename?: string
  showLineNumbers?: boolean
  highlightLines?: number[]
}

export function CodeBlock({ 
  code, 
  language = 'bash', 
  filename,
  showLineNumbers = false,
  highlightLines = []
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const lines = code.trim().split('\n')

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="group relative my-4 rounded-lg overflow-hidden border border-white/[0.06] bg-[#0a0a0a]"
    >
      {/* Header */}
      {(filename || language) && (
        <div className="flex items-center justify-between px-4 py-2 border-b border-white/[0.06] bg-white/[0.02]">
          <div className="flex items-center gap-2">
            {language === 'bash' || language === 'shell' ? (
              <Terminal className="w-3 h-3 text-white/30" />
            ) : (
              <FileCode className="w-3 h-3 text-white/30" />
            )}
            <span className="text-xs font-medium text-white/40 uppercase tracking-wider">
              {filename || language}
            </span>
          </div>
          <button
            onClick={copyToClipboard}
            className="opacity-0 group-hover:opacity-100 p-1.5 rounded hover:bg-white/5 transition-all"
          >
            {copied ? (
              <Check className="w-3 h-3 text-lime" />
            ) : (
              <Copy className="w-3 h-3 text-white/40" />
            )}
          </button>
        </div>
      )}

      {/* Code */}
      <div className="relative overflow-x-auto">
        <pre className="p-4 text-sm leading-relaxed font-mono">
          <code>
            {lines.map((line, i) => (
              <div 
                key={i}
                className={cn(
                  "flex",
                  highlightLines.includes(i + 1) && "bg-lime/5 -mx-4 px-4 border-l-2 border-lime"
                )}
              >
                {showLineNumbers && (
                  <span className="select-none pr-4 text-white/20 text-right w-8">
                    {i + 1}
                  </span>
                )}
                <span className="text-white/80">{line || ' '}</span>
              </div>
            ))}
          </code>
        </pre>
      </div>

      {/* Copy button (no header) */}
      {!filename && !language && (
        <button
          onClick={copyToClipboard}
          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 p-1.5 rounded bg-white/5 hover:bg-white/10 transition-all"
        >
          {copied ? (
            <Check className="w-3 h-3 text-lime" />
          ) : (
            <Copy className="w-3 h-3 text-white/40" />
          )}
        </button>
      )}
    </motion.div>
  )
}

interface InlineCodeProps {
  children: React.ReactNode
}

export function InlineCode({ children }: InlineCodeProps) {
  return (
    <code className="px-1.5 py-0.5 text-sm font-mono bg-white/[0.06] text-lime/90 rounded border border-white/[0.06]">
      {children}
    </code>
  )
}
