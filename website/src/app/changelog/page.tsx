'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { changelog, formatChangelogDate, changeTypeColors, getLatestVersion } from '@/lib/changelog'
import { Tag, ExternalLink } from 'lucide-react'

export default function ChangelogPage() {
  const latestVersion = getLatestVersion()

  return (
    <div className="min-h-screen bg-black pt-24 pb-16">
      <div className="max-w-3xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">Changelog</h1>
          <p className="text-xl text-white/60">
            All notable changes and version history
          </p>
          <div className="mt-6 flex items-center gap-4">
            <a
              href="https://pypi.org/project/zllm-zse/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 bg-lime/20 text-lime rounded-lg hover:bg-lime/30 transition-colors"
            >
              <Tag className="w-4 h-4" />
              Latest: v{latestVersion}
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </motion.div>

        <div className="space-y-12">
          {changelog.map((entry, index) => (
            <motion.article
              key={entry.version}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="border-l-2 border-white/10 pl-8 relative"
            >
              {/* Version dot */}
              <div className="absolute -left-[9px] top-0 w-4 h-4 bg-lime rounded-full" />

              {/* Version header */}
              <div className="mb-6">
                <div className="flex items-center gap-3 mb-2">
                  <h2 className="text-2xl font-bold text-white">v{entry.version}</h2>
                  {index === 0 && (
                    <span className="px-2 py-0.5 text-xs font-medium bg-lime/20 text-lime rounded">
                      Latest
                    </span>
                  )}
                </div>
                <p className="text-white/60 text-sm">{formatChangelogDate(entry.date)}</p>
                <h3 className="text-lg text-white/80 mt-2">{entry.title}</h3>
                <p className="text-white/60 mt-1">{entry.description}</p>
              </div>

              {/* Changes list */}
              <div className="space-y-3">
                {entry.changes.map((change, i) => {
                  const colors = changeTypeColors[change.type]
                  return (
                    <div key={i} className="flex gap-3 items-start">
                      <span
                        className={`px-2 py-0.5 text-xs font-medium rounded ${colors.bg} ${colors.text} uppercase flex-shrink-0`}
                      >
                        {change.type}
                      </span>
                      <p className="text-white/80">{change.text}</p>
                    </div>
                  )
                })}
              </div>
            </motion.article>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-16 text-center border border-white/10 rounded-xl p-8 bg-white/[0.02]"
        >
          <h2 className="text-xl font-semibold text-white mb-4">
            Want to see what's coming next?
          </h2>
          <p className="text-white/60 mb-6">
            Check out our roadmap and upcoming features on GitHub
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <a
              href="https://github.com/zse/zse/milestones"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-lime text-black font-medium rounded-lg hover:bg-lime/90 transition-colors"
            >
              View Roadmap
              <ExternalLink className="w-4 h-4" />
            </a>
            <a
              href="https://github.com/zse/zse/releases"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 border border-white/20 text-white font-medium rounded-lg hover:border-lime/50 transition-colors"
            >
              GitHub Releases
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
