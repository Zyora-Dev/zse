'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface TOCItem {
  id: string
  title: string
  level: number
}

interface TableOfContentsProps {
  items: TOCItem[]
}

export function TableOfContents({ items }: TableOfContentsProps) {
  const [activeId, setActiveId] = useState<string>('')

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id)
          }
        })
      },
      { rootMargin: '-80px 0% -80% 0%' }
    )

    items.forEach((item) => {
      const element = document.getElementById(item.id)
      if (element) observer.observe(element)
    })

    return () => observer.disconnect()
  }, [items])

  if (items.length === 0) return null

  return (
    <nav className="w-48 flex-shrink-0 sticky top-20 h-[calc(100vh-5rem)] overflow-y-auto hidden xl:block">
      <div className="py-6 pl-4">
        <p className="text-xs font-semibold text-white/40 uppercase tracking-wider mb-3">
          On this page
        </p>
        <ul className="space-y-1">
          {items.map((item) => (
            <li key={item.id}>
              <a
                href={`#${item.id}`}
                className={cn(
                  "block py-1 text-sm transition-colors border-l border-transparent",
                  item.level === 2 && "pl-3",
                  item.level === 3 && "pl-5",
                  activeId === item.id
                    ? "text-lime border-l-lime"
                    : "text-white/40 hover:text-white/70"
                )}
              >
                {item.title}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}

interface DocHeaderProps {
  title: string
  description?: string
  badge?: string
}

export function DocHeader({ title, description, badge }: DocHeaderProps) {
  return (
    <motion.header
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-8 pb-6 border-b border-white/[0.06]"
    >
      {badge && (
        <span className="inline-block px-2.5 py-1 text-xs font-medium bg-lime/20 text-lime rounded mb-3">
          {badge}
        </span>
      )}
      <h1 className="text-3xl font-bold text-white mb-3">{title}</h1>
      {description && (
        <p className="text-base text-white/60 leading-relaxed max-w-2xl">
          {description}
        </p>
      )}
    </motion.header>
  )
}

interface DocSectionProps {
  id?: string
  title: string
  children: React.ReactNode
}

export function DocSection({ id, title, children }: DocSectionProps) {
  return (
    <section className="mb-10">
      <h2 
        id={id} 
        className="text-xl font-semibold text-white mb-4 scroll-mt-24"
      >
        {title}
      </h2>
      <div className="text-base text-white/70 leading-relaxed">
        {children}
      </div>
    </section>
  )
}

interface DocSubSectionProps {
  id?: string
  title: string
  children: React.ReactNode
}

export function DocSubSection({ id, title, children }: DocSubSectionProps) {
  return (
    <div className="mb-6">
      <h3 
        id={id} 
        className="text-lg font-semibold text-white/90 mb-3 scroll-mt-24"
      >
        {title}
      </h3>
      <div className="text-base text-white/70 leading-relaxed">
        {children}
      </div>
    </div>
  )
}

interface DocNavProps {
  prev?: { title: string; href: string }
  next?: { title: string; href: string }
}

export function DocNav({ prev, next }: DocNavProps) {
  return (
    <div className="mt-12 pt-6 border-t border-white/[0.06] flex justify-between gap-4">
      {prev ? (
        <a
          href={prev.href}
          className="group flex-1 p-4 rounded-lg border border-white/[0.06] hover:border-lime/30 transition-colors"
        >
          <p className="text-xs text-white/40 mb-1">Previous</p>
          <p className="text-sm text-white group-hover:text-lime transition-colors">
            ← {prev.title}
          </p>
        </a>
      ) : <div />}
      {next ? (
        <a
          href={next.href}
          className="group flex-1 p-4 rounded-lg border border-white/[0.06] hover:border-lime/30 transition-colors text-right"
        >
          <p className="text-xs text-white/40 mb-1">Next</p>
          <p className="text-sm text-white group-hover:text-lime transition-colors">
            {next.title} →
          </p>
        </a>
      ) : <div />}
    </div>
  )
}
