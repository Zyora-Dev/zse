'use client'

import Link from 'next/link'
import Image from 'next/image'
import { motion } from 'framer-motion'
import { useState } from 'react'
import { Menu, X, Github, BookOpen, Newspaper, Calendar, Users, MessageSquare, BarChart2 } from 'lucide-react'

const navItems = [
  { href: '/docs', label: 'Documentation', icon: BookOpen },
  { href: '/benchmarks', label: 'Benchmarks', icon: BarChart2 },
  { href: '/blog', label: 'Blog', icon: Newspaper },
  { href: '/events', label: 'Events', icon: Calendar },
  { href: '/about', label: 'About', icon: Users },
  { href: '/community', label: 'Community', icon: MessageSquare },
]

export function Navigation() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-lg border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="flex items-center"
            >
              <Image
                src="/images/zllm-logo.png"
                alt="ZSE Logo"
                width={120}
                height={40}
                className="h-8 w-auto"
                priority
              />
            </motion.div>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="px-4 py-2 text-sm text-gray-400 hover:text-white hover:bg-white/5 rounded-lg transition-all duration-200"
              >
                {item.label}
              </Link>
            ))}
          </div>

          {/* Actions */}
          <div className="hidden md:flex items-center gap-3">
            <Link
              href="https://github.com/Zyora-Dev/zse"
              target="_blank"
              className="flex items-center gap-2 px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors"
            >
              <Github className="w-4 h-4" />
              GitHub
            </Link>
            <Link
              href="/docs/quickstart"
              className="px-4 py-2 text-sm font-medium bg-lime text-black rounded-lg hover:bg-lime/90 transition-colors"
            >
              Get Started
            </Link>
          </div>

          {/* Mobile menu button */}
          <button
            className="md:hidden p-2 text-gray-400 hover:text-white"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation */}
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="md:hidden bg-black/95 border-b border-white/5"
        >
          <div className="px-4 py-4 space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsOpen(false)}
                className="flex items-center gap-3 px-4 py-3 text-gray-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
              >
                <item.icon className="w-5 h-5 text-lime" />
                {item.label}
              </Link>
            ))}
            <hr className="border-white/10 my-4" />
            <Link
              href="https://github.com/Zyora-Dev/zse"
              target="_blank"
              className="flex items-center gap-3 px-4 py-3 text-gray-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
            >
              <Github className="w-5 h-5" />
              GitHub
            </Link>
            <Link
              href="/docs/quickstart"
              className="block px-4 py-3 text-center font-medium bg-lime text-black rounded-lg hover:bg-lime/90 transition-colors"
            >
              Get Started
            </Link>
          </div>
        </motion.div>
      )}
    </nav>
  )
}
