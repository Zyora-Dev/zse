import Link from 'next/link'
import Image from 'next/image'
import { Github, Twitter, Linkedin } from 'lucide-react'
import { DiscordIcon } from './icons/discord'

const footerLinks = {
  Product: [
    { href: '/docs', label: 'Documentation' },
    { href: '/docs/quickstart', label: 'Quick Start' },
    { href: '/benchmarks', label: 'Benchmarks' },
    { href: '/blog', label: 'Blog' },
    { href: '/changelog', label: 'Changelog' },
  ],
  Resources: [
    { href: '/community', label: 'Community' },
    { href: '/events', label: 'Events' },
    { href: 'https://github.com/Zyora-Dev/zse', label: 'GitHub' },
    { href: 'https://pypi.org/project/zllm-zse/', label: 'PyPI' },
  ],
  Company: [
    { href: '/about', label: 'About' },
    { href: '/contact', label: 'Contact' },
    { href: '/privacy', label: 'Privacy' },
    { href: '/terms', label: 'Terms' },
  ],
}

export function Footer() {
  return (
    <footer className="bg-black border-t border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <Link href="/" className="inline-block">
              <Image
                src="/images/zllm-logo.png"
                alt="ZSE Logo"
                width={120}
                height={40}
                className="h-8 w-auto"
              />
            </Link>
            <p className="mt-4 text-sm text-gray-500">
              Ultra memory-efficient LLM inference engine
            </p>
            <div className="flex gap-4 mt-6">
              <Link
                href="https://github.com/Zyora-Dev/zse"
                target="_blank"
                className="text-gray-500 hover:text-lime transition-colors"
              >
                <Github className="w-5 h-5" />
              </Link>
              <Link
                href="https://x.com/zyoralabs"
                target="_blank"
                className="text-gray-500 hover:text-lime transition-colors"
              >
                <Twitter className="w-5 h-5" />
              </Link>
              <Link
                href="https://discord.gg/f9JKreJA"
                target="_blank"
                className="text-gray-500 hover:text-lime transition-colors"
              >
                <DiscordIcon className="w-5 h-5" />
              </Link>
              <Link
                href="https://www.linkedin.com/company/zyora-labs/"
                target="_blank"
                className="text-gray-500 hover:text-lime transition-colors"
              >
                <Linkedin className="w-5 h-5" />
              </Link>
            </div>
          </div>

          {/* Links */}
          {Object.entries(footerLinks).map(([title, links]) => (
            <div key={title}>
              <h3 className="text-sm font-semibold text-white mb-4">{title}</h3>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      className="text-sm text-gray-500 hover:text-lime transition-colors"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom */}
        <div className="mt-12 pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-gray-600">
            © {new Date().getFullYear()} Zyora Labs. All rights reserved.
          </p>
          <div className="flex flex-col items-center md:items-end gap-1">
            <p className="text-sm text-gray-500">
              யாதும் ஊரே, யாவரும் கேளிர் <span className="text-lime">✦</span>
            </p>
            <p className="text-xs text-gray-600">
              Every place is home, everyone is kin
            </p>
          </div>
        </div>
      </div>
    </footer>
  )
}
