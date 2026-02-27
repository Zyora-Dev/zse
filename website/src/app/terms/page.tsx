'use client'

import { motion } from 'framer-motion'
import { Scale, FileText, Code, AlertTriangle, Users, Gavel, Mail } from 'lucide-react'

const sections = [
  {
    icon: FileText,
    title: 'Acceptance of Terms',
    intro: 'By accessing or using ZSE (the "Software") or this website, you agree to be bound by these Terms of Service. If you do not agree, please do not use our services.',
    bullets: [
      'These terms apply to all users, contributors, and visitors',
      'We may update these terms from time to time',
      'Continued use after changes constitutes acceptance',
    ],
  },
  {
    icon: Code,
    title: 'License Grant',
    intro: 'ZSE is open-source software licensed under the Apache License 2.0:',
    bullets: [
      'You may use, modify, and distribute the software freely',
      'You must include the original copyright notice and license',
      'You may use it for commercial purposes',
      'Modifications must be clearly marked',
      'The license does not grant trademark rights',
    ],
    outro: 'For the full license text, see our GitHub repository or visit apache.org/licenses/LICENSE-2.0.',
  },
  {
    icon: AlertTriangle,
    title: 'Disclaimer of Warranties',
    intro: 'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND:',
    bullets: [
      'No warranty of merchantability or fitness for a particular purpose',
      'No guarantee that the software will be error-free or uninterrupted',
      'Use at your own risk',
      'We do not warrant results from using the software',
    ],
    outro: 'This disclaimer is consistent with the Apache 2.0 license terms.',
  },
  {
    icon: Scale,
    title: 'Limitation of Liability',
    intro: 'To the maximum extent permitted by law:',
    bullets: [
      'Zyora Labs shall not be liable for any indirect, incidental, or consequential damages',
      'Our total liability is limited to the amount you paid us (if any)',
      'This limitation applies regardless of the legal theory',
    ],
    outro: 'Some jurisdictions do not allow these limitations, so they may not apply to you.',
  },
  {
    icon: Users,
    title: 'Community Guidelines',
    intro: 'When contributing to or participating in the ZSE community:',
    bullets: [
      'Be respectful and inclusive to all community members',
      'Do not submit malicious code or intentionally harmful contributions',
      'Follow our contribution guidelines on GitHub',
      'Report security vulnerabilities responsibly',
      'Do not use the project for illegal activities',
    ],
  },
  {
    icon: Gavel,
    title: 'Governing Law',
    intro: 'These terms are governed by the laws of India:',
    bullets: [
      'Any disputes shall be resolved in the courts of Tamil Nadu, India',
      'If any provision is found unenforceable, the rest remains in effect',
      'Our failure to enforce a right does not waive that right',
    ],
  },
  {
    icon: Mail,
    title: 'Contact',
    intro: 'For questions about these terms:',
    items: [
      { label: 'Email', text: 'zse@zyoralabs.com' },
      { label: 'GitHub', text: 'Open an issue on our repository' },
    ],
  },
]

export default function TermsPage() {
  return (
    <div className="min-h-screen bg-black pt-24 pb-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center justify-center w-16 h-16 bg-lime/10 rounded-2xl mb-6">
            <Scale className="w-8 h-8 text-lime" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-3">
            Terms of Service
          </h1>
          <p className="text-base text-white/60">
            Last updated: February 2025
          </p>
        </motion.div>

        {/* TL;DR Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-lime/5 border border-lime/20 rounded-xl p-6 mb-12"
        >
          <h2 className="text-lg font-semibold text-lime mb-3">TL;DR</h2>
          <ul className="space-y-2 text-white/80">
            <li className="flex items-start gap-2">
              <span className="text-lime mt-1">✓</span>
              <span>ZSE is free, open-source under Apache 2.0</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-lime mt-1">✓</span>
              <span>Use it for anything, including commercial projects</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-lime mt-1">✓</span>
              <span>No warranties — use at your own risk</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-lime mt-1">✓</span>
              <span>Be respectful when participating in our community</span>
            </li>
          </ul>
        </motion.div>

        {/* Sections */}
        <div className="space-y-8">
          {sections.map((section, index) => (
            <motion.section
              key={section.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 + index * 0.05 }}
              className="border border-white/10 rounded-xl p-8 bg-white/[0.02]"
            >
              <div className="flex items-start gap-4 mb-4">
                <div className="w-10 h-10 bg-white/5 rounded-lg flex items-center justify-center flex-shrink-0">
                  <section.icon className="w-5 h-5 text-lime" />
                </div>
                <h2 className="text-xl font-semibold text-white pt-1.5">{section.title}</h2>
              </div>
              <div className="text-white/70 leading-relaxed space-y-4">
                {section.intro && <p>{section.intro}</p>}
                {section.items && (
                  <ul className="space-y-2">
                    {section.items.map((item, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-lime">•</span>
                        <span><strong className="text-white font-semibold">{item.label}:</strong> {item.text}</span>
                      </li>
                    ))}
                  </ul>
                )}
                {section.bullets && (
                  <ul className="space-y-2">
                    {section.bullets.map((bullet, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-lime">•</span>
                        <span>{bullet}</span>
                      </li>
                    ))}
                  </ul>
                )}
                {section.outro && <p>{section.outro}</p>}
              </div>
            </motion.section>
          ))}
        </div>

        {/* Footer Note */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center"
        >
          <p className="text-white/40 text-sm">
            These terms apply to the ZSE website and software maintained by Zyora Labs.
            <br />
            For licensing questions, refer to the Apache 2.0 license in our repository.
          </p>
        </motion.div>
      </div>
    </div>
  )
}
