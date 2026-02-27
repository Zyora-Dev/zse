'use client'

import { motion } from 'framer-motion'
import { Shield, Eye, Database, Mail, Globe, Lock, FileText } from 'lucide-react'

const sections = [
  {
    icon: Eye,
    title: 'Information We Collect',
    intro: 'As an open-source project, we collect minimal information:',
    items: [
      { label: 'Website Analytics', text: 'Basic, anonymized usage data (page views, country) to improve our documentation. We do not use cookies for tracking.' },
      { label: 'Contact Form', text: 'When you contact us, we collect your name, email, and message to respond to your inquiry.' },
      { label: 'GitHub', text: "If you contribute to ZSE, GitHub collects information according to their privacy policy." },
      { label: 'PyPI', text: 'Package download statistics are collected by PyPI, not by us.' },
    ],
  },
  {
    icon: Database,
    title: 'How We Use Your Information',
    intro: 'We use the information we collect to:',
    bullets: [
      'Respond to your inquiries and support requests',
      'Improve our documentation and website',
      "Communicate important updates about ZSE (only if you've opted in)",
    ],
    outro: 'We never sell your data. As an open-source project, our goal is to build great software, not to monetize your information.',
  },
  {
    icon: Lock,
    title: 'Data Security',
    intro: 'We take reasonable measures to protect your information:',
    bullets: [
      'All website traffic is encrypted via HTTPS',
      'Contact form submissions are transmitted securely',
      'We limit access to personal information to team members who need it',
    ],
    outro: 'As an open-source project, our codebase is publicly auditable.',
  },
  {
    icon: Globe,
    title: 'Third-Party Services',
    intro: 'Our website may use the following third-party services:',
    items: [
      { label: 'Vercel', text: "Website hosting (see Vercel's privacy policy)" },
      { label: 'GitHub', text: 'Source code hosting and collaboration' },
      { label: 'PyPI', text: 'Python package distribution' },
      { label: 'ZeptoMail', text: 'Email delivery for contact form' },
    ],
    outro: 'Each service has its own privacy policy governing their data practices.',
  },
  {
    icon: FileText,
    title: 'Open Source Commitment',
    intro: 'ZSE is licensed under Apache 2.0, which means:',
    bullets: [
      'The source code is freely available for inspection',
      'You can verify exactly what our software does',
      'No hidden telemetry or data collection in the ZSE library',
      'Community contributions are welcome and transparent',
    ],
    outro: 'We believe in transparency and user privacy as core values.',
  },
  {
    icon: Mail,
    title: 'Contact Us',
    intro: 'If you have questions about this privacy policy or your data:',
    items: [
      { label: 'Email', text: 'zse@zyoralabs.com' },
      { label: 'GitHub', text: 'Open an issue on our repository' },
    ],
    outro: "We'll respond to privacy-related inquiries within 30 days.",
  },
]

export default function PrivacyPage() {
  return (
    <div className="min-h-screen bg-black pt-24 pb-16">
      <div className="max-w-4xl mx-auto px-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center justify-center w-16 h-16 bg-lime/10 rounded-2xl mb-6">
            <Shield className="w-8 h-8 text-lime" />
          </div>
          <h1 className="text-3xl font-bold text-white mb-3">
            Privacy Policy
          </h1>
          <p className="text-base text-white/60 max-w-2xl mx-auto">
            Your privacy matters. As an open-source project, we're committed to transparency.
          </p>
          <p className="mt-4 text-sm text-white/40">
            Last updated: February 25, 2026
          </p>
        </motion.div>

        {/* Summary Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-12 p-6 rounded-xl bg-lime/5 border border-lime/20"
        >
          <h2 className="text-lg font-semibold text-lime mb-3">TL;DR</h2>
          <ul className="space-y-2 text-white/80">
            <li className="flex items-start gap-2">
              <span className="text-lime mt-1">✓</span>
              <span>We collect minimal data needed to operate</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-lime mt-1">✓</span>
              <span>We never sell your personal information</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-lime mt-1">✓</span>
              <span>ZSE library contains no telemetry or tracking</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-lime mt-1">✓</span>
              <span>Our code is open source and auditable</span>
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
            This privacy policy applies to the ZSE website and related services operated by Zyora Labs.
            <br />
            The ZSE open-source library itself does not collect any user data.
          </p>
        </motion.div>
      </div>
    </div>
  )
}
