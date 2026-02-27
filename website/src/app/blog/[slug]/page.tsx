'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { getBlogPost, blogPosts } from '@/lib/blog'
import { formatDate } from '@/lib/utils'
import { ArrowLeft } from 'lucide-react'

export default function BlogPostPage() {
  const params = useParams()
  const slug = params.slug as string
  const post = getBlogPost(slug)

  if (!post) {
    return (
      <div className="min-h-screen bg-black pt-24 pb-16">
        <div className="max-w-3xl mx-auto px-6 text-center">
          <h1 className="text-2xl font-bold text-white mb-4">Post not found</h1>
          <Link href="/blog" className="text-lime hover:underline">
            ← Back to blog
          </Link>
        </div>
      </div>
    )
  }

  // Get related posts
  const relatedPosts = blogPosts
    .filter(p => p.slug !== post.slug && p.tags.some(t => post.tags.includes(t)))
    .slice(0, 3)

  return (
    <div className="min-h-screen bg-black pt-24 pb-16">
      <article className="max-w-3xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Link
            href="/blog"
            className="inline-flex items-center gap-2 text-white/60 hover:text-lime transition-colors mb-8"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to blog
          </Link>

          <header className="mb-12">
            <div className="flex gap-2 mb-4">
              {post.tags.map(tag => (
                <span
                  key={tag}
                  className="px-3 py-1 text-xs font-medium bg-lime/20 text-lime rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6">
              {post.title}
            </h1>
            <div className="flex items-center gap-4 text-white/60">
              <span>{post.author}</span>
              <span>•</span>
              <span>{formatDate(post.date)}</span>
              <span>•</span>
              <span>{post.readTime}</span>
            </div>
          </header>

          <div className="prose prose-invert prose-lg max-w-none">
            {post.content.split('\n').map((paragraph, i) => {
              if (paragraph.startsWith('# ')) {
                return (
                  <h1 key={i} className="text-3xl font-bold text-white mt-12 mb-6">
                    {paragraph.slice(2)}
                  </h1>
                )
              }
              if (paragraph.startsWith('## ')) {
                return (
                  <h2 key={i} className="text-2xl font-semibold text-white mt-10 mb-4">
                    {paragraph.slice(3)}
                  </h2>
                )
              }
              if (paragraph.startsWith('### ')) {
                return (
                  <h3 key={i} className="text-xl font-semibold text-white mt-8 mb-3">
                    {paragraph.slice(4)}
                  </h3>
                )
              }
              if (paragraph.startsWith('```')) {
                return null // Skip code block markers
              }
              if (paragraph.startsWith('- **')) {
                const match = paragraph.match(/\*\*(.+?)\*\*: (.+)/)
                if (match) {
                  return (
                    <p key={i} className="text-white/80 mb-2 pl-4">
                      <strong className="text-lime">{match[1]}</strong>: {match[2]}
                    </p>
                  )
                }
              }
              if (paragraph.startsWith('- ')) {
                return (
                  <p key={i} className="text-white/80 mb-2 pl-4">
                    • {paragraph.slice(2)}
                  </p>
                )
              }
              if (paragraph.startsWith('|')) {
                // Table handling
                const cells = paragraph.split('|').filter(c => c.trim())
                if (cells.every(c => c.includes('---'))) return null
                return (
                  <div key={i} className="flex border-b border-white/10 py-2">
                    {cells.map((cell, j) => (
                      <div
                        key={j}
                        className={`flex-1 ${j === 0 ? 'font-medium text-white' : 'text-white/60'}`}
                      >
                        {cell.trim().replace(/\*\*/g, '')}
                      </div>
                    ))}
                  </div>
                )
              }
              if (paragraph.includes('`')) {
                const parts = paragraph.split(/(`[^`]+`)/)
                return (
                  <p key={i} className="text-white/80 mb-4">
                    {parts.map((part, j) =>
                      part.startsWith('`') ? (
                        <code
                          key={j}
                          className="bg-white/10 px-2 py-0.5 rounded text-lime text-sm"
                        >
                          {part.slice(1, -1)}
                        </code>
                      ) : (
                        part
                      )
                    )}
                  </p>
                )
              }
              if (paragraph.trim()) {
                return (
                  <p key={i} className="text-white/80 mb-4">
                    {paragraph}
                  </p>
                )
              }
              return null
            })}
          </div>

          {relatedPosts.length > 0 && (
            <section className="mt-16 pt-12 border-t border-white/10">
              <h2 className="text-xl font-semibold text-white mb-6">Related Posts</h2>
              <div className="grid gap-4">
                {relatedPosts.map(related => (
                  <Link
                    key={related.slug}
                    href={`/blog/${related.slug}`}
                    className="block p-4 border border-white/10 rounded-lg hover:border-lime/50 transition-colors"
                  >
                    <h3 className="text-white font-medium hover:text-lime transition-colors">
                      {related.title}
                    </h3>
                    <p className="text-white/60 text-sm mt-1">{related.excerpt}</p>
                  </Link>
                ))}
              </div>
            </section>
          )}
        </motion.div>
      </article>
    </div>
  )
}
