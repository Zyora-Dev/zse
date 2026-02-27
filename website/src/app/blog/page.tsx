'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { blogPosts, BlogPost } from '@/lib/blog'
import { formatDate } from '@/lib/utils'

function BlogCard({ post, index }: { post: BlogPost; index: number }) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className="group"
    >
      <Link href={`/blog/${post.slug}`}>
        <div className="border border-white/10 rounded-xl p-6 hover:border-lime/50 transition-all duration-300 bg-white/[0.02] hover:bg-white/[0.04]">
          {post.featured && (
            <span className="inline-block px-3 py-1 text-xs font-medium bg-lime/20 text-lime rounded-full mb-4">
              Featured
            </span>
          )}
          <h2 className="text-xl font-semibold text-white mb-3 group-hover:text-lime transition-colors">
            {post.title}
          </h2>
          <p className="text-white/60 mb-4 line-clamp-2">{post.excerpt}</p>
          <div className="flex items-center justify-between text-sm text-white/40">
            <div className="flex items-center gap-2">
              <span>{post.author}</span>
              <span>•</span>
              <span>{formatDate(post.date)}</span>
            </div>
            <span>{post.readTime}</span>
          </div>
          <div className="flex gap-2 mt-4">
            {post.tags.map(tag => (
              <span
                key={tag}
                className="px-2 py-1 text-xs bg-white/5 text-white/50 rounded"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      </Link>
    </motion.article>
  )
}

export default function BlogPage() {
  const featuredPosts = blogPosts.filter(p => p.featured)
  const regularPosts = blogPosts.filter(p => !p.featured)

  return (
    <div className="min-h-screen bg-black pt-24 pb-16">
      <div className="max-w-5xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">Blog</h1>
          <p className="text-xl text-white/60">
            Updates, tutorials, and insights from the ZSE team
          </p>
        </motion.div>

        {featuredPosts.length > 0 && (
          <section className="mb-12">
            <h2 className="text-lg font-semibold text-lime mb-6">Featured</h2>
            <div className="space-y-6">
              {featuredPosts.map((post, index) => (
                <BlogCard key={post.id} post={post} index={index} />
              ))}
            </div>
          </section>
        )}

        <section>
          <h2 className="text-lg font-semibold text-white/80 mb-6">All Posts</h2>
          <div className="space-y-6">
            {regularPosts.map((post, index) => (
              <BlogCard key={post.id} post={post} index={index + featuredPosts.length} />
            ))}
          </div>
        </section>

        {blogPosts.length === 0 && (
          <div className="text-center py-12">
            <p className="text-white/60">No blog posts yet. Check back soon!</p>
          </div>
        )}
      </div>
    </div>
  )
}
