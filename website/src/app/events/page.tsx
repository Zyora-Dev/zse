'use client'

import Link from 'next/link'
import { motion, useScroll, useTransform, useInView } from 'framer-motion'
import { useRef } from 'react'
import { events, formatEventDate, Event } from '@/lib/events'
import { Calendar, MapPin, Clock, ExternalLink, Bell, Sparkles } from 'lucide-react'

const eventTypeColors: Record<string, { bg: string; text: string }> = {
  meetup: { bg: 'bg-blue-500/20', text: 'text-blue-400' },
  webinar: { bg: 'bg-purple-500/20', text: 'text-purple-400' },
  conference: { bg: 'bg-orange-500/20', text: 'text-orange-400' },
  workshop: { bg: 'bg-green-500/20', text: 'text-green-400' },
}

// Scroll-animated section wrapper
function SectionWithScrollAnimation({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })
  
  return (
    <motion.section
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={`mb-16 ${className}`}
    >
      {children}
    </motion.section>
  )
}

function EventCard({ event, index }: { event: Event; index: number }) {
  const colors = eventTypeColors[event.type]
  const isPast = new Date(event.date) < new Date()

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      className={`border border-white/10 rounded-xl p-6 ${isPast ? 'opacity-60' : ''} hover:border-lime/50 transition-all duration-300 bg-white/[0.02]`}
    >
      <div className="flex items-start justify-between gap-4 mb-4">
        <div>
          <span className={`inline-block px-3 py-1 text-xs font-medium rounded-full ${colors.bg} ${colors.text} mb-3`}>
            {event.type}
          </span>
          {event.featured && !isPast && (
            <span className="inline-block ml-2 px-3 py-1 text-xs font-medium bg-lime/20 text-lime rounded-full mb-3">
              Featured
            </span>
          )}
          <h3 className="text-xl font-semibold text-white">{event.title}</h3>
        </div>
        {isPast && (
          <span className="text-xs text-white/40 bg-white/5 px-2 py-1 rounded">
            Past Event
          </span>
        )}
      </div>

      <p className="text-white/60 mb-6">{event.description}</p>

      <div className="space-y-2 text-sm text-white/50">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4" />
          <span>{formatEventDate(event.date)}</span>
        </div>
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4" />
          <span>{event.time}</span>
        </div>
        <div className="flex items-center gap-2">
          <MapPin className="w-4 h-4" />
          <span>{event.location}</span>
        </div>
      </div>

      {event.link && !isPast && (
        <a
          href={event.link}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 mt-6 px-4 py-2 bg-lime text-black font-medium rounded-lg hover:bg-lime/90 transition-colors"
        >
          Register
          <ExternalLink className="w-4 h-4" />
        </a>
      )}
    </motion.div>
  )
}

export default function EventsPage() {
  const now = new Date()
  const pastEvents = events
    .filter(e => new Date(e.date) < now)
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())

  const headerRef = useRef(null)
  const { scrollYProgress } = useScroll({
    target: headerRef,
    offset: ["start start", "end start"]
  })
  
  const headerY = useTransform(scrollYProgress, [0, 1], [0, 100])
  const headerOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const headerScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.95])

  // Character animation variants
  const titleText = "Events"
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.08, delayChildren: 0.2 }
    }
  }
  const letterVariants = {
    hidden: { opacity: 0, y: 50, rotateX: -90 },
    visible: {
      opacity: 1,
      y: 0,
      rotateX: 0,
      transition: { type: "spring", damping: 12, stiffness: 100 }
    }
  }

  return (
    <div className="min-h-screen bg-black pt-24 pb-16">
      <div className="max-w-4xl mx-auto px-6">
        {/* Innovative Header */}
        <motion.div
          ref={headerRef}
          style={{ y: headerY, opacity: headerOpacity, scale: headerScale }}
          className="relative text-center mb-20 py-8"
        >
          {/* Background glow effect */}
          <div className="absolute inset-0 -z-10">
            <motion.div
              animate={{ 
                scale: [1, 1.2, 1],
                opacity: [0.3, 0.5, 0.3]
              }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[300px] bg-lime/20 rounded-full blur-[100px]"
            />
          </div>

          {/* Floating decorative elements */}
          <div className="absolute inset-0 overflow-hidden -z-10">
            {[...Array(6)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-1 bg-lime/40 rounded-full"
                style={{
                  left: `${15 + i * 15}%`,
                  top: `${20 + (i % 3) * 25}%`,
                }}
                animate={{
                  y: [0, -20, 0],
                  opacity: [0.2, 0.6, 0.2],
                  scale: [1, 1.5, 1],
                }}
                transition={{
                  duration: 3 + i * 0.5,
                  repeat: Infinity,
                  delay: i * 0.3,
                }}
              />
            ))}
          </div>

          {/* Icon with animated ring */}
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: "spring", damping: 15, stiffness: 100 }}
            className="relative inline-flex items-center justify-center mb-8"
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="absolute w-20 h-20 rounded-full border border-lime/20 border-dashed"
            />
            <motion.div
              animate={{ rotate: -360 }}
              transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
              className="absolute w-24 h-24 rounded-full border border-lime/10"
              style={{ borderStyle: 'dotted' }}
            />
            <div className="w-14 h-14 bg-gradient-to-br from-lime/20 to-lime/5 rounded-2xl flex items-center justify-center backdrop-blur-sm border border-lime/20">
              <Calendar className="w-7 h-7 text-lime" />
            </div>
          </motion.div>

          {/* Animated title with character animation */}
          <motion.h1
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="text-3xl font-bold mb-4 flex justify-center gap-1"
          >
            {titleText.split('').map((char, i) => (
              <motion.span
                key={i}
                variants={letterVariants}
                className="inline-block bg-gradient-to-b from-white via-white to-white/60 bg-clip-text text-transparent"
              >
                {char}
              </motion.span>
            ))}
          </motion.h1>

          {/* Subtitle with line animation */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="relative"
          >
            <p className="text-base text-white/60 max-w-md mx-auto">
              Join us at upcoming events, webinars, and workshops
            </p>
            
            {/* Animated underline */}
            <motion.div
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ delay: 0.8, duration: 0.6 }}
              className="mt-4 mx-auto w-24 h-px bg-gradient-to-r from-transparent via-lime/50 to-transparent"
            />
          </motion.div>

          {/* Sparkle decorations */}
          <motion.div
            animate={{ rotate: [0, 15, 0, -15, 0] }}
            transition={{ duration: 5, repeat: Infinity }}
            className="absolute top-4 right-1/4"
          >
            <Sparkles className="w-4 h-4 text-lime/40" />
          </motion.div>
          <motion.div
            animate={{ rotate: [0, -15, 0, 15, 0] }}
            transition={{ duration: 4, repeat: Infinity, delay: 1 }}
            className="absolute bottom-8 left-1/4"
          >
            <Sparkles className="w-3 h-3 text-lime/30" />
          </motion.div>
        </motion.div>

        {/* No Upcoming Events - Stay Tuned */}
        <SectionWithScrollAnimation>
          <h2 className="text-lg font-semibold text-lime mb-6 flex items-center gap-2">
            <motion.span
              initial={{ width: 0 }}
              whileInView={{ width: 24 }}
              transition={{ duration: 0.5 }}
              className="h-px bg-lime inline-block"
            />
            Upcoming Events
          </h2>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="border border-white/10 rounded-xl p-12 bg-white/[0.02] text-center"
          >
            {/* Animated Bell Icon */}
            <motion.div
              animate={{ 
                rotate: [0, -10, 10, -10, 10, 0],
                scale: [1, 1.1, 1]
              }}
              transition={{ 
                duration: 2,
                repeat: Infinity,
                repeatDelay: 3
              }}
              className="inline-flex items-center justify-center w-20 h-20 bg-lime/10 rounded-full mb-6"
            >
              <Bell className="w-10 h-10 text-lime" />
            </motion.div>

            <motion.h3
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-2xl font-semibold text-white mb-3"
            >
              No New Events
            </motion.h3>

            <motion.p
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-white/60 mb-6 max-w-md mx-auto"
            >
              We're planning something exciting! Stay tuned for upcoming webinars, workshops, and meetups.
            </motion.p>

            {/* Animated dots */}
            <motion.div className="flex justify-center gap-2 mb-8">
              {[0, 1, 2].map((i) => (
                <motion.span
                  key={i}
                  animate={{ 
                    opacity: [0.3, 1, 0.3],
                    scale: [0.8, 1.2, 0.8]
                  }}
                  transition={{ 
                    duration: 1.5,
                    repeat: Infinity,
                    delay: i * 0.2
                  }}
                  className="w-2 h-2 bg-lime rounded-full"
                />
              ))}
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="flex flex-col sm:flex-row gap-4 justify-center"
            >
              <a
                href="https://discord.gg/f9JKreJA"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-lime text-black font-medium rounded-lg hover:bg-lime/90 transition-colors"
              >
                Join Discord for Updates
              </a>
              <Link
                href="/contact"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 border border-white/20 text-white font-medium rounded-lg hover:bg-white/5 transition-colors"
              >
                Get Notified
              </Link>
            </motion.div>
          </motion.div>
        </SectionWithScrollAnimation>

        {pastEvents.length > 0 && (
          <SectionWithScrollAnimation>
            <h2 className="text-lg font-semibold text-white/60 mb-6 flex items-center gap-2">
              <motion.span
                initial={{ width: 0 }}
                whileInView={{ width: 24 }}
                transition={{ duration: 0.5 }}
                className="h-px bg-white/40 inline-block"
              />
              Past Events
            </h2>
            <div className="grid gap-6">
              {pastEvents.map((event, index) => (
                <EventCard key={event.id} event={event} index={index} />
              ))}
            </div>
          </SectionWithScrollAnimation>
        )}

        {events.length === 0 && pastEvents.length === 0 && (
          <div className="text-center py-12">
            <p className="text-white/60">No events scheduled yet. Check back soon!</p>
          </div>
        )}

        <motion.section
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="mt-16 p-8 border border-white/10 rounded-xl bg-white/[0.02] text-center relative overflow-hidden"
        >
          {/* Background decoration */}
          <div className="absolute inset-0 -z-10">
            <div className="absolute top-0 right-0 w-32 h-32 bg-lime/5 rounded-full blur-3xl" />
            <div className="absolute bottom-0 left-0 w-24 h-24 bg-lime/5 rounded-full blur-2xl" />
          </div>
          
          <motion.h2 
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-2xl font-semibold text-white mb-4"
          >
            Want to host a ZSE event?
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3 }}
            className="text-white/60 mb-6"
          >
            We'd love to collaborate with communities, universities, and companies to spread knowledge about efficient LLM inference.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
          >
            <Link
              href="/contact"
              className="inline-flex items-center gap-2 px-6 py-3 bg-lime text-black font-medium rounded-lg hover:bg-lime/90 transition-colors"
            >
              Get in touch
            </Link>
          </motion.div>
        </motion.section>
      </div>
    </div>
  )
}
