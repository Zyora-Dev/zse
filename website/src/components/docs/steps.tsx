'use client'

import { motion } from 'framer-motion'
import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Step {
  title: string
  description?: string
  content?: React.ReactNode
  code?: string
}

interface StepsProps {
  steps: Step[]
}

export function Steps({ steps }: StepsProps) {
  return (
    <div className="my-6 space-y-0">
      {steps.map((step, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="relative pl-8 pb-8 last:pb-0"
        >
          {/* Vertical line */}
          {index < steps.length - 1 && (
            <div className="absolute left-[11px] top-6 bottom-0 w-px bg-white/10" />
          )}
          
          {/* Step number */}
          <div className="absolute left-0 top-0 w-6 h-6 rounded-full bg-lime/20 border border-lime/40 flex items-center justify-center">
            <span className="text-xs font-bold text-lime">{index + 1}</span>
          </div>

          {/* Content */}
          <div>
            <h4 className="text-base font-semibold text-white mb-1">
              {step.title}
            </h4>
            {step.description && (
              <p className="text-sm text-white/50 mb-3">
                {step.description}
              </p>
            )}
            {step.code && (
              <pre className="bg-white/[0.03] border border-white/[0.06] rounded-lg p-3 overflow-x-auto my-2">
                <code className="text-sm text-lime/90 font-mono">{step.code}</code>
              </pre>
            )}
            {step.content && (
              <div className="text-sm text-white/70">
                {step.content}
              </div>
            )}
          </div>
        </motion.div>
      ))}
    </div>
  )
}

interface FeatureListProps {
  features: string[]
}

export function FeatureList({ features }: FeatureListProps) {
  return (
    <ul className="my-4 space-y-2">
      {features.map((feature, index) => (
        <motion.li
          key={index}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.05 }}
          className="flex items-start gap-2"
        >
          <Check className="w-4 h-4 text-lime mt-0.5 flex-shrink-0" />
          <span className="text-sm text-white/70">{feature}</span>
        </motion.li>
      ))}
    </ul>
  )
}

interface CardProps {
  title: string
  description?: string
  icon?: React.ElementType
  href?: string
  children?: React.ReactNode
}

export function Card({ title, description, icon: Icon, href, children }: CardProps) {
  const Wrapper = href ? 'a' : 'div'
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={href ? { y: -2 } : undefined}
    >
      <Wrapper
        {...(href ? { href, className: "block" } : {})}
        className={cn(
          "p-4 rounded-lg border border-white/[0.06] bg-white/[0.02]",
          href && "hover:border-lime/30 hover:bg-white/[0.04] transition-all cursor-pointer"
        )}
      >
        {Icon && (
          <div className="w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center mb-3">
            <Icon className="w-4 h-4 text-lime" />
          </div>
        )}
        <h4 className="text-base font-semibold text-white mb-1">{title}</h4>
        {description && (
          <p className="text-sm text-white/50">{description}</p>
        )}
        {children}
      </Wrapper>
    </motion.div>
  )
}

interface CardGridProps {
  children: React.ReactNode
  columns?: 2 | 3
}

export function CardGrid({ children, columns = 2 }: CardGridProps) {
  return (
    <div className={cn(
      "grid gap-4 my-6",
      columns === 2 && "md:grid-cols-2",
      columns === 3 && "md:grid-cols-3"
    )}>
      {children}
    </div>
  )
}
