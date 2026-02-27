'use client'

import { motion } from 'framer-motion'
import { Info, AlertTriangle, CheckCircle, Lightbulb, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

type CalloutType = 'info' | 'warning' | 'success' | 'tip' | 'danger'

interface CalloutProps {
  type?: CalloutType
  title?: string
  children: React.ReactNode
}

const calloutConfig: Record<CalloutType, { 
  icon: React.ElementType
  bg: string
  border: string
  iconColor: string
  titleColor: string
}> = {
  info: {
    icon: Info,
    bg: 'bg-blue-500/5',
    border: 'border-blue-500/20',
    iconColor: 'text-blue-400',
    titleColor: 'text-blue-300',
  },
  warning: {
    icon: AlertTriangle,
    bg: 'bg-yellow-500/5',
    border: 'border-yellow-500/20',
    iconColor: 'text-yellow-400',
    titleColor: 'text-yellow-300',
  },
  success: {
    icon: CheckCircle,
    bg: 'bg-green-500/5',
    border: 'border-green-500/20',
    iconColor: 'text-green-400',
    titleColor: 'text-green-300',
  },
  tip: {
    icon: Lightbulb,
    bg: 'bg-lime/5',
    border: 'border-lime/20',
    iconColor: 'text-lime',
    titleColor: 'text-lime',
  },
  danger: {
    icon: AlertCircle,
    bg: 'bg-red-500/5',
    border: 'border-red-500/20',
    iconColor: 'text-red-400',
    titleColor: 'text-red-300',
  },
}

export function Callout({ type = 'info', title, children }: CalloutProps) {
  const config = calloutConfig[type]
  const Icon = config.icon

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      className={cn(
        "my-4 px-4 py-3 rounded-lg border-l-2",
        config.bg,
        config.border
      )}
    >
      <div className="flex gap-3">
        <Icon className={cn("w-4 h-4 mt-0.5 flex-shrink-0", config.iconColor)} />
        <div className="flex-1 min-w-0">
          {title && (
            <p className={cn("text-xs font-semibold uppercase tracking-wider mb-1", config.titleColor)}>
              {title}
            </p>
          )}
          <div className="text-sm text-white/70 leading-relaxed">
            {children}
          </div>
        </div>
      </div>
    </motion.div>
  )
}
