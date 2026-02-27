import type { Metadata } from 'next'
import { Montserrat } from 'next/font/google'
import './globals.css'
import { Navigation } from '@/components/navigation'
import { Footer } from '@/components/footer'

const montserrat = Montserrat({ 
  subsets: ['latin'],
  variable: '--font-montserrat',
})

export const metadata: Metadata = {
  metadataBase: new URL('https://zllm.in'),
  title: 'ZSE - Z Server Engine',
  description: 'Ultra memory-efficient LLM inference engine. 3.9s cold start for 7B models.',
  keywords: ['LLM', 'inference', 'GPU', 'CUDA', 'machine learning', 'AI', 'memory efficient'],
  authors: [{ name: 'Zyora Labs' }],
  icons: {
    icon: '/images/zllm-fav.png',
    shortcut: '/images/zllm-fav.png',
    apple: '/images/zllm-fav.png',
  },
  openGraph: {
    title: 'ZSE - Z Server Engine',
    description: 'Ultra memory-efficient LLM inference engine',
    url: 'https://zllm.in',
    siteName: 'ZSE',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ZSE - Z Server Engine',
    description: 'Ultra memory-efficient LLM inference engine',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${montserrat.variable} font-sans antialiased min-h-screen bg-black`}>
        <Navigation />
        <main className="min-h-screen">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  )
}
