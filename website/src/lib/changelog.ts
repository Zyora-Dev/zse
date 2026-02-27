export interface ChangelogEntry {
  version: string
  date: string
  title: string
  description: string
  changes: {
    type: 'added' | 'changed' | 'fixed' | 'removed' | 'security'
    text: string
  }[]
}

export const changelog: ChangelogEntry[] = [
  {
    version: '0.1.3',
    date: '2026-02-27',
    title: '72B Model Benchmark Support',
    description: 'Verified 72B model benchmarks showing 79× speedup over bitsandbytes.',
    changes: [
      { type: 'added', text: 'Verified Qwen 72B benchmark: 6.5s cold start (79× faster than bitsandbytes)' },
      { type: 'added', text: 'llama.cpp GGUF comparison: ZSE 1.6× faster on 72B models' },
      { type: 'changed', text: 'Updated website with H200 GPU benchmark results' },
      { type: 'added', text: 'New benchmark scripts for 70B+ models' },
    ],
  },
  {
    version: '0.1.2',
    date: '2026-02-25',
    title: 'Documentation and packaging fixes',
    description: 'Improved documentation and fixed PyPI classifier issues.',
    changes: [
      { type: 'fixed', text: 'Fixed invalid PyPI classifier (removed CUDA language)' },
      { type: 'changed', text: 'Updated README with verified 32B benchmarks' },
      { type: 'changed', text: 'Added honest VRAM threshold notes' },
      { type: 'added', text: 'PyPI badge in README' },
    ],
  },
  {
    version: '0.1.1',
    date: '2026-02-24',
    title: 'Bug fixes and improvements',
    description: 'Minor bug fixes and documentation updates.',
    changes: [
      { type: 'fixed', text: 'Fixed package installation with optional GGUF support' },
      { type: 'changed', text: 'Improved error messages for missing models' },
      { type: 'added', text: 'Support for custom model paths' },
    ],
  },
  {
    version: '0.1.0',
    date: '2026-02-23',
    title: 'Initial Release',
    description: 'First public release of ZSE with core functionality.',
    changes: [
      { type: 'added', text: 'zQuantize: INT4/NF4 pre-quantization with .zse format' },
      { type: 'added', text: 'zServe: OpenAI-compatible API server' },
      { type: 'added', text: 'zInfer: CLI inference tool' },
      { type: 'added', text: 'GGUF import support' },
      { type: 'added', text: 'Streaming token generation' },
      { type: 'added', text: 'Multi-model management' },
    ],
  },
]

export function getLatestVersion(): string {
  return changelog[0]?.version || '0.0.0'
}

export function getChangelogByVersion(version: string): ChangelogEntry | undefined {
  return changelog.find(entry => entry.version === version)
}

export function formatChangelogDate(date: string): string {
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

export const changeTypeColors: Record<string, { bg: string; text: string }> = {
  added: { bg: 'bg-green-500/20', text: 'text-green-400' },
  changed: { bg: 'bg-blue-500/20', text: 'text-blue-400' },
  fixed: { bg: 'bg-yellow-500/20', text: 'text-yellow-400' },
  removed: { bg: 'bg-red-500/20', text: 'text-red-400' },
  security: { bg: 'bg-purple-500/20', text: 'text-purple-400' },
}
