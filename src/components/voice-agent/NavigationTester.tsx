'use client'

import { useVoiceClient } from '../../lib/hooks/use-voice-client'

const TEST_PAGES = [
  'dashboard',
  'database-query',
  'file-query',
  'tables',
  'users',
  'ai-results',
  'company-structure'
]

export function NavigationTester() {
  const { currentPage, testNavigation, isConnected } = useVoiceClient()

  if (!isConnected) return null

  return (
    <div className="fixed bottom-4 right-4 bg-black/80 text-white p-4 rounded-lg text-sm font-mono z-50 max-w-xs">
      <div className="mb-2 text-center font-bold">🧭 Test Navigation</div>
      <div className="mb-2 text-xs text-gray-400">Current: {currentPage}</div>
      <div className="grid grid-cols-2 gap-1">
        {TEST_PAGES.map(page => (
          <button
            key={page}
            onClick={() => testNavigation(page)}
            className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors"
          >
            {page}
          </button>
        ))}
      </div>
    </div>
  )
} 