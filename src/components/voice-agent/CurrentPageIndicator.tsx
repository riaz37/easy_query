'use client'

import { useVoiceClient } from '../../lib/hooks/use-voice-client'

export function CurrentPageIndicator() {
  const { currentPage, previousPage, isConnected } = useVoiceClient()

  if (!isConnected) return null

  return (
    <div className="fixed bottom-4 left-4 bg-black/80 text-white px-3 py-2 rounded-lg text-sm font-mono z-50">
      <div className="flex items-center space-x-2">
        <span className="text-green-400">●</span>
        <span>Page: {currentPage}</span>
        {previousPage && (
          <span className="text-gray-400">← {previousPage}</span>
        )}
      </div>
    </div>
  )
} 