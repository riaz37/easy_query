import { VoiceAgent } from '@/components/voice-agent'

export default function VoiceAgentPage() {
  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Voice Agent
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Interact with ESAP using voice commands and natural language
        </p>
      </div>
      
      <VoiceAgent />
      
      <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
          🧪 Voice Agent Testing
        </h3>
        <p className="text-blue-700 dark:text-blue-300 text-sm mb-3">
          Test the voice navigation system and verify all integrations are working properly.
        </p>
        <a
          href="/voice-agent/test"
          className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Go to Test Page
        </a>
      </div>
    </div>
  )
} 