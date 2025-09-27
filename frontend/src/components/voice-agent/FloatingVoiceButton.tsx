'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Mic, MicOff, Play, Navigation, Search, Upload, FileText } from 'lucide-react'
import { useVoiceAgent } from '@/components/providers/VoiceAgentContextProvider'

export function FloatingVoiceButton() {
  const {
    isConnected,
    isInConversation,
    connectionStatus,
    messages,
    isReady,
    isLoading,
    connect,
    disconnect,
    startConversation,
    stopConversation,
    clearMessages
  } = useVoiceAgent()
  
  const [isExpanded, setIsExpanded] = useState(false)
  const [voiceNavigationStatus, setVoiceNavigationStatus] = useState('Ready')
  const [lastVoiceCommand, setLastVoiceCommand] = useState<string | null>(null)

  // Listen for voice navigation events
  useEffect(() => {
    // Only set up event listeners if service is ready
    if (isLoading || !isReady) {
      return
    }

    const handleVoiceEvent = (event: CustomEvent) => {
      const { type, element_name, page, user_id } = event.detail
      setVoiceNavigationStatus(type)
      setLastVoiceCommand(`${type}: ${element_name || page || 'action executed'}${user_id ? ` (User: ${user_id})` : ''}`)
    }

    // Add event listeners for voice navigation
    window.addEventListener('voice-database-search', handleVoiceEvent as EventListener)
    window.addEventListener('voice-file-search', handleVoiceEvent as EventListener)
    window.addEventListener('voice-file-upload', handleVoiceEvent as EventListener)
    window.addEventListener('voice-generate-report', handleVoiceEvent as EventListener)
    window.addEventListener('voice-navigate', handleVoiceEvent as EventListener)

    return () => {
      window.removeEventListener('voice-database-search', handleVoiceEvent as EventListener)
      window.removeEventListener('voice-file-search', handleVoiceEvent as EventListener)
      window.removeEventListener('voice-file-upload', handleVoiceEvent as EventListener)
      window.removeEventListener('voice-generate-report', handleVoiceEvent as EventListener)
      window.removeEventListener('voice-navigate', handleVoiceEvent as EventListener)
    }
  }, [isLoading, isReady])

  // Don't render if service is not ready
  if (isLoading || !isReady) {
    return null
  }

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded)
  }

  const getStatusColor = () => {
    if (isInConversation) return 'bg-green-500'
    if (isConnected) return 'bg-blue-500'
    return 'bg-gray-500'
  }

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Main Floating Button */}
      <Button
        onClick={toggleExpanded}
        className={`w-16 h-16 rounded-full shadow-lg transition-all duration-300 ${getStatusColor()} hover:scale-110`}
        size="lg"
      >
        {isInConversation ? (
          <Play className="w-6 h-6 text-white" />
        ) : isConnected ? (
          <Mic className="w-6 h-6 text-white" />
        ) : (
          <MicOff className="w-6 h-6 text-white" />
        )}
      </Button>

      {/* Expanded Panel */}
      {isExpanded && (
        <div className="absolute bottom-20 right-0 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 p-4 space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Voice Agent</h3>
            <Badge variant={isConnected ? 'default' : 'secondary'}>
              {connectionStatus}
            </Badge>
          </div>

          {/* Voice Navigation Status */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <Navigation className="w-4 h-4" />
              <span>Status:</span>
              <Badge variant={voiceNavigationStatus === 'Ready' ? 'default' : 'secondary'}>
                {voiceNavigationStatus}
              </Badge>
            </div>
            {lastVoiceCommand && (
              <div className="text-xs text-muted-foreground">
                {lastVoiceCommand}
              </div>
            )}
          </div>

          {/* Connection Controls */}
          <div className="flex gap-2">
            {!isConnected ? (
              <Button onClick={connect} size="sm" className="flex-1">
                Connect
              </Button>
            ) : (
              <Button onClick={disconnect} variant="outline" size="sm" className="flex-1">
                Disconnect
              </Button>
            )}
          </div>

          {/* Conversation Controls */}
          {isConnected && (
            <div className="flex gap-2">
              {!isInConversation ? (
                <Button onClick={startConversation} size="sm" className="flex-1">
                  Start
                </Button>
              ) : (
                <Button onClick={stopConversation} variant="outline" size="sm" className="flex-1">
                  Stop
                </Button>
              )}
            </div>
          )}

          {/* Quick Actions */}
          {isConnected && (
            <div className="space-y-2">
              <div className="text-sm font-medium">Quick Actions:</div>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.dispatchEvent(new CustomEvent('voice-database-search', { detail: { type: 'Database Search', element_name: 'Database Query' } }))}
                  className="flex items-center gap-2"
                >
                  <Search className="w-3 h-3" />
                  Search
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.dispatchEvent(new CustomEvent('voice-file-upload', { detail: { type: 'File Upload', element_name: 'File Upload Area' } }))}
                  className="flex items-center gap-2"
                >
                  <Upload className="w-3 h-3" />
                  Upload
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.dispatchEvent(new CustomEvent('voice-generate-report', { detail: { type: 'Generate Report', element_name: 'Report Generator' } }))}
                  className="flex items-center gap-2"
                >
                  <FileText className="w-3 h-3" />
                  Report
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.dispatchEvent(new CustomEvent('voice-navigate', { detail: { type: 'Navigate', page: 'Dashboard' } }))}
                  className="flex items-center gap-2"
                >
                  <Navigation className="w-3 h-3" />
                  Navigate
                </Button>
              </div>
            </div>
          )}

          {/* Recent Messages */}
          {messages.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Recent Messages</span>
                <Button variant="outline" size="sm" onClick={clearMessages}>
                  Clear
                </Button>
              </div>
              <div className="max-h-32 overflow-y-auto space-y-1">
                {messages.slice(-3).map((message) => (
                  <div key={message.id} className="text-xs p-2 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="outline" className="text-xs">
                        {message.type}
                      </Badge>
                      <span className="text-muted-foreground">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="truncate">{message.content}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
} 