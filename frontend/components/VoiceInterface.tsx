'use client'

import React, { useState, useCallback } from 'react'
import { useVoiceClient } from '@/lib/voiceClient'

interface VoiceInterfaceProps {
  onConnectionChange?: (isConnected: boolean) => void
}

export function VoiceInterface({ onConnectionChange }: VoiceInterfaceProps) {
  const {
    isConnected,
    isInConversation,
    connectionStatus,
    connect,
    disconnect,
    startConversation,
    stopConversation
  } = useVoiceClient()

  const [isConnecting, setIsConnecting] = useState(false)

  const handleConnect = useCallback(async () => {
    setIsConnecting(true)
    try {
      await connect()
    } catch (error) {
      console.error('Connection failed:', error)
    } finally {
      setIsConnecting(false)
    }
  }, [connect])

  const getStatusText = () => {
    if (isConnecting) return 'Connecting...'
    if (!isConnected) return 'Disconnected'
    if (isInConversation) return 'Listening...'
    return 'Connected'
  }

  const getStatusColor = () => {
    if (isConnecting) return 'text-voice-primary'
    if (!isConnected) return 'text-voice-muted'
    if (isInConversation) return 'text-voice-accent'
    return 'text-voice-secondary'
  }

  return (
    <div className="space-y-6">
      
      {/* Status Display */}
      <div className="text-center">
        <div className={`text-2xl font-semibold mb-2 ${getStatusColor()}`}>
          {getStatusText()}
        </div>
        
        {/* Visual Indicator */}
        <div className="flex justify-center mb-4">
          <div className="relative">
            {isInConversation ? (
              <div className="voice-indicator listening" />
            ) : isConnected ? (
              <div className="voice-indicator connected" />
            ) : (
              <div className="w-16 h-16 rounded-full bg-gray-300" />
            )}
          </div>
        </div>

        <p className="text-voice-muted text-sm mb-4">
          Press to talk or start continuous voice conversation
        </p>
      </div>

      {/* Control Buttons */}
      <div className="flex flex-col space-y-3">
        {!isConnected ? (
          <button
            onClick={handleConnect}
            disabled={isConnecting}
            className={`voice-button-primary ${isConnecting ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isConnecting ? 'Connecting...' : 'Connect to Voice Agent'}
          </button>
        ) : (
          <>
            <div className="flex space-x-3">
              {!isInConversation ? (
                <button
                  onClick={startConversation}
                  className="flex-1 voice-button-success"
                >
                  Start Conversation
                </button>
              ) : (
                <button
                  onClick={stopConversation}
                  className="flex-1 voice-button-warning"
                >
                  Stop Conversation
                </button>
              )}
              
              <button
                onClick={disconnect}
                className="px-4 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
              >
                Disconnect
              </button>
            </div>
          </>
        )}
      </div>

      {/* Privacy & Security Notice */}
      {isConnected && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-sm">
          <div className="flex items-center mb-2">
            <div className="w-2 h-2 bg-voice-accent rounded-full mr-2" />
            <span className="text-voice-text font-medium">Your conversation is private and secure</span>
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-voice-secondary rounded-full mr-2" />
            <span className="text-voice-text font-medium">AI-powered with advanced function calling</span>
          </div>
        </div>
      )}

      {/* Connection Status */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-voice-text font-medium">Connection Status:</span>
          <span className={`font-semibold ${getStatusColor()}`}>
            {connectionStatus}
          </span>
        </div>
      </div>
    </div>
  )
}
