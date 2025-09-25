'use client'

import React, { useEffect, useState } from 'react'
import { useVoiceAgent } from '@/components/providers/VoiceAgentContextProvider'
import { useVoiceAgentConnection } from '@/lib/hooks'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { VoiceMessage } from '@/lib/voice-agent/types'

interface PersistentVoiceAgentProps {
  className?: string
}

export function PersistentVoiceAgent({ className }: PersistentVoiceAgentProps) {
  const {
    isConnected,
    isInConversation,
    connectionStatus,
    currentPage,
    previousPage,
    messages,
    isReady,
    connect,
    disconnect,
    startConversation,
    stopConversation,
    clearMessages,
    sendMessage,
    getContext,
    clearContext
  } = useVoiceAgent()

  const {
    isConnecting,
    lastConnectedAt,
    connectionAttempts,
    autoReconnectEnabled,
    reconnect,
    toggleAutoReconnect,
    resetConnectionState
  } = useVoiceAgentConnection()

  const [contextInfo, setContextInfo] = useState<any>(null)
  const [showContext, setShowContext] = useState(false)

  // Load context info
  useEffect(() => {
    if (isReady) {
      const context = getContext()
      setContextInfo(context)
    }
  }, [isReady, messages, currentPage, getContext])

  const handleConnect = async () => {
    try {
      await connect()
    } catch (error) {
      console.error('Failed to connect:', error)
    }
  }

  const handleDisconnect = async () => {
    try {
      await disconnect()
    } catch (error) {
      console.error('Failed to disconnect:', error)
    }
  }

  const handleStartConversation = async () => {
    try {
      await startConversation()
    } catch (error) {
      console.error('Failed to start conversation:', error)
    }
  }

  const handleStopConversation = () => {
    stopConversation()
  }

  const handleClearMessages = () => {
    clearMessages()
  }

  const handleClearContext = () => {
    clearContext()
    setContextInfo(null)
  }

  const handleReconnect = async () => {
    try {
      await reconnect()
    } catch (error) {
      console.error('Failed to reconnect:', error)
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'connected':
        return 'bg-green-500'
      case 'connecting':
        return 'bg-yellow-500'
      case 'disconnected':
        return 'bg-gray-500'
      case 'error':
        return 'bg-red-500'
      default:
        return 'bg-gray-500'
    }
  }

  if (!isReady) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            🎤 Voice Agent
            <Badge variant="secondary">Not Ready</Badge>
          </CardTitle>
          <CardDescription>
            Voice agent is not ready. Please ensure you are authenticated.
          </CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Connection Status Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            🎤 Persistent Voice Agent
            <Badge 
              variant={isConnected ? "default" : "secondary"}
              className={getStatusColor(connectionStatus)}
            >
              {connectionStatus}
            </Badge>
          </CardTitle>
          <CardDescription>
            Maintains connection and context across page navigations
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Connection Info */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium">Current Page:</span>
              <Badge variant="outline" className="ml-2">{currentPage}</Badge>
            </div>
            <div>
              <span className="font-medium">Previous Page:</span>
              <Badge variant="outline" className="ml-2">{previousPage || 'None'}</Badge>
            </div>
            <div>
              <span className="font-medium">Connection Attempts:</span>
              <Badge variant="outline" className="ml-2">{connectionAttempts}</Badge>
            </div>
            <div>
              <span className="font-medium">Auto Reconnect:</span>
              <Badge variant={autoReconnectEnabled ? "default" : "secondary"} className="ml-2">
                {autoReconnectEnabled ? 'Enabled' : 'Disabled'}
              </Badge>
            </div>
          </div>

          {/* Connection Actions */}
          <div className="flex flex-wrap gap-2">
            {!isConnected ? (
              <Button 
                onClick={handleConnect} 
                disabled={isConnecting}
                className="bg-green-600 hover:bg-green-700"
              >
                {isConnecting ? 'Connecting...' : 'Connect'}
              </Button>
            ) : (
              <Button 
                onClick={handleDisconnect}
                variant="destructive"
              >
                Disconnect
              </Button>
            )}
            
            {isConnected && (
              <>
                {!isInConversation ? (
                  <Button 
                    onClick={handleStartConversation}
                    className="bg-blue-600 hover:bg-blue-700"
                  >
                    Start Conversation
                  </Button>
                ) : (
                  <Button 
                    onClick={handleStopConversation}
                    variant="outline"
                  >
                    Stop Conversation
                  </Button>
                )}
                
                <Button 
                  onClick={handleReconnect}
                  variant="outline"
                >
                  Reconnect
                </Button>
              </>
            )}
            
            <Button 
              onClick={toggleAutoReconnect}
              variant="outline"
              size="sm"
            >
              {autoReconnectEnabled ? 'Disable' : 'Enable'} Auto Reconnect
            </Button>
            
            <Button 
              onClick={resetConnectionState}
              variant="outline"
              size="sm"
            >
              Reset State
            </Button>
          </div>

          {/* Context Info */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">Context Information</h4>
              <Button
                onClick={() => setShowContext(!showContext)}
                variant="ghost"
                size="sm"
              >
                {showContext ? 'Hide' : 'Show'} Context
              </Button>
            </div>
            
            {showContext && contextInfo && (
              <Card className="bg-gray-50">
                <CardContent className="pt-4">
                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div>
                      <span className="font-medium">Messages:</span> {contextInfo.messages?.length || 0}
                    </div>
                    <div>
                      <span className="font-medium">In Conversation:</span> {contextInfo.isInConversation ? 'Yes' : 'No'}
                    </div>
                    <div>
                      <span className="font-medium">Last Connected:</span> {lastConnectedAt ? formatTimestamp(lastConnectedAt.toISOString()) : 'Never'}
                    </div>
                    <div>
                      <span className="font-medium">Status:</span> {contextInfo.connectionStatus}
                    </div>
                  </div>
                  
                  <div className="mt-4 flex gap-2">
                    <Button 
                      onClick={handleClearMessages}
                      variant="outline"
                      size="sm"
                    >
                      Clear Messages
                    </Button>
                    <Button 
                      onClick={handleClearContext}
                      variant="outline"
                      size="sm"
                    >
                      Clear All Context
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Messages Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Message History ({messages.length})</span>
            <Button 
              onClick={handleClearMessages}
              variant="outline"
              size="sm"
            >
              Clear
            </Button>
          </CardTitle>
          <CardDescription>
            Messages are persisted across page navigations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {messages.length === 0 ? (
              <p className="text-gray-500 text-center py-4">No messages yet</p>
            ) : (
              messages.slice(-10).map((message: VoiceMessage) => (
                <div key={message.id} className="flex items-start gap-2 p-2 rounded border">
                  <div className={`w-2 h-2 rounded-full mt-2 ${
                    message.type === 'user' ? 'bg-blue-500' :
                    message.type === 'assistant' ? 'bg-green-500' :
                    message.type === 'system' ? 'bg-gray-500' :
                    'bg-red-500'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span className="font-medium">{message.type}</span>
                      <span>{formatTimestamp(message.timestamp)}</span>
                    </div>
                    <p className="text-sm mt-1 break-words">{message.content}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Test Message Input */}
      <Card>
        <CardHeader>
          <CardTitle>Test Message</CardTitle>
          <CardDescription>
            Send a test message to verify the connection
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Type a test message..."
              className="flex-1 px-3 py-2 border rounded-md"
              onKeyPress={(e) => {
                if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                  sendMessage(e.currentTarget.value.trim())
                  e.currentTarget.value = ''
                }
              }}
            />
            <Button 
              onClick={() => {
                const input = document.querySelector('input[placeholder="Type a test message..."]') as HTMLInputElement
                if (input?.value.trim()) {
                  sendMessage(input.value.trim())
                  input.value = ''
                }
              }}
              disabled={!isConnected}
            >
              Send
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 