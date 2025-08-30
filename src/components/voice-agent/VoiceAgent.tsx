'use client'

import React, { useState, useEffect } from 'react'
import { useVoiceClient } from '@/lib/hooks/use-voice-client'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Mic, MicOff, Play, Square, MessageSquare, Trash2, Navigation, Search, Upload, FileText, BarChart3 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'

export function VoiceAgent() {
  const {
    isConnected,
    isInConversation,
    connectionStatus,
    messages,
    connect,
    disconnect,
    startConversation,
    stopConversation,
    clearMessages,
    sendMessage
  } = useVoiceClient()

  const [textInput, setTextInput] = useState('')
  const [voiceNavigationStatus, setVoiceNavigationStatus] = useState('Ready')
  const [lastVoiceCommand, setLastVoiceCommand] = useState('')

  // Listen for voice navigation events
  useEffect(() => {
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
  }, [])

  const handleSendMessage = () => {
    if (textInput.trim()) {
      sendMessage(textInput.trim())
      setTextInput('')
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mic className="w-5 h-5" />
            ESAP Voice Agent
          </CardTitle>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Voice Navigation Status */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <Navigation className="w-4 h-4" />
              <span>Voice Navigation:</span>
              <Badge variant={voiceNavigationStatus === 'Ready' ? 'default' : 'secondary'}>
                {voiceNavigationStatus}
              </Badge>
            </div>
            {lastVoiceCommand && (
              <div className="text-xs text-muted-foreground">
                Last command: {lastVoiceCommand}
              </div>
            )}
          </div>

          {/* Connection Controls */}
          <div className="flex gap-2">
            {!isConnected ? (
              <Button onClick={connect} className="flex items-center gap-2">
                <Mic className="w-4 h-4" />
                Connect
              </Button>
            ) : (
              <Button onClick={disconnect} variant="outline" className="flex items-center gap-2">
                <MicOff className="w-4 h-4" />
                Disconnect
              </Button>
            )}
          </div>

          {/* Conversation Controls */}
          {isConnected && (
            <div className="flex gap-2">
              {!isInConversation ? (
                <Button onClick={startConversation} className="flex items-center gap-2">
                  <Play className="w-4 h-4" />
                  Start Conversation
                </Button>
              ) : (
                <Button onClick={stopConversation} variant="outline" className="flex items-center gap-2">
                  <Square className="w-4 h-4" />
                  Stop Conversation
                </Button>
              )}
            </div>
          )}

          {/* Quick Voice Actions */}
          {isConnected && (
            <div className="space-y-2">
              <div className="text-sm font-medium">Quick Voice Actions:</div>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.dispatchEvent(new CustomEvent('voice-database-search', { detail: { type: 'Database Search', element_name: 'Database Query' } }))}
                  className="flex items-center gap-2"
                >
                  <Search className="w-3 h-3" />
                  Database Search
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.dispatchEvent(new CustomEvent('voice-file-upload', { detail: { type: 'File Upload', element_name: 'File Upload Area' } }))}
                  className="flex items-center gap-2"
                >
                  <Upload className="w-3 h-3" />
                  File Upload
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.dispatchEvent(new CustomEvent('voice-generate-report', { detail: { type: 'Generate Report', element_name: 'Report Generator' } }))}
                  className="flex items-center gap-2"
                >
                  <FileText className="w-3 h-3" />
                  Generate Report
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

          {/* Text Input */}
          {isConnected && (
            <div className="space-y-2">
              <div className="flex gap-2">
                <Input
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type a message..."
                  className="flex-1"
                />
                <Button onClick={handleSendMessage} disabled={!textInput.trim()}>
                  <MessageSquare className="w-4 h-4" />
                </Button>
              </div>
            </div>
          )}

          {/* Messages */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">Messages</div>
              <Button variant="outline" size="sm" onClick={clearMessages}>
                <Trash2 className="w-3 h-3" />
                Clear
              </Button>
            </div>
            
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {messages.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  No messages yet. Connect to start chatting!
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    className={`p-3 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800'
                        : message.type === 'assistant'
                        ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800'
                        : message.type === 'error'
                        ? 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'
                        : 'bg-gray-50 dark:bg-gray-900/20 border border-gray-200 dark:border-gray-800'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <Badge variant="outline" className="text-xs">
                        {message.type}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-sm">{message.content}</div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Connection Status */}
          <div className="text-center text-sm text-muted-foreground">
            Status: {connectionStatus}
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 