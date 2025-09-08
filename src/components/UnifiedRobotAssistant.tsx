'use client'

import React, { useState, useEffect, useRef } from 'react'
import Image from 'next/image'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Mic, 
  MicOff, 
  Play, 
  MessageCircle, 
  X, 
  Send, 
  Trash2, 
  Minimize2, 
  Maximize2,
  Navigation, 
  Bot
} from 'lucide-react'
import { useVoiceAgent } from '@/components/providers/VoiceAgentContextProvider'
import { useTextConversation } from '@/components/providers/TextConversationContextProvider'
import { useTheme } from '@/store/theme-store'
import { cn } from '@/lib/utils'

type AssistantMode = 'voice' | 'text'

export function UnifiedRobotAssistant() {
  const theme = useTheme()
  const [mode, setMode] = useState<AssistantMode>('voice')
  const [isExpanded, setIsExpanded] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [inputMessage, setInputMessage] = useState('')
  const [unreadCount, setUnreadCount] = useState(0)
  const [voiceNavigationStatus, setVoiceNavigationStatus] = useState('Ready')
  const [lastVoiceCommand, setLastVoiceCommand] = useState<string | null>(null)

  // Voice Agent Context
  const {
    isConnected: isVoiceConnected,
    isInConversation,
    connectionStatus: voiceConnectionStatus,
    messages: voiceMessages,
    isReady: isVoiceReady,
    isLoading: isVoiceLoading,
    connect: connectVoice,
    disconnect: disconnectVoice,
    startConversation,
    stopConversation,
    clearMessages: clearVoiceMessages
  } = useVoiceAgent()

  // Text Conversation Context
  const {
    isConnected: isTextConnected,
    connectionStatus: textConnectionStatus,
    messages: textMessages,
    isTyping,
    isReady: isTextReady,
    isLoading: isTextLoading,
    connect: connectText,
    disconnect: disconnectText,
    sendMessage,
    clearMessages: clearTextMessages
  } = useTextConversation()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [mode === 'text' ? textMessages : voiceMessages])

  // Track unread messages when chat is minimized
  useEffect(() => {
    if (!isExpanded || isMinimized) {
      const currentMessages = mode === 'text' ? textMessages : voiceMessages
      if (currentMessages.length > 0) {
        setUnreadCount(prev => prev + 1)
      }
    } else {
      setUnreadCount(0)
    }
  }, [mode === 'text' ? textMessages.length : voiceMessages.length, isExpanded, isMinimized])

  // Focus input when expanded
  useEffect(() => {
    if (isExpanded && !isMinimized && inputRef.current && mode === 'text') {
      inputRef.current.focus()
    }
  }, [isExpanded, isMinimized, mode])

  // Listen for voice navigation events
  useEffect(() => {
    if (isVoiceLoading || !isVoiceReady) {
      return
    }

    const handleVoiceEvent = (event: CustomEvent) => {
      const { type, element_name, page, user_id } = event.detail
      setVoiceNavigationStatus(type)
      setLastVoiceCommand(`${type}: ${element_name || page || 'action executed'}${user_id ? ` (User: ${user_id})` : ''}`)
    }

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
  }, [isVoiceLoading, isVoiceReady])

  // Don't render if services are not ready
  if ((isVoiceLoading || !isVoiceReady) && (isTextLoading || !isTextReady)) {
    return null
  }

  const handleSendMessage = () => {
    if (!inputMessage.trim() || !isTextConnected) return

    sendMessage(inputMessage.trim())
    setInputMessage('')
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded)
    if (!isExpanded) {
      setIsMinimized(false)
      setUnreadCount(0)
    }
  }

  const toggleMinimized = () => {
    setIsMinimized(!isMinimized)
    if (isMinimized) {
      setUnreadCount(0)
    }
  }

  const getStatusColor = () => {
    if (mode === 'voice') {
      if (isInConversation) return 'bg-emerald-500 hover:bg-emerald-600 shadow-emerald-500/30'
      if (isVoiceConnected) return 'bg-emerald-500 hover:bg-emerald-600 shadow-emerald-500/20'
      return 'bg-gray-500 hover:bg-gray-600'
    } else {
      if (isTextConnected) return 'bg-emerald-500 hover:bg-emerald-600 shadow-emerald-500/20'
      return 'bg-gray-500 hover:bg-gray-600'
    }
  }

  const formatTime = (date: Date | string | number) => {
    try {
      const dateObj = date instanceof Date ? date : new Date(date)
      if (isNaN(dateObj.getTime())) {
        return 'Invalid time'
      }
      return dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } catch (error) {
      console.error('Error formatting time:', error, 'Input:', date)
      return 'Invalid time'
    }
  }

  const currentMessages = mode === 'text' ? textMessages : voiceMessages
  const isConnected = mode === 'text' ? isTextConnected : isVoiceConnected
  const connectionStatus = mode === 'text' ? textConnectionStatus : voiceConnectionStatus

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Main Robot Button */}
      <div className="relative">
        <Button
          onClick={toggleExpanded}
          className={cn(
            'w-16 h-16 rounded-full shadow-lg transition-all duration-300 hover:scale-110',
            getStatusColor()
          )}
          size="lg"
        >
          <Image
            src="/autopilot.svg"
            alt="AI Assistant"
            width={32}
            height={32}
            className="w-8 h-8 text-white"
          />
        </Button>

        {/* Unread Badge */}
        {unreadCount > 0 && (
          <Badge
            className="absolute -top-2 -right-2 bg-red-500 text-white px-2 py-1 text-xs rounded-full min-w-[20px] h-5"
          >
            {unreadCount > 99 ? '99+' : unreadCount}
          </Badge>
        )}
      </div>

      {/* Expanded Panel */}
      {isExpanded && (
        <div className={cn(
          "absolute bottom-20 rounded-2xl shadow-2xl transition-all duration-300 flex flex-col backdrop-blur-xl",
          isMinimized ? "w-80 h-12 right-0" : "w-96 sm:w-96 md:w-[28rem] h-[500px] right-0",
          theme === "dark" 
            ? "bg-white/5 border border-emerald-500/20 shadow-emerald-500/10" 
            : "bg-white/95 border border-emerald-500/30 shadow-emerald-500/20"
        )}
        style={{
          right: '0',
          maxWidth: 'calc(100vw - 2rem)',
          maxHeight: 'calc(100vh - 8rem)',
          background: theme === "dark"
            ? "linear-gradient(158.39deg, rgba(255, 255, 255, 0.06) 14.19%, rgba(255, 255, 255, 0.000015) 50.59%, rgba(255, 255, 255, 0.000015) 68.79%, rgba(255, 255, 255, 0.015) 105.18%)"
            : "linear-gradient(158.39deg, rgba(255, 255, 255, 0.98) 14.19%, rgba(240, 249, 245, 0.95) 50.59%, rgba(255, 255, 255, 0.98) 68.79%, rgba(240, 249, 245, 0.95) 105.18%)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          boxShadow: theme === "dark"
            ? "0 8px 32px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1)"
            : "0 8px 32px rgba(16, 185, 129, 0.12), 0 1px 0 rgba(255, 255, 255, 0.8) inset"
        }}>
          {/* Animated glow dots in corners */}
          <div className="absolute top-4 left-4 w-1 h-1 bg-emerald-400 rounded-full animate-pulse opacity-60" 
               style={{ boxShadow: "0 0 8px #10b981" }} />
          <div className="absolute top-8 left-2 w-0.5 h-0.5 bg-emerald-300 rounded-full animate-pulse opacity-40" 
               style={{ boxShadow: "0 0 4px #6ee7b7" }} />
          <div className="absolute bottom-12 left-6 w-0.5 h-0.5 bg-emerald-400 rounded-full animate-pulse opacity-50" 
               style={{ boxShadow: "0 0 6px #10b981" }} />
          <div className="absolute bottom-4 left-2 w-1 h-1 bg-emerald-300 rounded-full animate-pulse opacity-30" 
               style={{ boxShadow: "0 0 4px #6ee7b7" }} />
          
          {/* Header */}
          <div className={cn(
            "flex items-center justify-between p-4 flex-shrink-0",
            theme === "dark" 
              ? "border-b border-emerald-500/20" 
              : "border-b border-emerald-500/30"
          )}>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500/20 to-emerald-600/20 border border-emerald-400/30 flex items-center justify-center">
                <Bot className={cn(
                  "w-5 h-5",
                  theme === "dark" ? "text-emerald-400" : "text-emerald-600"
                )} />
              </div>
              <h3 className={cn(
                "text-lg font-semibold truncate",
                theme === "dark" ? "text-white" : "text-gray-900"
              )}>AI Assistant</h3>
              <Badge 
                variant={isConnected ? 'default' : 'secondary'} 
                className={cn(
                  "text-xs",
                  isConnected 
                    ? "bg-emerald-500 text-white" 
                    : theme === "dark" 
                      ? "bg-gray-700 text-gray-300" 
                      : "bg-gray-200 text-gray-600"
                )}
              >
                {connectionStatus}
              </Badge>
            </div>
            <div className="flex items-center gap-1">
              <Button
                onClick={toggleMinimized}
                variant="ghost"
                size="sm"
                className="w-8 h-8 p-0"
              >
                {isMinimized ? <Maximize2 className="w-4 h-4" /> : <Minimize2 className="w-4 h-4" />}
              </Button>
              <Button
                onClick={() => setIsExpanded(false)}
                variant="ghost"
                size="sm"
                className="w-8 h-8 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {!isMinimized && (
            <div className="flex flex-col flex-1 min-h-0">
              {/* Mode Toggle */}
              <div className={cn(
                "flex p-3 flex-shrink-0",
                theme === "dark" 
                  ? "border-b border-emerald-500/20" 
                  : "border-b border-emerald-500/30"
              )}>
                <div className={cn(
                  "flex rounded-lg p-1 w-full",
                  theme === "dark" 
                    ? "bg-emerald-500/10" 
                    : "bg-emerald-50"
                )}>
                  <Button
                    onClick={() => setMode('voice')}
                    size="sm"
                    className={cn(
                      "flex-1 border-0",
                      mode === 'voice' 
                        ? "bg-emerald-500 text-white hover:bg-emerald-600" 
                        : theme === "dark" 
                          ? "bg-transparent text-emerald-300 hover:bg-emerald-500/20" 
                          : "bg-transparent text-emerald-600 hover:bg-emerald-100"
                    )}
                  >
                    <Mic className="w-4 h-4 mr-2" />
                    Voice
                  </Button>
                  <Button
                    onClick={() => setMode('text')}
                    size="sm"
                    className={cn(
                      "flex-1 border-0",
                      mode === 'text' 
                        ? "bg-emerald-500 text-white hover:bg-emerald-600" 
                        : theme === "dark" 
                          ? "bg-transparent text-emerald-300 hover:bg-emerald-500/20" 
                          : "bg-transparent text-emerald-600 hover:bg-emerald-100"
                    )}
                  >
                    <MessageCircle className="w-4 h-4 mr-2" />
                    Text
                  </Button>
                </div>
              </div>

              {/* Voice Mode Content */}
              {mode === 'voice' && (
                <ScrollArea className="flex-1 min-h-0">
                  <div className="flex flex-col p-3 space-y-3">
                  {/* Voice Navigation Status */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-sm">
                      <Navigation className={cn(
                        "w-4 h-4",
                        theme === "dark" ? "text-emerald-400" : "text-emerald-600"
                      )} />
                      <span className={cn(
                        theme === "dark" ? "text-gray-300" : "text-gray-700"
                      )}>Status:</span>
                      <Badge 
                        variant={voiceNavigationStatus === 'Ready' ? 'default' : 'secondary'}
                        className={cn(
                          voiceNavigationStatus === 'Ready' 
                            ? "bg-emerald-500 text-white" 
                            : theme === "dark" 
                              ? "bg-gray-700 text-gray-300" 
                              : "bg-gray-200 text-gray-600"
                        )}
                      >
                        {voiceNavigationStatus}
                      </Badge>
                    </div>
                    {lastVoiceCommand && (
                      <div className={cn(
                        "text-xs",
                        theme === "dark" ? "text-emerald-300/70" : "text-emerald-600/70"
                      )}>
                        {lastVoiceCommand}
                      </div>
                    )}
                  </div>

                  {/* Connection Controls */}
                  <div className="flex gap-2">
                    {!isVoiceConnected ? (
                      <Button 
                        onClick={connectVoice} 
                        size="sm" 
                        className="flex-1 bg-emerald-500 hover:bg-emerald-600 text-white"
                      >
                        Connect Voice
                      </Button>
                    ) : (
                      <Button 
                        onClick={disconnectVoice} 
                        variant="outline" 
                        size="sm" 
                        className={cn(
                          "flex-1",
                          theme === "dark" 
                            ? "border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/20" 
                            : "border-emerald-500 text-emerald-600 hover:bg-emerald-50"
                        )}
                      >
                        Disconnect
                      </Button>
                    )}
                  </div>

                  {/* Conversation Controls */}
                  {isVoiceConnected && (
                    <div className="flex gap-2">
                      {!isInConversation ? (
                        <Button 
                          onClick={startConversation} 
                          size="sm" 
                          className="flex-1 bg-emerald-500 hover:bg-emerald-600 text-white"
                        >
                          Start Conversation
                        </Button>
                      ) : (
                        <Button 
                          onClick={stopConversation} 
                          variant="outline" 
                          size="sm" 
                          className={cn(
                            "flex-1",
                            theme === "dark" 
                              ? "border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/20" 
                              : "border-emerald-500 text-emerald-600 hover:bg-emerald-50"
                          )}
                        >
                          Stop Conversation
                        </Button>
                      )}
                    </div>
                  )}


                  {/* Recent Voice Messages */}
                  {voiceMessages.length > 0 && (
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className={cn(
                          "text-sm font-medium",
                          theme === "dark" ? "text-white" : "text-gray-900"
                        )}>Recent Messages</span>
                        <Button 
                          variant="outline" 
                          size="sm" 
                          onClick={clearVoiceMessages}
                          className={cn(
                            theme === "dark" 
                              ? "border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/20" 
                              : "border-emerald-500 text-emerald-600 hover:bg-emerald-50"
                          )}
                        >
                          Clear
                        </Button>
                      </div>
                      <ScrollArea className="max-h-24">
                        <div className="space-y-1">
                          {voiceMessages.slice(-3).map((message) => (
                            <div key={message.id} className={cn(
                              "text-xs p-2 rounded",
                              theme === "dark" 
                                ? "bg-emerald-500/10 border border-emerald-500/20" 
                                : "bg-emerald-50 border border-emerald-200"
                            )}>
                              <div className="flex items-center gap-2 mb-1">
                                <Badge 
                                  variant="outline" 
                                  className={cn(
                                    "text-xs",
                                    theme === "dark" 
                                      ? "border-emerald-500/50 text-emerald-300" 
                                      : "border-emerald-500 text-emerald-600"
                                  )}
                                >
                                  {message.type}
                                </Badge>
                                <span className={cn(
                                  theme === "dark" ? "text-emerald-300/70" : "text-emerald-600/70"
                                )}>
                                  {formatTime(message.timestamp)}
                                </span>
                              </div>
                              <div className={cn(
                                "truncate",
                                theme === "dark" ? "text-gray-300" : "text-gray-700"
                              )}>{message.content}</div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  )}
                  </div>
                </ScrollArea>
              )}

              {/* Text Mode Content */}
              {mode === 'text' && (
                <div className="flex flex-col flex-1 min-h-0">
                  {/* Connection Controls */}
                  {!isTextConnected && (
                    <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
                      <Button onClick={connectText} size="sm" className="w-full">
                        Connect to Chat
                      </Button>
                    </div>
                  )}

                  {/* Messages Area */}
                  <ScrollArea className="flex-1 p-4 min-h-0">
                    <div className="space-y-4">
                      {textMessages.length === 0 && isTextConnected && (
                        <div className="text-center text-muted-foreground py-8">
                          <MessageCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                          <p>Start a conversation!</p>
                        </div>
                      )}

                      {textMessages.map((message) => (
                        <div
                          key={message.id}
                          className={cn(
                            "flex",
                            message.type === 'user' ? 'justify-end' : 'justify-start'
                          )}
                        >
                          <div
                            className={cn(
                              "max-w-[80%] rounded-lg px-3 py-2 text-sm",
                              message.type === 'user'
                                ? 'bg-blue-500 text-white'
                                : message.type === 'system'
                                ? 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 italic'
                                : 'bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100'
                            )}
                          >
                            <div className="whitespace-pre-wrap break-words">
                              {message.content}
                            </div>
                            <div
                              className={cn(
                                "text-xs mt-1 opacity-70",
                                message.type === 'user' ? 'text-blue-100' : 'text-muted-foreground'
                              )}
                            >
                              {formatTime(message.timestamp)}
                            </div>
                          </div>
                        </div>
                      ))}

                      {isTyping && (
                        <div className="flex justify-start">
                          <div className="bg-gray-100 dark:bg-gray-700 rounded-lg px-3 py-2 text-sm">
                            <div className="flex items-center gap-1">
                              <div className="flex gap-1">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                              </div>
                              <span className="text-xs text-muted-foreground ml-2">typing...</span>
                            </div>
                          </div>
                        </div>
                      )}

                      <div ref={messagesEndRef} />
                    </div>
                  </ScrollArea>

                  {/* Input Area */}
                  <div className="p-3 border-t border-gray-200 dark:border-gray-700 space-y-2 flex-shrink-0">
                    <div className="flex gap-2">
                      <Input
                        ref={inputRef}
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder={isTextConnected ? "Type a message..." : "Connect to start chatting"}
                        disabled={!isTextConnected}
                        className="flex-1"
                      />
                      <Button
                        onClick={handleSendMessage}
                        disabled={!isTextConnected || !inputMessage.trim()}
                        size="sm"
                        className="px-3"
                      >
                        <Send className="w-4 h-4" />
                      </Button>
                    </div>

                    {/* Quick Actions */}
                    {isTextConnected && (
                      <div className="flex justify-between items-center">
                        <Button
                          onClick={clearTextMessages}
                          variant="ghost"
                          size="sm"
                          className="text-xs text-muted-foreground"
                          disabled={textMessages.length === 0}
                        >
                          <Trash2 className="w-3 h-3 mr-1" />
                          Clear
                        </Button>
                        <Button
                          onClick={disconnectText}
                          variant="ghost"
                          size="sm"
                          className="text-xs text-muted-foreground"
                        >
                          Disconnect
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
