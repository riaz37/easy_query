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
type RobotState = 'idle' | 'selecting' | 'chatting' | 'talking'

export function UnifiedRobotAssistant() {
  const theme = useTheme()
  const [mode, setMode] = useState<AssistantMode>('voice')
  const [robotState, setRobotState] = useState<RobotState>('idle')
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

  const handleRobotClick = () => {
    if (robotState === 'idle') {
      setRobotState('selecting')
    } else if (robotState === 'selecting') {
      setRobotState('idle')
    } else if (robotState === 'talking') {
      // Disconnect voice and return to idle
      disconnectVoice()
      setRobotState('idle')
    } else {
      // If in chatting mode, close the interface
      setRobotState('idle')
      setIsExpanded(false)
      setIsMinimized(false)
    }
  }

  const handleModeSelect = (selectedMode: AssistantMode) => {
    setMode(selectedMode)
    if (selectedMode === 'voice') {
      setRobotState('talking')
      connectVoice()
    } else {
      setRobotState('chatting')
      setIsExpanded(true)
      connectText()
    }
  }

  const toggleMinimized = () => {
    setIsMinimized(!isMinimized)
    if (isMinimized) {
      setUnreadCount(0)
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
        <button
          onClick={handleRobotClick}
          className={cn(
            'transition-all duration-300 hover:scale-110 focus:outline-none active:outline-none',
            'p-2 rounded-lg relative',
            robotState === 'talking' && 'animate-pulse',
            robotState === 'talking' && 'cursor-pointer'
          )}
          title={robotState === 'talking' ? 'Click to disconnect voice' : 'Click to interact'}
        >
          {/* Pulsing Glow Ring */}
          <div className={cn(
            'absolute inset-0 rounded-lg transition-all duration-1000',
            'bg-gradient-to-r from-emerald-400/20 to-emerald-600/20',
            'animate-pulse',
            robotState === 'talking' && 'from-emerald-400/40 to-emerald-600/40',
            'hover:from-emerald-400/30 hover:to-emerald-600/30'
          )} style={{
            animationDuration: '2s',
            animationTimingFunction: 'ease-in-out',
          }} />
          
          <Image
            src="/autopilot.svg"
            alt="AI Assistant"
            width={48}
            height={48}
            className="w-12 h-12 relative z-10"
          />
        </button>


        {/* Talking indicator dots */}
        {robotState === 'talking' && isInConversation && (
          <div className="absolute -top-2 -right-2 flex gap-1">
            <div className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-1 h-1 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
        )}

        {/* Expanding rings for active states */}
        {robotState === 'talking' && isInConversation && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute inset-0 border-2 border-emerald-400/30 rounded-lg animate-ping" style={{ animationDuration: '1.5s' }}></div>
            <div className="absolute inset-0 border-2 border-emerald-400/20 rounded-lg animate-ping" style={{ animationDuration: '1.5s', animationDelay: '0.5s' }}></div>
          </div>
        )}

        {/* Attention-seeking pulse for idle state */}
        {robotState === 'idle' && (
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute inset-0 border-2 border-emerald-400/20 rounded-lg animate-pulse" style={{ animationDuration: '3s' }}></div>
          </div>
        )}

        {/* Robot Question Speech Bubble */}
        {robotState === 'selecting' && (
          <div className="absolute bottom-6 right-20 flex items-center animate-in slide-in-from-right-2 duration-300">
            {/* Speech bubble */}
            <div className={cn(
              "px-4 py-2 rounded-2xl shadow-lg backdrop-blur-xl relative whitespace-nowrap",
              theme === "dark" 
                ? "bg-white/10 border border-emerald-500/20" 
                : "bg-white/95 border border-emerald-500/30"
            )}>
              {/* Speech bubble tail pointing to robot */}
              <div className={cn(
                "absolute -right-2 top-1/2 transform -translate-y-1/2 w-0 h-0",
                "border-l-[12px] border-l-emerald-500/30 border-t-[8px] border-t-transparent border-b-[8px] border-b-transparent"
              )}></div>
              
              <div className="flex flex-col gap-2">
                <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">
                  What would you like to do?
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleModeSelect('voice')}
                    className="px-3 py-1 text-xs bg-emerald-500 hover:bg-emerald-600 text-white rounded-md transition-colors"
                  >
                    Talk
                  </button>
                  <button
                    onClick={() => handleModeSelect('text')}
                    className="px-3 py-1 text-xs bg-emerald-500 hover:bg-emerald-600 text-white rounded-md transition-colors"
                  >
                    Chat
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Voice Mode - Speech Bubble beside existing robot */}
      {robotState === 'talking' && (
        <div className="absolute bottom-6 right-20 animate-in slide-in-from-right-2 duration-300">
          {/* Speech bubble */}
          <div className={cn(
            "px-4 py-2 rounded-2xl shadow-lg backdrop-blur-xl relative w-80",
            theme === "dark" 
              ? "bg-white/10 border border-emerald-500/20" 
              : "bg-white/95 border border-emerald-500/30"
          )}>
            {/* Speech bubble tail pointing to robot */}
            <div className={cn(
              "absolute -right-2 top-1/2 transform -translate-y-1/2 w-0 h-0",
              "border-l-[12px] border-l-emerald-500/30 border-t-[8px] border-t-transparent border-b-[8px] border-b-transparent"
            )}></div>
            
            {voiceMessages.length > 0 ? (
              <div className="flex flex-col gap-2">
                <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400 break-words">
                  {voiceMessages[voiceMessages.length - 1]?.content}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400 self-end">
                  {formatTime(voiceMessages[voiceMessages.length - 1]?.timestamp)}
                </span>
              </div>
            ) : (
              <span className="text-sm text-gray-600 dark:text-gray-300">
                {isInConversation ? "Listening..." : "Ready to talk"}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Expanded Panel - Text Chat Mode */}
      {robotState === 'chatting' && isExpanded && (
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
                <Image
                  src="/autopilot.svg"
                  alt="AI Assistant"
                  width={20}
                  height={20}
                  className="w-5 h-5"
                />
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


              {/* Text Mode Content */}
              {mode === 'text' && (
                <div className="flex flex-col flex-1 min-h-0">
                  {/* Connection Loading State */}
                  {!isTextConnected && (
                    <div className="p-3 bg-emerald-50 dark:bg-emerald-900/20 border-b border-emerald-200 dark:border-emerald-700 flex-shrink-0">
                      <div className="flex items-center justify-center gap-2 text-emerald-600 dark:text-emerald-400">
                        <div className="w-4 h-4 border-2 border-emerald-600 border-t-transparent rounded-full animate-spin"></div>
                        <span className="text-sm font-medium">Connecting to chat...</span>
                      </div>
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
                              "max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-lg backdrop-blur-xl",
                              message.type === 'user'
                                ? 'bg-emerald-500 text-white'
                                : message.type === 'system'
                                ? theme === "dark" 
                                  ? 'bg-white/10 border border-emerald-500/20 text-emerald-300 italic' 
                                  : 'bg-emerald-50 border border-emerald-200 text-emerald-600 italic'
                                : theme === "dark"
                                  ? 'bg-white/10 border border-emerald-500/20 text-white'
                                  : 'bg-white/95 border border-emerald-200 text-gray-900'
                            )}
                          >
                            <div className="whitespace-pre-wrap break-words">
                              {message.content}
                            </div>
                            <div
                              className={cn(
                                "text-xs mt-2 opacity-70",
                                message.type === 'user' ? 'text-emerald-100' : 'text-emerald-500/70'
                              )}
                            >
                              {formatTime(message.timestamp)}
                            </div>
                          </div>
                        </div>
                      ))}

                      {isTyping && (
                        <div className="flex justify-start">
                          <div className={cn(
                            "rounded-2xl px-4 py-3 text-sm shadow-lg backdrop-blur-xl",
                            theme === "dark"
                              ? "bg-white/10 border border-emerald-500/20"
                              : "bg-white/95 border border-emerald-200"
                          )}>
                            <div className="flex items-center gap-1">
                              <div className="flex gap-1">
                                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                              </div>
                              <span className="text-xs text-emerald-500/70 ml-2">typing...</span>
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
                      <div className="flex justify-start items-center">
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
