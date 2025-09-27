'use client'

import React, { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { MessageCircle, X, Send, Trash2, Minimize2, Maximize2 } from 'lucide-react'
import { useTextConversation } from '@/components/providers/TextConversationContextProvider'
import { cn } from '@/lib/utils'

export function FloatingTextButton() {
  const {
    isConnected,
    connectionStatus,
    messages,
    isTyping,
    isReady,
    isLoading,
    connect,
    disconnect,
    sendMessage,
    clearMessages
  } = useTextConversation()

  const [isExpanded, setIsExpanded] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [inputMessage, setInputMessage] = useState('')
  const [unreadCount, setUnreadCount] = useState(0)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  // Track unread messages when chat is minimized
  useEffect(() => {
    if (!isExpanded || isMinimized) {
      setUnreadCount(prev => prev + 1)
    } else {
      setUnreadCount(0)
    }
  }, [messages.length, isExpanded, isMinimized])

  // Focus input when expanded
  useEffect(() => {
    if (isExpanded && !isMinimized && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isExpanded, isMinimized])

  // Don't render if service is not ready
  if (isLoading || !isReady) {
    return null
  }

  const handleSendMessage = () => {
    if (!inputMessage.trim() || !isConnected) return

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
    if (isConnected) return 'bg-blue-500 hover:bg-blue-600'
    return 'bg-gray-500 hover:bg-gray-600'
  }

  const formatTime = (date: Date | string | number) => {
    try {
      // Ensure we have a valid Date object
      const dateObj = date instanceof Date ? date : new Date(date)
      
      // Check if the date is valid
      if (isNaN(dateObj.getTime())) {
        return 'Invalid time'
      }
      
      return dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } catch (error) {
      console.error('Error formatting time:', error, 'Input:', date)
      return 'Invalid time'
    }
  }

  return (
    <div className="fixed bottom-6 left-6 z-50">
      {/* Main Floating Button */}
      <div className="relative">
        <Button
          onClick={toggleExpanded}
          className={cn(
            'w-16 h-16 rounded-full shadow-lg transition-all duration-300 hover:scale-110',
            getStatusColor()
          )}
          size="lg"
        >
          <MessageCircle className="w-6 h-6 text-white" />
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

      {/* Chat Panel */}
      {isExpanded && (
        <div className={cn(
          "absolute bottom-20 left-0 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 transition-all duration-300 flex flex-col",
          isMinimized ? "w-80 h-12" : "w-96 h-[500px]"
        )}>
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
            <div className="flex items-center gap-2">
              <MessageCircle className="w-5 h-5" />
              <h3 className="text-lg font-semibold">Text Chat</h3>
              <Badge variant={isConnected ? 'default' : 'secondary'} className="text-xs">
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
              {/* Connection Controls */}
              {!isConnected && (
                <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
                  <Button onClick={connect} size="sm" className="w-full">
                    Connect to Chat
                  </Button>
                </div>
              )}

              {/* Messages Area */}
              <ScrollArea className="flex-1 p-4 min-h-0">
                <div className="space-y-4">
                  {messages.length === 0 && isConnected && (
                    <div className="text-center text-muted-foreground py-8">
                      <MessageCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                      <p>Start a conversation!</p>
                    </div>
                  )}

                  {messages.map((message) => (
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
                    placeholder={isConnected ? "Type a message..." : "Connect to start chatting"}
                    disabled={!isConnected}
                    className="flex-1"
                  />
                  <Button
                    onClick={handleSendMessage}
                    disabled={!isConnected || !inputMessage.trim()}
                    size="sm"
                    className="px-3"
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                </div>

                {/* Quick Actions */}
                {isConnected && (
                  <div className="flex justify-between items-center">
                    <Button
                      onClick={clearMessages}
                      variant="ghost"
                      size="sm"
                      className="text-xs text-muted-foreground"
                      disabled={messages.length === 0}
                    >
                      <Trash2 className="w-3 h-3 mr-1" />
                      Clear
                    </Button>
                    <Button
                      onClick={disconnect}
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
  )
}
