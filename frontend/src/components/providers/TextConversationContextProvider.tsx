'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useAuthContext } from '@/components/providers'
import { TextConversationService, TextMessage, TextConversationState } from '@/lib/voice-agent/services/TextConversationService'

interface TextConversationContextType {
  // Service state
  isConnected: boolean
  connectionStatus: string
  messages: TextMessage[]
  isTyping: boolean

  // Service status
  isReady: boolean
  isLoading: boolean

  // Actions
  connect: () => Promise<void>
  disconnect: () => void
  sendMessage: (content: string) => void
  clearMessages: () => void

  // Page context
  updateCurrentPage: (page: string) => void

  // State access
  getState: () => TextConversationState
}

const TextConversationContext = createContext<TextConversationContextType | undefined>(undefined)

interface TextConversationProviderProps {
  children: ReactNode
}

export function TextConversationProvider({ children }: TextConversationProviderProps) {
  const { user, isAuthenticated, isLoading: authLoading } = useAuthContext()

  // Text conversation state
  const [isConnected, setIsConnected] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Disconnected')
  const [messages, setMessages] = useState<TextMessage[]>([])
  const [isTyping, setIsTyping] = useState(false)

  // Service reference
  const [textService, setTextService] = useState<TextConversationService | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // Initialize text conversation service when user is authenticated
  useEffect(() => {
    if (authLoading) {
      return // Wait for auth to finish loading
    }

    if (!isAuthenticated || !user?.user_id) {
      // Clean up existing service if user is not authenticated
      if (textService) {
        textService.disconnect()
        setTextService(null)
      }

      // Reset state
      setIsConnected(false)
      setConnectionStatus('Disconnected')
      setMessages([])
      setIsTyping(false)
      return
    }

    // Create new service instance
    setIsLoading(true)

    try {
      const newService = new TextConversationService(user.user_id)

      // Set up event handlers
      newService.onStateChange = (state: TextConversationState) => {
        setIsConnected(state.isConnected)
        setConnectionStatus(state.connectionStatus)
        setMessages(state.messages)
        setIsTyping(state.isTyping)
      }

      newService.onMessage = (message: TextMessage) => {
        // Individual message handler (optional, state change handler already updates messages)
        console.log('💬 New message received:', message)
      }

      newService.onError = (error: Error) => {
        console.error('💬 Text conversation error:', error)
      }

      setTextService(newService)
      console.log('💬 Text conversation service initialized for user:', user.user_id)
    } catch (error) {
      console.error('💬 Failed to initialize text conversation service:', error)
    } finally {
      setIsLoading(false)
    }

    // Cleanup on unmount or when user changes
    return () => {
      if (textService) {
        textService.cleanup()
      }
    }
  }, [isAuthenticated, user?.user_id, authLoading])

  // Connection methods
  const connect = async () => {
    if (!textService || !isReady) {
      console.log('💬 Cannot connect: Service not ready')
      return
    }

    try {
      await textService.connect()
    } catch (error) {
      console.error('💬 Failed to connect:', error)
    }
  }

  const disconnect = () => {
    if (!textService) return

    try {
      textService.disconnect()
    } catch (error) {
      console.error('💬 Failed to disconnect:', error)
    }
  }

  const sendMessage = (content: string) => {
    if (!textService || !isReady) {
      console.log('💬 Cannot send message: Service not ready')
      return
    }

    if (!content.trim()) {
      console.log('💬 Cannot send empty message')
      return
    }

    textService.sendMessage(content.trim())
  }

  const clearMessages = () => {
    if (!textService || !isReady) {
      console.log('💬 Cannot clear messages: Service not ready')
      return
    }

    textService.clearMessages()
  }

  const updateCurrentPage = (page: string) => {
    if (!textService || !isReady) {
      console.log('💬 Cannot update page: Service not ready')
      return
    }

    textService.updateCurrentPage(page)
  }

  const getState = (): TextConversationState => {
    if (!textService || !isReady) {
      return {
        isConnected: false,
        connectionStatus: 'Disconnected',
        messages: [],
        isTyping: false
      }
    }

    return textService.getState()
  }

  // Computed values
  const isReady = !!textService && isAuthenticated && !!user?.user_id

  const contextValue: TextConversationContextType = {
    // State
    isConnected,
    connectionStatus,
    messages,
    isTyping,

    // Status
    isReady,
    isLoading: authLoading || isLoading,

    // Actions
    connect,
    disconnect,
    sendMessage,
    clearMessages,
    updateCurrentPage,
    getState,
  }

  return (
    <TextConversationContext.Provider value={contextValue}>
      {children}
    </TextConversationContext.Provider>
  )
}

// Custom hook to use text conversation context
export function useTextConversation() {
  const context = useContext(TextConversationContext)

  if (context === undefined) {
    throw new Error('useTextConversation must be used within a TextConversationProvider')
  }

  return context
}
