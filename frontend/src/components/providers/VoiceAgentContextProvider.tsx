'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { useAuthContext } from '@/components/providers'
import { VoiceAgentService } from '@/lib/voice-agent'
import { VoiceMessage } from '@/lib/voice-agent/types'

interface VoiceAgentContextType {
  // Service state
  isConnected: boolean
  isInConversation: boolean
  connectionStatus: string
  currentPage: string
  previousPage: string | null
  
  // Messages
  messages: VoiceMessage[]
  
  // Service status
  isReady: boolean
  isLoading: boolean
  
  // Actions
  connect: () => Promise<void>
  disconnect: () => Promise<void>
  startConversation: () => Promise<void>
  stopConversation: () => void
  clearMessages: () => void
  sendMessage: (message: string) => void
  
  // Navigation actions
  navigateToPage: (page: string) => void
  clickElement: (elementName: string) => void
  executeSearch: (query: string, type: 'database' | 'file') => void
  handleFileUpload: (descriptions: string[], tableNames: string[]) => void
  viewReport: (request: string) => void
  generateReport: (query: string) => void
  testNavigation: (page: string) => void
  
  // Debug methods
  refreshPageState: () => void
  getCurrentPageState: () => { currentPage: string; previousPage: string | null }
  
  // Context management
  getContext: () => any
  setContext: (context: any) => void
  clearContext: () => void
}

const VoiceAgentContext = createContext<VoiceAgentContextType | undefined>(undefined)

interface VoiceAgentProviderProps {
  children: ReactNode
}

export function VoiceAgentProvider({ children }: VoiceAgentProviderProps) {
  const { user, isAuthenticated, isLoading: authLoading } = useAuthContext()
  
  // Voice agent state
  const [isConnected, setIsConnected] = useState(false)
  const [isInConversation, setIsInConversation] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Disconnected')
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [previousPage, setPreviousPage] = useState<string | null>(null)
  const [messages, setMessages] = useState<VoiceMessage[]>([])
  
  // Service reference
  const [voiceAgentService, setVoiceAgentService] = useState<VoiceAgentService | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // Initialize voice agent service when user is authenticated
  useEffect(() => {
    if (authLoading) {
      return // Wait for auth to finish loading
    }

    if (!isAuthenticated || !user?.user_id) {
      // Clean up existing service if user is not authenticated
      if (voiceAgentService) {
        voiceAgentService.cleanup()
        setVoiceAgentService(null)
      }
      
      // Reset state
      setIsConnected(false)
      setIsInConversation(false)
      setConnectionStatus('Disconnected')
      setMessages([])
      setCurrentPage('dashboard')
      setPreviousPage(null)
      return
    }

    // Use singleton pattern to get or create service
    setIsLoading(true)
    
    try {
      const newService = VoiceAgentService.getInstance(
        user.user_id,
        // State change handler
        (state) => {
          setIsConnected(state.isConnected)
          setIsInConversation(state.isInConversation)
          setConnectionStatus(state.connectionStatus)
          setCurrentPage(state.currentPage)
          setPreviousPage(state.previousPage)
        },
        // Message handler
        (message) => {
          setMessages(prev => [...prev, message])
        }
      )
      
      setVoiceAgentService(newService)
      console.log('🎤 Voice agent service initialized for user:', user.user_id)
    } catch (error) {
      console.error('🎤 Failed to initialize voice agent service:', error)
    } finally {
      setIsLoading(false)
    }

    // Cleanup on unmount or when user changes
    return () => {
      // Don't cleanup the service here as it's now a singleton
      // The service will be cleaned up when the user changes or the app unmounts
    }
  }, [isAuthenticated, user?.user_id, authLoading])

  // Connection methods
  const connect = async () => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot connect: Service not ready')
      return
    }
    
    try {
      await voiceAgentService.connect()
    } catch (error) {
      console.error('🎤 Failed to connect:', error)
    }
  }

  const disconnect = async () => {
    if (!voiceAgentService) return
    
    try {
      await voiceAgentService.disconnect()
    } catch (error) {
      console.error('🎤 Failed to disconnect:', error)
    }
  }

  const startConversation = async () => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot start conversation: Service not ready')
      return
    }
    
    try {
      await voiceAgentService.startConversation()
    } catch (error) {
      console.error('🎤 Failed to start conversation:', error)
    }
  }

  const stopConversation = () => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot stop conversation: Service not ready')
      return
    }
    
    voiceAgentService.stopConversation()
  }

  const clearMessages = () => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot clear messages: Service not ready')
      return
    }
    
    voiceAgentService.clearMessages()
  }

  const sendMessage = (message: string) => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot send message: Service not ready')
      return
    }
    
    voiceAgentService.sendMessage(message)
  }

  // Navigation methods
  const navigateToPage = (page: string) => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot navigate: Service not ready')
      return
    }
    
    voiceAgentService.navigateToPage(page)
  }

  const clickElement = (elementName: string) => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot click element: Service not ready')
      return
    }
    
    voiceAgentService.clickElement(elementName)
  }

  const executeSearch = (query: string, type: 'database' | 'file') => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot execute search: Service not ready')
      return
    }
    
    voiceAgentService.executeSearch(query, type)
  }

  const handleFileUpload = (descriptions: string[], tableNames: string[]) => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot handle file upload: Service not ready')
      return
    }
    
    voiceAgentService.handleFileUpload(descriptions, tableNames)
  }

  const viewReport = (request: string) => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot view report: Service not ready')
      return
    }
    
    voiceAgentService.viewReport(request)
  }

  const generateReport = (query: string) => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot generate report: Service not ready')
      return
    }
    
    voiceAgentService.generateReport(query)
  }

  const testNavigation = (page: string) => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot test navigation: Service not ready')
      return
    }
    
    voiceAgentService.testNavigation(page)
  }

  // Debug methods
  const refreshPageState = () => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot refresh page state: Service not ready')
      return
    }
    
    voiceAgentService.refreshPageState()
  }

  const getCurrentPageState = () => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot get page state: Service not ready')
      return { currentPage: 'unknown', previousPage: null }
    }
    
    return voiceAgentService.getCurrentPageState() || { currentPage: 'unknown', previousPage: null }
  }

  // Context management methods
  const getContext = () => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot get context: Service not ready')
      return null
    }
    
    return voiceAgentService.getContext()
  }

  const setContext = (context: any) => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot set context: Service not ready')
      return
    }
    
    voiceAgentService.setContext(context)
  }

  const clearContext = () => {
    if (!voiceAgentService || !isReady) {
      console.log('🎤 Cannot clear context: Service not ready')
      return
    }
    
    voiceAgentService.clearContext()
  }

  // Computed values
  const isReady = !!voiceAgentService && isAuthenticated && !!user?.user_id

  const contextValue: VoiceAgentContextType = {
    // State
    isConnected,
    isInConversation,
    connectionStatus,
    currentPage,
    previousPage,
    messages,
    
    // Status
    isReady,
    isLoading: authLoading || isLoading,
    
    // Actions
    connect,
    disconnect,
    startConversation,
    stopConversation,
    clearMessages,
    sendMessage,
    
    // Navigation
    navigateToPage,
    clickElement,
    executeSearch,
    handleFileUpload,
    viewReport,
    generateReport,
    testNavigation,
    
    // Debug
    refreshPageState,
    getCurrentPageState,
    
    // Context management
    getContext,
    setContext,
    clearContext,
  }

  return (
    <VoiceAgentContext.Provider value={contextValue}>
      {children}
    </VoiceAgentContext.Provider>
  )
}

// Custom hook to use voice agent context
export function useVoiceAgent() {
  const context = useContext(VoiceAgentContext)
  
  if (context === undefined) {
    throw new Error('useVoiceAgent must be used within a VoiceAgentProvider')
  }
  
  return context
} 