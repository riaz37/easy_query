'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { VoiceAgentService } from '../voice-agent'
import { VoiceClientHook, VoiceMessage } from '../voice-agent/types'

export function useVoiceClient(): VoiceClientHook {
  // State
  const [isConnected, setIsConnected] = useState(false)
  const [isInConversation, setIsInConversation] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Disconnected')
  const [messages, setMessages] = useState<VoiceMessage[]>([])
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [previousPage, setPreviousPage] = useState<string | null>(null)

  // Service reference
  const voiceAgentServiceRef = useRef<VoiceAgentService | null>(null)

  // Initialize voice agent service
  useEffect(() => {
    voiceAgentServiceRef.current = new VoiceAgentService(
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

    // Cleanup on unmount
    return () => {
      voiceAgentServiceRef.current?.cleanup()
    }
  }, [])

  // Connection methods
  const connect = useCallback(async () => {
    try {
      await voiceAgentServiceRef.current?.connect()
    } catch (error) {
      console.error('Failed to connect:', error)
    }
  }, [])

  const disconnect = useCallback(async () => {
    try {
      await voiceAgentServiceRef.current?.disconnect()
    } catch (error) {
      console.error('Failed to disconnect:', error)
    }
  }, [])

  const startConversation = useCallback(async () => {
    try {
      await voiceAgentServiceRef.current?.startConversation()
    } catch (error) {
      console.error('Failed to start conversation:', error)
    }
  }, [])

  const stopConversation = useCallback(() => {
    voiceAgentServiceRef.current?.stopConversation()
  }, [])

  // Utility methods
  const clearMessages = useCallback(() => {
    voiceAgentServiceRef.current?.clearMessages()
  }, [])

  const sendMessage = useCallback((message: string) => {
    voiceAgentServiceRef.current?.sendMessage(message)
  }, [])

  // Navigation methods
  const navigateToPage = useCallback((page: string) => {
    voiceAgentServiceRef.current?.navigateToPage(page)
  }, [])

  const clickElement = useCallback((elementName: string) => {
    voiceAgentServiceRef.current?.clickElement(elementName)
  }, [])

  const executeSearch = useCallback((query: string, type: 'database' | 'file') => {
    voiceAgentServiceRef.current?.executeSearch(query, type)
  }, [])

  const handleFileUpload = useCallback((descriptions: string[], tableNames: string[]) => {
    voiceAgentServiceRef.current?.handleFileUpload(descriptions, tableNames)
  }, [])

  const viewReport = useCallback((request: string) => {
    voiceAgentServiceRef.current?.viewReport(request)
  }, [])

  const generateReport = useCallback((query: string) => {
    voiceAgentServiceRef.current?.generateReport(query)
  }, [])

  const testNavigation = useCallback((page: string) => {
    voiceAgentServiceRef.current?.testNavigation(page)
  }, [])

  // Debug methods
  const refreshPageState = useCallback(() => {
    voiceAgentServiceRef.current?.refreshPageState()
  }, [])

  const getCurrentPageState = useCallback(() => {
    return voiceAgentServiceRef.current?.getCurrentPageState() || { currentPage: 'unknown', previousPage: null }
  }, [])

  return {
    isConnected,
    isInConversation,
    connectionStatus,
    messages,
    currentPage,
    previousPage,
    
    // Connection methods
    connect,
    disconnect,
    startConversation,
    stopConversation,
    
    // Utility methods
    clearMessages,
    sendMessage,
    
    // Navigation methods
    navigateToPage,
    clickElement,
    executeSearch,
    handleFileUpload,
    viewReport,
    generateReport,
    testNavigation,
    
    // Debug methods
    refreshPageState,
    getCurrentPageState,
  }
}