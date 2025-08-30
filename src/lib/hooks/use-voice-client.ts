'use client'

import { useState, useCallback, useRef, useEffect } from 'react'

export interface VoiceMessage {
  id: string
  type: 'user' | 'assistant' | 'system' | 'error' | 'tool_call' | 'tool_result'
  content: string
  timestamp: Date
  isAudio?: boolean
}

interface VoiceClientHook {
  isConnected: boolean
  isInConversation: boolean
  connectionStatus: string
  messages: VoiceMessage[]
  
  // Connection methods
  connect: () => void
  disconnect: () => void
  startConversation: () => void
  stopConversation: () => void
  
  // Utility methods
  clearMessages: () => void
  sendMessage: (message: string) => void
}

export function useVoiceClient(): VoiceClientHook {
  // Connection state
  const [isConnected, setIsConnected] = useState(false)
  const [isInConversation, setIsInConversation] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Disconnected')
  const [messages, setMessages] = useState<VoiceMessage[]>([])
  
  // WebSocket ref
  const voiceWsRef = useRef<WebSocket | null>(null)
  
  // Backend URL from environment
  const BACKEND_URL = process.env.NEXT_PUBLIC_PROD_BACKEND_URL || 'https://176.9.16.194:8200'

  const addMessage = useCallback((message: Omit<VoiceMessage, 'id' | 'timestamp'>) => {
    const newMessage: VoiceMessage = {
      ...message,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    }
    console.log('🎤 Adding message:', newMessage)
    setMessages(prev => [...prev, newMessage])
  }, [])

  const connect = useCallback(async () => {
    if (isConnected || voiceWsRef.current?.readyState === WebSocket.OPEN) {
      console.log('🎤 Already connected')
      return
    }

    try {
      setConnectionStatus('Connecting...')
      
      // Health check first
      try {
        const healthResponse = await fetch(`${BACKEND_URL}/voice/health`)
        if (!healthResponse.ok) {
          throw new Error(`Health check failed: ${healthResponse.status}`)
        }
        console.log('✅ Backend health check passed')
      } catch (error) {
        console.error('❌ Backend health check failed:', error)
        addMessage({
          type: 'error',
          content: 'Cannot reach backend - please ensure it is running'
        })
        setConnectionStatus('Connection Failed')
        return
      }

      // Connect to WebSocket
      const wsUrl = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://') + '/voice/ws'
      console.log('🎤 Connecting to:', wsUrl)
      
      const voiceWs = new WebSocket(wsUrl)
      
      voiceWs.onopen = () => {
        console.log('✅ Voice WebSocket connected')
        setIsConnected(true)
        setConnectionStatus('Connected')
        addMessage({
          type: 'system',
          content: '✅ Connected to ESAP Voice Agent'
        })
      }
      
      voiceWs.onmessage = (event) => {
        try {
          console.log('🎤 Message received:', event.data)
          
          let data: any
          
          // Handle different message types
          if (event.data instanceof Blob) {
            // Skip binary data for now
            console.log('🎤 Binary message received, skipping')
            return
          } else if (typeof event.data === 'string') {
            data = JSON.parse(event.data)
          } else {
            data = event.data
          }
          
          // Handle the message
          if (data.type === 'assistant' && data.text) {
            addMessage({
              type: 'assistant',
              content: data.text
            })
          } else if (data.type === 'user' && data.text) {
            addMessage({
              type: 'user',
              content: data.text,
              isAudio: true
            })
          } else if (data.type === 'system') {
            addMessage({
              type: 'system',
              content: data.message || 'System message received'
            })
          } else if (data.interaction_type) {
            addMessage({
              type: 'system',
              content: `Voice navigation: ${data.interaction_type}`
            })
          } else {
            addMessage({
              type: 'system',
              content: 'Message received from backend'
            })
          }
          
        } catch (error) {
          console.error('Error handling message:', error)
          addMessage({
            type: 'error',
            content: 'Error processing message from backend'
          })
        }
      }
      
      voiceWs.onclose = (event) => {
        console.log('🔌 Voice WebSocket disconnected:', event.code, event.reason)
        setIsConnected(false)
        setIsInConversation(false)
        setConnectionStatus('Disconnected')
        addMessage({
          type: 'system',
          content: '👋 Disconnected from voice agent'
        })
      }
      
      voiceWs.onerror = (error) => {
        console.error('❌ Voice WebSocket error:', error)
        setIsConnected(false)
        setIsInConversation(false)
        setConnectionStatus('Connection Failed')
        addMessage({
          type: 'error',
          content: '❌ Connection error - Backend may not be running'
        })
      }
      
      voiceWsRef.current = voiceWs
      
    } catch (error) {
      console.error('Failed to connect:', error)
      setConnectionStatus('Connection Failed')
      addMessage({
        type: 'error',
        content: `Failed to connect: ${error instanceof Error ? error.message : 'Unknown error'}`
      })
    }
  }, [isConnected, addMessage])

  const disconnect = useCallback(() => {
    console.log('🎤 Disconnecting...')
    
    if (voiceWsRef.current) {
      voiceWsRef.current.close()
      voiceWsRef.current = null
    }
    
    setIsConnected(false)
    setIsInConversation(false)
    setConnectionStatus('Disconnected')
  }, [])

  const startConversation = useCallback(() => {
    if (!isConnected) {
      console.log('🎤 Cannot start conversation: not connected')
      return
    }

    setIsInConversation(true)
    addMessage({
      type: 'system',
      content: '🎙️ Voice conversation started! Speak naturally.'
    })
  }, [isConnected, addMessage])

  const stopConversation = useCallback(() => {
    setIsInConversation(false)
    addMessage({
      type: 'system',
      content: '⏸️ Conversation paused'
    })
  }, [addMessage])

  const sendMessage = useCallback((message: string) => {
    if (!isConnected || !voiceWsRef.current) {
      console.log('🎤 Cannot send message: not connected')
      return
    }

    try {
      const payload = {
        type: 'text_message',
        text: message,
        timestamp: new Date().toISOString()
      }
      
      voiceWsRef.current.send(JSON.stringify(payload))
      
      addMessage({
        type: 'user',
        content: message
      })
      
    } catch (error) {
      console.error('Error sending message:', error)
      addMessage({
        type: 'error',
        content: 'Failed to send message'
      })
    }
  }, [isConnected, addMessage])

  const clearMessages = useCallback(() => {
    setMessages([])
    console.log('🧹 Messages cleared')
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    isConnected,
    isInConversation,
    connectionStatus,
    messages,
    
    // Connection methods
    connect,
    disconnect,
    startConversation,
    stopConversation,
    
    // Utility methods
    clearMessages,
    sendMessage,
  }
}