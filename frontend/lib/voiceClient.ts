'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { config, getRTVIConfig } from './config'

export interface VoiceMessage {
  id: string
  type: 'user' | 'assistant' | 'system' | 'error' | 'tool_call' | 'tool_result'
  content: string
  timestamp: Date
  isAudio?: boolean
  toolCall?: ToolCall
  toolResult?: ToolResult
}

export interface ToolCall {
  toolName: string
  action: string
  parameters: Record<string, any>
  timestamp: Date
  status: 'pending' | 'executing' | 'completed' | 'failed'
}

export interface ToolResult {
  toolName: string
  action: string
  result: any
  timestamp: Date
  success: boolean
  displayData?: {
    title: string
    summary: string
    details: Record<string, any>
    visualizations?: any[]
  }
}

interface VoiceClientHook {
  isConnected: boolean
  isInConversation: boolean
  connectionStatus: string
  messages: VoiceMessage[]
  toolCallsInProgress: ToolCall[]
  recentToolResults: ToolResult[]
  isToolsConnected: boolean
  
  // Connection methods
  connect: () => void
  disconnect: () => void
  startConversation: () => void
  stopConversation: () => void
  connectToolWebSocket: () => void
  
  // Utility methods
  clearMessages: () => void
  clearToolCalls: () => void
}

export function useVoiceClient(): VoiceClientHook {
  // Connection state
  const [isConnected, setIsConnected] = useState(false)
  const [isInConversation, setIsInConversation] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Disconnected')
  const [messages, setMessages] = useState<VoiceMessage[]>([])
  
  // Tool state
  const [toolCallsInProgress, setToolCallsInProgress] = useState<ToolCall[]>([])
  const [recentToolResults, setRecentToolResults] = useState<ToolResult[]>([])
  const [isToolsConnected, setIsToolsConnected] = useState(false)
  
  // Client refs
  const mainWebSocketRef = useRef<any>(null) // RTVI client
  const toolWebSocketRef = useRef<WebSocket | null>(null)
  const productWebSocketRef = useRef<WebSocket | null>(null)

  const addMessage = useCallback((message: Omit<VoiceMessage, 'id' | 'timestamp'>) => {
    const newMessage: VoiceMessage = {
      ...message,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    }
    console.log('🎤 Adding message:', newMessage)
    setMessages(prev => {
      const updated = [...prev, newMessage]
      console.log('🎤 Total messages now:', updated.length)
      return updated
    })
  }, [])

  const connectToolWebSocket = useCallback(() => {
    if (toolWebSocketRef.current?.readyState === WebSocket.OPEN) {
      console.log('🔧 Tool WebSocket already connected')
      return
    }

    console.log('🔧 Attempting to connect to Tool WebSocket...')
    
    // For development with self-signed certificates, we need to handle SSL errors
    const toolWsUrl = config.websocket.tools(config.defaultUserId.trim())
    console.log('🔧 Tool WebSocket URL:', toolWsUrl)
    
    const toolWs = new WebSocket(toolWsUrl)
    
    toolWs.onopen = () => {
      console.log('✅ Tool WebSocket connected successfully')
      setIsToolsConnected(true)
      addMessage({
        type: 'system',
        content: '🔧 Tool WebSocket connected'
      })
    }
    
    toolWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('🔧 Tool WebSocket message received:', data)
        
        // Dispatch message to WebSocket monitor
        window.dispatchEvent(new CustomEvent('websocket-message', {
          detail: event
        }))
        
        handleToolAction(data)
      } catch (error) {
        console.error('Error parsing tool message:', error)
      }
    }
    
    toolWs.onclose = (event) => {
      console.log('🔌 Tool WebSocket disconnected:', event.code, event.reason)
      setIsToolsConnected(false)
      addMessage({
        type: 'system',
        content: '🔌 Tool WebSocket disconnected'
      })
    }
    
    toolWs.onerror = (error) => {
      console.error('❌ Tool WebSocket error:', error)
      setIsToolsConnected(false)
      addMessage({
        type: 'error',
        content: '❌ Tool WebSocket connection error - Backend may not be running or SSL certificate issue'
      })
      
      // Retry connection after 5 seconds
      setTimeout(() => {
        if (!isToolsConnected) {
          console.log('🔄 Retrying tool WebSocket connection...')
          connectToolWebSocket()
        }
      }, 5000)
    }
    
    toolWebSocketRef.current = toolWs
  }, [addMessage])

  const connect = useCallback(async () => {
    if (isConnected) {
      return
    }

    try {
      setConnectionStatus('Connecting...')
      
      // Connect to tool WebSocket first
      connectToolWebSocket()

      // Import the WebSocket transport (this handles protobuf serialization)
      const { WebSocketTransport } = await import('@pipecat-ai/websocket-transport')
      
      const transport = new WebSocketTransport()
      
      const rtviConfig = {
        transport,
        params: getRTVIConfig(),
        enableMic: true,
        enableCam: false,
        callbacks: {
          onConnected: () => {
            setIsConnected(true)
            setConnectionStatus('Connected')
            addMessage({
              type: 'system',
              content: '✅ Connected to ESAP Voice Agent'
            })
          },
          onDisconnected: () => {
            setIsConnected(false)
            setIsInConversation(false)
            setConnectionStatus('Disconnected')
            addMessage({
              type: 'system',
              content: '👋 Disconnected from voice conversation'
            })
          },
          onBotReady: (data: any) => {
            console.log(`🎤 Voice bot ready:`, data)
            addMessage({
              type: 'system',
              content: '🤖 AI is ready to chat!'
            })
          },
          onUserTranscript: (data: any) => {
            console.log('🎤 User transcript received:', data)
            if (data.text && data.text.trim()) {
              addMessage({
                type: 'user',
                content: data.text,
                isAudio: true
              })
            }
          },
          onBotTranscript: (data: any) => {
            console.log('🎤 Bot transcript received:', data)
            if (data.text && data.text.trim()) {
              addMessage({
                type: 'assistant',
                content: data.text
              })
            }
          },
          onMessageError: (error: any) => {
            console.error('🎤 Message error:', error)
            addMessage({
              type: 'error',
              content: `Message error: ${error.message || 'Unknown error'}`
            })
          },
          onError: (error: any) => {
            console.error('🎤 Error:', error)
            addMessage({
              type: 'error',
              content: `Connection error: ${error.message || 'Unknown error'}`
            })
          },
        },
      }

      // Import RTVI client dynamically
      const { RTVIClient } = await import('@pipecat-ai/client-js')
      const rtviClient = new RTVIClient(rtviConfig)
      
      console.log('🎤 Initializing devices...')
      await rtviClient.initDevices()

      console.log('🎤 Connecting to voice bot...')
      await rtviClient.connect()

      // Store the client reference
      mainWebSocketRef.current = rtviClient as any

    } catch (error) {
      console.error('Failed to connect:', error)
      setConnectionStatus('Connection Failed')
      addMessage({
        type: 'error',
        content: `Failed to connect to ESAP Voice Agent: ${error instanceof Error ? error.message : 'Unknown error'}`
      })
    }
  }, [addMessage, connectToolWebSocket, isConnected])

  const disconnect = useCallback(async () => {
    console.log('🎤 Disconnecting from voice conversation...')
    
    if (mainWebSocketRef.current) {
      try {
        await (mainWebSocketRef.current as any).disconnect()
      } catch (error) {
        console.error('Error disconnecting RTVI client:', error)
      }
      mainWebSocketRef.current = null
    }
    
    if (toolWebSocketRef.current) {
      toolWebSocketRef.current.close()
      toolWebSocketRef.current = null
    }
    
    if (productWebSocketRef.current) {
      productWebSocketRef.current.close()
      productWebSocketRef.current = null
    }
    
    setIsConnected(false)
    setIsInConversation(false)
    setConnectionStatus('Disconnected')
  }, [])

  const startConversation = useCallback(async () => {
    console.log('🎤 Starting voice conversation...')
    
    if (!isConnected || isInConversation) {
      console.log('🎤 Cannot start conversation: not connected or already in conversation')
      return
    }

    try {
      setIsInConversation(true)
      
      addMessage({
        type: 'system',
        content: '🎙️ Voice conversation started! Speak naturally.'
      })
      
    } catch (error) {
      console.error('🎤 Error starting conversation:', error)
      addMessage({
        type: 'error',
        content: `Failed to start conversation: ${(error as Error).message}`
      })
      setIsInConversation(false)
    }
  }, [isConnected, isInConversation, addMessage])

  const stopConversation = useCallback(() => {
    console.log('🎤 Stopping voice conversation...')
    
    setIsInConversation(false)
    
    addMessage({
      type: 'system',
      content: '⏸️ Conversation paused'
    })
  }, [addMessage])

  // Handle tool actions
  const handleToolAction = useCallback((data: any) => {
    console.log('🔧 Tool action:', data)
    
    switch (data.type) {
      case 'mssql_search_result':
        addMessage({
          type: 'system',
          content: `🔍 MSSQL Search: ${data.action}`,
        })
        break
      
      case 'quotation_command':
        addMessage({
          type: 'system',
          content: `🔧 Quotation Tool: ${data.action}`,
        })
        break
      
      case 'navigation_command':
        addMessage({
          type: 'system',
          content: `🧭 Navigation Tool: ${data.action}`,
        })
        break
      
      case 'product_command':
        addMessage({
          type: 'system',
          content: `📦 Product Info Tool: ${data.action}`,
        })
        break

      default:
        addMessage({
          type: 'system',
          content: `🔧 Tool activated: ${data.type}`,
        })
    }
  }, [addMessage])

  const clearMessages = useCallback(() => {
    setMessages([])
    console.log('🧹 Messages cleared')
  }, [])

  const clearToolCalls = useCallback(() => {
    setToolCallsInProgress([])
    setRecentToolResults([])
    console.log('🧹 Tool calls cleared')
  }, [])

  // Auto-connect tool WebSocket on mount
  useEffect(() => {
    console.log('🔧 Auto-connecting tool WebSocket...')
    
    // First test if backend is reachable
    fetch(config.api.health)
      .then(response => {
        if (response.ok) {
          console.log('✅ Backend health check passed')
          connectToolWebSocket()
        } else {
          console.error('❌ Backend health check failed:', response.status)
          addMessage({
            type: 'error',
            content: `❌ Backend health check failed: ${response.status}`
          })
        }
      })
      .catch(error => {
        console.error('❌ Backend health check error:', error)
        addMessage({
          type: 'error',
          content: '❌ Cannot reach backend - please ensure it is running on https://176.9.16.194:8200'
        })
      })
  }, [connectToolWebSocket, addMessage])

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
    toolCallsInProgress,
    recentToolResults,
    isToolsConnected,
    
    // Connection methods
    connect,
    disconnect,
    startConversation,
    stopConversation,
    connectToolWebSocket,
    
    // Utility methods
    clearMessages,
    clearToolCalls,
  }
}
