import { useState, useEffect, useCallback } from 'react'
import { useVoiceAgent } from '@/components/providers/VoiceAgentContextProvider'

interface VoiceAgentConnectionState {
  isConnected: boolean
  isConnecting: boolean
  connectionStatus: string
  lastConnectedAt: Date | null
  connectionAttempts: number
  autoReconnectEnabled: boolean
}

interface VoiceAgentConnectionActions {
  connect: () => Promise<void>
  disconnect: () => Promise<void>
  reconnect: () => Promise<void>
  toggleAutoReconnect: () => void
  resetConnectionState: () => void
}

export function useVoiceAgentConnection(): VoiceAgentConnectionState & VoiceAgentConnectionActions {
  const { isConnected, connectionStatus, isReady } = useVoiceAgent()
  
  const [connectionState, setConnectionState] = useState<VoiceAgentConnectionState>({
    isConnected: false,
    isConnecting: false,
    connectionStatus: 'Disconnected',
    lastConnectedAt: null,
    connectionAttempts: 0,
    autoReconnectEnabled: true
  })

  // Update connection state when voice agent state changes
  useEffect(() => {
    setConnectionState(prev => ({
      ...prev,
      isConnected,
      connectionStatus,
      lastConnectedAt: isConnected ? new Date() : prev.lastConnectedAt
    }))
  }, [isConnected, connectionStatus])

  // Auto-reconnect logic
  useEffect(() => {
    if (!connectionState.autoReconnectEnabled || !isReady) return

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && !isConnected && isReady) {
        console.log('🎤 Page became visible, attempting to reconnect...')
        reconnect()
      }
    }

    const handleOnline = () => {
      if (!isConnected && isReady) {
        console.log('🎤 Network came online, attempting to reconnect...')
        reconnect()
      }
    }

    // Listen for page visibility changes
    document.addEventListener('visibilitychange', handleVisibilityChange)
    
    // Listen for network online events
    window.addEventListener('online', handleOnline)

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('online', handleOnline)
    }
  }, [connectionState.autoReconnectEnabled, isConnected, isReady])

  // Connection methods
  const connect = useCallback(async () => {
    if (!isReady) {
      console.log('🎤 Cannot connect: Voice agent not ready')
      return
    }

    setConnectionState(prev => ({
      ...prev,
      isConnecting: true,
      connectionAttempts: prev.connectionAttempts + 1
    }))

    try {
      const { connect: connectVoiceAgent } = useVoiceAgent()
      await connectVoiceAgent()
      
      setConnectionState(prev => ({
        ...prev,
        isConnecting: false,
        lastConnectedAt: new Date()
      }))
    } catch (error) {
      console.error('🎤 Connection failed:', error)
      setConnectionState(prev => ({
        ...prev,
        isConnecting: false
      }))
      throw error
    }
  }, [isReady])

  const disconnect = useCallback(async () => {
    setConnectionState(prev => ({
      ...prev,
      isConnecting: true
    }))

    try {
      const { disconnect: disconnectVoiceAgent } = useVoiceAgent()
      await disconnectVoiceAgent()
      
      setConnectionState(prev => ({
        ...prev,
        isConnecting: false
      }))
    } catch (error) {
      console.error('🎤 Disconnect failed:', error)
      setConnectionState(prev => ({
        ...prev,
        isConnecting: false
      }))
      throw error
    }
  }, [])

  const reconnect = useCallback(async () => {
    if (connectionState.isConnecting) return

    try {
      await disconnect()
      // Small delay before reconnecting
      await new Promise(resolve => setTimeout(resolve, 1000))
      await connect()
    } catch (error) {
      console.error('🎤 Reconnection failed:', error)
      throw error
    }
  }, [connect, disconnect, connectionState.isConnecting])

  const toggleAutoReconnect = useCallback(() => {
    setConnectionState(prev => ({
      ...prev,
      autoReconnectEnabled: !prev.autoReconnectEnabled
    }))
  }, [])

  const resetConnectionState = useCallback(() => {
    setConnectionState({
      isConnected: false,
      isConnecting: false,
      connectionStatus: 'Disconnected',
      lastConnectedAt: null,
      connectionAttempts: 0,
      autoReconnectEnabled: true
    })
  }, [])

  return {
    ...connectionState,
    connect,
    disconnect,
    reconnect,
    toggleAutoReconnect,
    resetConnectionState
  }
} 