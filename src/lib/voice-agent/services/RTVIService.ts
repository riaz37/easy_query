import { VOICE_AGENT_CONFIG } from '../config'
import { MessageService } from './MessageService'
import { NavigationService } from './NavigationService'

export class RTVIService {
  private rtviClient: any = null
  private isInitialized = false

  // Event handlers
  private onConnected?: () => void
  private onDisconnected?: () => void
  private onBotReady?: (data: any) => void
  private onUserTranscript?: (data: any) => void
  private onBotTranscript?: (data: any) => void
  private onError?: (error: Error) => void

  constructor(
    onConnected?: () => void,
    onDisconnected?: () => void,
    onBotReady?: (data: any) => void,
    onUserTranscript?: (data: any) => void,
    onBotTranscript?: (data: any) => void,
    onError?: (error: Error) => void
  ) {
    this.onConnected = onConnected
    this.onDisconnected = onDisconnected
    this.onBotReady = onBotReady
    this.onUserTranscript = onUserTranscript
    this.onBotTranscript = onBotTranscript
    this.onError = onError
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('🎤 RTVI client already initialized')
      return
    }

    try {
      console.log('🎤 Initializing RTVI client for audio...')
      
      // Import RTVI client dynamically
      const { RTVIClient } = await import('@pipecat-ai/client-js')
      const { WebSocketTransport } = await import('@pipecat-ai/websocket-transport')
      
      const transport = new WebSocketTransport()
      
      const rtviConfig = {
        transport,
        params: {
          baseUrl: VOICE_AGENT_CONFIG.BACKEND_URL,
          endpoints: { connect: VOICE_AGENT_CONFIG.WEBSOCKET_ENDPOINTS.VOICE_CONNECT }
        },
        enableMic: VOICE_AGENT_CONFIG.RTVI.ENABLE_MIC,
        enableCam: VOICE_AGENT_CONFIG.RTVI.ENABLE_CAM,
        callbacks: {
          onConnected: () => {
            console.log('✅ RTVI client connected')
            this.onConnected?.()
          },
          onDisconnected: () => {
            console.log('🔌 RTVI client disconnected')
            this.onDisconnected?.()
          },
          onBotReady: (data: any) => {
            console.log('🎤 Voice bot ready:', data)
            this.onBotReady?.(data)
          },
          onUserTranscript: (data: any) => {
            console.log('🎤 User transcript received:', data)
            this.onUserTranscript?.(data)
          },
          onBotTranscript: (data: any) => {
            console.log('🎤 Bot transcript received:', data)
            this.onBotTranscript?.(data)
          },
          onMessageError: (error: any) => {
            console.error('🎤 RTVI message error:', error)
            this.onError?.(new Error(`Audio error: ${error.message || 'Unknown error'}`))
          },
          onError: (error: any) => {
            console.error('🎤 RTVI error:', error)
            this.onError?.(new Error(`Audio connection error: ${error.message || 'Unknown error'}`))
          },
        },
      }

      this.rtviClient = new RTVIClient(rtviConfig)
      
      console.log('🎤 Initializing audio devices...')
      await this.rtviClient.initDevices()

      this.isInitialized = true
      console.log('✅ RTVI client initialized successfully')
      
    } catch (error) {
      console.error('Failed to initialize RTVI client:', error)
      this.onError?.(new Error(`Failed to initialize audio: ${error instanceof Error ? error.message : 'Unknown error'}`))
      throw error
    }
  }

  async connect(): Promise<void> {
    if (!this.isInitialized || !this.rtviClient) {
      throw new Error('RTVI client not initialized')
    }

    try {
      console.log('🎤 Connecting to voice bot...')
      await this.rtviClient.connect()
      console.log('✅ RTVI client connected to voice bot')
      
    } catch (error) {
      console.error('Failed to connect RTVI client:', error)
      this.onError?.(new Error(`Failed to connect to voice bot: ${error instanceof Error ? error.message : 'Unknown error'}`))
      throw error
    }
  }

  async disconnect(): Promise<void> {
    if (!this.rtviClient) {
      console.log('🎤 RTVI client not connected')
      return
    }

    try {
      await this.rtviClient.disconnect()
      console.log('✅ RTVI client disconnected')
      this.rtviClient = null
      this.isInitialized = false
      
    } catch (error) {
      console.error('Error disconnecting RTVI client:', error)
      this.onError?.(new Error(`Failed to disconnect: ${error instanceof Error ? error.message : 'Unknown error'}`))
    }
  }

  isConnected(): boolean {
    return this.rtviClient !== null && this.isInitialized
  }

  // Audio processing only - navigation is handled by backend commands
  processBotTranscript(text: string): void {
    if (!text || !text.trim()) return

    console.log('🎤 Bot transcript received (audio only):', text)
    // Navigation commands come from backend via WebSocket, not from parsing transcripts
  }

  // Audio device management
  async getAudioDevices(): Promise<MediaDeviceInfo[]> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      return devices.filter(device => device.kind === 'audioinput')
    } catch (error) {
      console.error('Failed to get audio devices:', error)
      return []
    }
  }

  async setAudioDevice(deviceId: string): Promise<void> {
    if (!this.rtviClient) {
      throw new Error('RTVI client not connected')
    }

    try {
      // This would depend on the specific RTVI client API
      // await this.rtviClient.setAudioDevice(deviceId)
      console.log('🎤 Audio device set to:', deviceId)
    } catch (error) {
      console.error('Failed to set audio device:', error)
      throw error
    }
  }

  // Audio level monitoring
  startAudioLevelMonitoring(): void {
    if (!this.rtviClient) return

    try {
      // This would depend on the specific RTVI client API
      // this.rtviClient.startAudioLevelMonitoring()
      console.log('🎤 Audio level monitoring started')
    } catch (error) {
      console.error('Failed to start audio level monitoring:', error)
    }
  }

  stopAudioLevelMonitoring(): void {
    if (!this.rtviClient) return

    try {
      // This would depend on the specific RTVI client API
      // this.rtviClient.stopAudioLevelMonitoring()
      console.log('🎤 Audio level monitoring stopped')
    } catch (error) {
      console.error('Failed to stop audio level monitoring:', error)
    }
  }

  // Mute/unmute functionality
  async setMuted(muted: boolean): Promise<void> {
    if (!this.rtviClient) return

    try {
      // This would depend on the specific RTVI client API
      // await this.rtviClient.setMuted(muted)
      console.log('🎤 Audio muted:', muted)
    } catch (error) {
      console.error('Failed to set muted state:', error)
      throw error
    }
  }

  isMuted(): boolean {
    // This would depend on the specific RTVI client API
    // return this.rtviClient?.isMuted() || false
    return false
  }

  // Cleanup
  cleanup(): void {
    this.disconnect()
    this.rtviClient = null
    this.isInitialized = false
  }
} 