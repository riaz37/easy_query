import { VOICE_AGENT_CONFIG, getWebSocketUrl, getHealthCheckUrl } from '../config'
import { MessageService } from './MessageService'
import { NavigationData } from '../types'

export class WebSocketService {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = VOICE_AGENT_CONFIG.RTVI.RECONNECTION_ATTEMPTS
  private reconnectTimeout: NodeJS.Timeout | null = null

  // Event handlers
  private onMessageHandler?: (data: any) => void
  private onConnectionChange?: (isConnected: boolean) => void
  private onError?: (error: Error) => void

  constructor(
    onMessage?: (data: any) => void,
    onConnectionChange?: (isConnected: boolean) => void,
    onError?: (error: Error) => void
  ) {
    this.onMessageHandler = onMessage
    this.onConnectionChange = onConnectionChange
    this.onError = onError
  }

  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('🎤 WebSocket already connected')
      return
    }

    try {
      // Health check first
      await this.performHealthCheck()
      
      // Connect to WebSocket with user_id parameter
      const baseWsUrl = getWebSocketUrl(VOICE_AGENT_CONFIG.BACKEND_URL, VOICE_AGENT_CONFIG.WEBSOCKET_ENDPOINTS.VOICE_WS)
      const wsUrl = `${baseWsUrl}?user_id=${VOICE_AGENT_CONFIG.DEFAULTS.USER_ID}`
      console.log('🎤 Connecting to WebSocket:', wsUrl)
      
      this.ws = new WebSocket(wsUrl)
      this.setupWebSocketHandlers()
      
    } catch (error) {
      console.error('❌ WebSocket connection failed:', error)
      this.handleError(error as Error)
    }
  }

  private async performHealthCheck(): Promise<void> {
    try {
      const healthUrl = getHealthCheckUrl(VOICE_AGENT_CONFIG.BACKEND_URL)
      const response = await fetch(healthUrl)
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`)
      }
      
      console.log('✅ Backend health check passed')
    } catch (error) {
      console.error('❌ Backend health check failed:', error)
      throw new Error('Cannot reach backend - please ensure it is running')
    }
  }

  private setupWebSocketHandlers(): void {
    if (!this.ws) return

    this.ws.onopen = () => {
      console.log('✅ Voice WebSocket connected successfully')
      console.log('🎤 WebSocket URL:', this.ws?.url)
      console.log('🎤 WebSocket readyState:', this.ws?.readyState)
      this.reconnectAttempts = 0
      this.onConnectionChange?.(true)
      
      if (this.onMessageHandler) {
        this.onMessageHandler({
          type: 'system',
          message: 'WebSocket connected for tool communication'
        })
      }
    }

    this.ws.onmessage = (event) => {
      this.handleWebSocketMessage(event)
    }

    this.ws.onclose = (event) => {
      console.log('🔌 Voice WebSocket disconnected:', event.code, event.reason)
      this.onConnectionChange?.(false)
      
      if (this.onMessageHandler) {
        this.onMessageHandler({
          type: 'system',
          message: 'WebSocket disconnected'
        })
      }

      // Attempt reconnection if not manually closed
      if (event.code !== 1000) {
        this.scheduleReconnection()
      }
    }

    this.ws.onerror = (error) => {
      console.error('❌ Voice WebSocket error:', error)
      this.handleError(new Error('WebSocket connection error'))
    }
  }

  private handleWebSocketMessage(event: MessageEvent): void {
    try {
      console.log('🎤 WebSocket message received:', event.data)
      
      let data: any
      
      if (event.data instanceof Blob) {
        // Skip binary data for WebSocket - this will be handled by RTVI
        console.log('🎤 Binary message received on WebSocket, skipping (handled by RTVI)')
        return
      } else if (typeof event.data === 'string') {
        data = JSON.parse(event.data)
        console.log('🎤 Parsed WebSocket data:', data)
      } else {
        data = event.data
        console.log('🎤 WebSocket data (non-string):', data)
      }
      
      // Log the message structure for debugging
      if (data.type === 'navigation_result') {
        console.log('🧭 Navigation result message structure:', {
          type: data.type,
          action: data.action,
          hasData: !!data.data,
          dataType: data.data?.Action_type,
          targetPage: data.data?.page,
          interactionType: data.data?.interaction_type
        })
      }
      
      // Handle different message types
      this.processWebSocketMessage(data)
      
    } catch (error) {
      console.error('Error handling WebSocket message:', error)
      this.handleError(new Error('Error processing message from backend'))
    }
  }

  private processWebSocketMessage(data: any): void {
    console.log('🎤 Processing WebSocket message:', data)
    
    // Handle navigation_result type (new format)
    if (data.type === 'navigation_result' && data.data?.Action_type === 'navigation') {
      console.log('🧭 ✅ Navigation result received from backend:', data.data)
      console.log('🧭 📋 Message details:', {
        id: data.id,
        timestamp: data.timestamp,
        toolName: data.toolName,
        action: data.action,
        success: data.success
      })
      
      // Execute the navigation command using the data field
      console.log('🧭 🚀 About to execute navigation command...')
      this.executeNavigationCommand(data.data)
      
      // Notify message handler for logging
      this.onMessageHandler?.({
        type: 'navigation',
        content: `🧭 ${data.data.interaction_type}: ${data.data.page}`,
        navigationData: data.data
      })
      
    } else if (data.Action_type === 'navigation') {
      // Legacy format - Navigation command from backend - execute immediately
      console.log('🧭 Navigation command received from backend (legacy format):', data)
      
      // Execute the navigation command
      this.executeNavigationCommand(data)
      
      // Notify message handler for logging
      this.onMessageHandler?.({
        type: 'navigation',
        content: `🧭 ${data.interaction_type}: ${data.page}`,
        navigationData: data
      })
      
    } else if (data.type === 'system') {
      this.onMessageHandler?.({
        type: 'system',
        message: data.message || 'System message received'
      })
      
    } else if (data.interaction_type) {
      this.onMessageHandler?.({
        type: 'system',
        content: `Voice interaction: ${data.interaction_type}`
      })
      
    } else if (data.type === 'tool_call' || data.type === 'tool_result') {
      this.onMessageHandler?.({
        type: data.type,
        content: data.content || 'Tool interaction'
      })
      
    } else {
      this.onMessageHandler?.({
        type: 'system',
        content: 'Message received from backend'
      })
    }
  }

  sendMessage(message: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('🎤 Cannot send message: WebSocket not connected')
      return
    }

    try {
      const payload = {
        type: 'text_message',
        text: message,
        timestamp: new Date().toISOString()
      }
      
      this.ws.send(JSON.stringify(payload))
      console.log('🎤 Text message sent via WebSocket')
      
    } catch (error) {
      console.error('Error sending message:', error)
      this.handleError(new Error('Failed to send message'))
    }
  }

  private scheduleReconnection(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('🎤 Max reconnection attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000) // Exponential backoff, max 10s
    
    console.log(`🎤 Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`)
    
    this.reconnectTimeout = setTimeout(() => {
      console.log(`🎤 Attempting reconnection ${this.reconnectAttempts}`)
      this.connect()
    }, delay)
  }

  private handleError(error: Error): void {
    console.error('🎤 WebSocket service error:', error)
    this.onError?.(error)
  }

  // Execute navigation commands from backend
  private executeNavigationCommand(data: any): void {
    try {
      console.log('🧭 Executing navigation command from backend:', data)
      
      // Import NavigationService dynamically to avoid circular dependencies
      import('./NavigationService').then(({ NavigationService }) => {
        console.log('🧭 NavigationService imported successfully')
        
        switch (data.interaction_type) {
          case 'page_navigation':
            if (data.page) {
              // Check if we're already on the target page to prevent unnecessary navigation
              const currentPage = NavigationService.getCurrentPage()
              console.log('🧭 Current page from NavigationService:', currentPage)
              console.log('🧭 Target page from backend:', data.page)
              
              if (data.page === currentPage) {
                console.log('🧭 Already on target page:', data.page, '- but executing anyway for backend sync')
                // Execute navigation even if same page to ensure backend state sync
              }
              
              console.log('🧭 Executing page navigation to:', data.page, 'from:', currentPage)
              NavigationService.executeNavigation(data.page)
              
              // Log successful navigation execution
              console.log('🧭 ✅ Navigation command executed successfully')
            } else {
              console.warn('🧭 No page specified in navigation command')
            }
            break
            
          case 'button_click':
            if (data.element_name) {
              console.log('🧭 Executing button click on:', data.element_name)
              NavigationService.executeElementClick(data.element_name)
            }
            break
            
          case 'database_search':
            if (data.search_query) {
              console.log('🧭 Executing database search for:', data.search_query)
              NavigationService.executeSearch(data.search_query, 'database')
            }
            break
            
          case 'file_search':
            if (data.search_query) {
              console.log('🧭 Executing file search for:', data.search_query)
              NavigationService.executeSearch(data.search_query, 'file')
            }
            break
            
          case 'file_upload':
            if (data.file_descriptions && data.table_names) {
              console.log('🧭 Executing file upload:', data.file_descriptions, data.table_names)
              NavigationService.executeFileUpload(data.file_descriptions, data.table_names)
            }
            break
            
          case 'view_report':
            if (data.report_request) {
              console.log('🧭 Executing view report:', data.report_request)
              NavigationService.executeViewReport(data.report_request)
            }
            break
            
          case 'generate_report':
            if (data.report_query) {
              console.log('🧭 Executing generate report:', data.report_query)
              NavigationService.executeGenerateReport(data.report_query)
            }
            break
            
          default:
            console.warn('🧭 Unknown interaction type from backend:', data.interaction_type)
        }
      }).catch(error => {
        console.error('🧭 Failed to import NavigationService:', error)
      })
      
    } catch (error) {
      console.error('🧭 Error executing navigation command:', error)
      this.handleError(error as Error)
    }
  }

  disconnect(): void {
    console.log('🎤 Disconnecting WebSocket...')
    
    // Clear reconnection timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }
    
    // Close WebSocket
    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect')
      this.ws = null
    }
    
    this.reconnectAttempts = 0
    this.onConnectionChange?.(false)
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }

  getConnectionState(): string {
    if (!this.ws) return 'Disconnected'
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'Connecting...'
      case WebSocket.OPEN:
        return 'Connected'
      case WebSocket.CLOSING:
        return 'Disconnecting...'
      case WebSocket.CLOSED:
        return 'Disconnected'
      default:
        return 'Unknown'
    }
  }
} 