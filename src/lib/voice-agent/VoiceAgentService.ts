import { VOICE_AGENT_CONFIG } from './config'
import { MessageService } from './services/MessageService'
import { NavigationService } from './services/NavigationService'
import { WebSocketService } from './services/WebSocketService'
import { RTVIService } from './services/RTVIService'
import { VoiceMessage, VoiceClientState, VoiceClientActions, VoiceClientHook } from './types'

export class VoiceAgentService {
  private webSocketService: WebSocketService
  private rtviService: RTVIService
  private state: VoiceClientState = {
    isConnected: false,
    isInConversation: false,
    connectionStatus: 'Disconnected',
    messages: [],
    currentPage: 'dashboard',
    previousPage: null
  }

  // State change callbacks
  private onStateChange?: (state: VoiceClientState) => void
  private onMessage?: (message: VoiceMessage) => void

  constructor(
    onStateChange?: (state: VoiceClientState) => void,
    onMessage?: (message: VoiceMessage) => void
  ) {
    this.onStateChange = onStateChange
    this.onMessage = onMessage

    // Initialize services
    this.webSocketService = new WebSocketService(
      this.handleWebSocketMessage.bind(this),
      this.handleWebSocketConnectionChange.bind(this),
      this.handleWebSocketError.bind(this)
    )

    this.rtviService = new RTVIService(
      this.handleRTVIConnected.bind(this),
      this.handleRTVIDisconnected.bind(this),
      this.handleRTVIBotReady.bind(this),
      this.handleRTVIUserTranscript.bind(this),
      this.handleRTVIBotTranscript.bind(this),
      this.handleRTVIError.bind(this)
    )

    // Initialize page detection
    this.initializePageDetection()
  }

  private initializePageDetection(): void {
    // Detect initial page using centralized detection
    this.state.currentPage = this.detectCurrentPage()
    
    // Listen for route changes
    if (typeof window !== 'undefined') {
      // Listen for popstate (browser back/forward buttons)
      const handlePopState = () => {
        const newPage = this.detectCurrentPage()
        this.updatePageState(newPage, 'browser_navigation')
      }
      
      // Listen for hash changes
      const handleHashChange = () => {
        const newPage = this.detectCurrentPage()
        this.updatePageState(newPage, 'hash_change')
      }
      
      // Listen for beforeunload to capture page changes
      const handleBeforeUnload = () => {
        // Store current page before unload
        sessionStorage.setItem('voice_agent_last_page', this.state.currentPage)
      }
      
      // Listen for load to detect page changes
      const handleLoad = () => {
        const newPage = this.detectCurrentPage()
        const lastPage = sessionStorage.getItem('voice_agent_last_page')
        
        if (lastPage && lastPage !== newPage) {
          this.updatePageState(newPage, 'page_load')
        }
      }
      
      // Add event listeners
      window.addEventListener('popstate', handlePopState)
      window.addEventListener('hashchange', handleHashChange)
      window.addEventListener('beforeunload', handleBeforeUnload)
      window.addEventListener('load', handleLoad)
      
      // Cleanup will be handled by the hook
    }
  }

  // Centralized page detection
  private detectCurrentPage(): string {
    if (typeof window !== 'undefined') {
      const path = window.location.pathname
      const page = path === '/' ? 'dashboard' : path.substring(1)
      console.log('🧭 Detected current page from URL:', page)
      return page
    }
    return 'dashboard'
  }

  // Centralized page state update
  private updatePageState(newPage: string, source: string = 'unknown'): void {
    if (newPage !== this.state.currentPage) {
      this.state.previousPage = this.state.currentPage
      this.state.currentPage = newPage
      
      console.log(`🧭 Page changed from ${this.state.previousPage} to ${this.state.currentPage} (source: ${source})`)
      
      // Update NavigationService state to keep it in sync
      NavigationService.updatePage(newPage)
      
      // Notify state change
      this.notifyStateChange()
    }
  }

  private notifyStateChange(): void {
    this.onStateChange?.({ ...this.state })
  }

  private addMessage(message: Omit<VoiceMessage, 'id' | 'timestamp'>): void {
    const newMessage = MessageService.createMessage(
      message.type,
      message.content,
      message
    )
    
    this.state.messages = [...this.state.messages, newMessage]
    this.onMessage?.(newMessage)
    this.notifyStateChange()
  }

  // WebSocket event handlers
  private handleWebSocketMessage(data: any): void {
    if (data.type === 'system') {
      this.addMessage(MessageService.createSystemMessage(data.message || 'System message received'))
    } else if (data.type === 'navigation') {
      this.addMessage(MessageService.createNavigationMessage(data.content, data.navigationData))
    } else if (data.type === 'tool_call' || data.type === 'tool_result') {
      this.addMessage(MessageService.createToolMessage(data.type, data.content || 'Tool interaction'))
    }
  }

  private handleWebSocketConnectionChange(isConnected: boolean): void {
    this.state.connectionStatus = this.webSocketService.getConnectionState()
    this.notifyStateChange()
  }

  private handleWebSocketError(error: Error): void {
    this.addMessage(MessageService.createErrorMessage(error.message))
  }

  // RTVI event handlers
  private handleRTVIConnected(): void {
    this.state.isConnected = true
    this.state.connectionStatus = 'Connected'
    this.addMessage(MessageService.createSystemMessage('Connected to ESAP Voice Agent - Audio ready!'))
    this.notifyStateChange()
  }

  private handleRTVIDisconnected(): void {
    this.state.isConnected = false
    this.state.isInConversation = false
    this.state.connectionStatus = 'Disconnected'
    this.addMessage(MessageService.createSystemMessage('Disconnected from voice agent'))
    this.notifyStateChange()
  }

  private handleRTVIBotReady(data: any): void {
    this.addMessage(MessageService.createSystemMessage('AI is ready to chat!'))
  }

  private handleRTVIUserTranscript(data: any): void {
    if (data.text && data.text.trim()) {
      this.addMessage(MessageService.createUserMessage(data.text, true))
    }
  }

  private handleRTVIBotTranscript(data: any): void {
    if (data.text && data.text.trim()) {
      this.addMessage(MessageService.createAssistantMessage(data.text))
      
      // Audio processing only - navigation commands come from backend via WebSocket
      this.rtviService.processBotTranscript(data.text)
    }
  }

  private handleRTVIError(error: Error): void {
    this.addMessage(MessageService.createErrorMessage(error.message))
  }

  // Public API methods
  async connect(): Promise<void> {
    try {
      this.state.connectionStatus = 'Connecting...'
      this.notifyStateChange()

      // Connect WebSocket first
      await this.webSocketService.connect()
      
      // Then connect RTVI for audio
      try {
        await this.rtviService.initialize()
        await this.rtviService.connect()
      } catch (rtviError) {
        console.error('RTVI connection failed, but WebSocket is available:', rtviError)
        this.state.connectionStatus = 'Audio Failed - Tools Available'
        this.notifyStateChange()
      }
      
    } catch (error) {
      this.state.connectionStatus = 'Connection Failed'
      this.addMessage(MessageService.createErrorMessage(
        `Failed to connect: ${error instanceof Error ? error.message : 'Unknown error'}`
      ))
      this.notifyStateChange()
      throw error
    }
  }

  async disconnect(): Promise<void> {
    console.log('🎤 Disconnecting voice agent...')
    
    // Disconnect RTVI
    await this.rtviService.disconnect()
    
    // Disconnect WebSocket
    this.webSocketService.disconnect()
    
    // Reset state
    this.state.isConnected = false
    this.state.isInConversation = false
    this.state.connectionStatus = 'Disconnected'
    this.notifyStateChange()
  }

  async startConversation(): Promise<void> {
    if (!this.state.isConnected) {
      throw new Error('Cannot start conversation: not connected')
    }

    this.state.isInConversation = true
    this.addMessage(MessageService.createSystemMessage('Voice conversation started! Speak naturally.'))
    this.notifyStateChange()
  }

  stopConversation(): void {
    this.state.isInConversation = false
    this.addMessage(MessageService.createSystemMessage('Conversation paused'))
    this.notifyStateChange()
  }

  clearMessages(): void {
    this.state.messages = []
    this.notifyStateChange()
  }

  sendMessage(message: string): void {
    if (!this.state.isConnected) {
      console.warn('Cannot send message: not connected')
      return
    }

    this.webSocketService.sendMessage(message)
    this.addMessage(MessageService.createUserMessage(message))
  }

  // Navigation methods
  navigateToPage(page: string): void {
    NavigationService.executeNavigation(page)
  }

  // Debug method to refresh page state
  refreshPageState(): void {
    const detectedPage = this.detectCurrentPage()
    console.log('🧭 Refreshing page state. Detected:', detectedPage, 'Current:', this.state.currentPage)
    this.updatePageState(detectedPage, 'manual_refresh')
  }

  // Get current page state for debugging
  getCurrentPageState(): { currentPage: string; previousPage: string | null } {
    return {
      currentPage: this.state.currentPage,
      previousPage: this.state.previousPage
    }
  }

  clickElement(elementName: string): void {
    NavigationService.executeElementClick(elementName)
  }

  executeSearch(query: string, type: 'database' | 'file'): void {
    NavigationService.executeSearch(query, type)
  }

  handleFileUpload(descriptions: string[], tableNames: string[]): void {
    NavigationService.executeFileUpload(descriptions, tableNames)
  }

  viewReport(request: string): void {
    NavigationService.executeViewReport(request)
  }

  generateReport(query: string): void {
    NavigationService.executeGenerateReport(query)
  }

  testNavigation(page: string): void {
    this.navigateToPage(page)
  }

  // Getters
  getState(): VoiceClientState {
    return { ...this.state }
  }

  // Cleanup
  cleanup(): void {
    this.disconnect()
    this.rtviService.cleanup()
  }
} 