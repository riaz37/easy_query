import { VOICE_AGENT_CONFIG } from './config'
import { MessageService } from './services/MessageService'
import { WebSocketService } from './services/WebSocketService'
import { RTVIService } from './services/RTVIService'
import { ButtonActionService } from './services/ButtonActionService'
import { NavigationActionManager } from './services/NavigationActionManager'
import { VoiceMessage, VoiceClientState, VoiceClientActions, VoiceClientHook } from './types'

export class VoiceAgentService {
  private webSocketService: WebSocketService
  private rtviService: RTVIService
  private userId: string
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
    userId: string,
    onStateChange?: (state: VoiceClientState) => void,
    onMessage?: (message: VoiceMessage) => void
  ) {
    this.userId = userId
    this.onStateChange = onStateChange
    this.onMessage = onMessage

    // Initialize services
    this.webSocketService = new WebSocketService(
      this.userId
    )

    // Set up WebSocket event handlers
    this.webSocketService.onMessage = this.handleWebSocketMessage.bind(this)
    this.webSocketService.onConnectionChange = this.handleWebSocketConnectionChange.bind(this)
    this.webSocketService.onError = this.handleWebSocketError.bind(this)

    this.rtviService = new RTVIService(
      this.userId,
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
      return page
    }
    return 'dashboard'
  }

  // Centralized page state update
  private updatePageState(newPage: string, source: string = 'unknown'): void {
    if (newPage !== this.state.currentPage) {
      this.state.previousPage = this.state.currentPage
      this.state.currentPage = newPage
      
      // Update NavigationActionManager state to keep it in sync
      NavigationActionManager.setCurrentPage(newPage)
      
      // Update ButtonActionService state to keep it in sync
      ButtonActionService.setCurrentPage(newPage)
      
      // Check if we should reconnect with the new page
      // Only reconnect for major page changes, not for sub-navigation
      if (this.shouldReconnectForPageChange(this.state.previousPage, newPage)) {
        // Reconnect asynchronously to avoid blocking the UI
        setTimeout(() => {
          this.reconnectWithCurrentPage().catch(error => {
            console.error('🧭 Failed to reconnect after page change:', error)
          })
        }, 1000) // Small delay to ensure page is fully loaded
      }
      
      // Notify state change
      this.notifyStateChange()
    }
  }

  // Determine if a page change warrants reconnecting the voice agent
  private shouldReconnectForPageChange(previousPage: string | null, newPage: string): boolean {
    if (!previousPage || !this.state.isConnected) return false
    
    // Major page changes that should trigger reconnection
    const majorPages = ['dashboard', 'database-query', 'file-query', 'ai-results', 'auth', 'users']
    
    const isPreviousMajor = majorPages.includes(previousPage)
    const isNewMajor = majorPages.includes(newPage)
    
    // Reconnect if moving between major pages
    return isPreviousMajor && isNewMajor && previousPage !== newPage
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
  private handleWebSocketMessage(message: VoiceMessage): void {
    this.addMessage(message)
  }

  private handleWebSocketConnectionChange(isConnected: boolean): void {
    this.state.isConnected = isConnected
    this.state.connectionStatus = isConnected ? 'Connected' : 'Disconnected'
    this.notifyStateChange()
  }

  private handleWebSocketError(error: Error): void {
    this.addMessage(MessageService.createErrorMessage(`WebSocket error: ${error.message}`))
    this.state.connectionStatus = 'Error'
    this.notifyStateChange()
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

  private handleButtonActionMessage(data: any): void {
    
    if (data.result && data.result.element_name) {
      const elementName = data.result.element_name
      const context = {
        page: data.result.page,
        previousPage: data.result.previous_page,
        timestamp: data.result.timestamp,
        user_id: data.result.user_id,
        success: data.result.success,
        // Add action-specific context
        search_query: data.result.search_query,
        upload_request: data.result.upload_request,
        report_request: data.result.report_request,
        report_query: data.result.report_query
      }
      
      // Execute the button action
      this.executeButtonAction(elementName, context)
      
      // Add a message to show the action was executed
      this.addMessage(MessageService.createSystemMessage(
        `Voice agent executed: ${elementName}${data.result.search_query ? ` - ${data.result.search_query}` : ''}`
      ))
    }
  }

  // Public API methods
  async connect(): Promise<void> {
    if (!this.userId || this.userId === 'frontend_user') {
      throw new Error('Cannot connect: Invalid or missing user ID')
    }
    
    try {
      this.state.connectionStatus = 'Connecting...'
      this.notifyStateChange()

      // Connect WebSocket first with current page and user ID
      await this.webSocketService.connect(this.state.currentPage, this.userId)
      
      // Then connect RTVI for audio with current page
      try {
        await this.rtviService.initialize(this.state.currentPage)
        await this.rtviService.connect(this.state.currentPage)
      } catch (rtviError) {
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

  async reconnectWithCurrentPage(): Promise<void> {
    if (!this.state.isConnected) {
      return
    }

    try {
      // Disconnect current connections
      await this.disconnect()
      
      // Reconnect with current page
      await this.connect()
      
    } catch (error) {
      this.addMessage(MessageService.createErrorMessage(
        `Failed to reconnect: ${error instanceof Error ? error.message : 'Unknown error'}`
      ))
    }
  }

  async disconnect(): Promise<void> {
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
      return
    }

    this.webSocketService.sendMessage(message)
    this.addMessage(MessageService.createUserMessage(message))
  }

  // Navigation methods
  navigateToPage(page: string): void {
    // Use NavigationActionManager for navigation actions
    NavigationActionManager.executeNavigationAction(`go to ${page}`)
  }

  // Debug method to refresh page state
  refreshPageState(): void {
    const detectedPage = this.detectCurrentPage()
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
    // Use ButtonActionService for specific button actions
    ButtonActionService.executeButtonAction(elementName)
  }

  executeButtonAction(elementName: string, context?: any): void {
    ButtonActionService.executeButtonAction(elementName, context)
  }

  handleFileUpload(descriptions: string[], tableNames: string[]): void {
    // Use ButtonActionService for file upload actions
    ButtonActionService.executeButtonAction('upload', { descriptions, tableNames })
  }

  viewReport(request: string): void {
    // Use ButtonActionService for view report actions
    ButtonActionService.executeButtonAction('view report', { request })
  }

  generateReport(query: string): void {
    // Use ButtonActionService for generate report actions
    ButtonActionService.executeButtonAction('report generation', { query })
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