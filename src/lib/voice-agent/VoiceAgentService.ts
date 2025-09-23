import { VOICE_AGENT_CONFIG } from "./config";
import { MessageService } from "./services/MessageService";
import { WebSocketService } from "./services/WebSocketService";
import { RTVIService } from "./services/RTVIService";
import { getButtonMapping, BUTTON_MAPPING_CONFIG } from "./config/default-button-actions";
import { NavigationActionManager } from "./services/NavigationActionManager";
// Removed old ButtonRegistrationService - using mapping system only
import {
  VoiceMessage,
  VoiceClientState,
  VoiceClientActions,
  VoiceClientHook,
} from "./types";

// Global context storage keys
const CONTEXT_KEYS = {
  MESSAGES: "voice_agent_messages",
  CONVERSATION_STATE: "voice_agent_conversation_state",
  USER_PREFERENCES: "voice_agent_user_preferences",
  SESSION_DATA: "voice_agent_session_data",
  LAST_PAGE: "voice_agent_last_page",
  CONNECTION_STATE: "voice_agent_connection_state",
} as const;

// Singleton instance
let globalVoiceAgentService: VoiceAgentService | null = null;

export class VoiceAgentService {
  private webSocketService: WebSocketService;
  private rtviService: RTVIService;
  private userId: string;
  private state: VoiceClientState = {
    isConnected: false,
    isInConversation: false,
    connectionStatus: "Disconnected",
    messages: [],
    currentPage: "dashboard",
    previousPage: null,
  };

  // State change callbacks
  private onStateChange?: (state: VoiceClientState) => void;
  private onMessage?: (message: VoiceMessage) => void;

  // Context persistence
  private contextPersistenceEnabled = true;
  private autoReconnectEnabled = true;
  private reconnectInterval: NodeJS.Timeout | null = null;

  constructor(
    userId: string,
    onStateChange?: (state: VoiceClientState) => void,
    onMessage?: (message: VoiceMessage) => void,
  ) {
    this.userId = userId;
    this.onStateChange = onStateChange;
    this.onMessage = onMessage;

    // Initialize services
    this.webSocketService = new WebSocketService(this.userId);

    // Set up WebSocket event handlers
    this.webSocketService.onMessage = this.handleWebSocketMessage.bind(this);
    this.webSocketService.onConnectionChange =
      this.handleWebSocketConnectionChange.bind(this);
    this.webSocketService.onError = this.handleWebSocketError.bind(this);

    this.rtviService = new RTVIService(
      this.userId,
      this.handleRTVIConnected.bind(this),
      this.handleRTVIDisconnected.bind(this),
      this.handleRTVIBotReady.bind(this),
      this.handleRTVIUserTranscript.bind(this),
      this.handleRTVIBotTranscript.bind(this),
      this.handleRTVIError.bind(this),
    );

    // Load persisted context
    this.loadPersistedContext();

    // Initialize page detection
    this.initializePageDetection();

    // Set up auto-reconnection
    this.setupAutoReconnection();

    // Initialize button registration service
    this.initializeButtonRegistration();
  }

  // Singleton pattern
  static getInstance(
    userId: string,
    onStateChange?: (state: VoiceClientState) => void,
    onMessage?: (message: VoiceMessage) => void,
  ): VoiceAgentService {
    if (!globalVoiceAgentService || globalVoiceAgentService.userId !== userId) {
      if (globalVoiceAgentService) {
        globalVoiceAgentService.cleanup();
      }
      globalVoiceAgentService = new VoiceAgentService(
        userId,
        onStateChange,
        onMessage,
      );
    } else {
      // Update callbacks for existing instance
      globalVoiceAgentService.onStateChange = onStateChange;
      globalVoiceAgentService.onMessage = onMessage;
    }
    return globalVoiceAgentService;
  }

  // Initialize button registration service
  private async initializeButtonRegistration(): Promise<void> {
    try {
      console.log("🎤 Initializing button registration service...");
      await buttonRegistrationService.initialize();
      console.log("🎤 Button registration service initialized successfully");
    } catch (error) {
      console.error(
        "🎤 Failed to initialize button registration service:",
        error,
      );
    }
  }

  // Context persistence methods
  private loadPersistedContext(): void {
    if (!this.contextPersistenceEnabled || typeof window === "undefined")
      return;

    try {
      // Load messages
      const persistedMessages = sessionStorage.getItem(CONTEXT_KEYS.MESSAGES);
      if (persistedMessages) {
        const messages = JSON.parse(persistedMessages);
        this.state.messages = messages.filter((msg: VoiceMessage) => {
          // Only keep messages from the last 24 hours
          const msgTime = new Date(msg.timestamp).getTime();
          const now = Date.now();
          return now - msgTime < 24 * 60 * 60 * 1000;
        });
      }

      // Load conversation state
      const conversationState = sessionStorage.getItem(
        CONTEXT_KEYS.CONVERSATION_STATE,
      );
      if (conversationState) {
        const state = JSON.parse(conversationState);
        this.state.isInConversation = state.isInConversation || false;
      }

      // Load last page
      const lastPage = sessionStorage.getItem(CONTEXT_KEYS.LAST_PAGE);
      if (lastPage) {
        this.state.previousPage = lastPage;
      }

      // Load connection state
      const connectionState = sessionStorage.getItem(
        CONTEXT_KEYS.CONNECTION_STATE,
      );
      if (connectionState) {
        const state = JSON.parse(connectionState);
        this.state.connectionStatus = state.connectionStatus || "Disconnected";
        this.state.isConnected = state.isConnected || false;
      }

      console.log("🎤 Loaded persisted context:", {
        messagesCount: this.state.messages.length,
        isInConversation: this.state.isInConversation,
        previousPage: this.state.previousPage,
        connectionStatus: this.state.connectionStatus,
      });
    } catch (error) {
      console.error("🎤 Failed to load persisted context:", error);
    }
  }

  private savePersistedContext(): void {
    if (!this.contextPersistenceEnabled || typeof window === "undefined")
      return;

    try {
      // Save messages (keep last 50 messages to avoid storage limits)
      const messagesToSave = this.state.messages.slice(-50);
      sessionStorage.setItem(
        CONTEXT_KEYS.MESSAGES,
        JSON.stringify(messagesToSave),
      );

      // Save conversation state
      sessionStorage.setItem(
        CONTEXT_KEYS.CONVERSATION_STATE,
        JSON.stringify({
          isInConversation: this.state.isInConversation,
        }),
      );

      // Save current page as previous page for next navigation
      sessionStorage.setItem(CONTEXT_KEYS.LAST_PAGE, this.state.currentPage);

      // Save connection state
      sessionStorage.setItem(
        CONTEXT_KEYS.CONNECTION_STATE,
        JSON.stringify({
          isConnected: this.state.isConnected,
          connectionStatus: this.state.connectionStatus,
        }),
      );
    } catch (error) {
      console.error("🎤 Failed to save persisted context:", error);
    }
  }

  // Auto-reconnection setup
  private setupAutoReconnection(): void {
    if (!this.autoReconnectEnabled) return;

    // Check connection every 30 seconds
    this.reconnectInterval = setInterval(() => {
      if (this.state.isConnected && !this.webSocketService.isConnected()) {
        console.log("🎤 Auto-reconnecting due to lost connection...");
        this.reconnectWithCurrentPage().catch((error) => {
          console.error("🎤 Auto-reconnection failed:", error);
        });
      }
    }, 30000); // 30 seconds
  }

  // Enhanced page detection with context preservation
  private initializePageDetection(): void {
    // Detect initial page using centralized detection
    this.state.currentPage = this.detectCurrentPage();

    // Listen for route changes
    if (typeof window !== "undefined") {
      // Listen for popstate (browser back/forward buttons)
      const handlePopState = () => {
        const newPage = this.detectCurrentPage();
        this.updatePageState(newPage, "browser_navigation");
      };

      // Listen for hash changes
      const handleHashChange = () => {
        const newPage = this.detectCurrentPage();
        this.updatePageState(newPage, "hash_change");
      };

      // Listen for SPA voice-navigation events
      const handleVoiceNavigation = (event: Event) => {
        try {
          const detail = (event as CustomEvent).detail;
          if (detail?.page) {
            this.updatePageState(detail.page, "voice_navigation");
          }
        } catch (_) {}
      };

      // Listen for beforeunload to capture page changes and save context
      const handleBeforeUnload = () => {
        // Save current context before page unload
        this.savePersistedContext();
        sessionStorage.setItem("voice_agent_last_page", this.state.currentPage);
      };

      // Listen for load to detect page changes and restore context
      const handleLoad = () => {
        const newPage = this.detectCurrentPage();
        const lastPage = sessionStorage.getItem("voice_agent_last_page");

        if (lastPage && lastPage !== newPage) {
          this.updatePageState(newPage, "page_load");
        }

        // Restore connection if it was previously connected
        this.restoreConnectionIfNeeded();
      };

      // Add event listeners
      window.addEventListener("popstate", handlePopState);
      window.addEventListener("hashchange", handleHashChange);
      window.addEventListener(
        "voice-navigation",
        handleVoiceNavigation as EventListener,
      );
      window.addEventListener("beforeunload", handleBeforeUnload);
      window.addEventListener("load", handleLoad);

      // Cleanup will be handled by the hook
    }
  }

  // Restore connection if it was previously connected
  private async restoreConnectionIfNeeded(): Promise<void> {
    const connectionState = sessionStorage.getItem(
      CONTEXT_KEYS.CONNECTION_STATE,
    );
    if (connectionState) {
      try {
        const state = JSON.parse(connectionState);
        if (state.isConnected && !this.state.isConnected) {
          console.log("🎤 Restoring previous connection...");
          await this.connect();
        }
      } catch (error) {
        console.error("🎤 Failed to restore connection:", error);
      }
    }
  }

  // Centralized page detection
  private detectCurrentPage(): string {
    if (typeof window !== "undefined") {
      const path = window.location.pathname;
      const page = path === "/" ? "dashboard" : path.substring(1);
      return page;
    }
    return "dashboard";
  }

  // Enhanced page state update with context preservation
  private updatePageState(newPage: string, source: string = "unknown"): void {
    if (newPage !== this.state.currentPage) {
      // Save context before page change
      this.savePersistedContext();

      const previousPage = this.state.currentPage;
      this.state.previousPage = previousPage;
      this.state.currentPage = newPage;

      // Update NavigationActionManager state to keep it in sync
      NavigationActionManager.setCurrentPage(newPage);

      // Page change is handled by ButtonRegistry through backend responses

      // Notify backend of page change without reconnecting
      try {
        this.webSocketService.notifyPageChange(newPage);
      } catch (err) {
        console.warn("🎤 Failed to notify backend of page change:", err);
      }

      // Notify state change
      this.notifyStateChange();

      // Save context after page change
      this.savePersistedContext();
    }
  }

  // Determine if a page change warrants reconnecting the voice agent
  private shouldReconnectForPageChange(
    previousPage: string | null,
    newPage: string,
  ): boolean {
    if (!previousPage || !this.state.isConnected) return false;

    // Major page changes that should trigger reconnection
    const majorPages = [
      "dashboard",
      "database-query",
      "file-query",
      "ai-reports",
      "auth",
      "users",
    ];

    const isPreviousMajor = majorPages.includes(previousPage);
    const isNewMajor = majorPages.includes(newPage);

    // Reconnect if moving between major pages
    return isPreviousMajor && isNewMajor && previousPage !== newPage;
  }

  private notifyStateChange(): void {
    this.onStateChange?.({ ...this.state });
    // Save context whenever state changes
    this.savePersistedContext();
  }

  private addMessage(message: Omit<VoiceMessage, "id" | "timestamp">): void {
    const newMessage = MessageService.createMessage(
      message.type,
      message.content,
      message,
    );

    this.state.messages = [...this.state.messages, newMessage];
    this.onMessage?.(newMessage);
    this.notifyStateChange();
  }

  // WebSocket event handlers
  private handleWebSocketMessage(message: VoiceMessage): void {
    this.addMessage(message);
  }

  private handleWebSocketConnectionChange(isConnected: boolean): void {
    this.state.isConnected = isConnected;
    this.state.connectionStatus = isConnected ? "Connected" : "Disconnected";
    this.notifyStateChange();
  }

  private handleWebSocketError(error: Error): void {
    this.addMessage(
      MessageService.createErrorMessage(`WebSocket error: ${error.message}`),
    );
    this.state.connectionStatus = "Error";
    this.notifyStateChange();
  }

  // RTVI event handlers
  private handleRTVIConnected(): void {
    this.state.isConnected = true;
    this.state.connectionStatus = "Connected";
    this.addMessage(
      MessageService.createSystemMessage(
        "Connected to Easy Query Voice Agent - Audio ready!",
      ),
    );
    this.notifyStateChange();
  }

  private handleRTVIDisconnected(): void {
    this.state.isConnected = false;
    this.state.isInConversation = false;
    this.state.connectionStatus = "Disconnected";
    this.addMessage(
      MessageService.createSystemMessage("Disconnected from voice agent"),
    );
    this.notifyStateChange();
  }

  private handleRTVIBotReady(data: any): void {
    this.addMessage(MessageService.createSystemMessage("AI is ready to chat!"));
  }

  private handleRTVIUserTranscript(data: any): void {
    if (data.text && data.text.trim()) {
      this.addMessage(MessageService.createUserMessage(data.text, true));
    }
  }

  private handleRTVIBotTranscript(data: any): void {
    if (data.text && data.text.trim()) {
      this.addMessage(MessageService.createAssistantMessage(data.text));

      // Audio processing only - navigation commands come from backend via WebSocket
      this.rtviService.processBotTranscript(data.text);
    }
  }

  private handleRTVIError(error: Error): void {
    this.addMessage(MessageService.createErrorMessage(error.message));
  }

  private handleButtonActionMessage(data: any): void {
    if (data.result && data.result.element_name) {
      const elementName = data.result.element_name;
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
        report_query: data.result.report_query,
      };

      // Execute the button action
      this.executeButtonAction(elementName, context);

      // Add a message to show the action was executed
      this.addMessage(
        MessageService.createSystemMessage(
          `Voice agent executed: ${elementName}${data.result.search_query ? ` - ${data.result.search_query}` : ""}`,
        ),
      );
    }
  }

  // Public API methods
  async connect(): Promise<void> {
    if (!this.userId || this.userId === "frontend_user") {
      throw new Error("Cannot connect: Invalid or missing user ID");
    }

    try {
      this.state.connectionStatus = "Connecting...";
      this.notifyStateChange();

      // Connect WebSocket first with current page and user ID
      await this.webSocketService.connect(this.state.currentPage, this.userId);

      // Then connect RTVI for audio with current page
      try {
        await this.rtviService.initialize(this.state.currentPage);
        await this.rtviService.connect(this.state.currentPage);
      } catch (rtviError) {
        this.state.connectionStatus = "Audio Failed - Tools Available";
        this.notifyStateChange();
      }
    } catch (error) {
      this.state.connectionStatus = "Connection Failed";
      this.addMessage(
        MessageService.createErrorMessage(
          `Failed to connect: ${error instanceof Error ? error.message : "Unknown error"}`,
        ),
      );
      this.notifyStateChange();
      throw error;
    }
  }

  async reconnectWithCurrentPage(): Promise<void> {
    if (!this.state.isConnected) {
      return;
    }

    try {
      // Disconnect current connections
      await this.disconnect();

      // Reconnect with current page
      await this.connect();
    } catch (error) {
      this.addMessage(
        MessageService.createErrorMessage(
          `Failed to reconnect: ${error instanceof Error ? error.message : "Unknown error"}`,
        ),
      );
    }
  }

  async disconnect(): Promise<void> {
    // Disconnect RTVI
    await this.rtviService.disconnect();

    // Disconnect WebSocket
    this.webSocketService.disconnect();

    // Reset state
    this.state.isConnected = false;
    this.state.isInConversation = false;
    this.state.connectionStatus = "Disconnected";
    this.notifyStateChange();
  }

  async startConversation(): Promise<void> {
    if (!this.state.isConnected) {
      throw new Error("Cannot start conversation: not connected");
    }

    this.state.isInConversation = true;
    this.addMessage(
      MessageService.createSystemMessage(
        "Voice conversation started! Speak naturally.",
      ),
    );
    this.notifyStateChange();
  }

  stopConversation(): void {
    this.state.isInConversation = false;
    this.addMessage(MessageService.createSystemMessage("Conversation paused"));
    this.notifyStateChange();
  }

  clearMessages(): void {
    this.state.messages = [];
    this.notifyStateChange();
  }

  sendMessage(message: string): void {
    if (!this.state.isConnected) {
      return;
    }

    this.webSocketService.sendMessage(message);
    this.addMessage(MessageService.createUserMessage(message));
  }

  // Navigation methods
  navigateToPage(page: string): void {
    // Use NavigationActionManager for navigation actions
    NavigationActionManager.executeNavigationAction(`go to ${page}`);
  }

  // Debug method to refresh page state
  refreshPageState(): void {
    const detectedPage = this.detectCurrentPage();
    this.updatePageState(detectedPage, "manual_refresh");
  }

  // Get current page state for debugging
  getCurrentPageState(): { currentPage: string; previousPage: string | null } {
    return {
      currentPage: this.state.currentPage,
      previousPage: this.state.previousPage,
    };
  }

  clickElement(elementName: string): void {
    // Use ButtonRegistry for button actions
    this.executeButtonAction(elementName);
  }

  async executeButtonAction(elementName: string, context?: any): Promise<void> {
    try {
      console.log(`🎯 VoiceAgent executing button action: ${elementName}`);
      
      const mapping = getButtonMapping(elementName);
      
      if (!mapping) {
        console.error(`🎯 No mapping found for element: ${elementName}`);
        console.error(`🎯 Available mappings:`, Object.keys(BUTTON_MAPPING_CONFIG));
        return;
      }
      
      // Execute using mapping system
      const success = await this.executeMappedButtonAction(mapping, context);
      
      if (success) {
        console.log(`🎯 Successfully executed button action: ${elementName}`);
      } else {
        console.warn(`🎯 Failed to execute button action: ${elementName}`);
      }
    } catch (error) {
      console.error(`🎯 Error executing button action: ${elementName}`, error);
    }
  }

  /**
   * Execute button action using mapping configuration
   */
  private async executeMappedButtonAction(mapping: any, context?: any): Promise<boolean> {
    console.log(`🎯 Executing mapped button action: ${mapping.name}`);
    
    // Handle workflow: fill textarea first if needed
    if (mapping.workflow?.fillTextarea && context?.search_query) {
      const textareaFilled = await this.fillTextarea(mapping, context.search_query);
      if (!textareaFilled) {
        console.warn(`🎯 Failed to fill textarea for: ${mapping.name}`);
        return false;
      }
      
      // Small delay to allow React to process the state change
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Try each selector in order
    for (let i = 0; i < mapping.selectors.length; i++) {
      const selector = mapping.selectors[i];
      console.log(`🎯 Trying selector ${i + 1}/${mapping.selectors.length}: ${selector}`);
      
      try {
        const button = document.querySelector(selector) as HTMLElement;
        
        if (button) {
          // Validate button if validation rules exist
          if (mapping.validation && !this.validateButton(button, mapping.validation)) {
            console.warn(`🎯 Button found but validation failed for selector: ${selector}`);
            continue;
          }
          
          // Click the button
          button.click();
          console.log(`✅ Successfully clicked button: ${mapping.name}`);
          return true;
        }
      } catch (error) {
        console.warn(`🎯 Error with selector ${selector}:`, error);
      }
    }
    
    // If all selectors fail, try fuzzy matching
    console.warn(`🎯 All selectors failed for: ${mapping.name}`);
    return await this.tryFuzzyMatching(mapping);
  }

  /**
   * Fill textarea with search query
   */
  private async fillTextarea(mapping: any, searchQuery: string): Promise<boolean> {
    if (!mapping.workflow?.textareaSelectors) {
      console.warn(`🎯 No textarea selectors defined for: ${mapping.name}`);
      return false;
    }
    
    console.log(`🎯 Filling textarea with query: "${searchQuery}"`);
    
    // Try each textarea selector
    for (let i = 0; i < mapping.workflow.textareaSelectors.length; i++) {
      const selector = mapping.workflow.textareaSelectors[i];
      console.log(`🎯 Trying textarea selector ${i + 1}/${mapping.workflow.textareaSelectors.length}: ${selector}`);
      
      try {
        const textarea = document.querySelector(selector) as HTMLTextAreaElement;
        
        if (textarea) {
          console.log(`🎯 Textarea found with selector: ${selector}`);
          
          // Set the value using React's synthetic event system
          const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
          if (nativeInputValueSetter) {
            nativeInputValueSetter.call(textarea, searchQuery);
          }
          
          // Create and dispatch a proper React synthetic event
          const inputEvent = new Event('input', { bubbles: true });
          Object.defineProperty(inputEvent, 'target', {
            writable: false,
            value: textarea
          });
          Object.defineProperty(inputEvent, 'currentTarget', {
            writable: false,
            value: textarea
          });
          
          // Dispatch the event
          textarea.dispatchEvent(inputEvent);
          
          console.log(`✅ Successfully filled textarea with: "${searchQuery}"`);
          return true;
        }
      } catch (error) {
        console.warn(`🎯 Error with textarea selector ${selector}:`, error);
      }
    }
    
    console.warn(`🎯 No textarea found for: ${mapping.name}`);
    return false;
  }

  /**
   * Validate button matches expected criteria
   */
  private validateButton(button: HTMLElement, validation: any): boolean {
    if (!validation) return true;
    
    // Check expected text
    if (validation.expectedText) {
      const buttonText = button.textContent?.trim();
      if (!buttonText?.includes(validation.expectedText)) {
        console.warn(`🎯 Text validation failed. Expected: ${validation.expectedText}, Got: ${buttonText}`);
        return false;
      }
    }
    
    // Check expected class
    if (validation.expectedClass) {
      if (!button.className.includes(validation.expectedClass)) {
        console.warn(`🎯 Class validation failed. Expected: ${validation.expectedClass}, Got: ${button.className}`);
        return false;
      }
    }
    
    // Check if button is visible and clickable
    if (!button.offsetParent) {
      console.warn(`🎯 Button is not visible`);
      return false;
    }
    
    return true;
  }

  /**
   * Fallback fuzzy matching when all selectors fail
   */
  private async tryFuzzyMatching(mapping: any): Promise<boolean> {
    console.log(`🎯 Attempting fuzzy matching for: ${mapping.name}`);
    
    const allButtons = Array.from(document.querySelectorAll('button')) as HTMLElement[];
    console.log(`🎯 Found ${allButtons.length} buttons on page`);
    
    // Try to find button by partial text match
    if (mapping.validation?.expectedText) {
      const expectedText = mapping.validation.expectedText.toLowerCase();
      const matchingButton = allButtons.find(btn => {
        const buttonText = btn.textContent?.toLowerCase().trim();
        return buttonText && (
          buttonText.includes(expectedText) || 
          expectedText.includes(buttonText) ||
          buttonText.split(' ').some(word => expectedText.includes(word))
        );
      });
      
      if (matchingButton) {
        console.log(`🎯 Found button via fuzzy matching: ${matchingButton.textContent?.trim()}`);
        matchingButton.click();
        console.log(`✅ Successfully clicked button via fuzzy matching`);
        return true;
      }
    }
    
    console.error(`🎯 Failed to find button: ${mapping.name}`);
    return false;
  }

  handleFileUpload(descriptions: string[], tableNames: string[]): void {
    // Use ButtonRegistry for file upload actions
    this.executeButtonAction("upload", {
      descriptions,
      tableNames,
    });
  }

  viewReport(request: string): void {
    // Use ButtonRegistry for view report actions
    this.executeButtonAction("view report", { request });
  }

  generateReport(query: string): void {
    // Use ButtonRegistry for generate report actions
    this.executeButtonAction("report generation", { query });
  }

  testNavigation(page: string): void {
    this.navigateToPage(page);
  }

  // Context management methods
  getContext(): VoiceClientState {
    return { ...this.state };
  }

  setContext(context: Partial<VoiceClientState>): void {
    this.state = { ...this.state, ...context };
    this.notifyStateChange();
  }

  clearContext(): void {
    if (typeof window !== "undefined") {
      Object.values(CONTEXT_KEYS).forEach((key) => {
        sessionStorage.removeItem(key);
      });
    }
    this.state.messages = [];
    this.state.isInConversation = false;
    this.notifyStateChange();
  }

  // Getters
  getState(): VoiceClientState {
    return { ...this.state };
  }

  // Cleanup
  cleanup(): void {
    // Clear auto-reconnection interval
    if (this.reconnectInterval) {
      clearInterval(this.reconnectInterval);
      this.reconnectInterval = null;
    }

    // Save context before cleanup
    this.savePersistedContext();

    // Disconnect services
    this.disconnect();
    this.rtviService.cleanup();

    // Clear global instance if this is the global instance
    if (globalVoiceAgentService === this) {
      globalVoiceAgentService = null;
    }
  }
}
