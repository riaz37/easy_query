import { VOICE_AGENT_CONFIG } from "../config";
// Removed old ButtonRegistry - using mapping system only
// Removed old ButtonRegistrationService - using mapping system only
import { getButtonMapping, ButtonMappingConfig, BUTTON_MAPPING_CONFIG } from "../config/default-button-actions";

export interface TextMessage {
  id: string;
  type: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  user_id?: string;
}

export interface TextConversationState {
  isConnected: boolean;
  connectionStatus: string;
  messages: TextMessage[];
  isTyping: boolean;
}

export class TextConversationService {
  private textWs: WebSocket | null = null;
  private toolsWs: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private currentPage = "voice-control";
  private userId: string;
  private messageIdCounter = 0;
  private autoReconnectEnabled = true;
  private reconnectInterval: NodeJS.Timeout | null = null;

  // Event handlers
  onStateChange?: (state: TextConversationState) => void;
  onMessage?: (message: TextMessage) => void;
  onError?: (error: Error) => void;

  // Internal state
  private state: TextConversationState = {
    isConnected: false,
    connectionStatus: "Disconnected",
    messages: [],
    isTyping: false,
  };

  constructor(userId: string) {
    this.userId = userId;

    // Initialize page detection and persistence
    this.initializePageDetection();
    this.setupAutoReconnection();
    
    // Restore connection if it was previously connected
    this.restoreConnectionIfNeeded();
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const baseUrl = VOICE_AGENT_CONFIG.BACKEND_URL;
        const wsBaseUrl = baseUrl.replace(
          /^https?:\/\//,
          baseUrl.startsWith("https") ? "wss://" : "ws://",
        );

        // Connect to text conversation endpoint for AI responses
        const textWsUrl = `${wsBaseUrl}/voice/ws/text-conversation?user_id=${this.userId}&current_page=${this.currentPage}`;
        console.log("🔗 Connecting to text conversation WebSocket:", textWsUrl);

        // Connect to tools endpoint for navigation results
        const toolsWsUrl = `${wsBaseUrl}/voice/ws/tools?user_id=${this.userId}&current_page=${this.currentPage}`;
        console.log("🔗 Connecting to tools WebSocket:", toolsWsUrl);

        let textConnected = false;
        let toolsConnected = false;

        // Connect to text conversation WebSocket
        this.textWs = new WebSocket(textWsUrl);

        this.textWs.onopen = () => {
          console.log("💬 Text conversation WebSocket connected");
          textConnected = true;
          if (toolsConnected) {
            this.onBothConnected(resolve);
          }
        };

        this.textWs.onmessage = (event) => {
          try {
            let data;

            // Try to parse as JSON first
            try {
              data = JSON.parse(event.data);
            } catch (jsonError) {
              // If JSON parsing fails, treat as plain text message
              console.log("💬 Received plain text message:", event.data);
              data = {
                type: "message",
                content: event.data,
                timestamp: new Date().toISOString(),
              };
            }

            this.handleTextMessage(data);
          } catch (error) {
            console.error("💬 Error handling text message:", error);
            this.handleError(error as Error);
          }
        };

        this.textWs.onclose = (event) => {
          console.log(
            "💬 Text conversation WebSocket disconnected:",
            event.code,
            event.reason,
          );
          this.updateState({
            isConnected: false,
            connectionStatus: "Disconnected",
            isTyping: false,
          });

          if (event.code !== 1000) {
            this.handleReconnection();
          }
        };

        this.textWs.onerror = (error) => {
          console.error("💬 Text conversation WebSocket error:", error);
          this.handleError(new Error("Text WebSocket connection error"));
          reject(new Error("Text WebSocket connection error"));
        };

        // Connect to tools WebSocket
        this.toolsWs = new WebSocket(toolsWsUrl);

        this.toolsWs.onopen = () => {
          console.log("💬 Tools WebSocket connected");
          toolsConnected = true;
          if (textConnected) {
            this.onBothConnected(resolve);
          }
        };

        this.toolsWs.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log("💬 Tools message received:", data);
            this.handleToolsMessage(data);
          } catch (error) {
            console.error("💬 Error handling tools message:", error);
            this.handleError(error as Error);
          }
        };

        this.toolsWs.onclose = (event) => {
          console.log(
            "💬 Tools WebSocket disconnected:",
            event.code,
            event.reason,
          );
          if (event.code !== 1000) {
            this.handleReconnection();
          }
        };

        this.toolsWs.onerror = (error) => {
          console.error("💬 Tools WebSocket error:", error);
          this.handleError(new Error("Tools WebSocket connection error"));
          reject(new Error("Tools WebSocket connection error"));
        };
      } catch (error) {
        this.handleError(error as Error);
        reject(error);
      }
    });
  }

  private onBothConnected(resolve: () => void): void {
    this.reconnectAttempts = 0;
    this.updateState({
      isConnected: true,
      connectionStatus: "Connected",
    });

    // Send initial connection message to text endpoint
    this.sendSystemMessage({
      type: "connection",
      user_id: this.userId,
      current_page: this.currentPage,
      timestamp: new Date().toISOString(),
    });

    resolve();
  }

  private handleTextMessage(data: any): void {
    try {
      console.log("💬 Text message received:", data);

      switch (data.type) {
        case "message":
          this.addMessage({
            id: this.generateMessageId(),
            type: "assistant",
            content: data.content || data.message || "Empty message",
            timestamp: new Date(data.timestamp || Date.now()),
            user_id: data.user_id,
          });
          break;

        case "typing_start":
          this.updateState({ isTyping: true });
          break;

        case "typing_stop":
          this.updateState({ isTyping: false });
          break;

        case "system":
          this.addMessage({
            id: this.generateMessageId(),
            type: "system",
            content: data.message || data.content || "System message",
            timestamp: new Date(data.timestamp || Date.now()),
          });
          break;

        case "error":
          this.addMessage({
            id: this.generateMessageId(),
            type: "system",
            content: `Error: ${data.message || data.error || "Unknown error"}`,
            timestamp: new Date(),
          });
          break;

        default:
          console.log("💬 Unknown text message type:", data.type);
          // Try to handle as a general message
          if (data.content || data.message) {
            this.addMessage({
              id: this.generateMessageId(),
              type: "assistant",
              content: data.content || data.message,
              timestamp: new Date(data.timestamp || Date.now()),
            });
          }
      }
    } catch (error) {
      console.error("💬 Error handling text message:", error);
      this.handleError(error as Error);
    }
  }

  private handleToolsMessage(data: any): void {
    try {
      console.log("💬 Tools message received:", data);

      switch (data.type) {
        case "navigation_result":
          this.handleNavigationResult(data);
          break;

        case "button_action_result":
          this.handleButtonActionResult(data);
          break;

        case "system":
          this.addMessage({
            id: this.generateMessageId(),
            type: "system",
            content: data.message || data.content || "System message",
            timestamp: new Date(data.timestamp || Date.now()),
          });
          break;

        default:
          console.log("💬 Unknown tools message type:", data.type);
      }
    } catch (error) {
      console.error("💬 Error handling tools message:", error);
      this.handleError(error as Error);
    }
  }

  private handleNavigationResult(data: any): void {
    try {
      console.log("💬 Navigation command received:", {
        action: data.action,
        type: data.type,
        result: data.result,
      });

      // Add system message to show navigation action
      const targetPage =
        data.result?.page || data.result?.value || "unknown page";
      this.addMessage({
        id: this.generateMessageId(),
        type: "system",
        content: `Navigating to ${targetPage}...`,
        timestamp: new Date(),
      });

      // Execute backend command directly
      if (data.action === "navigate" && data.result?.page) {
        // Navigation command - page is in data.result
        console.log("💬 Executing navigation to:", data.result.page);
        this.executeDirectNavigation(data.result.page, data.result);
      } else if (data.action === "navigate" && data.data?.page) {
        // Fallback: Navigation command - page is in data.data
        console.log("💬 Executing navigation to:", data.data.page);
        this.executeDirectNavigation(data.data.page, data.data);
      } else if (data.action === "click" && data.result?.element_name) {
        // Click command - execute using ButtonRegistry
        console.log("💬 Executing click action:", data.result.element_name);
        this.executeRegisteredButtonAction(
          data.result.element_name,
          data.result,
          "text",
        );

        // Check if it should also navigate
        if (data.result.page && data.result.page !== this.currentPage) {
          console.log(
            "💬 Click action includes navigation to:",
            data.result.page,
          );
          this.executeDirectNavigation(data.result.page, data.result);
        }
      } else if (data.action === "click" && data.data?.element_name) {
        // Fallback: Click command - execute using ButtonRegistry
        console.log("💬 Executing click action:", data.data.element_name);
        this.executeRegisteredButtonAction(
          data.data.element_name,
          data.data,
          "text",
        );

        // Check if it should also navigate
        if (data.data.page && data.data.page !== this.currentPage) {
          console.log(
            "💬 Click action includes navigation to:",
            data.data.page,
          );
          this.executeDirectNavigation(data.data.page, data.data);
        }
      } else {
        console.log("💬 Navigation command processed:", data.action);
        console.log("💬 Full data received:", JSON.stringify(data, null, 2));
      }
    } catch (error) {
      console.error("💬 Error handling navigation result:", error);
      this.handleError(error as Error);
    }
  }

  private handleButtonActionResult(data: any): void {
    try {
      console.log("💬 Handling button action result:", data);

      // Add system message to show button action
      this.addMessage({
        id: this.generateMessageId(),
        type: "system",
        content: `Executing action: ${data.result?.element_name || "unknown element"}`,
        timestamp: new Date(),
      });

      if (data.result?.element_name) {
        // Extract element name from the result object
        const elementName = data.result.element_name;
        const context = data.result.context || {};
        console.log("💬 Executing button action:", elementName);

        // Execute using ButtonRegistry
        this.executeRegisteredButtonAction(elementName, context, "text");
      } else {
        console.warn("💬 No element name specified in button action result");
      }
    } catch (error) {
      console.error("💬 Error handling button action result:", error);
      this.handleError(error as Error);
    }
  }

  /**
   * Execute registered button action using robust mapping system
   * This method uses the new mapping configuration for reliable button targeting
   */
  private async executeRegisteredButtonAction(
    elementName: string,
    context: any,
    source: "voice" | "text",
  ): Promise<void> {
    console.log(
      "💬 Executing mapped button action:",
      elementName,
      "with context:",
      context,
    );

    try {
      // Get button mapping configuration
      const mapping = getButtonMapping(elementName);
      
      if (!mapping) {
        console.error(`💬 No mapping found for element: ${elementName}`);
        console.error(`💬 Available mappings:`, Object.keys(BUTTON_MAPPING_CONFIG));
        console.error(`💬 Please add mapping for: "${elementName}"`);
        
        this.addMessage({
          id: this.generateMessageId(),
          type: "system",
          content: `No mapping found for action: ${elementName}`,
          timestamp: new Date(),
        });
        return;
      }

      // Execute using mapping system
      const success = await this.executeMappedButtonAction(mapping, context, source);
      
      if (success) {
        console.log(`💬 Successfully executed mapped button action: ${elementName}`);
        this.addMessage({
          id: this.generateMessageId(),
          type: "system",
          content: `Executed action: ${mapping.name}`,
          timestamp: new Date(),
        });
      } else {
        console.warn(`💬 Failed to execute mapped button action: ${elementName}`);
        this.addMessage({
          id: this.generateMessageId(),
          type: "system",
          content: `Failed to execute action: ${mapping.name}`,
          timestamp: new Date(),
        });
      }
    } catch (error) {
      console.error("💬 Error executing mapped button action:", error);
      this.addMessage({
        id: this.generateMessageId(),
        type: "system",
        content: `Error executing action: ${elementName}`,
        timestamp: new Date(),
      });
    }
  }

  /**
   * Execute button action using mapping configuration
   */
  private async executeMappedButtonAction(
    mapping: ButtonMappingConfig,
    context: any,
    source: "voice" | "text"
  ): Promise<boolean> {
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
  private async fillTextarea(mapping: ButtonMappingConfig, searchQuery: string): Promise<boolean> {
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
  private validateButton(button: HTMLElement, validation: ButtonMappingConfig['validation']): boolean {
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
    
    // Check expected ID
    if (validation.expectedId) {
      if (button.id !== validation.expectedId) {
        console.warn(`🎯 ID validation failed. Expected: ${validation.expectedId}, Got: ${button.id}`);
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
  private async tryFuzzyMatching(mapping: ButtonMappingConfig): Promise<boolean> {
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

  /**
   * Execute direct navigation based on backend response
   * This method dispatches a SPA event instead of reloading the page
   */
  private executeDirectNavigation(targetPage: string, result: any): void {
    console.log("💬 Executing direct navigation to:", targetPage);

    try {
      // Update current page tracking
      this.currentPage = targetPage;

      // Dispatch event for SPA navigation handled by VoiceNavigationHandler
      if (typeof window !== "undefined") {
        const event = new CustomEvent("voice-navigation", {
          detail: {
            page: targetPage,
            previousPage: result?.previous_page || null,
            type: "page_navigation",
            context: result || {},
            timestamp: new Date().toISOString(),
            source: "text_conversation",
          },
        });
        window.dispatchEvent(event);
      }

      // Add success message
      this.addMessage({
        id: this.generateMessageId(),
        type: "system",
        content: `Successfully navigated to ${targetPage}`,
        timestamp: new Date(),
      });
    } catch (error) {
      console.error("💬 Error during direct navigation:", error);
      this.addMessage({
        id: this.generateMessageId(),
        type: "system",
        content: `Navigation failed: ${error.message}`,
        timestamp: new Date(),
      });
      // As a last resort, fallback to full reload
      if (typeof window !== "undefined") {
        window.location.href = `/${targetPage.toLowerCase().replace(/\s+/g, "-")}`;
      }
    }
  }

  sendMessage(content: string): void {
    if (!this.textWs || this.textWs.readyState !== WebSocket.OPEN) {
      console.warn("💬 Cannot send message: Text WebSocket not connected");
      return;
    }

    // Add user message to local state first
    const userMessage: TextMessage = {
      id: this.generateMessageId(),
      type: "user",
      content,
      timestamp: new Date(),
      user_id: this.userId,
    };

    this.addMessage(userMessage);

    // Send to server
    const payload = {
      type: "message",
      content,
      user_id: this.userId,
      current_page: this.currentPage,
      timestamp: new Date().toISOString(),
    };

    try {
      this.textWs.send(JSON.stringify(payload));
      console.log("💬 Message sent:", payload);
    } catch (error) {
      console.error("💬 Error sending message:", error);
      this.addMessage({
        id: this.generateMessageId(),
        type: "system",
        content: "Failed to send message. Please try again.",
        timestamp: new Date(),
      });
    }
  }

  private sendSystemMessage(payload: any): void {
    if (!this.textWs || this.textWs.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      this.textWs.send(JSON.stringify(payload));
      console.log("💬 System message sent:", payload);
    } catch (error) {
      console.error("💬 Error sending system message:", error);
    }
  }

  private addMessage(message: TextMessage): void {
    this.state.messages = [...this.state.messages, message];
    this.updateState({ messages: this.state.messages });
    this.onMessage?.(message);
  }

  private updateState(updates: Partial<TextConversationState>): void {
    this.state = { ...this.state, ...updates };
    this.onStateChange?.(this.state);
  }

  private generateMessageId(): string {
    return `text-msg-${Date.now()}-${this.messageIdCounter++}`;
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.addMessage({
        id: this.generateMessageId(),
        type: "system",
        content: "Connection lost. Maximum reconnection attempts reached.",
        timestamp: new Date(),
      });
      this.onError?.(new Error("Max reconnection attempts reached"));
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    this.updateState({
      connectionStatus: `Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`,
    });

    this.reconnectTimeout = setTimeout(() => {
      this.connect().catch((error) => {
        this.handleError(error);
      });
    }, delay);
  }

  private handleError(error: Error): void {
    console.error("💬 Text conversation error:", error);
    this.onError?.(error);

    this.addMessage({
      id: this.generateMessageId(),
      type: "system",
      content: `Error: ${error.message}`,
      timestamp: new Date(),
    });
  }

  disconnect(): void {
    // Clear reconnection timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Close both WebSockets
    if (this.textWs) {
      this.textWs.close(1000, "Manual disconnect");
      this.textWs = null;
    }

    if (this.toolsWs) {
      this.toolsWs.close(1000, "Manual disconnect");
      this.toolsWs = null;
    }

    this.reconnectAttempts = 0;
    this.updateState({
      isConnected: false,
      connectionStatus: "Disconnected",
      isTyping: false,
    });
  }

  clearMessages(): void {
    this.updateState({ messages: [] });
  }

  getState(): TextConversationState {
    return { ...this.state };
  }

  isConnected(): boolean {
    return (
      this.state.isConnected &&
      this.textWs?.readyState === WebSocket.OPEN &&
      this.toolsWs?.readyState === WebSocket.OPEN
    );
  }

  // Update current page context
  updateCurrentPage(page: string): void {
    // Only update and notify if page actually changed
    if (page === this.currentPage) {
      return;
    }

    const previousPage = this.currentPage;
    this.currentPage = page;

    // Notify both servers if connected
    if (this.textWs?.readyState === WebSocket.OPEN) {
      this.sendSystemMessage({
        type: "page_change",
        user_id: this.userId,
        current_page: page,
        previous_page: previousPage,
        timestamp: new Date().toISOString(),
      });
    }

    if (this.toolsWs?.readyState === WebSocket.OPEN) {
      try {
        this.toolsWs.send(
          JSON.stringify({
            type: "page_change",
            user_id: this.userId,
            current_page: page,
            previous_page: previousPage,
            timestamp: new Date().toISOString(),
          }),
        );
      } catch (error) {
        console.error("💬 Error sending page change to tools:", error);
      }
    }

    console.log(`💬 Page changed from ${previousPage} to ${page}`);
  }

  // Context persistence methods (similar to VoiceAgentService)
  private savePersistedContext(): void {
    try {
      const contextData = {
        messages: this.state.messages,
        currentPage: this.currentPage,
        userId: this.userId,
        timestamp: new Date().toISOString(),
      };

      sessionStorage.setItem('text_conversation_context', JSON.stringify(contextData));
      sessionStorage.setItem('text_conversation_last_page', this.currentPage);
      
      // Save connection state
      sessionStorage.setItem(
        'text_conversation_connection_state',
        JSON.stringify({
          isConnected: this.state.isConnected,
          connectionStatus: this.state.connectionStatus,
        }),
      );
    } catch (error) {
      console.error('💬 Failed to save persisted context:', error);
    }
  }

  private restorePersistedContext(): boolean {
    try {
      const contextData = sessionStorage.getItem('text_conversation_context');
      if (!contextData) return false;

      const parsed = JSON.parse(contextData);
      
      // Restore messages if they exist and are recent (within 1 hour)
      if (parsed.messages && Array.isArray(parsed.messages)) {
        const messageAge = Date.now() - new Date(parsed.timestamp).getTime();
        if (messageAge < 3600000) { // 1 hour
          this.state.messages = parsed.messages;
          this.updateState({ messages: this.state.messages });
          console.log('💬 Restored persisted messages:', parsed.messages.length);
        }
      }

      // Restore current page
      if (parsed.currentPage) {
        this.currentPage = parsed.currentPage;
        console.log('💬 Restored current page:', this.currentPage);
      }

      return true;
    } catch (error) {
      console.error('💬 Failed to restore persisted context:', error);
      return false;
    }
  }

  private async restoreConnectionIfNeeded(): Promise<void> {
    const connectionState = sessionStorage.getItem('text_conversation_connection_state');
    if (!connectionState) return;

    try {
      const parsed = JSON.parse(connectionState);
      if (parsed.isConnected) {
        console.log('💬 Restoring text conversation connection...');
        await this.connect();
      }
    } catch (error) {
      console.error('💬 Failed to restore connection:', error);
    }
  }

  private setupAutoReconnection(): void {
    if (!this.autoReconnectEnabled) return;

    // Check connection every 30 seconds
    this.reconnectInterval = setInterval(() => {
      if (this.state.isConnected && !this.isConnected()) {
        console.log('💬 Auto-reconnecting due to lost connection...');
        this.handleReconnection();
      }
    }, 30000); // 30 seconds
  }

  private initializePageDetection(): void {
    // Detect initial page
    this.currentPage = this.detectCurrentPage();

    // Listen for route changes
    if (typeof window !== 'undefined') {
      // Listen for popstate (browser back/forward buttons)
      const handlePopState = () => {
        const newPage = this.detectCurrentPage();
        this.updatePageState(newPage, 'browser_navigation');
      };

      // Listen for hash changes
      const handleHashChange = () => {
        const newPage = this.detectCurrentPage();
        this.updatePageState(newPage, 'hash_change');
      };

      // Listen for SPA navigation events
      const handleNavigation = (event: Event) => {
        try {
          const detail = (event as CustomEvent).detail;
          if (detail?.page) {
            this.updatePageState(detail.page, 'spa_navigation');
          }
        } catch (_) {}
      };

      // Listen for beforeunload to save context
      const handleBeforeUnload = () => {
        this.savePersistedContext();
        sessionStorage.setItem('text_conversation_last_page', this.currentPage);
      };

      // Listen for load to detect page changes and restore context
      const handleLoad = () => {
        const newPage = this.detectCurrentPage();
        const lastPage = sessionStorage.getItem('text_conversation_last_page');

        if (lastPage && lastPage !== newPage) {
          this.updatePageState(newPage, 'page_load');
        }

        // Restore context and connection if needed
        this.restorePersistedContext();
        this.restoreConnectionIfNeeded();
      };

      // Add event listeners
      window.addEventListener('popstate', handlePopState);
      window.addEventListener('hashchange', handleHashChange);
      window.addEventListener('voice-navigation', handleNavigation as EventListener);
      window.addEventListener('beforeunload', handleBeforeUnload);
      window.addEventListener('load', handleLoad);
    }
  }

  private detectCurrentPage(): string {
    if (typeof window === 'undefined') return 'dashboard';

    const pathname = window.location.pathname;
    const path = pathname.replace(/^\//, '').toLowerCase();

    const pageMap: Record<string, string> = {
      '': 'dashboard',
      dashboard: 'dashboard',
      'file-query': 'file-query',
      'database-query': 'database-query',
      tables: 'tables',
      users: 'users',
      'ai-reports': 'ai-reports',
      'company-structure': 'company-structure',
      'voice-control': 'voice-control',
      'user-configuration': 'user-configuration',
    };

    return pageMap[path] || path || 'dashboard';
  }

  private updatePageState(newPage: string, source: string): void {
    if (newPage === this.currentPage) return;

    const previousPage = this.currentPage;
    this.currentPage = newPage;

    console.log(`💬 Page state updated from ${previousPage} to ${newPage} (source: ${source})`);

    // Update page context on server if connected
    this.updateCurrentPage(newPage);

    // Save context after page change
    this.savePersistedContext();
  }

  // Cleanup method
  cleanup(): void {
    // Clear intervals
    if (this.reconnectInterval) {
      clearInterval(this.reconnectInterval);
      this.reconnectInterval = null;
    }

    // Clear timeouts
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Disconnect WebSockets
    this.disconnect();
  }
}
