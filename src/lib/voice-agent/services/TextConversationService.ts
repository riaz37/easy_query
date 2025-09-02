import { VOICE_AGENT_CONFIG } from "../config";
import { buttonRegistry, ButtonExecutionContext } from "./ButtonRegistry";
import { buttonRegistrationService } from "./ButtonRegistrationService";

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

    // Initialize button registration service
    this.initializeButtonRegistration();
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
   * Execute registered button action using ButtonRegistry
   * This method uses the scalable button registry system
   */
  private async executeRegisteredButtonAction(
    elementName: string,
    context: any,
    source: "voice" | "text",
  ): Promise<void> {
    console.log(
      "💬 Executing registered button action:",
      elementName,
      "with context:",
      context,
    );

    try {
      const executionContext: ButtonExecutionContext = {
        elementName,
        page: context.page,
        previousPage: context.previous_page,
        context,
        source,
        timestamp: new Date().toISOString(),
      };

      const result = await buttonRegistry.execute(
        elementName,
        executionContext,
      );

      if (result.success) {
        console.log(`💬 Successfully executed button action: ${elementName}`);
        this.addMessage({
          id: this.generateMessageId(),
          type: "system",
          content: `Executed action: ${elementName}`,
          timestamp: new Date(),
        });
      } else {
        console.warn(
          `💬 Failed to execute button action: ${elementName}`,
          result.error,
        );
        this.addMessage({
          id: this.generateMessageId(),
          type: "system",
          content: `Failed to execute action: ${elementName} - ${result.error}`,
          timestamp: new Date(),
        });
      }
    } catch (error) {
      console.error("💬 Error executing registered button action:", error);
      this.addMessage({
        id: this.generateMessageId(),
        type: "system",
        content: `Error executing action: ${elementName}`,
        timestamp: new Date(),
      });
    }
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

  // Initialize button registration service
  private async initializeButtonRegistration(): Promise<void> {
    try {
      console.log("💬 Initializing button registration service...");
      await buttonRegistrationService.initialize();
      console.log("💬 Button registration service initialized successfully");
    } catch (error) {
      console.error(
        "💬 Failed to initialize button registration service:",
        error,
      );
    }
  }
}
