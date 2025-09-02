import { VOICE_AGENT_CONFIG } from "../config";
import { MessageService } from "./MessageService";
import { ButtonActionManager } from "./ButtonActionManager";
import { VoiceMessage, InteractionType } from "../types";
import { buttonRegistry, ButtonExecutionContext } from "./ButtonRegistry";

export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private currentPage = "dashboard";
  private userId: string;

  // Event handlers
  onMessage?: (message: VoiceMessage) => void;
  onConnectionChange?: (isConnected: boolean) => void;
  onError?: (error: Error) => void;

  constructor(userId: string) {
    this.userId = userId;
  }

  connect(currentPage: string, userId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.currentPage = currentPage;
        this.userId = userId;

        // Construct the full WebSocket URL with the base backend URL
        const baseUrl = VOICE_AGENT_CONFIG.BACKEND_URL;
        const wsBaseUrl = baseUrl.replace(
          /^https?:\/\//,
          baseUrl.startsWith("https") ? "wss://" : "ws://",
        );
        const wsUrl = `${wsBaseUrl}${VOICE_AGENT_CONFIG.WEBSOCKET_ENDPOINTS.VOICE_WS}?user_id=${userId}&current_page=${currentPage}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          this.onConnectionChange?.(true);
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            this.handleError(error as Error);
          }
        };

        this.ws.onclose = (event) => {
          this.onConnectionChange?.(false);

          if (event.code !== 1000) {
            this.handleReconnection();
          }
        };

        this.ws.onerror = (error) => {
          this.handleError(new Error("WebSocket connection error"));
          reject(new Error("WebSocket connection error"));
        };
      } catch (error) {
        this.handleError(error as Error);
      }
    });
  }

  private handleMessage(data: any): void {
    try {
      console.log("🎤 WebSocket message received:", data);

      if (data.type === "navigation_result") {
        this.handleNavigationResult(data);
      } else if (data.type === "button_action_result") {
        this.handleButtonActionResult(data);
      } else if (data.type === "system") {
        this.onMessage?.(
          MessageService.createSystemMessage(data.message || "System message"),
        );
      } else {
        console.log("🎤 Unknown message type:", data.type);
      }
    } catch (error) {
      this.handleError(error as Error);
    }
  }

  private handleNavigationResult(data: any): void {
    try {
      console.log("🧭 Backend command received:", {
        action: data.action,
        type: data.type,
        data: data.data,
        result: data.result,
      });

      // Execute backend command directly
      if (data.action === "navigate" && data.data?.page) {
        // Navigation command - page is in data.data
        const targetPage = data.data.page;
        console.log("🧭 Executing navigation to:", targetPage);
        this.executeDirectNavigation(targetPage, data.data);
      } else if (data.action === "click" && data.data?.element_name) {
        // Click command - execute using ButtonRegistry
        console.log("🖱️ Executing click action:", data.data.element_name);
        this.executeRegisteredButtonAction(
          data.data.element_name,
          data.data,
          "voice",
        );

        // Check if it should also navigate
        if (data.data.page && data.data.page !== this.currentPage) {
          console.log(
            "🧭 Click action includes navigation to:",
            data.data.page,
          );
          this.executeDirectNavigation(data.data.page, data.data);
        }
      } else if (
        data.data?.Action_type === "clicked" &&
        data.data?.element_name
      ) {
        // Fallback click command - execute using ButtonRegistry
        console.log(
          "🖱️ Executing fallback click action:",
          data.data.element_name,
        );
        this.executeRegisteredButtonAction(
          data.data.element_name,
          data.data,
          "voice",
        );

        // Check if it should also navigate
        if (data.data.page && data.data.page !== this.currentPage) {
          console.log(
            "🧭 Fallback click action includes navigation to:",
            data.data.page,
          );
          this.executeDirectNavigation(data.data.page, data.data);
        }
      } else {
        console.log(
          "🧭 Backend command processed:",
          data.action || data.data?.Action_type,
        );
        console.log("🧭 Full data received:", JSON.stringify(data, null, 2));
      }
    } catch (error) {
      this.handleError(error as Error);
    }
  }

  private handleButtonActionResult(data: any): void {
    try {
      console.log("🖱️ Handling button action result:", data);

      if (data.result?.element_name) {
        // Extract element name from the result object
        const elementName = data.result.element_name;
        const context = data.result.context || {};
        console.log("🖱️ Executing button action:", elementName);

        // Execute using ButtonRegistry
        this.executeRegisteredButtonAction(elementName, context, "voice");
      } else {
        console.warn("🖱️ No element name specified in button action result");
      }
    } catch (error) {
      this.handleError(error as Error);
    }
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.onError?.(new Error("Max reconnection attempts reached"));
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    this.reconnectTimeout = setTimeout(() => {
      this.connect(this.currentPage, this.userId).catch((error) => {
        this.handleError(error);
      });
    }, delay);
  }

  private handleError(error: Error): void {
    this.onError?.(error);
  }

  disconnect(): void {
    // Clear reconnection timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    // Close WebSocket
    if (this.ws) {
      this.ws.close(1000, "Manual disconnect");
      this.ws = null;
    }

    this.reconnectAttempts = 0;
    this.onConnectionChange?.(false);
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getConnectionState(): string {
    if (!this.ws) return "Disconnected";

    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return "Connecting...";
      case WebSocket.OPEN:
        return "Connected";
      case WebSocket.CLOSING:
        return "Closing...";
      case WebSocket.CLOSED:
        return "Disconnected";
      default:
        return "Unknown";
    }
  }

  sendMessage(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // console.warn('🎤 Cannot send message: WebSocket not connected')
    }
  }

  /**
   * Notify backend about page change without reconnecting
   */
  notifyPageChange(newPage: string): void {
    this.currentPage = newPage;
    const payload = {
      type: "page_change",
      user_id: this.userId,
      page: newPage,
      timestamp: new Date().toISOString(),
    };
    this.sendMessage(payload);
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
      "🖱️ Executing registered button action:",
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
        console.log(`🖱️ Successfully executed button action: ${elementName}`);
      } else {
        console.warn(
          `🖱️ Failed to execute button action: ${elementName}`,
          result.error,
        );
      }
    } catch (error) {
      console.error("🖱️ Error executing registered button action:", error);
    }
  }

  /**
   * Execute direct navigation based on backend response
   * This method dispatches a SPA event instead of reloading the page
   */
  private executeDirectNavigation(targetPage: string, result: any): void {
    console.log("🧭 Executing direct navigation to:", targetPage);

    try {
      // Update current page tracking
      this.currentPage = targetPage;

      // Dispatch event for SPA navigation handled by VoiceNavigationHandler
      if (typeof window !== "undefined") {
        const event = new CustomEvent("voice-navigation", {
          detail: {
            page: targetPage,
            previousPage: null,
            type: "page_navigation",
            context: result || {},
            timestamp: new Date().toISOString(),
          },
        });
        window.dispatchEvent(event);
      }
    } catch (error) {
      console.error("🧭 Error during direct navigation:", error);
      // As a last resort, fallback to full reload
      if (typeof window !== "undefined") {
        window.location.href = `/${targetPage.toLowerCase().replace(/\s+/g, "-")}`;
      }
    }
  }
}
