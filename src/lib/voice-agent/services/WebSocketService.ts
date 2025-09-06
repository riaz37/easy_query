import { VOICE_AGENT_CONFIG } from "../config";
import { MessageService } from "./MessageService";
import { VoiceMessage, InteractionType } from "../types";
// Removed old ButtonRegistry - using mapping system only
import { getButtonMapping, ButtonMappingConfig, BUTTON_MAPPING_CONFIG } from "../config/default-button-actions";

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
   * Execute registered button action using robust mapping system
   * This method uses the new mapping configuration for reliable button targeting
   */
  private async executeRegisteredButtonAction(
    elementName: string,
    context: any,
    source: "voice" | "text",
  ): Promise<void> {
    console.log(
      "🖱️ Executing mapped button action:",
      elementName,
      "with context:",
      context,
    );

    try {
      // Get button mapping configuration
      const mapping = getButtonMapping(elementName);
      
      if (!mapping) {
        console.error(`🖱️ No mapping found for element: ${elementName}`);
        console.error(`🖱️ Available mappings:`, Object.keys(BUTTON_MAPPING_CONFIG));
        console.error(`🖱️ Please add mapping for: "${elementName}"`);
        return;
      }

      // Execute using mapping system
      const success = await this.executeMappedButtonAction(mapping, context, source);
      
      if (success) {
        console.log(`🖱️ Successfully executed mapped button action: ${elementName}`);
      } else {
        console.warn(`🖱️ Failed to execute mapped button action: ${elementName}`);
      }
    } catch (error) {
      console.error("🖱️ Error executing mapped button action:", error);
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
    
    // Check expected text (handle both string and array)
    if (validation.expectedText) {
      const buttonText = button.textContent?.trim();
      const expectedTexts = Array.isArray(validation.expectedText) 
        ? validation.expectedText 
        : [validation.expectedText];
      
      const textMatches = expectedTexts.some(expectedText => 
        buttonText?.includes(expectedText)
      );
      
      if (!textMatches) {
        console.warn(`🎯 Text validation failed. Expected one of: ${expectedTexts.join(', ')}, Got: ${buttonText}`);
        return false;
      }
    }
    
    // Check expected class (handle both string and array)
    if (validation.expectedClass) {
      const expectedClasses = Array.isArray(validation.expectedClass) 
        ? validation.expectedClass 
        : [validation.expectedClass];
      
      const classMatches = expectedClasses.some(expectedClass => 
        button.className.includes(expectedClass)
      );
      
      if (!classMatches) {
        console.warn(`🎯 Class validation failed. Expected one of: ${expectedClasses.join(', ')}, Got: ${button.className}`);
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
    
    // Log all available buttons for debugging
    allButtons.forEach((btn, index) => {
      console.log(`🎯 Button ${index}:`, {
        text: btn.textContent?.trim(),
        id: btn.id,
        className: btn.className,
        visible: !!btn.offsetParent
      });
    });
    
    // Try to find button by partial text match
    if (mapping.validation?.expectedText) {
      const expectedTexts = Array.isArray(mapping.validation.expectedText) 
        ? mapping.validation.expectedText 
        : [mapping.validation.expectedText];
      
      const matchingButton = allButtons.find(btn => {
        const buttonText = btn.textContent?.toLowerCase().trim();
        return buttonText && expectedTexts.some(expectedText => {
          const lowerExpectedText = expectedText.toLowerCase();
          return (
            buttonText.includes(lowerExpectedText) || 
            lowerExpectedText.includes(buttonText) ||
            buttonText.split(' ').some(word => lowerExpectedText.includes(word))
          );
        });
      });
      
      if (matchingButton) {
        console.log(`🎯 Found button via fuzzy matching: ${matchingButton.textContent?.trim()}`);
        matchingButton.click();
        console.log(`✅ Successfully clicked button via fuzzy matching`);
        return true;
      }
    }
    
    console.error(`🎯 Failed to find button: ${mapping.name}`);
    console.error(`🎯 Available buttons:`, allButtons.map(btn => ({
      text: btn.textContent?.trim(),
      id: btn.id,
      className: btn.className
    })));
    
    return false;
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
