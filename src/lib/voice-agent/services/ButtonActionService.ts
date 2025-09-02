import { MessageService } from './MessageService'
import { NavigationData, InteractionType } from '../types'

// Define button action configuration interface
interface ButtonActionConfig {
  actionType: InteractionType
  handler: (context?: any) => void
  params: {
    clicked: boolean
    element_name: string
    param: string
    value: string
    context?: string
  }
  customParams?: Record<string, any>
}

// Define button action registry
interface ButtonActionRegistry {
  [key: string]: ButtonActionConfig
}

export class ButtonActionService {
  private static currentPage = 'dashboard'
  private static buttonActions: ButtonActionRegistry = new Map()

  static setCurrentPage(page: string): void {
    this.currentPage = page
  }

  static getCurrentPage(): string {
    return this.currentPage
  }

  /**
   * Register a button action with its configuration
   * This allows for dynamic registration of button actions
   */
  static registerButtonAction(
    elementName: string, 
    config: Omit<ButtonActionConfig, 'params'> & { 
      elementName: string 
    }
  ): void {
    const actionConfig: ButtonActionConfig = {
      ...config,
      params: {
        clicked: true,
        element_name: config.elementName,
        param: 'clicked,name',
        value: `true,${config.elementName}`,
        context: config.context || 'voice_agent_triggered'
      }
    }
    
    this.buttonActions.set(elementName.toLowerCase(), actionConfig)
    console.log(`🖱️ Registered button action: ${elementName}`, actionConfig)
  }

  /**
   * Register multiple button actions at once
   */
  static registerButtonActions(actions: Array<{ elementName: string } & Omit<ButtonActionConfig, 'params'>>): void {
    actions.forEach(action => this.registerButtonAction(action.elementName, action))
  }

  /**
   * Unregister a button action
   */
  static unregisterButtonAction(elementName: string): void {
    this.buttonActions.delete(elementName.toLowerCase())
    console.log(`🖱️ Unregistered button action: ${elementName}`)
  }

  /**
   * Execute button click action based on element name
   * This method now uses the registered configurations instead of hardcoded switch statements
   */
  static executeButtonAction(elementName: string, context?: any): void {
    console.log('🖱️ ButtonActionService executing button action:', elementName, context)
    
    const normalizedName = elementName.toLowerCase()
    const actionConfig = this.buttonActions.get(normalizedName)
    
    if (actionConfig) {
      console.log(`🖱️ Found registered action for: ${elementName}`)
      this.executeRegisteredAction(actionConfig, context)
    } else {
      console.warn(`🖱️ No registered action found for: ${elementName}`)
      this.executeGenericButtonAction(elementName, context)
    }
  }

  /**
   * Execute a registered button action
   */
  private static executeRegisteredAction(actionConfig: ButtonActionConfig, context?: any): void {
    try {
      // Create navigation data for logging
      const navigationData = this.createButtonActionData(actionConfig.actionType, {
        ...actionConfig.params,
        ...actionConfig.customParams,
        context: context || actionConfig.params.context
      })

      // Emit button action event
      this.emitButtonActionEvent(actionConfig.actionType, navigationData)
      
      // Execute the registered handler
      actionConfig.handler(context)
      
    } catch (error) {
      console.error(`🖱️ Error executing button action: ${actionConfig.params.element_name}`, error)
      this.emitButtonActionEvent('error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        element_name: actionConfig.params.element_name,
        context: context || actionConfig.params.context
      })
    }
  }

  /**
   * Handle generic button actions for unknown button names
   */
  private static executeGenericButtonAction(elementName: string, context?: any): void {
    console.log('🖱️ Executing generic button action:', elementName)
    
    const navigationData = this.createButtonActionData('button_click', {
      clicked: true,
      element_name: elementName,
      param: 'clicked,name',
      value: `true,${elementName}`,
      context: context || 'voice_agent_triggered'
    })

    // Emit generic button action event
    this.emitButtonActionEvent('button_click', navigationData)
    
    // Try to find and click the button by text content
    this.findAndClickButtonByText(elementName)
  }

  /**
   * Find and click a button by its text content
   * This is a fallback for when we can't identify the button by name
   */
  private static findAndClickButtonByText(buttonText: string): void {
    if (typeof window === 'undefined') return

    // Find buttons by text content
    const buttons = Array.from(document.querySelectorAll('button'))
    const targetButton = buttons.find(button => {
      const text = button.textContent?.toLowerCase().trim()
      return text && text.includes(buttonText.toLowerCase())
    })

    if (targetButton) {
      console.log('🖱️ Found button by text, clicking:', buttonText)
      targetButton.click()
    } else {
      console.warn('🖱️ Could not find button with text:', buttonText)
    }
  }

  /**
   * Create navigation data for button actions
   */
  private static createButtonActionData(
    interactionType: InteractionType,
    options: Partial<NavigationData> = {}
  ): NavigationData {
    return MessageService.createNavigationData(
      'navigation',
      interactionType,
      this.currentPage,
      null, // previousPage
      {
        timestamp: new Date().toISOString(),
        user_id: 'frontend_user',
        success: true,
        error_message: null,
        ...options
      }
    )
  }

  /**
   * Emit button action events
   */
  private static emitButtonActionEvent(type: string, data: NavigationData): void {
    if (typeof window !== 'undefined') {
      const event = new CustomEvent(`voice-agent-${type}`, {
        detail: { ...data, timestamp: new Date().toISOString() }
      })
      window.dispatchEvent(event)
    }
  }

  /**
   * Get all available button actions for the current page
   * This can be used to provide voice agent with available actions
   */
  static getAvailableButtonActions(): string[] {
    if (typeof window === 'undefined') return []

    const buttons = Array.from(document.querySelectorAll('button'))
    return buttons
      .map(button => button.textContent?.trim())
      .filter(text => text && text.length > 0)
      .slice(0, 10) // Limit to first 10 buttons
  }

  /**
   * Check if a specific button action is available
   */
  static isButtonActionAvailable(actionName: string): boolean {
    const availableActions = this.getAvailableButtonActions()
    return availableActions.some(action => 
      action.toLowerCase().includes(actionName.toLowerCase())
    )
  }
} 