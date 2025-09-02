import { buttonRegistry } from './ButtonRegistry'
import { defaultButtonActions } from '../config/default-button-actions'

/**
 * ButtonRegistrationService - Handles initialization and management of button actions
 *
 * This service:
 * 1. Registers all default button actions on initialization
 * 2. Provides methods to register custom button actions
 * 3. Handles auto-discovery of DOM buttons
 * 4. Manages lifecycle of button registrations
 */

export class ButtonRegistrationService {
  private static instance: ButtonRegistrationService | null = null
  private isInitialized = false
  private autoDiscoveryEnabled = true
  private discoveryInterval: NodeJS.Timeout | null = null
  private lastDiscoveryTime = 0
  private discoveryIntervalMs = 5000 // 5 seconds

  private constructor() {}

  static getInstance(): ButtonRegistrationService {
    if (!this.instance) {
      this.instance = new ButtonRegistrationService()
    }
    return this.instance
  }

  /**
   * Initialize the button registration system
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.log('🎯 ButtonRegistrationService already initialized')
      return
    }

    console.log('🎯 ButtonRegistrationService initializing...')

    try {
      // Register all default button actions
      this.registerDefaultActions()

      // Set up auto-discovery of DOM buttons
      if (this.autoDiscoveryEnabled) {
        this.startAutoDiscovery()
      }

      // Set up event listeners for button registry
      this.setupEventListeners()

      this.isInitialized = true
      console.log('🎯 ButtonRegistrationService initialized successfully')
    } catch (error) {
      console.error('🎯 Failed to initialize ButtonRegistrationService:', error)
      throw error
    }
  }

  /**
   * Register all default button actions
   */
  private registerDefaultActions(): void {
    console.log(`🎯 Registering ${defaultButtonActions.length} default button actions...`)

    buttonRegistry.registerMany(defaultButtonActions)

    const stats = buttonRegistry.getStats()
    console.log('🎯 Default actions registered:', {
      totalActions: stats.totalActions,
      totalAliases: stats.totalAliases,
      totalCategories: stats.totalCategories
    })
  }

  /**
   * Set up event listeners for button registry events
   */
  private setupEventListeners(): void {
    buttonRegistry.onButtonExecuted = (result) => {
      console.log(`🎯 Button executed: ${result.elementName}`, {
        success: result.success,
        executionTime: result.executionTime,
        error: result.error
      })

      // Dispatch custom event for other parts of the app to listen to
      if (typeof window !== 'undefined') {
        const event = new CustomEvent('button-action-executed', {
          detail: result
        })
        window.dispatchEvent(event)
      }
    }

    buttonRegistry.onButtonRegistered = (action) => {
      console.log(`🎯 Button action registered: ${action.name} (${action.id})`)
    }

    buttonRegistry.onButtonUnregistered = (actionId) => {
      console.log(`🎯 Button action unregistered: ${actionId}`)
    }
  }

  /**
   * Start auto-discovery of DOM buttons
   */
  private startAutoDiscovery(): void {
    if (typeof window === 'undefined') return

    console.log('🎯 Starting auto-discovery of DOM buttons...')

    // Initial discovery
    this.discoverDOMButtons()

    // Set up periodic discovery
    this.discoveryInterval = setInterval(() => {
      this.discoverDOMButtons()
    }, this.discoveryIntervalMs)

    // Set up mutation observer for real-time discovery
    this.setupMutationObserver()
  }

  /**
   * Stop auto-discovery
   */
  private stopAutoDiscovery(): void {
    if (this.discoveryInterval) {
      clearInterval(this.discoveryInterval)
      this.discoveryInterval = null
    }
    console.log('🎯 Auto-discovery stopped')
  }

  /**
   * Discover buttons in the DOM and register dynamic actions
   */
  private discoverDOMButtons(): void {
    const now = Date.now()
    if (now - this.lastDiscoveryTime < this.discoveryIntervalMs) {
      return // Throttle discovery
    }
    this.lastDiscoveryTime = now

    if (typeof window === 'undefined') return

    try {
      const buttons = Array.from(document.querySelectorAll('button, [role="button"], a[href]'))
      let discoveredCount = 0

      buttons.forEach(button => {
        const text = button.textContent?.trim()
        const ariaLabel = button.getAttribute('aria-label')
        const dataAction = button.getAttribute('data-action')
        const id = button.id

        // Create a unique identifier for the button
        const buttonId = dataAction || id || text?.toLowerCase().replace(/\s+/g, '-')

        if (buttonId && !buttonRegistry.isRegistered(buttonId)) {
          // Create a dynamic button action
          const buttonAction = {
            id: buttonId,
            name: text || ariaLabel || buttonId,
            aliases: [
              text?.toLowerCase(),
              ariaLabel?.toLowerCase(),
              dataAction
            ].filter(Boolean) as string[],
            description: `Auto-discovered: ${text || ariaLabel || buttonId}`,
            category: 'auto-discovered',
            handler: () => {
              console.log(`🎯 Auto-discovered button clicked: ${buttonId}`)
              ;(button as HTMLElement).click()
            }
          }

          buttonRegistry.register(buttonAction)
          discoveredCount++
        }
      })

      if (discoveredCount > 0) {
        console.log(`🎯 Auto-discovered ${discoveredCount} new buttons`)
      }
    } catch (error) {
      console.error('🎯 Error during button discovery:', error)
    }
  }

  /**
   * Set up mutation observer for real-time button discovery
   */
  private setupMutationObserver(): void {
    if (typeof window === 'undefined') return

    const observer = new MutationObserver((mutations) => {
      let shouldDiscover = false

      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              const element = node as Element
              // Check if new buttons were added
              if (element.tagName === 'BUTTON' ||
                  element.querySelector('button') ||
                  element.getAttribute('role') === 'button') {
                shouldDiscover = true
              }
            }
          })
        }
      })

      if (shouldDiscover) {
        // Debounce discovery to avoid excessive calls
        setTimeout(() => this.discoverDOMButtons(), 100)
      }
    })

    observer.observe(document.body, {
      childList: true,
      subtree: true
    })

    console.log('🎯 Mutation observer set up for real-time button discovery')
  }

  /**
   * Register a custom button action
   */
  registerCustomAction(action: {
    id: string
    name: string
    handler: (context?: any) => void | Promise<void>
    aliases?: string[]
    description?: string
    category?: string
    validation?: (context?: any) => boolean
  }): void {
    buttonRegistry.register({
      id: action.id,
      name: action.name,
      handler: action.handler,
      aliases: action.aliases,
      description: action.description,
      category: action.category || 'custom',
      validation: action.validation
    })

    console.log(`🎯 Custom button action registered: ${action.name}`)
  }

  /**
   * Register multiple custom actions
   */
  registerCustomActions(actions: Array<{
    id: string
    name: string
    handler: (context?: any) => void | Promise<void>
    aliases?: string[]
    description?: string
    category?: string
    validation?: (context?: any) => boolean
  }>): void {
    actions.forEach(action => this.registerCustomAction(action))
  }

  /**
   * Unregister a button action
   */
  unregisterAction(actionId: string): boolean {
    const result = buttonRegistry.unregister(actionId)
    if (result) {
      console.log(`🎯 Unregistered button action: ${actionId}`)
    } else {
      console.warn(`🎯 Failed to unregister button action: ${actionId}`)
    }
    return result
  }

  /**
   * Get all registered actions
   */
  getRegisteredActions() {
    return buttonRegistry.getActions()
  }

  /**
   * Get actions by category
   */
  getActionsByCategory(category: string) {
    return buttonRegistry.getActionsByCategory(category)
  }

  /**
   * Get all categories
   */
  getCategories() {
    return buttonRegistry.getCategories()
  }

  /**
   * Get execution statistics
   */
  getExecutionStats() {
    return buttonRegistry.getStats()
  }

  /**
   * Get execution history
   */
  getExecutionHistory() {
    return buttonRegistry.getExecutionHistory()
  }

  /**
   * Clear execution history
   */
  clearExecutionHistory() {
    buttonRegistry.clearHistory()
  }

  /**
   * Enable or disable auto-discovery
   */
  setAutoDiscovery(enabled: boolean): void {
    if (enabled && !this.autoDiscoveryEnabled) {
      this.autoDiscoveryEnabled = true
      this.startAutoDiscovery()
    } else if (!enabled && this.autoDiscoveryEnabled) {
      this.autoDiscoveryEnabled = false
      this.stopAutoDiscovery()
    }
  }

  /**
   * Export configuration for debugging
   */
  exportConfig() {
    return buttonRegistry.exportConfig()
  }

  /**
   * Reset the service (useful for testing)
   */
  reset(): void {
    this.stopAutoDiscovery()
    buttonRegistry.reset()
    this.isInitialized = false
    console.log('🎯 ButtonRegistrationService reset complete')
  }

  /**
   * Cleanup when shutting down
   */
  cleanup(): void {
    this.stopAutoDiscovery()
    console.log('🎯 ButtonRegistrationService cleaned up')
  }

  /**
   * Check if service is initialized
   */
  isReady(): boolean {
    return this.isInitialized
  }
}

// Export singleton instance
export const buttonRegistrationService = ButtonRegistrationService.getInstance()
