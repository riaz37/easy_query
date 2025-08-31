import { ButtonActionService } from './ButtonActionService'
import { 
  BUTTON_ACTION_CONFIGS, 
  ButtonActionDefinition, 
  getAllButtonActions,
  findButtonAction,
  getButtonActionsByCategory 
} from '../config/button-actions'

export class ButtonActionManager {
  private static isInitialized = false
  private static registeredActions = new Set<string>()

  /**
   * Initialize the button action manager
   * This should be called once during app startup
   */
  static initialize(): void {
    if (this.isInitialized) {
      console.log('🖱️ ButtonActionManager already initialized')
      return
    }

    console.log('🖱️ Initializing ButtonActionManager...')
    
    // Register all button actions from configuration
    this.registerAllButtonActions()
    
    // Set up automatic button detection
    this.setupAutomaticButtonDetection()
    
    this.isInitialized = true
    console.log('🖱️ ButtonActionManager initialized successfully')
  }

  /**
   * Register all button actions from the configuration
   */
  private static registerAllButtonActions(): void {
    const allActions = getAllButtonActions()
    
    allActions.forEach(action => {
      this.registerButtonAction(action)
    })
    
    console.log(`🖱️ Registered ${allActions.length} button actions`)
  }

  /**
   * Register a single button action
   */
  private static registerButtonAction(actionDef: ButtonActionDefinition): void {
    const actionKey = actionDef.elementName.toLowerCase()
    
    if (this.registeredActions.has(actionKey)) {
      console.log(`🖱️ Button action already registered: ${actionDef.elementName}`)
      return
    }

    // Create the action configuration
    const actionConfig = {
      actionType: actionDef.actionType,
      handler: this.createActionHandler(actionDef),
      context: actionDef.context,
      customParams: actionDef.customParams
    }

    // Register the main action
    ButtonActionService.registerButtonAction(actionDef.elementName, actionConfig)
    this.registeredActions.add(actionKey)

    // Register aliases if they exist
    if (actionDef.aliases) {
      actionDef.aliases.forEach(alias => {
        const aliasKey = alias.toLowerCase()
        if (!this.registeredActions.has(aliasKey)) {
          ButtonActionService.registerButtonAction(alias, actionConfig)
          this.registeredActions.add(aliasKey)
          console.log(`🖱️ Registered alias: ${alias} -> ${actionDef.elementName}`)
        }
      })
    }
  }

  /**
   * Create an action handler function based on the action definition
   */
  private static createActionHandler(actionDef: ButtonActionDefinition): (context?: any) => void {
    return (context?: any) => {
      console.log(`🖱️ Executing action: ${actionDef.elementName}`, { context, actionDef })
      
      // Execute the appropriate handler based on the action type
      switch (actionDef.handler) {
        case 'executeViewReportAction':
          this.executeViewReportAction(context)
          break
        case 'executeReportGenerationAction':
          this.executeReportGenerationAction(context)
          break
        case 'executeFileUploadAction':
          this.executeFileUploadAction(context)
          break
        case 'executeFileSearchAction':
          this.executeFileSearchAction(context)
          break
        case 'executeNavigationAction':
          this.executeNavigationAction(actionDef.elementName, context)
          break
        case 'executeCreateTableAction':
          this.executeCreateTableAction(context)
          break
        case 'executeDeleteTableAction':
          this.executeDeleteTableAction(context)
          break
        case 'executeDataExportAction':
          this.executeDataExportAction(context)
          break
        case 'executeCreateUserAction':
          this.executeCreateUserAction(context)
          break
        case 'executeDeleteUserAction':
          this.executeDeleteUserAction(context)
          break
        case 'executePasswordChangeAction':
          this.executePasswordChangeAction(context)
          break
        case 'executeAddBusinessRuleAction':
          this.executeAddBusinessRuleAction(context)
          break
        case 'executeEditBusinessRuleAction':
          this.executeEditBusinessRuleAction(context)
          break
        case 'executeDeleteBusinessRuleAction':
          this.executeDeleteBusinessRuleAction(context)
          break
        default:
          console.warn(`🖱️ Unknown action handler: ${actionDef.handler}`)
          this.executeGenericAction(actionDef, context)
      }
    }
  }

  /**
   * Set up automatic button detection for dynamically added buttons
   */
  private static setupAutomaticButtonDetection(): void {
    if (typeof window === 'undefined') return

    // Use MutationObserver to detect new buttons being added to the DOM
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              const element = node as Element
              this.detectButtonsInElement(element)
            }
          })
        }
      })
    })

    // Start observing
    observer.observe(document.body, {
      childList: true,
      subtree: true
    })

    console.log('🖱️ Automatic button detection enabled')
  }

  /**
   * Detect buttons in a specific element and register them if they match our patterns
   */
  private static detectButtonsInElement(element: Element): void {
    const buttons = element.querySelectorAll('button')
    
    buttons.forEach(button => {
      const buttonText = button.textContent?.trim()
      if (buttonText) {
        // Check if this button matches any of our action patterns
        const matchingAction = findButtonAction(buttonText)
        if (matchingAction && !this.registeredActions.has(buttonText.toLowerCase())) {
          console.log(`🖱️ Auto-detected button: ${buttonText}`)
          this.registerButtonAction(matchingAction)
        }
      }
    })
  }

  /**
   * Get all available button actions
   */
  static getAvailableActions(): ButtonActionDefinition[] {
    return getAllButtonActions()
  }

  /**
   * Get button actions by category
   */
  static getActionsByCategory(categoryName: string): ButtonActionDefinition[] {
    return getButtonActionsByCategory(categoryName)
  }

  /**
   * Check if a button action is available
   */
  static isActionAvailable(actionName: string): boolean {
    return !!findButtonAction(actionName)
  }

  /**
   * Get action information
   */
  static getActionInfo(actionName: string): ButtonActionDefinition | undefined {
    return findButtonAction(actionName)
  }

  // Action handler implementations
  private static executeViewReportAction(context?: any): void {
    console.log('📊 Executing view report action')
    if (typeof window !== 'undefined') {
      const hasResults = sessionStorage.getItem('reportResults')
      if (hasResults) {
        window.location.href = '/ai-results'
      } else {
        this.showMessage('No reports available to view. Please generate a report first.')
      }
    }
  }

  private static executeReportGenerationAction(context?: any): void {
    console.log('📈 Executing report generation action')
    this.dispatchEvent('voice-agent-generate-report', {
      action: 'generate_report',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeFileUploadAction(context?: any): void {
    console.log('📤 Executing file upload action')
    this.dispatchEvent('voice-agent-file-upload', {
      action: 'file_upload',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeFileSearchAction(context?: any): void {
    console.log('🔍 Executing file search action')
    const searchQuery = context?.search_query || 'Search file system'
    this.dispatchEvent('voice-agent-file-search', {
      action: 'file_search',
      searchQuery,
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeNavigationAction(pageName: string, context?: any): void {
    console.log('🧭 Executing navigation action to:', pageName)
    if (typeof window !== 'undefined') {
      const pagePath = this.getPagePath(pageName)
      if (pagePath) {
        window.location.href = pagePath
      } else {
        console.warn('🧭 Unknown page for navigation:', pageName)
      }
    }
  }

  private static executeCreateTableAction(context?: any): void {
    console.log('📋 Executing create table action')
    this.dispatchEvent('voice-agent-create-table', {
      action: 'create_table',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeDeleteTableAction(context?: any): void {
    console.log('🗑️ Executing delete table action')
    this.dispatchEvent('voice-agent-delete-table', {
      action: 'delete_table',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeDataExportAction(context?: any): void {
    console.log('📤 Executing data export action')
    this.dispatchEvent('voice-agent-export-data', {
      action: 'export_data',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeCreateUserAction(context?: any): void {
    console.log('👤 Executing create user action')
    this.dispatchEvent('voice-agent-create-user', {
      action: 'create_user',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeDeleteUserAction(context?: any): void {
    console.log('🗑️ Executing delete user action')
    this.dispatchEvent('voice-agent-delete-user', {
      action: 'delete_user',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executePasswordChangeAction(context?: any): void {
    console.log('🔐 Executing password change action')
    this.dispatchEvent('voice-agent-change-password', {
      action: 'change_password',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeAddBusinessRuleAction(context?: any): void {
    console.log('📋 Executing add business rule action')
    this.dispatchEvent('voice-agent-add-business-rule', {
      action: 'add_business_rule',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeEditBusinessRuleAction(context?: any): void {
    console.log('✏️ Executing edit business rule action')
    this.dispatchEvent('voice-agent-edit-business-rule', {
      action: 'edit_business_rule',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeDeleteBusinessRuleAction(context?: any): void {
    console.log('🗑️ Executing delete business rule action')
    this.dispatchEvent('voice-agent-delete-business-rule', {
      action: 'delete_business_rule',
      context: context || 'voice_agent_triggered'
    })
  }

  private static executeGenericAction(actionDef: ButtonActionDefinition, context?: any): void {
    console.log('🖱️ Executing generic action:', actionDef.elementName)
    this.dispatchEvent('voice-agent-generic-action', {
      action: actionDef.elementName,
      actionType: actionDef.actionType,
      context: context || 'voice_agent_triggered'
    })
  }

  // Utility methods
  private static getPagePath(pageName: string): string | null {
    const pageMappings: Record<string, string> = {
      'dashboard': '/',
      'database query': '/database-query',
      'file query': '/file-query',
      'tables': '/tables',
      'users': '/users',
      'ai-results': '/ai-results',
      'company-structure': '/company-structure'
    }
    
    return pageMappings[pageName.toLowerCase()] || null
  }

  private static dispatchEvent(eventName: string, detail: any): void {
    if (typeof window !== 'undefined') {
      const event = new CustomEvent(eventName, {
        detail: { ...detail, timestamp: new Date().toISOString() }
      })
      window.dispatchEvent(event)
    }
  }

  private static showMessage(message: string, type: 'info' | 'warning' | 'error' = 'info'): void {
    this.dispatchEvent('voice-agent-show-message', {
      type,
      message,
      timestamp: new Date().toISOString()
    })
  }

  /**
   * Cleanup and reset the manager
   */
  static cleanup(): void {
    this.isInitialized = false
    this.registeredActions.clear()
    console.log('🖱️ ButtonActionManager cleaned up')
  }
} 