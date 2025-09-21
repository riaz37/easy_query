import { 
  NAVIGATION_ACTION_CONFIGS, 
  NavigationActionDefinition, 
  getAllNavigationActions,
  findNavigationAction,
  getNavigationActionsByCategory,
  getNavigationActionsByPage,
  getNavigationActionsForCurrentPage,
  searchNavigationActions
} from '../config/navigation-actions'

export class NavigationActionManager {
  private static isInitialized = false
  private static registeredActions = new Map<string, NavigationActionDefinition>()
  private static currentPage = 'dashboard'
  private static previousPages: string[] = []
  private static maxHistorySize = 10

  /**
   * Initialize the navigation action manager
   * This should be called once during app startup
   */
  static initialize(): void {
    if (this.isInitialized) {
      console.log('🧭 NavigationActionManager already initialized')
      return
    }

    console.log('🧭 Initializing NavigationActionManager...')
    
    // Register all navigation actions from configuration
    this.registerAllNavigationActions()
    
    // Set up automatic navigation detection
    this.setupAutomaticNavigationDetection()
    
    this.isInitialized = true
    console.log('🧭 NavigationActionManager initialized successfully')
  }

  /**
   * Register all navigation actions from the configuration
   */
  private static registerAllNavigationActions(): void {
    const allActions = getAllNavigationActions()
    
    allActions.forEach(action => {
      this.registerNavigationAction(action)
    })
    
    console.log(`🧭 Registered ${allActions.length} navigation actions`)
  }

  /**
   * Register a single navigation action
   */
  private static registerNavigationAction(actionDef: NavigationActionDefinition): void {
    const actionKey = actionDef.actionName.toLowerCase()
    
    if (this.registeredActions.has(actionKey)) {
      console.log(`🧭 Navigation action already registered: ${actionDef.actionName}`)
      return
    }

    // Register the main action
    this.registeredActions.set(actionKey, actionDef)

    // Register aliases if they exist
    if (actionDef.aliases) {
      actionDef.aliases.forEach(alias => {
        const aliasKey = alias.toLowerCase()
        if (!this.registeredActions.has(aliasKey)) {
          this.registeredActions.set(aliasKey, actionDef)
          console.log(`🧭 Registered navigation alias: ${alias} -> ${actionDef.actionName}`)
        }
      })
    }
  }

  /**
   * Execute a navigation action by name
   */
  static executeNavigationAction(actionName: string, context?: any): void {
    console.log('🧭 NavigationActionManager executing navigation action:', actionName, context)
    
    const normalizedName = actionName.toLowerCase()
    const actionDef = this.registeredActions.get(normalizedName)
    
    if (actionDef) {
      console.log(`🧭 Found registered navigation action: ${actionDef.actionName}`)
      this.executeRegisteredNavigationAction(actionDef, context)
    } else {
      console.warn(`🧭 No registered navigation action found for: ${actionName}`)
      this.executeGenericNavigationAction(actionName, context)
    }
  }

  /**
   * Execute a registered navigation action
   */
  private static executeRegisteredNavigationAction(actionDef: NavigationActionDefinition, context?: any): void {
    try {
      // Check conditions before executing
      if (!this.checkActionConditions(actionDef, context)) {
        console.warn(`🧭 Navigation action conditions not met: ${actionDef.actionName}`)
        return
      }

      // Execute the appropriate handler based on the action type
      switch (actionDef.handler) {
        case 'executePageNavigation':
          this.executePageNavigation(actionDef.targetPage, context)
          break
        case 'executeLogout':
          this.executeLogout(context)
          break
        case 'executeGoBack':
          this.executeGoBack(context)
          break
        case 'executePageRefresh':
          this.executePageRefresh(context)
          break
        default:
          console.warn(`🧭 Unknown navigation handler: ${actionDef.handler}`)
          this.executeGenericNavigationAction(actionDef.actionName, context)
      }
      
    } catch (error) {
      console.error(`🧭 Error executing navigation action: ${actionDef.actionName}`, error)
      this.showNavigationError(`Failed to execute navigation: ${actionDef.actionName}`)
    }
  }

  /**
   * Check if navigation action conditions are met
   */
  private static checkActionConditions(actionDef: NavigationActionDefinition, context?: any): boolean {
    const conditions = actionDef.conditions
    
    if (!conditions) return true

    // Check authentication requirement
    if (conditions.requiresAuth && !this.isUserAuthenticated()) {
      console.warn('🧭 Navigation requires authentication')
      return false
    }

    // Check database requirement
    if (conditions.requiresDatabase && !this.hasDatabaseAccess()) {
      console.warn('🧭 Navigation requires database access')
      return false
    }

    // Check permissions
    if (conditions.requiresPermissions && !this.hasRequiredPermissions(conditions.requiresPermissions)) {
      console.warn('🧭 Navigation requires specific permissions')
      return false
    }

    // Check page-specific conditions
    if (conditions.pageSpecific && !conditions.pageSpecific.includes(this.currentPage)) {
      console.warn('🧭 Navigation not available on current page')
      return false
    }

    return true
  }

  /**
   * Execute page navigation
   */
  private static executePageNavigation(targetPage: string, context?: any): void {
    console.log('🧭 Executing page navigation to:', targetPage)
    
    // Store current page in history before navigating
    this.addToHistory(this.currentPage)
    
    // Update current page
    this.currentPage = targetPage
    
    // Dispatch navigation event for VoiceNavigationHandler to handle
    if (typeof window !== 'undefined') {
      console.log('🧭 Dispatching voice-navigation event for page:', targetPage)
      const event = new CustomEvent('voice-navigation', {
        detail: {
          page: targetPage,
          previousPage: this.currentPage,
          type: 'page_navigation',
          context: context || 'voice_agent_triggered',
          timestamp: new Date().toISOString()
        }
      })
      window.dispatchEvent(event)
    }
  }

  /**
   * Execute logout
   */
  private static executeLogout(context?: any): void {
    console.log('🧭 Executing logout')
    
    // Dispatch logout event
    this.dispatchNavigationEvent('logout', {
      action: 'logout',
      context: context || 'voice_agent_triggered'
    })
    
    // Navigate to auth page
    this.executePageNavigation('auth', context)
  }

  /**
   * Execute go back navigation
   */
  private static executeGoBack(context?: any): void {
    console.log('🧭 Executing go back navigation')
    
    const steps = context?.steps || 1
    const targetPage = this.getPreviousPage(steps)
    
    if (targetPage) {
      console.log(`🧭 Going back ${steps} step(s) to: ${targetPage}`)
      this.executePageNavigation(targetPage, context)
    } else {
      console.warn('🧭 No previous page available')
      this.showNavigationError('No previous page available')
    }
  }

  /**
   * Execute page refresh
   */
  private static executePageRefresh(context?: any): void {
    console.log('🧭 Executing page refresh')
    
    if (typeof window !== 'undefined') {
      window.location.reload()
    }
  }

  /**
   * Execute generic navigation action
   */
  private static executeGenericNavigationAction(actionName: string, context?: any): void {
    console.log('🧭 Executing generic navigation action:', actionName)
    
    // Try to find a matching action by searching
    const matchingAction = searchNavigationActions(actionName)
    
    if (matchingAction.length > 0) {
      console.log('🧭 Found potential matches:', matchingAction.map(a => a.actionName))
      // Execute the first matching action
      this.executeRegisteredNavigationAction(matchingAction[0], context)
    } else {
      console.warn('🧭 No matching navigation action found')
      this.showNavigationError(`No navigation action found for: ${actionName}`)
    }
  }

  /**
   * Set current page (called when page changes)
   */
  static setCurrentPage(page: string): void {
    if (page !== this.currentPage) {
      this.addToHistory(this.currentPage)
      this.currentPage = page
      console.log('🧭 Current page updated to:', page)
    }
  }

  /**
   * Get current page
   */
  static getCurrentPage(): string {
    return this.currentPage
  }

  /**
   * Get previous page
   */
  static getPreviousPage(steps: number = 1): string | null {
    const index = this.previousPages.length - steps
    return index >= 0 ? this.previousPages[index] : null
  }

  /**
   * Get navigation history
   */
  static getNavigationHistory(): string[] {
    return [...this.previousPages]
  }

  /**
   * Add page to navigation history
   */
  private static addToHistory(page: string): void {
    if (page && page !== this.currentPage) {
      this.previousPages.push(page)
      
      // Limit history size
      if (this.previousPages.length > this.maxHistorySize) {
        this.previousPages.shift()
      }
      
      console.log('🧭 Added to navigation history:', page)
    }
  }

  /**
   * Get page path for navigation
   */
  private static getPagePath(pageName: string): string | null {
    const pageMappings: Record<string, string> = {
      'dashboard': '/',
      'database-query': '/database-query',
      'file-query': '/file-query',
      'tables': '/tables',
      'users': '/users',
      'ai-reports': '/ai-reports',
      'company-structure': '/company-structure',
      'user-configuration': '/user-configuration',
      'business-rules': '/business-rules',
      'auth': '/auth',
      'help': '/help'
    }
    
    return pageMappings[pageName.toLowerCase()] || null
  }

  /**
   * Set up automatic navigation detection
   */
  private static setupAutomaticNavigationDetection(): void {
    if (typeof window === 'undefined') return

    // Listen for route changes
    const handleRouteChange = () => {
      const path = window.location.pathname
      const page = path === '/' ? 'dashboard' : path.substring(1)
      this.setCurrentPage(page)
    }

    // Listen for popstate (browser back/forward buttons)
    window.addEventListener('popstate', handleRouteChange)
    
    // Listen for hash changes
    window.addEventListener('hashchange', handleRouteChange)
    
    // Initial page detection
    handleRouteChange()
    
    console.log('🧭 Automatic navigation detection enabled')
  }

  // Utility methods for condition checking
  private static isUserAuthenticated(): boolean {
    // This would check the actual auth state
    // For now, return true as a placeholder
    return true
  }

  private static hasDatabaseAccess(): boolean {
    // This would check if user has database access
    // For now, return true as a placeholder
    return true
  }

  private static hasRequiredPermissions(permissions: string[]): boolean {
    // This would check user permissions
    // For now, return true as a placeholder
    return true
  }

  // Event dispatching
  private static dispatchNavigationEvent(eventName: string, detail: any): void {
    if (typeof window !== 'undefined') {
      const event = new CustomEvent(`voice-agent-${eventName}`, {
        detail: { ...detail, timestamp: new Date().toISOString() }
      })
      window.dispatchEvent(event)
    }
  }

  // Error handling
  private static showNavigationError(message: string): void {
    this.dispatchNavigationEvent('navigation-error', {
      type: 'error',
      message,
      timestamp: new Date().toISOString()
    })
  }

  // Public API methods
  static getAvailableActions(): NavigationActionDefinition[] {
    return getAllNavigationActions()
  }

  static getActionsByCategory(categoryName: string): NavigationActionDefinition[] {
    return getNavigationActionsByCategory(categoryName)
  }

  static getActionsByPage(targetPage: string): NavigationActionDefinition[] {
    return getNavigationActionsByPage(targetPage)
  }

  static getActionsForCurrentPage(): NavigationActionDefinition[] {
    return getNavigationActionsForCurrentPage(this.currentPage)
  }

  static searchActions(searchTerm: string): NavigationActionDefinition[] {
    return searchNavigationActions(searchTerm)
  }

  static isActionAvailable(actionName: string): boolean {
    return !!findNavigationAction(actionName)
  }

  static getActionInfo(actionName: string): NavigationActionDefinition | undefined {
    return findNavigationAction(actionName)
  }

  /**
   * Cleanup and reset the manager
   */
  static cleanup(): void {
    this.isInitialized = false
    this.registeredActions.clear()
    this.previousPages = []
    console.log('🧭 NavigationActionManager cleaned up')
  }
} 