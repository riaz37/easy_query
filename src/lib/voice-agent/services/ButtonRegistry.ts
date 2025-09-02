/**
 * ButtonRegistry - Scalable button registration and execution system
 *
 * This system allows:
 * 1. Backend to decide WHEN to execute button actions (source of truth)
 * 2. Frontend to register HOW to execute button actions
 * 3. Easy scalability - register any number of button actions
 * 4. Consistent execution across voice and text systems
 */

export interface ButtonAction {
  id: string
  name: string
  aliases?: string[]
  description?: string
  handler: (context?: any) => void | Promise<void>
  selector?: string
  validation?: (context?: any) => boolean
  category?: string
}

export interface ButtonExecutionContext {
  elementName: string
  page?: string
  previousPage?: string
  context?: any
  source: 'voice' | 'text'
  timestamp: string
}

export interface ButtonExecutionResult {
  success: boolean
  error?: string
  elementName: string
  executionTime: number
  context?: any
}

export class ButtonRegistry {
  private static instance: ButtonRegistry | null = null
  private actions = new Map<string, ButtonAction>()
  private aliases = new Map<string, string>() // alias -> actionId mapping
  private categories = new Map<string, Set<string>>() // category -> Set of actionIds
  private executionHistory: ButtonExecutionResult[] = []
  private maxHistorySize = 100

  // Event handlers
  onButtonExecuted?: (result: ButtonExecutionResult) => void
  onButtonRegistered?: (action: ButtonAction) => void
  onButtonUnregistered?: (actionId: string) => void

  private constructor() {}

  static getInstance(): ButtonRegistry {
    if (!this.instance) {
      this.instance = new ButtonRegistry()
    }
    return this.instance
  }

  /**
   * Register a button action
   */
  register(action: ButtonAction): void {
    const normalizedId = this.normalizeId(action.id)

    // Store the action
    this.actions.set(normalizedId, action)

    // Register aliases
    if (action.aliases) {
      action.aliases.forEach(alias => {
        this.aliases.set(this.normalizeId(alias), normalizedId)
      })
    }

    // Register category
    if (action.category) {
      if (!this.categories.has(action.category)) {
        this.categories.set(action.category, new Set())
      }
      this.categories.get(action.category)!.add(normalizedId)
    }

    console.log(`🎯 ButtonRegistry: Registered action '${action.name}' with id '${normalizedId}'`)
    this.onButtonRegistered?.(action)
  }

  /**
   * Register multiple actions at once
   */
  registerMany(actions: ButtonAction[]): void {
    actions.forEach(action => this.register(action))
  }

  /**
   * Unregister a button action
   */
  unregister(actionId: string): boolean {
    const normalizedId = this.normalizeId(actionId)
    const action = this.actions.get(normalizedId)

    if (!action) {
      return false
    }

    // Remove from actions
    this.actions.delete(normalizedId)

    // Remove aliases
    if (action.aliases) {
      action.aliases.forEach(alias => {
        this.aliases.delete(this.normalizeId(alias))
      })
    }

    // Remove from categories
    if (action.category) {
      const categorySet = this.categories.get(action.category)
      if (categorySet) {
        categorySet.delete(normalizedId)
        if (categorySet.size === 0) {
          this.categories.delete(action.category)
        }
      }
    }

    console.log(`🎯 ButtonRegistry: Unregistered action '${normalizedId}'`)
    this.onButtonUnregistered?.(normalizedId)
    return true
  }

  /**
   * Execute a button action when commanded by backend
   */
  async execute(elementName: string, context: ButtonExecutionContext): Promise<ButtonExecutionResult> {
    const startTime = Date.now()
    const result: ButtonExecutionResult = {
      success: false,
      elementName,
      executionTime: 0,
      context: context.context
    }

    try {
      // Find the action
      const action = this.findAction(elementName)

      if (!action) {
        result.error = `No registered action found for '${elementName}'`
        console.warn(`🎯 ButtonRegistry: ${result.error}`)
        return result
      }

      // Validate if needed
      if (action.validation && !action.validation(context.context)) {
        result.error = `Validation failed for action '${elementName}'`
        console.warn(`🎯 ButtonRegistry: ${result.error}`)
        return result
      }

      console.log(`🎯 ButtonRegistry: Executing action '${action.name}' for '${elementName}'`)

      // Execute the action
      await action.handler(context.context)

      result.success = true
      result.executionTime = Date.now() - startTime

      console.log(`🎯 ButtonRegistry: Successfully executed '${elementName}' in ${result.executionTime}ms`)

    } catch (error) {
      result.error = error instanceof Error ? error.message : 'Unknown error'
      result.executionTime = Date.now() - startTime
      console.error(`🎯 ButtonRegistry: Error executing '${elementName}':`, error)
    }

    // Store in history
    this.addToHistory(result)

    // Notify listeners
    this.onButtonExecuted?.(result)

    return result
  }

  /**
   * Find a registered action by name or alias
   */
  private findAction(elementName: string): ButtonAction | null {
    const normalizedName = this.normalizeId(elementName)

    // Try direct match first
    let action = this.actions.get(normalizedName)
    if (action) return action

    // Try alias match
    const aliasTarget = this.aliases.get(normalizedName)
    if (aliasTarget) {
      action = this.actions.get(aliasTarget)
      if (action) return action
    }

    // Try partial matches (fuzzy search)
    for (const [actionId, registeredAction] of this.actions.entries()) {
      if (this.isPartialMatch(normalizedName, actionId) ||
          this.isPartialMatch(normalizedName, registeredAction.name)) {
        return registeredAction
      }

      // Check aliases for partial matches
      if (registeredAction.aliases) {
        for (const alias of registeredAction.aliases) {
          if (this.isPartialMatch(normalizedName, alias)) {
            return registeredAction
          }
        }
      }
    }

    return null
  }

  /**
   * Check if two strings partially match (fuzzy matching)
   */
  private isPartialMatch(search: string, target: string): boolean {
    const normalizedTarget = this.normalizeId(target)
    return normalizedTarget.includes(search) || search.includes(normalizedTarget)
  }

  /**
   * Normalize string for consistent matching
   */
  private normalizeId(str: string): string {
    return str.toLowerCase().trim().replace(/[^a-z0-9]/g, '')
  }

  /**
   * Get all registered actions
   */
  getActions(): ButtonAction[] {
    return Array.from(this.actions.values())
  }

  /**
   * Get actions by category
   */
  getActionsByCategory(category: string): ButtonAction[] {
    const actionIds = this.categories.get(category)
    if (!actionIds) return []

    return Array.from(actionIds)
      .map(id => this.actions.get(id))
      .filter((action): action is ButtonAction => action !== undefined)
  }

  /**
   * Get all categories
   */
  getCategories(): string[] {
    return Array.from(this.categories.keys())
  }

  /**
   * Check if an action is registered
   */
  isRegistered(elementName: string): boolean {
    return this.findAction(elementName) !== null
  }

  /**
   * Get execution history
   */
  getExecutionHistory(): ButtonExecutionResult[] {
    return [...this.executionHistory]
  }

  /**
   * Clear execution history
   */
  clearHistory(): void {
    this.executionHistory = []
  }

  /**
   * Add result to execution history
   */
  private addToHistory(result: ButtonExecutionResult): void {
    this.executionHistory.push(result)

    // Keep history size manageable
    if (this.executionHistory.length > this.maxHistorySize) {
      this.executionHistory = this.executionHistory.slice(-this.maxHistorySize)
    }
  }

  /**
   * Get statistics about registered actions
   */
  getStats(): {
    totalActions: number
    totalAliases: number
    totalCategories: number
    successfulExecutions: number
    failedExecutions: number
    averageExecutionTime: number
  } {
    const successful = this.executionHistory.filter(r => r.success)
    const failed = this.executionHistory.filter(r => !r.success)
    const avgTime = this.executionHistory.length > 0
      ? this.executionHistory.reduce((sum, r) => sum + r.executionTime, 0) / this.executionHistory.length
      : 0

    return {
      totalActions: this.actions.size,
      totalAliases: this.aliases.size,
      totalCategories: this.categories.size,
      successfulExecutions: successful.length,
      failedExecutions: failed.length,
      averageExecutionTime: Math.round(avgTime)
    }
  }

  /**
   * Reset the registry (useful for testing)
   */
  reset(): void {
    this.actions.clear()
    this.aliases.clear()
    this.categories.clear()
    this.executionHistory = []
    console.log('🎯 ButtonRegistry: Reset complete')
  }

  /**
   * Export configuration for debugging
   */
  exportConfig(): {
    actions: Array<{id: string, name: string, aliases?: string[], category?: string}>
    aliases: Array<{alias: string, target: string}>
    categories: Array<{category: string, actionCount: number}>
  } {
    return {
      actions: Array.from(this.actions.values()).map(action => ({
        id: action.id,
        name: action.name,
        aliases: action.aliases,
        category: action.category
      })),
      aliases: Array.from(this.aliases.entries()).map(([alias, target]) => ({
        alias,
        target
      })),
      categories: Array.from(this.categories.entries()).map(([category, actionIds]) => ({
        category,
        actionCount: actionIds.size
      }))
    }
  }
}

// Export singleton instance
export const buttonRegistry = ButtonRegistry.getInstance()
