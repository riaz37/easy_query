import { useEffect, useCallback } from 'react'
import { NavigationActionManager } from '@/lib/voice-agent/services/NavigationActionManager'
import { NavigationActionDefinition } from '@/lib/voice-agent/config/navigation-actions'

export function useNavigationActionManager() {
  // Initialize the NavigationActionManager on mount
  useEffect(() => {
    console.log('🧭 Initializing NavigationActionManager in useNavigationActionManager hook')
    NavigationActionManager.initialize()

    // Cleanup on unmount
    return () => {
      NavigationActionManager.cleanup()
    }
  }, [])

  // Core navigation functionality
  const executeNavigationAction = useCallback((actionName: string, context?: any): void => {
    NavigationActionManager.executeNavigationAction(actionName, context)
  }, [])

  const setCurrentPage = useCallback((page: string): void => {
    NavigationActionManager.setCurrentPage(page)
  }, [])

  const getCurrentPage = useCallback((): string => {
    return NavigationActionManager.getCurrentPage()
  }, [])

  const getPreviousPage = useCallback((steps: number = 1): string | null => {
    return NavigationActionManager.getPreviousPage(steps)
  }, [])

  const getNavigationHistory = useCallback((): string[] => {
    return NavigationActionManager.getNavigationHistory()
  }, [])

  // Action management
  const getAvailableActions = useCallback((): NavigationActionDefinition[] => {
    return NavigationActionManager.getAvailableActions()
  }, [])

  const getActionsByCategory = useCallback((categoryName: string): NavigationActionDefinition[] => {
    return NavigationActionManager.getActionsByCategory(categoryName)
  }, [])

  const getActionsByPage = useCallback((targetPage: string): NavigationActionDefinition[] => {
    return NavigationActionManager.getActionsByPage(targetPage)
  }, [])

  const getActionsForCurrentPage = useCallback((): NavigationActionDefinition[] => {
    return NavigationActionManager.getActionsForCurrentPage()
  }, [])

  const searchActions = useCallback((searchTerm: string): NavigationActionDefinition[] => {
    return NavigationActionManager.searchActions(searchTerm)
  }, [])

  const isActionAvailable = useCallback((actionName: string): boolean => {
    return NavigationActionManager.isActionAvailable(actionName)
  }, [])

  const getActionInfo = useCallback((actionName: string): NavigationActionDefinition | undefined => {
    return NavigationActionManager.getActionInfo(actionName)
  }, [])

  // Advanced functionality
  const getActionsByType = useCallback((actionType: string): NavigationActionDefinition[] => {
    const allActions = getAvailableActions()
    return allActions.filter(action => action.actionType === actionType)
  }, [getAvailableActions])

  const getActionsByContext = useCallback((context: string): NavigationActionDefinition[] => {
    const allActions = getAvailableActions()
    return allActions.filter(action => action.context === context)
  }, [getAvailableActions])

  const getActionsByPermission = useCallback((permission: string): NavigationActionDefinition[] => {
    const allActions = getAvailableActions()
    return allActions.filter(action => 
      action.conditions?.requiresPermissions?.includes(permission)
    )
  }, [getAvailableActions])

  const getPublicActions = useCallback((): NavigationActionDefinition[] => {
    const allActions = getAvailableActions()
    return allActions.filter(action => action.conditions?.isPublic === true)
  }, [getAvailableActions])

  const getAuthenticatedActions = useCallback((): NavigationActionDefinition[] => {
    const allActions = getAvailableActions()
    return allActions.filter(action => action.conditions?.requiresAuth === true)
  }, [getAvailableActions])

  // Utility functions
  const getTotalActionCount = useCallback((): number => {
    return getAvailableActions().length
  }, [getAvailableActions])

  const getCategoryCount = useCallback((): number => {
    const categories = new Set<string>()
    getAvailableActions().forEach(action => {
      if (action.metadata?.category) {
        categories.add(action.metadata.category)
      }
    })
    return categories.size
  }, [getAvailableActions])

  const getAvailableTargetPages = useCallback((): string[] => {
    const pages = new Set<string>()
    getAvailableActions().forEach(action => pages.add(action.targetPage))
    return Array.from(pages)
  }, [getAvailableActions])

  const getAvailableContexts = useCallback((): string[] => {
    const contexts = new Set<string>()
    getAvailableActions().forEach(action => contexts.add(action.context))
    return Array.from(contexts)
  }, [getAvailableActions])

  // Quick navigation helpers
  const goToDashboard = useCallback((context?: any): void => {
    executeNavigationAction('go to dashboard', context)
  }, [executeNavigationAction])

  const goToDatabaseQuery = useCallback((context?: any): void => {
    executeNavigationAction('go to database query', context)
  }, [executeNavigationAction])

  const goToFileQuery = useCallback((context?: any): void => {
    executeNavigationAction('go to file query', context)
  }, [executeNavigationAction])

  const goToTables = useCallback((context?: any): void => {
    executeNavigationAction('go to tables', context)
  }, [executeNavigationAction])

  const goToUsers = useCallback((context?: any): void => {
    executeNavigationAction('go to users', context)
  }, [executeNavigationAction])

  const goToAIResults = useCallback((context?: any): void => {
    executeNavigationAction('go to ai results', context)
  }, [executeNavigationAction])

  const goBack = useCallback((steps: number = 1, context?: any): void => {
    executeNavigationAction('go back', { ...context, steps })
  }, [executeNavigationAction])

  const refreshPage = useCallback((context?: any): void => {
    executeNavigationAction('refresh page', context)
  }, [executeNavigationAction])

  const logout = useCallback((context?: any): void => {
    executeNavigationAction('logout', context)
  }, [executeNavigationAction])

  return {
    // Core navigation
    executeNavigationAction,
    setCurrentPage,
    getCurrentPage,
    getPreviousPage,
    getNavigationHistory,
    
    // Action management
    getAvailableActions,
    getActionsByCategory,
    getActionsByPage,
    getActionsForCurrentPage,
    searchActions,
    isActionAvailable,
    getActionInfo,
    
    // Advanced functionality
    getActionsByType,
    getActionsByContext,
    getActionsByPermission,
    getPublicActions,
    getAuthenticatedActions,
    
    // Utility functions
    getTotalActionCount,
    getCategoryCount,
    getAvailableTargetPages,
    getAvailableContexts,
    
    // Quick navigation helpers
    goToDashboard,
    goToDatabaseQuery,
    goToFileQuery,
    goToTables,
    goToUsers,
    goToAIResults,
    goBack,
    refreshPage,
    logout
  }
} 