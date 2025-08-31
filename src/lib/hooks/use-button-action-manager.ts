import { useEffect, useCallback } from 'react'
import { ButtonActionManager } from '@/lib/voice-agent/services/ButtonActionManager'
import { ButtonActionDefinition } from '@/lib/voice-agent/config/button-actions'

export function useButtonActionManager() {
  // Initialize the ButtonActionManager on mount
  useEffect(() => {
    console.log('🖱️ Initializing ButtonActionManager in useButtonActionManager hook')
    ButtonActionManager.initialize()

    // Cleanup on unmount
    return () => {
      ButtonActionManager.cleanup()
    }
  }, [])

  // Get all available button actions
  const getAvailableActions = useCallback((): ButtonActionDefinition[] => {
    return ButtonActionManager.getAvailableActions()
  }, [])

  // Get button actions by category
  const getActionsByCategory = useCallback((categoryName: string): ButtonActionDefinition[] => {
    return ButtonActionManager.getActionsByCategory(categoryName)
  }, [])

  // Check if a button action is available
  const isActionAvailable = useCallback((actionName: string): boolean => {
    return ButtonActionManager.isActionAvailable(actionName)
  }, [])

  // Get action information
  const getActionInfo = useCallback((actionName: string): ButtonActionDefinition | undefined => {
    return ButtonActionManager.getActionInfo(actionName)
  }, [])

  // Get actions by action type
  const getActionsByType = useCallback((actionType: string): ButtonActionDefinition[] => {
    const allActions = getAvailableActions()
    return allActions.filter(action => action.actionType === actionType)
  }, [getAvailableActions])

  // Search actions by text
  const searchActions = useCallback((searchTerm: string): ButtonActionDefinition[] => {
    const allActions = getAvailableActions()
    const normalizedSearch = searchTerm.toLowerCase()
    
    return allActions.filter(action => 
      action.elementName.toLowerCase().includes(normalizedSearch) ||
      action.description.toLowerCase().includes(normalizedSearch) ||
      action.aliases?.some(alias => alias.toLowerCase().includes(normalizedSearch))
    )
  }, [getAvailableActions])

  // Get action categories
  const getActionCategories = useCallback(() => {
    const allActions = getAvailableActions()
    const categories = new Set<string>()
    
    allActions.forEach(action => {
      if (action.context) {
        categories.add(action.context)
      }
    })
    
    return Array.from(categories)
  }, [getAvailableActions])

  return {
    // Core functionality
    getAvailableActions,
    getActionsByCategory,
    isActionAvailable,
    getActionInfo,
    
    // Advanced functionality
    getActionsByType,
    searchActions,
    getActionCategories,
    
    // Utility functions
    getTotalActionCount: () => getAvailableActions().length,
    getCategoryCount: () => getActionCategories().length
  }
} 