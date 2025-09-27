import { useCallback } from 'react'
import { BUTTON_MAPPING_CONFIG, getButtonMapping, getButtonMappingsForPage } from '@/lib/voice-agent/config/default-button-actions'

export function useButtonActionManager() {
  // Get all available button mappings
  const getAvailableActions = useCallback(() => {
    return Object.entries(BUTTON_MAPPING_CONFIG).map(([key, mapping]) => ({
      id: mapping.id,
      name: mapping.name,
      elementName: key,
      description: mapping.description,
      page: mapping.page,
      selectors: mapping.selectors,
      aliases: mapping.aliases,
      category: 'mapped'
    }))
  }, [])

  // Get button actions by page
  const getActionsByPage = useCallback((pageName: string) => {
    return getButtonMappingsForPage(pageName)
  }, [])

  // Check if a button action is available
  const isActionAvailable = useCallback((elementName: string): boolean => {
    return !!getButtonMapping(elementName)
  }, [])

  // Get action information
  const getActionInfo = useCallback((elementName: string) => {
    return getButtonMapping(elementName)
  }, [])

  // Get actions by page
  const getActionsByType = useCallback((pageName: string) => {
    return getButtonMappingsForPage(pageName)
  }, [])

  // Search actions by text
  const searchActions = useCallback((searchTerm: string) => {
    const allActions = getAvailableActions()
    const normalizedSearch = searchTerm.toLowerCase()
    
    return allActions.filter(action => 
      action.name.toLowerCase().includes(normalizedSearch) ||
      action.description?.toLowerCase().includes(normalizedSearch) ||
      action.aliases?.some(alias => alias.toLowerCase().includes(normalizedSearch)) ||
      action.elementName.toLowerCase().includes(normalizedSearch)
    )
  }, [getAvailableActions])

  // Get available pages
  const getActionCategories = useCallback(() => {
    const pages = new Set<string>()
    Object.values(BUTTON_MAPPING_CONFIG).forEach(mapping => {
      pages.add(mapping.page)
    })
    return Array.from(pages)
  }, [])

  // Get mapping configuration
  const getMappingConfig = useCallback(() => {
    return BUTTON_MAPPING_CONFIG
  }, [])

  return {
    // Core functionality
    getAvailableActions,
    getActionsByPage,
    isActionAvailable,
    getActionInfo,
    
    // Advanced functionality
    getActionsByType,
    searchActions,
    getActionCategories,
    getMappingConfig,
    
    // Utility functions
    getTotalActionCount: () => getAvailableActions().length,
    getCategoryCount: () => getActionCategories().length,
    getPageCount: () => getActionCategories().length
  }
} 