import { InteractionType } from '../types'

// Define navigation action configuration interface
export interface NavigationActionDefinition {
  actionName: string
  actionType: InteractionType
  targetPage: string
  handler: string // Reference to handler function name
  context: string
  description: string
  aliases?: string[] // Alternative names for the same navigation action
  parameters?: {
    required?: string[]
    optional?: string[]
    defaultValues?: Record<string, any>
  }
  conditions?: {
    requiresAuth?: boolean
    requiresDatabase?: boolean
    requiresPermissions?: string[]
    pageSpecific?: string[] // Only available on specific pages
  }
  metadata?: {
    icon?: string
    category?: string
    priority?: number
    isPublic?: boolean
  }
}

// Define navigation action categories
export interface NavigationActionCategory {
  name: string
  description: string
  actions: NavigationActionDefinition[]
}

// Navigation action configurations organized by category
export const NAVIGATION_ACTION_CONFIGS: NavigationActionCategory[] = [
  {
    name: 'Core Navigation',
    description: 'Primary page navigation actions',
    actions: [
      {
        actionName: 'go to dashboard',
        actionType: 'page_navigation',
        targetPage: 'dashboard',
        handler: 'executePageNavigation',
        context: 'core_navigation',
        description: 'Navigate to the main dashboard page',
        aliases: ['dashboard', 'home', 'main page', 'start', 'landing page', 'go home'],
        parameters: {
          optional: ['section']
        },
        conditions: {
          requiresAuth: true,
          isPublic: true
        },
        metadata: {
          icon: 'home',
          category: 'core',
          priority: 1
        }
      },
      {
        actionName: 'go to database-query',
        actionType: 'page_navigation',
        targetPage: 'database-query',
        handler: 'executePageNavigation',
        context: 'core_navigation',
        description: 'Navigate to the database query page',
        aliases: ['database query', 'database', 'sql', 'query', 'db query', 'database search', 'go to database', 'database-query'],
        parameters: {
          optional: ['query', 'mode']
        },
        conditions: {
          requiresAuth: true,
          requiresDatabase: true
        },
        metadata: {
          icon: 'database',
          category: 'core',
          priority: 2
        }
      },
      {
        actionName: 'go to file-query',
        actionType: 'page_navigation',
        targetPage: 'file-query',
        handler: 'executePageNavigation',
        context: 'core_navigation',
        description: 'Navigate to the file query and upload page',
        aliases: ['file query', 'file', 'files', 'file search', 'file upload', 'file management', 'go to files', 'file-query'],
        parameters: {
          optional: ['search', 'upload']
        },
        conditions: {
          requiresAuth: true
        },
        metadata: {
          icon: 'file',
          category: 'core',
          priority: 3
        }
      }
    ]
  },
  {
    name: 'Data Management',
    description: 'Data and table management navigation',
    actions: [
      {
        actionName: 'go to tables',
        actionType: 'page_navigation',
        targetPage: 'tables',
        handler: 'executePageNavigation',
        context: 'data_management',
        description: 'Navigate to the table management page',
        aliases: ['tables', 'table management', 'manage tables', 'go to tables', 'table view'],
        parameters: {
          optional: ['action', 'tableName']
        },
        conditions: {
          requiresAuth: true,
          requiresDatabase: true
        },
        metadata: {
          icon: 'table',
          category: 'data',
          priority: 4
        }
      },
      {
        actionName: 'go to ai-reports',
        actionType: 'page_navigation',
        targetPage: 'ai-reports',
        handler: 'executePageNavigation',
        context: 'data_management',
        description: 'Navigate to the AI analysis reports page',
        aliases: ['ai reports', 'reports', 'ai analysis', 'view reports', 'go to reports', 'ai-reports'],
        parameters: {
          optional: ['reportId', 'filter']
        },
        conditions: {
          requiresAuth: true
        },
        metadata: {
          icon: 'chart',
          category: 'data',
          priority: 5
        }
      }
    ]
  },
  {
    name: 'User Management',
    description: 'User account and access management navigation',
    actions: [
      {
        actionName: 'go to users',
        actionType: 'page_navigation',
        targetPage: 'users',
        handler: 'executePageNavigation',
        context: 'user_management',
        description: 'Navigate to the user management page',
        aliases: ['users', 'user management', 'manage users', 'go to users', 'user view'],
        parameters: {
          optional: ['action', 'userId']
        },
        conditions: {
          requiresAuth: true,
          requiresPermissions: ['user_management']
        },
        metadata: {
          icon: 'users',
          category: 'admin',
          priority: 6
        }
      },
      {
        actionName: 'go to user-configuration',
        actionType: 'page_navigation',
        targetPage: 'user-configuration',
        handler: 'executePageNavigation',
        context: 'user_management',
        description: 'Navigate to the user configuration and settings page',
        aliases: ['user configuration', 'user config', 'settings', 'user settings', 'go to settings', 'user-configuration'],
        parameters: {
          optional: ['section']
        },
        conditions: {
          requiresAuth: true
        },
        metadata: {
          icon: 'settings',
          category: 'admin',
          priority: 7
        }
      }
    ]
  },
  {
    name: 'Business Logic',
    description: 'Business rules and company structure navigation',
    actions: [
      {
        actionName: 'go to business-rules',
        actionType: 'page_navigation',
        targetPage: 'business-rules',
        handler: 'executePageNavigation',
        context: 'business_logic',
        description: 'Navigate to the business rules management page',
        aliases: ['business rules', 'rules', 'business logic', 'go to rules', 'business-rules'],
        parameters: {
          optional: ['action', 'ruleId']
        },
        conditions: {
          requiresAuth: true,
          requiresPermissions: ['business_rules']
        },
        metadata: {
          icon: 'rules',
          category: 'business',
          priority: 8
        }
      },
      {
        actionName: 'go to company-structure',
        actionType: 'page_navigation',
        targetPage: 'company-structure',
        handler: 'executePageNavigation',
        context: 'company_management',
        description: 'Navigate to the company structure management page',
        aliases: ['company structure', 'company', 'structure', 'company management', 'go to company', 'company-structure'],
        parameters: {
          optional: ['view', 'companyId']
        },
        conditions: {
          requiresAuth: true,
          requiresPermissions: ['company_structure']
        },
        metadata: {
          icon: 'organization',
          category: 'business',
          priority: 9
        }
      }
    ]
  },
  {
    name: 'Authentication',
    description: 'Login, logout, and authentication navigation',
    actions: [
      {
        actionName: 'go to login',
        actionType: 'page_navigation',
        targetPage: 'auth',
        handler: 'executePageNavigation',
        context: 'authentication',
        description: 'Navigate to the login page',
        aliases: ['login', 'sign in', 'authentication', 'go to login', 'signin'],
        parameters: {
          optional: ['redirect']
        },
        conditions: {
          requiresAuth: false,
          isPublic: true
        },
        metadata: {
          icon: 'login',
          category: 'auth',
          priority: 10
        }
      },
      {
        actionName: 'logout',
        actionType: 'logout',
        targetPage: 'auth',
        handler: 'executeLogout',
        context: 'authentication',
        description: 'Log out of the current session',
        aliases: ['sign out', 'signout', 'log out', 'end session', 'disconnect'],
        parameters: {},
        conditions: {
          requiresAuth: true
        },
        metadata: {
          icon: 'logout',
          category: 'auth',
          priority: 11
        }
      }
    ]
  },
  {
    name: 'Dynamic Navigation',
    description: 'Context-aware and dynamic navigation actions',
    actions: [
      {
        actionName: 'go back',
        actionType: 'navigation_back',
        targetPage: 'previous',
        handler: 'executeGoBack',
        context: 'dynamic_navigation',
        description: 'Navigate back to the previous page',
        aliases: ['back', 'previous page', 'go back', 'return', 'navigate back'],
        parameters: {
          optional: ['steps']
        },
        conditions: {
          requiresAuth: true
        },
        metadata: {
          icon: 'arrow-left',
          category: 'dynamic',
          priority: 12
        }
      },
      {
        actionName: 'refresh page',
        actionType: 'page_refresh',
        targetPage: 'current',
        handler: 'executePageRefresh',
        context: 'dynamic_navigation',
        description: 'Refresh the current page',
        aliases: ['refresh', 'reload', 'reload page', 'refresh current page'],
        parameters: {},
        conditions: {
          requiresAuth: true
        },
        metadata: {
          icon: 'refresh',
          category: 'dynamic',
          priority: 13
        }
      },
      {
        actionName: 'go to help',
        actionType: 'page_navigation',
        targetPage: 'help',
        handler: 'executePageNavigation',
        context: 'dynamic_navigation',
        description: 'Navigate to help and documentation',
        aliases: ['help', 'documentation', 'support', 'go to help', 'docs'],
        parameters: {
          optional: ['topic', 'section']
        },
        conditions: {
          requiresAuth: false,
          isPublic: true
        },
        metadata: {
          icon: 'help',
          category: 'dynamic',
          priority: 14
        }
      }
    ]
  }
]

// Helper function to get all navigation actions flattened
export const getAllNavigationActions = (): NavigationActionDefinition[] => {
  return NAVIGATION_ACTION_CONFIGS.flatMap(category => category.actions)
}

// Helper function to find navigation action by name
export const findNavigationAction = (actionName: string): NavigationActionDefinition | undefined => {
  const normalizedName = actionName.toLowerCase()
  return getAllNavigationActions().find(action => 
    action.actionName.toLowerCase() === normalizedName ||
    action.aliases?.some(alias => alias.toLowerCase() === normalizedName)
  )
}

// Helper function to get navigation actions by category
export const getNavigationActionsByCategory = (categoryName: string): NavigationActionDefinition[] => {
  const category = NAVIGATION_ACTION_CONFIGS.find(cat => cat.name.toLowerCase() === categoryName.toLowerCase())
  return category?.actions || []
}

// Helper function to get navigation actions by target page
export const getNavigationActionsByPage = (targetPage: string): NavigationActionDefinition[] => {
  return getAllNavigationActions().filter(action => action.targetPage === targetPage)
}

// Helper function to get navigation actions available on current page
export const getNavigationActionsForCurrentPage = (currentPage: string): NavigationActionDefinition[] => {
  return getAllNavigationActions().filter(action => 
    !action.conditions?.pageSpecific || 
    action.conditions.pageSpecific?.includes(currentPage)
  )
}

// Helper function to get all available target pages
export const getAvailableTargetPages = (): string[] => {
  const pages = new Set<string>()
  getAllNavigationActions().forEach(action => pages.add(action.targetPage))
  return Array.from(pages)
}

// Helper function to validate navigation action configuration
export const validateNavigationActionConfig = (config: NavigationActionDefinition): boolean => {
  return !!(
    config.actionName &&
    config.actionType &&
    config.targetPage &&
    config.handler &&
    config.context &&
    config.description
  )
}

// Helper function to get navigation actions by action type
export const getNavigationActionsByType = (actionType: InteractionType): NavigationActionDefinition[] => {
  return getAllNavigationActions().filter(action => action.actionType === actionType)
}

// Helper function to search navigation actions by text
export const searchNavigationActions = (searchTerm: string): NavigationActionDefinition[] => {
  const normalizedSearch = searchTerm.toLowerCase()
  return getAllNavigationActions().filter(action => 
    action.actionName.toLowerCase().includes(normalizedSearch) ||
    action.description.toLowerCase().includes(normalizedSearch) ||
    action.aliases?.some(alias => alias.toLowerCase().includes(normalizedSearch)) ||
    action.targetPage.toLowerCase().includes(normalizedSearch)
  )
}

// Export default configuration for easy import
export default NAVIGATION_ACTION_CONFIGS 