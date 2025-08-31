import { InteractionType } from '../types'

// Define button action configuration interface
export interface ButtonActionDefinition {
  elementName: string
  actionType: InteractionType
  handler: string // Reference to handler function name
  context: string
  description: string
  customParams?: Record<string, any>
  aliases?: string[] // Alternative names for the same action
}

// Define button action categories
export interface ButtonActionCategory {
  name: string
  description: string
  actions: ButtonActionDefinition[]
}

// Button action configurations organized by category
export const BUTTON_ACTION_CONFIGS: ButtonActionCategory[] = [
  {
    name: 'Reports',
    description: 'Report generation and viewing actions',
    actions: [
      {
        elementName: 'view report',
        actionType: 'view_report',
        handler: 'executeViewReportAction',
        context: 'view_report',
        description: 'Navigate to view generated reports',
        aliases: ['view reports', 'show report', 'show reports', 'reports']
      },
      {
        elementName: 'report generation',
        actionType: 'generate_report',
        handler: 'executeReportGenerationAction',
        context: 'report_generation',
        description: 'Start the report generation process',
        aliases: ['generate report', 'create report', 'new report', 'start report']
      },
      {
        elementName: 'generate report',
        actionType: 'generate_report',
        handler: 'executeReportGenerationAction',
        context: 'report_generation',
        description: 'Start the report generation process',
        aliases: ['report generation', 'create report', 'new report', 'start report']
      }
    ]
  },
  {
    name: 'File Operations',
    description: 'File upload and search actions',
    actions: [
      {
        elementName: 'upload',
        actionType: 'file_upload',
        handler: 'executeFileUploadAction',
        context: 'file_upload',
        description: 'Trigger file upload process',
        aliases: ['upload file', 'upload files', 'add file', 'add files', 'import file']
      },
      {
        elementName: 'search',
        actionType: 'file_search',
        handler: 'executeFileSearchAction',
        context: 'file_search',
        description: 'Search through file system',
        aliases: ['search files', 'find file', 'find files', 'file search', 'search file system']
      }
    ]
  },
  {
    name: 'Navigation',
    description: 'Page navigation and routing actions',
    actions: [
      {
        elementName: 'dashboard',
        actionType: 'page_navigation',
        handler: 'executeNavigationAction',
        context: 'navigation',
        description: 'Navigate to dashboard page',
        aliases: ['home', 'main page', 'start', 'landing page']
      },
      {
        elementName: 'database query',
        actionType: 'page_navigation',
        handler: 'executeNavigationAction',
        context: 'navigation',
        description: 'Navigate to database query page',
        aliases: ['database', 'sql', 'query', 'db query', 'database search']
      },
      {
        elementName: 'file query',
        actionType: 'page_navigation',
        handler: 'executeNavigationAction',
        context: 'navigation',
        description: 'Navigate to file query page',
        aliases: ['file', 'files', 'file search', 'file upload', 'file management']
      }
    ]
  },
  {
    name: 'Data Operations',
    description: 'Data manipulation and management actions',
    actions: [
      {
        elementName: 'create table',
        actionType: 'table_creation',
        handler: 'executeCreateTableAction',
        context: 'table_management',
        description: 'Create a new database table',
        aliases: ['new table', 'add table', 'table creation', 'create database table']
      },
      {
        elementName: 'delete table',
        actionType: 'table_deletion',
        handler: 'executeDeleteTableAction',
        context: 'table_management',
        description: 'Delete an existing database table',
        aliases: ['remove table', 'drop table', 'delete database table', 'remove database table']
      },
      {
        elementName: 'export data',
        actionType: 'data_export',
        handler: 'executeDataExportAction',
        context: 'data_management',
        description: 'Export data from database or files',
        aliases: ['download data', 'export to excel', 'export to csv', 'download report']
      }
    ]
  },
  {
    name: 'User Management',
    description: 'User account and access management actions',
    actions: [
      {
        elementName: 'create user',
        actionType: 'user_creation',
        handler: 'executeCreateUserAction',
        context: 'user_management',
        description: 'Create a new user account',
        aliases: ['new user', 'add user', 'user creation', 'register user']
      },
      {
        elementName: 'delete user',
        actionType: 'user_deletion',
        handler: 'executeDeleteUserAction',
        context: 'user_management',
        description: 'Delete an existing user account',
        aliases: ['remove user', 'user removal', 'delete account', 'remove account']
      },
      {
        elementName: 'change password',
        actionType: 'password_change',
        handler: 'executePasswordChangeAction',
        context: 'user_management',
        description: 'Change user password',
        aliases: ['update password', 'reset password', 'new password', 'modify password']
      }
    ]
  },
  {
    name: 'Business Rules',
    description: 'Business logic and rule management actions',
    actions: [
      {
        elementName: 'add business rule',
        actionType: 'business_rule_creation',
        handler: 'executeAddBusinessRuleAction',
        context: 'business_rules',
        description: 'Add a new business rule',
        aliases: ['new business rule', 'create business rule', 'business rule creation', 'add rule']
      },
      {
        elementName: 'edit business rule',
        actionType: 'business_rule_editing',
        handler: 'executeEditBusinessRuleAction',
        context: 'business_rules',
        description: 'Edit an existing business rule',
        aliases: ['modify business rule', 'update business rule', 'change business rule', 'edit rule']
      },
      {
        elementName: 'delete business rule',
        actionType: 'business_rule_deletion',
        handler: 'executeDeleteBusinessRuleAction',
        context: 'business_rules',
        description: 'Delete an existing business rule',
        aliases: ['remove business rule', 'business rule removal', 'delete rule', 'remove rule']
      }
    ]
  }
]

// Helper function to get all button actions flattened
export const getAllButtonActions = (): ButtonActionDefinition[] => {
  return BUTTON_ACTION_CONFIGS.flatMap(category => category.actions)
}

// Helper function to find button action by element name
export const findButtonAction = (elementName: string): ButtonActionDefinition | undefined => {
  const normalizedName = elementName.toLowerCase()
  return getAllButtonActions().find(action => 
    action.elementName.toLowerCase() === normalizedName ||
    action.aliases?.some(alias => alias.toLowerCase() === normalizedName)
  )
}

// Helper function to get button actions by category
export const getButtonActionsByCategory = (categoryName: string): ButtonActionDefinition[] => {
  const category = BUTTON_ACTION_CONFIGS.find(cat => cat.name.toLowerCase() === categoryName.toLowerCase())
  return category?.actions || []
}

// Helper function to get all available action types
export const getAvailableActionTypes = (): InteractionType[] => {
  const actionTypes = new Set<InteractionType>()
  getAllButtonActions().forEach(action => actionTypes.add(action.actionType))
  return Array.from(actionTypes)
}

// Helper function to validate button action configuration
export const validateButtonActionConfig = (config: ButtonActionDefinition): boolean => {
  return !!(
    config.elementName &&
    config.actionType &&
    config.handler &&
    config.context &&
    config.description
  )
}

// Export default configuration for easy import
export default BUTTON_ACTION_CONFIGS 