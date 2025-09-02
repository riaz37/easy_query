import { ButtonAction } from '../services/ButtonRegistry'

/**
 * Default button actions for common UI elements
 * These can be easily extended and customized
 */

export const defaultButtonActions: ButtonAction[] = [
  // Navigation buttons
  {
    id: 'dashboard',
    name: 'Dashboard',
    aliases: ['home', 'main', 'dashboard button'],
    description: 'Navigate to dashboard',
    category: 'navigation',
    handler: () => {
      if (typeof window !== 'undefined') {
        window.location.href = '/'
      }
    }
  },

  {
    id: 'database-query',
    name: 'Database Query',
    aliases: ['database', 'query', 'sql'],
    description: 'Navigate to database query page',
    category: 'navigation',
    handler: () => {
      if (typeof window !== 'undefined') {
        window.location.href = '/database-query'
      }
    }
  },

  {
    id: 'file-query',
    name: 'File Query',
    aliases: ['file', 'files', 'file search'],
    description: 'Navigate to file query page',
    category: 'navigation',
    handler: () => {
      if (typeof window !== 'undefined') {
        window.location.href = '/file-query'
      }
    }
  },

  {
    id: 'tables',
    name: 'Tables',
    aliases: ['table', 'data tables', 'manage tables'],
    description: 'Navigate to tables page',
    category: 'navigation',
    handler: () => {
      if (typeof window !== 'undefined') {
        window.location.href = '/tables'
      }
    }
  },

  {
    id: 'users',
    name: 'Users',
    aliases: ['user management', 'manage users', 'user page'],
    description: 'Navigate to users page',
    category: 'navigation',
    handler: () => {
      if (typeof window !== 'undefined') {
        window.location.href = '/users'
      }
    }
  },

  {
    id: 'ai-results',
    name: 'AI Results',
    aliases: ['results', 'reports', 'ai reports'],
    description: 'Navigate to AI results page',
    category: 'navigation',
    handler: () => {
      if (typeof window !== 'undefined') {
        window.location.href = '/ai-results'
      }
    }
  },

  // Form actions
  {
    id: 'submit',
    name: 'Submit',
    aliases: ['submit button', 'send', 'save'],
    description: 'Click submit button',
    category: 'forms',
    handler: () => {
      const submitBtn = document.querySelector('button[type="submit"], input[type="submit"], .submit-btn') as HTMLElement
      if (submitBtn) {
        submitBtn.click()
        console.log('🎯 Clicked submit button')
      } else {
        console.warn('🎯 Submit button not found')
      }
    }
  },

  {
    id: 'cancel',
    name: 'Cancel',
    aliases: ['cancel button', 'close', 'dismiss'],
    description: 'Click cancel button',
    category: 'forms',
    handler: () => {
      const cancelBtn = document.querySelector('button[type="button"]:contains("Cancel"), .cancel-btn, .close-btn') as HTMLElement
      if (cancelBtn) {
        cancelBtn.click()
        console.log('🎯 Clicked cancel button')
      } else {
        console.warn('🎯 Cancel button not found')
      }
    }
  },

  {
    id: 'reset',
    name: 'Reset',
    aliases: ['reset button', 'clear form'],
    description: 'Click reset button',
    category: 'forms',
    handler: () => {
      const resetBtn = document.querySelector('button[type="reset"], .reset-btn') as HTMLElement
      if (resetBtn) {
        resetBtn.click()
        console.log('🎯 Clicked reset button')
      } else {
        console.warn('🎯 Reset button not found')
      }
    }
  },

  // File operations
  {
    id: 'upload',
    name: 'Upload',
    aliases: ['upload file', 'file upload', 'upload button'],
    description: 'Click upload button',
    category: 'files',
    handler: () => {
      const uploadBtn = document.querySelector('.upload-btn, input[type="file"], button:contains("Upload")') as HTMLElement
      if (uploadBtn) {
        uploadBtn.click()
        console.log('🎯 Clicked upload button')
      } else {
        console.warn('🎯 Upload button not found')
      }
    }
  },

  {
    id: 'download',
    name: 'Download',
    aliases: ['download button', 'export', 'save file'],
    description: 'Click download button',
    category: 'files',
    handler: () => {
      const downloadBtn = document.querySelector('.download-btn, button:contains("Download"), a[download]') as HTMLElement
      if (downloadBtn) {
        downloadBtn.click()
        console.log('🎯 Clicked download button')
      } else {
        console.warn('🎯 Download button not found')
      }
    }
  },

  // Modal and dialog actions
  {
    id: 'confirm',
    name: 'Confirm',
    aliases: ['confirm button', 'ok', 'yes'],
    description: 'Click confirm button',
    category: 'dialogs',
    handler: () => {
      const confirmBtn = document.querySelector('.confirm-btn, button:contains("Confirm"), button:contains("OK"), button:contains("Yes")') as HTMLElement
      if (confirmBtn) {
        confirmBtn.click()
        console.log('🎯 Clicked confirm button')
      } else {
        console.warn('🎯 Confirm button not found')
      }
    }
  },

  // Search actions
  {
    id: 'search',
    name: 'Search',
    aliases: ['search button', 'find', 'lookup'],
    description: 'Click search button',
    category: 'search',
    handler: () => {
      const searchBtn = document.querySelector('.search-btn, button[type="search"], button:contains("Search")') as HTMLElement
      if (searchBtn) {
        searchBtn.click()
        console.log('🎯 Clicked search button')
      } else {
        console.warn('🎯 Search button not found')
      }
    }
  },

  {
    id: 'filter',
    name: 'Filter',
    aliases: ['filter button', 'apply filter'],
    description: 'Click filter button',
    category: 'search',
    handler: () => {
      const filterBtn = document.querySelector('.filter-btn, button:contains("Filter")') as HTMLElement
      if (filterBtn) {
        filterBtn.click()
        console.log('🎯 Clicked filter button')
      } else {
        console.warn('🎯 Filter button not found')
      }
    }
  },

  // CRUD operations
  {
    id: 'create',
    name: 'Create',
    aliases: ['create button', 'add', 'new'],
    description: 'Click create/add button',
    category: 'crud',
    handler: () => {
      const createBtn = document.querySelector('.create-btn, .add-btn, button:contains("Create"), button:contains("Add"), button:contains("New")') as HTMLElement
      if (createBtn) {
        createBtn.click()
        console.log('🎯 Clicked create button')
      } else {
        console.warn('🎯 Create button not found')
      }
    }
  },

  {
    id: 'edit',
    name: 'Edit',
    aliases: ['edit button', 'modify', 'update'],
    description: 'Click edit button',
    category: 'crud',
    handler: () => {
      const editBtn = document.querySelector('.edit-btn, button:contains("Edit"), button:contains("Update")') as HTMLElement
      if (editBtn) {
        editBtn.click()
        console.log('🎯 Clicked edit button')
      } else {
        console.warn('🎯 Edit button not found')
      }
    }
  },

  {
    id: 'delete',
    name: 'Delete',
    aliases: ['delete button', 'remove', 'trash'],
    description: 'Click delete button',
    category: 'crud',
    handler: () => {
      const deleteBtn = document.querySelector('.delete-btn, button:contains("Delete"), button:contains("Remove")') as HTMLElement
      if (deleteBtn) {
        deleteBtn.click()
        console.log('🎯 Clicked delete button')
      } else {
        console.warn('🎯 Delete button not found')
      }
    }
  },

  // User-specific actions
  {
    id: 'vector-database-access',
    name: 'Vector Database Access',
    aliases: ['vector db', 'vector access', 'database access'],
    description: 'Click vector database access button',
    category: 'user-management',
    handler: () => {
      // Look for the specific button by text content
      const buttons = Array.from(document.querySelectorAll('button'))
      const vectorBtn = buttons.find(btn =>
        btn.textContent?.toLowerCase().includes('vector') &&
        btn.textContent?.toLowerCase().includes('database')
      )

      if (vectorBtn) {
        vectorBtn.click()
        console.log('🎯 Clicked vector database access button')
      } else {
        console.warn('🎯 Vector database access button not found')
      }
    }
  },

  {
    id: 'change-password',
    name: 'Change Password',
    aliases: ['password', 'change pwd', 'update password'],
    description: 'Click change password button',
    category: 'user-management',
    handler: () => {
      const pwdBtn = document.querySelector('button:contains("Password"), .password-btn') as HTMLElement
      if (pwdBtn) {
        pwdBtn.click()
        console.log('🎯 Clicked change password button')
      } else {
        console.warn('🎯 Change password button not found')
      }
    }
  },

  // Menu and navigation
  {
    id: 'menu',
    name: 'Menu',
    aliases: ['menu button', 'hamburger', 'sidebar'],
    description: 'Click menu button',
    category: 'navigation',
    handler: () => {
      const menuBtn = document.querySelector('.menu-btn, .hamburger, button[aria-label*="menu"]') as HTMLElement
      if (menuBtn) {
        menuBtn.click()
        console.log('🎯 Clicked menu button')
      } else {
        console.warn('🎯 Menu button not found')
      }
    }
  },

  // Generic fallback action
  {
    id: 'generic-click',
    name: 'Generic Click',
    aliases: ['click', 'press', 'tap'],
    description: 'Generic button click fallback',
    category: 'generic',
    handler: (context?: any) => {
      // This is a fallback that tries to find any button with matching text
      const searchText = context?.elementName || context?.element_name
      if (!searchText) {
        console.warn('🎯 No element name provided for generic click')
        return
      }

      const buttons = Array.from(document.querySelectorAll('button, [role="button"], a'))
      const matchingButton = buttons.find(btn => {
        const text = btn.textContent?.toLowerCase().trim()
        const searchLower = searchText.toLowerCase().trim()
        return text && (text.includes(searchLower) || searchLower.includes(text))
      })

      if (matchingButton) {
        (matchingButton as HTMLElement).click()
        console.log(`🎯 Generic click executed for: ${searchText}`)
      } else {
        console.warn(`🎯 No button found for generic click: ${searchText}`)
      }
    }
  }
]

// Helper function to create custom button actions easily
export function createButtonAction(
  id: string,
  name: string,
  handler: (context?: any) => void | Promise<void>,
  options?: {
    aliases?: string[]
    description?: string
    category?: string
    validation?: (context?: any) => boolean
    selector?: string
  }
): ButtonAction {
  return {
    id,
    name,
    handler,
    aliases: options?.aliases,
    description: options?.description,
    category: options?.category || 'custom',
    validation: options?.validation,
    selector: options?.selector
  }
}

// Helper function to create DOM-based button action
export function createDOMButtonAction(
  id: string,
  name: string,
  selector: string,
  options?: {
    aliases?: string[]
    description?: string
    category?: string
    validation?: (context?: any) => boolean
  }
): ButtonAction {
  return createButtonAction(
    id,
    name,
    () => {
      const element = document.querySelector(selector) as HTMLElement
      if (element) {
        element.click()
        console.log(`🎯 Clicked element with selector: ${selector}`)
      } else {
        console.warn(`🎯 Element not found with selector: ${selector}`)
      }
    },
    {
      ...options,
      selector
    }
  )
}
