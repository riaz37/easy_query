// Removed old ButtonAction interface - using mapping system only

/**
 * Button Mapping Configuration
 * Maps backend element_name to frontend button configurations
 */
export interface ButtonMappingConfig {
  id: string
  name: string
  page: string
  selectors: string[]
  validation?: {
    expectedText?: string | string[]
    expectedClass?: string | string[]
    expectedId?: string
  }
  aliases?: string[]
  description?: string
  workflow?: {
    fillTextarea?: boolean
    textareaSelectors?: string[]
  }
}

/**
 * Central mapping configuration for all button actions
 * Maps backend element_name to frontend button selectors
 */
export const BUTTON_MAPPING_CONFIG: Record<string, ButtonMappingConfig> = {
  // Vector Database Access Button
  "vector database access": {
    id: "add-vector-db-access",
    name: "Add Vector DB Access",
    page: "users",
    selectors: [
      "[data-action='vector-db-access']",                  // Data attribute
      "#add-vector-db-btn",                               // ID selector
      "button[aria-label*='vector']",                     // Aria label
      "button.bg-purple-600"                              // Class-based
    ],
    validation: {
      expectedText: "Add Vector DB Access",
      expectedClass: "bg-purple-600"
    },
    aliases: ["add vector db", "vector database access", "create vector database"],
    description: "Click add vector database access button"
  },

  // MSSQL Database Access Button  
  "manage database access": {
    id: "add-mssql-access",
    name: "Add MSSQL Access", 
    page: "users",
    selectors: [
      "[data-action='mssql-access']",                     // Data attribute
      "#add-mssql-btn",                                   // ID selector
      "button[aria-label*='mssql']",                      // Aria label
      "button.bg-blue-600"                                // Class-based
    ],
    validation: {
      expectedText: "Add MSSQL Access",
      expectedClass: "bg-blue-600"
    },
    aliases: ["add mssql", "mssql access", "create mssql access"],
    description: "Click add MSSQL access button"
  },

  // File Upload Area (to open file picker)
  "upload": {
    id: "file-upload-area",
    name: "Open File Picker",
    page: "file-query",
    selectors: [
      "[data-element='upload-area']",                     // Upload area (drag & drop)
      "#upload-area",                                     // ID selector
      "div[class*='border-dashed']",                      // Class-based
      "div[class*='upload']"                              // Generic upload div
    ],
    validation: {
      expectedText: "Click to upload files or drag and drop",
      expectedClass: "border-dashed"
    },
    aliases: ["upload file", "select files", "choose files", "add files"],
    description: "Click upload area to open file picker"
  },

  // File Upload Button (to actually upload selected files)
  "upload files": {
    id: "file-upload-button",
    name: "Upload Selected Files",
    page: "file-query",
    selectors: [
      "button[data-element='upload-button']",             // Upload button data-element
      "#upload-button",                                   // ID selector
      "button[aria-label*='upload']",                     // Aria label
      "button.bg-blue-600"                                // Class-based
    ],
    validation: {
      expectedText: "Upload",
      expectedClass: "bg-blue-600"
    },
    aliases: ["upload selected files", "upload pending files", "process files"],
    description: "Click upload button to upload selected files"
  },

  // Search Button (works for both file-query and database-query pages)
  "search": {
    id: "search-button",
    name: "Search/Query",
    page: "both", // Works on both pages
    selectors: [
      // Prioritize data-element selectors (most specific)
      "button[data-element='file-query-submit']",         // File query submit button
      "button[data-element='query-submit']",              // Database query submit button
      "#search-button",                                   // ID selector
      "button[aria-label*='search']",                     // Aria label
      // More specific class selectors to avoid mode toggle conflicts
      "button.bg-green-600[type='submit']",               // File query button with type
      "button.bg-blue-600[type='submit']",                // Database query button with type
      // Fallback selectors (less specific, but should work)
      "button.bg-green-600",                              // File query button class
      "button.bg-blue-600"                                // Database query button class
    ],
    validation: {
      expectedText: ["Execute Query", "Ask Question"],    // Multiple valid texts
      expectedClass: ["bg-green-600", "bg-blue-600"]     // Multiple valid classes
    },
    aliases: ["search", "search files", "file search", "database search", "query database", "ask question", "execute query"],
    description: "Fill textarea with search query and click submit button (works on both file-query and database-query pages)",
    // Enhanced workflow: fill textarea + click button
    workflow: {
      fillTextarea: true,
      textareaSelectors: [
        "textarea[data-element='file-query-input']",      // File query textarea
        "textarea[data-element='query-input']",           // Database query textarea
        "#query-input",                                   // ID selector
        "textarea[placeholder*='query']",                 // Placeholder-based
        "textarea[placeholder*='question']"               // Placeholder-based
      ]
    }
  }
}

// Removed old ButtonAction conversion functions - using mapping system directly

/**
 * Get button mapping by element name
 */
export function getButtonMapping(elementName: string): ButtonMappingConfig | undefined {
  return BUTTON_MAPPING_CONFIG[elementName];
}

/**
 * Get all button mappings for a specific page
 */
export function getButtonMappingsForPage(page: string): ButtonMappingConfig[] {
  return Object.values(BUTTON_MAPPING_CONFIG).filter(mapping => mapping.page === page)
}

// Removed old ButtonAction helper functions - using mapping system directly
