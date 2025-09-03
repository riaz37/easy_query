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
    expectedText?: string
    expectedClass?: string
    expectedId?: string
  }
  aliases?: string[]
  description?: string
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
      "button:contains('Add Vector DB Access')",           // Primary text match
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
      "button:contains('Add MSSQL Access')",              // Primary text match
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
