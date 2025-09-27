export const VOICE_AGENT_CONFIG = {
  // Backend configuration - use environment variables with fallbacks
  BACKEND_URL: process.env.NEXT_PUBLIC_API_BASE_URL || 'https://176.9.16.194:8200',

  // WebSocket endpoints
  WEBSOCKET_ENDPOINTS: {
    VOICE_WS: '/voice/ws/tools',  // Updated to match backend endpoint
    VOICE_CONNECT: '/voice/connect',
    HEALTH_CHECK: '/voice/health'
  },

  // RTVI configuration
  RTVI: {
    ENABLE_MIC: true,
    ENABLE_CAM: false,
    CONNECTION_TIMEOUT: 10000,
    RECONNECTION_ATTEMPTS: 3
  },

  // Navigation patterns for voice commands
  NAVIGATION_PATTERNS: {
    DASHBOARD: ['dashboard', 'home', 'main page', 'start'],
    FILE_QUERY: ['file', 'upload', 'files', 'file query', 'file search', 'file upload'],
    DATABASE_QUERY: ['database', 'sql', 'query', 'database query', 'database search'],
    TABLES: ['table', 'tables', 'show tables', 'manage tables'],
    USERS: ['user', 'users', 'user management', 'manage users'],
    REPORTS: ['report', 'ai reports', 'ai-reports', 'generate report'],
    COMPANY_STRUCTURE: ['company', 'hierarchy', 'company structure', 'organization']
  },

  // Page mappings
  PAGE_MAPPINGS: {
    'dashboard': 'dashboard',
    'file-query': 'file-query',
    'database-query': 'database-query',
    'tables': 'tables',
    'users': 'users',
    'ai-reports': 'ai-reports',
    'company-structure': 'company-structure'
  },

  // Message delays and timeouts
  TIMING: {
    NAVIGATION_DELAY: 500,
    MESSAGE_DEBOUNCE: 300,
    CONNECTION_TIMEOUT: 10000
  },

  // Default values
  DEFAULTS: {
    USER_ID: 'frontend_user',  // Updated to match backend expectation
    CURRENT_PAGE: 'dashboard',
    MESSAGE_ID_PREFIX: 'voice-'
  }
} as const

export const getWebSocketUrl = (baseUrl: string, endpoint: string): string => {
  return baseUrl.replace(/^https?:\/\//, 'wss://') + endpoint
}

export const getHealthCheckUrl = (baseUrl: string): string => {
  return baseUrl + VOICE_AGENT_CONFIG.WEBSOCKET_ENDPOINTS.HEALTH_CHECK
}

// Helper function to get the RTVI config with current page
export const getRTVIConfig = (currentPage?: string) => {
  const baseUrl = VOICE_AGENT_CONFIG.BACKEND_URL
  const connectEndpoint = currentPage
    ? `${VOICE_AGENT_CONFIG.WEBSOCKET_ENDPOINTS.VOICE_CONNECT}?current_page=${encodeURIComponent(currentPage)}`
    : VOICE_AGENT_CONFIG.WEBSOCKET_ENDPOINTS.VOICE_CONNECT

  return {
    baseUrl,
    endpoints: {
      connect: connectEndpoint
    },
  }
} 