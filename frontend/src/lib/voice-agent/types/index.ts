export interface VoiceMessage {
  id: string
  type: 'user' | 'assistant' | 'system' | 'error' | 'tool_call' | 'tool_result' | 'navigation'
  content: string
  timestamp: Date
  isAudio?: boolean
  navigationData?: NavigationData
}

export interface NavigationData {
  action_type: string
  param: string
  value: string
  page: string
  previous_page: string | null
  interaction_type: string
  clicked: boolean
  element_name: string | null
  search_query: string | null
  report_request: string | null
  report_query: string | null
  upload_request: string | null
  db_id: string | null
  table_specific: boolean
  tables: string[]
  file_descriptions: string[]
  table_names: string[]
  context: string | null
  timestamp: string
  user_id: string
  success: boolean
  error_message: string | null
}

export interface VoiceClientState {
  isConnected: boolean
  isInConversation: boolean
  connectionStatus: string
  messages: VoiceMessage[]
  currentPage: string
  previousPage: string | null
  isReady: boolean
}

export interface VoiceClientActions {
  connect: () => Promise<void>
  disconnect: () => Promise<void>
  startConversation: () => Promise<void>
  stopConversation: () => void
  clearMessages: () => void
  sendMessage: (message: string) => void
  navigateToPage: (page: string) => void
  clickElement: (elementName: string) => void
  executeSearch: (query: string, type: 'database' | 'file') => void
  handleFileUpload: (descriptions: string[], tableNames: string[]) => void
  viewReport: (request: string) => void
  generateReport: (query: string) => void
  testNavigation: (page: string) => void
  
  // Debug methods
  refreshPageState: () => void
  getCurrentPageState: () => { currentPage: string; previousPage: string | null }
}

export interface VoiceClientHook extends VoiceClientState, VoiceClientActions {}

export type InteractionType = 
  | 'page_navigation'
  | 'button_click'
  | 'database_search'
  | 'file_search'
  | 'file_upload'
  | 'view_report'
  | 'generate_report'

export interface NavigationEvent {
  page: string
  previousPage: string | null
  type: InteractionType
}

export interface ClickEvent {
  elementName: string
  page: string
  type: InteractionType
}

export interface SearchEvent {
  query: string
  type: 'database' | 'file'
  page: string
  interactionType: InteractionType
}

export interface UploadEvent {
  descriptions: string[]
  tableNames: string[]
  page: string
  type: InteractionType
}

export interface ReportEvent {
  request?: string
  query?: string
  page: string
  type: InteractionType
} 