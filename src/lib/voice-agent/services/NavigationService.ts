import { VOICE_AGENT_CONFIG } from '../config'
import { MessageService } from './MessageService'
import { NavigationData, InteractionType } from '../types'

export class NavigationService {
  private static currentPage = 'dashboard'
  private static previousPage: string | null = null

  // This method is now primarily used for external state updates
  static detectCurrentPage(): string {
    if (typeof window !== 'undefined') {
      const path = window.location.pathname
      const page = path === '/' ? 'dashboard' : path.substring(1)
      console.log('🧭 NavigationService detected current page from URL:', page)
      return page
    }
    return 'dashboard'
  }

  static getCurrentPage(): string {
    return this.currentPage
  }

  static getPreviousPage(): string | null {
    return this.previousPage
  }

  // Update page state - called by VoiceAgentService to keep in sync
  static updatePage(newPage: string): void {
    if (newPage !== this.currentPage) {
      this.previousPage = this.currentPage
      this.currentPage = newPage
      console.log('🧭 NavigationService state updated:', this.previousPage, '→', this.currentPage)
    }
  }

  // Force refresh page state from URL (for debugging)
  static refreshPageState(): void {
    const detectedPage = this.detectCurrentPage()
    this.updatePage(detectedPage)
  }

  // This method is no longer used - navigation is now backend-driven
  // Kept for potential future use or debugging
  static detectNavigationFromVoice(text: string): string | null {
    console.log('🧭 Navigation detection disabled - using backend commands instead')
    return null
  }

  static createNavigationData(
    interactionType: InteractionType,
    page: string,
    options: Partial<NavigationData> = {}
  ): NavigationData {
    return MessageService.createNavigationData(
      'navigation',
      interactionType,
      page,
      this.previousPage,
      options
    )
  }

  static executeNavigation(page: string): void {
    console.log('🧭 Executing navigation to:', page)
    
    // Update internal state
    this.updatePage(page)
    
    // Emit navigation event
    this.emitNavigationEvent(page, 'page_navigation')
    
    // Direct navigation as fallback
    if (typeof window !== 'undefined') {
      console.log('🧭 Direct navigation to:', `/${page}`)
      window.location.href = `/${page}`
    }
  }

  static executeElementClick(elementName: string): void {
    console.log('🖱️ Executing click on:', elementName)
    
    const navigationData = this.createNavigationData('button_click', this.currentPage, {
      clicked: true,
      element_name: elementName,
      param: 'clicked,name',
      value: `true,${elementName}`
    })

    // Emit click event
    this.emitClickEvent(elementName, 'button_click')
    
    // Add message for logging
    MessageService.createNavigationMessage(
      MessageService.formatNavigationContent('button_click', elementName),
      navigationData
    )
  }

  static executeSearch(query: string, type: 'database' | 'file'): void {
    console.log('🔍 Executing search:', type, query)
    
    const interactionType = type === 'database' ? 'database_search' : 'file_search'
    const navigationData = this.createNavigationData(interactionType, this.currentPage, {
      search_query: query,
      param: type === 'database' ? 'search,question' : 'query,table_specific,tables[]',
      value: type === 'database' ? `true,${query}` : `${query},false,[]`
    })

    // Emit search event
    this.emitSearchEvent(query, type, interactionType)
    
    // Add message for logging
    MessageService.createNavigationMessage(
      MessageService.formatNavigationContent(interactionType, query),
      navigationData
    )
  }

  static executeFileUpload(descriptions: string[], tableNames: string[]): void {
    console.log('📤 Executing file upload:', descriptions, tableNames)
    
    const navigationData = this.createNavigationData('file_upload', this.currentPage, {
      file_descriptions: descriptions,
      table_names: tableNames,
      param: 'file_descriptions[],table_names[]',
      value: `[${descriptions.map(d => `"${d}"`).join(',')}],[${tableNames.map(t => `"${t}"`).join(',')}]`,
      upload_request: `Upload files: ${descriptions.join(', ')} to tables: ${tableNames.join(', ')}`,
      context: `file_descriptions:${descriptions.join(',')},table_names:${tableNames.join(',')}`
    })

    // Emit upload event
    this.emitUploadEvent(descriptions, tableNames, 'file_upload')
    
    // Add message for logging
    MessageService.createNavigationMessage(
      MessageService.formatNavigationContent('file_upload', `${descriptions.length} files`),
      navigationData
    )
  }

  static executeViewReport(request: string): void {
    console.log('📊 Executing view report:', request)
    
    const navigationData = this.createNavigationData('view_report', this.currentPage, {
      report_request: request,
      clicked: true,
      element_name: 'view report',
      param: 'clicked,name,report_request',
      value: `true,view report,${request}`
    })

    // Emit view report event
    this.emitReportEvent({ request }, 'view_report')
    
    // Add message for logging
    MessageService.createNavigationMessage(
      MessageService.formatNavigationContent('view_report', request),
      navigationData
    )
  }

  static executeGenerateReport(query: string): void {
    console.log('📈 Executing generate report:', query)
    
    const navigationData = this.createNavigationData('generate_report', this.currentPage, {
      report_query: query,
      clicked: true,
      element_name: 'report generation',
      param: 'clicked,name,report_query',
      value: `true,report generation,${query}`,
      context: 'report generation'
    })

    // Emit generate report event
    this.emitReportEvent({ query }, 'generate_report')
    
    // Add message for logging
    MessageService.createNavigationMessage(
      MessageService.formatNavigationContent('generate_report', query),
      navigationData
    )
  }

  // Event emission methods
  private static emitNavigationEvent(page: string, type: InteractionType): void {
    window.dispatchEvent(new CustomEvent('voice-navigation', {
      detail: { page, previousPage: this.previousPage, type }
    }))
  }

  private static emitClickEvent(elementName: string, type: InteractionType): void {
    window.dispatchEvent(new CustomEvent('voice-click', {
      detail: { elementName, page: this.currentPage, type }
    }))
  }

  private static emitSearchEvent(query: string, searchType: 'database' | 'file', interactionType: InteractionType): void {
    window.dispatchEvent(new CustomEvent('voice-search', {
      detail: { query, type: searchType, page: this.currentPage, interactionType }
    }))
  }

  private static emitUploadEvent(descriptions: string[], tableNames: string[], type: InteractionType): void {
    window.dispatchEvent(new CustomEvent('voice-upload', {
      detail: { descriptions, tableNames, page: this.currentPage, type }
    }))
  }

  private static emitReportEvent(data: { request?: string; query?: string }, type: InteractionType): void {
    window.dispatchEvent(new CustomEvent('voice-report', {
      detail: { ...data, page: this.currentPage, type }
    }))
  }
} 