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

  // Event emission methods
  private static emitNavigationEvent(page: string, type: InteractionType): void {
    window.dispatchEvent(new CustomEvent('voice-navigation', {
      detail: { page, previousPage: this.previousPage, type }
    }))
  }
} 