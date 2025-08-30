import { VoiceMessage, NavigationData } from '../types'

export class MessageService {
  private static generateMessageId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  }

  static createMessage(
    type: VoiceMessage['type'],
    content: string,
    options: Partial<Omit<VoiceMessage, 'id' | 'timestamp' | 'type' | 'content'>> = {}
  ): VoiceMessage {
    return {
      id: this.generateMessageId(),
      type,
      content,
      timestamp: new Date(),
      ...options
    }
  }

  static createSystemMessage(content: string): VoiceMessage {
    return this.createMessage('system', content)
  }

  static createErrorMessage(content: string): VoiceMessage {
    return this.createMessage('error', content)
  }

  static createUserMessage(content: string, isAudio = false): VoiceMessage {
    return this.createMessage('user', content, { isAudio })
  }

  static createAssistantMessage(content: string): VoiceMessage {
    return this.createMessage('assistant', content)
  }

  static createNavigationMessage(
    content: string,
    navigationData: NavigationData
  ): VoiceMessage {
    return this.createMessage('navigation', content, { navigationData })
  }

  static createToolMessage(
    type: 'tool_call' | 'tool_result',
    content: string
  ): VoiceMessage {
    return this.createMessage(type, content)
  }

  static formatNavigationContent(interactionType: string, page: string): string {
    const emojiMap: Record<string, string> = {
      'page_navigation': '🧭',
      'button_click': '🖱️',
      'database_search': '🔍',
      'file_search': '🔍',
      'file_upload': '📤',
      'view_report': '📊',
      'generate_report': '📈'
    }

    const emoji = emojiMap[interactionType] || '🎯'
    return `${emoji} ${interactionType.replace('_', ' ')}: ${page}`
  }

  static formatSystemContent(message: string): string {
    const emojiMap: Record<string, string> = {
      'connected': '✅',
      'disconnected': '👋',
      'conversation_started': '🎙️',
      'conversation_paused': '⏸️',
      'bot_ready': '🤖',
      'websocket_connected': '✅',
      'websocket_disconnected': '🔌',
      'websocket_error': '❌',
      'audio_ready': '🎤',
      'connection_failed': '❌'
    }

    for (const [key, emoji] of Object.entries(emojiMap)) {
      if (message.toLowerCase().includes(key)) {
        return `${emoji} ${message}`
      }
    }

    return message
  }

  static createNavigationData(
    actionType: string,
    interactionType: string,
    page: string,
    previousPage: string | null,
    options: Partial<NavigationData> = {}
  ): NavigationData {
    return {
      action_type: actionType,
      param: '',
      value: '',
      page,
      previous_page: previousPage,
      interaction_type: interactionType,
      clicked: false,
      element_name: null,
      search_query: null,
      report_request: null,
      report_query: null,
      upload_request: null,
      db_id: null,
      table_specific: false,
      tables: [],
      file_descriptions: [],
      table_names: [],
      context: null,
      timestamp: new Date().toISOString(),
      user_id: 'voice_user',
      success: true,
      error_message: null,
      ...options
    }
  }
} 