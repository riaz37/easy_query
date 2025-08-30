'use client'

import React from 'react'
import { VoiceInterface } from '@/components/VoiceInterface'
import { ConversationDisplay } from '@/components/ConversationDisplay'
import { WebSocketMonitor } from '@/components/WebSocketMonitor'
import { ToolsResultsDisplay } from '@/components/ToolsResultsDisplay'
import { ToolsWebSocketTest } from '@/components/ToolsWebSocketTest'
import { EnvironmentSwitcher } from '@/components/EnvironmentSwitcher'
import { ConnectionTest } from '@/components/ConnectionTest'
import { useVoiceClient } from '@/lib/voiceClient'
import { Mic, Brain, MessageCircle, Settings, Database } from 'lucide-react'

export default function VoiceAgentPage() {
  const {
    messages,
    clearMessages,
    clearToolCalls,
    isConnected,
    isToolsConnected,
    connect,
    disconnect,
    connectToolWebSocket
  } = useVoiceClient()

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-green-50 to-teal-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-blue-100 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Mic className="w-8 h-8 text-voice-primary" />
                <Brain className="w-8 h-8 text-voice-secondary" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">MSSQL Search Agent</h1>
                <p className="text-sm text-gray-600">AI-Powered Database Query Assistant</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Connection Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
                <span className="text-sm font-medium text-gray-700">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
              
              {/* Tools Status */}
              <div className="flex items-center space-x-2 bg-blue-100 px-3 py-1 rounded-full">
                <Settings className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium text-blue-800">
                  Tools: {isToolsConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
          
          {/* Description */}
          <p className="mt-2 text-gray-600 text-center max-w-2xl mx-auto">
            Advanced AI assistant with voice interaction for MSSQL database queries and search operations
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Conversation Area - Takes up 2/4 of the width */}
          <div className="lg:col-span-2 space-y-6">
            {/* Conversation Display */}
            <div className="bg-white rounded-2xl shadow-xl border border-blue-100 overflow-hidden backdrop-blur-sm">
              <div className="bg-gradient-to-r from-voice-primary via-voice-secondary to-teal-500 px-6 py-4">
                <div className="flex items-center space-x-3 text-white">
                  <MessageCircle className="w-6 h-6" />
                  <h2 className="text-lg font-semibold">Conversation</h2>
                </div>
              </div>
              <div className="p-6 h-96">
                <ConversationDisplay 
                  messages={messages}
                  onClear={clearMessages}
                />
              </div>
            </div>

            {/* Voice Interface */}
            <div className="bg-white rounded-2xl shadow-xl border border-green-100 overflow-hidden backdrop-blur-sm">
              <div className="bg-gradient-to-r from-green-500 via-emerald-600 to-teal-500 px-6 py-4">
                <div className="flex items-center space-x-3 text-white">
                  <Mic className="w-6 h-6" />
                  <h2 className="text-lg font-semibold">Voice Control</h2>
                </div>
              </div>
              <div className="p-6">
                <VoiceInterface />
              </div>
            </div>
          </div>

          {/* Tools Monitor - Takes up 2/4 of the width */}
          <div className="lg:col-span-2 space-y-6">
            {/* Environment Configuration */}
            <EnvironmentSwitcher />
            
            {/* Connection Test */}
            <ConnectionTest />
            
            {/* Enhanced Tools Results Display */}
            <ToolsResultsDisplay 
              isConnected={isToolsConnected} 
              onConnect={connectToolWebSocket}
              onDisconnect={disconnect}
            />
            
            {/* Tools WebSocket Direct Test */}
            <ToolsWebSocketTest />
            
            {/* WebSocket Raw Monitor (for debugging) */}
            <WebSocketMonitor 
              isConnected={isToolsConnected} 
              onConnect={connectToolWebSocket}
              onDisconnect={disconnect}
            />
          </div>
        </div>
      </main>

      {/* Status Footer */}
      <footer className="mt-12 bg-white/50 backdrop-blur-sm border-t border-blue-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-center items-center space-x-8 text-gray-600">
            <div className="flex items-center space-x-2">
              <MessageCircle className="w-5 h-5 text-voice-primary" />
              <span className="text-sm">Real-time Conversation</span>
            </div>
            <div className="flex items-center space-x-2">
              <Database className="w-5 h-5 text-voice-secondary" />
              <span className="text-sm">MSSQL Search</span>
            </div>
            <div className="flex items-center space-x-2">
              <Mic className="w-5 h-5 text-voice-accent" />
              <span className="text-sm">Voice Interaction</span>
            </div>
            <div className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-voice-primary" />
              <span className="text-sm">AI Powered</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
