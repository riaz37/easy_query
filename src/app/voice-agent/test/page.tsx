'use client'

import { useState, useEffect } from 'react'
import { VoiceNavigationHandler } from '@/lib/voice-agent/voice-navigation-handler'

export default function VoiceAgentTestPage() {
  const [navigationHandler, setNavigationHandler] = useState<VoiceNavigationHandler | null>(null)
  const [testResults, setTestResults] = useState<string[]>([])

  useEffect(() => {
    // Initialize navigation handler for testing
    const handler = new VoiceNavigationHandler()
    setNavigationHandler(handler)

    return () => {
      handler.disconnect()
    }
  }, [])

  const addTestResult = (result: string) => {
    setTestResults(prev => [...prev, `${new Date().toLocaleTimeString()}: ${result}`])
  }

  const testDatabaseSearch = () => {
    addTestResult('Testing database search...')
    const event = new CustomEvent('voice-database-search', {
      detail: { 
        query: 'financial reports',
        user_id: 'test-user-123'
      }
    })
    window.dispatchEvent(event)
  }

  const testFileSearch = () => {
    addTestResult('Testing file search...')
    const event = new CustomEvent('voice-file-search', {
      detail: { 
        search_query: 'sales data',
        table_specific: true,
        tables: ['sales', 'finance'],
        user_id: 'test-user-123'
      }
    })
    window.dispatchEvent(event)
  }

  const testFileUpload = () => {
    addTestResult('Testing file upload...')
    const event = new CustomEvent('voice-file-upload', {
      detail: { 
        upload_request: 'sales data',
        file_descriptions: ['sales data'],
        table_names: ['finance'],
        user_id: 'test-user-123'
      }
    })
    window.dispatchEvent(event)
  }

  const testViewReport = () => {
    addTestResult('Testing view report...')
    const event = new CustomEvent('voice-view-report', {
      detail: { 
        reportRequest: 'monthly sales report',
        user_id: 'test-user-123'
      }
    })
    window.dispatchEvent(event)
  }

  const testGenerateReport = () => {
    addTestResult('Testing generate report...')
    const event = new CustomEvent('voice-generate-report', {
      detail: { 
        reportQuery: 'quarterly financial summary',
        user_id: 'test-user-123'
      }
    })
    window.dispatchEvent(event)
  }

  const testSetDatabase = () => {
    addTestResult('Testing set database...')
    const event = new CustomEvent('voice-set-database', {
      detail: { 
        dbId: 'finance_db',
        user_id: 'test-user-123'
      }
    })
    window.dispatchEvent(event)
  }

  const clearResults = () => {
    setTestResults([])
  }

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Voice Agent Test Page
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Test the voice navigation system and event handling
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Test Controls */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Test Controls</h2>
          
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={testDatabaseSearch}
              className="p-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              Test Database Search
            </button>
            
            <button
              onClick={testFileSearch}
              className="p-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
            >
              Test File Search
            </button>
            
            <button
              onClick={testFileUpload}
              className="p-3 bg-purple-500 text-white rounded-lg hover:bg-purple-600 transition-colors"
            >
              Test File Upload
            </button>
            
            <button
              onClick={testViewReport}
              className="p-3 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
            >
              Test View Report
            </button>
            
            <button
              onClick={testGenerateReport}
              className="p-3 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
            >
              Test Generate Report
            </button>
            
            <button
              onClick={testSetDatabase}
              className="p-3 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors"
            >
              Test Set Database
            </button>
          </div>

          <button
            onClick={clearResults}
            className="w-full p-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
          >
            Clear Results
          </button>
        </div>

        {/* Test Results */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Test Results</h2>
          
          <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg h-96 overflow-y-auto">
            {testResults.length === 0 ? (
              <p className="text-gray-500 dark:text-gray-400 text-center mt-8">
                No test results yet. Click the test buttons to see results.
              </p>
            ) : (
              <div className="space-y-2">
                {testResults.map((result, index) => (
                  <div key={index} className="text-sm p-2 bg-white dark:bg-gray-700 rounded border">
                    {result}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Element Highlighting Test */}
      <div className="mt-8 space-y-4">
        <h2 className="text-xl font-semibold">Element Highlighting Test</h2>
        <p className="text-gray-600 dark:text-gray-400">
          These elements have voice navigation data attributes and should be highlightable:
        </p>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 border rounded-lg">
            <input
              type="text"
              placeholder="Database search input"
              className="w-full p-2 border rounded"
              data-element="search"
              data-voice-target="test-database-search"
            />
          </div>
          
          <div className="p-4 border rounded-lg">
            <button
              className="w-full p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              data-element="search"
              data-voice-target="test-search-button"
            >
              Search Button
            </button>
          </div>
          
          <div className="p-4 border rounded-lg">
            <div
              className="w-full h-20 border-2 border-dashed border-gray-300 rounded-lg flex items-center justify-center cursor-pointer hover:border-blue-400"
              data-element="upload"
              data-voice-target="test-upload-area"
            >
              Upload Area
            </div>
          </div>
          
          <div className="p-4 border rounded-lg">
            <button
              className="w-full p-2 bg-green-500 text-white rounded hover:bg-green-600"
              data-element="upload"
              data-voice-target="test-upload-button"
            >
              Upload Button
            </button>
          </div>
        </div>
      </div>
    </div>
  )
} 