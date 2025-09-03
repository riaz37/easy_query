'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Play,
  Square,
  Settings,
  Info,
  Download,
  Upload,
  Save,
  Search,
  Filter,
  Plus,
  Edit,
  Trash2,
  RefreshCw
} from 'lucide-react'
import { BUTTON_MAPPING_CONFIG, getButtonMapping } from '@/lib/voice-agent/config/default-button-actions'

export function ButtonSystemExample() {
  const [registeredActions, setRegisteredActions] = useState<any[]>([])
  const [executionHistory, setExecutionHistory] = useState<any[]>([])
  const [stats, setStats] = useState<any>({})
  const [customActionName, setCustomActionName] = useState('')
  const [customActionCode, setCustomActionCode] = useState('console.log("Custom action executed!")')
  const [testElementName, setTestElementName] = useState('')

  // Load data on mount
  useEffect(() => {
    loadData()

    // Listen for button executions
    const handleButtonExecution = (event: any) => {
      loadData() // Refresh data when buttons are executed
    }

    window.addEventListener('button-action-executed', handleButtonExecution)

    return () => {
      window.removeEventListener('button-action-executed', handleButtonExecution)
    }
  }, [])

  const loadData = () => {
    setRegisteredActions(buttonRegistrationService.getRegisteredActions())
    setExecutionHistory(buttonRegistrationService.getExecutionHistory().slice(-10))
    setStats(buttonRegistrationService.getExecutionStats())
  }

  const registerCustomAction = () => {
    if (!customActionName.trim()) return

    try {
      // Create handler function from code
      const handler = new Function('context', customActionCode)

      buttonRegistrationService.registerCustomAction({
        id: customActionName.toLowerCase().replace(/\s+/g, '-'),
        name: customActionName,
        aliases: [customActionName.toLowerCase()],
        category: 'custom-demo',
        handler: handler
      })

      setCustomActionName('')
      setCustomActionCode('console.log("Custom action executed!")')
      loadData()
    } catch (error) {
      console.error('Error registering custom action:', error)
      alert('Error in custom action code: ' + error.message)
    }
  }

  const testButtonExecution = async () => {
    if (!testElementName.trim()) return

    try {
      const result = await buttonRegistry.execute(testElementName, {
        elementName: testElementName,
        source: 'test',
        timestamp: new Date().toISOString(),
        context: { test: true }
      })

      console.log('Test execution result:', result)
      loadData()
    } catch (error) {
      console.error('Error testing button execution:', error)
    }
  }

  const categories = buttonRegistrationService.getCategories()

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white flex items-center justify-center gap-3">
            <Settings className="w-10 h-10 text-blue-500" />
            Button System Example
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Demonstration of the scalable button registration and execution system
          </p>
        </div>

        {/* Statistics */}
        <Card className="border-2 border-blue-200 dark:border-blue-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Info className="w-5 h-5" />
              System Statistics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{stats.totalActions || 0}</div>
                <div className="text-sm text-gray-600">Total Actions</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{stats.totalCategories || 0}</div>
                <div className="text-sm text-gray-600">Categories</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{stats.successfulExecutions || 0}</div>
                <div className="text-sm text-gray-600">Successful</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">{stats.failedExecutions || 0}</div>
                <div className="text-sm text-gray-600">Failed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">{stats.averageExecutionTime || 0}ms</div>
                <div className="text-sm text-gray-600">Avg Time</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Demo Buttons */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="w-5 h-5" />
                Demo Buttons
                <Badge variant="outline">Try voice/text: "Click save button"</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Save clicked')}
                >
                  <Save className="w-4 h-4" />
                  Save
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Upload clicked')}
                >
                  <Upload className="w-4 h-4" />
                  Upload
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Download clicked')}
                >
                  <Download className="w-4 h-4" />
                  Download
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Search clicked')}
                >
                  <Search className="w-4 h-4" />
                  Search
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Filter clicked')}
                >
                  <Filter className="w-4 h-4" />
                  Filter
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Add clicked')}
                >
                  <Plus className="w-4 h-4" />
                  Add
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Edit clicked')}
                >
                  <Edit className="w-4 h-4" />
                  Edit
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Delete clicked')}
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </Button>
                <Button
                  variant="outline"
                  className="flex items-center gap-2"
                  onClick={() => console.log('Vector DB clicked')}
                >
                  Vector Database Access
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Custom Action Registration */}
          <Card>
            <CardHeader>
              <CardTitle>Register Custom Action</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Input
                placeholder="Action name (e.g., 'My Custom Button')"
                value={customActionName}
                onChange={(e) => setCustomActionName(e.target.value)}
              />
              <Textarea
                placeholder="JavaScript handler code"
                value={customActionCode}
                onChange={(e) => setCustomActionCode(e.target.value)}
                rows={4}
              />
              <Button onClick={registerCustomAction} disabled={!customActionName.trim()}>
                Register Action
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Test Section */}
        <Card>
          <CardHeader>
            <CardTitle>Test Button Execution</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="Element name to test (e.g., 'save', 'upload', 'vector database access')"
                value={testElementName}
                onChange={(e) => setTestElementName(e.target.value)}
                className="flex-1"
              />
              <Button onClick={testButtonExecution} disabled={!testElementName.trim()}>
                Test Execute
              </Button>
              <Button onClick={loadData} variant="outline">
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Tabs for detailed views */}
        <Tabs defaultValue="registered" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="registered">Registered Actions</TabsTrigger>
            <TabsTrigger value="categories">Categories</TabsTrigger>
            <TabsTrigger value="history">Execution History</TabsTrigger>
          </TabsList>

          <TabsContent value="registered" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>All Registered Actions ({registeredActions.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {registeredActions.map((action, index) => (
                      <div key={index} className="p-3 border rounded-lg">
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="font-medium">{action.name}</div>
                            <div className="text-sm text-gray-600">ID: {action.id}</div>
                            {action.aliases && action.aliases.length > 0 && (
                              <div className="text-xs text-gray-500">
                                Aliases: {action.aliases.join(', ')}
                              </div>
                            )}
                          </div>
                          <Badge variant="outline">{action.category}</Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="categories" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Actions by Category</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {categories.map(category => {
                    const categoryActions = buttonRegistrationService.getActionsByCategory(category)
                    return (
                      <div key={category} className="p-3 border rounded-lg">
                        <div className="font-medium mb-2 flex items-center justify-between">
                          {category}
                          <Badge variant="secondary">{categoryActions.length}</Badge>
                        </div>
                        <div className="text-sm space-y-1">
                          {categoryActions.slice(0, 5).map(action => (
                            <div key={action.id} className="text-gray-600">
                              • {action.name}
                            </div>
                          ))}
                          {categoryActions.length > 5 && (
                            <div className="text-gray-500">...and {categoryActions.length - 5} more</div>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Recent Execution History</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-64">
                  <div className="space-y-2">
                    {executionHistory.map((execution, index) => (
                      <div key={index} className="p-3 border rounded-lg">
                        <div className="flex justify-between items-start">
                          <div>
                            <div className="font-medium">{execution.elementName}</div>
                            <div className="text-sm text-gray-600">
                              Time: {execution.executionTime}ms
                            </div>
                            {execution.error && (
                              <div className="text-xs text-red-600">
                                Error: {execution.error}
                              </div>
                            )}
                          </div>
                          <Badge variant={execution.success ? 'default' : 'destructive'}>
                            {execution.success ? 'Success' : 'Failed'}
                          </Badge>
                        </div>
                      </div>
                    ))}
                    {executionHistory.length === 0 && (
                      <div className="text-center text-gray-500 py-8">
                        No execution history yet. Try clicking some buttons or testing execution!
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Instructions */}
        <Card className="bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800">
          <CardHeader>
            <CardTitle className="text-green-900 dark:text-green-100">
              How to Test the Button System
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-green-800 dark:text-green-200">
            <p><strong>Voice Commands:</strong> "Click save button", "Press upload", "Execute vector database access"</p>
            <p><strong>Text Commands:</strong> Type the same commands in the text chat</p>
            <p><strong>Direct Testing:</strong> Use the test input above to execute actions directly</p>
            <p><strong>Custom Actions:</strong> Register your own actions and test them immediately</p>
            <p><strong>Monitoring:</strong> Watch the execution history and statistics update in real-time</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
