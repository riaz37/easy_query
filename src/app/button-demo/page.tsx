'use client'

import React, { useState } from 'react'
import { useButtonActionManager } from '@/lib/hooks/use-button-action-manager'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Search, 
  Filter, 
  Info, 
  Play, 
  FileText, 
  Upload, 
  Database, 
  Users, 
  Settings,
  Plus,
  Trash2,
  Edit,
  MousePointer,
  Zap,
  Globe,
  Navigation,
  ArrowLeft,
  RefreshCw,
  LogOut,
  Eye,
  Download,
  Save,
  X
} from 'lucide-react'

export default function ButtonDemoPage() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [selectedAction, setSelectedAction] = useState<any>(null)
  
  const {
    getAvailableActions,
    getActionsByCategory,
    getActionCategories,
    searchActions,
    getTotalActionCount,
    getCategoryCount,
    isActionAvailable,
    getActionInfo
  } = useButtonActionManager()

  const allActions = getAvailableActions()
  const categories = getActionCategories()
  const filteredActions = searchTerm 
    ? searchActions(searchTerm)
    : selectedCategory === 'all' 
      ? allActions 
      : getActionsByCategory(selectedCategory)

  const getCategoryIcon = (category: string) => {
    switch (category.toLowerCase()) {
      case 'navigation':
        return <Navigation className="w-4 h-4" />
      case 'forms':
        return <FileText className="w-4 h-4" />
      case 'files':
        return <Upload className="w-4 h-4" />
      case 'dialogs':
        return <Eye className="w-4 h-4" />
      case 'search':
        return <Search className="w-4 h-4" />
      case 'crud':
        return <Database className="w-4 h-4" />
      case 'user-management':
        return <Users className="w-4 h-4" />
      case 'generic':
        return <MousePointer className="w-4 h-4" />
      default:
        return <Info className="w-4 h-4" />
    }
  }

  const getActionTypeColor = (category: string) => {
    switch (category) {
      case 'navigation':
        return 'bg-blue-500'
      case 'forms':
        return 'bg-green-500'
      case 'files':
        return 'bg-purple-500'
      case 'dialogs':
        return 'bg-orange-500'
      case 'search':
        return 'bg-indigo-500'
      case 'crud':
        return 'bg-teal-500'
      case 'user-management':
        return 'bg-pink-500'
      case 'generic':
        return 'bg-gray-500'
      default:
        return 'bg-gray-500'
    }
  }

  const handleActionClick = (action: any) => {
    setSelectedAction(action)
    // Execute the action
    if (action.handler) {
      try {
        action.handler()
        console.log(`🎯 Executed action: ${action.name}`)
      } catch (error) {
        console.error(`🎯 Error executing action: ${action.name}`, error)
      }
    }
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Button Action Demo</h1>
        <p className="text-muted-foreground">
          Interactive showcase of the scalable button action system. Click any action to see it in action!
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MousePointer className="w-5 h-5" />
                Available Button Actions
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                {getTotalActionCount()} total actions across {getCategoryCount()} categories
              </p>
            </CardHeader>
            
            <CardContent className="space-y-6">
              {/* Statistics */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{getTotalActionCount()}</div>
                  <div className="text-sm text-blue-500">Total Actions</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{getCategoryCount()}</div>
                  <div className="text-sm text-green-500">Categories</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">{filteredActions.length}</div>
                  <div className="text-sm text-purple-500">Filtered Actions</div>
                </div>
              </div>

              {/* Search and Filter */}
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                    <Input
                      placeholder="Search button actions..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <Button
                    variant={selectedCategory === 'all' ? 'default' : 'outline'}
                    onClick={() => setSelectedCategory('all')}
                    size="sm"
                  >
                    All
                  </Button>
                  {categories.map(category => (
                    <Button
                      key={category}
                      variant={selectedCategory === category ? 'default' : 'outline'}
                      onClick={() => setSelectedCategory(category)}
                      size="sm"
                    >
                      {getCategoryIcon(category)}
                      <span className="ml-2">{category}</span>
                    </Button>
                  ))}
                </div>
              </div>

              {/* Actions Display */}
              <Tabs defaultValue="grid" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="grid">Grid View</TabsTrigger>
                  <TabsTrigger value="list">List View</TabsTrigger>
                </TabsList>
                
                <TabsContent value="grid" className="mt-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {filteredActions.map((action, index) => (
                      <Card 
                        key={index} 
                        className="hover:shadow-md transition-shadow cursor-pointer"
                        onClick={() => handleActionClick(action)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex items-center gap-2">
                              <Badge 
                                variant="secondary" 
                                className={`${getActionTypeColor(action.category || 'default')} text-white`}
                              >
                                {action.category || 'default'}
                              </Badge>
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {action.category || 'uncategorized'}
                            </div>
                          </div>
                          
                          <h4 className="font-medium mb-2">{action.name}</h4>
                          <p className="text-sm text-muted-foreground mb-3">
                            {action.description || 'No description available'}
                          </p>
                          
                          {action.aliases && action.aliases.length > 0 && (
                            <div className="space-y-1">
                              <div className="text-xs font-medium text-muted-foreground">Aliases:</div>
                              <div className="flex flex-wrap gap-1">
                                {action.aliases.slice(0, 3).map((alias, aliasIndex) => (
                                  <Badge key={aliasIndex} variant="outline" className="text-xs">
                                    {alias}
                                  </Badge>
                                ))}
                                {action.aliases.length > 3 && (
                                  <Badge variant="outline" className="text-xs">
                                    +{action.aliases.length - 3} more
                                  </Badge>
                                )}
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </TabsContent>
                
                <TabsContent value="list" className="mt-6">
                  <div className="space-y-2">
                    {filteredActions.map((action, index) => (
                      <Card 
                        key={index} 
                        className="cursor-pointer hover:shadow-md transition-shadow"
                        onClick={() => handleActionClick(action)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <Badge 
                                variant="secondary" 
                                className={`${getActionTypeColor(action.category || 'default')} text-white`}
                              >
                                {action.category || 'default'}
                              </Badge>
                              <div>
                                <h4 className="font-medium">{action.name}</h4>
                                <p className="text-sm text-muted-foreground">{action.description || 'No description available'}</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-xs text-muted-foreground">{action.category || 'uncategorized'}</div>
                              {action.aliases && action.aliases.length > 0 && (
                                <div className="text-xs text-muted-foreground">
                                  {action.aliases.length} alias{action.aliases.length !== 1 ? 'es' : ''}
                                </div>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </TabsContent>
              </Tabs>

              {/* No Results */}
              {filteredActions.length === 0 && (
                <div className="text-center py-8">
                  <Info className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-muted-foreground mb-2">
                    No actions found
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {searchTerm 
                      ? `No actions match "${searchTerm}"`
                      : 'No actions available in this category'
                    }
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Action Details */}
          {selectedAction && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Action Details
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedAction(null)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-medium text-lg">{selectedAction.name}</h4>
                  <Badge 
                    variant="secondary" 
                    className={`${getActionTypeColor(selectedAction.category || 'default')} text-white mt-1`}
                  >
                    {selectedAction.category || 'default'}
                  </Badge>
                </div>
                
                {selectedAction.description && (
                  <div>
                    <h5 className="font-medium text-sm mb-1">Description</h5>
                    <p className="text-sm text-muted-foreground">{selectedAction.description}</p>
                  </div>
                )}
                
                {selectedAction.aliases && selectedAction.aliases.length > 0 && (
                  <div>
                    <h5 className="font-medium text-sm mb-2">Aliases</h5>
                    <div className="flex flex-wrap gap-1">
                      {selectedAction.aliases.map((alias, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {alias}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
                
                <div>
                  <h5 className="font-medium text-sm mb-2">Action ID</h5>
                  <code className="text-xs bg-gray-100 px-2 py-1 rounded">{selectedAction.id}</code>
                </div>
                
                <Button 
                  onClick={() => handleActionClick(selectedAction)}
                  className="w-full"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Execute Action
                </Button>
              </CardContent>
            </Card>
          )}

          {/* System Information */}
          <Card className="bg-blue-50 border-blue-200">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-600 mt-0.5" />
                <div>
                  <h4 className="font-medium text-blue-900 mb-2">How It Works</h4>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• Button actions are automatically registered from configuration files</li>
                    <li>• New actions can be added by updating the configuration</li>
                    <li>• Aliases provide multiple ways to reference the same action</li>
                    <li>• Actions are categorized for better organization</li>
                    <li>• Click any action to see it execute in real-time</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Quick Test Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full justify-start"
                onClick={() => {
                  const action = getActionInfo('dashboard')
                  if (action) handleActionClick(action)
                }}
              >
                <Globe className="w-4 h-4 mr-2" />
                Test Dashboard Navigation
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full justify-start"
                onClick={() => {
                  const action = getActionInfo('search')
                  if (action) handleActionClick(action)
                }}
              >
                <Search className="w-4 h-4 mr-2" />
                Test Search Action
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full justify-start"
                onClick={() => {
                  const action = getActionInfo('upload')
                  if (action) handleActionClick(action)
                }}
              >
                <Upload className="w-4 h-4 mr-2" />
                Test Upload Action
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
