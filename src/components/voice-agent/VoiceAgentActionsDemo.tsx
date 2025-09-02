'use client'

import React, { useState } from 'react'
import { useButtonActionManager } from '@/lib/hooks/use-button-action-manager'
import { useNavigationActionManager } from '@/lib/hooks/use-navigation-action-manager'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Search, 
  Info, 
  Play, 
  FileText, 
  Upload, 
  Database, 
  Users, 
  Settings,
  Navigation,
  MousePointer,
  Zap,
  Globe,
  ArrowLeft,
  RefreshCw,
  LogOut
} from 'lucide-react'

export function VoiceAgentActionsDemo() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedTab, setSelectedTab] = useState('overview')
  
  // Button Action Manager
  const {
    getAvailableActions: getAvailableButtonActions,
    getActionsByCategory: getButtonActionsByCategory,
    getTotalActionCount: getTotalButtonActionCount,
    getCategoryCount: getButtonCategoryCount
  } = useButtonActionManager()

  // Navigation Action Manager
  const {
    getAvailableActions: getAvailableNavigationActions,
    getActionsByCategory: getNavigationActionsByCategory,
    getTotalActionCount: getTotalNavigationActionCount,
    getCategoryCount: getNavigationCategoryCount,
    getCurrentPage,
    getNavigationHistory,
    goToDashboard,
    goToDatabaseQuery,
    goToFileQuery,
    goBack,
    refreshPage
  } = useNavigationActionManager()

  const allButtonActions = getAvailableButtonActions()
  const allNavigationActions = getAvailableNavigationActions()
  const buttonCategories = getButtonActionsByCategory('Reports') // Example category
  const navigationCategories = getNavigationActionsByCategory('Core Navigation') // Example category
  const currentPage = getCurrentPage()
  const navigationHistory = getNavigationHistory()

  const getActionTypeIcon = (actionType: string) => {
    switch (actionType) {
      case 'view_report':
        return <FileText className="w-4 h-4" />
      case 'generate_report':
        return <Play className="w-4 h-4" />
      case 'file_upload':
        return <Upload className="w-4 h-4" />
      case 'file_search':
        return <Search className="w-4 h-4" />
      case 'page_navigation':
        return <Navigation className="w-4 h-4" />
      case 'logout':
        return <LogOut className="w-4 h-4" />
      case 'page_refresh':
        return <RefreshCw className="w-4 h-4" />
      case 'navigation_back':
        return <ArrowLeft className="w-4 h-4" />
      default:
        return <Zap className="w-4 h-4" />
    }
  }

  const getActionTypeColor = (actionType: string) => {
    switch (actionType) {
      case 'view_report':
        return 'bg-blue-500'
      case 'generate_report':
        return 'bg-green-500'
      case 'file_upload':
        return 'bg-purple-500'
      case 'file_search':
        return 'bg-orange-500'
      case 'page_navigation':
        return 'bg-indigo-500'
      case 'logout':
        return 'bg-red-500'
      case 'page_refresh':
        return 'bg-yellow-500'
      case 'navigation_back':
        return 'bg-gray-500'
      default:
        return 'bg-gray-500'
    }
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            Voice Agent Actions & Navigation Demo
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Comprehensive demo of the scalable button action and navigation action systems
          </p>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {/* Overview Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{getTotalButtonActionCount()}</div>
              <div className="text-sm text-blue-500">Button Actions</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{getTotalNavigationActionCount()}</div>
              <div className="text-sm text-green-500">Navigation Actions</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{getButtonCategoryCount()}</div>
              <div className="text-sm text-purple-500">Button Categories</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">{getNavigationCategoryCount()}</div>
              <div className="text-sm text-orange-500">Navigation Categories</div>
            </div>
          </div>

          {/* Current Page and Navigation History */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Globe className="w-4 h-4" />
                  Current Page
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-lg font-medium">{currentPage}</div>
                <div className="text-sm text-muted-foreground">Active page for voice agent</div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Navigation className="w-4 h-4" />
                  Navigation History
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-sm space-y-1">
                  {navigationHistory.slice(-3).map((page, index) => (
                    <div key={index} className="text-muted-foreground">
                      {navigationHistory.length - 3 + index + 1}. {page}
                    </div>
                  ))}
                  {navigationHistory.length === 0 && (
                    <div className="text-muted-foreground">No navigation history</div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Quick Navigation Actions */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Navigation className="w-4 h-4" />
                Quick Navigation Actions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                <Button onClick={() => goToDashboard()} variant="outline" size="sm">
                  Dashboard
                </Button>
                <Button onClick={() => goToDatabaseQuery()} variant="outline" size="sm">
                  Database Query
                </Button>
                <Button onClick={() => goToFileQuery()} variant="outline" size="sm">
                  File Query
                </Button>
                <Button onClick={() => goBack()} variant="outline" size="sm">
                  Go Back
                </Button>
                <Button onClick={() => refreshPage()} variant="outline" size="sm">
                  Refresh Page
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Main Tabs */}
          <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="button-actions">Button Actions</TabsTrigger>
              <TabsTrigger value="navigation-actions">Navigation Actions</TabsTrigger>
              <TabsTrigger value="voice-commands">Voice Commands</TabsTrigger>
            </TabsList>
            
            {/* Overview Tab */}
            <TabsContent value="overview" className="mt-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Button Actions Overview */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <MousePointer className="w-4 h-4" />
                      Button Actions Overview
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {allButtonActions.slice(0, 5).map((action, index) => (
                        <div key={index} className="flex items-center gap-2">
                          <Badge 
                            variant="secondary" 
                            className={`${getActionTypeColor(action.actionType)} text-white text-xs`}
                          >
                            {action.actionType}
                          </Badge>
                          <span className="text-sm">{action.elementName}</span>
                        </div>
                      ))}
                      {allButtonActions.length > 5 && (
                        <div className="text-xs text-muted-foreground">
                          +{allButtonActions.length - 5} more actions...
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Navigation Actions Overview */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Navigation className="w-4 h-4" />
                      Navigation Actions Overview
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {allNavigationActions.slice(0, 5).map((action, index) => (
                        <div key={index} className="flex items-center gap-2">
                          <Badge 
                            variant="secondary" 
                            className={`${getActionTypeColor(action.actionType)} text-white text-xs`}
                          >
                            {action.actionType}
                          </Badge>
                          <span className="text-sm">{action.actionName}</span>
                        </div>
                      ))}
                      {allNavigationActions.length > 5 && (
                        <div className="text-xs text-muted-foreground">
                          +{allNavigationActions.length - 5} more actions...
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            {/* Button Actions Tab */}
            <TabsContent value="button-actions" className="mt-6">
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <Input
                    placeholder="Search button actions..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="max-w-md"
                  />
                  <Badge variant="outline">
                    {allButtonActions.length} total actions
                  </Badge>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {allButtonActions.map((action, index) => (
                    <Card key={index} className="hover:shadow-md transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Badge 
                              variant="secondary" 
                              className={`${getActionTypeColor(action.actionType)} text-white`}
                            >
                              {action.actionType}
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {action.context}
                          </div>
                        </div>
                        
                        <h4 className="font-medium mb-2">{action.elementName}</h4>
                        <p className="text-sm text-muted-foreground mb-3">
                          {action.description}
                        </p>
                        
                        {action.aliases && action.aliases.length > 0 && (
                          <div className="space-y-1">
                            <div className="text-xs font-medium text-muted-foreground">Aliases:</div>
                            <div className="flex flex-wrap gap-1">
                              {action.aliases.slice(0, 2).map((alias, aliasIndex) => (
                                <Badge key={aliasIndex} variant="outline" className="text-xs">
                                  {alias}
                                </Badge>
                              ))}
                              {action.aliases.length > 2 && (
                                <Badge variant="outline" className="text-xs">
                                  +{action.aliases.length - 2} more
                                </Badge>
                              )}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </TabsContent>

            {/* Navigation Actions Tab */}
            <TabsContent value="navigation-actions" className="mt-6">
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <Badge variant="outline">
                    {allNavigationActions.length} total actions
                  </Badge>
                  <Badge variant="outline">
                    Current page: {currentPage}
                  </Badge>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {allNavigationActions.map((action, index) => (
                    <Card key={index} className="hover:shadow-md transition-shadow">
                      <CardContent className="p-4">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <Badge 
                              variant="secondary" 
                              className={`${getActionTypeColor(action.actionType)} text-white`}
                            >
                              {action.actionType}
                            </Badge>
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {action.context}
                          </div>
                        </div>
                        
                        <h4 className="font-medium mb-2">{action.actionName}</h4>
                        <p className="text-sm text-muted-foreground mb-2">
                          {action.description}
                        </p>
                        
                        <div className="text-xs text-muted-foreground mb-3">
                          Target: {action.targetPage}
                        </div>
                        
                        {action.aliases && action.aliases.length > 0 && (
                          <div className="space-y-1">
                            <div className="text-xs font-medium text-muted-foreground">Aliases:</div>
                            <div className="flex flex-wrap gap-1">
                              {action.aliases.slice(0, 2).map((alias, aliasIndex) => (
                                <Badge key={aliasIndex} variant="outline" className="text-xs">
                                  {alias}
                                </Badge>
                              ))}
                              {action.aliases.length > 2 && (
                                <Badge variant="outline" className="text-xs">
                                  +{action.aliases.length - 2} more
                                </Badge>
                              )}
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            </TabsContent>

            {/* Voice Commands Tab */}
            <TabsContent value="voice-commands" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    Voice Command Examples
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Button Action Commands */}
                    <div>
                      <h4 className="font-medium mb-3 text-blue-600">Button Actions</h4>
                      <div className="space-y-2 text-sm">
                        <div className="p-2 bg-blue-50 rounded">
                          "Click view report" → Executes view report action
                        </div>
                        <div className="p-2 bg-blue-50 rounded">
                          "Generate report" → Executes report generation
                        </div>
                        <div className="p-2 bg-blue-50 rounded">
                          "Upload file" → Executes file upload action
                        </div>
                        <div className="p-2 bg-blue-50 rounded">
                          "Search files" → Executes file search action
                        </div>
                      </div>
                    </div>

                    {/* Navigation Commands */}
                    <div>
                      <h4 className="font-medium mb-3 text-green-600">Navigation Actions</h4>
                      <div className="space-y-2 text-sm">
                        <div className="p-2 bg-green-50 rounded">
                          "Go to dashboard" → Navigates to dashboard
                        </div>
                        <div className="p-2 bg-green-50 rounded">
                          "Go to database query" → Navigates to database page
                        </div>
                        <div className="p-2 bg-green-50 rounded">
                          "Go back" → Returns to previous page
                        </div>
                        <div className="p-2 bg-green-50 rounded">
                          "Refresh page" → Reloads current page
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* System Information */}
          <Card className="bg-gray-50 border-gray-200">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-gray-600 mt-0.5" />
                <div>
                  <h4 className="font-medium text-gray-900 mb-2">How the Scalable System Works</h4>
                  <ul className="text-sm text-gray-800 space-y-1">
                    <li>• <strong>Configuration-Driven:</strong> All actions are defined in configuration files</li>
                    <li>• <strong>Automatic Registration:</strong> Actions are automatically registered on startup</li>
                    <li>• <strong>Alias Support:</strong> Multiple ways to reference the same action</li>
                    <li>• <strong>Condition Checking:</strong> Actions check permissions and requirements</li>
                    <li>• <strong>Easy Extension:</strong> Add new actions by updating configuration files</li>
                    <li>• <strong>Voice Integration:</strong> All actions are available to the voice agent</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </CardContent>
      </Card>
    </div>
  )
} 