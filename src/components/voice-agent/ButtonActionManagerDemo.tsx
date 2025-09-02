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
  Edit
} from 'lucide-react'

export function ButtonActionManagerDemo() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  
  const {
    getAvailableActions,
    getActionsByCategory,
    getActionCategories,
    searchActions,
    getTotalActionCount,
    getCategoryCount
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
      case 'reports':
        return <FileText className="w-4 h-4" />
      case 'file operations':
        return <Upload className="w-4 h-4" />
      case 'navigation':
        return <Database className="w-4 h-4" />
      case 'data operations':
        return <Database className="w-4 h-4" />
      case 'user management':
        return <Users className="w-4 h-4" />
      case 'business rules':
        return <Settings className="w-4 h-4" />
      default:
        return <Info className="w-4 h-4" />
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
      case 'table_creation':
        return 'bg-teal-500'
      case 'user_creation':
        return 'bg-pink-500'
      case 'business_rule_creation':
        return 'bg-amber-500'
      default:
        return 'bg-gray-500'
    }
  }

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="w-5 h-5" />
            Button Action Manager Demo
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            This demo shows the scalable button action system that automatically manages voice agent interactions
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
            <div className="flex gap-2">
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
                  <Card key={index}>
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <Badge 
                            variant="secondary" 
                            className={`${getActionTypeColor(action.actionType)} text-white`}
                          >
                            {action.actionType}
                          </Badge>
                          <div>
                            <h4 className="font-medium">{action.elementName}</h4>
                            <p className="text-sm text-muted-foreground">{action.description}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-xs text-muted-foreground">{action.context}</div>
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

          {/* Action Registration Info */}
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
                    <li>• Voice agent can execute any registered action</li>
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