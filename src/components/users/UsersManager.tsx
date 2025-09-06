"use client";

import React, { useState, useMemo, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Users, 
  Database, 
  Brain, 
  Plus, 
  Search, 
  Edit, 
  Trash2, 
  Shield,
  UserCheck,
  AlertCircle,
  RefreshCw,
  // Loader2
} from "lucide-react";
import { PageLoader, InlineLoader } from "@/components/ui/loading";
import { useUsersManager } from "./hooks/useUsersManager";
import { CreateDatabaseAccessModal } from "./modals/CreateDatabaseAccessModal";
import { CreateVectorDBAccessModal } from "./modals/CreateVectorDBAccessModal";
import { useTheme } from "@/store/theme-store";

export function UsersManager() {
  // Theme
  const theme = useTheme();
  const isDark = theme === 'dark';
  
  // Local state for modals
  const [activeTab, setActiveTab] = useState("mssql");
  const [isDatabaseModalOpen, setIsDatabaseModalOpen] = useState(false);
  const [isVectorDBModalOpen, setIsVectorDBModalOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<string>("");
  const [editingUser, setEditingUser] = useState<string>("");

  const {
    searchTerm,
    setSearchTerm,
    userAccessConfigs,
    userConfigs,
    userConfigLoading,
    loadUserAccessConfigs,
    loadUserConfigs,
    extractNameFromEmail,
    availableDatabases,
  } = useUsersManager();

  // Load data on component mount
  useEffect(() => {
    loadUserAccessConfigs();
    loadUserConfigs();
  }, [loadUserAccessConfigs, loadUserConfigs]);

  // Filtered data based on search term
  const filteredUserAccess = useMemo(() => {
    if (!userAccessConfigs || !Array.isArray(userAccessConfigs)) return [];
    if (!searchTerm.trim()) return userAccessConfigs;
    
    return userAccessConfigs.filter(config =>
      config && config.user_id && (
        config.user_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        extractNameFromEmail(config.user_id).toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
  }, [userAccessConfigs, searchTerm, extractNameFromEmail]);

  // Filter users by access type
  const mssqlUsers = useMemo(() => {
    return filteredUserAccess.filter((config) =>
      config.database_access?.parent_databases?.length > 0 ||
      config.database_access?.sub_databases?.some((sub: any) => sub.databases?.length > 0)
    );
  }, [filteredUserAccess]);

  const vectorDBUsers = useMemo(() => {
    return userConfigs.filter((config) => 
      config.db_id && config.table_names && config.table_names.length > 0
    );
  }, [userConfigs]);

  // Handle modal operations
  const handleCreateMSSQLAccess = () => {
    setSelectedUser("");
    setEditingUser("");
    setIsDatabaseModalOpen(true);
  };

  const handleCreateVectorDBAccess = () => {
    setSelectedUser("");
    setEditingUser("");
    setIsVectorDBModalOpen(true);
  };

  const handleEditUser = (userId: string, type: 'mssql' | 'vector') => {
    setSelectedUser(userId);
    setEditingUser(userId);
    if (type === 'mssql') {
      setIsDatabaseModalOpen(true);
    } else {
      setIsVectorDBModalOpen(true);
    }
  };

  const handleModalSuccess = () => {
    loadUserAccessConfigs();
    loadUserConfigs();
    setIsDatabaseModalOpen(false);
    setIsVectorDBModalOpen(false);
    setSelectedUser("");
    setEditingUser("");
  };

  const handleModalClose = () => {
    setIsDatabaseModalOpen(false);
    setIsVectorDBModalOpen(false);
    setSelectedUser("");
    setEditingUser("");
  };

  // Helper functions
  const getDatabaseName = (dbId: number) => {
    const database = availableDatabases?.find(db => db.db_id === dbId);
    return database ? database.db_name : `DB ${dbId}`;
  };

  const formatTableNames = (tableNames: string[]) => {
    if (!tableNames || tableNames.length === 0) return 'No tables';
    if (tableNames.length <= 3) return tableNames.join(', ');
    return `${tableNames.slice(0, 3).join(', ')} +${tableNames.length - 3} more`;
  };

  const getAccessLevelBadge = (config: any) => {
    const hasMSSQL = config.database_access?.parent_databases?.length > 0 || 
                     config.database_access?.sub_databases?.some((sub: any) => sub.databases?.length > 0);
    const hasVectorDB = config.access_level >= 2;
    
    if (hasMSSQL && hasVectorDB) {
      return <Badge className="bg-green-600 hover:bg-green-700">Full Access</Badge>;
    } else if (hasMSSQL) {
      return <Badge className="bg-blue-600 hover:bg-blue-700">MSSQL Only</Badge>;
    } else if (hasVectorDB) {
      return <Badge className="bg-purple-600 hover:bg-purple-700">Vector DB Only</Badge>;
    } else {
      return <Badge variant="secondary">No Access</Badge>;
    }
  };

  const getDatabaseCount = (config: any) => {
    const parentCount = config.database_access?.parent_databases?.length || 0;
    const subCount = config.database_access?.sub_databases?.reduce((total: number, sub: any) => 
      total + (sub.databases?.length || 0), 0) || 0;
    return parentCount + subCount;
  };

  if (userConfigLoading || !activeTab) {
    return (
      <PageLoader
        size="lg"
        variant="primary"
        message="Loading Users"
        description="Please wait while we load user access configurations..."
        showProgress={false}
      />
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className={`text-3xl font-bold ${isDark ? 'text-white' : 'text-gray-900'} flex items-center gap-3`}>
                <Users className="h-8 w-8 text-emerald-400" />
                User Access Management
              </h1>
              <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} mt-2`}>
                Manage user access to MSSQL databases and vector databases
              </p>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={handleCreateMSSQLAccess}
                className="bg-blue-500 hover:bg-blue-600 text-white shadow-lg hover:shadow-blue-500/25 transition-all duration-200"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add MSSQL Access
              </Button>
              <Button
                onClick={handleCreateVectorDBAccess}
                className="bg-purple-500 hover:bg-purple-600 text-white shadow-lg hover:shadow-purple-500/25 transition-all duration-200"
              >
                <Brain className="w-4 h-4 mr-2" />
                Add Vector DB Access
              </Button>
            </div>
          </div>

          {/* Search */}
          <div className="relative max-w-md">
            <Search className={`absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 ${isDark ? 'text-emerald-400' : 'text-emerald-600'}`} />
            <Input
              placeholder="Search users by email..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`pl-10 transition-all duration-200 focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 ${
                isDark 
                  ? 'bg-slate-800/70 border-slate-600 text-white hover:border-slate-500' 
                  : 'bg-white border-gray-300 text-gray-900 hover:border-gray-400'
              } placeholder:${isDark ? 'text-gray-400' : 'text-gray-500'}`}
            />
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card className={`transition-all duration-200 hover:scale-105 hover:shadow-lg ${
            isDark 
              ? "bg-gradient-to-br from-slate-800/80 to-slate-700/80 border-slate-600 hover:border-slate-500" 
              : "bg-gradient-to-br from-white to-gray-50 border-gray-200 shadow-sm hover:shadow-md"
          }`}>
            <CardHeader className="pb-3">
              <CardTitle className={`text-sm font-medium ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>Total Users</CardTitle>
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${isDark ? 'text-emerald-400' : 'text-emerald-600'}`}>{filteredUserAccess.length}</div>
            </CardContent>
          </Card>
          
          <Card className={`transition-all duration-200 hover:scale-105 hover:shadow-lg ${
            isDark 
              ? "bg-gradient-to-br from-blue-900/30 to-blue-800/20 border-blue-600/50 hover:border-blue-500" 
              : "bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 shadow-sm hover:shadow-md"
          }`}>
            <CardHeader className="pb-3">
              <CardTitle className={`text-sm font-medium ${isDark ? 'text-blue-300' : 'text-blue-600'}`}>MSSQL Access</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-500">{mssqlUsers.length}</div>
            </CardContent>
          </Card>
          
          <Card className={`transition-all duration-200 hover:scale-105 hover:shadow-lg ${
            isDark 
              ? "bg-gradient-to-br from-purple-900/30 to-purple-800/20 border-purple-600/50 hover:border-purple-500" 
              : "bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 shadow-sm hover:shadow-md"
          }`}>
            <CardHeader className="pb-3">
              <CardTitle className={`text-sm font-medium ${isDark ? 'text-purple-300' : 'text-purple-600'}`}>Vector DB Access</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-500">{vectorDBUsers.length}</div>
            </CardContent>
          </Card>
          
          <Card className={`transition-all duration-200 hover:scale-105 hover:shadow-lg ${
            isDark 
              ? "bg-gradient-to-br from-emerald-900/30 to-emerald-800/20 border-emerald-600/50 hover:border-emerald-500" 
              : "bg-gradient-to-br from-emerald-50 to-emerald-100 border-emerald-200 shadow-sm hover:shadow-md"
          }`}>
            <CardHeader className="pb-3">
              <CardTitle className={`text-sm font-medium ${isDark ? 'text-emerald-300' : 'text-emerald-600'}`}>Full Access</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-emerald-500">
                {filteredUserAccess.filter(config => 
                  (config.database_access?.parent_databases?.length > 0 || 
                   config.database_access?.sub_databases?.some((sub: any) => sub.databases?.length > 0)) &&
                  config.access_level >= 2
                ).length}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs value={activeTab || "mssql"} onValueChange={setActiveTab} className="w-full">
          <TabsList className={`grid w-full grid-cols-2 transition-all duration-200 ${
            isDark 
              ? 'bg-slate-800/70 border border-slate-600' 
              : 'bg-gray-100 border border-gray-200'
          }`}>
            <TabsTrigger
              value="mssql"
              className={`flex items-center gap-2 transition-all duration-200 ${
                isDark 
                  ? 'text-gray-300 data-[state=active]:bg-blue-600 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-blue-500/25 hover:bg-slate-700 hover:text-white' 
                  : 'text-gray-700 data-[state=active]:bg-blue-500 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-blue-500/25 hover:bg-gray-200 hover:text-gray-900'
              }`}
            >
              <Database className="h-4 w-4" />
              MSSQL Database Access
            </TabsTrigger>
            <TabsTrigger
              value="vector"
              className={`flex items-center gap-2 transition-all duration-200 ${
                isDark 
                  ? 'text-gray-300 data-[state=active]:bg-purple-600 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-purple-500/25 hover:bg-slate-700 hover:text-white' 
                  : 'text-gray-700 data-[state=active]:bg-purple-500 data-[state=active]:text-white data-[state=active]:shadow-lg data-[state=active]:shadow-purple-500/25 hover:bg-gray-200 hover:text-gray-900'
              }`}
            >
              <Brain className="h-4 w-4" />
              Vector Database Access
            </TabsTrigger>
          </TabsList>

          {/* MSSQL Database Access Tab */}
          <TabsContent value="mssql" className="mt-6">
            <Card className={`transition-all duration-200 ${
              isDark 
                ? "bg-gradient-to-br from-slate-800/80 to-slate-700/60 border-slate-600 hover:border-slate-500" 
                : "bg-gradient-to-br from-white to-gray-50 border-gray-200 shadow-sm hover:shadow-md"
            }`}>
              <CardHeader>
                <CardTitle className={`${isDark ? 'text-white' : 'text-gray-900'} flex items-center gap-2`}>
                  <Database className="h-5 w-5 text-blue-500" />
                  MSSQL Database Access Users
                </CardTitle>
                <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} text-sm`}>
                  Users with access to MSSQL databases for data operations
                </p>
              </CardHeader>
              <CardContent>
                {mssqlUsers.length === 0 ? (
                  <div className="text-center py-12">
                    <Database className={`h-12 w-12 ${isDark ? 'text-blue-400' : 'text-blue-500'} mx-auto mb-4`} />
                    <h3 className={`text-lg font-medium ${isDark ? 'text-white' : 'text-gray-900'} mb-2`}>No MSSQL Access Users</h3>
                    <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} mb-4`}>
                      No users have been granted access to MSSQL databases yet.
                    </p>
                    <Button onClick={handleCreateMSSQLAccess} className="bg-blue-500 hover:bg-blue-600 shadow-lg hover:shadow-blue-500/25 transition-all duration-200">
                      <Plus className="w-4 h-4 mr-2" />
                      Grant First Access
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {mssqlUsers.map((config) => (
                      <div
                        key={config.user_id}
                        className={`flex items-center justify-between p-4 rounded-lg border transition-all duration-200 hover:scale-[1.02] hover:shadow-lg ${
                          isDark 
                            ? 'bg-gradient-to-r from-slate-700/40 to-slate-600/30 border-slate-600 hover:border-blue-500/50 hover:shadow-blue-500/10' 
                            : 'bg-gradient-to-r from-gray-50 to-white border-gray-200 hover:border-blue-300 hover:shadow-blue-500/10'
                        }`}
                      >
                        <div className="flex items-center gap-4">
                          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center shadow-lg">
                            <UserCheck className="w-5 h-5 text-white" />
                          </div>
                          <div>
                            <div className="flex items-center gap-3 mb-1">
                              <h4 className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                                {extractNameFromEmail(config.user_id)}
                              </h4>
                              {getAccessLevelBadge(config)}
                            </div>
                            <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{config.user_id}</p>
                            <div className={`flex items-center gap-4 mt-2 text-xs ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
                              <span>Databases: {getDatabaseCount(config)}</span>
                              <span>Sub-companies: {config.sub_company_ids?.length || 0}</span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            onClick={() => handleEditUser(config.user_id, 'mssql')}
                            variant="outline"
                            size="sm"
                            className={`transition-all duration-200 ${
                              isDark 
                                ? "border-blue-500/50 text-blue-300 hover:bg-blue-500/20 hover:border-blue-400" 
                                : "border-blue-300 text-blue-600 hover:bg-blue-50 hover:border-blue-400"
                            }`}
                          >
                            <Edit className="w-4 h-4 mr-2" />
                            Edit
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Vector Database Access Tab */}
          <TabsContent value="vector" className="mt-6">
            <Card className={`transition-all duration-200 ${
              isDark 
                ? "bg-gradient-to-br from-slate-800/80 to-slate-700/60 border-slate-600 hover:border-slate-500" 
                : "bg-gradient-to-br from-white to-gray-50 border-gray-200 shadow-sm hover:shadow-md"
            }`}>
              <CardHeader>
                <CardTitle className={`${isDark ? 'text-white' : 'text-gray-900'} flex items-center gap-2`}>
                  <Brain className="h-5 w-5 text-purple-500" />
                  Vector Database Access Users
                </CardTitle>
                <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} text-sm`}>
                  Users with access to vector databases for AI and ML operations
                </p>
                <div className="flex justify-end mt-2">
                  <Button
                    onClick={loadUserConfigs}
                    variant="outline"
                    size="sm"
                    className={`transition-all duration-200 ${
                      isDark 
                        ? "border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:border-purple-400" 
                        : "border-purple-300 text-purple-600 hover:bg-purple-50 hover:border-purple-400"
                    }`}
                    disabled={userConfigLoading}
                  >
                    {userConfigLoading ? (
                      <InlineLoader size="sm" variant="primary" className="mr-2" />
                    ) : (
                      <RefreshCw className="h-4 w-4 mr-2" />
                    )}
                    Refresh
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                {userConfigLoading ? (
                  <div className="text-center py-12">
                    <PageLoader
                      size="lg"
                      variant="primary"
                      message="Loading Vector DB Access Users"
                      description="Fetching user configurations from the server..."
                      showProgress={false}
                    />
                  </div>
                ) : vectorDBUsers.length === 0 ? (
                  <div className="text-center py-12">
                    <Brain className={`h-12 w-12 ${isDark ? 'text-purple-400' : 'text-purple-500'} mx-auto mb-4`} />
                    <h3 className={`text-lg font-medium ${isDark ? 'text-white' : 'text-gray-900'} mb-2`}>No Vector DB Access Users</h3>
                    <p className={`${isDark ? 'text-gray-300' : 'text-gray-600'} mb-4`}>
                      No users have been granted access to vector databases yet.
                    </p>
                    <Button onClick={handleCreateVectorDBAccess} className="bg-purple-500 hover:bg-purple-600 shadow-lg hover:shadow-purple-500/25 transition-all duration-200">
                      <Plus className="w-4 h-4 mr-2" />
                      Grant First Access
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {vectorDBUsers.map((config) => (
                      <div
                        key={config.user_id}
                        className={`flex items-center justify-between p-4 rounded-lg border transition-all duration-200 hover:scale-[1.02] hover:shadow-lg ${
                          isDark 
                            ? 'bg-gradient-to-r from-slate-700/40 to-slate-600/30 border-slate-600 hover:border-purple-500/50 hover:shadow-purple-500/10' 
                            : 'bg-gradient-to-r from-gray-50 to-white border-gray-200 hover:border-purple-300 hover:shadow-purple-500/10'
                        }`}
                      >
                        <div className="flex items-center gap-4">
                          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg">
                            <Brain className="w-5 h-5 text-white" />
                          </div>
                          <div>
                            <div className="flex items-center gap-3 mb-1">
                              <h4 className={`font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                                {extractNameFromEmail(config.user_id)}
                              </h4>
                              {getAccessLevelBadge(config)}
                            </div>
                            <p className={`text-sm ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>{config.user_id}</p>
                            <div className={`flex items-center gap-4 mt-2 text-xs ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
                              <span>Access Level: {config.access_level}</span>
                              <span>Database: {getDatabaseName(config.db_id)}</span>
                              <span>Tables: {formatTableNames(config.table_names)}</span>
                            </div>
                            {config.table_names && config.table_names.length > 3 && (
                              <div className={`mt-2 text-xs ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                                <details className="cursor-pointer">
                                  <summary className={isDark ? 'hover:text-gray-300' : 'hover:text-gray-800'}>Show all tables</summary>
                                  <div className="mt-2 pl-4">
                                    {config.table_names.map((table, index) => (
                                      <div key={index} className={isDark ? 'text-gray-400' : 'text-gray-600'}>
                                        • {table}
                                      </div>
                                    ))}
                                  </div>
                                </details>
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Button
                            onClick={() => handleEditUser(config.user_id, 'vector')}
                            variant="outline"
                            size="sm"
                            className={`transition-all duration-200 ${
                              isDark 
                                ? "border-purple-500/50 text-purple-300 hover:bg-purple-500/20 hover:border-purple-400" 
                                : "border-purple-300 text-purple-600 hover:bg-purple-50 hover:border-purple-400"
                            }`}
                          >
                            <Edit className="w-4 h-4 mr-2" />
                            Edit
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Modals */}
        <CreateDatabaseAccessModal
          isOpen={isDatabaseModalOpen}
          onClose={handleModalClose}
          onSuccess={handleModalSuccess}
          selectedUser={selectedUser}
          editingUser={editingUser}
        />

        <CreateVectorDBAccessModal
          isOpen={isVectorDBModalOpen}
          onClose={handleModalClose}
          onSuccess={handleModalSuccess}
          selectedUser={selectedUser}
          editingUser={editingUser}
        />
    </div>
  );
}
