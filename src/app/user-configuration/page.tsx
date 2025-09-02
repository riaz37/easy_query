"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";

import { toast } from "sonner";
import {
  Settings,
  User,
  CheckCircle,
  AlertCircle,
  Loader2,
  RefreshCw,
  Database,
  Shield,
  FileText,
  ArrowRight,
  Globe,
  Key,
  Edit3,
  Save,
  X,
  RotateCcw,
} from "lucide-react";

import { useAuthContext } from "@/components/providers/AuthContextProvider";
import { useDatabaseContext } from "@/components/providers/DatabaseContextProvider";
import { useBusinessRulesContext } from "@/components/providers/BusinessRulesContextProvider";
import { ServiceRegistry } from "@/lib/api";

interface DatabaseInfo {
  db_id: number;
  db_name: string;
  db_url: string;
  db_type: string;
  is_current: boolean;
  business_rule?: string;
}

export default function UserConfigurationPage() {
  const { user, isAuthenticated } = useAuthContext();
  const {
    currentDatabaseId,
    currentDatabaseName,
    setCurrentDatabase,
    loadDatabasesFromConfig,
    setLoading: setDatabaseLoading,
    setError: setDatabaseError,
    availableDatabases,
  } = useDatabaseContext();
  const {
    businessRules,
    loadBusinessRulesFromConfig,
    updateBusinessRules,
    refreshBusinessRules,
    hasBusinessRules,
    businessRulesCount,
    setLoading: setBusinessRulesLoading,
    setError: setBusinessRulesError,
  } = useBusinessRulesContext();

  // State
  const [loading, setLoading] = useState(false);
  const [databases, setDatabases] = useState<DatabaseInfo[]>([]);
  const [activeTab, setActiveTab] = useState("overview");
  const hasLoadedRef = useRef(false);

  // Business rules editing state
  const [isEditingRules, setIsEditingRules] = useState(false);
  const [editedRulesContent, setEditedRulesContent] = useState("");
  const [hasUnsavedRulesChanges, setHasUnsavedRulesChanges] = useState(false);
  const [rulesContentError, setRulesContentError] = useState<string | null>(
    null,
  );

  // Load user configuration
  const loadUserConfiguration = useCallback(async () => {
    // Check if we already have configuration loaded from storage
    if (
      currentDatabaseId &&
      availableDatabases &&
      availableDatabases.length > 0
    ) {
      return;
    }

    setLoading(true);
    setDatabaseLoading(true);
    setBusinessRulesLoading(true);
    setDatabaseError(null);
    setBusinessRulesError(null);

    try {
      // Load accessible databases
      const databasesResponse =
        await ServiceRegistry.database.getAllDatabases();

      if (databasesResponse.success) {
        const dbList = databasesResponse.data.map((db) => ({
          db_id: db.id,
          db_name: db.name,
          db_url: db.url,
          db_type: db.type,
          is_current: db.id === currentDatabaseId,
          business_rule: db.metadata?.businessRule || "",
        }));

        // Update local state
        setDatabases(dbList);

        // Update database context provider
        const dbConfigs = dbList.map((db) => ({
          db_id: db.db_id,
          db_name: db.db_name,
          db_url: db.db_url,
          db_type: db.db_type,
          business_rule: db.business_rule,
          is_active: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        }));
        loadDatabasesFromConfig(dbConfigs);

        // Get current database from backend and load business rules
        const currentDBInfo =
          await ServiceRegistry.userCurrentDB.getUserCurrentDB(user?.user_id);
        if (
          currentDBInfo.success &&
          currentDBInfo.data &&
          currentDBInfo.data.db_id
        ) {
          // Update the current database in context
          setCurrentDatabase(
            currentDBInfo.data.db_id,
            currentDBInfo.data.db_name || "Unknown",
          );

          // Update local state to mark the current database
          setDatabases((prev) =>
            prev.map((db) => ({
              ...db,
              is_current: db.db_id === currentDBInfo.data.db_id,
            })),
          );

          // Extract business rules from the response and update the context
          const businessRules = currentDBInfo.data.business_rule || "";
          loadBusinessRulesFromConfig(businessRules);
        } else {
          // No current database set
          loadBusinessRulesFromConfig("");
        }
      } else {
        throw new Error(databasesResponse.error || "Failed to load databases");
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to load configuration";
      setDatabaseError(errorMessage);
      setBusinessRulesError(errorMessage);
      toast.error("Failed to load configuration");
    } finally {
      setLoading(false);
      setDatabaseLoading(false);
      setBusinessRulesLoading(false);
    }
  }, [
    user?.user_id,
    setCurrentDatabase,
    loadBusinessRulesFromConfig,
    loadDatabasesFromConfig,
    setDatabaseLoading,
    setBusinessRulesLoading,
    setDatabaseError,
    setBusinessRulesError,
    currentDatabaseId,
    availableDatabases,
  ]);

  // Load user configuration on mount
  useEffect(() => {
    if (isAuthenticated && !loading && !hasLoadedRef.current) {
      hasLoadedRef.current = true;
      loadUserConfiguration();
    }
  }, [isAuthenticated, loadUserConfiguration, loading]);

  // Reset loaded flag when user changes
  useEffect(() => {
    hasLoadedRef.current = false;
  }, [user?.user_id]);

  // Manual refresh function (separate from automatic loading)
  const handleManualRefresh = useCallback(async () => {
    hasLoadedRef.current = false;
    // Force reload by clearing storage check
    setLoading(true);
    setDatabaseLoading(true);
    setBusinessRulesLoading(true);
    await loadUserConfiguration();
  }, [loadUserConfiguration]);

  // Load business rules for a specific database
  const loadBusinessRulesForDatabase = useCallback(
    async (databaseId: number) => {
      try {
        setBusinessRulesLoading(true);
        setBusinessRulesError(null);

        const response =
          await ServiceRegistry.businessRules.getBusinessRules(databaseId);
        if (response.success && response.data) {
          loadBusinessRulesFromConfig(response.data.content || "");
        } else {
          loadBusinessRulesFromConfig("");
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : "Failed to load business rules";
        setBusinessRulesError(errorMessage);
        loadBusinessRulesFromConfig("");
      } finally {
        setBusinessRulesLoading(false);
      }
    },
    [
      loadBusinessRulesFromConfig,
      setBusinessRulesLoading,
      setBusinessRulesError,
    ],
  );

  // Handle database selection
  const handleDatabaseChange = useCallback(
    async (databaseId: number) => {
      try {
        setDatabaseLoading(true);
        setDatabaseError(null);

        const selectedDB = databases.find((db) => db.db_id === databaseId);
        if (selectedDB) {
          // Set the current database in backend
          const response = await ServiceRegistry.userCurrentDB.setUserCurrentDB(
            {
              db_id: databaseId,
            },
            user?.user_id,
          );

          if (response.success) {
            // Update database context provider
            setCurrentDatabase(databaseId, selectedDB.db_name);

            // Update local state
            setDatabases((prev) =>
              prev.map((db) => ({
                ...db,
                is_current: db.db_id === databaseId,
              })),
            );

            // Get the current database info which includes business rules
            const currentDBInfo =
              await ServiceRegistry.userCurrentDB.getUserCurrentDB(
                user?.user_id,
              );
            if (currentDBInfo.success && currentDBInfo.data) {
              const businessRules = currentDBInfo.data.business_rule || "";
              loadBusinessRulesFromConfig(businessRules);
            } else {
              loadBusinessRulesFromConfig("");
            }

            toast.success(`Switched to database: ${selectedDB.db_name}`);
          } else {
            throw new Error(response.error || "Failed to set current database");
          }
        } else {
          throw new Error("Selected database not found");
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : "Failed to switch database";
        setDatabaseError(errorMessage);
        toast.error("Failed to switch database");
      } finally {
        setDatabaseLoading(false);
      }
    },
    [
      databases,
      user?.user_id,
      setCurrentDatabase,
      loadBusinessRulesFromConfig,
      setDatabaseLoading,
      setDatabaseError,
    ],
  );

  // Handle business rules refresh
  const handleBusinessRulesRefresh = async () => {
    try {
      if (currentDatabaseId) {
        setBusinessRulesLoading(true);
        setBusinessRulesError(null);

        const currentDBInfo =
          await ServiceRegistry.userCurrentDB.getUserCurrentDB(user?.user_id);
        if (currentDBInfo.success && currentDBInfo.data) {
          const businessRules = currentDBInfo.data.business_rule || "";
          loadBusinessRulesFromConfig(businessRules);
          toast.success("Business rules refreshed successfully");
        } else {
          loadBusinessRulesFromConfig("");
          toast.success("Business rules refreshed (no rules found)");
        }
      } else {
        toast.error("No database selected");
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to refresh business rules";
      setBusinessRulesError(errorMessage);
      toast.error("Failed to refresh business rules");
    } finally {
      setBusinessRulesLoading(false);
    }
  };

  // Business rules editing handlers
  const handleRulesEdit = useCallback(() => {
    setIsEditingRules(true);
    setEditedRulesContent(businessRules.content);
    setHasUnsavedRulesChanges(false);
    setRulesContentError(null);
  }, [businessRules.content]);

  const handleRulesSave = useCallback(async () => {
    if (!currentDatabaseId) {
      toast.error("No database selected");
      return;
    }

    // Validate content
    if (!editedRulesContent.trim()) {
      setRulesContentError("Business rules content cannot be empty");
      return;
    }

    if (editedRulesContent.length < 10) {
      setRulesContentError(
        "Business rules content is too short (minimum 10 characters)",
      );
      return;
    }

    if (editedRulesContent.length > 50000) {
      setRulesContentError(
        "Business rules content is too long (maximum 50,000 characters)",
      );
      return;
    }

    setLoading(true);
    setBusinessRulesLoading(true);
    setRulesContentError(null);

    try {
      const response = await ServiceRegistry.businessRules.updateBusinessRules(
        editedRulesContent,
        currentDatabaseId,
      );

      if (response.success) {
        // Update business rules context
        updateBusinessRules(editedRulesContent);

        // Reset editing state
        setIsEditingRules(false);
        setHasUnsavedRulesChanges(false);
        setEditedRulesContent("");

        toast.success("Business rules updated successfully");

        // Refresh the configuration to get updated data
        await loadUserConfiguration();
      } else {
        throw new Error(response.error || "Failed to update business rules");
      }
    } catch (error: any) {
      const errorMessage = error.message || "Failed to update business rules";
      setRulesContentError(errorMessage);
      setBusinessRulesError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setLoading(false);
      setBusinessRulesLoading(false);
    }
  }, [
    currentDatabaseId,
    editedRulesContent,
    updateBusinessRules,
    loadUserConfiguration,
    setBusinessRulesLoading,
    setBusinessRulesError,
  ]);

  const handleRulesCancel = useCallback(() => {
    setIsEditingRules(false);
    setEditedRulesContent("");
    setHasUnsavedRulesChanges(false);
    setRulesContentError(null);
  }, []);

  const handleRulesReset = useCallback(() => {
    setEditedRulesContent(businessRules.content);
    setHasUnsavedRulesChanges(false);
    setRulesContentError(null);
  }, [businessRules.content]);

  const handleRulesContentChange = useCallback(
    (content: string) => {
      setEditedRulesContent(content);
      setHasUnsavedRulesChanges(content !== businessRules.content);
      setRulesContentError(null);
    },
    [businessRules.content],
  );

  if (!isAuthenticated) {
    return (
      <div className="container mx-auto p-6">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <AlertCircle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
              <h2 className="text-xl font-semibold mb-2">
                Authentication Required
              </h2>
              <p className="text-gray-600">
                Please log in to access your configuration.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="w-full min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900 pt-20">
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Settings className="w-8 h-8 text-emerald-400" />
            User Configuration
          </h1>
          <p className="text-gray-400 text-lg">
            Manage your database settings, business rules, and preferences
          </p>
        </div>

        {/* Main Configuration Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-slate-800/50">
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <User className="w-4 h-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="database" className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              Database Settings
            </TabsTrigger>
            <TabsTrigger
              value="business-rules"
              className="flex items-center gap-2"
            >
              <Shield className="w-4 h-4" />
              Business Rules
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6 mt-6">
            {/* User Info Card */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                  <User className="h-5 w-5 text-blue-400" />
                  User Information
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label className="text-gray-400">User ID</Label>
                    <div className="text-white font-medium">
                      {user?.user_id}
                    </div>
                  </div>
                  <div>
                    <Label className="text-gray-400">Email</Label>
                    <div className="text-white font-medium">
                      {user?.email || "Not provided"}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Current Status Card */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Current Status</CardTitle>
                <CardDescription className="text-gray-400">
                  Overview of your current configuration
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Database Status */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Database className="w-5 h-5 text-blue-400" />
                      <span className="text-white font-medium">
                        Database Context
                      </span>
                    </div>
                    <div className="ml-7">
                      {currentDatabaseName ? (
                        <div className="space-y-2">
                          <div className="text-white">
                            {currentDatabaseName}
                          </div>
                          <Badge
                            variant="outline"
                            className="text-green-400 border-green-400"
                          >
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Active
                          </Badge>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="text-gray-400">
                            No database selected
                          </div>
                          <Badge
                            variant="outline"
                            className="text-yellow-400 border-yellow-400"
                          >
                            <AlertCircle className="w-3 h-3 mr-1" />
                            Not Configured
                          </Badge>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Business Rules Status */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Shield className="w-5 h-5 text-emerald-400" />
                      <span className="text-white font-medium">
                        Business Rules
                      </span>
                    </div>
                    <div className="ml-7">
                      {businessRules.status === "loaded" && hasBusinessRules ? (
                        <div className="space-y-2">
                          <div className="text-white">
                            {businessRulesCount} rules active
                          </div>
                          <Badge
                            variant="outline"
                            className="text-green-400 border-green-400"
                          >
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Active
                          </Badge>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className="text-gray-400">
                            No rules configured
                          </div>
                          <Badge
                            variant="outline"
                            className="text-yellow-400 border-yellow-400"
                          >
                            <AlertCircle className="w-3 h-3 mr-1" />
                            Not Configured
                          </Badge>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Quick Actions</CardTitle>
                <CardDescription className="text-gray-400">
                  Common configuration tasks
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Button
                    variant="outline"
                    onClick={() => setActiveTab("database")}
                    className="w-full bg-slate-700 border-slate-600 text-white hover:bg-slate-600"
                  >
                    <Database className="w-4 h-4 mr-2" />
                    Configure Database
                  </Button>

                  <Button
                    variant="outline"
                    onClick={() => setActiveTab("business-rules")}
                    className="w-full bg-slate-700 border-slate-600 text-white hover:bg-slate-600"
                  >
                    <Shield className="w-4 h-4 mr-2" />
                    Manage Business Rules
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Database Settings Tab */}
          <TabsContent value="database" className="space-y-6 mt-6">
            {/* Database Selection */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">Database Selection</CardTitle>
                <CardDescription className="text-gray-400">
                  Choose your current working database
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {loading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="w-6 h-6 animate-spin text-emerald-400" />
                    <span className="ml-2 text-gray-400">
                      Loading databases...
                    </span>
                  </div>
                ) : (
                  <>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {databases.map((db) => (
                        <Card
                          key={db.db_id}
                          className={`cursor-pointer transition-all hover:scale-105 ${
                            db.is_current
                              ? "bg-emerald-900/30 border-emerald-500"
                              : "bg-slate-700/50 border-slate-600 hover:border-slate-500"
                          }`}
                          onClick={() => handleDatabaseChange(db.db_id)}
                        >
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between mb-2">
                              <h3 className="font-semibold text-white">
                                {db.db_name}
                              </h3>
                              {db.is_current && (
                                <CheckCircle className="w-5 h-5 text-emerald-400" />
                              )}
                            </div>
                            <div className="text-sm text-gray-400 space-y-1">
                              <div>Type: {db.db_type}</div>
                              <div className="truncate">URL: {db.db_url}</div>
                              <div>
                                Rules:{" "}
                                {db.is_current &&
                                businessRules.status === "loaded"
                                  ? `${businessRules.content.length} chars`
                                  : db.business_rule
                                    ? `${db.business_rule.length} chars`
                                    : "None"}
                              </div>
                            </div>
                            <Badge
                              variant={db.is_current ? "default" : "secondary"}
                              className={`mt-2 ${
                                db.is_current
                                  ? "bg-emerald-600 text-white"
                                  : "bg-slate-600 text-gray-300"
                              }`}
                            >
                              {db.is_current ? "Current" : "Select"}
                            </Badge>
                          </CardContent>
                        </Card>
                      ))}
                    </div>

                    <div className="flex justify-between items-center pt-4">
                      <Button
                        variant="outline"
                        onClick={handleManualRefresh}
                        disabled={loading}
                      >
                        <RefreshCw
                          className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`}
                        />
                        Refresh
                      </Button>

                      <div className="text-sm text-gray-400">
                        {databases.length} database
                        {databases.length !== 1 ? "s" : ""} available
                      </div>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Business Rules Tab */}
          <TabsContent value="business-rules" className="space-y-6 mt-6">
            {/* Business Rules Status */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-white">
                  <Shield className="h-5 w-5 text-emerald-400" />
                  Business Rules Status
                </CardTitle>
                <CardDescription className="text-gray-400">
                  Current business rules configuration for{" "}
                  {currentDatabaseName || "selected database"}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label className="text-gray-400">Status</Label>
                    <div className="flex items-center gap-2 mt-1">
                      {businessRules.status === "loaded" && hasBusinessRules ? (
                        <Badge
                          variant="outline"
                          className="text-green-400 border-green-400"
                        >
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Active
                        </Badge>
                      ) : (
                        <Badge
                          variant="outline"
                          className="text-yellow-400 border-yellow-400"
                        >
                          <AlertCircle className="w-3 h-3 mr-1" />
                          Not Configured
                        </Badge>
                      )}
                    </div>
                  </div>

                  <div>
                    <Label className="text-gray-400">Rules Count</Label>
                    <div className="text-white font-medium mt-1">
                      {businessRules.status === "loaded"
                        ? businessRulesCount
                        : "0"}
                    </div>
                  </div>

                  <div>
                    <Label className="text-gray-400">Content Length</Label>
                    <div className="text-white font-medium mt-1">
                      {businessRules.content
                        ? `${businessRules.content.length} characters`
                        : "0 characters"}
                    </div>
                  </div>
                </div>

                <Separator className="bg-slate-600" />

                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-400">
                    Last updated:{" "}
                    {businessRules.lastUpdated
                      ? new Date(businessRules.lastUpdated).toLocaleString()
                      : "Never"}
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={handleBusinessRulesRefresh}
                      disabled={businessRules.status === "loading"}
                      className="border-slate-600 text-slate-300 hover:bg-slate-700"
                    >
                      <RefreshCw
                        className={`w-4 h-4 mr-2 ${businessRules.status === "loading" ? "animate-spin" : ""}`}
                      />
                      Refresh Status
                    </Button>

                    {!isEditingRules ? (
                      <Button
                        onClick={handleRulesEdit}
                        disabled={!currentDatabaseId}
                        className="bg-emerald-600 hover:bg-emerald-700 text-white"
                      >
                        <Edit3 className="w-4 h-4 mr-2" />
                        Edit Rules
                      </Button>
                    ) : (
                      <div className="flex gap-2">
                        <Button
                          onClick={handleRulesSave}
                          disabled={loading || !hasUnsavedRulesChanges}
                          className="bg-green-600 hover:bg-green-700 text-white"
                        >
                          <Save className="w-4 h-4 mr-2" />
                          Save
                        </Button>
                        <Button
                          onClick={handleRulesCancel}
                          variant="outline"
                          className="border-gray-600 text-gray-300 hover:bg-gray-700"
                        >
                          <X className="w-4 h-4 mr-2" />
                          Cancel
                        </Button>
                        <Button
                          onClick={handleRulesReset}
                          variant="outline"
                          disabled={!hasUnsavedRulesChanges}
                          className="border-blue-600 text-blue-300 hover:bg-blue-700"
                        >
                          <RotateCcw className="w-4 h-4 mr-2" />
                          Reset
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Business Rules Editor */}
            {currentDatabaseId && (
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <FileText className="h-5 w-5 text-blue-400" />
                    Business Rules Editor
                  </CardTitle>
                  <CardDescription className="text-gray-400">
                    {isEditingRules
                      ? "Edit business rules for the current database"
                      : "View and edit business rules"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {isEditingRules ? (
                    <div className="space-y-4">
                      <div>
                        <Label
                          htmlFor="businessRules"
                          className="text-gray-400"
                        >
                          Business Rules Content
                        </Label>
                        <Textarea
                          id="businessRules"
                          value={editedRulesContent}
                          onChange={(e) =>
                            handleRulesContentChange(e.target.value)
                          }
                          placeholder="Enter your business rules here..."
                          className="mt-2 bg-slate-700/50 border-slate-600 text-white min-h-[200px] resize-y"
                        />
                        {rulesContentError && (
                          <p className="text-red-400 text-sm mt-2">
                            {rulesContentError}
                          </p>
                        )}
                        <div className="flex justify-between text-sm text-gray-400 mt-2">
                          <span>{editedRulesContent.length} characters</span>
                          <span>
                            {
                              editedRulesContent
                                .split("\n")
                                .filter((line) => line.trim().length > 0).length
                            }{" "}
                            rules
                          </span>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div>
                        <Label className="text-gray-400">
                          Current Business Rules
                        </Label>
                        <div className="mt-2 p-4 bg-slate-700/50 border border-slate-600 rounded-md min-h-[200px] max-h-[400px] overflow-y-auto">
                          {businessRules.content ? (
                            <pre className="text-white whitespace-pre-wrap text-sm font-mono">
                              {businessRules.content}
                            </pre>
                          ) : (
                            <p className="text-gray-400 italic">
                              No business rules configured for this database.
                            </p>
                          )}
                        </div>
                        <div className="flex justify-between text-sm text-gray-400 mt-2">
                          <span>
                            {businessRules.content?.length || 0} characters
                          </span>
                          <span>{businessRulesCount} rules</span>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Business Rules Context Info */}
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">
                  Context Information
                </CardTitle>
                <CardDescription className="text-gray-400">
                  How business rules are applied in your system
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 text-sm text-gray-300">
                  <div className="flex items-start gap-2">
                    <Globe className="w-4 h-4 text-blue-400 mt-0.5" />
                    <span>
                      Business rules are automatically applied to all database
                      queries
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Key className="w-4 h-4 text-emerald-400 mt-0.5" />
                    <span>
                      Rules ensure data integrity and compliance across your
                      system
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <Database className="w-4 h-4 text-purple-400 mt-0.5" />
                    <span>
                      Rules are database-specific and automatically loaded when
                      switching databases
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
