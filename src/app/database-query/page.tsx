"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext, useDatabaseContext } from "@/components/providers";
import { useDatabaseOperations } from "@/lib/hooks/use-database-operations";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Database,
  Play,
  History,
  BarChart3,
  AlertCircle,
  ArrowLeft,
  Clock,
  User,
  FileText,
} from "lucide-react";
import { DatabaseQueryForm } from "@/components/database-query/DatabaseQueryForm";
import { QueryHistoryPanel } from "@/components/database-query/QueryHistoryPanel";
import { QueryModeToggle } from "@/components/database-query/QueryModeToggle";
import { IntegratedReportGenerator } from "@/components/database-query/IntegratedReportGenerator";

export default function DatabaseQueryPage() {
  const router = useRouter();
  const { user } = useAuthContext();
  const { currentDatabase, hasCurrentDatabase } = useDatabaseContext();
  const {
    loading,
    error,
    sendQuery,
    fetchQueryHistory,
    history,
    historyLoading,
  } = useDatabaseOperations();

  // Local state
  const [showHistory, setShowHistory] = useState(false);
  const [currentQuery, setCurrentQuery] = useState<string>("");
  const [queryMode, setQueryMode] = useState<'query' | 'reports'>('query');

  // Load query history on mount
  useEffect(() => {
    if (user?.user_id) {
      fetchQueryHistory();
    }
  }, [user?.user_id, fetchQueryHistory]);

  const handleBackToDashboard = () => {
    router.push("/");
  };

  const handleQuerySubmit = async (query: string) => {
    if (!query.trim()) {
      toast.error("Please enter a query");
      return;
    }

    if (!hasCurrentDatabase) {
      toast.error("Please select a database first");
      return;
    }

    if (!user?.user_id) {
      toast.error("User not authenticated. Please log in again.");
      return;
    }

    try {
      setCurrentQuery(query);
      console.log("Sending query with user ID:", user.user_id); // Debug log

      const response = await sendQuery({
        question: query,
        userId: user.user_id, // Use actual user ID, no fallback
      });

      if (response.success && response.data) {
        const resultData = {
          query: query,
          userId: user.user_id,
          timestamp: new Date().toISOString(),
          result: {
            success: true,
            data: response.data.payload?.data || response.data.data || [],
            sql: response.data.payload?.sql || "",
            status_code: response.data.status_code || 200,
          },
        };

        // Store results in sessionStorage (like AI results page)
        sessionStorage.setItem(
          "databaseQueryResult",
          JSON.stringify(resultData)
        );

        // Redirect to results page
        router.push("/database-query-results");
      } else {
        toast.error("Query failed", {
          description: response.error || "Unknown error occurred",
        });
      }
    } catch (error) {
      console.error("Query execution error:", error);
      toast.error("Query execution failed", {
        description:
          error instanceof Error ? error.message : "Unknown error occurred",
      });
    }
  };

  const handleViewHistory = () => {
    setShowHistory(!showHistory);
  };

  return (
    <div className="w-full min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900">
      {/* Add top padding to account for fixed navbar */}
      <div className="pt-24 pb-8">
        <div className="container mx-auto px-4">
          <div className="max-w-7xl mx-auto">
            {/* Header with Back Button */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500/30 to-blue-600/20 rounded-xl flex items-center justify-center border border-blue-500/40">
                    <Database className="w-6 h-6 text-blue-400" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold text-white">
                      Database Query
                    </h1>
                    <p className="text-gray-400">
                      Run quick queries or generate comprehensive AI-powered reports
                    </p>
                  </div>
                </div>
                <Button
                  onClick={handleBackToDashboard}
                  variant="outline"
                  className="border-blue-400/30 text-blue-400 hover:bg-blue-400/10"
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Dashboard
                </Button>
              </div>

              {/* Feature Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                <Card className="bg-gray-900/50 border-blue-400/30">
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                        <Play className="w-4 h-4 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          Quick Queries
                        </h3>
                        <p className="text-gray-400 text-sm">
                          Instant results for simple questions
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gray-900/50 border-purple-400/30">
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                        <FileText className="w-4 h-4 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          AI Reports
                        </h3>
                        <p className="text-gray-400 text-sm">
                          Comprehensive analysis & insights
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gray-900/50 border-green-400/30">
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                        <BarChart3 className="w-4 h-4 text-green-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          Smart Results
                        </h3>
                        <p className="text-gray-400 text-sm">
                          AI-powered data insights
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gray-900/50 border-blue-400/30">
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                        <History className="w-4 h-4 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          Business Rules
                        </h3>
                        <p className="text-gray-400 text-sm">
                          Automatic compliance & validation
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Database Selection */}
            <div className="mb-6">
              <Card className="bg-gray-900/50 border-blue-400/30">
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                        <Database className="w-4 h-4 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          {hasCurrentDatabase
                            ? currentDatabase?.db_name
                            : "No Database Selected"}
                        </h3>
                        <p className="text-gray-400 text-sm">
                          {hasCurrentDatabase
                            ? `Connected to ${
                                currentDatabase?.db_type || "MSSQL"
                              } database`
                            : "Please select a database in User Configuration"}
                        </p>
                      </div>
                    </div>
                    {hasCurrentDatabase && (
                      <Badge
                        variant="outline"
                        className="border-green-400/30 text-green-400"
                      >
                        ✓ Connected
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Mode Toggle */}
            <div className="mb-6">
              <QueryModeToggle
                mode={queryMode}
                onModeChange={setQueryMode}
                hasDatabase={hasCurrentDatabase}
              />
              
              {/* Quick Navigation */}
              <div className="mt-4 flex items-center gap-2 text-sm text-gray-400">
                <span>Quick access:</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => router.push('/ai-results')}
                  className="text-purple-400 hover:text-purple-300 hover:bg-purple-400/10"
                >
                  <FileText className="w-4 h-4 mr-1" />
                  View AI Reports
                </Button>
              </div>
            </div>

            {/* Query Form or Report Generator */}
            <div className="mb-8">
              {queryMode === 'query' ? (
              <DatabaseQueryForm
                onSubmit={handleQuerySubmit}
                loading={loading}
                hasDatabase={hasCurrentDatabase}
              />
              ) : (
                <IntegratedReportGenerator
                  userId={user?.user_id}
                  onReportComplete={(results) => {
                    console.log('Report completed:', results);
                    // Store results for the results page
                    sessionStorage.setItem('reportResults', JSON.stringify(results));
                  }}
                />
              )}
            </div>

            {/* Error Display */}
            {error && (
              <div className="mb-6">
                <Card className="bg-red-900/20 border-red-500/30">
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <AlertCircle className="w-5 h-5 text-red-400" />
                      <div>
                        <h3 className="text-red-400 font-medium">
                          Query Error
                        </h3>
                        <p className="text-red-300 text-sm">{error}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Empty State */}
            <div className="mt-8">
              <Card className="bg-gray-900/50 border-blue-400/30">
                <CardContent className="pt-12 pb-12 text-center">
                  <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Database className="w-8 h-8 text-blue-400" />
                  </div>
                  <h3 className="text-white text-lg font-medium mb-2">
                    Ready to Get Started
                  </h3>
                  <p className="text-gray-400 mb-4">
                    {queryMode === 'query' 
                      ? 'Ask your question in natural language above to get started'
                      : 'Describe what you want to analyze and generate comprehensive reports'
                    }
                  </p>
                  {!hasCurrentDatabase && (
                    <p className="text-yellow-400 text-sm mb-4">
                      Please select a database first
                    </p>
                  )}
                  <div className="flex justify-center gap-2 mt-4">
                    <Badge variant="outline" className="border-blue-400/30 text-blue-400">
                      {queryMode === 'query' ? 'Quick Query Mode' : 'AI Report Mode'}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* History Panel */}
            <div className="fixed bottom-6 right-6 z-50">
              <Button
                onClick={handleViewHistory}
                variant="outline"
                size="lg"
                className="bg-gray-900/90 border-blue-400/30 text-blue-400 hover:bg-blue-400/10 backdrop-blur-sm"
              >
                <History className="w-5 h-5 mr-2" />
                Query History
              </Button>
            </div>

            {/* History Sidebar */}
            {showHistory && (
              <QueryHistoryPanel
                history={history}
                loading={historyLoading}
                onClose={() => setShowHistory(false)}
                onQuerySelect={(query) => {
                  setCurrentQuery(query);
                  setShowHistory(false);
                }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
