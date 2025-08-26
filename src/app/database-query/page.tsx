"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext, useDatabaseContext } from "@/components/providers";
import { useDatabaseOperations } from "@/lib/hooks/use-database-operations";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
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
  Loader2,
  Brain,
  Zap,
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
  const [queryProgress, setQueryProgress] = useState(0);
  const [processingSteps, setProcessingSteps] = useState<string[]>([]);
  const [isReportGenerating, setIsReportGenerating] = useState(false);

  // Load query history on mount
  useEffect(() => {
    if (user?.user_id) {
      fetchQueryHistory();
    }
  }, [user?.user_id, fetchQueryHistory]);

  // Simulate progress when loading
  useEffect(() => {
    if (loading) {
      setQueryProgress(0);
      setProcessingSteps([
        "Analyzing your question...",
        "Connecting to database...",
        "Processing query...",
        "Applying business rules...",
        "Generating results..."
      ]);
      
      const interval = setInterval(() => {
        setQueryProgress(prev => {
          if (prev >= 90) {
            clearInterval(interval);
            return 90;
          }
          return prev + 10;
        });
      }, 800);

      return () => clearInterval(interval);
    } else {
      setQueryProgress(0);
      setProcessingSteps([]);
    }
  }, [loading]);

  // Handle report generation state
  const handleReportComplete = (results: any) => {
    console.log('Report completed:', results);
    setIsReportGenerating(false);
    // Store results for the results page
    sessionStorage.setItem('reportResults', JSON.stringify(results));
  };

  const handleReportStart = () => {
    setIsReportGenerating(true);
  };

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
      {/* Loading Overlay */}
      {loading && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-gray-900/95 border border-blue-400/30 rounded-2xl p-8 max-w-md w-full mx-4">
            <div className="text-center space-y-6">
              {/* Animated Brain Icon */}
              <div className="relative">
                <div className="w-20 h-20 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full flex items-center justify-center mx-auto border border-blue-400/30">
                  <Brain className="w-10 h-10 text-blue-400 animate-pulse" />
                </div>
                <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                  <Zap className="w-3 h-3 text-white" />
                </div>
              </div>

              {/* Progress Bar */}
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-blue-400">Processing Query</span>
                  <span className="text-gray-400">{queryProgress}%</span>
                </div>
                <Progress value={queryProgress} className="h-2" />
              </div>

              {/* Current Step */}
              <div className="space-y-3">
                <div className="flex items-center justify-center gap-2 text-blue-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm font-medium">AI Processing</span>
                </div>
                <p className="text-gray-300 text-sm">
                  {processingSteps[Math.floor(queryProgress / 20)] || "Preparing..."}
                </p>
              </div>

              {/* Query Preview */}
              <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
                <p className="text-xs text-gray-400 mb-1">Processing:</p>
                <p className="text-white text-sm font-medium">
                  {currentQuery.length > 60 
                    ? currentQuery.substring(0, 60) + "..." 
                    : currentQuery
                  }
                </p>
              </div>

              {/* Tips */}
              <div className="text-xs text-gray-500 space-y-1">
                <p>• This may take a few moments</p>
                <p>• AI is analyzing your question and database</p>
                <p>• Results will appear automatically</p>
              </div>
            </div>
          </div>
        </div>
      )}

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
                  disabled={loading || (queryMode === 'reports' && isReportGenerating)}
                  className={`border-blue-400/30 text-blue-400 hover:bg-blue-400/10 transition-all duration-300 ${
                    loading || (queryMode === 'reports' && isReportGenerating) ? 'opacity-60 cursor-not-allowed' : ''
                  }`}
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  {loading 
                    ? 'Processing...' 
                    : (queryMode === 'reports' && isReportGenerating)
                    ? 'Generating Report...'
                    : 'Back to Dashboard'
                  }
                </Button>
              </div>

              {/* Feature Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
                <Card className={`bg-gray-900/50 border-blue-400/30 transition-all duration-300 ${
                  loading || (queryMode === 'reports' && isReportGenerating) ? 'opacity-60 scale-95' : ''
                }`}>
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 ${
                        loading || (queryMode === 'reports' && isReportGenerating) ? 'bg-blue-500/30 animate-pulse' : 'bg-blue-500/20'
                      }`}>
                        <Play className="w-4 h-4 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          Quick Queries
                        </h3>
                        <p className="text-gray-400 text-sm">
                          {loading || (queryMode === 'reports' && isReportGenerating) ? "Processing..." : "Instant results for simple questions"}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className={`bg-gray-900/50 border-purple-400/30 transition-all duration-300 ${
                  loading || (queryMode === 'reports' && isReportGenerating) ? 'opacity-60 scale-95' : ''
                }`}>
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 ${
                        loading || (queryMode === 'reports' && isReportGenerating) ? 'bg-purple-500/30 animate-pulse' : 'bg-purple-500/20'
                      }`}>
                        <FileText className="w-4 h-4 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          AI Reports
                        </h3>
                        <p className="text-gray-400 text-sm">
                          {loading || (queryMode === 'reports' && isReportGenerating) ? "Processing..." : "Comprehensive analysis & insights"}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className={`bg-gray-900/50 border-green-400/30 transition-all duration-300 ${
                  loading || (queryMode === 'reports' && isReportGenerating) ? 'opacity-60 scale-95' : ''
                }`}>
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 ${
                        loading || (queryMode === 'reports' && isReportGenerating) ? 'bg-green-500/30 animate-pulse' : 'bg-green-500/20'
                      }`}>
                        <BarChart3 className="w-4 h-4 text-green-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          Smart Results
                        </h3>
                        <p className="text-gray-400 text-sm">
                          {loading || (queryMode === 'reports' && isReportGenerating) ? "Processing..." : "AI-powered data insights"}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className={`bg-gray-900/50 border-blue-400/30 transition-all duration-300 ${
                  loading || (queryMode === 'reports' && isReportGenerating) ? 'opacity-60 scale-95' : ''
                }`}>
                  <CardContent className="pt-6">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 ${
                        loading || (queryMode === 'reports' && isReportGenerating) ? 'bg-purple-500/30 animate-pulse' : 'bg-purple-500/20'
                      }`}>
                        <History className="w-4 h-4 text-purple-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          Business Rules
                        </h3>
                        <p className="text-gray-400 text-sm">
                          {loading || (queryMode === 'reports' && isReportGenerating) ? "Processing..." : "Automatic compliance & validation"}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Database Selection */}
            <div className="mb-6">
              <Card className={`bg-gray-900/50 border-blue-400/30 transition-all duration-300 ${
                loading || (queryMode === 'reports' && isReportGenerating) ? 'border-blue-500/60 bg-blue-900/20' : ''
              }`}>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 ${
                        loading || (queryMode === 'reports' && isReportGenerating)
                          ? 'bg-blue-500/40 border border-blue-400/60' 
                          : 'bg-blue-500/20'
                      }`}>
                        {loading || (queryMode === 'reports' && isReportGenerating) ? (
                          <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                        ) : (
                          <Database className="w-4 h-4 text-blue-400" />
                        )}
                      </div>
                      <div>
                        <h3 className="text-white font-medium flex items-center gap-2">
                          {hasCurrentDatabase
                            ? currentDatabase?.db_name
                            : "No Database Selected"}
                          {(loading || (queryMode === 'reports' && isReportGenerating)) && (
                            <Badge variant="outline" className="border-blue-400/50 text-blue-400 text-xs">
                              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                              {queryMode === 'reports' ? 'Generating Report' : 'Processing'}
                            </Badge>
                          )}
                        </h3>
                        <p className="text-gray-400 text-sm">
                          {loading 
                            ? "Database is processing your query..."
                            : (queryMode === 'reports' && isReportGenerating)
                            ? "Database is generating your report..."
                            : hasCurrentDatabase
                            ? `Connected to ${
                                currentDatabase?.db_type || "MSSQL"
                              } database`
                            : "Please select a database in User Configuration"
                          }
                        </p>
                      </div>
                    </div>
                    {hasCurrentDatabase && (
                      <Badge
                        variant="outline"
                        className={`transition-all duration-300 ${
                          loading || (queryMode === 'reports' && isReportGenerating)
                            ? 'border-blue-400/50 text-blue-400 bg-blue-900/20' 
                            : 'border-green-400/30 text-green-400'
                        }`}
                      >
                        {loading || (queryMode === 'reports' && isReportGenerating) ? (
                          <>
                            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                            {queryMode === 'reports' ? 'Generating' : 'Busy'}
                          </>
                        ) : (
                          '✓ Connected'
                        )}
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
                loading={loading || (queryMode === 'reports' && isReportGenerating)}
              />
              
              {/* Quick Navigation */}
              <div className="mt-4 flex items-center gap-2 text-sm text-gray-400">
                <span>Quick access:</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => router.push('/ai-results')}
                  disabled={loading || (queryMode === 'reports' && isReportGenerating)}
                  className={`transition-all duration-300 ${
                    loading || (queryMode === 'reports' && isReportGenerating)
                      ? 'text-gray-500 cursor-not-allowed' 
                      : 'text-purple-400 hover:text-purple-300 hover:bg-purple-400/10'
                  }`}
                >
                  <FileText className="w-4 h-4 mr-1" />
                  {loading 
                    ? 'Processing...' 
                    : (queryMode === 'reports' && isReportGenerating)
                    ? 'Generating Report...'
                    : 'View AI Reports'
                  }
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
                currentQuery={currentQuery}
              />
              ) : (
                <IntegratedReportGenerator
                  userId={user?.user_id}
                  onReportComplete={handleReportComplete}
                  onReportStart={handleReportStart}
                  isReportGenerating={isReportGenerating}
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
              <Card className={`bg-gray-900/50 border-blue-400/30 transition-all duration-300 ${
                loading || (queryMode === 'reports' && isReportGenerating) ? 'border-blue-500/60 bg-blue-900/20' : ''
              }`}>
                <CardContent className="pt-12 pb-12 text-center">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 ${
                    loading || (queryMode === 'reports' && isReportGenerating)
                      ? 'bg-blue-500/40 border border-blue-400/60 scale-110' 
                      : 'bg-blue-500/20'
                  }`}>
                    {loading || (queryMode === 'reports' && isReportGenerating) ? (
                      <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
                    ) : (
                      <Database className="w-8 h-8 text-blue-400" />
                    )}
                  </div>
                  <h3 className="text-white text-lg font-medium mb-2">
                    {loading 
                      ? "Processing Your Query..." 
                      : (queryMode === 'reports' && isReportGenerating)
                      ? "Generating Your Report..."
                      : "Ready to Get Started"
                    }
                  </h3>
                  <p className="text-gray-400 mb-4">
                    {loading 
                      ? "AI is analyzing your question and generating results. Please wait..."
                      : (queryMode === 'reports' && isReportGenerating)
                      ? "AI is analyzing multiple data sources and generating comprehensive insights. This may take several minutes..."
                      : queryMode === 'query' 
                      ? 'Ask your question in natural language above to get started'
                      : 'Describe what you want to analyze and generate comprehensive reports'
                    }
                  </p>
                  {!hasCurrentDatabase && !loading && !(queryMode === 'reports' && isReportGenerating) && (
                    <p className="text-yellow-400 text-sm mb-4">
                      Please select a database first
                    </p>
                  )}
                  {(loading || (queryMode === 'reports' && isReportGenerating)) && (
                    <div className="mb-4">
                      <div className="flex items-center justify-center gap-2 text-blue-400 text-sm">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>{queryMode === 'reports' ? 'Generating Report...' : 'Processing...'}</span>
                      </div>
                      <Progress value={queryProgress} className="h-1 mt-2 max-w-xs mx-auto" />
                    </div>
                  )}
                  <div className="flex justify-center gap-2 mt-4">
                    <Badge variant="outline" className="border-blue-400/30 text-blue-400">
                      {loading 
                        ? 'AI Processing' 
                        : (queryMode === 'reports' && isReportGenerating)
                        ? 'AI Report Generation'
                        : queryMode === 'query' 
                        ? 'Quick Query Mode' 
                        : 'AI Report Mode'
                      }
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
                disabled={loading || (queryMode === 'reports' && isReportGenerating)}
                className={`transition-all duration-300 ${
                  loading || (queryMode === 'reports' && isReportGenerating)
                    ? 'bg-gray-800/90 border-blue-400/50 text-blue-400/70 cursor-not-allowed' 
                    : 'bg-gray-900/90 border-blue-400/30 text-blue-400 hover:bg-blue-400/10 backdrop-blur-sm'
                }`}
              >
                {loading || (queryMode === 'reports' && isReportGenerating) ? (
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                ) : (
                  <History className="w-5 h-5 mr-2" />
                )}
                {loading 
                  ? "Processing..." 
                  : (queryMode === 'reports' && isReportGenerating)
                  ? "Generating Report..."
                  : "Query History"
                }
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
