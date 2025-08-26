"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
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
import { ReportGenerator } from "@/components/reports";

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

  // Memoize the processing steps to prevent recreation
  const defaultProcessingSteps = useMemo(() => [
    "Analyzing your question...",
    "Connecting to database...",
    "Processing query...",
    "Applying business rules...",
    "Generating results..."
  ], []);

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
      setProcessingSteps(defaultProcessingSteps);
      
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
  }, [loading, defaultProcessingSteps]);

  // Memoize handlers to prevent unnecessary re-renders
  const handleReportComplete = useCallback((results: any) => {
    console.log('Report completed:', results);
    setIsReportGenerating(false);
    // Store results for the results page
    sessionStorage.setItem('reportResults', JSON.stringify(results));
  }, []);

  const handleReportStart = useCallback(() => {
    setIsReportGenerating(true);
  }, []);

  const handleBackToDashboard = useCallback(() => {
    router.push("/");
  }, [router]);

  const handleQuerySubmit = useCallback(async (query: string) => {
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
        user_id: user.user_id,
        query: query,
        database_id: currentDatabase?.database_id,
      });

      console.log("Query response:", response);
      toast.success("Query submitted successfully!");
    } catch (error) {
      console.error("Query submission failed:", error);
      toast.error("Failed to submit query. Please try again.");
    }
  }, [hasCurrentDatabase, user?.user_id, sendQuery, currentDatabase?.database_id]);

  const handleModeChange = useCallback((mode: 'query' | 'reports') => {
    setQueryMode(mode);
    // Reset states when switching modes
    if (mode === 'query') {
      setCurrentQuery("");
      setIsReportGenerating(false);
    } else {
      setCurrentQuery("");
    }
  }, []);

  const handleToggleHistory = useCallback(() => {
    setShowHistory(prev => !prev);
  }, []);

  // Memoize the current status to prevent unnecessary re-renders
  const currentStatus = useMemo(() => {
    if (loading) return 'loading';
    if (queryMode === 'reports' && isReportGenerating) return 'generating_report';
    return 'ready';
  }, [loading, queryMode, isReportGenerating]);

  // Memoize the status message
  const statusMessage = useMemo(() => {
    switch (currentStatus) {
      case 'loading':
        return "Processing Your Query...";
      case 'generating_report':
        return "Generating Your Report...";
      default:
        return "Ready to Get Started";
    }
  }, [currentStatus]);

  // Memoize the status description
  const statusDescription = useMemo(() => {
    switch (currentStatus) {
      case 'loading':
        return "AI is analyzing your question and generating results. Please wait...";
      case 'generating_report':
        return "AI is analyzing multiple data sources and generating comprehensive insights. This may take several minutes...";
      default:
        return queryMode === 'query' 
          ? 'Ask your question in natural language above to get started'
          : 'Describe what you want to analyze and generate comprehensive reports';
    }
  }, [currentStatus, queryMode]);

  // Memoize the status badge
  const statusBadge = useMemo(() => {
    switch (currentStatus) {
      case 'loading':
        return 'AI Processing';
      case 'generating_report':
        return 'AI Report Generation';
      default:
        return queryMode === 'query' ? 'Quick Query Mode' : 'AI Report Mode';
    }
  }, [currentStatus, queryMode]);

  return (
    <div className="w-full min-h-screen relative bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-900">
      {/* Add top padding to account for fixed navbar */}
      <div className="pt-24 pb-8">
        <div className="container mx-auto px-4">
          <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-blue-500/30 to-blue-600/20 rounded-xl flex items-center justify-center border border-blue-500/40">
                    <Database className="w-6 h-6 text-blue-400" />
                  </div>
                  <div>
                    <h1 className="text-3xl font-bold text-white">
                      Database Query & AI Reports
                    </h1>
                    <p className="text-gray-400">
                      Run quick queries or generate comprehensive AI-powered reports
                    </p>
                  </div>
                </div>

                {/* Status Badges */}
                <div className="flex items-center gap-3">
                  {user?.user_id && (
                    <Badge variant="outline" className="border-green-400/30 text-green-400">
                      <User className="w-4 h-4 mr-2" />
                      User: {user.user_id}
                    </Badge>
                  )}
                  {currentDatabase && (
                    <Badge variant="outline" className="border-blue-400/30 text-blue-400">
                      <Database className="w-4 h-4 mr-2" />
                      DB: {currentDatabase.database_name}
                    </Badge>
                  )}
                </div>
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
                      <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                        <History className="w-4 h-4 text-blue-400" />
                      </div>
                      <div>
                        <h3 className="text-white font-medium">
                          Query History
                        </h3>
                        <p className="text-gray-400 text-sm">
                          Track and reuse past queries
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Mode Toggle */}
            <div className="mb-6">
              <div className="flex items-center justify-between">
                <QueryModeToggle
                  mode={queryMode}
                  onModeChange={handleModeChange}
                  hasDatabase={hasCurrentDatabase}
                  loading={loading}
                />
                
                <div className="flex items-center gap-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleToggleHistory}
                    className="border-blue-400/30 text-blue-400 hover:bg-blue-400/10"
                  >
                    <History className="w-4 h-4 mr-2" />
                    {showHistory ? 'Hide History' : 'Show History'}
                  </Button>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => router.push('/ai-results')}
                    className="border-purple-400/30 text-purple-400 hover:bg-purple-400/10"
                  >
                    <BarChart3 className="w-4 h-4 mr-2" />
                    {loading 
                      ? 'Processing...' 
                      : (queryMode === 'reports' && isReportGenerating)
                      ? 'Generating Report...'
                      : 'View AI Reports'
                    }
                  </Button>
                </div>
              </div>
            </div>

            {/* Main Content */}
            <div className="space-y-6">
              {/* Query Form or Report Generator */}
              {queryMode === 'query' ? (
                <DatabaseQueryForm
                  onSubmit={handleQuerySubmit}
                  loading={loading}
                  hasDatabase={hasCurrentDatabase}
                  currentQuery={currentQuery}
                />
              ) : (
                <ReportGenerator
                  userId={user?.user_id}
                  onReportComplete={handleReportComplete}
                  onReportStart={handleReportStart}
                  isReportGenerating={isReportGenerating}
                />
              )}

              {/* Error Display */}
              {error && (
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
              )}

              {/* Empty State */}
              <Card className={`bg-gray-900/50 border-blue-400/30 transition-all duration-300 ${
                currentStatus !== 'ready' ? 'border-blue-500/60 bg-blue-900/20' : ''
              }`}>
                <CardContent className="pt-12 pb-12 text-center">
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 ${
                    currentStatus !== 'ready'
                      ? 'bg-blue-500/40 border border-blue-400/60 scale-110' 
                      : 'bg-blue-500/20'
                  }`}>
                    {currentStatus !== 'ready' ? (
                      <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
                    ) : (
                      <Database className="w-8 h-8 text-blue-400" />
                    )}
                  </div>
                  <h3 className="text-white text-lg font-medium mb-2">
                    {statusMessage}
                  </h3>
                  <p className="text-gray-400 mb-4">
                    {statusDescription}
                  </p>
                  {!hasCurrentDatabase && currentStatus === 'ready' && (
                    <p className="text-yellow-400 text-sm mb-4">
                      Please select a database first
                    </p>
                  )}
                  {currentStatus !== 'ready' && (
                    <div className="mb-4">
                      <div className="flex items-center justify-center gap-2 text-blue-400 text-sm">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>{currentStatus === 'generating_report' ? 'Generating Report...' : 'Processing...'}</span>
                      </div>
                      <Progress value={queryProgress} className="h-1 mt-2 max-w-xs mx-auto" />
                    </div>
                  )}
                  <div className="flex justify-center gap-2 mt-4">
                    <Badge variant="outline" className="border-blue-400/30 text-blue-400">
                      {statusBadge}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {/* Query History Panel - Fixed positioning */}
      {showHistory && (
        <QueryHistoryPanel
          history={history}
          loading={historyLoading}
          onClose={() => setShowHistory(false)}
          onQuerySelect={(query) => {
            setCurrentQuery(query);
            setShowHistory(false); // Close panel after selecting a query
          }}
        />
      )}
    </div>
  );
}
