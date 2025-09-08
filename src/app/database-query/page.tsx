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
  // Loader2,
  Brain,
  Zap,
} from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { DatabaseQueryForm } from "@/components/database-query/DatabaseQueryForm";
import { QueryHistoryPanel } from "@/components/database-query/QueryHistoryPanel";
import { QueryModeToggle } from "@/components/database-query/QueryModeToggle";
import { ReportGenerator } from "@/components/reports";
import { QueryResultOverlay } from "@/components/ui/query-result-overlay";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";
import { useTheme } from "@/store/theme-store";
import { DatabaseQueryStatsCards, DatabaseQueryHeader } from "@/components/database-query/components";

export default function DatabaseQueryPage() {
  const router = useRouter();
  const { user } = useAuthContext();
  const { currentDatabase, hasCurrentDatabase } = useDatabaseContext();
  const theme = useTheme();
  const isDark = theme === "dark";
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
  const [showResultOverlay, setShowResultOverlay] = useState(false);
  const [completedQuery, setCompletedQuery] = useState<string>("");

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

  // Voice agent event handling
  useEffect(() => {
    const handleVoiceAgentGenerateReport = (event: CustomEvent) => {
      console.log('🎤 Voice agent generate report event:', event.detail)
      if (event.detail.action === 'generate_report') {
        // Switch to reports mode and trigger report generation
        setQueryMode('reports')
        // The ReportGenerator component will handle the actual generation
      }
    }

    const handleVoiceAgentShowMessage = (event: CustomEvent) => {
      console.log('🎤 Voice agent show message event:', event.detail)
      if (event.detail.type === 'info') {
        toast.info(event.detail.message)
      }
    }

    // Add event listeners
    window.addEventListener('voice-agent-generate-report', handleVoiceAgentGenerateReport as EventListener)
    window.addEventListener('voice-agent-show-message', handleVoiceAgentShowMessage as EventListener)

    // Cleanup
    return () => {
      window.removeEventListener('voice-agent-generate-report', handleVoiceAgentGenerateReport as EventListener)
      window.removeEventListener('voice-agent-show-message', handleVoiceAgentShowMessage as EventListener)
    }
  }, [])

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

  const handleViewResults = useCallback(() => {
    setShowResultOverlay(false);
    router.push('/database-query-results');
  }, [router]);

  const handleDismissOverlay = useCallback(() => {
    setShowResultOverlay(false);
  }, []);

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
        userId: user.user_id,
        question: query,
        database_id: currentDatabase?.database_id,
      });

      console.log("Query response:", response);
      toast.success("Query submitted successfully!");
      
      // Store results in sessionStorage for the results page
      if (response.success && response.data) {
        const queryResult = {
          query: query,
          userId: user.user_id,
          databaseId: currentDatabase?.database_id,
          timestamp: new Date().toISOString(),
          result: response.data
        };
        sessionStorage.setItem('databaseQueryResult', JSON.stringify(queryResult));
        
        // Show overlay instead of immediate redirect
        setCompletedQuery(query);
        setShowResultOverlay(true);
      }
    } catch (error) {
      console.error("Query submission failed:", error);
      toast.error("Failed to submit query. Please try again.");
    }
  }, [hasCurrentDatabase, user?.user_id, sendQuery, currentDatabase?.database_id, router]);

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


  return (
    <PageLayout background="enhanced" backgroundIntensity="medium" maxWidth="7xl">
      {/* Header */}
      <DatabaseQueryHeader
        user={user}
        currentDatabase={currentDatabase}
        isDark={isDark}
      />

      {/* Stats Cards */}
      <DatabaseQueryStatsCards isDark={isDark} />

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
                    className="card-button-enhanced"
                  >
                    <History className="w-4 h-4 mr-2" />
                    {showHistory ? 'Hide History' : 'Show History'}
                  </Button>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => router.push('/ai-results')}
                    className="card-button-enhanced"
                    data-voice-action="view report"
                    data-voice-element="view report"
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
                <div className="card-enhanced">
                  <div className="card-content-enhanced">
                    <div className="flex items-center gap-3">
                      <AlertCircle className="w-5 h-5 text-red-400" />
                      <div>
                        <h3 className="text-red-400 font-medium">
                          Query Error
                        </h3>
                        <p className="text-red-300 text-sm">{error}</p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

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

      {/* Query Result Overlay */}
      <QueryResultOverlay
        isVisible={showResultOverlay}
        onViewResults={handleViewResults}
        queryText={completedQuery}
      />
    </PageLayout>
  );
}
