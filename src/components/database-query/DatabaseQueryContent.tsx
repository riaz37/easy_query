"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useAuthContext, useDatabaseContext } from "@/components/providers";
import { useDatabaseOperations } from "@/lib/hooks/use-database-operations";
import { useTaskCreator } from "@/components/task-manager";

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
  Brain,
  Zap,
} from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { DatabaseQueryForm } from "@/components/database-query/DatabaseQueryForm";
import { QueryHistoryPanel } from "@/components/database-query/QueryHistoryPanel";
import { QueryModeToggle } from "@/components/database-query/QueryModeToggle";
import { ReportGeneratorBackground as ReportGenerator } from "@/components/reports";
import { QueryResultOverlay } from "@/components/ui/query-result-overlay";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";
import { useTheme } from "@/store/theme-store";
import { DatabaseQueryStatsCards, DatabaseQueryHeader } from "@/components/database-query/components";

export function DatabaseQueryContent() {
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
  const { createQueryTask, executeTask } = useTaskCreator();

  // State
  const [queryMode, setQueryMode] = useState<'query' | 'reports'>('query');
  const [showHistory, setShowHistory] = useState(false);
  const [showResultOverlay, setShowResultOverlay] = useState(false);
  const [completedQuery, setCompletedQuery] = useState("");
  const [currentQuery, setCurrentQuery] = useState("");
  const [queryProgress, setQueryProgress] = useState(0);
  const [processingSteps, setProcessingSteps] = useState<string[]>([]);

  // Memoize the processing steps to prevent recreation
  const defaultProcessingSteps = useMemo(() => [
    "Analyzing your query...",
    "Connecting to database...",
    "Generating SQL...",
    "Executing query...",
    "Processing results...",
    "Formatting data...",
    "Preparing response..."
  ], []);

  // Voice agent event handlers
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

  // Load query history on mount
  useEffect(() => {
    if (user?.user_id) {
      fetchQueryHistory();
    }
  }, [user?.user_id, fetchQueryHistory]);

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

    // Create a background task for query execution
    const taskId = createQueryTask(
      `Query: ${query.substring(0, 50)}${query.length > 50 ? '...' : ''}`,
      `Executing database query: "${query}"`,
      {
        user_id: user.user_id,
        query: query,
        database_id: currentDatabase?.database_id,
      }
    );

    // Execute the task in background
    executeTask(
      taskId,
      async () => {
        try {
          setCurrentQuery(query);
          console.log("Sending query with user ID:", user.user_id);

          const response = await sendQuery({
            userId: user.user_id,
            question: query,
            database_id: currentDatabase?.database_id,
          });

          console.log("Query response:", response);
          toast.success("Query submitted successfully!");
          
          // Show result overlay
          setCompletedQuery(query);
          setShowResultOverlay(true);
          
          return response;
        } catch (error) {
          console.error("Query failed:", error);
          toast.error("Failed to execute query. Please try again.");
          throw error;
        }
      }
    ).catch((error) => {
      console.error('Failed to execute query:', error);
    });
  }, [user?.user_id, hasCurrentDatabase, currentDatabase?.database_id, sendQuery, createQueryTask, executeTask]);

  const handleModeChange = useCallback((mode: 'query' | 'reports') => {
    setQueryMode(mode);
  }, []);

  const handleViewResults = useCallback(() => {
    setShowResultOverlay(false);
    // Navigate to results page or show results in a modal
    router.push('/database-query-results');
  }, [router]);

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
          
          <Button
            variant="outline"
            onClick={handleToggleHistory}
            className="flex items-center gap-2 border-gray-600 text-gray-300 hover:bg-gray-700"
          >
            <History className="w-4 h-4" />
            History
            {history.length > 0 && (
              <Badge variant="secondary" className="ml-1">
                {history.length}
              </Badge>
            )}
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="space-y-6">
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
            onReportComplete={(results) => {
              console.log('Report completed:', results);
              toast.success('Report generated successfully!');
            }}
            onReportStart={() => {
              console.log('Report generation started');
            }}
          />
        )}
      </div>

      {/* Query History Panel */}
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
