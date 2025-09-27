"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { useAuthContext, useDatabaseContext } from "@/components/providers";
import { useDatabaseOperations } from "@/lib/hooks/use-database-operations";
import { useTaskCreator } from "@/components/task-manager";
import { useReports } from "@/lib/hooks/use-reports";

import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Label } from "@/components/ui/label";
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
import { ContentWrapper } from "@/components/layout/ContentWrapper";
import { useTheme } from "@/store/theme-store";
import { DatabaseQueryStatsCards, DatabaseQueryHeader } from "@/components/database-query/components";
import { QuickSuggestions } from "@/components/file-query/QuickSuggestions";
import { QueryForm } from "@/components/shared/QueryForm";
import { IntegratedReportGenerator } from "@/components/database-query/IntegratedReportGenerator";

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
  const reports = useReports();

  // State
  const [queryMode, setQueryMode] = useState<'query' | 'reports'>('query');
  const [showHistory, setShowHistory] = useState(false);
  const [showResultOverlay, setShowResultOverlay] = useState(false);
  const [completedQuery, setCompletedQuery] = useState("");
  const [currentQuery, setCurrentQuery] = useState("");
  const [queryProgress, setQueryProgress] = useState(0);
  const [processingSteps, setProcessingSteps] = useState<string[]>([]);
  const [queryInput, setQueryInput] = useState("");
  const [selectedModel, setSelectedModel] = useState("gemini");

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

  // ESC key handler for closing overlays/panels
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (showResultOverlay) {
          setShowResultOverlay(false);
        } else if (showHistory) {
          setShowHistory(false);
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [showResultOverlay, showHistory]);

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
      `${queryMode === 'query' ? 'Query' : 'Report'}: ${query.substring(0, 50)}${query.length > 50 ? '...' : ''}`,
      `${queryMode === 'query' ? 'Executing database query' : 'Generating AI report'}: "${query}"`,
      {
        user_id: user.user_id,
        query: query,
        database_id: currentDatabase?.database_id,
        mode: queryMode,
      }
    );

    // Execute the task in background
    executeTask(
      taskId,
      async () => {
        try {
          setCurrentQuery(query);
          setQueryInput(query);
          console.log("Sending database query with user ID:", user.user_id);

          // Only handle database queries here - AI reports are handled by IntegratedReportGenerator
          const response = await sendQuery({
            userId: user.user_id,
            question: query,
            database_id: currentDatabase?.database_id,
            model: selectedModel,
          });

          console.log("Query response:", response);
          toast.success("Query submitted successfully!");
          
          // Store query result in sessionStorage for the results page
          const queryResult = {
            query: query,
            userId: user.user_id,
            timestamp: new Date().toISOString(),
            result: {
              payload: {
                data: response.data?.payload?.data || []
              }
            }
          };
          sessionStorage.setItem("databaseQueryResult", JSON.stringify(queryResult));
          
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
      console.error("Failed to execute query:", error);
    });
  }, [user?.user_id, hasCurrentDatabase, currentDatabase?.database_id, sendQuery, createQueryTask, executeTask, queryMode, reports, selectedModel]);

  const handleModeChange = useCallback((mode: 'query' | 'reports') => {
    setQueryMode(mode);
    // Close any open overlays when changing modes
    setShowResultOverlay(false);
    setShowHistory(false);
  }, []);

  const handleViewResults = useCallback(() => {
    setShowResultOverlay(false);
    // Navigate to appropriate results page based on query mode
    if (queryMode === 'reports') {
      router.push('/ai-reports');
    } else {
      router.push('/database-query-results');
    }
  }, [router, queryMode]);

  const handleToggleHistory = useCallback(() => {
    setShowHistory(prev => {
      // Close result overlay when opening history
      if (!prev) {
        setShowResultOverlay(false);
      }
      return !prev;
    });
  }, []);

  // Close history panel handler
  const handleCloseHistory = useCallback(() => {
    setShowHistory(false);
  }, []);

  // Close result overlay handler
  const handleCloseResultOverlay = useCallback(() => {
    setShowResultOverlay(false);
    setCompletedQuery("");
  }, []);

  // Query selection handler
  const handleQuerySelect = useCallback((query: string) => {
    setQueryInput(query);
    setShowHistory(false); // Close panel after selecting a query
  }, []);

  return (
    <PageLayout
      background={["frame", "gridframe"]}
      maxWidth="7xl"
      className="database-query-page min-h-screen flex flex-col justify-center py-6"
    >
      <div className="mt-2">
        <style
          dangerouslySetInnerHTML={{
            __html: `
            .database-query-page textarea {
              background: var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04)) !important;
            }
          `,
          }}
        />
      
      {/* Welcome Header */}
      <ContentWrapper className="mb-12">
        <div className="flex items-center justify-between">
        <div>
          <h1 
            className="text-4xl font-bold mb-2 block"
            style={{
              background:
                "radial-gradient(70.83% 118.23% at 55.46% 50%, #0DAC5C 0%, #FFFFFF 84.18%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              color: "transparent",
              display: "block",
              backgroundSize: "100% 100%",
              backgroundRepeat: "no-repeat",
            }}
          >
            Hi there, {user?.username || ""}
          </h1>
          <p 
            className="text-xl block"
            style={{
              background:
                "radial-gradient(70.83% 118.23% at 55.46% 50%, #0DAC5C 0%, #FFFFFF 84.18%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              color: "transparent",
              display: "block",
              backgroundSize: "100% 100%",
              backgroundRepeat: "no-repeat",
            }}
          >
            What would you like to know?
          </p>
        </div>
        <Button
          variant="outline"
          className="text-white flex items-center gap-2"
          style={{
            background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
            border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
            height: "48px",
            minWidth: "64px",
            borderRadius: "99px",
          }}
          onClick={handleToggleHistory}
        >
          <Image
            src="/file-query/history.svg"
            alt="History"
            width={16}
            height={16}
            className="h-4 w-4"
          />
          History
          {history.length > 0 && (
            <Badge variant="secondary" className="ml-1">
              {history.length}
            </Badge>
          )}
        </Button>
        </div>
      </ContentWrapper>

      {/* Mode Toggle */}
      <ContentWrapper className="mb-8">
        <QueryModeToggle
          mode={queryMode}
          onModeChange={handleModeChange}
          hasDatabase={hasCurrentDatabase}
          loading={loading}
        />
      </ContentWrapper>

      {/* Main Content - Full Width */}
      <ContentWrapper className="mb-16">
        <div className="space-y-6">
          <div className="px-2 py-2 query-content-gradient">
            <div className="flex items-start">
              <Image
                src="/file-query/filerobot.svg"
                alt="Database Robot"
                width={120}
                height={120}
                className="flex-shrink-0 -ml-2"
              />
              <div className="flex flex-col justify-start pt-5 -ml-4 z-10">
                <h3 className="text-white font-semibold text-xl">
                  {queryMode === 'query' ? 'Database Query' : 'AI Reports'}
                </h3>
              </div>
            </div>

            <div className="relative z-10">
              {queryMode === 'query' ? (
                <QueryForm
                  query={queryInput}
                  setQuery={setQueryInput}
                  isExecuting={loading}
                  onExecuteClick={() => handleQuerySubmit(queryInput)}
                  placeholder="Ask your question in natural language... (e.g., 'Show me all users from last month')"
                  placeholderType="database"
                  buttonText="Ask"
                  showClearButton={true}
                  disabled={!hasCurrentDatabase}
                  enableTypewriter={true}
                  model={selectedModel}
                  onModelChange={setSelectedModel}
                  showModelSelector={true}
                />
              ) : (
                <IntegratedReportGenerator
                  userId={user?.user_id}
                  onReportComplete={(results) => {
                    console.log("Report completed:", results);
                    setCompletedQuery(queryInput);
                    setShowResultOverlay(true);
                  }}
                  onReportStart={() => {
                    console.log("Report generation started");
                  }}
                  isReportGenerating={reports.isGenerating}
                />
              )}
            </div>
          </div>
        </div>
      </ContentWrapper>

      {/* Quick Suggestions Section */}
      <ContentWrapper className="mb-8">
        <QuickSuggestions
          title={queryMode === 'query' ? "Database Query Suggestions" : "AI Report Suggestions"}
          suggestions={
            queryMode === 'query' ? [
              { 
                text: "Show me all users from last month", 
                query: "Show me all users from last month",
                icon: <User className="h-4 w-4 text-green-400" />
              },
              { 
                text: "What are the top performing products?", 
                query: "What are the top performing products?",
                icon: <BarChart3 className="h-4 w-4 text-green-400" />
              },
              { 
                text: "Find orders with total amount greater than $1000", 
                query: "Find orders with total amount greater than $1000",
                icon: <Database className="h-4 w-4 text-green-400" />
              },
              { 
                text: "Show me revenue trends over time", 
                query: "Show me revenue trends over time",
                icon: <BarChart3 className="h-4 w-4 text-green-400" />
              },
            ] : [
              { 
                text: "Generate a comprehensive sales report for Q4", 
                query: "Generate a comprehensive sales report for Q4",
                icon: <FileText className="h-4 w-4 text-green-400" />
              },
              { 
                text: "Create a customer analytics dashboard", 
                query: "Create a customer analytics dashboard",
                icon: <BarChart3 className="h-4 w-4 text-green-400" />
              },
              { 
                text: "Generate a financial performance summary", 
                query: "Generate a financial performance summary",
                icon: <Database className="h-4 w-4 text-green-400" />
              },
              { 
                text: "Create a user engagement report", 
                query: "Create a user engagement report",
                icon: <User className="h-4 w-4 text-green-400" />
              },
            ]
          }
          onQuerySelect={handleQuerySelect}
        />
      </ContentWrapper>

      {/* Query History Panel */}
      {showHistory && (
        <QueryHistoryPanel
          history={history}
          loading={historyLoading}
          onClose={handleCloseHistory}
          onQuerySelect={handleQuerySelect}
        />
      )}

      {/* Query Result Overlay */}
      <QueryResultOverlay
        isVisible={showResultOverlay}
        onViewResults={handleViewResults}
        onClose={handleCloseResultOverlay}
        queryText={completedQuery}
        queryMode={queryMode}
      />
      </div>
    </PageLayout>
  );
}