"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { useAuthContext, useDatabaseContext } from "@/components/providers";
import { useDatabaseOperations } from "@/lib/hooks/use-database-operations";
import { useTaskCreator } from "@/components/task-manager";

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
    <PageLayout
      background={["frame", "gridframe"]}
      maxWidth="7xl"
      className="database-query-page"
    >
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
      <div className="flex items-center justify-between mb-8">
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

      {/* Mode Toggle */}
      <div className="flex items-center gap-3 mb-8">
        <Label htmlFor="query-mode" className="text-white font-medium">
          Quick Query
        </Label>
        <button
          onClick={() => setQueryMode(queryMode === 'query' ? 'reports' : 'query')}
          className="relative inline-flex h-6 w-11 items-center transition-colors rounded-full"
          style={{
            backgroundColor: "var(--white-12, rgba(255, 255, 255, 0.12))",
            backdropFilter: "blur(29.09090805053711px)"
          }}
        >
          <span
            className={`inline-block h-4 w-4 transform transition-transform rounded-full ${
              queryMode === 'query' ? "translate-x-6" : "translate-x-1"
            }`}
            style={{
              backgroundColor: "var(--primary-light, rgba(158, 251, 205, 1))"
            }}
          />
        </button>
        <Label htmlFor="query-mode" className="text-white font-medium">
          AI Reports
        </Label>
      </div>

      {/* Main Content - Full Width */}
      <div className="space-y-6">
        {queryMode === 'query' ? (
          <div className="p-6 query-content-gradient">
            <div className="flex items-start">
              <Image
                src="/file-query/filerobot.svg"
                alt="Database Robot"
                width={120}
                height={120}
                className="flex-shrink-0"
              />
              <div className="flex flex-col justify-start pt-5 -ml-8 z-10">
                <h3 className="text-white font-semibold text-2xl">
                  Database Query
                </h3>
              </div>
            </div>

            <div className="relative -mt-16 px-4 z-10">
              <DatabaseQueryForm
                onSubmit={handleQuerySubmit}
                loading={loading}
                hasDatabase={hasCurrentDatabase}
                currentQuery={currentQuery}
              />
            </div>
          </div>
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

      {/* Quick Suggestions Section */}
      <div className="mt-12">
        <h3 className="text-xl font-semibold text-white mb-6">
          Quick suggestion
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((index) => (
            <div
              key={index}
              className="p-4 query-content-gradient"
            >
              <div className="space-y-2">
                <p className="text-sm text-slate-400">
                  Use time references: 'last week', 'this month', 'yesterday'
                </p>
                <div className="flex justify-center">
                  <div className="w-6 h-6 bg-green-400 flex items-center justify-center rounded">
                    <BarChart3 className="h-4 w-4 text-white" />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
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
