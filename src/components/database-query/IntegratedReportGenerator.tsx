"use client";

import React, { useState, useEffect, useMemo } from "react";
import { useReports } from "@/lib/hooks/use-reports";
import { useReportStructure } from "@/lib/hooks/use-report-structure";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  FileText,
  Play,
  Square,
  Clock,
  CheckCircle,
  AlertCircle,
  Sparkles,
  BarChart3,
  Loader2,
} from "lucide-react";
import { useUserContext } from "@/lib/hooks/use-user-context";

interface IntegratedReportGeneratorProps {
  userId?: string;
  onReportComplete?: (results: any) => void;
}

export function IntegratedReportGenerator({
  userId,
  onReportComplete,
}: IntegratedReportGeneratorProps) {
  const { user } = useUserContext();
  const [userQuery, setUserQuery] = useState("");
  const [selectedStructure, setSelectedStructure] =
    useState<string>("financial_report");

  // Hooks
  const reports = useReports();
  const reportStructure = useReportStructure();

  // Load report structure on mount - only when userId changes
  useEffect(() => {
    if (userId) {
      reportStructure.loadStructure(userId);
    }
  }, [userId]); // intentionally removed reportStructure from deps

  // Handle report completion
  useEffect(() => {
    if (reports.reportResults && onReportComplete) {
      onReportComplete(reports.reportResults);
    }
  }, [reports.reportResults, onReportComplete]);

  // Generate report
  const handleGenerateReport = async () => {
    if (!user?.user_id || !userQuery.trim()) return;

    console.log('Starting report generation for user:', user.user_id);
    console.log('User query:', userQuery);

    try {
      const taskId = await reports.generateReport({
        user_id: user.user_id,
        user_query: userQuery,
      });

      console.log('Report generation started, task ID:', taskId);

      // Start monitoring the task with proper callbacks
      reports.startMonitoring(taskId, {
        onProgress: (status) => {
          console.log("Report progress:", status);
        },
        onComplete: (results) => {
          console.log("Report completed:", results);
          // Store results for the results page
          if (onReportComplete) {
            onReportComplete(results);
          }
        },
        onError: (error) => {
          console.error("Report failed:", error);
        },
        pollInterval: 2000, // Poll every 2 seconds
      });
    } catch (error) {
      console.error("Failed to generate report:", error);
    }
  };

  // Generate report and wait for completion
  const handleGenerateReportAndWait = async () => {
    if (!user?.user_id || !userQuery.trim()) return;

    try {
      await reports.generateReportAndWait({
        user_id: user.user_id,
        user_query: userQuery,
      });
    } catch (error) {
      console.error("Failed to generate report:", error);
    }
  };

  // Stop monitoring
  const handleStopMonitoring = () => {
    reports.stopMonitoring();
  };

  // Get progress message
  const getProgressMessage = () => {
    if (!reports.currentTask) return "";
    
    const task = reports.currentTask;
    
    switch (task.status) {
      case 'pending':
        return 'Report generation queued and waiting to start...';
      case 'processing':
        if (task.current_step) {
          return `Processing: ${task.current_step}`;
        }
        return `Processing queries... (${task.processed_queries}/${task.total_queries})`;
      case 'completed':
        return 'Report generation completed successfully!';
      case 'failed':
        return `Failed: ${task.error || 'Unknown error'}`;
      default:
        return task.progress || 'Processing...';
    }
  };

  // Get detailed progress info
  const getDetailedProgressInfo = () => {
    if (!reports.currentTask) return null;
    
    const task = reports.currentTask;
    const processed = task.processed_queries || 0;
    const total = task.total_queries || 0;
    const successful = task.successful_queries || 0;
    const failed = task.failed_queries || 0;
    
    return {
      processed,
      total,
      successful,
      failed,
      remaining: total - processed,
      successRate: total > 0 ? Math.round((successful / total) * 100) : 0,
    };
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-green-500";
      case "failed":
        return "bg-red-500";
      case "processing":
        return "bg-blue-500";
      case "pending":
        return "bg-yellow-500";
      default:
        return "bg-gray-500";
    }
  };

  // Memoize the report structure section to prevent unnecessary re-renders
  const reportStructureSection = useMemo(() => {
    if (reportStructure.isLoading) {
      return (
        <Card className="bg-gray-900/50 border-purple-400/30">
          <CardContent className="pt-12 pb-12 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400 mx-auto mb-4"></div>
            <p className="text-gray-400">Loading report templates...</p>
          </CardContent>
        </Card>
      );
    }

    if (!reportStructure.structure) return null;

    return (
      <Card className="bg-gray-900/50 border-purple-400/30">
        <CardHeader>
          <CardTitle className="text-purple-400 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Report Template
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <label className="text-sm font-medium text-white">
              Select Report Type
            </label>
            <select
              value={selectedStructure}
              onChange={(e) => setSelectedStructure(e.target.value)}
              className="w-full p-3 border rounded-lg bg-gray-800/50 border-purple-400/30 text-white"
            >
              {Object.keys(reportStructure.structure).map((key) => (
                <option key={key} value={key}>
                  {key
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                </option>
              ))}
            </select>
            <div className="text-xs text-gray-400 bg-gray-800/30 p-3 rounded border border-gray-700">
              <div className="font-medium mb-2">Template Preview:</div>
              <div className="whitespace-pre-wrap text-gray-300">
                {reportStructure.structure[selectedStructure]}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }, [reportStructure.structure, reportStructure.isLoading, selectedStructure]);

  return (
    <div className="space-y-6">
      {/* Report Structure Selection */}
      {reportStructureSection}

      {/* Query Input */}
      <Card className="bg-gray-900/50 border-purple-400/30">
        <CardHeader>
          <CardTitle className="text-purple-400 flex items-center gap-2">
            <Sparkles className="w-5 h-5" />
            AI Report Generation
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-white">
              What would you like to know?
            </label>
            <Textarea
              value={userQuery}
              onChange={(e) => setUserQuery(e.target.value)}
              placeholder="e.g., Show me the financial report of May, or Generate a comprehensive sales analysis for Q2"
              className="min-h-[120px] bg-gray-800/50 border-purple-400/30 text-white placeholder:text-gray-400 resize-none"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 flex-wrap">
            <Button
              onClick={handleGenerateReport}
              disabled={!userQuery.trim() || reports.isGenerating}
              className="bg-purple-600 hover:bg-purple-700 text-white flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              Generate Report
            </Button>

            <Button
              onClick={handleGenerateReportAndWait}
              disabled={!userQuery.trim() || reports.isGenerating}
              variant="outline"
              className="border-purple-400/30 text-purple-400 hover:bg-purple-400/10 flex items-center gap-2"
            >
              <Clock className="h-4 w-4" />
              Generate & Wait
            </Button>

            {reports.currentTask && (
              <Button
                onClick={() => reports.refreshTaskStatus(reports.currentTask!.task_id)}
                variant="outline"
                className="border-blue-400/30 text-blue-400 hover:bg-blue-400/10 flex items-center gap-2"
              >
                <Loader2 className="h-4 w-4" />
                Refresh Status
              </Button>
            )}

            {reports.isGenerating && (
              <Button
                onClick={handleStopMonitoring}
                variant="destructive"
                className="flex items-center gap-2"
              >
                <Square className="h-4 w-4" />
                Stop
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Progress Display */}
      {reports.isGenerating && (
        <Card className="bg-gray-900/50 border-purple-400/30">
          <CardHeader>
            <CardTitle className="text-purple-400 flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin" />
              Report Generation Progress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-white">{getProgressMessage()}</span>
                <span className="text-purple-400 font-medium">
                  {reports.progress}%
                </span>
              </div>
              <Progress value={reports.progress} className="w-full" />
              
              {/* Detailed Progress Stats */}
              {(() => {
                const progressInfo = getDetailedProgressInfo();
                if (!progressInfo) return null;
                
                return (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <div className="text-center p-2 bg-gray-800/30 rounded border border-gray-700">
                      <div className="text-white font-medium">{progressInfo.processed}</div>
                      <div className="text-gray-400">Processed</div>
                    </div>
                    <div className="text-center p-2 bg-gray-800/30 rounded border border-gray-700">
                      <div className="text-white font-medium">{progressInfo.remaining}</div>
                      <div className="text-gray-400">Remaining</div>
                    </div>
                    <div className="text-center p-2 bg-gray-800/30 rounded border border-gray-700">
                      <div className="text-green-400 font-medium">{progressInfo.successRate}%</div>
                      <div className="text-gray-400">Success Rate</div>
                    </div>
                    <div className="text-center p-2 bg-gray-800/30 rounded border border-gray-700">
                      <div className="text-red-400 font-medium">{progressInfo.failed}</div>
                      <div className="text-gray-400">Failed</div>
                    </div>
                  </div>
                );
              })()}
              
              {reports.estimatedTimeRemaining && (
                <div className="text-xs text-gray-400 flex items-center gap-2">
                  <Clock className="w-4 h-4" />
                  Estimated time remaining: {reports.estimatedTimeRemaining}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Current Task Status */}
      {reports.currentTask && (
        <Card className="bg-gray-900/50 border-purple-400/30">
          <CardHeader>
            <CardTitle className="text-purple-400 flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Task Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            {/* Real-time Status Bar */}
            <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-300">Current Status:</span>
                <Badge className={getStatusColor(reports.currentTask.status)}>
                  {reports.currentTask.status.toUpperCase()}
                </Badge>
              </div>
              <div className="text-sm text-white">
                {reports.currentTask.current_step || 'Initializing...'}
              </div>
              {reports.currentTask.progress && (
                <div className="text-xs text-gray-400 mt-1">
                  {reports.currentTask.progress}
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-white">
                  {reports.currentTask.total_queries}
                </div>
                <div className="text-sm text-gray-400">Total Queries</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400">
                  {reports.currentTask.successful_queries}
                </div>
                <div className="text-sm text-gray-400">Successful</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-400">
                  {reports.currentTask.failed_queries}
                </div>
                <div className="text-sm text-gray-400">Failed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {reports.currentTask.progress_percentage}%
                </div>
                <div className="text-sm text-gray-400">Progress</div>
              </div>
            </div>

            <Separator className="my-4 bg-gray-700" />

            {/* Processing Details */}
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Processing Time:</span>
                <span className="text-white">
                  {reports.currentTask.processing_time_seconds 
                    ? `${reports.currentTask.processing_time_seconds}s`
                    : 'Calculating...'
                  }
                </span>
              </div>
              
              {reports.currentTask.started_at && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Started:</span>
                  <span className="text-white">
                    {new Date(reports.currentTask.started_at).toLocaleTimeString()}
                  </span>
                </div>
              )}
              
              {reports.currentTask.completed_at && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Completed:</span>
                  <span className="text-white">
                    {new Date(reports.currentTask.completed_at).toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Error Display */}
      {reports.error && (
        <Alert
          variant="destructive"
          className="border-red-500/30 bg-red-900/20"
        >
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="text-red-300">
            {reports.error}
          </AlertDescription>
        </Alert>
      )}

      {/* Report Results Preview */}
      {reports.reportResults && (
        <Card className="bg-gray-900/50 border-green-400/30">
          <CardHeader>
            <CardTitle className="text-green-400 flex items-center gap-2">
              <CheckCircle className="w-5 h-5" />
              Report Generated Successfully!
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">
                    {reports.reportResults.total_queries}
                  </div>
                  <div className="text-sm text-gray-400">Total Queries</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">
                    {reports.reportResults.successful_queries}
                  </div>
                  <div className="text-sm text-gray-400">Successful</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-400">
                    {reports.reportResults.failed_queries}
                  </div>
                  <div className="text-sm text-gray-400">Failed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400">
                    {reports.reportResults.database_id}
                  </div>
                  <div className="text-sm text-gray-400">Database ID</div>
                </div>
              </div>

              <div className="text-center">
                <Button
                  onClick={() => {
                    // Store results and redirect to detailed view
                    sessionStorage.setItem(
                      "reportResults",
                      JSON.stringify(reports.reportResults)
                    );
                    window.open("/ai-results", "_blank");
                  }}
                  className="bg-green-600 hover:bg-green-700 text-white"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  View Full Report
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
 