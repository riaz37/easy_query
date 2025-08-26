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
  Brain,
  Zap,
  Database,
  TrendingUp,
  FileBarChart,
} from "lucide-react";
import { useUserContext } from "@/lib/hooks/use-user-context";

interface IntegratedReportGeneratorProps {
  userId?: string;
  onReportComplete?: (results: any) => void;
  onReportStart?: () => void;
  isReportGenerating?: boolean;
}

export function IntegratedReportGenerator({
  userId,
  onReportComplete,
  onReportStart,
  isReportGenerating = false,
}: IntegratedReportGeneratorProps) {
  const { user } = useUserContext();
  const [userQuery, setUserQuery] = useState("");
  const [selectedStructure, setSelectedStructure] =
    useState<string>("financial_report");
  const [reportProgress, setReportProgress] = useState(0);
  const [processingSteps, setProcessingSteps] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);

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

  // Enhanced progress tracking for report generation
  useEffect(() => {
    if (reports.isGenerating) {
      setReportProgress(0);
      setProcessingTime(0);
      setCurrentStep(0);
      
      // Define processing steps for reports
      setProcessingSteps([
        "Analyzing your report request...",
        "Connecting to database...",
        "Generating SQL queries...",
        "Executing database queries...",
        "Processing business rules...",
        "Analyzing data patterns...",
        "Generating insights...",
        "Compiling final report..."
      ]);
      
      // Simulate progress with real-time updates
      const progressInterval = setInterval(() => {
        setReportProgress(prev => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + 2;
        });
      }, 1000);

      // Update current step based on progress
      const stepInterval = setInterval(() => {
        setCurrentStep(prev => {
          const newStep = Math.floor((reportProgress / 100) * processingSteps.length);
          return Math.min(newStep, processingSteps.length - 1);
        });
      }, 2000);

      // Track processing time
      const timeInterval = setInterval(() => {
        setProcessingTime(prev => prev + 1);
      }, 1000);

      return () => {
        clearInterval(progressInterval);
        clearInterval(stepInterval);
        clearInterval(timeInterval);
      };
    } else {
      setReportProgress(0);
      setProcessingTime(0);
      setCurrentStep(0);
      setProcessingSteps([]);
    }
  }, [reports.isGenerating, reportProgress, processingSteps.length]);

  // Generate report
  const handleGenerateReport = async () => {
    if (!user?.user_id || !userQuery.trim()) return;

    console.log('Starting report generation for user:', user.user_id);
    console.log('User query:', userQuery);

    // Notify parent component that report generation has started
    if (onReportStart) {
      onReportStart();
    }

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
          // Update progress based on real status
          if (status.progress_percentage) {
            setReportProgress(status.progress_percentage);
          }
          if (status.current_step) {
            setCurrentStep(processingSteps.findIndex(step => 
              step.toLowerCase().includes(status.current_step.toLowerCase())
            ) || currentStep);
          }
        },
        onComplete: (results) => {
          console.log("Report completed:", results);
          setReportProgress(100);
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

    // Notify parent component that report generation has started
    if (onReportStart) {
      onReportStart();
    }

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

  // Format time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
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
      <Card className={`bg-gray-900/50 border-purple-400/30 transition-all duration-300 ${
        reports.isGenerating ? 'opacity-60 scale-95' : ''
      }`}>
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
              disabled={reports.isGenerating}
              className="w-full p-3 border rounded-lg bg-gray-800/50 border-purple-400/30 text-white disabled:opacity-50"
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
  }, [reportStructure.structure, reportStructure.isLoading, selectedStructure, reports.isGenerating]);

  return (
    <div className="space-y-6">
      {/* Loading Overlay for Report Generation */}
      {reports.isGenerating && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-gray-900/95 border border-purple-400/30 rounded-2xl p-8 max-w-md w-full mx-4">
            <div className="text-center space-y-6">
              {/* Animated Brain Icon */}
              <div className="relative">
                <div className="w-20 h-20 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-full flex items-center justify-center mx-auto border border-purple-400/30">
                  <Brain className="w-10 h-10 text-purple-400 animate-pulse" />
                </div>
                <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                  <Zap className="w-3 h-3 text-white" />
                </div>
              </div>

              {/* Progress Bar */}
              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-purple-400">Generating Report</span>
                  <span className="text-gray-400">{reportProgress}%</span>
                </div>
                <Progress value={reportProgress} className="h-2" />
              </div>

              {/* Current Step */}
              <div className="space-y-3">
                <div className="flex items-center justify-center gap-2 text-purple-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm font-medium">AI Report Generation</span>
                </div>
                <p className="text-gray-300 text-sm">
                  {processingSteps[currentStep] || "Preparing..."}
                </p>
              </div>

              {/* Query Preview */}
              <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
                <p className="text-xs text-gray-400 mb-1">Processing:</p>
                <p className="text-white text-sm font-medium">
                  {userQuery.length > 60 
                    ? userQuery.substring(0, 60) + "..." 
                    : userQuery
                  }
                </p>
              </div>

              {/* Processing Stats */}
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div className="text-center">
                  <div className="text-purple-400 font-medium">Time</div>
                  <div className="text-gray-300">{formatTime(processingTime)}</div>
                </div>
                <div className="text-center">
                  <div className="text-purple-400 font-medium">Step</div>
                  <div className="text-gray-300">{currentStep + 1}/{processingSteps.length}</div>
                </div>
              </div>

              {/* Tips */}
              <div className="text-xs text-gray-500 space-y-1">
                <p>• This may take several minutes</p>
                <p>• AI is analyzing multiple data sources</p>
                <p>• Report will be generated automatically</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Report Structure Selection */}
      {reportStructureSection}

      {/* Query Input */}
      <Card className={`bg-gray-900/50 border-purple-400/30 transition-all duration-300 ${
        reports.isGenerating ? 'opacity-60 scale-95' : ''
      }`}>
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
              disabled={reports.isGenerating}
            />
          </div>

          {/* Enhanced Progress Indicator for Report Generation */}
          {reports.isGenerating && (
            <div className="space-y-3 p-4 bg-purple-900/20 border border-purple-400/30 rounded-lg">
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2 text-purple-400">
                  <Brain className="w-4 h-4 animate-pulse" />
                  <span>AI Generating Your Report</span>
                </div>
                <div className="flex items-center gap-2 text-gray-400">
                  <Clock className="w-4 h-4" />
                  <span>{formatTime(processingTime)}</span>
                </div>
              </div>
              
              <Progress value={reportProgress} className="h-2" />
              
              <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span>Analyzing request</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <span>Connecting to DB</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                  <span>Processing data</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                  <span>Generating insights</span>
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-2 flex-wrap">
            <Button
              onClick={handleGenerateReport}
              disabled={!userQuery.trim() || reports.isGenerating}
              className="bg-purple-600 hover:bg-purple-700 text-white flex items-center gap-2 min-w-[160px]"
            >
              {reports.isGenerating ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Generate Report
                </>
              )}
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

      {/* Processing Status */}
      {reports.isGenerating && (
        <div className="p-3 bg-green-900/20 border border-green-400/30 rounded-lg">
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <Zap className="w-4 h-4 animate-pulse" />
            <span>Your report is being generated by AI. This may take several minutes...</span>
          </div>
        </div>
      )}
    </div>
  );
}
 