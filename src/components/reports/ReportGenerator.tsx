"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useReports } from "@/lib/hooks/use-reports";
import { useUserContext } from "@/lib/hooks/use-user-context";
import { useUserConfiguration } from "@/components/user-configuration/hooks/useUserConfiguration";
import { useTaskCreator } from "@/components/task-manager";
import { useTaskStore } from "@/store/task-store";
import { ReportStructureSelector } from "./ReportStructureSelector";
import { ReportQueryInput } from "./ReportQueryInput";
import { ReportActionButtons } from "./ReportActionButtons";
import { ReportResultsPreview } from "./ReportResultsPreview";

interface ReportGeneratorProps {
  userId?: string;
  configId?: number;
  onReportComplete?: (results: any) => void;
  onReportStart?: () => void;
  isReportGenerating?: boolean;
}

export function ReportGenerator({
  userId,
  configId,
  onReportComplete,
  onReportStart,
  isReportGenerating = false,
}: ReportGeneratorProps) {
  const { user } = useUserContext();
  const [userQuery, setUserQuery] = useState("");
  const [selectedStructure, setSelectedStructure] = useState<string>("financial_report");
  const [reportProgress, setReportProgress] = useState(0);
  const [processingSteps, setProcessingSteps] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [processingTime, setProcessingTime] = useState(0);

  // Hooks
  const reports = useReports();
  const { reportStructure, reportStructureLoading, reportStructureError } = useUserConfiguration();
  const { createReportTask, executeTask } = useTaskCreator();

  // Use ref to track current progress without causing re-renders
  const progressRef = useRef(reportProgress);
  useEffect(() => {
    progressRef.current = reportProgress;
  }, [reportProgress]);

  // Memoize the processing steps to prevent recreation
  const defaultProcessingSteps = useMemo(() => [
    "Analyzing your report request...",
    "Connecting to database...",
    "Generating SQL queries...",
    "Executing database queries...",
    "Processing business rules...",
    "Analyzing data patterns...",
    "Generating insights...",
    "Compiling final report..."
  ], []);

  // Parse report structure from string
  const parsedReportStructure = useMemo(() => {
    if (!reportStructure) return null;
    try {
      return JSON.parse(reportStructure);
    } catch (error) {
      console.error('Failed to parse report structure:', error);
      return null;
    }
  }, [reportStructure]);

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
      setProcessingSteps(defaultProcessingSteps);
      
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
          const newStep = Math.floor((progressRef.current / 100) * defaultProcessingSteps.length);
          return Math.min(newStep, defaultProcessingSteps.length - 1);
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
  }, [reports.isGenerating, defaultProcessingSteps]); // Fixed dependencies

  // Memoize handlers to prevent unnecessary re-renders
  const handleGenerateReport = useCallback(async () => {
    if (!user?.user_id || !userQuery.trim()) return;

    console.log('Starting report generation for user:', user.user_id);
    console.log('User query:', userQuery);

    // Create a background task
    const taskId = createReportTask(
      `AI Report: ${userQuery.substring(0, 50)}${userQuery.length > 50 ? '...' : ''}`,
      `Generating AI report for query: "${userQuery}"`,
      {
        user_id: user.user_id,
        user_query: userQuery,
        selected_structure: selectedStructure,
      }
    );

    // Execute the task in background
    executeTask(
      taskId,
      async () => {
        // Notify parent component that report generation has started
        if (onReportStart) {
          onReportStart();
        }

        const reportTaskId = await reports.generateReport({
          user_id: user.user_id,
          user_query: userQuery,
        });

        console.log('Report generation started, backend task ID:', reportTaskId);

        // Update the task with the backend task ID
        const { updateTask, getTaskById } = useTaskStore.getState();
        const currentTask = getTaskById(taskId);
        console.log('Current task before update:', currentTask);
        console.log('Backend task ID to store:', reportTaskId);
        
        updateTask(taskId, {
          metadata: {
            ...(currentTask?.metadata || {}),
            backend_task_id: reportTaskId
          }
        });
        
        // Verify the update
        const updatedTask = getTaskById(taskId);
        console.log('Updated task metadata:', updatedTask?.metadata);

        // Start monitoring the task with proper callbacks
        return new Promise((resolve, reject) => {
          reports.startMonitoring(reportTaskId, {
            onProgress: (status) => {
              console.log("Report progress:", status);
              // Update task progress
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
              resolve(results);
            },
            onError: (error) => {
              console.error("Report failed:", error);
              reject(error);
            },
            pollInterval: 2000, // Poll every 2 seconds
          });
        });
      },
      (progress) => {
        // Progress callback for task system
        setReportProgress(progress);
      }
    ).catch((error) => {
      console.error('Failed to start report generation:', error);
    });
  }, [user?.user_id, userQuery, onReportStart, onReportComplete, reports, processingSteps, currentStep, selectedStructure, createReportTask, executeTask]);

  const handleGenerateReportAndWait = useCallback(async () => {
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
  }, [user?.user_id, userQuery, onReportStart, reports]);

  const handleStopMonitoring = useCallback(() => {
    reports.stopMonitoring();
  }, [reports]);

  // Memoize the format time utility
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  // Memoize the setUserQuery handler
  const handleUserQueryChange = useCallback((query: string) => {
    setUserQuery(query);
  }, []);

  // Memoize the setSelectedStructure handler
  const handleStructureChange = useCallback((structure: string) => {
    setSelectedStructure(structure);
  }, []);

  return (
    <div className="space-y-6">

      {/* Report Structure Selection */}
      <ReportStructureSelector
        reportStructure={parsedReportStructure}
        selectedStructure={selectedStructure}
        setSelectedStructure={handleStructureChange}
        isGenerating={reports.isGenerating}
        loading={reportStructureLoading}
        error={reportStructureError}
      />

      {/* Query Input */}
      <ReportQueryInput
        userQuery={userQuery}
        setUserQuery={handleUserQueryChange}
        isGenerating={reports.isGenerating}
        reportProgress={reportProgress}
        processingTime={processingTime}
        formatTime={formatTime}
      />

      {/* Enhanced Progress Indicator for Report Generation */}
      {reports.isGenerating && (
        <ReportProgressIndicator
          reportProgress={reportProgress}
          processingTime={processingTime}
          formatTime={formatTime}
        />
      )}

      {/* Action Buttons */}
      <ReportActionButtons
        userQuery={userQuery}
        isGenerating={reports.isGenerating}
        onGenerateReport={handleGenerateReport}
        onGenerateReportAndWait={handleGenerateReportAndWait}
        onStopMonitoring={handleStopMonitoring}
        reports={reports}
      />

      {/* Progress Display */}
      {reports.isGenerating && (
        <ReportProgressDisplay
          reports={reports}
          processingSteps={processingSteps}
        />
      )}

      {/* Current Task Status */}
      {reports.currentTask && (
        <ReportTaskStatus
          currentTask={reports.currentTask}
          reports={reports}
        />
      )}

      {/* Error Display */}
      {reports.error && (
        <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
          <div className="flex items-center gap-2 text-red-300">
            <span className="text-sm">Error: {reports.error}</span>
          </div>
        </div>
      )}

      {/* Report Results Preview */}
      {reports.reportResults && (
        <ReportResultsPreview
          reportResults={reports.reportResults}
        />
      )}

      {/* Processing Status */}
      {reports.isGenerating && (
        <ReportProcessingStatus />
      )}
    </div>
  );
} 