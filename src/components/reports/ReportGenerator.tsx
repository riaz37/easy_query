"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useReports } from "@/lib/hooks/use-reports";
import { useReportStructure } from "@/lib/hooks/use-report-structure";
import { useUserContext } from "@/lib/hooks/use-user-context";
import { ReportStructureSelector } from "./ReportStructureSelector";
import { ReportQueryInput } from "./ReportQueryInput";
import { ReportProgressOverlay } from "./ReportProgressOverlay";
import { ReportProgressIndicator } from "./ReportProgressIndicator";
import { ReportActionButtons } from "./ReportActionButtons";
import { ReportProgressDisplay } from "./ReportProgressDisplay";
import { ReportTaskStatus } from "./ReportTaskStatus";
import { ReportResultsPreview } from "./ReportResultsPreview";
import { ReportProcessingStatus } from "./ReportProcessingStatus";

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
  const reportStructure = useReportStructure();

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

  // Load report structure on mount - only when userId changes
  useEffect(() => {
    if (userId && !reportStructure.structure) {
      reportStructure.loadStructure(userId);
    }
  }, [userId]); // Removed reportStructure from deps to prevent infinite loops

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
  }, [user?.user_id, userQuery, onReportStart, onReportComplete, reports, processingSteps, currentStep]);

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
      {/* Loading Overlay for Report Generation */}
      {reports.isGenerating && (
        <ReportProgressOverlay
          reportProgress={reportProgress}
          processingSteps={processingSteps}
          currentStep={currentStep}
          processingTime={processingTime}
          userQuery={userQuery}
          formatTime={formatTime}
        />
      )}

      {/* Report Structure Selection */}
      <ReportStructureSelector
        reportStructure={reportStructure}
        selectedStructure={selectedStructure}
        setSelectedStructure={handleStructureChange}
        isGenerating={reports.isGenerating}
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