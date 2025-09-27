"use client";

import React, { useState, useCallback, useMemo } from "react";
import { useUserContext } from "@/lib/hooks/use-user-context";
import { useUserConfiguration } from "@/components/user-configuration/hooks/useUserConfiguration";
import { useTaskCreator } from "@/components/task-manager";
import { ServiceRegistry } from "@/lib/api";
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

export function ReportGeneratorBackground({
  userId,
  configId,
  onReportComplete,
  onReportStart,
  isReportGenerating = false,
}: ReportGeneratorProps) {
  const { user } = useUserContext();
  const [userQuery, setUserQuery] = useState("");
  const [selectedStructure, setSelectedStructure] =
    useState<string>("financial_report");

  // Hooks
  const { reportStructure, reportStructureLoading, reportStructureError } =
    useUserConfiguration();
  const { createReportTask, executeTask } = useTaskCreator();

  // Parse report structure from string
  const parsedReportStructure = useMemo(() => {
    if (!reportStructure) return null;
    try {
      return JSON.parse(reportStructure);
    } catch (error) {
      console.error("Failed to parse report structure:", error);
      return null;
    }
  }, [reportStructure]);

  // Memoize handlers to prevent unnecessary re-renders
  const handleGenerateReport = useCallback(async () => {
    if (!user?.user_id || !userQuery.trim()) return;

    console.log("Starting report generation for user:", user.user_id);
    console.log("User query:", userQuery);

    // Create a background task
    const taskId = createReportTask(
      `AI Report: ${userQuery.substring(0, 50)}${
        userQuery.length > 50 ? "..." : ""
      }`,
      `Generating AI report for query: "${userQuery}"`,
      {
        user_id: user.user_id,
        user_query: userQuery,
        selected_structure: selectedStructure,
      }
    );

    // Execute the task in background
    executeTask(taskId, async () => {
      // Notify parent component that report generation has started
      if (onReportStart) {
        onReportStart();
      }

      // Generate report using ServiceRegistry directly

      let reportResponse;
      try {
        reportResponse = await ServiceRegistry.reports.generateReport({
          user_id: user.user_id,
          user_query: userQuery,
        });
      } catch (error) {
        console.error(
          "ServiceRegistry.reports.generateReport threw an error:",
          error
        );
        throw new Error(
          `Service error: ${
            error instanceof Error ? error.message : "Unknown error"
          }`
        );
      }


      // Check if the response indicates success
      if (!reportResponse || !reportResponse.task_id) {
        console.error(
          "Report generation failed: Invalid response structure",
          reportResponse
        );
        throw new Error("Invalid response from report generation service");
      }

      // Check if the status indicates success
      if (reportResponse.status !== "accepted") {
        console.error(
          "Report generation failed:",
          reportResponse.message || "Unknown error"
        );
        throw new Error(
          reportResponse.message || "Report generation was not accepted"
        );
      }

        const reportTaskId = reportResponse.task_id;

      // Start monitoring the task with multi-task support
      return new Promise(async (resolve, reject) => {
        // Use the multi-task monitor instead of the single-task reports.startMonitoring
        const { TaskUtils } = await import("@/store/task-store");

        TaskUtils.startReportMonitoring(taskId, reportTaskId, {
          onProgress: (status) => {
            // Progress updates are handled by the task store
          },
          onComplete: (results) => {
            // Store results for the results page
            if (onReportComplete) {
              onReportComplete(results);
            }
            resolve(results);
          },
          onError: (error) => {
            console.error(`[Task ${taskId}] Report failed:`, error);
            reject(error);
          },
          pollInterval: 2000, // Poll every 2 seconds
        });
      });
    }).catch((error) => {
      console.error("Failed to start report generation:", error);
    });
  }, [
    user?.user_id,
    userQuery,
    onReportStart,
    onReportComplete,
    selectedStructure,
    createReportTask,
    executeTask,
  ]);

  const handleGenerateReportAndWait = useCallback(async () => {
    // For now, same as regular generation since we're using background tasks
    await handleGenerateReport();
  }, [handleGenerateReport]);

  const handleStopMonitoring = useCallback(() => {
    // Stop monitoring is now handled by the task system
    console.log("Stop monitoring requested - handled by task system");
  }, []);

  // Memoize the format time utility
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
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
    <ReportQueryInput
      userQuery={userQuery}
      setUserQuery={handleUserQueryChange}
      isGenerating={false} // No longer blocking UI
      reportProgress={0}
      processingTime={0}
      formatTime={formatTime}
      onGenerate={handleGenerateReport}
    />
  );
}
