"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { useReports } from "@/lib/hooks/use-reports";
import { useUserContext } from "@/lib/hooks/use-user-context";
import { useTaskCreator } from "@/components/task-manager";
import { Button } from "@/components/ui/button";
import { Play, Clock, Brain } from "lucide-react";

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
  const reports = useReports();
  const { createReportTask, executeTask } = useTaskCreator();
  
  const [userQuery, setUserQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [isMultiLine, setIsMultiLine] = useState(false);

  // Auto-resize textarea based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      const minHeight = 64;
      const maxHeight = 200;
      const newHeight = Math.min(Math.max(scrollHeight, minHeight), maxHeight);
      textarea.style.height = `${newHeight}px`;
      setIsMultiLine(newHeight > minHeight);
    }
  }, [userQuery]);

  // Handle report completion
  useEffect(() => {
    if (reports.reportResults && onReportComplete) {
      onReportComplete(reports.reportResults);
    }
  }, [reports.reportResults, onReportComplete]);

  // Handle report start
  useEffect(() => {
    if (reports.isGenerating && onReportStart) {
      onReportStart();
    }
  }, [reports.isGenerating, onReportStart]);

  const handleClear = () => {
    setUserQuery("");
    setIsMultiLine(false);
    if (textareaRef.current) {
      textareaRef.current.style.height = '64px';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerateReport();
    }
  };

  const handleGenerateReport = useCallback(async () => {
    if (!user?.user_id || !userQuery.trim()) return;

    // Create a background task
    const taskId = createReportTask(
      `AI Report: ${userQuery.substring(0, 50)}${userQuery.length > 50 ? '...' : ''}`,
      `Generating AI report for query: "${userQuery}"`,
      {
        user_id: user.user_id,
        user_query: userQuery,
      }
    );

    // Execute the task in background
    executeTask(
      taskId,
      async () => {
        if (onReportStart) {
          onReportStart();
        }

        const reportTaskId = await reports.generateReport({
          user_id: user.user_id,
          user_query: userQuery,
        });

        return new Promise((resolve, reject) => {
          reports.startMonitoring(reportTaskId, {
            onProgress: (status) => {
              console.log("Report progress:", status);
            },
            onComplete: (results) => {
              console.log("Report completed:", results);
              if (onReportComplete) {
                onReportComplete(results);
              }
              resolve(results);
            },
            onError: (error) => {
              console.error("Report failed:", error);
              reject(error);
            },
            pollInterval: 2000,
          });
        });
      }
    ).catch((error) => {
      console.error('Failed to start report generation:', error);
    });
  }, [user?.user_id, userQuery, onReportStart, onReportComplete, reports, createReportTask, executeTask]);

  const handleGenerateReportAndWait = useCallback(async () => {
    if (!user?.user_id || !userQuery.trim()) return;

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

  return (
    <div className="space-y-4">
      {/* Query Input - With inline buttons like QueryForm */}
      <div className="relative">
        <textarea
          ref={textareaRef}
          value={userQuery}
          onChange={(e) => setUserQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="e.g., Show me the financial report of May, or Generate a comprehensive sales analysis for Q2"
          className="w-full min-h-16 px-4 pr-24 py-4 bg-slate-800/50 text-white placeholder-slate-400 focus:outline-none border-0 resize-none overflow-y-auto"
          style={{
            background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
            borderRadius: "16px",
            outline: "none",
            border: "none",
            minHeight: "64px",
            maxHeight: "200px",
          }}
          disabled={isReportGenerating}
          rows={1}
        />
        
        <div className={`absolute right-2 flex gap-2 items-center ${isMultiLine ? 'bottom-3' : 'top-3'}`}>
          <Button
            onClick={handleGenerateReport}
            disabled={isReportGenerating || !userQuery.trim()}
            className="text-xs cursor-pointer"
            style={{
              background: "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
              border: "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
              color: "var(--p-main, rgba(19, 245, 132, 1))",
              borderRadius: "99px",
              height: "40px",
              minWidth: "60px",
            }}
          >
            {isReportGenerating ? "Generating..." : "Generate"}
          </Button>
        </div>
      </div>

      {/* Error Display */}
      {reports.error && (
        <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
          <div className="flex items-center gap-2 text-red-300">
            <Brain className="h-4 w-4" />
            <span className="text-sm">Error: {reports.error}</span>
          </div>
        </div>
      )}
    </div>
  );
}
 