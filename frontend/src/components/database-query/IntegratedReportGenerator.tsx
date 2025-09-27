"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { useReports } from "@/lib/hooks/use-reports";
import { useUserContext } from "@/lib/hooks/use-user-context";
import { useTaskCreator } from "@/components/task-manager";
import { Button } from "@/components/ui/button";
import { Play, Clock, Brain } from "lucide-react";
import { CustomTypewriter, TYPEWRITER_TEXTS } from "@/components/ui/custom-typewriter";

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
  const textareaRef = useRef<HTMLDivElement>(null);
  const [isMultiLine, setIsMultiLine] = useState(false);
  const [hasUserInteracted, setHasUserInteracted] = useState(false);

  // Auto-resize contentEditable div based on content - EXACTLY like QueryForm
  useEffect(() => {
    const div = textareaRef.current;
    if (div) {
      // Only update content if it's different to avoid conflicts
      if (div.textContent !== userQuery) {
        div.textContent = userQuery;
      }
      
      // Reset height to auto to get the correct scrollHeight
      div.style.height = 'auto';
      
      // Calculate the new height
      const scrollHeight = div.scrollHeight;
      const minHeight = 24; // Single line height
      const maxHeight = 200; // Maximum height before scrollbar appears
      
      // Set height with constraints
      const newHeight = Math.min(Math.max(scrollHeight, minHeight), maxHeight);
      div.style.height = `${newHeight}px`;
      
      // Determine if div is multi-line (more than minimum height)
      setIsMultiLine(newHeight > minHeight);
    }
  }, [userQuery]);

  // Set initial height on mount
  useEffect(() => {
    const div = textareaRef.current;
    if (div) {
      div.style.height = '24px'; // Set initial single line height
    }
  }, []);

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
      textareaRef.current.style.height = '24px'; // Reset to single line height
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Stop typewriter effect when user starts typing
    setHasUserInteracted(true);
    
    // Allow Enter to create new lines, but prevent form submission
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleGenerateReport();
    }
  };

  const handleFocus = () => {
    // Stop typewriter effect when user focuses on the input
    setHasUserInteracted(true);
  };

  const handleInput = (e: React.FormEvent) => {
    // Stop typewriter effect when user types
    setHasUserInteracted(true);
    const text = e.currentTarget.textContent || '';
    setUserQuery(text);
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

  return (
    <div 
      className="relative -mt-16 px-0 z-10"
      style={{
        background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
        borderRadius: "25px",
        padding: "16px",
      }}
    >
      {/* ContentEditable div - EXACTLY like QueryForm */}
      <div className="text-white text-sm leading-relaxed relative">
        <div
          ref={textareaRef}
          contentEditable
          suppressContentEditableWarning
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          onFocus={handleFocus}
          className="w-full text-white resize-none overflow-y-auto"
          style={{
            outline: "none",
            border: "none",
            minHeight: "24px", // Single line height
            maxHeight: "200px",
            width: "100%",
            boxSizing: "border-box",
            overflow: "auto",
            background: "transparent",
            padding: "0",
            margin: "0",
            lineHeight: "24px", // Single line height
            whiteSpace: "pre-wrap", // Allow wrapping
            paddingRight: isMultiLine ? "0" : "140px", // Space for buttons only when single line
            direction: "ltr", // Ensure left-to-right text direction
            textAlign: "left", // Ensure left alignment
          }}
          data-placeholder="e.g., Show me the financial report of May, or Generate a comprehensive sales analysis for Q2"
          suppressContentEditableWarning
        />
        
        {/* Placeholder overlay */}
        {!userQuery && !hasUserInteracted && (
          <div 
            className="absolute top-0 left-0 pointer-events-none"
            style={{
              lineHeight: "24px",
              paddingRight: isMultiLine ? "0" : "140px",
            }}
          >
            <CustomTypewriter
              texts={TYPEWRITER_TEXTS.reports}
              className="text-sm"
              typeSpeed={8}
              deleteSpeed={5}
              deleteChunkSize={3}
              pauseTime={600}
              loop={true}
              startDelay={200}
            />
          </div>
        )}

        {/* Buttons - inline when single line, separate section when multiline */}
        {!isMultiLine && (
          <div className="absolute right-0 top-0 flex gap-2 items-center" style={{ height: "24px" }}>
            <Button
              onClick={handleGenerateReport}
              disabled={isReportGenerating || !userQuery.trim()}
              className="text-xs cursor-pointer"
              style={{
                background:
                  "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                border:
                  "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                color: "var(--p-main, rgba(19, 245, 132, 1))",
                borderRadius: "99px",
                height: "40px",
                minWidth: "60px",
              }}
            >
              {isReportGenerating ? "Generating..." : "Generate"}
            </Button>
          </div>
        )}
      </div>

      {/* Separate button section when multiline */}
      {isMultiLine && (
        <div className="flex items-center justify-end gap-2 mt-2">
          <Button
            onClick={handleGenerateReport}
            disabled={isReportGenerating || !userQuery.trim()}
            className="text-xs cursor-pointer"
            style={{
              background:
                "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
              border:
                "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
              color: "var(--p-main, rgba(19, 245, 132, 1))",
              borderRadius: "99px",
              height: "40px",
              minWidth: "60px",
            }}
          >
            {isReportGenerating ? "Generating..." : "Generate"}
          </Button>
        </div>
      )}

      {/* Error Display */}
      {reports.error && (
        <div className="p-4 bg-red-900/20 border border-red-500/30 rounded-lg mt-4">
          <div className="flex items-center gap-2 text-red-300">
            <Brain className="h-4 w-4" />
            <span className="text-sm">Error: {reports.error}</span>
          </div>
        </div>
      )}
    </div>
  );
}