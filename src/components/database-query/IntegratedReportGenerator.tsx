"use client";

import React from "react";
import { useReports } from "@/lib/hooks/use-reports";
import { useUserContext } from "@/lib/hooks/use-user-context";
import { 
  ReportGenerator,
  ReportStructureSelector,
  ReportQueryInput,
  ReportProgressOverlay,
  ReportProgressIndicator,
  ReportActionButtons,
  ReportProgressDisplay,
  ReportTaskStatus,
  ReportResultsPreview,
  ReportProcessingStatus
} from "@/components/reports";

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

  // Handle report completion
  React.useEffect(() => {
    if (reports.reportResults && onReportComplete) {
      onReportComplete(reports.reportResults);
    }
  }, [reports.reportResults, onReportComplete]);

  // Handle report start
  React.useEffect(() => {
    if (reports.isGenerating && onReportStart) {
      onReportStart();
    }
  }, [reports.isGenerating, onReportStart]);

  return (
    <div className="space-y-6">
      {/* Use the existing ReportGenerator component */}
      <ReportGenerator
        userId={userId}
        onReportComplete={onReportComplete}
        onReportStart={onReportStart}
        isReportGenerating={isReportGenerating}
      />
    </div>
  );
}
 