import React from "react";
import { Button } from "@/components/ui/button";
import { Play, Clock, Square } from "lucide-react";
import { Spinner } from "@/components/ui/loading";

interface ReportActionButtonsProps {
  userQuery: string;
  isGenerating: boolean;
  onGenerateReport: () => void;
  onGenerateReportAndWait: () => void;
  onStopMonitoring: () => void;
  reports: any;
}

export function ReportActionButtons({
  userQuery,
  isGenerating,
  onGenerateReport,
  onGenerateReportAndWait,
  onStopMonitoring,
  reports,
}: ReportActionButtonsProps) {
  return (
    <div className="flex gap-2 flex-wrap">
      <Button
        onClick={onGenerateReport}
        disabled={!userQuery.trim() || isGenerating}
        className="bg-emerald-600 hover:bg-emerald-700 text-white flex items-center gap-2 min-w-[160px]"
        data-voice-action="report generation"
        data-voice-element="report generation"
      >
        {isGenerating ? (
          <>
            <Spinner size="sm" variant="primary" />
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
        onClick={onGenerateReportAndWait}
        disabled={!userQuery.trim() || isGenerating}
        variant="outline"
        className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10 flex items-center gap-2"
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
          <Spinner size="sm" variant="primary" />
          Refresh Status
        </Button>
      )}

      {isGenerating && (
        <Button
          onClick={onStopMonitoring}
          variant="destructive"
          className="flex items-center gap-2"
        >
          <Square className="h-4 w-4" />
          Stop
        </Button>
      )}
    </div>
  );
} 