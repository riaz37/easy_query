import React from "react";
import { Button } from "@/components/ui/button";
import { Play, Clock, Loader2, Square } from "lucide-react";

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
        className="bg-purple-600 hover:bg-purple-700 text-white flex items-center gap-2 min-w-[160px]"
      >
        {isGenerating ? (
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
        onClick={onGenerateReportAndWait}
        disabled={!userQuery.trim() || isGenerating}
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