import React from "react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { BarChart3 } from "lucide-react";

interface ReportTaskStatusProps {
  currentTask: any;
  reports: any;
}

export function ReportTaskStatus({
  currentTask,
  reports,
}: ReportTaskStatusProps) {
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

  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" />
            Task Status
          </div>
        </div>
        <div className="mt-4">
        {/* Real-time Status Bar */}
        <div className="mb-4 p-3 bg-gray-800/30 rounded border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-300">Current Status:</span>
            <Badge className={getStatusColor(currentTask.status)}>
              {currentTask.status.toUpperCase()}
            </Badge>
          </div>
          <div className="text-sm text-white">
            {currentTask.current_step || 'Initializing...'}
          </div>
          {currentTask.progress && (
            <div className="text-xs text-gray-400 mt-1">
              {currentTask.progress}
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">
              {currentTask.total_queries}
            </div>
            <div className="text-sm text-gray-400">Total Queries</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {currentTask.successful_queries}
            </div>
            <div className="text-sm text-gray-400">Successful</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">
              {currentTask.failed_queries}
            </div>
            <div className="text-sm text-gray-400">Failed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {currentTask.progress_percentage}%
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
              {currentTask.processing_time_seconds 
                ? `${currentTask.processing_time_seconds}s`
                : 'Calculating...'
              }
            </span>
          </div>
          
          {currentTask.started_at && (
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Started:</span>
              <span className="text-white">
                {new Date(currentTask.started_at).toLocaleTimeString()}
              </span>
            </div>
          )}
          
          {currentTask.completed_at && (
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Completed:</span>
              <span className="text-white">
                {new Date(currentTask.completed_at).toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>
        </div>
      </div>
    </div>
  );
} 