import React from "react";
import { Progress } from "@/components/ui/progress";
import { Loader2, Clock } from "lucide-react";

interface ReportProgressDisplayProps {
  reports: any;
  processingSteps: string[];
}

export function ReportProgressDisplay({
  reports,
  processingSteps,
}: ReportProgressDisplayProps) {
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

  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <Loader2 className="w-5 h-5 animate-spin text-emerald-400" />
            Report Generation Progress
          </div>
        </div>
        <div className="mt-4">
        <div className="space-y-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-white">{getProgressMessage()}</span>
            <span className="text-emerald-400 font-medium">
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
        </div>
      </div>
    </div>
  );
} 