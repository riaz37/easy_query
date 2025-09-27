"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  CheckCircle, 
  XCircle, 
  Clock, 
  FileText,
  Upload,
  AlertCircle
} from "lucide-react";
import { Spinner, ProgressLoader } from "@/components/ui/loading";
import { 
  SmartFileSystemResponse, 
  BundleTaskStatusResponse,
  IndividualTask 
} from "@/types/api";

interface FileUploadProgressProps {
  uploadResponse: SmartFileSystemResponse | null;
  bundleStatus: BundleTaskStatusResponse | null;
  isLoading: boolean;
  error: string | null;
  onRefreshStatus: () => void;
  onReset: () => void;
}

export function FileUploadProgress({
  uploadResponse,
  bundleStatus,
  isLoading,
  error,
  onRefreshStatus,
  onReset
}: FileUploadProgressProps) {
  const [timeElapsed, setTimeElapsed] = useState(0);

  // Timer effect for elapsed time
  useEffect(() => {
    if (!uploadResponse) return;

    const startTime = Date.now();
    const timer = setInterval(() => {
      setTimeElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(timer);
  }, [uploadResponse]);

  if (!uploadResponse && !bundleStatus && !isLoading && !error) {
    return null;
  }

  const getStatusIcon = () => {
    if (error) return <XCircle className="h-6 w-6 text-red-500" />;
    if (bundleStatus?.status === "COMPLETED") return <CheckCircle className="h-6 w-6 text-green-500" />;
    if (bundleStatus?.status === "PROCESSING") return <Spinner size="md" variant="accent-green" />;
    return <Clock className="h-6 w-6 text-yellow-500" />;
  };

  const getStatusColor = () => {
    if (error) return "text-red-400";
    if (bundleStatus?.status === "COMPLETED") return "text-green-400";
    if (bundleStatus?.status === "PROCESSING") return "text-emerald-400";
    return "text-yellow-400";
  };

  const getStatusText = () => {
    if (error) return "Upload Failed";
    if (bundleStatus?.status === "COMPLETED") return "Processing Complete";
    if (bundleStatus?.status === "PROCESSING") return "Processing Files";
    return "Upload Successful";
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card className="border-emerald-200 bg-emerald-50 dark:bg-emerald-900/20 dark:border-emerald-800">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-emerald-900 dark:text-emerald-100">
          <Upload className="h-5 w-5" />
          File Processing Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Upload Response Summary */}
        {uploadResponse && (
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-gray-900 dark:text-white">Upload Summary</h4>
              <Badge variant="outline" className="text-xs">
                Bundle ID: {uploadResponse.bundle_id.slice(0, 8)}...
              </Badge>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600 dark:text-gray-400">Total Files:</span>
                <span className="ml-2 text-gray-900 dark:text-white">{uploadResponse.total_files}</span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Processing Mode:</span>
                <span className="ml-2 text-gray-900 dark:text-white capitalize">{uploadResponse.processing_mode}</span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Unstructured:</span>
                <span className="ml-2 text-gray-900 dark:text-white">{uploadResponse.unstructured_files}</span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Semi-structured:</span>
                <span className="ml-2 text-gray-900 dark:text-white">{uploadResponse.semi_structured_files}</span>
              </div>
            </div>
          </div>
        )}

        {/* Processing Status */}
        {bundleStatus && (
          <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-gray-900 dark:text-white">Processing Status</h4>
              <div className="flex items-center gap-2">
                {getStatusIcon()}
                <span className={`text-sm font-medium ${getStatusColor()}`}>
                  {getStatusText()}
                </span>
              </div>
            </div>
            
            {/* Progress Bar */}
            <div className="space-y-2 mb-4">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Overall Progress</span>
                <span className="text-gray-900 dark:text-white">{bundleStatus.progress_percentage}%</span>
              </div>
              <ProgressLoader 
                progress={bundleStatus.progress_percentage} 
                size="sm"
                variant="accent-green"
                showPercentage={false}
              />
            </div>

            {/* File Counts */}
            <div className="grid grid-cols-3 gap-4 text-sm mb-4">
              <div className="text-center">
                <div className="text-lg font-semibold text-emerald-600 dark:text-emerald-400">
                  {bundleStatus.total_files}
                </div>
                <div className="text-gray-600 dark:text-gray-400">Total</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold text-green-600 dark:text-green-400">
                  {bundleStatus.completed_files}
                </div>
                <div className="text-gray-600 dark:text-gray-400">Completed</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-semibold text-red-600 dark:text-red-400">
                  {bundleStatus.failed_files}
                </div>
                <div className="text-gray-600 dark:text-gray-400">Failed</div>
              </div>
            </div>

            {/* Individual Task Status */}
            {bundleStatus.individual_tasks && bundleStatus.individual_tasks.length > 0 && (
              <div className="space-y-2">
                <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">File Details</h5>
                {bundleStatus.individual_tasks.map((task: IndividualTask, index: number) => (
                  <div key={task.task_id} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-gray-500" />
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {task.filename}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {task.status === 'completed' && (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      )}
                      {task.status === 'failed' && (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      {task.status === 'processing' && (
                        <Spinner size="sm" variant="accent-green" />
                      )}
                      <Badge 
                        variant={task.status === 'completed' ? 'default' : task.status === 'failed' ? 'destructive' : 'secondary'}
                        className="text-xs"
                      >
                        {task.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Timestamps */}
            <div className="grid grid-cols-2 gap-4 text-xs text-gray-500 dark:text-gray-400 mt-4 pt-4 border-t">
              <div>
                <span>Created:</span>
                <span className="ml-2">{new Date(bundleStatus.created_at).toLocaleString()}</span>
              </div>
              <div>
                <span>Updated:</span>
                <span className="ml-2">{new Date(bundleStatus.last_updated).toLocaleString()}</span>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-800">
            <div className="flex items-center gap-2 text-red-800 dark:text-red-200">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">Error:</span>
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end gap-2">
          {bundleStatus && bundleStatus.status !== "COMPLETED" && (
            <Button
              variant="outline"
              size="sm"
              onClick={onRefreshStatus}
              disabled={isLoading}
            >
              {isLoading ? (
                <Spinner size="sm" variant="accent-green" />
              ) : (
                <Clock className="h-4 w-4" />
              )}
              Refresh Status
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={onReset}
          >
            Reset
          </Button>
        </div>

        {/* Timer */}
        {uploadResponse && (
          <div className="text-center text-sm text-gray-500 dark:text-gray-400">
            Time elapsed: {formatTime(timeElapsed)}
          </div>
        )}
      </CardContent>
    </Card>
  );
} 