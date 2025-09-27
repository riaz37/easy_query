import React from 'react';
import { Badge } from '@/components/ui/badge';
import { Clock, FileText, CheckCircle, AlertCircle, BarChart3 } from 'lucide-react';

interface FileQueryStatsProps {
  query: string;
  resultCount: number;
  executionTime?: number;
  uploadedFilesCount: number;
  completedFilesCount: number;
  failedFilesCount: number;
  className?: string;
}

export function FileQueryStats({
  query,
  resultCount,
  executionTime,
  uploadedFilesCount,
  completedFilesCount,
  failedFilesCount,
  className = ""
}: FileQueryStatsProps) {
  const successRate = uploadedFilesCount > 0 ? (completedFilesCount / uploadedFilesCount) * 100 : 0;

  // Format execution time nicely
  const formatExecutionTime = (time: number) => {
    if (time < 1000) return `${time}ms`;
    if (time < 60000) return `${(time / 1000).toFixed(1)}s`;
    return `${(time / 60000).toFixed(1)}m`;
  };

  return (
    <div className={`${className} bg-gray-900/50 border border-gray-600/30 rounded-lg p-6`}>
      {/* Header */}
      <div className="flex items-center gap-2 mb-6">
        <div className="w-10 h-10 rounded-lg bg-emerald-900/30 border border-emerald-400/30 flex items-center justify-center">
          <BarChart3 className="w-5 h-5 text-emerald-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">Query Statistics</h3>
          <p className="text-sm text-gray-400">Performance metrics and file status</p>
        </div>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {/* Query Length */}
        <div className="text-center p-4 rounded-lg bg-emerald-900/20 border border-emerald-400/30">
          <div className="text-2xl font-bold text-emerald-400 mb-1">
            {query.length}
          </div>
          <div className="text-xs text-emerald-300/70">Characters</div>
        </div>

        {/* Results Count */}
        <div className="text-center p-4 rounded-lg bg-green-900/20 border border-green-400/30">
          <div className="text-2xl font-bold text-green-400 mb-1">
            {resultCount}
          </div>
          <div className="text-xs text-green-300/70">Results</div>
        </div>

        {/* Execution Time */}
        <div className="text-center p-4 rounded-lg bg-emerald-900/20 border border-emerald-400/30">
          <div className="text-2xl font-bold text-emerald-400 mb-1">
            {executionTime ? formatExecutionTime(executionTime) : 'N/A'}
          </div>
          <div className="text-xs text-emerald-300/70">Execution Time</div>
        </div>

        {/* Success Rate */}
        <div className="text-center p-4 rounded-lg bg-orange-900/20 border border-orange-400/30">
          <div className="text-2xl font-bold text-orange-400 mb-1">
            {successRate.toFixed(1)}%
          </div>
          <div className="text-xs text-orange-300/70">Success Rate</div>
        </div>
      </div>

      {/* File Status Summary */}
      {uploadedFilesCount > 0 && (
        <div className="mb-6 p-4 rounded-lg bg-gray-800/30 border border-gray-600/30">
          <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
            <FileText className="w-4 h-4 text-emerald-400" />
            File Processing Status
          </h4>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-400"></div>
              <span className="text-sm text-gray-300">
                Total: <span className="text-emerald-400 font-medium">{uploadedFilesCount}</span>
              </span>
            </div>
            
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400"></div>
              <span className="text-sm text-gray-300">
                Completed: <span className="text-green-400 font-medium">{completedFilesCount}</span>
              </span>
            </div>
            
            {failedFilesCount > 0 && (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-red-400"></div>
                <span className="text-sm text-gray-300">
                  Failed: <span className="text-red-400 font-medium">{failedFilesCount}</span>
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Query Preview */}
      <div className="p-4 rounded-lg bg-gray-800/30 border border-gray-600/30">
        <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
          <Clock className="w-4 h-4 text-emerald-400" />
          Query Preview
        </h4>
        <div className="p-3 bg-gray-900/50 border border-gray-600/30 rounded-lg">
          <p className="text-sm text-gray-300 leading-relaxed">
            {query}
          </p>
        </div>
      </div>
    </div>
  );
} 