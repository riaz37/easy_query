import React from "react";
import { Brain, Clock } from "lucide-react";

interface ReportProgressIndicatorProps {
  reportProgress: number;
  processingTime: number;
  formatTime: (seconds: number) => string;
}

export function ReportProgressIndicator({
  reportProgress,
  processingTime,
  formatTime,
}: ReportProgressIndicatorProps) {
  return (
    <div className="space-y-3 p-4 bg-emerald-900/20 border border-emerald-400/30 rounded-lg">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-emerald-400">
          <Brain className="w-4 h-4 animate-pulse" />
          <span>AI Generating Your Report</span>
        </div>
        <div className="flex items-center gap-2 text-gray-400">
          <Clock className="w-4 h-4" />
          <span>{formatTime(processingTime)}</span>
        </div>
      </div>
      
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div 
          className="bg-emerald-500 h-2 rounded-full transition-all duration-300"
          style={{ width: `${reportProgress}%` }}
        ></div>
      </div>
      
      <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span>Analyzing request</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
          <span>Connecting to DB</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
          <span>Processing data</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
          <span>Generating insights</span>
        </div>
      </div>
    </div>
  );
} 