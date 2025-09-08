import React from "react";
import { Textarea } from "@/components/ui/textarea";
import { Sparkles } from "lucide-react";

interface ReportQueryInputProps {
  userQuery: string;
  setUserQuery: (query: string) => void;
  isGenerating: boolean;
  reportProgress: number;
  processingTime: number;
  formatTime: (seconds: number) => string;
}

export function ReportQueryInput({
  userQuery,
  setUserQuery,
  isGenerating,
  reportProgress,
  processingTime,
  formatTime,
}: ReportQueryInputProps) {
  return (
    <div className={`card-enhanced transition-all duration-300 ${
      isGenerating ? 'opacity-60 scale-95' : ''
    }`}>
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            AI Report Generation
          </div>
        </div>
        <div className="mt-4 space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-white">
            What would you like to know?
          </label>
          <Textarea
            value={userQuery}
            onChange={(e) => setUserQuery(e.target.value)}
            placeholder="e.g., Show me the financial report of May, or Generate a comprehensive sales analysis for Q2"
            className="min-h-[120px] bg-gray-800/50 border-emerald-400/30 text-white placeholder:text-gray-400 resize-none"
            disabled={isGenerating}
          />
        </div>

        {/* Enhanced Progress Indicator for Report Generation */}
        {isGenerating && (
          <div className="space-y-3 p-4 bg-emerald-900/20 border border-emerald-400/30 rounded-lg">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2 text-emerald-400">
                <span>AI Generating Your Report</span>
              </div>
              <div className="flex items-center gap-2 text-gray-400">
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
        )}
        </div>
      </div>
    </div>
  );
} 