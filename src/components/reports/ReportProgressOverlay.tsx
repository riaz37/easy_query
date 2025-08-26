import React from "react";
import { Progress } from "@/components/ui/progress";
import { Brain, Zap, Loader2 } from "lucide-react";

interface ReportProgressOverlayProps {
  reportProgress: number;
  processingSteps: string[];
  currentStep: number;
  processingTime: number;
  userQuery: string;
  formatTime: (seconds: number) => string;
}

export function ReportProgressOverlay({
  reportProgress,
  processingSteps,
  currentStep,
  processingTime,
  userQuery,
  formatTime,
}: ReportProgressOverlayProps) {
  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-gray-900/95 border border-purple-400/30 rounded-2xl p-8 max-w-md w-full mx-4">
        <div className="text-center space-y-6">
          {/* Animated Brain Icon */}
          <div className="relative">
            <div className="w-20 h-20 bg-gradient-to-br from-purple-500/20 to-blue-500/20 rounded-full flex items-center justify-center mx-auto border border-purple-400/30">
              <Brain className="w-10 h-10 text-purple-400 animate-pulse" />
            </div>
            <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
              <Zap className="w-3 h-3 text-white" />
            </div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm">
              <span className="text-purple-400">Generating Report</span>
              <span className="text-gray-400">{reportProgress}%</span>
            </div>
            <Progress value={reportProgress} className="h-2" />
          </div>

          {/* Current Step */}
          <div className="space-y-3">
            <div className="flex items-center justify-center gap-2 text-purple-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm font-medium">AI Report Generation</span>
            </div>
            <p className="text-gray-300 text-sm">
              {processingSteps[currentStep] || "Preparing..."}
            </p>
          </div>

          {/* Query Preview */}
          <div className="bg-gray-800/50 rounded-lg p-3 border border-gray-700/50">
            <p className="text-xs text-gray-400 mb-1">Processing:</p>
            <p className="text-white text-sm font-medium">
              {userQuery.length > 60 
                ? userQuery.substring(0, 60) + "..." 
                : userQuery
              }
            </p>
          </div>

          {/* Processing Stats */}
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div className="text-center">
              <div className="text-purple-400 font-medium">Time</div>
              <div className="text-gray-300">{formatTime(processingTime)}</div>
            </div>
            <div className="text-center">
              <div className="text-purple-400 font-medium">Step</div>
              <div className="text-gray-300">{currentStep + 1}/{processingSteps.length}</div>
            </div>
          </div>

          {/* Tips */}
          <div className="text-xs text-gray-500 space-y-1">
            <p>• This may take several minutes</p>
            <p>• AI is analyzing multiple data sources</p>
            <p>• Report will be generated automatically</p>
          </div>
        </div>
      </div>
    </div>
  );
} 