import React from "react";
import Image from "next/image";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/loading/Spinner";

interface ReportQueryInputProps {
  userQuery: string;
  setUserQuery: (query: string) => void;
  isGenerating: boolean;
  reportProgress: number;
  processingTime: number;
  formatTime: (seconds: number) => string;
  onGenerate?: () => void;
}

export function ReportQueryInput({
  userQuery,
  setUserQuery,
  isGenerating,
  reportProgress,
  processingTime,
  formatTime,
  onGenerate,
}: ReportQueryInputProps) {
  const handleClear = () => {
    setUserQuery("");
  };

  return (
    <div
      className="p-6"
      style={{
        background:
          "linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)), linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%), linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)",
        border: "1.5px solid",
        borderImageSource:
          "linear-gradient(158.39deg, rgba(255, 255, 255, 0.06) 14.19%, rgba(255, 255, 255, 1.5e-05) 50.59%, rgba(255, 255, 255, 1.5e-05) 68.79%, rgba(255, 255, 255, 0.015) 105.18%)",
        borderRadius: "30px",
        backdropFilter: "blur(30px)",
      }}
    >
      <div className="flex items-start">
        <Image
          src="/file-query/filerobot.svg"
          alt="AI Report Robot"
          width={120}
          height={120}
          className="flex-shrink-0 -ml-6"
        />
        <div className="flex flex-col justify-start pt-5 -ml-8 z-10">
          <h3 className="text-white font-semibold text-2xl">
            AI Reports
          </h3>
        </div>
      </div>

      <div className="relative -mt-16 px-2 z-10">
        <Textarea
          value={userQuery}
          onChange={(e) => setUserQuery(e.target.value)}
          placeholder="e.g., Show me the financial report of May, or Generate a comprehensive sales analysis for Q2"
          className="w-full h-48 p-4 pr-32 bg-slate-800/50 text-white placeholder-slate-400 focus:outline-none resize-none border-0"
          style={{
            background:
              "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
            borderRadius: "16px",
            outline: "none",
            border: "none",
          }}
          disabled={isGenerating}
        />

        {/* Action Buttons */}
        <div className="absolute bottom-3 left-7 right-7 flex justify-start">
          <div className="flex gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={handleClear}
              className="text-xs"
              style={{
                background:
                  "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                border:
                  "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                color: "white",
                borderRadius: "99px",
                height: "48px",
                minWidth: "64px",
              }}
            >
              Clear
            </Button>
            <Button
              type="button"
              onClick={onGenerate}
              disabled={!userQuery.trim() || isGenerating}
              className="text-xs"
              style={{
                background:
                  "var(--components-button-Fill, rgba(255, 255, 255, 0.12))",
                border:
                  "1px solid var(--primary-16, rgba(19, 245, 132, 0.16))",
                color: "var(--p-main, rgba(19, 245, 132, 1))",
                borderRadius: "99px",
                height: "48px",
                minWidth: "64px",
              }}
            >
              {isGenerating ? "Generating..." : "Generate"}
            </Button>
          </div>
        </div>
      </div>

      {/* Enhanced Progress Indicator for Report Generation */}
      {isGenerating && (
        <div className="space-y-3 p-4 bg-emerald-900/20 border border-emerald-400/30 rounded-lg mt-4">
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
  );
} 