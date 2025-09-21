"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Clock } from "lucide-react";

interface ProcessingDetailsProps {
  reportResults: any;
}

export function ProcessingDetails({ reportResults }: ProcessingDetailsProps) {
  if (!reportResults.summary) return null;

  return (
    <div className="card-enhanced mt-6">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <Clock className="w-5 h-5 text-emerald-400" />
            Processing Details
          </div>
        </div>
        <div className="mt-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">
              {reportResults.summary.total_sections}
            </div>
            <div className="text-sm text-gray-400">Total Sections</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {reportResults.summary.success_rate}%
            </div>
            <div className="text-sm text-gray-400">Success Rate</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {reportResults.summary.total_processing_time?.toFixed(2) || "N/A"}s
            </div>
            <div className="text-sm text-gray-400">Total Time</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {reportResults.summary.average_processing_time?.toFixed(2) || "N/A"}s
            </div>
            <div className="text-sm text-gray-400">Avg per Query</div>
          </div>
        </div>

        <Separator className="my-4 bg-gray-700" />

        <div className="text-center">
          <div className="text-sm text-gray-400 mb-2">Processing Method:</div>
          <Badge variant="outline" className="border-green-400/30 text-green-400">
            {reportResults.summary.processing_method}
          </Badge>
        </div>
        </div>
      </div>
    </div>
  );
} 