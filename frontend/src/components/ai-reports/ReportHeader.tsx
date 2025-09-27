"use client";

import React from "react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { CheckCircle } from "lucide-react";

interface ReportHeaderProps {
  reportResults: any;
}

export function ReportHeader({ reportResults }: ReportHeaderProps) {
  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-emerald-400" />
            Report Overview
          </div>
        </div>
        <div className="mt-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">
              {reportResults.total_queries}
            </div>
            <div className="text-sm text-gray-400">Total Queries</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {reportResults.successful_queries}
            </div>
            <div className="text-sm text-gray-400">Successful</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {reportResults.failed_queries}
            </div>
            <div className="text-sm text-gray-400">Failed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-emerald-400">
              {reportResults.database_id}
            </div>
            <div className="text-sm text-gray-400">Database ID</div>
          </div>
        </div>

        <Separator className="my-4 bg-gray-700" />

        <div className="text-center">
          <div className="text-sm text-gray-400 mb-2">Success Rate:</div>
          <Badge variant="outline" className="border-green-400/30 text-green-400">
            {((reportResults.successful_queries / reportResults.total_queries) * 100).toFixed(1)}%
          </Badge>
        </div>
        </div>
      </div>
    </div>
  );
} 