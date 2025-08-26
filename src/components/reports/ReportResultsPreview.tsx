import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, CheckCircle } from "lucide-react";

interface ReportResultsPreviewProps {
  reportResults: any;
}

export function ReportResultsPreview({
  reportResults,
}: ReportResultsPreviewProps) {
  return (
    <Card className="bg-gray-900/50 border-green-400/30">
      <CardHeader>
        <CardTitle className="text-green-400 flex items-center gap-2">
          <CheckCircle className="w-5 h-5" />
          Report Generated Successfully!
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
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
              <div className="text-2xl font-bold text-red-400">
                {reportResults.failed_queries}
              </div>
              <div className="text-sm text-gray-400">Failed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">
                {reportResults.database_id}
              </div>
              <div className="text-sm text-gray-400">Database ID</div>
            </div>
          </div>

          <div className="text-center">
            <Button
              onClick={() => {
                // Store results and redirect to detailed view
                sessionStorage.setItem(
                  "reportResults",
                  JSON.stringify(reportResults)
                );
                window.open("/ai-results", "_blank");
              }}
              className="bg-green-600 hover:bg-green-700 text-white"
            >
              <FileText className="w-4 h-4 mr-2" />
              View Full Report
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 