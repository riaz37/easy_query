"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, Eye, FileText } from "lucide-react";
import { Spinner } from "@/components/ui/loading";

interface ExportControlsProps {
  reportResults: any;
  pdfGenerating: boolean;
  onDownloadPDF: () => void;
  onPreviewPDF: () => void;
  onDownloadText: () => void;
}

export function ExportControls({
  reportResults,
  pdfGenerating,
  onDownloadPDF,
  onPreviewPDF,
  onDownloadText,
}: ExportControlsProps) {
  return (
    <Card className="bg-gray-900/50 border-green-400/30">
      <CardHeader>
        <CardTitle className="text-green-400 flex items-center gap-2">
          <Download className="w-5 h-5" />
          Export Report
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">
                Export Report
              </h3>
              <p className="text-gray-400 text-sm">
                Download your AI-generated report in multiple formats
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button
              onClick={onDownloadPDF}
              disabled={pdfGenerating}
              className="bg-emerald-600 hover:bg-emerald-700 text-white"
            >
              <Download className="w-4 h-4 mr-2" />
              {pdfGenerating ? "Generating PDF..." : "Download PDF"}
            </Button>

            <Button
              onClick={onPreviewPDF}
              disabled={pdfGenerating}
              variant="outline"
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
            >
              <Eye className="w-4 h-4 mr-2" />
              Preview PDF
            </Button>

            <Button
              onClick={onDownloadText}
              variant="outline"
              className="border-green-400/30 text-green-400 hover:bg-green-400/10"
            >
              <FileText className="w-4 h-4 mr-2" />
              Download Text
            </Button>
          </div>

          {pdfGenerating && (
            <div className="bg-green-900/20 p-4 rounded-lg border border-green-400/30">
              <div className="flex items-center gap-3">
                <Spinner size="sm" variant="success" />
                <span className="text-green-400 text-sm">
                  Generating professional PDF report...
                </span>
              </div>
              <p className="text-gray-400 text-xs mt-2">
                This may take a few moments depending on the report size
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 