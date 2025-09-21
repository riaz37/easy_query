import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, CheckCircle, Download, Eye } from "lucide-react";
import { Spinner } from "@/components/ui/loading";
import { generateAndDownloadPDF, generatePDFBlob } from "@/lib/utils/smart-pdf-generator";

interface ReportResultsPreviewProps {
  reportResults: any;
}

export function ReportResultsPreview({
  reportResults,
}: ReportResultsPreviewProps) {
  const [pdfGenerating, setPdfGenerating] = useState(false);

  const handleDownloadPDF = async () => {
    setPdfGenerating(true);
    try {
      await generateAndDownloadPDF(
        reportResults, 
        `AI_Report_${reportResults.database_id}_${new Date().toISOString().split('T')[0]}.pdf`
      );
    } catch (error) {
      console.error('Failed to generate PDF:', error);
      alert('Failed to generate PDF. Please try again.');
    } finally {
      setPdfGenerating(false);
    }
  };

  const handlePreviewPDF = async () => {
    setPdfGenerating(true);
    try {
      const blob = await generatePDFBlob(reportResults);
      const url = URL.createObjectURL(blob);
      window.open(url, '_blank');
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to preview PDF:', error);
      alert('Failed to preview PDF. Please try again.');
    } finally {
      setPdfGenerating(false);
    }
  };

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

          <div className="text-center space-y-3">
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Button
                onClick={() => {
                  // Store results and redirect to detailed view
                  sessionStorage.setItem(
                    "reportResults",
                    JSON.stringify(reportResults)
                  );
                  window.open("/ai-reports", "_blank");
                }}
                className="bg-green-600 hover:bg-green-700 text-white"
                data-voice-action="view report"
                data-voice-element="view report"
              >
                <FileText className="w-4 h-4 mr-2" />
                View Full Report
              </Button>
              
              <Button
                onClick={handlePreviewPDF}
                disabled={pdfGenerating}
                variant="outline"
                className="border-blue-400/30 text-blue-400 hover:bg-blue-400/10"
              >
                {pdfGenerating ? (
                  <Spinner size="sm" variant="primary" className="mr-2" />
                ) : (
                  <Eye className="w-4 h-4 mr-2" />
                )}
                Preview PDF
              </Button>
              
              <Button
                onClick={handleDownloadPDF}
                disabled={pdfGenerating}
                className="bg-emerald-600 hover:bg-emerald-700 text-white"
              >
                {pdfGenerating ? (
                  <Spinner size="sm" variant="primary" className="mr-2" />
                ) : (
                  <Download className="w-4 h-4 mr-2" />
                )}
                Download PDF
              </Button>
            </div>
            
            {pdfGenerating && (
              <div className="text-sm text-gray-400">
                Generating PDF... This may take a few moments.
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 