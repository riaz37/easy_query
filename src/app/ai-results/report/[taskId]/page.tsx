"use client";

import React, { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, FileText, Download, Eye } from "lucide-react";
import { ESAPBrandLoader } from "@/components/ui/loading";
import { ReportResults } from "@/types/reports";
import { useTaskStore } from "@/store/task-store";
import {
  generateAndDownloadPDF,
  generatePDFBlob,
} from "@/lib/utils/smart-pdf-generator";
import {
  ReportHeader,
  ExportControls,
  LLMAnalysisOverview,
  ProcessingDetails,
  ReportSection,
} from "@/components/ai-results";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";

export default function ReportDetailPage() {
  const router = useRouter();
  const params = useParams();
  const { getTaskById } = useTaskStore();
  const [reportResults, setReportResults] = useState<ReportResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [pdfGenerating, setPdfGenerating] = useState(false);
  const [expandedAnalysis, setExpandedAnalysis] = useState<Set<number>>(
    new Set()
  );
  const [task, setTask] = useState<any>(null);

  const taskId = params.taskId as string;

  const toggleAnalysis = (index: number) => {
    const newExpanded = new Set(expandedAnalysis);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedAnalysis(newExpanded);
  };

  useEffect(() => {
    if (taskId) {
      const foundTask = getTaskById(taskId);
      if (foundTask && foundTask.status === 'completed' && foundTask.result) {
        setTask(foundTask);
        setReportResults(foundTask.result);
      } else {
        // Task not found or not completed
        console.error('Task not found or not completed:', taskId);
      }
    }
    setLoading(false);
  }, [taskId, getTaskById]);

  const handleBackToResults = () => {
    router.push("/ai-results");
  };

  const handleDownloadPDF = async () => {
    if (!reportResults) return;

    setPdfGenerating(true);
    try {
      await generateAndDownloadPDF(
        reportResults,
        `AI_Report_${taskId.substring(0, 8)}_${new Date().toISOString().split("T")[0]}.pdf`
      );
    } catch (error) {
      console.error("Failed to generate PDF:", error);
      alert("Failed to generate PDF. Please try again.");
    } finally {
      setPdfGenerating(false);
    }
  };

  const handlePreviewPDF = async () => {
    if (!reportResults) return;

    setPdfGenerating(true);
    try {
      const blob = await generatePDFBlob(reportResults);
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to preview PDF:", error);
      alert("Failed to preview PDF. Please try again.");
    } finally {
      setPdfGenerating(false);
    }
  };

  const handleDownloadText = () => {
    if (!reportResults) return;

    // Create a downloadable text report
    const reportText = generateReportText(reportResults);
    const blob = new Blob([reportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `AI_Report_${taskId.substring(0, 8)}_${new Date().toISOString().split("T")[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const generateReportText = (results: ReportResults): string => {
    let report = "AI-GENERATED REPORT\n";
    report += "=".repeat(50) + "\n\n";
    report += `Database ID: ${results.database_id}\n`;
    report += `Total Queries: ${results.total_queries}\n`;
    report += `Successful Queries: ${results.successful_queries}\n`;
    report += `Failed Queries: ${results.failed_queries}\n\n`;

    if (results.results) {
      results.results.forEach((section, index) => {
        report += `Section ${section.section_number}: ${section.section_name}\n`;
        report += "-".repeat(30) + "\n";
        report += `Query ${section.query_number}: ${section.query}\n`;
        report += `Status: ${section.success ? "Success" : "Failed"}\n\n`;
      });
    }

    return report;
  };

  if (loading) {
    return (
      <PageLayout background="enhanced" backgroundIntensity="high">
        <div className="text-center">
          <ESAPBrandLoader size="xl" className="mx-auto" />
          <p className="text-white mt-4">Loading report details...</p>
        </div>
      </PageLayout>
    );
  }

  if (!task || !reportResults) {
    return (
      <PageLayout background="enhanced" backgroundIntensity="high">
        <div className="text-center">
          <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-red-400" />
          </div>
          <h3 className="text-white text-lg font-medium mb-2">
            Report Not Found
          </h3>
          <p className="text-gray-400 mb-4">
            The requested report could not be found or is not completed.
          </p>
          <Button
            onClick={handleBackToResults}
            className="bg-emerald-600 hover:bg-emerald-700"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Reports
          </Button>
        </div>
      </PageLayout>
    );
  }

  return (
    <PageLayout background="enhanced" maxWidth="7xl" backgroundIntensity="high">
      <PageHeader
        title={`${task.title}`}
        description={
          <div className="flex items-center gap-2">
            <span>Generated on {task.completedAt?.toLocaleDateString()}</span>
            {task.metadata?.selected_structure && (
              <Badge variant="outline" className="text-xs">
                {task.metadata.selected_structure}
              </Badge>
            )}
          </div>
        }
        actions={
          <Button
            onClick={handleBackToResults}
            variant="outline"
            className="border-gray-600 text-gray-300 hover:bg-gray-700"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Reports
          </Button>
        }
      />

      {/* Export Controls */}
      <ExportControls
        reportResults={reportResults}
        pdfGenerating={pdfGenerating}
        onDownloadPDF={handleDownloadPDF}
        onPreviewPDF={handlePreviewPDF}
        onDownloadText={handleDownloadText}
      />

      {/* LLM Analysis Overview */}
      <LLMAnalysisOverview reportResults={reportResults} />

      {/* Report Sections */}
      {reportResults.results && reportResults.results.length > 0 && (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-white">
            Detailed Report Sections
          </h2>
          <p className="text-gray-400 mb-4">
            Expand each section below to view the complete AI analysis, data
            visualization, and detailed insights.
          </p>

          {reportResults.results.map((section, index) => (
            <ReportSection
              key={index}
              section={section}
              index={index}
              expandedAnalysis={expandedAnalysis}
              toggleAnalysis={toggleAnalysis}
            />
          ))}
        </div>
      )}

      {/* No Sections Message */}
      {(!reportResults.results || reportResults.results.length === 0) && (
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="pt-12 pb-12 text-center">
              <div className="w-16 h-16 bg-emerald-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileText className="w-8 h-8 text-emerald-400" />
              </div>
              <h3 className="text-white text-lg font-medium mb-2">
                Report Generated Successfully
              </h3>
              <p className="text-gray-400">
                The report has been generated with {reportResults.total_queries}{" "}
                queries.
                {reportResults.successful_queries > 0 && (
                  <span className="text-emerald-400">
                    {" "}
                    {reportResults.successful_queries} queries were successful.
                  </span>
                )}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Processing Details */}
      <ProcessingDetails reportResults={reportResults} />
    </PageLayout>
  );
}
