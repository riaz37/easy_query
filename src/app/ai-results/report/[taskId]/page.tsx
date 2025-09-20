"use client";

import React, { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, FileText, Download, Eye, Brain, Clock } from "lucide-react";
import { ESAPBrandLoader } from "@/components/ui/loading";
import { ReportResults } from "@/types/reports";
import { useTaskStore } from "@/store/task-store";
import { ServiceRegistry } from "@/lib/api";
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
  const [reportResults, setReportResults] = useState<ReportResults | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [pdfGenerating, setPdfGenerating] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Set<number>>(
    new Set()
  );
  const [task, setTask] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const taskId = params.taskId as string;

  const toggleSection = (index: number) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSections(newExpanded);
  };

  useEffect(() => {
    const fetchTask = async () => {
      if (!taskId) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);

        // First try to get from local store
        const localTask = getTaskById(taskId);
        if (localTask && localTask.status === "completed" && localTask.result) {
          setTask(localTask);
          setReportResults(localTask.result);
          setLoading(false);
          return;
        }

        // If not found locally, fetch from backend API
        const taskStatus = await ServiceRegistry.reports.getTaskStatus(taskId);

        if (taskStatus.status === "completed" && taskStatus.results) {
          // Create a task object from the API response
          const apiTask = {
            id: taskId,
            type: "report_generation" as const,
            title: taskStatus.user_query || "AI Report",
            description: taskStatus.user_query || "Generated AI Report",
            status: taskStatus.status,
            progress: taskStatus.progress_percentage || 100,
            createdAt: new Date(taskStatus.created_at),
            startedAt: taskStatus.started_at
              ? new Date(taskStatus.started_at)
              : undefined,
            completedAt: taskStatus.completed_at
              ? new Date(taskStatus.completed_at)
              : undefined,
            result: taskStatus.results,
            metadata: {
              backend_task_id: taskId,
              total_queries: taskStatus.total_queries,
              processed_queries: taskStatus.processed_queries,
              successful_queries: taskStatus.successful_queries,
              failed_queries: taskStatus.failed_queries,
            },
          };

          setTask(apiTask);
          setReportResults(taskStatus.results);
        } else {
          setError("Task not completed or no results available");
          console.error(
            "Task not completed or no results available:",
            taskStatus
          );
        }
      } catch (error) {
        setError("Failed to load report. Please try again.");
        console.error("Failed to fetch task:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchTask();
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
        `AI_Report_${taskId.substring(0, 8)}_${
          new Date().toISOString().split("T")[0]
        }.pdf`
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
    a.download = `AI_Report_${taskId.substring(0, 8)}_${
      new Date().toISOString().split("T")[0]
    }.txt`;
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
      <PageLayout background={["frame", "gridframe"]}>
        <div className="text-center">
          <ESAPBrandLoader size="xl" className="mx-auto" />
          <p className="text-white mt-4">Loading report details...</p>
        </div>
      </PageLayout>
    );
  }

  if (error) {
    return (
      <PageLayout background={["frame", "gridframe"]}>
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-red-400" />
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">
            Report Not Available
          </h2>
          <p className="text-gray-400 mb-6">{error}</p>
          <div className="flex gap-4 justify-center">
            <Button
              onClick={() => window.location.reload()}
              variant="outline"
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
            >
              Try Again
            </Button>
            <Button
              onClick={handleBackToResults}
              variant="outline"
              className="border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Reports
            </Button>
          </div>
        </div>
      </PageLayout>
    );
  }

  if (!task || !reportResults) {
    return (
      <PageLayout background={["frame", "gridframe"]}>
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
    <PageLayout background={["frame", "gridframe"]} maxWidth="7xl">
      <div className="modal-enhanced">
        <div 
          className="modal-content-enhanced overflow-hidden"
          style={{
            background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
linear-gradient(230.27deg, rgba(19, 245, 132, 0) 71.59%, rgba(19, 245, 132, 0.2) 98.91%),
linear-gradient(67.9deg, rgba(19, 245, 132, 0) 66.65%, rgba(19, 245, 132, 0.2) 100%)`,
            backdropFilter: "blur(30px)"
          }}
        >
          {/* Header Section */}
          <div className="p-6">
            {/* Title and Query Info */}
            <div className="mb-6">
              <h1 className="modal-title-enhanced text-3xl font-bold mb-4">
                AI Report Results
              </h1>
              <div className="p-4 rounded-lg" style={{
                background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                border: "1px solid var(--components-button-outlined, rgba(145, 158, 171, 0.32))"
              }}>
                <div className="flex items-center gap-2 mb-2">
                  <Brain className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400 font-medium">Query:</span>
                </div>
                <p className="text-white text-sm">{task.title}</p>
              </div>
            </div>

            {/* Action Bar */}
            <div className="flex items-center justify-between">
              {/* Export Controls */}
              <div className="flex items-center gap-3">
                <Button
                  onClick={handleDownloadPDF}
                  disabled={pdfGenerating}
                  variant="outline"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer"
                  style={{
                    background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                    borderRadius: "118.8px",
                    height: "48px",
                    minWidth: "120px"
                  }}
                >
                  <Download className="w-4 h-4 mr-2" />
                  {pdfGenerating ? "Generating..." : "Download PDF"}
                </Button>
                <Button
                  onClick={handlePreviewPDF}
                  disabled={pdfGenerating}
                  variant="outline"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer"
                  style={{
                    background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                    borderRadius: "118.8px",
                    height: "48px",
                    minWidth: "100px"
                  }}
                >
                  <Eye className="w-4 h-4 mr-2" />
                  Preview
                </Button>
                <Button
                  onClick={handleBackToResults}
                  variant="outline"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer"
                  style={{
                    background: "var(--components-paper-bg-paper-blur, rgba(255, 255, 255, 0.04))",
                    borderRadius: "118.8px",
                    height: "48px",
                    minWidth: "100px"
                  }}
                >
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back
                </Button>
              </div>
            </div>
          </div>

          {/* Content Section */}
          <div className="px-6 pb-6 space-y-6">
            {/* Report Overview */}
            <ReportHeader reportResults={reportResults} />

            {/* LLM Analysis Overview */}
            <LLMAnalysisOverview reportResults={reportResults} />

            {/* Report Sections */}
            {reportResults.results && reportResults.results.length > 0 && (
              <div className="space-y-6">
                <h2 className="modal-title-enhanced text-2xl font-bold mb-4">
                  Report Sections
                </h2>
                {reportResults.results.map((section, index) => (
                  <ReportSection
                    key={index}
                    section={section}
                    index={index}
                    expandedSections={expandedSections}
                    toggleSection={toggleSection}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </PageLayout>
  );
}
