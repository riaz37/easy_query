"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ArrowLeft, FileText } from "lucide-react";
import { ESAPBrandLoader } from "@/components/ui/loading";
import { ReportResults } from "@/types/reports";
import { generateAndDownloadPDF, generatePDFBlob } from "@/lib/utils/smart-pdf-generator";
import {
  ReportHeader,
  ExportControls,
  LLMAnalysisOverview,
  ProcessingDetails,
  ReportSection,
} from "@/components/ai-results";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";

export default function AIResultsPage() {
  const router = useRouter();
  const [reportResults, setReportResults] = useState<ReportResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [pdfGenerating, setPdfGenerating] = useState(false);
  const [expandedAnalysis, setExpandedAnalysis] = useState<Set<number>>(new Set());

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
    // Get results from sessionStorage
    const storedResults = sessionStorage.getItem("reportResults");
    if (storedResults) {
      try {
        const results = JSON.parse(storedResults);
        console.log("AI Results Page - Loaded data:", results);
        console.log("AI Results Page - Results structure:", {
          hasResults: !!results.results,
          resultsLength: results.results?.length,
          hasLLMAnalysis: results.results?.some((section: any) => section.graph_and_analysis?.llm_analysis),
          sampleSection: results.results?.[0]
        });
        setReportResults(results);
      } catch (error) {
        console.error("Failed to parse report results:", error);
      }
    }
    setLoading(false);
  }, []);

  const handleBackToQuery = () => {
    router.push("/database-query");
  };

  const handleDownloadPDF = async () => {
    if (!reportResults) return;

    setPdfGenerating(true);
    try {
      await generateAndDownloadPDF(
        reportResults,
        `AI_Report_${new Date().toISOString().split("T")[0]}.pdf`
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
    a.download = `AI_Report_${new Date().toISOString().split("T")[0]}.txt`;
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
      <PageLayout background="gradient">
        <div className="text-center">
          <ESAPBrandLoader size="xl" className="mx-auto" />
          <p className="text-white mt-4">Loading report results...</p>
        </div>
      </PageLayout>
    );
  }

  if (!reportResults) {
    return (
      <PageLayout background="gradient" maxWidth="4xl">
        <Card className="bg-gray-900/50 border-red-400/30">
          <CardContent className="pt-12 pb-12 text-center">
            <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <FileText className="w-8 h-8 text-red-400" />
            </div>
            <h3 className="text-white text-lg font-medium mb-2">
              No Report Results Found
            </h3>
            <p className="text-gray-400 mb-4">
              Please generate a report first from the Database Query page
            </p>
            <Button
              onClick={handleBackToQuery}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Database Query
            </Button>
          </CardContent>
        </Card>
      </PageLayout>
    );
  }

  return (
    <PageLayout background="gradient" maxWidth="7xl">
      <PageHeader
        title="AI-Generated Report"
        description={
          <>
            Comprehensive analysis and insights generated by AI
            {reportResults.results &&
              reportResults.results.some((section) => section.graph_and_analysis?.llm_analysis) && (
                <span className="text-blue-400 ml-2">
                  • Overview above, detailed analysis in sections below
                </span>
              )}
          </>
        }
        actions={
          <Button
            onClick={handleBackToQuery}
            variant="outline"
            className="border-gray-600 text-gray-300 hover:bg-gray-700"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Query
          </Button>
        }
      />

            {/* Report Header */}
            <ReportHeader reportResults={reportResults} />

            {/* Export Controls */}
            <ExportControls
              reportResults={reportResults}
              pdfGenerating={pdfGenerating}
              onDownloadPDF={handleDownloadPDF}
              onPreviewPDF={handlePreviewPDF}
              onDownloadText={handleDownloadText}
            />

            {/* LLM Analysis Overview */}
            <LLMAnalysisOverview
              reportResults={reportResults}
            />

            {/* Report Sections */}
            {reportResults.results && reportResults.results.length > 0 && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-white">Detailed Report Sections</h2>
                <p className="text-gray-400 mb-4">
                  Expand each section below to view the complete AI analysis, data visualization, and detailed insights.
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
              <Card className="bg-gray-900/50 border-gray-400/30">
                <CardContent className="pt-12 pb-12 text-center">
                  <div className="w-16 h-16 bg-gray-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
                    <FileText className="w-8 h-8 text-gray-400" />
                  </div>
                  <h3 className="text-white text-lg font-medium mb-2">
                    Report Generated Successfully
                  </h3>
                  <p className="text-gray-400">
                    The report has been generated with {reportResults.total_queries} queries.
                    {reportResults.successful_queries > 0 && (
                      <span className="text-green-400"> {reportResults.successful_queries} queries were successful.</span>
                    )}
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Processing Details */}
            <ProcessingDetails reportResults={reportResults} />
    </PageLayout>
  );
}
