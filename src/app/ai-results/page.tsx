"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { ESAPBrandLoader } from "@/components/ui/loading";
import { ReportResults } from "@/types/reports";
import { useTaskStore } from "@/store/task-store";
import {
  generateAndDownloadPDF,
} from "@/lib/utils/smart-pdf-generator";
import {
  ReportTaskList,
} from "@/components/ai-results";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";

export default function AIResultsPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(false);
  }, []);

  const handleBackToQuery = () => {
    router.push("/database-query");
  };

  const handleViewReport = (taskId: string, results: ReportResults) => {
    // For now, just download the report when view is clicked
    handleDownloadReport(taskId, results);
  };

  const handleDownloadReport = async (taskId: string, results: ReportResults) => {
    try {
      await generateAndDownloadPDF(
        results,
        `AI_Report_${taskId.substring(0, 8)}_${new Date().toISOString().split("T")[0]}.pdf`
      );
    } catch (error) {
      console.error("Failed to generate PDF:", error);
      alert("Failed to generate PDF. Please try again.");
    }
  };

  if (loading) {
    return (
      <PageLayout background="enhanced" backgroundIntensity="high">
        <div className="text-center">
          <ESAPBrandLoader size="xl" className="mx-auto" />
          <p className="text-white mt-4">Loading report results...</p>
        </div>
      </PageLayout>
    );
  }

  return (
    <PageLayout background="enhanced" maxWidth="7xl" backgroundIntensity="high">
      <PageHeader
        title="AI-Generated Reports"
        description="View and manage your completed AI-generated reports"
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

      <div className="space-y-6">
        <ReportTaskList
          onViewReport={handleViewReport}
          onDownloadReport={handleDownloadReport}
        />
      </div>
    </PageLayout>
  );
}