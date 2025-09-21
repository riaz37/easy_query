"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { ReportResults } from "@/types/reports";
import { generateAndDownloadPDF } from "@/lib/utils/smart-pdf-generator";
import { PaginatedReportList } from "@/components/ai-reports";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";

export default function AIReportsPage() {
  const router = useRouter();

  const handleBackToQuery = () => {
    router.push("/database-query");
  };

  const handleViewReport = (taskId: string, results: ReportResults) => {
    router.push(`/ai-reports/report/${taskId}`);
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

  return (
    <PageLayout background={["frame", "gridframe"]} maxWidth="7xl">
      <PageHeader
        title="AI Reports"
        description="View and manage your completed AI-generated reports based on your queries"
        enhancedTitle={true}
      />

      <div className="space-y-6">
        <PaginatedReportList
          onViewReport={handleViewReport}
          onDownloadReport={handleDownloadReport}
        />
      </div>
    </PageLayout>
  );
}