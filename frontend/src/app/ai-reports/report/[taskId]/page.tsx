"use client";

import React, { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { FileText, Brain } from "lucide-react";
import { EasyQueryBrandLoader } from "@/components/ui/loading";
import { ReportResults } from "@/types/reports";
import { useTaskStore } from "@/store/task-store";
import { ServiceRegistry } from "@/lib/api";
import { ReportSection } from "@/components/ai-reports";
import { PageLayout, PageHeader } from "@/components/layout/PageLayout";

export default function ReportDetailPage() {
  const router = useRouter();
  const params = useParams();
  const { getTaskById } = useTaskStore();
  const [reportResults, setReportResults] = useState<ReportResults | null>(
    null
  );
  const [loading, setLoading] = useState(true);
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

  if (loading) {
    return (
      <PageLayout background={["frame", "gridframe"]}>
        <div className="text-center">
          <EasyQueryBrandLoader size="xl" className="mx-auto" />
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
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 border border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10 rounded cursor-pointer"
            >
              Try Again
            </button>
            <button
              onClick={() => router.push("/ai-reports")}
              className="px-4 py-2 border border-emerald-400/30 text-emerald-400 hover:bg-emerald-400/10 rounded cursor-pointer"
            >
              Back to Reports
            </button>
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
          <button
            onClick={() => router.push("/ai-reports")}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded cursor-pointer"
          >
            Back to Reports
          </button>
        </div>
      </PageLayout>
    );
  }

  return (
    <PageLayout background={["frame", "gridframe"]} maxWidth="7xl">
      {/* Header Section */}
      <div className="mb-6">
        {/* Title and Query Info */}
        <div className="mb-6">
          <h1 className="modal-title-enhanced text-3xl font-bold mb-2">
            AI Report Results
          </h1>
          <p className="text-white text-sm">{task.title}</p>
        </div>
          </div>

      {/* Content Section */}
      <div className="space-y-6">
      {/* Report Sections */}
      {reportResults.results && reportResults.results.length > 0 && (
        <div className="space-y-6">
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
    </PageLayout>
  );
}
