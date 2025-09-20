"use client";

import React, { useState } from "react";
import { useTaskStore } from "@/store/task-store";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Clock,
  CheckCircle,
  XCircle,
  ChevronDown,
  Trash2,
  Eye,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface TaskIndicatorModalProps {
  isOpen: boolean;
  onClose: () => void;
  className?: string;
  width?: string;
  position?: "right" | "left";
  topOffset?: string;
}

export function TaskIndicatorModal({
  isOpen,
  onClose,
  className,
  width = "w-[28rem]",
  position = "right",
  topOffset = "top-20",
}: TaskIndicatorModalProps) {
  const {
    tasks,
    activeTasks,
    completedTasks,
    failedTasks,
    removeTask,
    clearCompletedTasks,
  } = useTaskStore();

  const [activeTab, setActiveTab] = useState<
    "all" | "active" | "completed" | "failed"
  >("all");

  const getTasksToShow = () => {
    switch (activeTab) {
      case "active":
        return activeTasks;
      case "completed":
        return completedTasks;
      case "failed":
        return failedTasks;
      default:
        return tasks;
    }
  };

  const formatDuration = (startedAt?: Date, completedAt?: Date) => {
    if (!startedAt) return "Not started";
    const end = completedAt || new Date();
    const duration = Math.floor((end.getTime() - startedAt.getTime()) / 1000);

    if (duration < 60) return `${duration}s`;
    if (duration < 3600)
      return `${Math.floor(duration / 60)}m ${duration % 60}s`;
    return `${Math.floor(duration / 3600)}h ${Math.floor(
      (duration % 3600) / 60
    )}m`;
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop overlay */}
      <div
        className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          width: "100vw",
          height: "100vh",
        }}
      />
      
      {/* Modal positioned relative to viewport */}
      <div
        className={cn(
          "fixed shadow-2xl overflow-hidden z-50 query-content-gradient",
          position === "right" ? "right-4" : "left-4",
          className
        )}
        style={{
          top: "5.5rem", // Position below navbar (64px + 24px gap)
          maxHeight: "calc(100vh - 7rem)",
          height: "auto",
          width: "clamp(20rem, 28rem, calc(100vw - 2rem))", // Responsive width with clamp
          borderRadius: "32px",
          boxShadow:
            "0 20px 40px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(19, 245, 132, 0.1)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6">
            <h3 className="text-xl font-bold text-white">Running Task List</h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="text-gray-400 hover:text-white hover:bg-white/10 rounded-full p-2"
            >
              <XCircle className="w-5 h-5" />
            </Button>
          </div>

          {/* Tabs Container */}
          <div className="p-4 rounded-2xl">
            <div className="flex gap-2">
              {[
                { key: "all", label: "All", count: tasks.length },
                { key: "active", label: "Active", count: activeTasks.length },
                {
                  key: "completed",
                  label: "Complete",
                  count: completedTasks.length,
                },
                { key: "failed", label: "Failed", count: failedTasks.length },
              ].map(({ key, label, count }) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key as any)}
                  className={cn(
                    "px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 whitespace-nowrap flex-shrink-0",
                    "hover:scale-105",
                    activeTab === key
                      ? "bg-white/10 text-white ring-2 ring-white/20"
                      : "bg-transparent text-gray-300 hover:bg-white/5 hover:ring-1 hover:ring-white/10"
                  )}
                >
                  {label} ({count})
                </button>
              ))}
            </div>
          </div>

          {/* Task List */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {getTasksToShow().length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No {activeTab} tasks</p>
              </div>
            ) : (
              getTasksToShow().map((task) => (
                <div
                  key={task.id}
                  className="rounded-xl p-4 border border-white/10 hover:border-white/20 transition-all duration-200 group"
                  style={{
                    background:
                      "var(--item-root-active-bgcolor, rgba(19, 245, 132, 0.08))",
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-4 flex-1">
                      {/* Modern Icon with Badge */}
                      <div className="relative">
                        <div className="w-10 h-10 bg-yellow-500/20 rounded-lg flex items-center justify-center border border-yellow-500/30">
                          <span className="text-yellow-300 font-bold text-sm">
                            1
                          </span>
                        </div>
                        {task.status === "running" && (
                          <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white/20 flex items-center justify-center">
                            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                          </div>
                        )}
                        {task.status === "completed" && (
                          <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white/20 flex items-center justify-center">
                            <CheckCircle className="w-2.5 h-2.5 text-white" />
                          </div>
                        )}
                        {task.status === "failed" && (
                          <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-red-500 rounded-full border-2 border-white/20 flex items-center justify-center">
                            <XCircle className="w-2.5 h-2.5 text-white" />
                          </div>
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <h4 className="text-base font-semibold text-white truncate mb-1">
                          {task.title}
                        </h4>
                        <p className="text-sm text-gray-300 mb-3 line-clamp-2">
                          {task.description}
                        </p>

                        {/* Progress Bar */}
                        <div className="mb-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-gray-400">
                              {task.status === "running"
                                ? "Processing..."
                                : task.status === "completed"
                                ? "Completed"
                                : task.status === "failed"
                                ? "Failed"
                                : "Pending"}
                            </span>
                            <span className="text-xs font-medium text-white">
                              100%
                            </span>
                          </div>
                          <div className="w-full bg-gray-700/50 rounded-full h-2">
                            <div
                              className="h-2 rounded-full transition-all duration-300"
                              style={{
                                width:
                                  task.status === "completed"
                                    ? "100%"
                                    : task.status === "running"
                                    ? "75%"
                                    : "25%",
                                backgroundColor:
                                  task.status === "completed"
                                    ? "#10b981"
                                    : task.status === "running"
                                    ? "#13f584"
                                    : task.status === "failed"
                                    ? "#ef4444"
                                    : "#f59e0b",
                              }}
                            ></div>
                          </div>
                        </div>

                        {/* Metadata */}
                        <div className="flex items-center justify-between text-xs text-gray-400">
                          <div className="flex items-center gap-4">
                            <span>
                              {task.createdAt.toLocaleDateString("en-GB", {
                                day: "2-digit",
                                month: "short",
                                year: "numeric",
                              })}
                            </span>
                            {task.startedAt && (
                              <span>
                                • Duration{" "}
                                {formatDuration(
                                  task.startedAt,
                                  task.completedAt
                                )}
                              </span>
                            )}
                            <span>• {task.type.replace("_", " ")}</span>
                          </div>
                        </div>

                        {/* Completion Celebration */}
                        {task.status === "completed" && task.result && (
                          <div className="mt-2 p-2 bg-green-900/20 border border-green-500/30 rounded text-xs text-green-300">
                            <div className="flex items-center gap-2">
                              <div className="w-1 h-1 bg-green-400 rounded-full animate-pulse"></div>
                              <span>Report generated successfully!</span>
                              <div className="ml-auto flex gap-1">
                                <div
                                  className="w-1 h-1 bg-green-400 rounded-full animate-bounce"
                                  style={{ animationDelay: "0ms" }}
                                ></div>
                                <div
                                  className="w-1 h-1 bg-green-400 rounded-full animate-bounce"
                                  style={{ animationDelay: "150ms" }}
                                ></div>
                                <div
                                  className="w-1 h-1 bg-green-400 rounded-full animate-bounce"
                                  style={{ animationDelay: "300ms" }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Error Message */}
                        {task.error && (
                          <div className="mt-2 p-2 bg-red-900/20 border border-red-500/30 rounded text-xs text-red-300">
                            <div className="flex items-center gap-2">
                              <div className="w-1 h-1 bg-red-400 rounded-full animate-pulse"></div>
                              <span>{task.error}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-2 ml-4">
                      {task.status === "completed" && task.result && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-8 w-8 p-0 text-gray-400 hover:text-white hover:bg-white/10 rounded-full transition-all duration-200"
                          onClick={() => {
                            const backendTaskId =
                              task.metadata?.backend_task_id;
                            const taskId = backendTaskId || task.id;
                            window.location.href = `/ai-results/report/${taskId}`;
                          }}
                        >
                          <Eye className="w-4 h-4" />
                        </Button>
                      )}

                      {(task.status === "completed" ||
                        task.status === "failed") && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-8 w-8 p-0 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-full transition-all duration-200"
                          onClick={() => removeTask(task.id)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Footer Actions */}
          {completedTasks.length > 0 && (
            <div className="p-6 border-t border-white/10">
              <Button
                variant="outline"
                size="sm"
                onClick={clearCompletedTasks}
                className="w-full text-gray-300 border-white/20 hover:bg-white/10 hover:border-white/30 rounded-xl transition-all duration-200"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear All Completed
              </Button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
