"use client";

import React, { useState, useEffect, useRef } from "react";
import { useTaskStore } from "@/store/task-store";
import { Clock, CheckCircle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface TaskIndicatorModalProps {
  isOpen: boolean;
  onClose: () => void;
  className?: string;
  triggerRef?: React.RefObject<HTMLDivElement>;
}

export function TaskIndicatorModal({
  isOpen,
  onClose,
  className,
  triggerRef,
}: TaskIndicatorModalProps) {
  const { tasks, activeTasks, completedTasks, failedTasks } = useTaskStore();
  const [activeTab, setActiveTab] = useState<
    "all" | "active" | "completed" | "failed"
  >("all");
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running":
        return <Clock className="w-4 h-4 text-green-400" />;
      case "completed":
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running":
        return "text-green-400 bg-green-500/20";
      case "completed":
        return "text-green-400 bg-green-500/20";
      case "failed":
        return "text-red-400 bg-red-500/20";
      default:
        return "text-yellow-400 bg-yellow-500/20";
    }
  };

  // Handle click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node) &&
        triggerRef?.current &&
        !triggerRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () =>
        document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen, onClose, triggerRef]);

  if (!isOpen) return null;

  return (
    <div className="absolute top-full right-0 mt-2 z-50 animate-in slide-in-from-top-4 duration-300">
      <div
        ref={dropdownRef}
        className={cn(
          "w-[28rem] max-h-[80vh] overflow-y-auto rounded-[32px] shadow-2xl border",
          className
        )}
        style={{
          background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
                        linear-gradient(246.02deg, rgba(19, 245, 132, 0) 91.9%, rgba(19, 245, 132, 0.2) 114.38%),
                        linear-gradient(59.16deg, rgba(19, 245, 132, 0) 71.78%, rgba(19, 245, 132, 0.2) 124.92%)`,
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-2 py-1">
          {/* Header */}
          <div className="px-4 py-3 border-b border-white/10">
            <h3 className="text-lg font-semibold text-white">
              Running Task List
            </h3>
          </div>

          {/* Tabs */}
          <div className="px-2 py-2">
            <div className="flex gap-1">
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
                    "text-white active:scale-95 transition-all duration-200 flex items-center gap-2 px-4 py-2.5 cursor-pointer text-sm font-medium rounded-full whitespace-nowrap flex-shrink-0",
                    activeTab === key
                      ? "bg-white/10"
                      : "bg-transparent hover:bg-white/5"
                  )}
                  style={{
                    backgroundColor:
                      activeTab === key
                        ? "rgba(255, 255, 255, 0.1)"
                        : "transparent",
                    borderRadius: "99px",
                  }}
                  onMouseEnter={(e) => {
                    if (activeTab !== key) {
                      e.currentTarget.style.backgroundColor =
                        "rgba(255, 255, 255, 0.05)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (activeTab !== key) {
                      e.currentTarget.style.backgroundColor = "transparent";
                    }
                  }}
                >
                  {label} ({count})
                </button>
              ))}
            </div>
          </div>

          {/* Task List */}
          <div className="px-2 py-1 max-h-80 overflow-y-auto">
            {getTasksToShow().length === 0 ? (
              <div className="text-center py-6 text-gray-400">
                <Clock className="w-6 h-6 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No {activeTab} tasks</p>
              </div>
            ) : (
              getTasksToShow().map((task) => (
                <div
                  key={task.id}
                  className="text-white active:scale-95 mx-1 transition-all duration-200 flex items-center gap-3 p-3 cursor-pointer rounded-full"
                  style={{
                    backgroundColor: "transparent",
                    borderRadius: "99px",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor =
                      "rgba(19, 245, 132, 0.08)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  {/* Status Icon */}
                  <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0">
                    {getStatusIcon(task.status)}
                  </div>

                  {/* Task Content */}
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium text-white truncate">
                      {task.title}
                    </div>
                    <div className="text-xs text-gray-400 mt-0.5">
                      {task.status} • {task.type.replace("_", " ")}
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div className="flex-shrink-0">
                    <span
                      className={cn(
                        "text-xs px-2 py-1 rounded-full",
                        getStatusColor(task.status)
                      )}
                    >
                      {task.status}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}