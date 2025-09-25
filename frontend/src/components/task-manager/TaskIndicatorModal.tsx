"use client";

import React, { useState, useEffect, useRef } from "react";
import { useTaskStore } from "@/store/task-store";
import { Badge } from "@/components/ui/badge";
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
  triggerRef?: React.RefObject<HTMLDivElement>;
}

export function TaskIndicatorModal({
  isOpen,
  onClose,
  className,
  width = "w-[28rem]",
  position = "right",
  topOffset = "top-20",
  triggerRef,
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
  
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0, right: 0 });
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

  // Calculate dropdown position
  useEffect(() => {
    if (isOpen && triggerRef?.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      
      const dropdownWidth = 448; // 28rem = 448px
      const dropdownHeight = Math.min(600, viewportHeight - 100); // Max height with some margin
      
      let left = triggerRect.left;
      let right = 0;
      
      // Adjust horizontal position based on available space
      if (position === "right") {
        if (triggerRect.right + dropdownWidth > viewportWidth) {
          // Position to the left of trigger
          left = triggerRect.right - dropdownWidth;
        } else {
          // Position to the right of trigger
          left = triggerRect.right;
        }
      } else {
        if (triggerRect.left - dropdownWidth < 0) {
          // Position to the right of trigger
          left = triggerRect.right;
        } else {
          // Position to the left of trigger
          left = triggerRect.left - dropdownWidth;
        }
      }
      
      // Ensure dropdown stays within viewport
      left = Math.max(16, Math.min(left, viewportWidth - dropdownWidth - 16));
      
      const top = Math.min(triggerRect.bottom + 8, viewportHeight - dropdownHeight - 16);
      
      setDropdownPosition({ top, left, right });
    }
  }, [isOpen, triggerRef, position]);

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
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [isOpen, onClose, triggerRef]);

  if (!isOpen) return null;

  return (
    <div
      ref={dropdownRef}
      className={cn(
        "fixed shadow-2xl overflow-hidden z-50 animate-in slide-in-from-top-2 duration-200",
        className
      )}
      style={{
        top: dropdownPosition.top,
        left: dropdownPosition.left,
        width: "28rem",
        maxHeight: "600px",
        borderRadius: "16px",
        background: "linear-gradient(0deg, rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)), linear-gradient(246.02deg, rgba(59, 130, 246, 0.1) 91.9%, rgba(59, 130, 246, 0.25) 114.38%), linear-gradient(59.16deg, rgba(59, 130, 246, 0.1) 71.78%, rgba(59, 130, 246, 0.25) 124.92%)",
        border: "1.5px solid",
        borderImageSource: "linear-gradient(158.39deg, rgba(59, 130, 246, 0.2) 14.19%, rgba(59, 130, 246, 0.05) 50.59%, rgba(59, 130, 246, 0.05) 68.79%, rgba(59, 130, 246, 0.1) 105.18%)",
        boxShadow: "0 20px 40px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(59, 130, 246, 0.2)",
      }}
      onClick={(e) => e.stopPropagation()}
    >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-6">
            <h3 className="text-xl font-bold text-white">Running Task List</h3>
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
                    "px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 whitespace-nowrap flex-shrink-0 cursor-pointer",
                    "hover:scale-105",
                    activeTab === key
                      ? "bg-white/10 text-white"
                      : "bg-transparent text-gray-300 hover:bg-white/5"
                  )}
                >
                  {label} ({count})
                </button>
              ))}
            </div>
          </div>

          {/* Task List */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3" style={{ maxHeight: "400px" }}>
            {getTasksToShow().length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No {activeTab} tasks</p>
              </div>
            ) : (
              getTasksToShow().map((task) => (
                <div
                  key={task.id}
                  className="rounded-xl p-4 border border-green-400/20 hover:border-green-400/30 transition-all duration-200 group"
                  style={{
                    background: "linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.08))",
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-4 flex-1">
                      {/* Simple Status Icon */}
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center border" style={{
                        backgroundColor: task.status === "completed" 
                          ? "rgba(34, 197, 94, 0.2)" 
                          : task.status === "running"
                          ? "rgba(34, 197, 94, 0.2)"
                          : task.status === "failed"
                          ? "rgba(239, 68, 68, 0.2)"
                          : "rgba(34, 197, 94, 0.2)",
                        borderColor: task.status === "completed" 
                          ? "rgba(34, 197, 94, 0.4)" 
                          : task.status === "running"
                          ? "rgba(34, 197, 94, 0.4)"
                          : task.status === "failed"
                          ? "rgba(239, 68, 68, 0.4)"
                          : "rgba(34, 197, 94, 0.4)"
                      }}>
                        {task.status === "running" && <Clock className="w-4 h-4 text-green-300" />}
                        {task.status === "completed" && <CheckCircle className="w-4 h-4 text-green-300" />}
                        {task.status === "failed" && <XCircle className="w-4 h-4 text-red-300" />}
                        {task.status === "pending" && <Clock className="w-4 h-4 text-green-300" />}
                      </div>

                      <div className="flex-1 min-w-0">
                        <h4 className="text-base font-semibold text-white truncate mb-1">
                          {task.title}
                        </h4>

                        {/* Simple Status */}
                        <div className="mb-2">
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            task.status === "completed"
                              ? "bg-green-500/20 text-green-300"
                              : task.status === "running"
                              ? "bg-green-500/20 text-green-300"
                              : task.status === "failed"
                              ? "bg-red-500/20 text-red-300"
                              : "bg-green-500/20 text-green-300"
                          }`}>
                            {task.status === "running"
                              ? "Processing"
                              : task.status === "completed"
                              ? "Completed"
                              : task.status === "failed"
                              ? "Failed"
                              : "Pending"}
                          </span>
                        </div>

                        {/* Simple Metadata */}
                        <div className="text-xs text-gray-400">
                          {task.type.replace("_", " ")}
                        </div>

                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>

        </div>
    </div>
  );
}
