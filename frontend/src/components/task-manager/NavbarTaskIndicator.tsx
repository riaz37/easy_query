"use client";

import React, { useRef } from 'react';
import { useTaskStore } from '@/store/task-store';
import { Badge } from '@/components/ui/badge';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { TaskIndicatorModal } from './TaskIndicatorModal';

interface NavbarTaskIndicatorProps {
  className?: string;
}

export function NavbarTaskIndicator({ className }: NavbarTaskIndicatorProps) {
  const {
    activeTasks,
    completedTasks,
    failedTasks,
    isTaskListOpen,
    toggleTaskList,
    getActiveTasksCount,
    getCompletedTasksCount,
    getFailedTasksCount,
  } = useTaskStore();
  
  const triggerRef = useRef<HTMLDivElement>(null);

  const activeCount = getActiveTasksCount();
  const completedCount = getCompletedTasksCount();
  const failedCount = getFailedTasksCount();
  const totalTasks = activeCount + completedCount + failedCount;

  // Don't show indicator if no tasks
  if (totalTasks === 0) return null;

  const getStatusText = () => {
    if (failedCount > 0) return 'failed';
    if (activeCount > 0) return 'running';
    if (completedCount > 0) return 'completed';
    return 'No tasks';
  };

  return (
    <>
      {/* Compact Task Indicator */}
      <div className={cn("relative", className)}>
        <div 
          ref={triggerRef}
          className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-full border transition-all duration-200 cursor-pointer group",
            "hover:shadow-lg"
          )}
          style={{
            background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
            color: 'var(--primary-main, rgba(19, 245, 132, 1))'
          }}
          onClick={toggleTaskList}
        >
          <span className="text-sm font-medium hidden sm:inline">{getStatusText()}</span>
          <Badge 
            variant="secondary" 
            className="ml-1 text-xs text-white"
            style={{
              background: 'var(--primary-dark, rgba(13, 172, 92, 1))'
            }}
          >
            {totalTasks}
          </Badge>
          <ChevronDown className={cn(
            "w-4 h-4 transition-transform duration-200",
            isTaskListOpen ? "rotate-180" : ""
          )} />
        </div>
      </div>

      {/* Reusable Task Indicator Modal */}
      <TaskIndicatorModal
        isOpen={isTaskListOpen}
        onClose={toggleTaskList}
        position="right"
        topOffset="top-20"
        className="z-50"
        triggerRef={triggerRef}
      />
    </>
  );
}