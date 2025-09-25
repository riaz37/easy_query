"use client";

import React, { useState, useEffect } from 'react';
import { useTaskStore } from '@/store/task-store';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  ChevronDown,
  ChevronUp,
  MoreHorizontal,
  Trash2,
  Eye
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface GlobalTaskIndicatorProps {
  className?: string;
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left';
}

export function GlobalTaskIndicator({ className }: GlobalTaskIndicatorProps) {
  const {
    activeTasks,
    completedTasks,
    failedTasks,
    isTaskListOpen,
    toggleTaskList,
    getActiveTasksCount,
    getCompletedTasksCount,
    getFailedTasksCount,
    clearCompletedTasks,
  } = useTaskStore();

  const [showNotification, setShowNotification] = useState(false);
  const [lastCompletedCount, setLastCompletedCount] = useState(0);

  const activeCount = getActiveTasksCount();
  const completedCount = getCompletedTasksCount();
  const failedCount = getFailedTasksCount();
  const totalTasks = activeCount + completedCount + failedCount;

  // Show notification when tasks complete
  useEffect(() => {
    if (completedCount > lastCompletedCount && lastCompletedCount > 0) {
      setShowNotification(true);
      setTimeout(() => setShowNotification(false), 3000);
    }
    setLastCompletedCount(completedCount);
  }, [completedCount, lastCompletedCount]);

  // Don't show indicator if no tasks
  if (totalTasks === 0) return null;

  const getStatusIcon = () => {
    if (failedCount > 0) return <XCircle className="w-4 h-4 text-red-400" />;
    if (activeCount > 0) return <Clock className="w-4 h-4 text-blue-400 animate-pulse" />;
    if (completedCount > 0) return <CheckCircle className="w-4 h-4 text-green-400" />;
    return <AlertCircle className="w-4 h-4 text-gray-400" />;
  };

  const getStatusText = () => {
    if (failedCount > 0) return `${failedCount} failed`;
    if (activeCount > 0) return `${activeCount} running`;
    if (completedCount > 0) return `${completedCount} completed`;
    return 'No tasks';
  };

  const getStatusColor = () => {
    if (failedCount > 0) return 'border-red-500/50 bg-red-500/10 text-red-400';
    if (activeCount > 0) return 'border-blue-500/50 bg-blue-500/10 text-blue-400';
    if (completedCount > 0) return 'border-green-500/50 bg-green-500/10 text-green-400';
    return 'border-gray-500/50 bg-gray-500/10 text-gray-400';
  };

  return (
    <>
      {/* Global Task Indicator */}
      <div className={cn(
        "fixed bottom-20 right-4 z-30 transition-all duration-300",
        "sm:bottom-20 md:bottom-4 lg:bottom-4",
        className
      )}>
        <div className="flex flex-col gap-2">
          {/* Notification for completed tasks */}
          {showNotification && (
            <div className="animate-in slide-in-from-right-2 duration-300">
              <div className="bg-green-600 text-white px-3 py-2 rounded-lg shadow-lg flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                <span className="text-sm font-medium">
                  {completedCount - lastCompletedCount} task(s) completed!
                </span>
              </div>
            </div>
          )}

          {/* Main Task Indicator */}
          <div className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg border shadow-lg backdrop-blur-sm",
            "hover:shadow-xl transition-all duration-200 cursor-pointer",
            getStatusColor()
          )} onClick={toggleTaskList}>
            {getStatusIcon()}
            <span className="text-sm font-medium">{getStatusText()}</span>
            <Badge variant="secondary" className="ml-1 text-xs">
              {totalTasks}
            </Badge>
            {isTaskListOpen ? (
              <ChevronDown className="w-4 h-4" />
            ) : (
              <ChevronUp className="w-4 h-4" />
            )}
          </div>

          {/* Quick Actions */}
          {completedCount > 0 && (
            <div className="flex gap-1">
              <Button
                size="sm"
                variant="outline"
                className="h-8 px-2 text-xs bg-white/10 border-white/20 text-white hover:bg-white/20"
                onClick={(e) => {
                  e.stopPropagation();
                  clearCompletedTasks();
                }}
              >
                <Trash2 className="w-3 h-3 mr-1" />
                Clear
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Task List Overlay */}
      {isTaskListOpen && (
        <div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm" onClick={toggleTaskList}>
          <div 
            className="fixed right-4 top-1/2 -translate-y-1/2 w-96 max-h-[80vh] bg-gray-900 border border-gray-700 rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <TaskListPanel />
          </div>
        </div>
      )}
    </>
  );
}

// Task List Panel Component
function TaskListPanel() {
  const {
    tasks,
    activeTasks,
    completedTasks,
    failedTasks,
    closeTaskList,
    removeTask,
    clearCompletedTasks,
  } = useTaskStore();

  const [activeTab, setActiveTab] = useState<'all' | 'active' | 'completed' | 'failed'>('all');

  const getTasksToShow = () => {
    switch (activeTab) {
      case 'active': return activeTasks;
      case 'completed': return completedTasks;
      case 'failed': return failedTasks;
      default: return tasks;
    }
  };

  const formatDuration = (startedAt?: Date, completedAt?: Date) => {
    if (!startedAt) return 'Not started';
    const end = completedAt || new Date();
    const duration = Math.floor((end.getTime() - startedAt.getTime()) / 1000);
    
    if (duration < 60) return `${duration}s`;
    if (duration < 3600) return `${Math.floor(duration / 60)}m ${duration % 60}s`;
    return `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`;
  };

  const getTaskIcon = (task: any) => {
    switch (task.status) {
      case 'running': return <Clock className="w-4 h-4 text-blue-400 animate-pulse" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'failed': return <XCircle className="w-4 h-4 text-red-400" />;
      case 'cancelled': return <XCircle className="w-4 h-4 text-gray-400" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <h3 className="text-lg font-semibold text-white">Task Manager</h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={closeTaskList}
          className="text-gray-400 hover:text-white"
        >
          <XCircle className="w-4 h-4" />
        </Button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        {[
          { key: 'all', label: 'All', count: tasks.length },
          { key: 'active', label: 'Active', count: activeTasks.length },
          { key: 'completed', label: 'Completed', count: completedTasks.length },
          { key: 'failed', label: 'Failed', count: failedTasks.length },
        ].map(({ key, label, count }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key as any)}
            className={cn(
              "flex-1 px-3 py-2 text-sm font-medium transition-colors cursor-pointer",
              "hover:bg-gray-800 border-b-2",
              activeTab === key
                ? "border-blue-500 text-blue-400"
                : "border-transparent text-gray-400 hover:text-white"
            )}
          >
            {label} ({count})
          </button>
        ))}
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
              className="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {getTaskIcon(task)}
                  <div className="flex-1 min-w-0">
                    <h4 className="text-sm font-medium text-white truncate">
                      {task.title}
                    </h4>
                    <p className="text-xs text-gray-400 mt-1 line-clamp-2">
                      {task.description}
                    </p>
                    
                    {/* Progress Bar */}
                    {task.status === 'running' && (
                      <div className="mt-2">
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>Progress</span>
                          <span>{task.progress}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-1.5">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                            style={{ width: `${task.progress}%` }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Duration */}
                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                      <span>Created: {task.createdAt.toLocaleTimeString()}</span>
                      {task.startedAt && (
                        <span>Duration: {formatDuration(task.startedAt, task.completedAt)}</span>
                      )}
                    </div>

                    {/* Error Message */}
                    {task.error && (
                      <div className="mt-2 p-2 bg-red-900/20 border border-red-500/30 rounded text-xs text-red-300">
                        {task.error}
                      </div>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1 ml-2">
                  {task.status === 'completed' && task.result && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0 text-gray-400 hover:text-white cursor-pointer"
                      onClick={() => {
                        // TODO: Implement view result functionality
                        console.log('View result:', task.result);
                      }}
                    >
                      <Eye className="w-3 h-3" />
                    </Button>
                  )}
                  
                  {(task.status === 'completed' || task.status === 'failed') && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0 text-gray-400 hover:text-red-400 cursor-pointer"
                      onClick={() => removeTask(task.id)}
                    >
                      <Trash2 className="w-3 h-3" />
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
        <div className="p-4 border-t border-gray-700">
          <Button
            variant="outline"
            size="sm"
            onClick={clearCompletedTasks}
            className="w-full text-gray-400 border-gray-600 hover:bg-gray-800 cursor-pointer"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Clear All Completed
          </Button>
        </div>
      )}
    </div>
  );
}
