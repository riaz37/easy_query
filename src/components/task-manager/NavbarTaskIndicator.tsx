"use client";

import React, { useState } from 'react';
import { useTaskStore } from '@/store/task-store';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  ChevronDown,
  MoreHorizontal,
  Trash2,
  Eye
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useTheme } from '@/store/theme-store';

interface NavbarTaskIndicatorProps {
  className?: string;
}

export function NavbarTaskIndicator({ className }: NavbarTaskIndicatorProps) {
  const theme = useTheme();
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

  const activeCount = getActiveTasksCount();
  const completedCount = getCompletedTasksCount();
  const failedCount = getFailedTasksCount();
  const totalTasks = activeCount + completedCount + failedCount;

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
      {/* Compact Task Indicator */}
      <div className={cn("relative", className)}>
        <div className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-full border transition-all duration-200 cursor-pointer group",
          "hover:shadow-lg",
          getStatusColor(),
          theme === "dark"
            ? "bg-gray-700/50 border-gray-600/30 hover:bg-gray-600/50"
            : "bg-gray-200/50 border-gray-300/30 hover:bg-gray-300/50"
        )} onClick={toggleTaskList}>
          {getStatusIcon()}
          <span className="text-sm font-medium hidden sm:inline">{getStatusText()}</span>
          <Badge variant="secondary" className="ml-1 text-xs bg-white/20 text-white">
            {totalTasks}
          </Badge>
          <ChevronDown className={cn(
            "w-4 h-4 transition-transform duration-200",
            isTaskListOpen ? "rotate-180" : ""
          )} />
        </div>
      </div>

      {/* Task List Overlay */}
      {isTaskListOpen && (
        <div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm" onClick={toggleTaskList}>
          <div 
            className="fixed right-4 top-20 w-96 max-h-[80vh] bg-gray-900 border border-gray-700 rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <TaskListPanel />
          </div>
        </div>
      )}
    </>
  );
}

// Task List Panel Component (reused from GlobalTaskIndicator)
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
    const iconClass = "w-4 h-4";
    
    switch (task.status) {
      case 'running': 
        return (
          <div className="relative">
            <Clock className={`${iconClass} text-blue-400 animate-spin`} />
            <div className="absolute inset-0 w-4 h-4 border-2 border-blue-400/30 rounded-full animate-ping"></div>
          </div>
        );
      case 'completed': 
        return (
          <div className="relative">
            <CheckCircle className={`${iconClass} text-green-400`} />
            <div className="absolute inset-0 w-4 h-4 bg-green-400/20 rounded-full animate-pulse"></div>
          </div>
        );
      case 'failed': 
        return (
          <div className="relative">
            <XCircle className={`${iconClass} text-red-400`} />
            <div className="absolute inset-0 w-4 h-4 bg-red-400/20 rounded-full animate-pulse"></div>
          </div>
        );
      case 'cancelled': 
        return <XCircle className={`${iconClass} text-gray-400`} />;
      case 'pending':
        return (
          <div className="relative">
            <Clock className={`${iconClass} text-yellow-400`} />
            <div className="absolute inset-0 w-4 h-4 border-2 border-yellow-400/30 rounded-full animate-pulse"></div>
          </div>
        );
      default: 
        return <Clock className={`${iconClass} text-gray-400`} />;
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
              "flex-1 px-3 py-2 text-sm font-medium transition-colors",
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
                    
                    {/* Simple Processing Display */}
                    {task.status === 'running' && (
                      <div className="mt-2">
                        <div className="flex items-center gap-2 text-xs text-blue-400">
                          <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></div>
                          <span>Processing...</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                          <div className="bg-blue-500 h-1 rounded-full w-1/3 animate-pulse"></div>
                        </div>
                      </div>
                    )}

                    {/* Pending State with Animation */}
                    {task.status === 'pending' && (
                      <div className="mt-2">
                        <div className="flex items-center gap-2 text-xs text-yellow-400">
                          <div className="w-1.5 h-1.5 bg-yellow-400 rounded-full animate-pulse"></div>
                          <span>Queued for processing...</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-1 mt-1">
                          <div className="bg-yellow-500 h-1 rounded-full w-1/3 animate-pulse"></div>
                        </div>
                      </div>
                    )}

                    {/* Enhanced Duration and Status Info */}
                    <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
                      <div className="flex items-center gap-4">
                      <span>Created: {task.createdAt.toLocaleTimeString()}</span>
                      {task.startedAt && (
                        <span>Duration: {formatDuration(task.startedAt, task.completedAt)}</span>
                      )}
                    </div>
                      
                      {/* Task Type Badge */}
                      <div className="flex items-center gap-1">
                        <div className={`w-2 h-2 rounded-full ${
                          task.type === 'report_generation' ? 'bg-purple-400' :
                          task.type === 'query_execution' ? 'bg-blue-400' :
                          task.type === 'data_processing' ? 'bg-green-400' :
                          'bg-gray-400'
                        }`}></div>
                        <span className="text-xs text-gray-400 capitalize">
                          {task.type.replace('_', ' ')}
                        </span>
                      </div>
                    </div>

                    {/* Completion Celebration */}
                    {task.status === 'completed' && task.result && (
                      <div className="mt-2 p-2 bg-green-900/20 border border-green-500/30 rounded text-xs text-green-300">
                        <div className="flex items-center gap-2">
                          <div className="w-1 h-1 bg-green-400 rounded-full animate-pulse"></div>
                          <span>Report generated successfully!</span>
                          <div className="ml-auto flex gap-1">
                            <div className="w-1 h-1 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                            <div className="w-1 h-1 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                            <div className="w-1 h-1 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
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
                <div className="flex items-center gap-1 ml-2">
                  {task.status === 'completed' && task.result && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0 text-gray-400 hover:text-white"
                      onClick={() => {
                        // Navigate to report detail page using backend task ID if available
                        console.log('=== TASK MANAGER DEBUG ===');
                        console.log('Full task object:', task);
                        console.log('Task metadata:', task.metadata);
                        console.log('Backend task ID:', task.metadata?.backend_task_id);
                        console.log('Local task ID:', task.id);
                        console.log('Task status:', task.status);
                        console.log('Task result:', task.result);
                        
                        const backendTaskId = task.metadata?.backend_task_id;
                        const taskId = backendTaskId || task.id;
                        console.log('Using task ID for navigation:', taskId);
                        console.log('========================');
                        
                        window.location.href = `/ai-results/report/${taskId}`;
                      }}
                    >
                      <Eye className="w-3 h-3" />
                    </Button>
                  )}
                  
                  {(task.status === 'completed' || task.status === 'failed') && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0 text-gray-400 hover:text-red-400"
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
            className="w-full text-gray-400 border-gray-600 hover:bg-gray-800"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Clear All Completed
          </Button>
        </div>
      )}
    </div>
  );
}
