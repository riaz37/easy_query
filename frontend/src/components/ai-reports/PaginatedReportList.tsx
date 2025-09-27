"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthContext } from '@/components/providers';
import { ServiceRegistry } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  FileText, 
  Clock, 
  CheckCircle, 
  XCircle, 
  Eye, 
  Download,
  Calendar,
  User,
  Database,
  ChevronLeft,
  ChevronRight,
  RefreshCw
} from 'lucide-react';
import { formatDistanceToNow, format, isValid, parseISO } from 'date-fns';
import { EasyQueryBrandLoader, ReportListSkeleton } from '@/components/ui/loading';

interface ReportTask {
  task_id: string;
  user_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: string;
  current_step: string;
  total_queries: number;
  processed_queries: number;
  successful_queries: number;
  failed_queries: number;
  created_at: string;
  started_at: string;
  completed_at?: string;
  progress_percentage: number;
  results?: any;
  error?: string;
  user_query?: string;
}

interface PaginatedReportListProps {
  onViewReport: (taskId: string, results: any) => void;
  onDownloadReport: (taskId: string, results: any) => void;
}

// Utility function to format timestamps safely
const formatTimestamp = (timestamp: string | undefined | null) => {
  if (!timestamp) return { relative: 'Unknown', absolute: '', raw: 'No timestamp' };
  
  try {
    // Try to parse the timestamp
    const date = typeof timestamp === 'string' ? parseISO(timestamp) : new Date(timestamp);
    
    if (!isValid(date)) {
      console.warn('Invalid timestamp:', timestamp);
      return { relative: 'Invalid date', absolute: '', raw: timestamp };
    }
    
    return {
      relative: formatDistanceToNow(date, { addSuffix: true }),
      absolute: format(date, 'MMM dd, yyyy HH:mm'),
      raw: timestamp
    };
  } catch (error) {
    console.warn('Error formatting timestamp:', timestamp, error);
    return { relative: 'Invalid date', absolute: '', raw: timestamp };
  }
};

export function PaginatedReportList({ onViewReport, onDownloadReport }: PaginatedReportListProps) {
  const router = useRouter();
  const { user } = useAuthContext();
  const [tasks, setTasks] = useState<ReportTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [totalTasks, setTotalTasks] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);

  const limit = 20; // Load more tasks to ensure we have enough completed ones
  const offset = currentPage * limit;

  // Load tasks from backend
  const loadTasks = useCallback(async (page: number = 0) => {
    if (!user?.user_id) return;

    try {
      setLoading(true);
      setError(null);
      
      const response = await ServiceRegistry.reports.getUserTasks(
        user.user_id, 
        limit, 
        page * limit
      );
      
      setTasks(response.tasks);
      setTotalTasks(response.total);
      setHasMore(response.hasMore);
    } catch (err) {
      console.error('Failed to load tasks:', err);
      setError('Failed to load reports. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [user?.user_id, limit]);

  // Load tasks on mount and when user changes
  useEffect(() => {
    loadTasks(currentPage);
  }, [loadTasks, currentPage]);

  // Filter completed tasks for display
  const completedTasks = tasks.filter(task => task.status === 'completed');
  
  // Calculate pagination for completed tasks only
  const completedTasksPerPage = 5;
  const startIndex = currentPage * completedTasksPerPage;
  const endIndex = startIndex + completedTasksPerPage;
  const paginatedCompletedTasks = completedTasks.slice(startIndex, endIndex);
  const hasMoreCompleted = endIndex < completedTasks.length;

  // Refresh tasks
  const handleRefresh = useCallback(() => {
    loadTasks(currentPage);
  }, [loadTasks, currentPage]);

  // Pagination handlers
  const handleNextPage = useCallback(() => {
    if (hasMoreCompleted) {
      setCurrentPage(prev => prev + 1);
    }
  }, [hasMoreCompleted]);

  const handlePrevPage = useCallback(() => {
    if (currentPage > 0) {
      setCurrentPage(prev => prev - 1);
    }
  }, [currentPage]);

  const handleViewReport = (task: ReportTask) => {
    setSelectedTask(task.task_id);
    if (task.results) {
      onViewReport(task.task_id, task.results);
    } else {
      // Navigate to task detail page if no results in list
      router.push(`/ai-reports/report/${task.task_id}`);
    }
  };

  const handleDownloadReport = (task: ReportTask) => {
    if (task.results) {
      onDownloadReport(task.task_id, task.results);
    }
  };

  if (loading && tasks.length === 0) {
    return (
      <ReportListSkeleton 
        reportCount={5}
        showActions={true}
        showPagination={true}
        size="md"
      />
    );
  }

  if (error) {
    return (
      <Card className="bg-red-900/20 border-red-500/30">
        <CardContent className="p-8 text-center">
          <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <XCircle className="w-8 h-8 text-red-400" />
          </div>
          <h3 className="text-white text-lg font-medium mb-2">
            Error Loading Reports
          </h3>
          <p className="text-red-300 mb-4">{error}</p>
          <Button onClick={handleRefresh} variant="outline" className="border-red-500 text-red-300 hover:bg-red-900/20">
            <RefreshCw className="w-4 h-4 mr-2" />
            Try Again
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (completedTasks.length === 0) {
    return (
      <Card className="bg-gray-900/50 border-gray-700">
        <CardContent className="p-8 text-center">
          <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-blue-400" />
          </div>
          <h3 className="text-white text-lg font-medium mb-2">
            No Completed Reports
          </h3>
          <p className="text-gray-400 mb-4">
            Generate some reports from the Database Query page to see them here.
          </p>
          <Button onClick={handleRefresh} variant="outline" className="border-gray-600 text-gray-300 hover:bg-gray-700">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="modal-enhanced">
      <div 
        className="modal-content-enhanced overflow-hidden"
        style={{
          background: `linear-gradient(0deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.03)),
linear-gradient(230.27deg, rgba(19, 245, 132, 0) 71.59%, rgba(19, 245, 132, 0.2) 98.91%),
linear-gradient(67.9deg, rgba(19, 245, 132, 0) 66.65%, rgba(19, 245, 132, 0.2) 100%)`,
          backdropFilter: "blur(30px)"
        }}
      >

        {/* Reports List */}
        <div className="p-6">
          <div className="rounded-t-xl overflow-hidden">
            <table className="w-full">
              <thead>
                <tr 
                  style={{
                    background: "var(--components-Table-Head-filled, rgba(145, 158, 171, 0.08))",
                    borderRadius: "12px 12px 0 0"
                  }}
                >
                  <th className="px-6 py-4 text-left rounded-tl-xl text-white font-medium text-sm">
                    Report Title
                  </th>
                  <th className="px-6 py-4 text-left text-white font-medium text-sm">
                    Created
                  </th>
                  <th className="px-6 py-4 text-left text-white font-medium text-sm">
                    Queries
                  </th>
                  <th className="px-6 py-4 text-right text-white font-medium text-sm rounded-tr-xl">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  // Show skeleton rows while loading
                  Array.from({ length: 3 }).map((_, index) => (
                    <tr key={`loading-row-${index}`} className="border-b border-white/10">
                      <td className="px-6 py-4">
                        <div className="space-y-2">
                          <div 
                            className="h-5 rounded animate-pulse w-48"
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                          <div 
                            className="h-4 rounded animate-pulse w-32"
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="space-y-1">
                          <div 
                            className="h-4 rounded animate-pulse w-24"
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                          <div 
                            className="h-3 rounded animate-pulse w-16"
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div 
                          className="h-4 rounded animate-pulse w-8"
                          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                        />
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center gap-2 justify-end">
                          <div 
                            className="rounded-full animate-pulse h-8 w-20"
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                          <div 
                            className="rounded-full animate-pulse h-8 w-24"
                            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))
                ) : paginatedCompletedTasks.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="px-6 py-8 text-center text-white">
                      No completed reports found
                    </td>
                  </tr>
                ) : (
                  paginatedCompletedTasks.map((task) => (
                    <tr 
                      key={task.task_id} 
                      className="border-b border-white/10 hover:bg-white/5 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <div className="text-white font-medium">
                          {task.user_query ? task.user_query.substring(0, 50) + (task.user_query.length > 50 ? '...' : '') : 'Generated Report'}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-white">
                        <div className="text-sm">
                          {formatTimestamp(task.completed_at || task.created_at).absolute}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-white">
                        {task.total_queries || 0}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center gap-2 justify-end">
                          <Button
                            onClick={() => handleViewReport(task)}
                            variant="outline"
                            className="border-0 text-white hover:bg-emerald-400/10"
                            size="sm"
                            style={{ 
                              borderRadius: "99px", 
                              height: "32px", 
                              minWidth: "80px",
                              background: "rgba(255, 255, 255, 0.04)"
                            }}
                          >
                            <Eye className="w-4 h-4 mr-1" />
                            View
                          </Button>
                          {task.results && (
                            <Button
                              onClick={() => handleDownloadReport(task)}
                              variant="outline"
                              className="border-0 text-white hover:bg-emerald-400/10"
                              size="sm"
                              style={{ 
                                borderRadius: "99px", 
                                height: "32px", 
                                minWidth: "90px",
                                background: "rgba(255, 255, 255, 0.04)"
                              }}
                            >
                              Download
                              <img src="/tables/download.svg" alt="Download" className="w-4 h-4 ml-1" />
                            </Button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Pagination Controls */}
        <div className="px-6 py-4 flex items-center justify-between">
          {loading ? (
            // Pagination skeleton while loading
            <>
              <div className="space-y-1">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse w-48" />
              </div>
              <div className="flex items-center gap-2">
                <div className="bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse h-8 w-20" />
                <div className="bg-gray-200 dark:bg-gray-700 rounded animate-pulse h-6 w-16" />
                <div className="bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse h-8 w-16" />
              </div>
            </>
          ) : (
            <>
              <div className="text-sm text-gray-400">
                Showing {paginatedCompletedTasks.length} of {completedTasks.length} completed reports
              </div>
              
              <div className="flex items-center gap-2">
                <Button
                  onClick={handlePrevPage}
                  disabled={currentPage === 0 || loading}
                  variant="outline"
                  size="sm"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer disabled:opacity-50"
                  style={{ borderRadius: "999px" }}
                >
                  <ChevronLeft className="w-4 h-4 mr-1" />
                  Previous
                </Button>
                
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-400">Page {currentPage + 1}</span>
                </div>
                
                <Button
                  onClick={handleNextPage}
                  disabled={!hasMoreCompleted || loading}
                  variant="outline"
                  size="sm"
                  className="border-0 text-white hover:bg-white/10 cursor-pointer disabled:opacity-50"
                  style={{ borderRadius: "999px" }}
                >
                  Next
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
