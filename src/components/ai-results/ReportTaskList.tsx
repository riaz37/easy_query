"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useTaskStore } from '@/store/task-store';
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
  Database
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface ReportTaskListProps {
  onViewReport: (taskId: string, results: any) => void;
  onDownloadReport: (taskId: string, results: any) => void;
}

export function ReportTaskList({ onViewReport, onDownloadReport }: ReportTaskListProps) {
  const router = useRouter();
  const { completedTasks, getTasksByType } = useTaskStore();
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  
  // Get all completed report generation tasks
  const reportTasks = getTasksByType('report_generation')
    .filter(task => task.status === 'completed' && task.result)
    .sort((a, b) => (b.completedAt?.getTime() || 0) - (a.completedAt?.getTime() || 0));

  const handleViewReport = (task: any) => {
    setSelectedTask(task.id);
    // Navigate to a new page with the report details
    router.push(`/ai-results/report/${task.id}`);
  };

  const handleDownloadReport = (task: any) => {
    onDownloadReport(task.id, task.result);
  };

  if (reportTasks.length === 0) {
    return (
      <Card className="bg-gray-900/50 border-gray-700">
        <CardContent className="p-8 text-center">
          <div className="w-16 h-16 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-blue-400" />
          </div>
          <h3 className="text-white text-lg font-medium mb-2">
            No Completed Reports
          </h3>
          <p className="text-gray-400">
            Generate some reports from the Database Query page to see them here.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">
          Completed Reports ({reportTasks.length})
        </h2>
        <Badge variant="outline" className="text-emerald-400 border-emerald-400">
          {reportTasks.length} reports ready
        </Badge>
      </div>

      <div className="grid gap-4">
        {reportTasks.map((task) => (
          <Card 
            key={task.id} 
            className={`bg-gray-900/50 border-gray-700 hover:border-gray-600 transition-colors ${
              selectedTask === task.id ? 'ring-2 ring-blue-500' : ''
            }`}
          >
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <CardTitle className="text-white text-lg mb-2">
                    {task.title}
                  </CardTitle>
                  <p className="text-gray-400 text-sm mb-3">
                    {task.description}
                  </p>
                  
                  {/* Task Metadata */}
                  <div className="flex flex-wrap gap-4 text-xs text-gray-500">
                    <div className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      {task.completedAt && formatDistanceToNow(task.completedAt, { addSuffix: true })}
                    </div>
                    <div className="flex items-center gap-1">
                      <User className="w-3 h-3" />
                      {task.metadata?.userId ? task.metadata.userId.substring(0, 8) + '...' : 'Unknown'}
                    </div>
                    {task.metadata?.selected_structure && (
                      <div className="flex items-center gap-1">
                        <Database className="w-3 h-3" />
                        {task.metadata.selected_structure}
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="flex items-center gap-2 ml-4">
                  <Badge variant="outline" className="text-emerald-400 border-emerald-400">
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Completed
                  </Badge>
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="pt-0">
              {/* Report Summary */}
              {task.result && (
                <div className="bg-gray-800/50 rounded-lg p-4 mb-4">
                  <h4 className="text-white font-medium mb-2">Report Summary</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Total Queries:</span>
                      <span className="text-white ml-2">{task.result.total_queries || 0}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Successful:</span>
                      <span className="text-emerald-400 ml-2">{task.result.successful_queries || 0}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Failed:</span>
                      <span className="text-red-400 ml-2">{task.result.failed_queries || 0}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Sections:</span>
                      <span className="text-white ml-2">{task.result.results?.length || 0}</span>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Action Buttons */}
              <div className="flex gap-2">
                <Button
                  onClick={() => handleViewReport(task)}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                  size="sm"
                >
                  <Eye className="w-4 h-4 mr-2" />
                  View Report
                </Button>
                <Button
                  onClick={() => handleDownloadReport(task)}
                  variant="outline"
                  className="border-gray-600 text-gray-300 hover:bg-gray-700"
                  size="sm"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
