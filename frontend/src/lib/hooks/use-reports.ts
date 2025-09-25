import { useState, useCallback, useRef, useEffect } from 'react';
import { ServiceRegistry } from '../api';
import {
  GenerateReportRequest,
  ReportTaskStatus,
  ReportResults,
  ReportGenerationOptions,
  UpdateReportStructureRequest,
  ReportHistoryItem,
  ReportFilterOptions
} from '../../types/reports';

interface UseReportsState {
  isGenerating: boolean;
  currentTask: ReportTaskStatus | null;
  reportResults: ReportResults | null;
  error: string | null;
  progress: number;
  estimatedTimeRemaining: string | null;
}

interface UseReportsReturn extends UseReportsState {
  // Report generation
  generateReport: (request: GenerateReportRequest, options?: ReportGenerationOptions) => Promise<string>;
  generateReportAndWait: (request: GenerateReportRequest, options?: ReportGenerationOptions) => Promise<ReportResults>;
  
  // Task monitoring
  startMonitoring: (taskId: string, options?: ReportGenerationOptions) => void;
  stopMonitoring: () => void;
  refreshTaskStatus: (taskId: string) => Promise<ReportTaskStatus>;
  
  // Report structure management
  updateReportStructure: (configId: number, structure: string) => Promise<void>;
  getReportStructure: (configId: number) => Promise<string>;
  
  // Report history
  getReportHistory: (filters?: ReportFilterOptions) => Promise<ReportHistoryItem[]>;
  
  // Task management
  deleteReportTask: (taskId: string) => Promise<void>;
  
  // Utility methods
  getProgressMessage: () => string;
  reset: () => void;
}

/**
 * Hook for managing report generation and monitoring
 */
export function useReports(): UseReportsReturn {
  const [state, setState] = useState<UseReportsState>({
    isGenerating: false,
    currentTask: null,
    reportResults: null,
    error: null,
    progress: 0,
    estimatedTimeRemaining: null,
  });

  const monitoringRef = useRef<{
    taskId: string | null;
    intervalId: NodeJS.Timeout | null;
    options: ReportGenerationOptions;
  }>({
    taskId: null,
    intervalId: null,
    options: {},
  });

  // Cleanup monitoring on unmount
  useEffect(() => {
    return () => {
      if (monitoringRef.current.intervalId) {
        clearInterval(monitoringRef.current.intervalId);
      }
    };
  }, []);

  /**
   * Generate a report in the background
   */
  const generateReport = useCallback(async (
    request: GenerateReportRequest,
    options: ReportGenerationOptions = {}
  ): Promise<string> => {
    try {
      setState(prev => ({
        ...prev,
        isGenerating: true,
        error: null,
        progress: 0,
      }));

      const response = await ServiceRegistry.reports.generateReport(request);
      
      setState(prev => ({
        ...prev,
        currentTask: {
          task_id: response.task_id,
          user_id: response.user_id,
          status: 'pending',
          progress: 'Report generation queued...',
          current_step: 'Queued',
          total_queries: 0,
          processed_queries: 0,
          successful_queries: 0,
          failed_queries: 0,
          created_at: response.timestamp,
          started_at: response.timestamp,
          progress_percentage: 0,
        },
      }));

      return response.task_id;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate report';
      setState(prev => ({
        ...prev,
        error: errorMessage,
        isGenerating: false,
      }));
      throw error;
    }
  }, []);

  /**
   * Generate a report and wait for completion
   */
  const generateReportAndWait = useCallback(async (
    request: GenerateReportRequest,
    options: ReportGenerationOptions = {}
  ): Promise<ReportResults> => {
    try {
      setState(prev => ({
        ...prev,
        isGenerating: true,
        error: null,
        progress: 0,
      }));

      const results = await ServiceRegistry.reports.generateReportAndWait(request, {
        ...options,
        onProgress: (status) => {
          setState(prev => ({
            ...prev,
            currentTask: status,
            progress: status.progress_percentage,
            estimatedTimeRemaining: ServiceRegistry.reports.getEstimatedTimeRemaining(status),
          }));
        },
        onComplete: (results) => {
          setState(prev => ({
            ...prev,
            isGenerating: false,
            reportResults: results,
            progress: 100,
            estimatedTimeRemaining: null,
          }));
        },
        onError: (error) => {
          setState(prev => ({
            ...prev,
            error: error.message,
            isGenerating: false,
            progress: 0,
          }));
        },
      });

      return results;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to generate report';
      setState(prev => ({
        ...prev,
        error: errorMessage,
        isGenerating: false,
        progress: 0,
      }));
      throw error;
    }
  }, []);

  /**
   * Start monitoring a report generation task
   */
  const startMonitoring = useCallback((
    taskId: string,
    options: ReportGenerationOptions = {}
  ) => {
    console.log('Starting monitoring for task:', taskId);
    
    // Stop any existing monitoring
    stopMonitoring();

    monitoringRef.current = {
      taskId,
      options,
      intervalId: null,
    };

    const pollInterval = options.pollInterval || 2000;
    console.log('Polling interval set to:', pollInterval, 'ms');

    const intervalId = setInterval(async () => {
      try {
        console.log('Polling task status for:', taskId);
        const status = await ServiceRegistry.reports.getTaskStatus(taskId);
        console.log('Task status received:', status);
        
        setState(prev => ({
          ...prev,
          currentTask: status,
          progress: status.progress_percentage,
          estimatedTimeRemaining: ServiceRegistry.reports.getEstimatedTimeRemaining(status),
        }));

        // Call progress callback
        options.onProgress?.(status);

        // Check if completed
        if (status.status === 'completed' && status.results) {
          console.log('Task completed successfully with results:', status.results);
          setState(prev => ({
            ...prev,
            isGenerating: false,
            reportResults: status.results,
            progress: 100,
            estimatedTimeRemaining: null,
          }));
          
          options.onComplete?.(status.results);
          stopMonitoring();
        }

        // Check if failed
        if (status.status === 'failed') {
          console.error('Task failed:', status.error);
          const error = new Error(status.error || 'Report generation failed');
          setState(prev => ({
            ...prev,
            error: error.message,
            isGenerating: false,
            progress: 0,
          }));
          
          options.onError?.(error);
          stopMonitoring();
        }
      } catch (error) {
        console.error('Error polling task status:', error);
        const errorMessage = error instanceof Error ? error.message : 'Failed to get task status';
        setState(prev => ({
          ...prev,
          error: errorMessage,
        }));
        
        options.onError?.(error as Error);
        stopMonitoring();
      }
    }, pollInterval);

    monitoringRef.current.intervalId = intervalId;
    console.log('Monitoring started successfully');
  }, []);

  /**
   * Stop monitoring the current task
   */
  const stopMonitoring = useCallback(() => {
    if (monitoringRef.current.intervalId) {
      clearInterval(monitoringRef.current.intervalId);
      monitoringRef.current.intervalId = null;
    }
    monitoringRef.current.taskId = null;
  }, []);

  /**
   * Refresh the status of a specific task
   */
  const refreshTaskStatus = useCallback(async (taskId: string): Promise<ReportTaskStatus> => {
    try {
      const status = await ServiceRegistry.reports.getTaskStatus(taskId);
      
      setState(prev => ({
        ...prev,
        currentTask: status,
        progress: status.progress_percentage,
        estimatedTimeRemaining: ServiceRegistry.reports.getEstimatedTimeRemaining(status),
      }));

      return status;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to refresh task status';
      setState(prev => ({
        ...prev,
        error: errorMessage,
      }));
      throw error;
    }
  }, []);

  /**
   * Update report structure
   */
  const updateReportStructure = useCallback(async (
    configId: number,
    structure: string
  ): Promise<void> => {
    try {
      const request: UpdateReportStructureRequest = { report_structure: structure };
      await ServiceRegistry.reports.updateReportStructure(configId, request);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to update report structure';
      setState(prev => ({
        ...prev,
        error: errorMessage,
      }));
      throw error;
    }
  }, []);

  /**
   * Get report structure
   */
  const getReportStructure = useCallback(async (configId: number): Promise<string> => {
    try {
      return await ServiceRegistry.reports.getReportStructure(configId);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get report structure';
      setState(prev => ({
        ...prev,
        error: errorMessage,
      }));
      throw error;
    }
  }, []);

  /**
   * Get report history
   */
  const getReportHistory = useCallback(async (filters?: ReportFilterOptions): Promise<ReportHistoryItem[]> => {
    try {
      return await ServiceRegistry.reports.getReportHistory(filters);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to get report history';
      setState(prev => ({
        ...prev,
        error: errorMessage,
      }));
      throw error;
    }
  }, []);

  /**
   * Delete a report task
   */
  const deleteReportTask = useCallback(async (taskId: string): Promise<void> => {
    try {
      await ServiceRegistry.reports.deleteReportTask(taskId);
      
      // If this was the current task, reset state
      if (state.currentTask?.task_id === taskId) {
        reset();
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete report task';
      setState(prev => ({
        ...prev,
        error: errorMessage,
      }));
      throw error;
    }
  }, [state.currentTask?.task_id]);

  /**
   * Get formatted progress message
   */
  const getProgressMessage = useCallback((): string => {
    if (!state.currentTask) return '';
    return ServiceRegistry.reports.getProgressMessage(state.currentTask);
  }, [state.currentTask]);

  /**
   * Reset the hook state
   */
  const reset = useCallback(() => {
    stopMonitoring();
    setState({
      isGenerating: false,
      currentTask: null,
      reportResults: null,
      error: null,
      progress: 0,
      estimatedTimeRemaining: null,
    });
  }, [stopMonitoring]);

  return {
    ...state,
    generateReport,
    generateReportAndWait,
    startMonitoring,
    stopMonitoring,
    refreshTaskStatus,
    updateReportStructure,
    getReportStructure,
    getReportHistory,
    deleteReportTask,
    getProgressMessage,
    reset,
  };
} 