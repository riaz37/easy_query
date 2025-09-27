import { BaseService } from './base';
import { API_ENDPOINTS } from '../endpoints';
import {
  GenerateReportRequest,
  GenerateReportResponse,
  ReportTaskStatus,
  UpdateReportStructureRequest,
  ReportResults,
  ReportHistoryItem,
  ReportFilterOptions,
  ReportGenerationOptions
} from '@/types/reports';

/**
 * Service for managing report generation and management
 */
export class ReportService extends BaseService {
  protected readonly serviceName = 'ReportService';

  /**
   * Start a background report generation task
   */
  async generateReport(request: GenerateReportRequest): Promise<GenerateReportResponse> {
    const response = await this.post<GenerateReportResponse>(
      API_ENDPOINTS.GENERATE_REPORT_BACKGROUND,
      request
    );
    return response.data;
  }

  /**
   * Get the status of a report generation task
   */
  async getTaskStatus(taskId: string): Promise<ReportTaskStatus> {
    const response = await this.get<ReportTaskStatus>(
      API_ENDPOINTS.GET_REPORT_TASK_STATUS(taskId)
    );
    return response.data;
  }

  /**
   * Update the report structure for a specific config
   */
  async updateReportStructure(
    configId: number,
    request: UpdateReportStructureRequest
  ): Promise<void> {
    await this.put(
      API_ENDPOINTS.UPDATE_REPORT_STRUCTURE(configId),
      request
    );
  }

  /**
   * Update the report structure for a specific user (using user current DB endpoint)
   */
  async updateUserReportStructure(
    userId: string,
    request: UpdateReportStructureRequest
  ): Promise<void> {
    await this.put(
      API_ENDPOINTS.SET_USER_CURRENT_DB(userId),
      request
    );
  }

  /**
   * Get the current report structure for a specific user
   */
  async getReportStructure(userId: string): Promise<string> {
    const response = await this.get<{ 
      data: { 
        report_structure: string;
        user_id: string;
        db_id: number;
        business_rule: string;
        table_info: any;
        db_schema: any;
        created_at: string;
        updated_at: string;
      } 
    }>(
      API_ENDPOINTS.GET_USER_CURRENT_DB(userId)
    );
    return response.data.data.report_structure;
  }

  /**
   * Get report generation history with optional filtering
   */
  async getReportHistory(filters?: ReportFilterOptions): Promise<ReportHistoryItem[]> {
    const params = filters ? this.buildFilterParams(filters) : {};
    const response = await this.get<ReportHistoryItem[]>(
      API_ENDPOINTS.GET_REPORT_HISTORY,
      params
    );
    return response.data;
  }

  /**
   * Get user tasks with pagination
   */
  async getUserTasks(userId: string, limit: number = 5, offset: number = 0): Promise<{
    tasks: any[];
    total: number;
    hasMore: boolean;
  }> {
    const response = await this.get<{
      tasks: any[];
      total: number;
      has_more: boolean;
    }>(
      API_ENDPOINTS.GET_USER_TASKS(userId),
      { limit, offset }
    );
    return {
      tasks: response.data.tasks || [],
      total: response.data.total || 0,
      hasMore: response.data.has_more || false,
    };
  }

  /**
   * Delete a report task
   */
  async deleteReportTask(taskId: string): Promise<void> {
    await this.delete(API_ENDPOINTS.DELETE_REPORT_TASK(taskId));
  }

  /**
   * Monitor a report generation task with progress updates
   */
  async monitorReportTask(
    taskId: string,
    options: ReportGenerationOptions = {}
  ): Promise<ReportResults> {
    const {
      onProgress,
      onComplete,
      onError,
      pollInterval = 2000,
      timeout = 300000 // 5 minutes default
    } = options;

    const startTime = Date.now();
    let lastStatus: ReportTaskStatus | null = null;

    while (true) {
      try {
        // Check timeout
        if (Date.now() - startTime > timeout) {
          throw new Error('Report generation timeout exceeded');
        }

        const status = await this.getTaskStatus(taskId);
        
        // Call progress callback if status changed
        if (!lastStatus || lastStatus.status !== status.status) {
          onProgress?.(status);
          lastStatus = status;
        }

        // Check if completed
        if (status.status === 'completed' && status.results) {
          onComplete?.(status.results);
          return status.results;
        }

        // Check if failed
        if (status.status === 'failed') {
          const error = new Error(status.error || 'Report generation failed');
          onError?.(error);
          throw error;
        }

        // Wait before next poll
        await this.delay(pollInterval);
      } catch (error) {
        onError?.(error as Error);
        throw error;
      }
    }
  }

  /**
   * Generate a report and wait for completion
   */
  async generateReportAndWait(
    request: GenerateReportRequest,
    options: ReportGenerationOptions = {}
  ): Promise<ReportResults> {
    // Start the report generation
    const response = await this.generateReport(request);
    
    // Monitor the task until completion
    return this.monitorReportTask(response.task_id, options);
  }

  /**
   * Build filter parameters for report history
   */
  private buildFilterParams(filters: ReportFilterOptions): Record<string, any> {
    const params: Record<string, any> = {};
    
    if (filters.status) params.status = filters.status;
    if (filters.date_from) params.date_from = filters.date_from;
    if (filters.date_to) params.date_to = filters.date_to;
    if (filters.user_id) params.user_id = filters.user_id;
    
    return params;
  }

  /**
   * Utility method to delay execution
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get a formatted progress message for a task status
   */
  getProgressMessage(status: ReportTaskStatus): string {
    switch (status.status) {
      case 'pending':
        return 'Report generation queued...';
      case 'processing':
        return `Processing queries: ${status.processed_queries}/${status.total_queries} (${status.progress_percentage}%)`;
      case 'completed':
        return 'Report generation completed successfully!';
      case 'failed':
        return `Report generation failed: ${status.error || 'Unknown error'}`;
      default:
        return 'Unknown status';
    }
  }

  /**
   * Get estimated time remaining for a task
   */
  getEstimatedTimeRemaining(status: ReportTaskStatus): string | null {
    if (status.status === 'completed' || status.status === 'failed') {
      return null;
    }

    if (status.progress_percentage === 0) {
      return 'Calculating...';
    }

    if (status.processing_time_seconds && status.progress_percentage > 0) {
      const elapsedSeconds = status.processing_time_seconds;
      const progressRatio = status.progress_percentage / 100;
      const estimatedTotalSeconds = elapsedSeconds / progressRatio;
      const remainingSeconds = estimatedTotalSeconds - elapsedSeconds;
      
      if (remainingSeconds < 60) {
        return `${Math.round(remainingSeconds)}s`;
      } else if (remainingSeconds < 3600) {
        return `${Math.round(remainingSeconds / 60)}m`;
      } else {
        return `${Math.round(remainingSeconds / 3600)}h ${Math.round((remainingSeconds % 3600) / 60)}m`;
      }
    }

    return 'Calculating...';
  }
} 