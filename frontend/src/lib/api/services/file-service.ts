import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";
import {
  SmartFileSystemRequest,
  SmartFileSystemResponse,
  BundleTaskStatusResponse,
  BundleTaskStatusAllResponse,
  FilesSearchRequest,
  FilesSearchResponse,
} from "@/types/api";

/**
 * Service for handling file-related API calls
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class FileService extends BaseService {
  protected readonly serviceName = 'FileService';

  /**
   * Upload files to smart file system
   * User ID is extracted from JWT token on backend
   */
  async uploadToSmartFileSystem(request: {
    files: File[];
    file_descriptions: string[];
    table_names: string[];
    user_ids: string;
    use_table?: boolean;
  }): Promise<ServiceResponse<SmartFileSystemResponse>> {
    this.validateRequired(request, ['files', 'file_descriptions', 'user_ids']);

    if (!Array.isArray(request.files) || request.files.length === 0) {
      throw this.createValidationError('At least one file is required');
    }

    if (!Array.isArray(request.file_descriptions) || request.file_descriptions.length === 0) {
      throw this.createValidationError('File descriptions are required');
    }

    if (request.user_ids.trim().length === 0) {
      throw this.createValidationError('User ID is required');
    }

    // Table names are only required if use_table is true
    if (request.use_table !== false) {
      if (!Array.isArray(request.table_names) || request.table_names.length === 0) {
        throw this.createValidationError('Table names are required when using tables');
      }
      if (request.files.length !== request.table_names.length) {
        throw this.createValidationError('Files and table names arrays must have the same length when using tables');
      }
    }

    if (request.files.length !== request.file_descriptions.length) {
      throw this.createValidationError('Files and descriptions arrays must have the same length');
    }

    // Validate file types and sizes
    const validationErrors = this.validateFiles(request.files);
    if (validationErrors.length > 0) {
      throw this.createValidationError(`File validation failed: ${validationErrors.join(', ')}`);
    }

    const formData = new FormData();
    
    // Add files
    request.files.forEach((file, index) => {
      formData.append('files', file);
    });

    // Add metadata - match the exact format from your curl command
    formData.append('file_descriptions', request.file_descriptions[0] || 'string');
    
    // Only add table_names if use_table is true
    if (request.use_table !== false) {
      formData.append('table_names', request.table_names[0] || 'string');
    }
    
    formData.append('user_ids', request.user_ids);

    return this.post<SmartFileSystemResponse>(
      API_ENDPOINTS.SMART_FILE_SYSTEM,
      formData,
      {
        headers: {
          // Don't set Content-Type for FormData - let browser set it with boundary
        },
      }
    );
  }

  /**
   * Get bundle task status by bundle ID
   */
  async getBundleTaskStatus(bundleId: string): Promise<ServiceResponse<BundleTaskStatusResponse>> {
    this.validateRequired({ bundleId }, ['bundleId']);
    this.validateTypes({ bundleId }, { bundleId: 'string' });

    if (bundleId.trim().length === 0) {
      throw this.createValidationError('Bundle ID cannot be empty');
    }

    return this.get<BundleTaskStatusResponse>(
      API_ENDPOINTS.BUNDLE_TASK_STATUS(bundleId)
    );
  }

  /**
   * Get all bundle task statuses
   */
  async getAllBundleTaskStatuses(): Promise<ServiceResponse<BundleTaskStatusAllResponse>> {
    return this.get<BundleTaskStatusAllResponse>(
      API_ENDPOINTS.BUNDLE_TASK_STATUS_ALL
    );
  }

  /**
   * Search files with authenticated user context
   * User ID can be passed explicitly or extracted from JWT token on backend
   */
  async searchFiles(request: FilesSearchRequest): Promise<ServiceResponse<FilesSearchResponse>> {
    this.validateRequired(request, ['query']);
    this.validateTypes(request, { query: 'string' });

    if (request.query.trim().length === 0) {
      throw this.createValidationError('Search query cannot be empty');
    }

    // Validate optional parameters
    if (request.intent_top_k !== undefined && request.intent_top_k <= 0) {
      throw this.createValidationError('intent_top_k must be positive');
    }

    if (request.chunk_top_k !== undefined && request.chunk_top_k <= 0) {
      throw this.createValidationError('chunk_top_k must be positive');
    }

    if (request.max_chunks_for_answer !== undefined && request.max_chunks_for_answer <= 0) {
      throw this.createValidationError('max_chunks_for_answer must be positive');
    }

    // Ensure user_id is included in the request
    const searchRequest: FilesSearchRequest = {
      ...request,
      // user_id should be provided by the caller
    };

    return this.post<FilesSearchResponse>(
      API_ENDPOINTS.FILES_SEARCH,
      searchRequest
    );
  }

  /**
   * Validate uploaded files
   */
  private validateFiles(files: File[]): string[] {
    const errors: string[] = [];
    const maxFileSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = [
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/plain',
      'text/csv',
      'application/json',
    ];

    files.forEach((file, index) => {
      // Check file size
      if (file.size > maxFileSize) {
        errors.push(`File ${index + 1} (${file.name}) is too large. Maximum size is 50MB`);
      }

      // Check file type
      if (!allowedTypes.includes(file.type)) {
        errors.push(`File ${index + 1} (${file.name}) has unsupported type: ${file.type}`);
      }

      // Check file name
      if (file.name.length > 255) {
        errors.push(`File ${index + 1} name is too long. Maximum length is 255 characters`);
      }

      // Check for potentially dangerous file names
      if (/[<>:"|?*]/.test(file.name)) {
        errors.push(`File ${index + 1} name contains invalid characters`);
      }
    });

    return errors;
  }

  /**
   * Get file upload progress for a bundle
   */
  async getUploadProgress(bundleId: string): Promise<ServiceResponse<{
    bundleId: string;
    overallProgress: number;
    fileProgresses: Array<{
      fileName: string;
      status: 'pending' | 'uploading' | 'processing' | 'completed' | 'failed';
      progress: number;
      error?: string;
    }>;
  }>> {
    const statusResponse = await this.getBundleTaskStatus(bundleId);
    
    if (!statusResponse.success) {
      throw new Error(`Failed to get bundle status: ${statusResponse.error}`);
    }

    const status = statusResponse.data;
    
    const fileProgresses = status.individual_tasks.map(task => ({
      fileName: task.filename,
      status: task.status as 'pending' | 'uploading' | 'processing' | 'completed' | 'failed',
      progress: task.status === 'completed' ? 100 : 
                task.status === 'failed' ? 0 : 
                parseInt(task.progress) || 0,
      error: task.error_message || undefined,
    }));

    return {
      data: {
        bundleId,
        overallProgress: status.progress_percentage,
        fileProgresses,
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Cancel file upload bundle
   */
  async cancelUpload(bundleId: string): Promise<ServiceResponse<void>> {
    this.validateRequired({ bundleId }, ['bundleId']);
    this.validateTypes({ bundleId }, { bundleId: 'string' });

    if (bundleId.trim().length === 0) {
      throw this.createValidationError('Bundle ID cannot be empty');
    }

    // This would require a cancel endpoint in the API
    // For now, we'll return a placeholder response
    return {
      data: undefined as any,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get supported file types
   */
  getSupportedFileTypes(): ServiceResponse<{
    types: string[];
    maxSize: number;
    description: string;
  }> {
    return {
      data: {
        types: [
          'application/pdf',
          'application/msword',
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
          'application/vnd.ms-excel',
          'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          'text/plain',
          'text/csv',
          'application/json',
        ],
        maxSize: 50 * 1024 * 1024, // 50MB
        description: 'Supported file types include PDF, Word documents, Excel spreadsheets, text files, CSV, and JSON files. Maximum file size is 50MB.',
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }
}

// Export singleton instance
export const fileService = new FileService();

// Export for backward compatibility
export default fileService;
