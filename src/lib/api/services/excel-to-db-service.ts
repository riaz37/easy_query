import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";
import {
  ExcelToDBHealthResponse,
  ExcelToDBPushDataRequest,
  ExcelToDBPushDataResponse,
  ExcelToDBGetAIMappingRequest,
  ExcelToDBGetAIMappingResponse,
} from "@/types/api";

/**
 * Service for Excel to Database operations
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class ExcelToDBService extends BaseService {
  protected readonly serviceName = 'ExcelToDBService';

  /**
   * Check health status of Excel to Database service
   */
  async checkHealth(): Promise<ServiceResponse<ExcelToDBHealthResponse>> {
    const response = await this.get<any>(API_ENDPOINTS.EXCEL_TO_DB_HEALTH);
    
    // Transform to expected format
    const healthResponse: ExcelToDBHealthResponse = {
      status: "success",
      message: "Health check completed",
      timestamp: new Date().toISOString(),
      ...response.data
    };

    return {
      data: healthResponse,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Push Excel data to database using provided column mapping
   * User ID is required for the API call
   */
  async pushDataToDatabase(
    request: ExcelToDBPushDataRequest
  ): Promise<ServiceResponse<ExcelToDBPushDataResponse>> {
    this.validateRequired(request, ['user_id', 'table_full_name', 'column_mapping', 'excel_file']);
    this.validateTypes(request, {
      table_full_name: 'string',
      skip_first_row: 'boolean',
    });

    if (request.table_full_name.trim().length === 0) {
      throw this.createValidationError('Table full name cannot be empty');
    }

    if (!request.column_mapping || typeof request.column_mapping !== 'object') {
      throw this.createValidationError('Column mapping is required and must be an object');
    }

    if (Object.keys(request.column_mapping).length === 0) {
      throw this.createValidationError('Column mapping cannot be empty');
    }

    if (!(request.excel_file instanceof File)) {
      throw this.createValidationError('Excel file is required and must be a File object');
    }

    // Validate file type
    const allowedTypes = [
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/csv',
    ];

    if (!allowedTypes.includes(request.excel_file.type)) {
      throw this.createValidationError('Invalid file type. Only Excel files (.xls, .xlsx) and CSV files are supported');
    }

    // Validate file size (e.g., 50MB limit to match component)
    const maxFileSize = 50 * 1024 * 1024; // 50MB
    if (request.excel_file.size > maxFileSize) {
      throw this.createValidationError('File size exceeds maximum limit of 50MB');
    }

    // Validate table name format - Updated to match actual API (can be just table name or schema.table_name)
    if (request.table_full_name.trim().length === 0) {
      throw this.createValidationError('Table full name cannot be empty');
    }

    // Validate column mapping
    this.validateColumnMapping(request.column_mapping);

    const formData = new FormData();
    formData.append('user_id', request.user_id);
    formData.append('table_full_name', request.table_full_name);
    formData.append('column_mapping', JSON.stringify(request.column_mapping));
    formData.append('skip_first_row', String(request.skip_first_row ?? true));
    formData.append('excel_file', request.excel_file);

    return this.post<ExcelToDBPushDataResponse>(
      API_ENDPOINTS.EXCEL_TO_DB_PUSH_DATA,
      formData,
      {
        headers: {
          // Don't set Content-Type for FormData - let browser set it with boundary
        },
      }
    );
  }

  /**
   * Get AI-powered column mapping suggestions
   * User ID is required for the API call
   */
  async getAIMapping(
    request: ExcelToDBGetAIMappingRequest
  ): Promise<ServiceResponse<ExcelToDBGetAIMappingResponse>> {
    this.validateRequired(request, ['user_id', 'table_full_name', 'excel_file']);
    this.validateTypes(request, { table_full_name: 'string' });

    if (request.table_full_name.trim().length === 0) {
      throw this.createValidationError('Table full name cannot be empty');
    }

    if (!(request.excel_file instanceof File)) {
      throw this.createValidationError('Excel file is required and must be a File object');
    }

    // Validate file type
    const allowedTypes = [
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/csv',
    ];

    if (!allowedTypes.includes(request.excel_file.type)) {
      throw this.createValidationError('Invalid file type. Only Excel files (.xls, .xlsx) and CSV files are supported');
    }

    // Validate table name format - Updated to match actual API (can be just table name or schema.table_name)
    if (request.table_full_name.trim().length === 0) {
      throw this.createValidationError('Table full name cannot be empty');
    }

    const formData = new FormData();
    formData.append('user_id', request.user_id);
    formData.append('table_full_name', request.table_full_name);
    formData.append('excel_file', request.excel_file);

    return this.post<ExcelToDBGetAIMappingResponse>(
      API_ENDPOINTS.EXCEL_TO_DB_GET_AI_MAPPING,
      formData,
      {
        headers: {
          // Don't set Content-Type for FormData - let browser set it with boundary
        },
      }
    );
  }

  /**
   * Validate Excel file format and content
   */
  async validateExcelFile(file: File): Promise<ServiceResponse<{
    isValid: boolean;
    errors: string[];
    warnings: string[];
    metadata: {
      fileName: string;
      fileSize: number;
      fileType: string;
      estimatedRows?: number;
      estimatedColumns?: number;
    };
  }>> {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Basic file validation
    if (!file) {
      errors.push('File is required');
      return {
        data: {
          isValid: false,
          errors,
          warnings,
          metadata: {
            fileName: '',
            fileSize: 0,
            fileType: '',
          },
        },
        success: true,
        timestamp: new Date().toISOString(),
      };
    }

    // File type validation
    const allowedTypes = [
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/csv',
    ];

    if (!allowedTypes.includes(file.type)) {
      errors.push('Invalid file type. Only Excel files (.xls, .xlsx) and CSV files are supported');
    }

    // File size validation
    const maxFileSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxFileSize) {
      errors.push('File size exceeds maximum limit of 10MB');
    }

    if (file.size === 0) {
      errors.push('File is empty');
    }

    // File name validation
    if (file.name.length > 255) {
      errors.push('File name is too long (maximum 255 characters)');
    }

    if (/[<>:"|?*]/.test(file.name)) {
      errors.push('File name contains invalid characters');
    }

    // Warnings
    if (file.size > 5 * 1024 * 1024) { // 5MB
      warnings.push('Large file detected. Processing may take longer');
    }

    return {
      data: {
        isValid: errors.length === 0,
        errors,
        warnings,
        metadata: {
          fileName: file.name,
          fileSize: file.size,
          fileType: file.type,
          // Actual row/column estimation would require parsing the file
          estimatedRows: undefined,
          estimatedColumns: undefined,
        },
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get supported file formats
   */
  getSupportedFormats(): ServiceResponse<{
    formats: Array<{
      extension: string;
      mimeType: string;
      description: string;
    }>;
    maxFileSize: number;
    recommendations: string[];
  }> {
    return {
      data: {
        formats: [
          {
            extension: '.xlsx',
            mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            description: 'Excel 2007+ format (recommended)',
          },
          {
            extension: '.xls',
            mimeType: 'application/vnd.ms-excel',
            description: 'Legacy Excel format',
          },
          {
            extension: '.csv',
            mimeType: 'text/csv',
            description: 'Comma-separated values',
          },
        ],
        maxFileSize: 10 * 1024 * 1024, // 10MB
        recommendations: [
          'Use .xlsx format for best compatibility',
          'Ensure column headers are in the first row',
          'Remove any merged cells or complex formatting',
          'Keep file size under 5MB for optimal performance',
          'Use consistent data types within each column',
        ],
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Validate column mapping structure
   */
  private validateColumnMapping(columnMapping: Record<string, string>): void {
    const errors: string[] = [];

    Object.entries(columnMapping).forEach(([excelColumn, dbColumn]) => {
      // Validate Excel column name
      if (!excelColumn || typeof excelColumn !== 'string' || excelColumn.trim().length === 0) {
        errors.push(`Invalid Excel column name: "${excelColumn}"`);
      }

      // Validate database column name
      if (!dbColumn || typeof dbColumn !== 'string' || dbColumn.trim().length === 0) {
        errors.push(`Invalid database column name: "${dbColumn}" for Excel column "${excelColumn}"`);
      }

      // Validate database column name format
      if (dbColumn && !/^[a-zA-Z][a-zA-Z0-9_]*$/.test(dbColumn)) {
        errors.push(`Database column name "${dbColumn}" must start with a letter and contain only letters, numbers, and underscores`);
      }
    });

    if (errors.length > 0) {
      throw this.createValidationError(`Column mapping validation failed: ${errors.join(', ')}`);
    }
  }

  /**
   * Preview Excel file data
   */
  async previewExcelData(file: File, maxRows: number = 10): Promise<ServiceResponse<{
    headers: string[];
    rows: string[][];
    totalRows: number;
    hasMoreRows: boolean;
  }>> {
    // This would typically involve parsing the Excel file
    // For now, return a placeholder response
    const validation = await this.validateExcelFile(file);
    
    if (!validation.data.isValid) {
      throw this.createValidationError(`File validation failed: ${validation.data.errors.join(', ')}`);
    }

    // In a real implementation, this would parse the Excel file
    // and return actual preview data
    return {
      data: {
        headers: [],
        rows: [],
        totalRows: 0,
        hasMoreRows: false,
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }
}

// Export singleton instance
export const excelToDBService = new ExcelToDBService();

// Export for backward compatibility
export default excelToDBService;