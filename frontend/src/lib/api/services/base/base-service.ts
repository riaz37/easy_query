import { apiClient } from "../../client";
import { transformResponse, transformErrorResponse } from "../../transformers";
import { ApiResponse, ApiRequestConfig } from "@/types/api";
// Dynamic import to avoid SSR/client-side module resolution issues
type NetworkErrorType = {
  new (message: string, statusCode?: number, context?: Record<string, any>): Error & {
    statusCode?: number;
    context?: Record<string, any>;
  };
};

/**
 * Standard service response type
 */
export interface ServiceResponse<T = any> {
  data: T;
  success: boolean;
  error?: string;
  timestamp: string;
}

/**
 * Service error with context
 */
export class ServiceError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
    public readonly context?: Record<string, any>
  ) {
    super(message);
    this.name = 'ServiceError';
  }
}

/**
 * Base service class providing standardized patterns for all API services
 */
export abstract class BaseService {
  protected abstract readonly serviceName: string;

  /**
   * Make a standardized API request
   */
  protected async request<T = any>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH',
    endpoint: string,
    data?: any,
    config?: Partial<ApiRequestConfig>
  ): Promise<ServiceResponse<T>> {
    const startTime = Date.now();
    
    try {
      this.logRequest(method, endpoint, data);

      let response: any;
      
      switch (method) {
        case 'GET':
          response = await apiClient.get(endpoint, data, config);
          break;
        case 'POST':
          response = await apiClient.post(endpoint, data, config);
          break;
        case 'PUT':
          response = await apiClient.put(endpoint, data, config);
          break;
        case 'DELETE':
          response = await apiClient.delete(endpoint, config);
          break;
        case 'PATCH':
          response = await apiClient.patch(endpoint, data, config);
          break;
        default:
          throw new ServiceError(`Unsupported HTTP method: ${method}`);
      }

      const serviceResponse = this.transformSuccessResponse(response);
      
      this.logResponse(method, endpoint, serviceResponse, Date.now() - startTime);
      
      return serviceResponse;
    } catch (error) {
      const serviceError = this.handleError(error, method, endpoint);
      
      this.logError(method, endpoint, serviceError, Date.now() - startTime);
      
      throw serviceError;
    }
  }

  /**
   * GET request helper
   */
  protected async get<T = any>(
    endpoint: string,
    params?: Record<string, any>,
    config?: Partial<ApiRequestConfig>
  ): Promise<ServiceResponse<T>> {
    return this.request<T>('GET', endpoint, params, config);
  }

  /**
   * POST request helper
   */
  protected async post<T = any>(
    endpoint: string,
    data?: any,
    config?: Partial<ApiRequestConfig>
  ): Promise<ServiceResponse<T>> {
    return this.request<T>('POST', endpoint, data, config);
  }

  /**
   * PUT request helper
   */
  protected async put<T = any>(
    endpoint: string,
    data?: any,
    config?: Partial<ApiRequestConfig>
  ): Promise<ServiceResponse<T>> {
    return this.request<T>('PUT', endpoint, data, config);
  }

  /**
   * DELETE request helper
   */
  protected async delete<T = any>(
    endpoint: string,
    config?: Partial<ApiRequestConfig>
  ): Promise<ServiceResponse<T>> {
    return this.request<T>('DELETE', endpoint, undefined, config);
  }

  /**
   * PATCH request helper
   */
  protected async patch<T = any>(
    endpoint: string,
    data?: any,
    config?: Partial<ApiRequestConfig>
  ): Promise<ServiceResponse<T>> {
    return this.request<T>('PATCH', endpoint, data, config);
  }

  /**
   * Transform successful response to standard format
   */
  private transformSuccessResponse<T = any>(response: any): ServiceResponse<T> {
    return {
      data: response,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Handle and transform errors consistently
   */
  private handleError(
    error: any,
    method: string,
    endpoint: string
  ): ServiceError {
    // Handle NetworkError from API client - check for statusCode property instead of instanceof
    if (error && error.statusCode) {
      return new ServiceError(
        this.getErrorMessage(error),
        error.statusCode,
        {
          method,
          endpoint,
          service: this.serviceName,
          originalError: error.message,
        }
      );
    }

    // Handle standard errors
    if (error instanceof Error) {
      return new ServiceError(
        error.message,
        undefined,
        {
          method,
          endpoint,
          service: this.serviceName,
        }
      );
    }

    // Handle unknown errors
    return new ServiceError(
      'An unknown error occurred',
      undefined,
      {
        method,
        endpoint,
        service: this.serviceName,
        originalError: String(error),
      }
    );
  }

  /**
   * Extract meaningful error message from error object
   */
  private getErrorMessage(error: any): string {
    // Priority order for error messages
    if (error.message) return error.message;
    if (error.response?.data?.message) return error.response.data.message;
    if (error.response?.statusText) return error.response.statusText;
    if (error.statusCode) return `HTTP ${error.statusCode} error`;
    
    return 'An error occurred';
  }

  /**
   * Validate required parameters
   */
  protected validateRequired(
    params: Record<string, any>,
    requiredFields: string[]
  ): void {
    const missing = requiredFields.filter(field => 
      params[field] === undefined || params[field] === null
    );

    if (missing.length > 0) {
      throw new ServiceError(
        `Missing required parameters: ${missing.join(', ')}`,
        400,
        { missingFields: missing, service: this.serviceName }
      );
    }
  }

  /**
   * Validate parameter types
   */
  protected validateTypes(
    params: Record<string, any>,
    typeValidations: Record<string, string>
  ): void {
    const errors: string[] = [];

    Object.entries(typeValidations).forEach(([field, expectedType]) => {
      const value = params[field];
      if (value !== undefined && value !== null) {
        const actualType = typeof value;
        if (actualType !== expectedType) {
          errors.push(`${field} must be ${expectedType}, got ${actualType}`);
        }
      }
    });

    if (errors.length > 0) {
      throw new ServiceError(
        `Type validation failed: ${errors.join(', ')}`,
        400,
        { validationErrors: errors, service: this.serviceName }
      );
    }
  }

  /**
   * Log request (only in development)
   */
  private logRequest(method: string, endpoint: string, data?: any): void {
    if (process.env.NODE_ENV === 'development') {
      console.log(`[${this.serviceName}] ${method} ${endpoint}`, {
        data: data ? (typeof data === 'object' ? JSON.stringify(data, null, 2) : data) : undefined,
        timestamp: new Date().toISOString(),
      });
    }
  }

  /**
   * Log successful response (only in development)
   */
  private logResponse(
    method: string,
    endpoint: string,
    response: ServiceResponse,
    duration: number
  ): void {
    if (process.env.NODE_ENV === 'development') {
      console.log(`[${this.serviceName}] ${method} ${endpoint} - Success (${duration}ms)`, {
        dataSize: response.data ? JSON.stringify(response.data).length : 0,
        timestamp: response.timestamp,
      });
    }
  }

  /**
   * Log error (always logged)
   */
  private logError(
    method: string,
    endpoint: string,
    error: ServiceError,
    duration: number
  ): void {
    console.error(`[${this.serviceName}] ${method} ${endpoint} - Error (${duration}ms)`, {
      message: error.message,
      statusCode: error.statusCode,
      context: error.context,
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Create a standardized error for business logic validation
   */
  protected createValidationError(message: string, details?: Record<string, any>): ServiceError {
    return new ServiceError(message, 400, {
      type: 'validation',
      service: this.serviceName,
      ...details,
    });
  }

  /**
   * Create a standardized error for not found resources
   */
  protected createNotFoundError(resource: string, identifier?: string | number): ServiceError {
    const message = identifier 
      ? `${resource} with identifier '${identifier}' not found`
      : `${resource} not found`;
    
    return new ServiceError(message, 404, {
      type: 'not_found',
      resource,
      identifier,
      service: this.serviceName,
    });
  }

  /**
   * Create a standardized error for authentication issues
   */
  protected createAuthError(message: string = 'Authentication required'): ServiceError {
    return new ServiceError(message, 401, {
      type: 'authentication',
      service: this.serviceName,
    });
  }

  /**
   * Create a standardized error for authorization issues
   */
  protected createAuthorizationError(message: string = 'Access denied'): ServiceError {
    return new ServiceError(message, 403, {
      type: 'authorization',
      service: this.serviceName,
    });
  }
} 