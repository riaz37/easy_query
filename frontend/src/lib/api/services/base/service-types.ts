/**
 * Standard service operation result
 */
export interface ServiceResult<T = any> {
  data: T;
  success: boolean;
  error?: string;
  timestamp: string;
}

/**
 * Pagination parameters for service requests
 */
export interface ServicePaginationParams {
  page?: number;
  limit?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

/**
 * Standard service filters
 */
export interface ServiceFilters {
  [key: string]: any;
}

/**
 * Service request with pagination
 */
export interface ServicePaginatedRequest {
  pagination?: ServicePaginationParams;
  filters?: ServiceFilters;
}

/**
 * Service response with pagination
 */
export interface ServicePaginatedResponse<T = any> {
  data: T[];
  success: boolean;
  error?: string;
  timestamp: string;
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}

/**
 * Standard CRUD operations interface
 */
export interface CRUDServiceOperations<TEntity, TCreateRequest, TUpdateRequest> {
  create(request: TCreateRequest): Promise<ServiceResult<TEntity>>;
  getById(id: string | number): Promise<ServiceResult<TEntity>>;
  getAll(request?: ServicePaginatedRequest): Promise<ServicePaginatedResponse<TEntity>>;
  update(id: string | number, request: TUpdateRequest): Promise<ServiceResult<TEntity>>;
  delete(id: string | number): Promise<ServiceResult<void>>;
}

/**
 * Service validation result
 */
export interface ServiceValidationResult {
  isValid: boolean;
  errors: string[];
  warnings?: string[];
}

/**
 * Service operation context
 */
export interface ServiceOperationContext {
  operation: string;
  service: string;
  timestamp: string;
  userId?: string;
  metadata?: Record<string, any>;
}

/**
 * Service health check result
 */
export interface ServiceHealthResult {
  service: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  details?: {
    uptime?: number;
    responseTime?: number;
    dependencies?: Record<string, 'healthy' | 'unhealthy'>;
    errors?: string[];
  };
}

/**
 * Service configuration interface
 */
export interface ServiceConfig {
  timeout?: number;
  retries?: number;
  enableLogging?: boolean;
  enableMetrics?: boolean;
}

/**
 * Service metrics interface
 */
export interface ServiceMetrics {
  service: string;
  operation: string;
  timestamp: string;
  duration: number;
  success: boolean;
  statusCode?: number;
  errorType?: string;
}

/**
 * Standard service error types
 */
export enum ServiceErrorType {
  VALIDATION = 'validation',
  NOT_FOUND = 'not_found',
  AUTHENTICATION = 'authentication',
  AUTHORIZATION = 'authorization',
  NETWORK = 'network',
  SERVER = 'server',
  UNKNOWN = 'unknown',
}

/**
 * Service error details
 */
export interface ServiceErrorDetails {
  type: ServiceErrorType;
  message: string;
  statusCode?: number;
  context?: Record<string, any>;
  timestamp: string;
}

/**
 * Async operation status
 */
export interface AsyncOperationStatus {
  operationId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  result?: any;
  error?: string;
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
}

/**
 * Service batch operation request
 */
export interface ServiceBatchRequest<T> {
  operations: T[];
  options?: {
    continueOnError?: boolean;
    maxConcurrency?: number;
  };
}

/**
 * Service batch operation response
 */
export interface ServiceBatchResponse<T> {
  results: Array<ServiceResult<T> | { error: string }>;
  summary: {
    total: number;
    successful: number;
    failed: number;
  };
  timestamp: string;
} 