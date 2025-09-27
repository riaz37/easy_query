// Export API client
export { apiClient } from './client';
export type { ApiClient } from './client';

// Export endpoints
export { API_ENDPOINTS } from './endpoints';

// Export transformers
export {
  transformResponse,
  transformPaginatedResponse,
  transformRequest,
  transformErrorResponse,
  transformFileUploadData,
  transformQueryParams,
} from './transformers';

// Export base service components
export {
  BaseService,
  ServiceError,
} from './services/base/base-service';

export {
  ServiceRegistry,
  getServiceHealthStatus,
  ServiceMetricsCollector,
} from './services/service-registry';

export type {
  ServiceResponse,
  ServiceResult,
  ServicePaginationParams,
  ServiceFilters,
  ServicePaginatedRequest,
  ServicePaginatedResponse,
  CRUDServiceOperations,
  ServiceValidationResult,
  ServiceOperationContext,
  ServiceHealthResult,
  ServiceConfig,
  ServiceMetrics,
  ServiceErrorDetails,
  AsyncOperationStatus,
  ServiceBatchRequest,
  ServiceBatchResponse,
  ServiceErrorType,
} from './services/base';

// Export all standardized services
export {
  queryService,
  historyService,
  databaseService,
  businessRulesService,
  userCurrentDBService,
  vectorDBService,
  fileService,
  authService,
  excelToDBService,
  newTableService,
  userAccessService,
} from './services/service-registry';

// Export service interfaces and types
export type {
  SearchQueryParams,
  DbQueryParams,
  QueryResultData,
} from './services/query-service';

export type {
  ConversationHistoryItem,
} from './services/history-service';

export type {
  DatabaseInfo,
  DatabaseReloadResult,
} from './services/database-service';

export type {
  VectorDBConfig,
  UserTableNamesResponse,
} from './services/vector-db-service';

export type {
  LoginRequest,
  SignupRequest,
  ChangePasswordRequest,
  AuthResponse,
  UserProfile,
  JWTPayload,
} from './services/auth-service';

// Legacy service exports (for backward compatibility - all services are now standardized)
// These are kept for any code that might still import the class directly

// Legacy exports for backward compatibility (all services are now standardized)
export { default as QueryService } from './services/query-service';
export { default as HistoryService } from './services/history-service';
export { default as DatabaseService } from './services/database-service';
export { default as BusinessRulesService } from './services/business-rules-service';
export { default as UserCurrentDBService } from './services/user-current-db-service';
export { default as VectorDBService } from './services/vector-db-service';
export { default as FileService } from './services/file-service';
export { default as AuthService } from './services/auth-service';
export { default as ExcelToDBService } from './services/excel-to-db-service';
export { default as NewTableService } from './services/new-table-service';
export { default as UserAccessService } from './services/user-access-service';

// Export types
export type { ApiResponse, ApiRequestConfig, ApiError } from '@/types/api';
