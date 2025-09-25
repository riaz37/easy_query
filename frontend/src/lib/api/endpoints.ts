const baseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL || "https://176.9.16.194:8200";

/**
 * API endpoint definitions
 * All endpoints use JWT authentication - user ID is extracted from token on backend
 */
export const API_ENDPOINTS = {
  // Authentication endpoints
  AUTH_SIGNUP: `${baseUrl}/Auth/signup`,
  AUTH_LOGIN: `${baseUrl}/Auth/login`,
  AUTH_PROFILE: `${baseUrl}/Auth/profile`,
  AUTH_CHANGE_PASSWORD: `${baseUrl}/Auth/change-password`,

  // Query endpoints
  SEARCH: `${baseUrl}/search`,
  QUERY: `${baseUrl}/mssql/query`,

  // File system endpoints
  SMART_FILE_SYSTEM: `${baseUrl}/files/smart_file_system`,
  BUNDLE_TASK_STATUS: (bundleId: string) =>
    `${baseUrl}/files/bundle_task_status/${bundleId}`,
  BUNDLE_TASK_STATUS_ALL: `${baseUrl}/files/bundle_task_status_all`,
  FILES_SEARCH: `${baseUrl}/files/search`,

  // History endpoints (authenticated - user ID from JWT)
  CONVERSATION_HISTORY: (userId: string) => `${baseUrl}/mssql/conversation-history/${encodeURIComponent(userId)}`,
  CLEAR_HISTORY: (userId: string) => `${baseUrl}/mssql/clear-history/${encodeURIComponent(userId)}`,

  // Database management endpoints
  RELOAD_DB: `${baseUrl}/mssql/reload-db`,

  // MSSQL Configuration endpoints
  SET_MSSQL_CONFIG: `${baseUrl}/mssql-config/mssql-config/set-config`,
  GET_MSSQL_CONFIGS: `${baseUrl}/mssql-config/mssql-config`,
  GET_MSSQL_CONFIG: (id: number) =>
    `${baseUrl}/mssql-config/mssql-config/${id}`,
  UPDATE_MSSQL_CONFIG: (id: number) =>
    `${baseUrl}/mssql-config/mssql-config/update-config/${id}`,

  // Parent Companies endpoints
  CREATE_PARENT_COMPANY: `${baseUrl}/mssql-config/parent-companies`,
  GET_PARENT_COMPANIES: `${baseUrl}/mssql-config/parent-companies`,
  GET_PARENT_COMPANY: (id: number) =>
    `${baseUrl}/mssql-config/parent-companies/${id}`,

  // Sub Companies endpoints
  CREATE_SUB_COMPANY: `${baseUrl}/mssql-config/sub-companies`,
  GET_SUB_COMPANIES: `${baseUrl}/mssql-config/sub-companies`,
  GET_SUB_COMPANY: (id: number) =>
    `${baseUrl}/mssql-config/sub-companies/${id}`,

  // User Access endpoints
  CREATE_USER_ACCESS: `${baseUrl}/mssql-config/user-access`,
  GET_USER_ACCESS_CONFIGS: `${baseUrl}/mssql-config/user-access`,
  GET_USER_ACCESS: (userId: string) =>
    `${baseUrl}/mssql-config/user-access/${encodeURIComponent(userId)}`,

  // Database Configuration endpoints
  CREATE_DATABASE_CONFIG: `${baseUrl}/database-config`,
  GET_DATABASE_CONFIGS: `${baseUrl}/database-config`,
  GET_DATABASE_CONFIG: (id: number) => `${baseUrl}/database-config/${id}`,
  UPDATE_DATABASE_CONFIG: (id: number) => `${baseUrl}/database-config/${id}`,
  DELETE_DATABASE_CONFIG: (id: number) => `${baseUrl}/database-config/${id}`,

  // User Configuration endpoints (authenticated)
  CREATE_USER_CONFIG: `${baseUrl}/user-config`,
  GET_USER_CONFIGS: `${baseUrl}/user-config`,
  GET_USER_CONFIG: `${baseUrl}/user-config/me`, // Authenticated endpoint
  UPDATE_USER_CONFIG: `${baseUrl}/user-config/me`, // Authenticated endpoint

  // Vector Database endpoints (authenticated)
  GET_VECTOR_DB_CONFIGS: `${baseUrl}/user-config`,
  GET_USER_CONFIG_BY_DB_AUTH: (dbId: number) =>
    `${baseUrl}/user-config/by-db/${dbId}`, // Authenticated endpoint
  GET_CONFIG_BY_ID: (id: number) => `${baseUrl}/config/${id}`,
  UPDATE_USER_CONFIG_BY_ID: (id: number) => `${baseUrl}/user-config/${id}`,

  // User Table Names endpoints (authenticated)
  GET_USER_TABLE_NAMES_AUTH: (userId: string) => `${baseUrl}/user/${encodeURIComponent(userId)}/table-names`, // Authenticated endpoint with user ID
  ADD_USER_TABLE_NAME_AUTH: (userId: string) => `${baseUrl}/user/${encodeURIComponent(userId)}/table-names`, // Authenticated endpoint with user ID
  DELETE_USER_TABLE_NAME_AUTH: (userId: string, tableName: string) =>
    `${baseUrl}/user/${encodeURIComponent(userId)}/table-names/${encodeURIComponent(tableName)}`, // Authenticated endpoint with user ID

  // User Current Database endpoints (authenticated)
  SET_USER_CURRENT_DB: (userId: string) => `${baseUrl}/mssql-config/user-current-db/${encodeURIComponent(userId)}`, // PUT endpoint for authenticated user
  GET_USER_CURRENT_DB: (userId: string) => `${baseUrl}/mssql-config/user-current-db/${encodeURIComponent(userId)}/lite`, // GET endpoint for authenticated user

  // MSSQL Config Advanced Operations
  GENERATE_TABLE_INFO: `${baseUrl}/mssql-config/generate-table-info`,
  GET_TASK_STATUS: (taskId: string) =>
    `${baseUrl}/mssql-config/mssql-config/tasks/${taskId}`,

  // Excel to Database endpoints (authenticated)
  EXCEL_TO_DB_HEALTH: `${baseUrl}/excel-to-db/health`,
  EXCEL_TO_DB_PUSH_DATA: `${baseUrl}/excel-to-db/excel-to-db/push-data`,
  EXCEL_TO_DB_GET_AI_MAPPING: `${baseUrl}/excel-to-db/excel-to-db/get-ai-mapping`,

  // New Table Management endpoints (authenticated)
  NEW_TABLE_CREATE: `${baseUrl}/new-table/create`,
  NEW_TABLE_GET_DATA_TYPES: `${baseUrl}/new-table/data-types`,
  NEW_TABLE_GET_USER_TABLES: (userId: string) => `${baseUrl}/new-table/user-tables/${encodeURIComponent(userId)}`,
  NEW_TABLE_UPDATE_BUSINESS_RULE: (userId: string) => `${baseUrl}/new-table/user-business-rule/${encodeURIComponent(userId)}`,
  NEW_TABLE_GET_BUSINESS_RULE: (userId: string) => `${baseUrl}/new-table/user-business-rule/${encodeURIComponent(userId)}`,

  // Report Generation endpoints
  GENERATE_REPORT_BACKGROUND: `${baseUrl}/reports/generate-report-background`,
  GET_REPORT_TASK_STATUS: (taskId: string) => `${baseUrl}/reports/task-status/${taskId}`,
  UPDATE_REPORT_STRUCTURE: (id: number) => `${baseUrl}/mssql-config/mssql-config/${id}/report-structure`,
  GET_REPORT_HISTORY: `${baseUrl}/reports/history`,
  DELETE_REPORT_TASK: (taskId: string) => `${baseUrl}/reports/task/${taskId}`,
  GET_USER_TASKS: (userId: string) => `${baseUrl}/reports/user-tasks/${encodeURIComponent(userId)}`,
};

/**
 * Helper function to build endpoint URL with query parameters
 */
export function buildEndpointWithQueryParams(
  endpoint: string,
  params: Record<string, any>,
): string {
  const queryParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined) {
      queryParams.append(key, String(value));
    }
  });

  const queryString = queryParams.toString();

  if (!queryString) {
    return endpoint;
  }

  return `${endpoint}?${queryString}`;
}

/**
 * Helper function to build endpoint URL with path parameters
 */
export function buildEndpointWithPathParams(
  endpointTemplate: string,
  params: Record<string, string | number>,
): string {
  let endpoint = endpointTemplate;

  Object.entries(params).forEach(([key, value]) => {
    endpoint = endpoint.replace(`:${key}`, String(value));
  });

  return endpoint;
}
