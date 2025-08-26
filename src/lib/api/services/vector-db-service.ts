import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";

export interface VectorDBConfig {
  db_id: number;
  db_config: {
    schema: string;
    DB_HOST: string;
    DB_NAME: string;
    DB_PORT: number;
    DB_USER: string;
    [key: string]: any; // Allow other properties
  };
  created_at?: string;
  updated_at: string;
}

export interface UserTableNamesResponse {
  table_names: string[];
}

/**
 * Service for managing vector database operations
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class VectorDBService extends BaseService {
  protected readonly serviceName = 'VectorDBService';

  /**
   * Get all available vector database configurations for the authenticated user
   */
  async getVectorDBConfigs(): Promise<ServiceResponse<VectorDBConfig[]>> {
    const response = await this.get<any>(API_ENDPOINTS.GET_DATABASE_CONFIGS);
    
    // Extract the configs array from the nested response structure
    const configs = response.data?.configs || [];
    
    return {
      data: configs,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get available table names for the authenticated user
   * User ID is passed as a parameter to the endpoint
   */
  async getUserTableNames(userId: string): Promise<ServiceResponse<string[]>> {
    if (!userId) {
      throw this.createValidationError('userId is required');
    }
    
    const endpoint = API_ENDPOINTS.GET_USER_TABLE_NAMES_AUTH(userId);
    const response = await this.get<any>(endpoint);
    
    // Handle different response structures
    let tableNames: string[] = [];
    
    if (response.data && Array.isArray(response.data)) {
      tableNames = response.data;
    } else if (response.data && typeof response.data === 'object') {
      // Handle the case where response might have table_names property
      tableNames = response.data.table_names || [];
    }
    
    return {
      data: tableNames,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Create vector database access for the authenticated user
   * User ID is extracted from JWT token on backend
   */
  async createVectorDBAccess(request: {
    vector_db_id: number;
    accessible_tables: string[];
    access_level: string;
  }): Promise<ServiceResponse<any>> {
    this.validateRequired(request, ['vector_db_id', 'accessible_tables', 'access_level']);
    this.validateTypes(request, {
      vector_db_id: 'number',
      access_level: 'string',
    });

    if (!Array.isArray(request.accessible_tables)) {
      throw this.createValidationError('accessible_tables must be an array');
    }

    if (request.vector_db_id <= 0) {
      throw this.createValidationError('vector_db_id must be positive');
    }

    const requestBody = {
        access_type: "vector_db",
        vector_db_id: request.vector_db_id,
        accessible_tables: request.accessible_tables,
        access_level: request.access_level,
    };

    return this.post<any>(API_ENDPOINTS.CREATE_USER_CONFIG, requestBody);
  }

  /**
   * Get user configuration by database ID for the authenticated user
   * User ID is extracted from JWT token on backend
   */
  async getUserConfigByDB(dbId: number): Promise<ServiceResponse<any>> {
    this.validateRequired({ dbId }, ['dbId']);
    this.validateTypes({ dbId }, { dbId: 'number' });

    if (dbId <= 0) {
      throw this.createValidationError('Database ID must be positive');
    }

    return this.get<any>(API_ENDPOINTS.GET_USER_CONFIG_BY_DB_AUTH(dbId));
  }

  /**
   * Add table name for the authenticated user
   * User ID is required as parameter
   */
  async addUserTableName(tableName: string, userId: string): Promise<ServiceResponse<any>> {
    this.validateRequired({ tableName, userId }, ['tableName', 'userId']);
    this.validateTypes({ tableName, userId }, { tableName: 'string', userId: 'string' });

    if (tableName.trim().length === 0) {
      throw this.createValidationError('Table name cannot be empty');
    }

    if (userId.trim().length === 0) {
      throw this.createValidationError('User ID cannot be empty');
    }

    // Validate table name format
    if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(tableName)) {
      throw this.createValidationError('Table name must start with a letter and contain only letters, numbers, and underscores');
    }

    return this.post<any>(API_ENDPOINTS.ADD_USER_TABLE_NAME_AUTH(userId), {
        table_name: tableName,
      });
  }

  /**
   * Delete table name for the authenticated user
   * User ID is required as parameter
   */
  async deleteUserTableName(tableName: string, userId: string): Promise<ServiceResponse<any>> {
    this.validateRequired({ tableName, userId }, ['tableName', 'userId']);
    this.validateTypes({ tableName, userId }, { tableName: 'string', userId: 'string' });

    if (tableName.trim().length === 0) {
      throw this.createValidationError('Table name cannot be empty');
    }

    if (userId.trim().length === 0) {
      throw this.createValidationError('User ID cannot be empty');
    }

    return this.delete<any>(API_ENDPOINTS.DELETE_USER_TABLE_NAME_AUTH(userId, tableName));
  }

  /**
   * Bulk add multiple table names
   */
  async addMultipleTableNames(tableNames: string[], userId: string): Promise<ServiceResponse<{
    successful: string[];
    failed: Array<{ tableName: string; error: string }>;
  }>> {
    this.validateRequired({ tableNames, userId }, ['tableNames', 'userId']);

    if (!Array.isArray(tableNames)) {
      throw this.createValidationError('tableNames must be an array');
    }

    if (tableNames.length === 0) {
      throw this.createValidationError('tableNames array cannot be empty');
    }

    if (userId.trim().length === 0) {
      throw this.createValidationError('User ID cannot be empty');
    }

    const results = await Promise.allSettled(
      tableNames.map(tableName => this.addUserTableName(tableName, userId))
    );

    const successful: string[] = [];
    const failed: Array<{ tableName: string; error: string }> = [];

    results.forEach((result, index) => {
      const tableName = tableNames[index];
      if (result.status === 'fulfilled') {
        successful.push(tableName);
      } else {
        failed.push({
          tableName,
          error: result.reason?.message || 'Unknown error',
        });
      }
    });

    return {
      data: { successful, failed },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get user table names with metadata
   */
  async getUserTableNamesWithMetadata(userId: string): Promise<ServiceResponse<{
    tableNames: string[];
    count: number;
    lastUpdated: string;
  }>> {
    if (!userId) {
      throw this.createValidationError('userId is required');
    }
    
    const response = await this.getUserTableNames(userId);
    
    return {
      data: {
        tableNames: response.data,
        count: response.data.length,
        lastUpdated: new Date().toISOString(),
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Validate table name format
   */
  validateTableName(tableName: string): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!tableName || typeof tableName !== 'string') {
      errors.push('Table name must be a string');
      return { isValid: false, errors };
    }

    if (tableName.trim().length === 0) {
      errors.push('Table name cannot be empty');
    }

    if (tableName.length > 128) {
      errors.push('Table name cannot be longer than 128 characters');
    }

    if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(tableName)) {
      errors.push('Table name must start with a letter and contain only letters, numbers, and underscores');
    }

    // Check for reserved keywords
    const reservedKeywords = ['select', 'insert', 'update', 'delete', 'drop', 'create', 'alter', 'table'];
    if (reservedKeywords.includes(tableName.toLowerCase())) {
      errors.push('Table name cannot be a reserved SQL keyword');
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }
}

// Export singleton instance
export const vectorDBService = new VectorDBService();

// Export for backward compatibility
export default vectorDBService; 