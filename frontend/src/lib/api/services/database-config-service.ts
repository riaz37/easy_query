import { apiClient } from "../client";
import { API_ENDPOINTS } from "../endpoints";

/**
 * Database Configuration Types
 */
export interface DatabaseConfig {
  DB_HOST: string;
  DB_PORT: number;
  DB_NAME: string;
  DB_USER: string;
  DB_PASSWORD: string;
  schema: string;
  user_id?: string; // Optional user ID for user association
}

export interface DatabaseConfigData {
  db_id: number;
  db_config: DatabaseConfig;
  created_at: string;
  updated_at: string;
}

export interface CreateDatabaseConfigRequest {
  db_config: DatabaseConfig;
  user_id?: string; // Optional user ID for user association
}

export interface CreateDatabaseConfigResponse {
  status: string;
  message: string;
  data: DatabaseConfigData;
}

export interface GetDatabaseConfigsResponse {
  status: string;
  message: string;
  data: {
    configs: DatabaseConfigData[];
    count: number;
  };
}

export interface GetDatabaseConfigResponse {
  status: string;
  message: string;
  data: DatabaseConfigData;
}

export interface UpdateDatabaseConfigRequest {
  db_config: DatabaseConfig;
}

export interface UpdateDatabaseConfigResponse {
  status: string;
  message: string;
  data: {
    db_id: number;
    updated_config: DatabaseConfigData;
  };
}

export interface DeleteDatabaseConfigResponse {
  status: string;
  message: string;
  data: null;
}

/**
 * Service for managing database configurations
 */
export class DatabaseConfigService {
  /**
   * Create a new database configuration
   */
  static async createDatabaseConfig(
    request: CreateDatabaseConfigRequest,
  ): Promise<DatabaseConfigData> {
    try {
      const response = await apiClient.post(
        API_ENDPOINTS.CREATE_DATABASE_CONFIG,
        request,
      );
      return response; // API client interceptor already extracted .data
    } catch (error) {
      console.error("Error creating database configuration:", error);
      throw error;
    }
  }

  /**
   * Get all database configurations
   */
  static async getDatabaseConfigs(): Promise<{
    configs: DatabaseConfigData[];
    count: number;
  }> {
    try {
      const response = await apiClient.get(API_ENDPOINTS.GET_DATABASE_CONFIGS);
      return response; // API client interceptor already extracted .data
    } catch (error) {
      console.error("Error fetching database configurations:", error);
      throw error;
    }
  }

  /**
   * Get a specific database configuration by ID
   */
  static async getDatabaseConfig(id: number): Promise<DatabaseConfigData> {
    try {
      const response = await apiClient.get(
        API_ENDPOINTS.GET_DATABASE_CONFIG(id),
      );
      return response; // API client interceptor already extracted .data
    } catch (error) {
      console.error(`Error fetching database configuration ${id}:`, error);
      throw error;
    }
  }

  /**
   * Update an existing database configuration
   */
  static async updateDatabaseConfig(
    id: number,
    request: UpdateDatabaseConfigRequest,
  ): Promise<{ db_id: number; updated_config: DatabaseConfigData }> {
    try {
      const response = await apiClient.put(
        API_ENDPOINTS.UPDATE_DATABASE_CONFIG(id),
        request,
      );
      return response; // API client interceptor already extracted .data
    } catch (error) {
      console.error(`Error updating database configuration ${id}:`, error);
      throw error;
    }
  }

  /**
   * Delete a database configuration
   */
  static async deleteDatabaseConfig(id: number): Promise<null> {
    try {
      const response = await apiClient.delete(
        API_ENDPOINTS.DELETE_DATABASE_CONFIG(id),
      );
      return response; // API client interceptor already extracted .data (null in this case)
    } catch (error) {
      console.error(`Error deleting database configuration ${id}:`, error);
      throw error;
    }
  }

  /**
   * Validate database configuration request
   */
  static validateDatabaseConfig(config: DatabaseConfig): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!config.DB_HOST?.trim()) {
      errors.push("Database host is required");
    }

    if (!config.DB_PORT || config.DB_PORT <= 0 || config.DB_PORT > 65535) {
      errors.push("Database port must be a valid port number (1-65535)");
    }

    if (!config.DB_NAME?.trim()) {
      errors.push("Database name is required");
    }

    if (!config.DB_USER?.trim()) {
      errors.push("Database user is required");
    }

    if (!config.DB_PASSWORD?.trim()) {
      errors.push("Database password is required");
    }

    if (!config.schema?.trim()) {
      errors.push("Database schema is required");
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }
}

export default DatabaseConfigService;
