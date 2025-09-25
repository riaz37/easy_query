import { apiClient } from "../client";
import { API_ENDPOINTS } from "../endpoints";

export interface MSSQLConfig {
  id: number;
  db_name: string;
  db_url: string;
  business_rule?: string;
  created_at?: string;
  updated_at?: string;
}

export interface MSSQLConfigsResponse {
  configs: MSSQLConfig[];
  count: number;
}

export interface TaskStatusResponse {
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  error?: string;
  result?: any;
}

export interface MSSQLConfigFormRequest {
  db_url: string;
  db_name: string;
  business_rule?: string;
  file?: File;
}

export interface MSSQLConfigTaskResponse {
  task_id: string;
  message: string;
}

/**
 * Service for managing MSSQL configurations
 */
export class MSSQLConfigService {
  /**
   * Get all MSSQL configurations (databases)
   */
  static async getMSSQLConfigs(): Promise<MSSQLConfig[]> {
    try {
      const response = await apiClient.get<MSSQLConfigsResponse>(
        API_ENDPOINTS.GET_MSSQL_CONFIGS
      );

      // With API client interceptor, response now contains just the data portion
      if (response && response.configs && Array.isArray(response.configs)) {
        return response.configs;
      } else if (Array.isArray(response)) {
        // If response is directly an array
        return response;
      } else {
        return [];
      }
    } catch (error: any) {
      console.error("Error fetching MSSQL configs:", error);
      throw new Error(
        error.response?.data?.message || "Failed to fetch MSSQL configurations"
      );
    }
  }

  /**
   * Get a specific MSSQL configuration by ID
   */
  static async getMSSQLConfig(id: number): Promise<MSSQLConfig> {
    try {
      const response = await apiClient.get<MSSQLConfig>(
        API_ENDPOINTS.GET_MSSQL_CONFIG(id)
      );

      return response;
    } catch (error: any) {
      console.error(`Error fetching MSSQL config ${id}:`, error);
      throw new Error(
        error.response?.data?.message || "Failed to fetch MSSQL configuration"
      );
    }
  }

  /**
   * Set MSSQL configuration (create new database)
   */
  static async setMSSQLConfig(config: MSSQLConfigFormRequest & { user_id: string }): Promise<MSSQLConfigTaskResponse> {
    try {
      console.log("Setting MSSQL config:", { ...config, file: config.file ? 'File present' : 'No file' });
      
      const formData = new FormData();
      formData.append("db_url", config.db_url);
      formData.append("db_name", config.db_name);
      formData.append("user_id", config.user_id);
      
      if (config.business_rule) {
        formData.append("business_rule", config.business_rule);
      }
      
      if (config.file) {
        formData.append("file", config.file);
      }

      console.log("FormData contents:");
      for (let [key, value] of formData.entries()) {
        console.log(`${key}:`, value);
      }

      console.log("Making request to:", API_ENDPOINTS.SET_MSSQL_CONFIG);
      
      const response = await apiClient.post<MSSQLConfigTaskResponse>(
        API_ENDPOINTS.SET_MSSQL_CONFIG,
        formData
      );

      console.log("Response received:", response);
      return response;
    } catch (error: any) {
      console.error("Error setting MSSQL config:", error);
      console.error("Error details:", {
        message: error.message,
        response: error.response,
        status: error.response?.status,
        data: error.response?.data
      });
      
      throw new Error(
        error.response?.data?.message || error.message || "Failed to set MSSQL configuration"
      );
    }
  }

  /**
   * Get task status by task ID
   */
  static async getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
    try {
      console.log("Getting task status for task ID:", taskId);
      console.log("Requesting from endpoint:", API_ENDPOINTS.GET_TASK_STATUS(taskId));
      
      const response = await apiClient.get<TaskStatusResponse>(
        API_ENDPOINTS.GET_TASK_STATUS(taskId)
      );
      
      console.log("Task status response received:", response);
      return response;
    } catch (error: any) {
      console.error(`Error fetching task status for ${taskId}:`, error);
      console.error("Error details:", {
        message: error.message,
        response: error.response,
        status: error.response?.status,
        data: error.response?.data
      });
      
      throw new Error(
        error.response?.data?.message || error.message || "Failed to fetch task status"
      );
    }
  }

  /**
   * Validate form configuration
   */
  static validateFormConfig(config: MSSQLConfigFormRequest & { user_id: string }): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!config.db_url?.trim()) {
      errors.push("Database URL is required");
    }

    if (!config.db_name?.trim()) {
      errors.push("Database name is required");
    }

    if (!config.user_id?.trim()) {
      errors.push("User ID is required");
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Update an MSSQL configuration
   */
  static async updateMSSQLConfig(
    id: number,
    updates: Partial<MSSQLConfig>
  ): Promise<MSSQLConfig> {
    try {
      const formData = new FormData();
      
      if (updates.db_name) formData.append("db_name", updates.db_name);
      if (updates.db_url) formData.append("db_url", updates.db_url);
      if (updates.business_rule !== undefined) formData.append("business_rule", updates.business_rule);

      const response = await apiClient.put(
        API_ENDPOINTS.UPDATE_MSSQL_CONFIG(id),
        formData
      );

      return response;
    } catch (error: any) {
      console.error(`Error updating MSSQL config ${id}:`, error);
      throw new Error(
        error.response?.data?.message || "Failed to update MSSQL configuration"
      );
    }
  }
}

export default MSSQLConfigService;
