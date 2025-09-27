import { apiClient } from "../client";
import { API_ENDPOINTS } from "../endpoints";
import {
  UserConfigCreateRequest,
  UserConfigCreateResponse,
  UserConfigResponse,
  UserConfigsListResponse,
  UserConfigByDbResponse,
  UserConfigUpdateRequest,
  UserConfigUpdateResponse,
  AddUserTableNameRequest,
  UserTableNameActionResponse,
  GetUserTableNamesResponse,
} from "@/types/api";

/**
 * Service for managing user configuration operations
 */
export class UserConfigService {
  /**
   * Create a new user configuration
   */
  static async createUserConfig(
    request: UserConfigCreateRequest
  ): Promise<UserConfigCreateResponse> {
    try {
      const response = await apiClient.post(
        API_ENDPOINTS.CREATE_USER_CONFIG,
        request
      );
      return response;
    } catch (error) {
      console.error("Error creating user configuration:", error);
      throw error;
    }
  }

  /**
   * Get all user configurations
   */
  static async getUserConfigs(): Promise<UserConfigsListResponse> {
    try {
      const response = await apiClient.get(API_ENDPOINTS.GET_USER_CONFIGS);
      return response;
    } catch (error) {
      console.error("Error fetching user configurations:", error);
      throw error;
    }
  }

  /**
   * Get user configuration by user ID
   */
  static async getUserConfig(userId: string): Promise<UserConfigResponse> {
    try {
      const response = await apiClient.get(
        API_ENDPOINTS.GET_USER_CONFIG(userId)
      );
      return response;
    } catch (error) {
      console.error(`Error fetching user configuration for ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Get user configuration by user ID and database ID
   */
  static async getUserConfigByDb(
    userId: string,
    dbId: number
  ): Promise<UserConfigByDbResponse> {
    try {
      const response = await apiClient.get(
        API_ENDPOINTS.GET_USER_CONFIG_BY_DB(userId, dbId)
      );
      return response;
    } catch (error) {
      console.error(
        `Error fetching user configuration for ${userId} with db ${dbId}:`,
        error
      );
      throw error;
    }
  }

  /**
   * Get configuration by config ID
   */
  static async getConfigById(id: number): Promise<UserConfigResponse> {
    try {
      const response = await apiClient.get(API_ENDPOINTS.GET_CONFIG_BY_ID(id));
      return response;
    } catch (error) {
      console.error(`Error fetching configuration ${id}:`, error);
      throw error;
    }
  }

  /**
   * Update user configuration
   */
  static async updateUserConfig(
    id: number,
    request: UserConfigUpdateRequest
  ): Promise<UserConfigUpdateResponse> {
    try {
      const response = await apiClient.put(
        API_ENDPOINTS.UPDATE_USER_CONFIG(id),
        request
      );
      return response;
    } catch (error) {
      console.error(`Error updating user configuration ${id}:`, error);
      throw error;
    }
  }

  /**
   * Add table name for user
   */
  static async addUserTableName(
    userId: string,
    request: AddUserTableNameRequest
  ): Promise<UserTableNameActionResponse> {
    try {
      const response = await apiClient.post(
        API_ENDPOINTS.ADD_USER_TABLE_NAME(userId),
        request
      );
      return response;
    } catch (error) {
      console.error(`Error adding table name for user ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Get table names for user
   */
  static async getUserTableNames(
    userId: string
  ): Promise<GetUserTableNamesResponse> {
    try {
      const response = await apiClient.get(
        API_ENDPOINTS.GET_USER_TABLE_NAMES(userId)
      );
      return response;
    } catch (error) {
      console.error(`Error fetching table names for user ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Delete table name for user
   */
  static async deleteUserTableName(
    userId: string,
    tableName: string
  ): Promise<UserTableNameActionResponse> {
    try {
      const response = await apiClient.delete(
        API_ENDPOINTS.DELETE_USER_TABLE_NAME(userId, tableName)
      );
      return response;
    } catch (error) {
      console.error(
        `Error deleting table name ${tableName} for user ${userId}:`,
        error
      );
      throw error;
    }
  }

  /**
   * Validate user config create request
   */
  static validateCreateRequest(request: UserConfigCreateRequest): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!request.user_id || request.user_id.trim() === "") {
      errors.push("User ID is required");
    }

    if (!request.db_id || request.db_id <= 0) {
      errors.push("Database ID is required and must be a positive number");
    }

    if (
      request.access_level === undefined ||
      request.access_level < 0 ||
      request.access_level > 10
    ) {
      errors.push("Access level must be between 0 and 10");
    }

    if (!Array.isArray(request.accessible_tables)) {
      errors.push("Accessible tables must be an array");
    }

    if (!Array.isArray(request.table_names)) {
      errors.push("Table names must be an array");
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate user config update request
   */
  static validateUpdateRequest(request: UserConfigUpdateRequest): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!request.db_id || request.db_id <= 0) {
      errors.push("Database ID is required and must be a positive number");
    }

    if (
      request.access_level === undefined ||
      request.access_level < 0 ||
      request.access_level > 10
    ) {
      errors.push("Access level must be between 0 and 10");
    }

    if (!Array.isArray(request.accessible_tables)) {
      errors.push("Accessible tables must be an array");
    }

    if (!Array.isArray(request.table_names)) {
      errors.push("Table names must be an array");
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate add table name request
   */
  static validateAddTableNameRequest(request: AddUserTableNameRequest): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!request.table_name || request.table_name.trim() === "") {
      errors.push("Table name is required");
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }
}

export default UserConfigService;
