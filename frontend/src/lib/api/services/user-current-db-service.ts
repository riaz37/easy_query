import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";
import { UserCurrentDBRequest, UserCurrentDBResponse } from "@/types/api";

/**
 * Service for managing user's current database configuration
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class UserCurrentDBService extends BaseService {
  protected readonly serviceName = 'UserCurrentDBService';

  /**
   * Set the current database for the authenticated user
   * User ID is extracted from JWT token on backend
   */
  async setUserCurrentDB(
    request: UserCurrentDBRequest,
    userId?: string
  ): Promise<ServiceResponse<UserCurrentDBResponse>> {
    // Validate request
    const validation = this.validateRequest(request);
    if (!validation.isValid) {
      throw this.createValidationError(`Invalid request: ${validation.errors.join(", ")}`);
    }

    // Use the updated endpoint that accepts userId parameter
    const endpoint = API_ENDPOINTS.SET_USER_CURRENT_DB(userId || '');

    // Use PUT method as per the API specification
    return this.put<UserCurrentDBResponse>(endpoint, request);
  }

  /**
   * Get the current database for the authenticated user
   * User ID is extracted from JWT token on backend
   */
  async getUserCurrentDB(userId?: string): Promise<ServiceResponse<UserCurrentDBResponse>> {
    // Use the updated endpoint that accepts userId parameter
    const endpoint = API_ENDPOINTS.GET_USER_CURRENT_DB(userId || '');

    return this.get<UserCurrentDBResponse>(endpoint);
  }

  /**
   * Validate user current database request
   */
  validateRequest(request: UserCurrentDBRequest): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!request.db_id || request.db_id <= 0) {
      errors.push("Database ID is required and must be positive");
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Check if user has a current database set
   */
  async hasCurrentDatabase(userId?: string): Promise<ServiceResponse<boolean>> {
    try {
      const response = await this.getUserCurrentDB(userId);
      const hasDatabase = !!(response.data && response.data.db_id);
      
      return {
        data: hasDatabase,
        success: true,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        data: false,
        success: true,
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Get current database with additional metadata
   */
  async getCurrentDatabaseInfo(userId?: string): Promise<ServiceResponse<{
    database: UserCurrentDBResponse;
    isConfigured: boolean;
    hasBusinessRules: boolean;
    tableCount: number;
  }>> {
    try {
      const dbResponse = await this.getUserCurrentDB(userId);
      
      if (!dbResponse.data) {
        return {
          data: {
            database: {} as UserCurrentDBResponse,
            isConfigured: false,
            hasBusinessRules: false,
            tableCount: 0,
          },
          success: true,
          timestamp: new Date().toISOString(),
        };
      }

      const database = dbResponse.data;
      const hasBusinessRules = !!(database.business_rule && database.business_rule.trim().length > 0);
      const tableCount = database.table_info?.tables?.length || 0;

      return {
        data: {
          database,
          isConfigured: true,
          hasBusinessRules,
          tableCount,
        },
        success: true,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      throw error;
    }
  }
}

// Export singleton instance
export const userCurrentDBService = new UserCurrentDBService();

// Export for backward compatibility
export default userCurrentDBService;
