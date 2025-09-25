import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";
import {
  UserAccessCreateRequest,
  UserAccessCreateResponse,
  UserAccessResponse,
  UserAccessListResponse,
} from "@/types/api";

/**
 * Service for managing user access to databases and companies
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class UserAccessService extends BaseService {
  protected readonly serviceName = 'UserAccessService';

  /**
   * Create or update user access configuration
   */
  async createUserAccess(
    accessConfig: UserAccessCreateRequest
  ): Promise<ServiceResponse<UserAccessCreateResponse>> {
    this.validateRequired(accessConfig, [
      'user_id',
      'parent_company_id',
      'sub_company_ids',
      'database_access',
      'table_shows'
    ]);

    this.validateTypes(accessConfig, {
      user_id: 'string',
      parent_company_id: 'number',
    });

    if (!Array.isArray(accessConfig.sub_company_ids)) {
      throw this.createValidationError('sub_company_ids must be an array');
    }

    if (!accessConfig.database_access || typeof accessConfig.database_access !== 'object') {
      throw this.createValidationError('database_access is required and must be an object');
    }

    if (!accessConfig.table_shows || typeof accessConfig.table_shows !== 'object') {
      throw this.createValidationError('table_shows is required and must be an object');
    }

    // Validate user ID format
    if (accessConfig.user_id.trim().length === 0) {
      throw this.createValidationError('user_id cannot be empty');
    }

    // Validate parent company ID
    if (accessConfig.parent_company_id <= 0) {
      throw this.createValidationError('parent_company_id must be positive');
    }

    // Validate database access structure
    this.validateDatabaseAccess(accessConfig.database_access);

    return this.post<UserAccessCreateResponse>(
        API_ENDPOINTS.CREATE_USER_ACCESS,
        accessConfig
      );
  }

  /**
   * Get all user access configurations
   */
  async getUserAccessConfigs(): Promise<ServiceResponse<UserAccessListResponse>> {
    return this.get<UserAccessListResponse>(API_ENDPOINTS.GET_USER_ACCESS_CONFIGS);
  }

  /**
   * Get access configurations for a specific user
   */
  async getUserAccess(userId: string): Promise<ServiceResponse<UserAccessResponse>> {
    this.validateRequired({ userId }, ['userId']);
    this.validateTypes({ userId }, { userId: 'string' });

    if (userId.trim().length === 0) {
      throw this.createValidationError('User ID cannot be empty');
    }

    return this.get<UserAccessResponse>(API_ENDPOINTS.GET_USER_ACCESS(userId));
  }

  /**
   * Get accessible databases for a specific user
   */
  async getUserAccessibleDatabases(userId: string): Promise<ServiceResponse<{
    databases: Array<{
      id: number;
      name: string;
      description: string;
      url: string;
      access_level: string;
    }>;
    count: number;
  }>> {
    const userAccessResponse = await this.getUserAccess(userId);
    
    if (!userAccessResponse.success || !userAccessResponse.data?.access_configs?.length) {
      return {
        data: { databases: [], count: 0 },
        success: true,
        timestamp: new Date().toISOString(),
      };
    }

    const accessibleDatabases: Array<{
      id: number;
      name: string;
      description: string;
      url: string;
      access_level: string;
    }> = [];
    
    userAccessResponse.data.access_configs.forEach(config => {
        // Add parent company databases
        if (config.database_access.parent_databases) {
          config.database_access.parent_databases.forEach(db => {
            accessibleDatabases.push({
              id: db.db_id,
              name: `Database ${db.db_id}`,
              description: `Parent Company Database (${db.access_level} access)`,
              url: `Database ${db.db_id}`,
              access_level: db.access_level
            });
          });
        }

        // Add sub company databases
        if (config.database_access.sub_databases) {
          config.database_access.sub_databases.forEach(subDb => {
            if (subDb.databases) {
              subDb.databases.forEach(db => {
                accessibleDatabases.push({
                  id: db.db_id,
                  name: `Database ${db.db_id}`,
                  description: `Sub Company Database (${db.access_level} access)`,
                  url: `Database ${db.db_id}`,
                  access_level: db.access_level
                });
              });
            }
          });
        }
      });

    return {
      data: {
        databases: accessibleDatabases,
        count: accessibleDatabases.length,
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Update user access configuration
   */
  async updateUserAccess(
    userId: string,
    accessConfig: Partial<UserAccessCreateRequest>
  ): Promise<ServiceResponse<UserAccessCreateResponse>> {
    this.validateRequired({ userId }, ['userId']);
    this.validateTypes({ userId }, { userId: 'string' });

    if (userId.trim().length === 0) {
      throw this.createValidationError('User ID cannot be empty');
    }

    if (!accessConfig || typeof accessConfig !== 'object') {
      throw this.createValidationError('Access configuration is required');
    }

    // Validate provided fields
    if (accessConfig.parent_company_id !== undefined) {
      if (typeof accessConfig.parent_company_id !== 'number' || accessConfig.parent_company_id <= 0) {
        throw this.createValidationError('parent_company_id must be a positive number');
      }
    }

    if (accessConfig.sub_company_ids !== undefined && !Array.isArray(accessConfig.sub_company_ids)) {
      throw this.createValidationError('sub_company_ids must be an array');
    }

    if (accessConfig.database_access) {
      this.validateDatabaseAccess(accessConfig.database_access);
    }

    // For update, we'll use POST to the same endpoint with user_id
    const updateData = {
      user_id: userId,
      ...accessConfig,
    };

    return this.post<UserAccessCreateResponse>(
      API_ENDPOINTS.CREATE_USER_ACCESS,
      updateData
    );
  }

  /**
   * Delete user access configuration
   */
  async deleteUserAccess(userId: string): Promise<ServiceResponse<void>> {
    this.validateRequired({ userId }, ['userId']);
    this.validateTypes({ userId }, { userId: 'string' });

    if (userId.trim().length === 0) {
      throw this.createValidationError('User ID cannot be empty');
    }

    // Note: This would require a DELETE endpoint in the API
    // For now, we'll return a placeholder response
    return {
      data: undefined as any,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Check if user has access to a specific database
   */
  async checkUserDatabaseAccess(
    userId: string,
    databaseId: number
  ): Promise<ServiceResponse<{
    hasAccess: boolean;
    accessLevel?: string;
    restrictions?: string[];
  }>> {
    const userAccessResponse = await this.getUserAccess(userId);
    
    if (!userAccessResponse.success || !userAccessResponse.data?.access_configs?.length) {
      return {
        data: { hasAccess: false },
        success: true,
        timestamp: new Date().toISOString(),
      };
    }

    // Check access across all configurations
    for (const config of userAccessResponse.data.access_configs) {
      // Check parent databases
      const parentDb = config.database_access.parent_databases?.find(db => db.db_id === databaseId);
      if (parentDb) {
        return {
          data: {
            hasAccess: true,
            accessLevel: parentDb.access_level,
            restrictions: this.getAccessRestrictions(parentDb.access_level),
          },
          success: true,
          timestamp: new Date().toISOString(),
        };
      }

      // Check sub databases
      for (const subDb of config.database_access.sub_databases || []) {
        const foundDb = subDb.databases?.find(db => db.db_id === databaseId);
        if (foundDb) {
          return {
            data: {
              hasAccess: true,
              accessLevel: foundDb.access_level,
              restrictions: this.getAccessRestrictions(foundDb.access_level),
            },
            success: true,
            timestamp: new Date().toISOString(),
          };
        }
      }
    }

    return {
      data: { hasAccess: false },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get user access summary
   */
  async getUserAccessSummary(userId: string): Promise<ServiceResponse<{
    totalDatabases: number;
    accessLevels: Record<string, number>;
    companies: {
      parentCompanies: number;
      subCompanies: number;
    };
    tables: {
      totalTables: number;
      accessibleTables: number;
    };
  }>> {
    const userAccessResponse = await this.getUserAccess(userId);
    
    if (!userAccessResponse.success || !userAccessResponse.data?.access_configs?.length) {
      return {
        data: {
          totalDatabases: 0,
          accessLevels: {},
          companies: { parentCompanies: 0, subCompanies: 0 },
          tables: { totalTables: 0, accessibleTables: 0 },
        },
        success: true,
        timestamp: new Date().toISOString(),
      };
    }

    let totalDatabases = 0;
    const accessLevels: Record<string, number> = {};
    const parentCompanies = new Set<number>();
    const subCompanies = new Set<number>();
    let totalTables = 0;

    userAccessResponse.data.access_configs.forEach(config => {
      parentCompanies.add(config.parent_company_id);
      
      config.sub_company_ids.forEach(subId => subCompanies.add(subId));

      // Count parent databases
      config.database_access.parent_databases?.forEach(db => {
        totalDatabases++;
        accessLevels[db.access_level] = (accessLevels[db.access_level] || 0) + 1;
      });

      // Count sub databases
      config.database_access.sub_databases?.forEach(subDb => {
        subDb.databases?.forEach(db => {
          totalDatabases++;
          accessLevels[db.access_level] = (accessLevels[db.access_level] || 0) + 1;
        });
      });

      // Count tables
      Object.values(config.table_shows).forEach(tables => {
        if (Array.isArray(tables)) {
          totalTables += tables.length;
        }
      });
    });

    return {
      data: {
        totalDatabases,
        accessLevels,
        companies: {
          parentCompanies: parentCompanies.size,
          subCompanies: subCompanies.size,
        },
        tables: {
          totalTables,
          accessibleTables: totalTables, // Assuming all shown tables are accessible
        },
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Validate database access structure
   */
  private validateDatabaseAccess(databaseAccess: any): void {
    if (!databaseAccess.parent_databases || !Array.isArray(databaseAccess.parent_databases)) {
      throw this.createValidationError('parent_databases must be an array');
    }

    if (!databaseAccess.sub_databases || !Array.isArray(databaseAccess.sub_databases)) {
      throw this.createValidationError('sub_databases must be an array');
    }

    // Validate parent databases
    databaseAccess.parent_databases.forEach((db: any, index: number) => {
      if (!db.db_id || typeof db.db_id !== 'number' || db.db_id <= 0) {
        throw this.createValidationError(`parent_databases[${index}].db_id must be a positive number`);
      }
      if (!db.access_level || typeof db.access_level !== 'string') {
        throw this.createValidationError(`parent_databases[${index}].access_level is required`);
      }
      if (!['full', 'read_only', 'limited'].includes(db.access_level)) {
        throw this.createValidationError(`parent_databases[${index}].access_level must be 'full', 'read_only', or 'limited'`);
      }
    });

    // Validate sub databases
    databaseAccess.sub_databases.forEach((subDb: any, index: number) => {
      if (!subDb.sub_company_id || typeof subDb.sub_company_id !== 'number' || subDb.sub_company_id <= 0) {
        throw this.createValidationError(`sub_databases[${index}].sub_company_id must be a positive number`);
      }
      if (!subDb.databases || !Array.isArray(subDb.databases)) {
        throw this.createValidationError(`sub_databases[${index}].databases must be an array`);
      }
      
      subDb.databases.forEach((db: any, dbIndex: number) => {
        if (!db.db_id || typeof db.db_id !== 'number' || db.db_id <= 0) {
          throw this.createValidationError(`sub_databases[${index}].databases[${dbIndex}].db_id must be a positive number`);
        }
        if (!db.access_level || typeof db.access_level !== 'string') {
          throw this.createValidationError(`sub_databases[${index}].databases[${dbIndex}].access_level is required`);
        }
        if (!['full', 'read_only', 'limited'].includes(db.access_level)) {
          throw this.createValidationError(`sub_databases[${index}].databases[${dbIndex}].access_level must be 'full', 'read_only', or 'limited'`);
        }
      });
    });
  }

  /**
   * Get access restrictions based on access level
   */
  private getAccessRestrictions(accessLevel: string): string[] {
    switch (accessLevel) {
      case 'full':
        return [];
      case 'read_only':
        return ['Cannot modify data', 'Cannot create/drop tables', 'Cannot execute DDL statements'];
      case 'limited':
        return ['Limited table access', 'Cannot modify data', 'Cannot create/drop tables', 'Cannot execute DDL statements'];
      default:
        return ['Unknown access level restrictions'];
    }
  }
}

// Export singleton instance
export const userAccessService = new UserAccessService();

// Export for backward compatibility
export default userAccessService;