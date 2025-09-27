import { API_ENDPOINTS } from '../endpoints';
import { BaseService, ServiceResponse } from './base';
import { MSSQLConfigData } from '@/types/api';

/**
 * Database configuration information
 */
export interface DatabaseInfo {
  id: number;
  name: string;
  url: string;
  type: string;
  status: 'active' | 'inactive' | 'error';
  lastUpdated: string;
  metadata?: Record<string, any>;
}

/**
 * Database reload result
 */
export interface DatabaseReloadResult {
  success: boolean;
  message: string;
  reloadedAt: string;
  affectedTables?: string[];
  duration?: number;
}

/**
 * Service for handling database management API calls
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class DatabaseService extends BaseService {
  protected readonly serviceName = 'DatabaseService';

  /**
   * Get all available databases for the authenticated user
   */
  async getAllDatabases(): Promise<ServiceResponse<DatabaseInfo[]>> {
    const response = await this.get<any>(API_ENDPOINTS.GET_MSSQL_CONFIGS);
    
    // Transform MSSQL config data to DatabaseInfo format
    let databases: DatabaseInfo[] = [];
    
    if (response.data) {
      // Handle different response structures
      const configsData = Array.isArray(response.data) 
        ? response.data 
        : response.data.configs || [];
      
      databases = configsData.map((config: MSSQLConfigData) => ({
        id: config.db_id,
        name: config.db_name,
        url: config.db_url,
        type: 'mssql',
        status: 'active' as const,
        lastUpdated: config.updated_at,
        metadata: {
          businessRule: config.business_rule,
          tableInfo: config.table_info,
          dbSchema: config.db_schema,
          createdAt: config.created_at,
        },
      }));
    }

    return {
      data: databases,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get a specific database by ID
   */
  async getDatabaseById(databaseId: number): Promise<ServiceResponse<DatabaseInfo>> {
    this.validateRequired({ databaseId }, ['databaseId']);
    this.validateTypes({ databaseId }, { databaseId: 'number' });

    if (databaseId <= 0) {
      throw this.createValidationError('Database ID must be a positive number');
    }

    const response = await this.get<MSSQLConfigData>(
      API_ENDPOINTS.GET_MSSQL_CONFIG(databaseId)
    );

    if (!response.data) {
      throw this.createNotFoundError('Database', databaseId);
    }

    const config = response.data;
    const databaseInfo: DatabaseInfo = {
      id: config.db_id,
      name: config.db_name,
      url: config.db_url,
      type: 'mssql',
      status: 'active',
      lastUpdated: config.updated_at,
      metadata: {
        businessRule: config.business_rule,
        tableInfo: config.table_info,
        dbSchema: config.db_schema,
        createdAt: config.created_at,
      },
    };

    return {
      data: databaseInfo,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Reload the database schema and metadata
   */
  async reloadDatabase(): Promise<ServiceResponse<DatabaseReloadResult>> {
    const startTime = Date.now();
    
    const response = await this.post<any>(API_ENDPOINTS.RELOAD_DB);
    
    const duration = Date.now() - startTime;
    
    // Transform response to standardized format
    const reloadResult: DatabaseReloadResult = {
      success: true,
      message: response.data?.message || 'Database reloaded successfully',
      reloadedAt: new Date().toISOString(),
      duration,
      affectedTables: response.data?.affectedTables || [],
    };

    return {
      data: reloadResult,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Test database connection
   */
  async testDatabaseConnection(databaseId: number): Promise<ServiceResponse<{
    connected: boolean;
    responseTime: number;
    error?: string;
  }>> {
    this.validateRequired({ databaseId }, ['databaseId']);
    this.validateTypes({ databaseId }, { databaseId: 'number' });

    if (databaseId <= 0) {
      throw this.createValidationError('Database ID must be a positive number');
    }

    const startTime = Date.now();
    
    try {
      // Try to get database info as a connection test
      await this.getDatabaseById(databaseId);
      
      const responseTime = Date.now() - startTime;
      
      return {
        data: {
          connected: true,
          responseTime,
        },
        success: true,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      const responseTime = Date.now() - startTime;
      
      return {
        data: {
          connected: false,
          responseTime,
          error: error instanceof Error ? error.message : 'Connection test failed',
        },
        success: true, // The test completed successfully, even though connection failed
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Get database statistics
   */
  async getDatabaseStats(databaseId: number): Promise<ServiceResponse<{
    tableCount: number;
    totalRows: number;
    sizeInMB: number;
    lastActivity: string;
    connectionCount: number;
  }>> {
    this.validateRequired({ databaseId }, ['databaseId']);
    this.validateTypes({ databaseId }, { databaseId: 'number' });

    if (databaseId <= 0) {
      throw this.createValidationError('Database ID must be a positive number');
    }

    // Get database info first
    const databaseResponse = await this.getDatabaseById(databaseId);
    
    if (!databaseResponse.success) {
      throw this.createNotFoundError('Database', databaseId);
    }

    const tableInfo = databaseResponse.data.metadata?.tableInfo;
    
    // Calculate stats from table info
    const tableCount = tableInfo?.tables?.length || 0;
    const totalRows = tableInfo?.tables?.reduce((sum: number, table: any) => {
      return sum + (table.row_count_sample || 0);
    }, 0) || 0;

    // Mock data for fields not available in current API
    const stats = {
      tableCount,
      totalRows,
      sizeInMB: 0, // Would need dedicated endpoint
      lastActivity: databaseResponse.data.lastUpdated,
      connectionCount: 1, // Would need dedicated endpoint
    };

    return {
      data: stats,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Validate database configuration
   */
  validateDatabaseConfig(config: {
    name: string;
    url: string;
    type: string;
  }): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!config.name || config.name.trim().length === 0) {
      errors.push('Database name is required');
    }

    if (!config.url || config.url.trim().length === 0) {
      errors.push('Database URL is required');
    }

    if (!config.type || config.type.trim().length === 0) {
      errors.push('Database type is required');
    }

    // Validate URL format for MSSQL
    if (config.type === 'mssql' && config.url) {
      const mssqlUrlPattern = /^(?:mssql:\/\/|Server=|Data Source=)/i;
      if (!mssqlUrlPattern.test(config.url)) {
        errors.push('Invalid MSSQL connection string format');
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }
}

// Export singleton instance
export const databaseService = new DatabaseService();

// Export for backward compatibility
export default databaseService;