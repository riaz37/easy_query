import { queryService } from './query-service';
import { historyService } from './history-service';
import { databaseService } from './database-service';
import { businessRulesService } from './business-rules-service';
import { userCurrentDBService } from './user-current-db-service';
import { vectorDBService } from './vector-db-service';
import { fileService } from './file-service';
import { authService } from './auth-service';
import { excelToDBService } from './excel-to-db-service';
import { newTableService } from './new-table-service';
import { userAccessService } from './user-access-service';
import { parentCompanyService } from './parent-company-service';
import { subCompanyService } from './sub-company-service';
import { UserConfigService } from './user-config-service';
import { DatabaseConfigService } from './database-config-service';
import { ReportService } from './report-service';
import { userTasksService } from './user-tasks-service';

/**
 * Central registry for all API services
 * 
 * Usage:
 * const results = await ServiceRegistry.query.search({ query: 'test' });
 * const dbInfo = await ServiceRegistry.database.getAllDatabases();
 * const user = await ServiceRegistry.auth.login(credentials);
 * 
 * Available services:
 * - query: Database query operations
 * - database: Database management
 * - businessRules: Business rules management
 * - userCurrentDB: User's current database
 * - vectorDB: Vector database operations
 * - file: File operations
 * - auth: Authentication
 * - excelToDB: Excel to database operations
 * - newTable: Table creation and management
 * - userAccess: User access management
 */
export const ServiceRegistry = {
  // Standardized services (using BaseService)
  query: queryService,
  history: historyService,
  database: databaseService,
  businessRules: businessRulesService,
  userCurrentDB: userCurrentDBService,
  vectorDB: vectorDBService,
  file: fileService,
  auth: authService,
  excelToDB: excelToDBService,
  newTable: newTableService,
  userAccess: userAccessService,
  parentCompanies: parentCompanyService,
  subCompanies: subCompanyService,
  userConfig: UserConfigService,
  databaseConfig: DatabaseConfigService,
  reports: new ReportService(),
  userTasks: userTasksService,
} as const;

/**
 * Service registry interface for type safety
 */
export interface ServiceRegistryInterface {
  // All services are now standardized (using BaseService)
  query: typeof queryService;
  history: typeof historyService;
  database: typeof databaseService;
  businessRules: typeof businessRulesService;
  userCurrentDB: typeof userCurrentDBService;
  vectorDB: typeof vectorDBService;
  file: typeof fileService;
  auth: typeof authService;
  excelToDB: typeof excelToDBService;
  newTable: typeof newTableService;
  userAccess: typeof userAccessService;
  parentCompanies: typeof parentCompanyService;
  subCompanies: typeof subCompanyService;
  userConfig: typeof UserConfigService;
  databaseConfig: typeof DatabaseConfigService;
  reports: ReportService;
  userTasks: typeof userTasksService;
}

/**
 * Get service health status for all services
 */
export async function getServiceHealthStatus(): Promise<{
  overall: 'healthy' | 'degraded' | 'unhealthy';
  services: Record<string, {
    status: 'healthy' | 'unhealthy';
    responseTime?: number;
    error?: string;
  }>;
  timestamp: string;
}> {
  const healthChecks = await Promise.allSettled([
    // Test all standardized services
    ServiceRegistry.query.getQueryStats().then(() => ({ service: 'query', status: 'healthy' as const })),
  
    ServiceRegistry.history.getHistoryStats().then(() => ({ service: 'history', status: 'healthy' as const })),
    ServiceRegistry.database.getAllDatabases().then(() => ({ service: 'database', status: 'healthy' as const })),
    ServiceRegistry.businessRules.getBusinessRulesForCurrentDatabase().then(() => ({ service: 'businessRules', status: 'healthy' as const })),
    ServiceRegistry.userCurrentDB.hasCurrentDatabase().then(() => ({ service: 'userCurrentDB', status: 'healthy' as const })),
    ServiceRegistry.vectorDB.getVectorDBConfigs().then(() => ({ service: 'vectorDB', status: 'healthy' as const })),
    ServiceRegistry.file.getSupportedFileTypes().then(() => ({ service: 'file', status: 'healthy' as const })),
    ServiceRegistry.auth.validateTokenFormat('test').then(() => ({ service: 'auth', status: 'healthy' as const })),
    ServiceRegistry.excelToDB.getSupportedFormats().then(() => ({ service: 'excelToDB', status: 'healthy' as const })),
    ServiceRegistry.newTable.getSupportedColumnTypes().then(() => ({ service: 'newTable', status: 'healthy' as const })),
    ServiceRegistry.userAccess.getUserAccessConfigs().then(() => ({ service: 'userAccess', status: 'healthy' as const })),
    ServiceRegistry.databaseConfig.getDatabaseConfigs().then(() => ({ service: 'databaseConfig', status: 'healthy' as const })),
  ]);

  const services: Record<string, { status: 'healthy' | 'unhealthy'; error?: string }> = {};
  let healthyCount = 0;

  const serviceNames = ['query', 'history', 'database', 'businessRules', 'userCurrentDB', 'vectorDB', 'file', 'auth', 'excelToDB', 'newTable', 'userAccess', 'databaseConfig'];

  healthChecks.forEach((result, index) => {
    const serviceName = serviceNames[index];

    if (result.status === 'fulfilled') {
      services[serviceName] = { status: 'healthy' };
      healthyCount++;
    } else {
      services[serviceName] = {
        status: 'unhealthy',
        error: result.reason?.message || 'Health check failed',
      };
    }
  });

  const totalServices = healthChecks.length;
  let overall: 'healthy' | 'degraded' | 'unhealthy';

  if (healthyCount === totalServices) {
    overall = 'healthy';
  } else if (healthyCount > 0) {
    overall = 'degraded';
  } else {
    overall = 'unhealthy';
  }

  return {
    overall,
    services,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Service metrics collector
 */
export class ServiceMetricsCollector {
  private static metrics: Array<{
    service: string;
    operation: string;
    duration: number;
    success: boolean;
    timestamp: string;
  }> = [];

  static recordMetric(
    service: string,
    operation: string,
    duration: number,
    success: boolean
  ): void {
    this.metrics.push({
      service,
      operation,
      duration,
      success,
      timestamp: new Date().toISOString(),
    });

    // Keep only last 1000 metrics to prevent memory leaks
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-1000);
    }
  }

  static getMetrics(service?: string): Array<{
    service: string;
    operation: string;
    duration: number;
    success: boolean;
    timestamp: string;
  }> {
    if (service) {
      return this.metrics.filter(m => m.service === service);
    }
    return [...this.metrics];
  }

  static getServiceStats(service: string): {
    totalOperations: number;
    successRate: number;
    averageResponseTime: number;
    errorCount: number;
  } {
    const serviceMetrics = this.getMetrics(service);
    const totalOperations = serviceMetrics.length;
    const successfulOperations = serviceMetrics.filter(m => m.success).length;
    const averageResponseTime = serviceMetrics.length > 0
      ? serviceMetrics.reduce((sum, m) => sum + m.duration, 0) / serviceMetrics.length
      : 0;

    return {
      totalOperations,
      successRate: totalOperations > 0 ? (successfulOperations / totalOperations) * 100 : 0,
      averageResponseTime,
      errorCount: totalOperations - successfulOperations,
    };
  }

  static clearMetrics(): void {
    this.metrics = [];
  }
}

/**
 * Get all standardized services list
 */
export function getStandardizedServices(): string[] {
  return [
    'query',
    'history', 
    'database',
    'businessRules',
    'userCurrentDB',
    'vectorDB',
    'file',
    'auth',
    'excelToDB',
    'newTable',
    'userAccess',
  ];
}

/**
 * Get legacy services list (all services are now standardized)
 */
export function getLegacyServices(): string[] {
  return [];
}

/**
 * Check if a service is standardized
 */
export function isServiceStandardized(serviceName: string): boolean {
  return getStandardizedServices().includes(serviceName);
}

// Export individual services for direct access
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
};

export default ServiceRegistry; 