/**
 * Business Rules Validator
 * Validates database queries against business rules content
 */

export interface BusinessRuleValidation {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

export class BusinessRulesValidator {
  /**
   * Validate a database query against business rules content
   */
  static validateQuery(
    query: string,
    businessRules: string,
    databaseId?: string
  ): BusinessRuleValidation {
    const errors: string[] = [];
    const warnings: string[] = [];
    const suggestions: string[] = [];

    if (!businessRules.trim()) {
      // No business rules configured - query is valid
      return { isValid: true, errors, warnings, suggestions };
    }

    // Basic SQL injection prevention
    if (this.containsSQLInjection(query)) {
      errors.push('Query contains potentially dangerous SQL patterns');
    }

    // Check for DDL operations (CREATE, DROP, ALTER, etc.)
    if (this.containsDDLOperations(query)) {
      errors.push('DDL operations (CREATE, DROP, ALTER) are not allowed');
    }

    // Check for system table access
    if (this.accessesSystemTables(query)) {
      warnings.push('Query accesses system tables - ensure this is intentional');
    }

    // Check for large result sets
    if (this.mayReturnLargeResultSet(query)) {
      warnings.push('Query may return a large result set - consider adding LIMIT');
    }

    // Check for missing WHERE clause on large tables
    if (this.missingWhereClause(query)) {
      warnings.push('Query missing WHERE clause - may affect performance on large tables');
    }

    // Apply custom business rules from configuration
    this.applyCustomBusinessRules(query, businessRules, errors, warnings, suggestions, databaseId);

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      suggestions
    };
  }

  /**
   * Check if query contains SQL injection patterns
   */
  private static containsSQLInjection(query: string): boolean {
    const dangerousPatterns = [
      /;\s*drop\s+table/i,
      /;\s*delete\s+from/i,
      /;\s*update\s+.+\s+set/i,
      /;\s*insert\s+into/i,
      /;\s*exec\s*\(/i,
      /;\s*execute\s*\(/i,
      /;\s*xp_cmdshell/i,
      /;\s*sp_executesql/i
    ];

    return dangerousPatterns.some(pattern => pattern.test(query));
  }

  /**
   * Check if query contains DDL operations
   */
  private static containsDDLOperations(query: string): boolean {
    const ddlPatterns = [
      /^\s*create\s+/i,
      /^\s*drop\s+/i,
      /^\s*alter\s+/i,
      /^\s*truncate\s+/i,
      /^\s*grant\s+/i,
      /^\s*revoke\s+/i,
      /^\s*deny\s+/i
    ];

    return ddlPatterns.some(pattern => pattern.test(query));
  }

  /**
   * Check if query accesses system tables
   */
  private static accessesSystemTables(query: string): boolean {
    const systemTablePatterns = [
      /from\s+sys\./i,
      /from\s+information_schema\./i,
      /from\s+master\./i,
      /from\s+tempdb\./i,
      /from\s+msdb\./i
    ];

    return systemTablePatterns.some(pattern => pattern.test(query));
  }

  /**
   * Check if query may return large result sets
   */
  private static mayReturnLargeResultSet(query: string): boolean {
    const largeResultPatterns = [
      /select\s+\*/i,
      /select\s+.*\s+from\s+\w+\s*(?!where|limit|top)/i
    ];

    return largeResultPatterns.some(pattern => pattern.test(query)) && 
           !query.toLowerCase().includes('limit') && 
           !query.toLowerCase().includes('top');
  }

  /**
   * Check if query is missing WHERE clause
   */
  private static missingWhereClause(query: string): boolean {
    const selectPattern = /select\s+.+\s+from\s+\w+/i;
    const wherePattern = /\bwhere\b/i;
    
    return selectPattern.test(query) && !wherePattern.test(query);
  }

  /**
   * Apply custom business rules from configuration
   */
  private static applyCustomBusinessRules(
    query: string, 
    businessRules: string,
    errors: string[],
    warnings: string[],
    suggestions: string[],
    databaseId?: string
  ): void {
    const rules = businessRules.toLowerCase();
    
    // Database-specific validation
    if (databaseId) {
      // Check for database-specific restrictions
      if (rules.includes('production database restrictions') && databaseId.includes('prod')) {
        warnings.push('Query executed on production database - ensure this is intentional');
      }
      
      if (rules.includes('test database only') && !databaseId.includes('test')) {
        errors.push('This query type is restricted to test databases only');
      }
      
      if (rules.includes('specific database access') && !this.isAllowedDatabase(databaseId, rules)) {
        errors.push(`Access to database ${databaseId} is restricted by business rules`);
      }
    }
    
    // Check for specific table restrictions
    if (rules.includes('no salary access') && query.toLowerCase().includes('salary')) {
      errors.push('Access to salary information is restricted by business rules');
    }

    if (rules.includes('no personal data') && this.containsPersonalData(query)) {
      errors.push('Access to personal data is restricted by business rules');
    }

    if (rules.includes('audit required') && this.isAuditRequired(query)) {
      warnings.push('This query type requires audit logging');
    }

    // Check for time-based restrictions
    if (rules.includes('business hours only') && !this.isBusinessHours()) {
      warnings.push('Query executed outside business hours');
    }

    // Check for user-specific restrictions
    if (rules.includes('admin only')) {
      warnings.push('This query type may require admin privileges');
    }

    // Check for specific table access restrictions
    if (rules.includes('restricted tables') && this.accessesRestrictedTables(query)) {
      errors.push('Access to restricted tables is blocked by business rules');
    }
  }

  /**
   * Check if query contains personal data
   */
  private static containsPersonalData(query: string): boolean {
    const personalDataPatterns = [
      /ssn|social\s*security/i,
      /credit\s*card|cc\s*num/i,
      /passport\s*number/i,
      /driver\s*license/i,
      /phone\s*number|phone\s*num/i,
      /email\s*address|email\s*addr/i
    ];

    return personalDataPatterns.some(pattern => pattern.test(query));
  }

  /**
   * Check if query requires audit logging
   */
  private static isAuditRequired(query: string): boolean {
    const auditRequiredPatterns = [
      /delete\s+from/i,
      /update\s+.+\s+set/i,
      /insert\s+into/i
    ];

    return auditRequiredPatterns.some(pattern => pattern.test(query));
  }

  /**
   * Check if current time is within business hours
   */
  private static isBusinessHours(): boolean {
    const now = new Date();
    const hour = now.getHours();
    return hour >= 8 && hour <= 18; // 8 AM to 6 PM
  }

  /**
   * Check if query accesses restricted tables
   */
  private static accessesRestrictedTables(query: string): boolean {
    const restrictedTablePatterns = [
      /from\s+users/i,
      /from\s+employees/i,
      /from\s+payroll/i,
      /from\s+salary/i,
      /from\s+personal_info/i
    ];

    return restrictedTablePatterns.some(pattern => pattern.test(query));
  }

  /**
   * Check if database is allowed based on business rules
   */
  private static isAllowedDatabase(databaseId: string, rules: string): boolean {
    // Extract allowed database patterns from rules
    const allowedPatterns = rules.match(/allowed databases?:\s*([^\n]+)/i);
    if (allowedPatterns) {
      const allowedDbs = allowedPatterns[1].split(',').map(db => db.trim().toLowerCase());
      return allowedDbs.some(pattern => databaseId.toLowerCase().includes(pattern));
    }
    
    // Default: allow all databases if no specific restrictions
    return true;
  }
} 