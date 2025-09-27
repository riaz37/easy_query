import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";
import {
  ParentCompanyCreateRequest,
  ParentCompanyResponse,
  ParentCompaniesListResponse,
  ParentCompanyData,
} from "@/types/api";

/**
 * Service for managing parent companies using standardized BaseService
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class ParentCompanyService extends BaseService {
  protected readonly serviceName = 'ParentCompanyService';

  /**
   * Create a new parent company
   */
  async createParentCompany(
    company: ParentCompanyCreateRequest
  ): Promise<ServiceResponse<ParentCompanyResponse>> {
    this.validateRequired(company, ['company_name', 'db_id']);
    this.validateTypes(company, {
      company_name: 'string',
      db_id: 'number',
    });

    if (company.company_name.trim().length === 0) {
      throw this.createValidationError('Company name cannot be empty');
    }

    if (company.db_id <= 0) {
      throw this.createValidationError('Database ID must be positive');
    }

    // Validate company name length
    if (company.company_name.length > 255) {
      throw this.createValidationError('Company name cannot be longer than 255 characters');
    }

    // Validate email format if provided
    if (company.contact_email && !this.isValidEmail(company.contact_email)) {
      throw this.createValidationError('Invalid email format');
    }

    return this.post<ParentCompanyResponse>(
      API_ENDPOINTS.CREATE_PARENT_COMPANY,
      {
        company_name: company.company_name.trim(),
        description: company.description?.trim() || '',
        db_id: company.db_id,
        address: company.address?.trim() || '',
        contact_email: company.contact_email?.toLowerCase().trim() || '',
      }
    );
  }

  /**
   * Get all parent companies
   */
  async getParentCompanies(): Promise<ServiceResponse<ParentCompanyData[]>> {
    const response = await this.get<ParentCompaniesListResponse>(
      API_ENDPOINTS.GET_PARENT_COMPANIES
    );

    // Transform response to extract the companies array
    let companies: ParentCompanyData[] = [];
    
    if (Array.isArray(response.data)) {
      companies = response.data;
    } else if (response.data && typeof response.data === 'object') {
      // Handle different response structures
      companies = response.data.companies || 
                  response.data.parent_companies || 
                  response.data.data || 
                  [];
    }

    return {
      data: companies,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get a specific parent company by ID
   */
  async getParentCompany(companyId: number): Promise<ServiceResponse<ParentCompanyData>> {
    this.validateRequired({ companyId }, ['companyId']);
    this.validateTypes({ companyId }, { companyId: 'number' });

    if (companyId <= 0) {
      throw this.createValidationError('Company ID must be positive');
    }

    return this.get<ParentCompanyData>(
      API_ENDPOINTS.GET_PARENT_COMPANY?.(companyId) || `/api/parent-companies/${companyId}`
    );
  }

  /**
   * Update a parent company
   */
  async updateParentCompany(
    companyId: number,
    updates: Partial<ParentCompanyCreateRequest>
  ): Promise<ServiceResponse<ParentCompanyData>> {
    this.validateRequired({ companyId }, ['companyId']);
    this.validateTypes({ companyId }, { companyId: 'number' });

    if (companyId <= 0) {
      throw this.createValidationError('Company ID must be positive');
    }

    // Validate updates
    if (updates.company_name !== undefined) {
      if (typeof updates.company_name !== 'string' || updates.company_name.trim().length === 0) {
        throw this.createValidationError('Company name cannot be empty');
      }
      if (updates.company_name.length > 255) {
        throw this.createValidationError('Company name cannot be longer than 255 characters');
      }
    }

    if (updates.contact_email !== undefined && updates.contact_email && !this.isValidEmail(updates.contact_email)) {
      throw this.createValidationError('Invalid email format');
    }

    if (updates.db_id !== undefined && updates.db_id <= 0) {
      throw this.createValidationError('Database ID must be positive');
    }

    // Clean up the updates
    const cleanUpdates = {
      ...updates,
      company_name: updates.company_name?.trim(),
      description: updates.description?.trim(),
      address: updates.address?.trim(),
      contact_email: updates.contact_email?.toLowerCase().trim(),
    };

    return this.put<ParentCompanyData>(
      API_ENDPOINTS.UPDATE_PARENT_COMPANY?.(companyId) || `/api/parent-companies/${companyId}`,
      cleanUpdates
    );
  }

  /**
   * Delete a parent company
   */
  async deleteParentCompany(companyId: number): Promise<ServiceResponse<void>> {
    this.validateRequired({ companyId }, ['companyId']);
    this.validateTypes({ companyId }, { companyId: 'number' });

    if (companyId <= 0) {
      throw this.createValidationError('Company ID must be positive');
    }

    return this.delete<void>(
      API_ENDPOINTS.DELETE_PARENT_COMPANY?.(companyId) || `/api/parent-companies/${companyId}`
    );
  }

  /**
   * Get parent companies by database ID
   */
  async getParentCompaniesByDatabase(databaseId: number): Promise<ServiceResponse<ParentCompanyData[]>> {
    this.validateRequired({ databaseId }, ['databaseId']);
    this.validateTypes({ databaseId }, { databaseId: 'number' });

    if (databaseId <= 0) {
      throw this.createValidationError('Database ID must be positive');
    }

    try {
      const response = await this.getParentCompanies();
      
      if (response.success) {
        const filteredCompanies = response.data.filter(company => company.db_id === databaseId);
        return {
          data: filteredCompanies,
          success: true,
          timestamp: new Date().toISOString(),
        };
      } else {
        throw new Error(response.error || 'Failed to get parent companies');
      }
    } catch (error) {
      throw error;
    }
  }

  /**
   * Get parent company statistics
   */
  async getParentCompanyStats(): Promise<ServiceResponse<{
    totalCompanies: number;
    companiesByDatabase: Record<number, number>;
    companiesWithSubCompanies: number;
    averageSubCompaniesPerParent: number;
  }>> {
    try {
      const response = await this.getParentCompanies();
      
      if (response.success) {
        const companies = response.data;
        const totalCompanies = companies.length;
        
        // Group by database
        const companiesByDatabase: Record<number, number> = {};
        companies.forEach(company => {
          companiesByDatabase[company.db_id] = (companiesByDatabase[company.db_id] || 0) + 1;
        });

        return {
          data: {
            totalCompanies,
            companiesByDatabase,
            companiesWithSubCompanies: 0, // Would need sub-company data to calculate
            averageSubCompaniesPerParent: 0, // Would need sub-company data to calculate
          },
          success: true,
          timestamp: new Date().toISOString(),
        };
      } else {
        throw new Error(response.error || 'Failed to get parent companies');
      }
    } catch (error) {
      throw error;
    }
  }

  /**
   * Validate email format
   */
  private isValidEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  /**
   * Validate parent company data
   */
  validateParentCompanyData(data: ParentCompanyCreateRequest): {
    isValid: boolean;
    errors: string[];
    warnings: string[];
  } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Required field validation
    if (!data.company_name || data.company_name.trim().length === 0) {
      errors.push('Company name is required');
    } else if (data.company_name.length > 255) {
      errors.push('Company name cannot be longer than 255 characters');
    }

    if (!data.db_id || data.db_id <= 0) {
      errors.push('Valid database ID is required');
    }

    // Optional field validation
    if (data.contact_email && !this.isValidEmail(data.contact_email)) {
      errors.push('Invalid email format');
    }

    if (data.description && data.description.length > 1000) {
      warnings.push('Description is quite long, consider shortening it');
    }

    if (data.address && data.address.length > 500) {
      warnings.push('Address is quite long, consider shortening it');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    };
  }
}

// Export singleton instance
export const parentCompanyService = new ParentCompanyService();

// Export for backward compatibility
export default parentCompanyService;