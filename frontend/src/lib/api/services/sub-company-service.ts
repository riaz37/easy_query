import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";
import {
  SubCompanyCreateRequest,
  SubCompanyResponse,
  SubCompaniesListResponse,
  SubCompanyData,
} from "@/types/api";

/**
 * Service for managing sub companies using standardized BaseService
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class SubCompanyService extends BaseService {
  protected readonly serviceName = 'SubCompanyService';

  /**
   * Create a new sub company
   */
  async createSubCompany(
    company: SubCompanyCreateRequest
  ): Promise<ServiceResponse<SubCompanyResponse>> {
    this.validateRequired(company, ['company_name', 'parent_company_id', 'db_id']);
    this.validateTypes(company, {
      company_name: 'string',
      parent_company_id: 'number',
      db_id: 'number',
    });

    if (company.company_name.trim().length === 0) {
      throw this.createValidationError('Company name cannot be empty');
    }

    if (company.parent_company_id <= 0) {
      throw this.createValidationError('Parent company ID must be positive');
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

    return this.post<SubCompanyResponse>(
      API_ENDPOINTS.CREATE_SUB_COMPANY,
      {
        company_name: company.company_name.trim(),
        description: company.description?.trim() || '',
        parent_company_id: company.parent_company_id,
        db_id: company.db_id,
        address: company.address?.trim() || '',
        contact_email: company.contact_email?.toLowerCase().trim() || '',
      }
    );
  }

  /**
   * Get all sub companies
   */
  async getSubCompanies(): Promise<ServiceResponse<SubCompanyData[]>> {
    const response = await this.get<SubCompaniesListResponse>(
      API_ENDPOINTS.GET_SUB_COMPANIES
    );

    // Transform response to extract the companies array
    let companies: SubCompanyData[] = [];
    
    if (Array.isArray(response.data)) {
      companies = response.data;
    } else if (response.data && typeof response.data === 'object') {
      // Handle different response structures
      companies = response.data.companies || 
                  response.data.sub_companies || 
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
   * Get a specific sub company by ID
   */
  async getSubCompany(companyId: number): Promise<ServiceResponse<SubCompanyData>> {
    this.validateRequired({ companyId }, ['companyId']);
    this.validateTypes({ companyId }, { companyId: 'number' });

    if (companyId <= 0) {
      throw this.createValidationError('Company ID must be positive');
    }

    return this.get<SubCompanyData>(
      API_ENDPOINTS.GET_SUB_COMPANY?.(companyId) || `/api/sub-companies/${companyId}`
    );
  }

  /**
   * Update a sub company
   */
  async updateSubCompany(
    companyId: number,
    updates: Partial<SubCompanyCreateRequest>
  ): Promise<ServiceResponse<SubCompanyData>> {
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

    if (updates.parent_company_id !== undefined && updates.parent_company_id <= 0) {
      throw this.createValidationError('Parent company ID must be positive');
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

    return this.put<SubCompanyData>(
      API_ENDPOINTS.UPDATE_SUB_COMPANY?.(companyId) || `/api/sub-companies/${companyId}`,
      cleanUpdates
    );
  }

  /**
   * Delete a sub company
   */
  async deleteSubCompany(companyId: number): Promise<ServiceResponse<void>> {
    this.validateRequired({ companyId }, ['companyId']);
    this.validateTypes({ companyId }, { companyId: 'number' });

    if (companyId <= 0) {
      throw this.createValidationError('Company ID must be positive');
    }

    return this.delete<void>(
      API_ENDPOINTS.DELETE_SUB_COMPANY?.(companyId) || `/api/sub-companies/${companyId}`
    );
  }

  /**
   * Get sub companies by parent company ID
   */
  async getSubCompaniesByParent(parentCompanyId: number): Promise<ServiceResponse<SubCompanyData[]>> {
    this.validateRequired({ parentCompanyId }, ['parentCompanyId']);
    this.validateTypes({ parentCompanyId }, { parentCompanyId: 'number' });

    if (parentCompanyId <= 0) {
      throw this.createValidationError('Parent company ID must be positive');
    }

    try {
      const response = await this.getSubCompanies();
      
      if (response.success) {
        const filteredCompanies = response.data.filter(company => company.parent_company_id === parentCompanyId);
        return {
          data: filteredCompanies,
          success: true,
          timestamp: new Date().toISOString(),
        };
      } else {
        throw new Error(response.error || 'Failed to get sub companies');
      }
    } catch (error) {
      throw error;
    }
  }

  /**
   * Get sub companies by database ID
   */
  async getSubCompaniesByDatabase(databaseId: number): Promise<ServiceResponse<SubCompanyData[]>> {
    this.validateRequired({ databaseId }, ['databaseId']);
    this.validateTypes({ databaseId }, { databaseId: 'number' });

    if (databaseId <= 0) {
      throw this.createValidationError('Database ID must be positive');
    }

    try {
      const response = await this.getSubCompanies();
      
      if (response.success) {
        const filteredCompanies = response.data.filter(company => company.db_id === databaseId);
        return {
          data: filteredCompanies,
          success: true,
          timestamp: new Date().toISOString(),
        };
      } else {
        throw new Error(response.error || 'Failed to get sub companies');
      }
    } catch (error) {
      throw error;
    }
  }

  /**
   * Get sub company statistics
   */
  async getSubCompanyStats(): Promise<ServiceResponse<{
    totalSubCompanies: number;
    subCompaniesByParent: Record<number, number>;
    subCompaniesByDatabase: Record<number, number>;
    averageSubCompaniesPerParent: number;
  }>> {
    try {
      const response = await this.getSubCompanies();
      
      if (response.success) {
        const companies = response.data;
        const totalSubCompanies = companies.length;
        
        // Group by parent company
        const subCompaniesByParent: Record<number, number> = {};
        const subCompaniesByDatabase: Record<number, number> = {};
        
        companies.forEach(company => {
          subCompaniesByParent[company.parent_company_id] = 
            (subCompaniesByParent[company.parent_company_id] || 0) + 1;
          
          subCompaniesByDatabase[company.db_id] = 
            (subCompaniesByDatabase[company.db_id] || 0) + 1;
        });

        const uniqueParents = Object.keys(subCompaniesByParent).length;
        const averageSubCompaniesPerParent = uniqueParents > 0 ? totalSubCompanies / uniqueParents : 0;

        return {
          data: {
            totalSubCompanies,
            subCompaniesByParent,
            subCompaniesByDatabase,
            averageSubCompaniesPerParent: Math.round(averageSubCompaniesPerParent * 100) / 100,
          },
          success: true,
          timestamp: new Date().toISOString(),
        };
      } else {
        throw new Error(response.error || 'Failed to get sub companies');
      }
    } catch (error) {
      throw error;
    }
  }

  /**
   * Check if a sub company can be created under a parent
   */
  async canCreateSubCompany(parentCompanyId: number): Promise<ServiceResponse<{
    canCreate: boolean;
    reason?: string;
    currentSubCompanies: number;
    maxAllowed?: number;
  }>> {
    try {
      const response = await this.getSubCompaniesByParent(parentCompanyId);
      
      if (response.success) {
        const currentSubCompanies = response.data.length;
        const maxAllowed = 50; // Example limit
        
        const canCreate = currentSubCompanies < maxAllowed;
        
        return {
          data: {
            canCreate,
            reason: canCreate ? undefined : `Maximum sub-companies limit (${maxAllowed}) reached`,
            currentSubCompanies,
            maxAllowed,
          },
          success: true,
          timestamp: new Date().toISOString(),
        };
      } else {
        throw new Error(response.error || 'Failed to check sub company creation eligibility');
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
   * Validate sub company data
   */
  validateSubCompanyData(data: SubCompanyCreateRequest): {
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

    if (!data.parent_company_id || data.parent_company_id <= 0) {
      errors.push('Valid parent company ID is required');
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
export const subCompanyService = new SubCompanyService();

// Export for backward compatibility
export default subCompanyService;