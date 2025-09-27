import { useState, useCallback } from "react";
import { ServiceRegistry } from "../api";
import {
  ParentCompanyCreateRequest,
  ParentCompanyData,
} from "@/types/api";

/**
 * Hook for Parent Company operations using standardized ServiceRegistry
 */
export function useParentCompanies() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [parentCompanies, setParentCompanies] = useState<ParentCompanyData[]>([]);

  /**
   * Get all parent companies
   */
  const getParentCompanies = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.parentCompanies.getParentCompanies();
      
      if (response.success) {
        setParentCompanies(response.data);
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to fetch parent companies');
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch parent companies");
      setParentCompanies([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Create a new parent company
   */
  const createParentCompany = useCallback(async (companyData: ParentCompanyCreateRequest) => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await ServiceRegistry.parentCompanies.createParentCompany(companyData);
      
      if (response.success) {
        setSuccess(true);
        // Refresh the list
        await getParentCompanies();
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to create parent company');
      }
    } catch (e: any) {
      setError(e.message || "Failed to create parent company");
      return null;
    } finally {
      setLoading(false);
    }
  }, [getParentCompanies]);

  /**
   * Update a parent company
   */
  const updateParentCompany = useCallback(async (
    companyId: number,
    companyData: Partial<ParentCompanyCreateRequest>
  ) => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await ServiceRegistry.parentCompanies.updateParentCompany(companyId, companyData);
      
      if (response.success) {
        setSuccess(true);
        // Refresh the list
        await getParentCompanies();
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to update parent company');
      }
    } catch (e: any) {
      setError(e.message || "Failed to update parent company");
      return null;
    } finally {
      setLoading(false);
    }
  }, [getParentCompanies]);

  /**
   * Delete a parent company
   */
  const deleteParentCompany = useCallback(async (companyId: number) => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await ServiceRegistry.parentCompanies.deleteParentCompany(companyId);
      
      if (response.success) {
        setSuccess(true);
        // Refresh the list
        await getParentCompanies();
        return true;
      } else {
        throw new Error(response.error || 'Failed to delete parent company');
      }
    } catch (e: any) {
      setError(e.message || "Failed to delete parent company");
      return false;
    } finally {
      setLoading(false);
    }
  }, [getParentCompanies]);

  /**
   * Get parent company by ID
   */
  const getParentCompany = useCallback(async (companyId: number) => {
    setError(null);
    try {
      const response = await ServiceRegistry.parentCompanies.getParentCompany(companyId);
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get parent company');
      }
    } catch (e: any) {
      setError(e.message || "Failed to get parent company");
      throw e;
    }
  }, []);

  /**
   * Get parent companies by database ID
   */
  const getParentCompaniesByDatabase = useCallback(async (databaseId: number) => {
    setError(null);
    try {
      const response = await ServiceRegistry.parentCompanies.getParentCompaniesByDatabase(databaseId);
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get parent companies by database');
      }
    } catch (e: any) {
      setError(e.message || "Failed to get parent companies by database");
      throw e;
    }
  }, []);

  /**
   * Get parent company statistics
   */
  const getParentCompanyStats = useCallback(async () => {
    setError(null);
    try {
      const response = await ServiceRegistry.parentCompanies.getParentCompanyStats();
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get parent company stats');
      }
    } catch (e: any) {
      setError(e.message || "Failed to get parent company stats");
      throw e;
    }
  }, []);

  /**
   * Validate parent company data
   */
  const validateParentCompany = useCallback((companyData: ParentCompanyCreateRequest) => {
    return ServiceRegistry.parentCompanies.validateParentCompanyData(companyData);
  }, []);

  // Clear functions
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearSuccess = useCallback(() => {
    setSuccess(false);
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setSuccess(false);
    setParentCompanies([]);
  }, []);

  return {
    // State
    loading,
    error,
    success,
    parentCompanies,

    // Actions
    getParentCompanies,
    createParentCompany,
    updateParentCompany,
    deleteParentCompany,
    getParentCompany,
    getParentCompaniesByDatabase,
    getParentCompanyStats,
    validateParentCompany,

    // Utilities
    clearError,
    clearSuccess,
    reset,
  };
}