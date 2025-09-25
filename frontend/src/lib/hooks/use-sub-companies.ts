import { useState, useCallback } from "react";
import { ServiceRegistry } from "../api";
import {
  SubCompanyCreateRequest,
  SubCompanyData,
} from "@/types/api";

/**
 * Hook for Sub Company operations using standardized ServiceRegistry
 */
export function useSubCompanies() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [subCompanies, setSubCompanies] = useState<SubCompanyData[]>([]);

  /**
   * Get all sub companies
   */
  const getSubCompanies = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.subCompanies.getSubCompanies();
      
      if (response.success) {
        setSubCompanies(response.data);
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to fetch sub companies');
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch sub companies");
      setSubCompanies([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Create a new sub company
   */
  const createSubCompany = useCallback(async (companyData: SubCompanyCreateRequest) => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await ServiceRegistry.subCompanies.createSubCompany(companyData);
      
      if (response.success) {
        setSuccess(true);
        // Refresh the list
        await getSubCompanies();
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to create sub company');
      }
    } catch (e: any) {
      setError(e.message || "Failed to create sub company");
      return null;
    } finally {
      setLoading(false);
    }
  }, [getSubCompanies]);

  /**
   * Update a sub company
   */
  const updateSubCompany = useCallback(async (
    companyId: number,
    companyData: Partial<SubCompanyCreateRequest>
  ) => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await ServiceRegistry.subCompanies.updateSubCompany(companyId, companyData);
      
      if (response.success) {
        setSuccess(true);
        // Refresh the list
        await getSubCompanies();
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to update sub company');
      }
    } catch (e: any) {
      setError(e.message || "Failed to update sub company");
      return null;
    } finally {
      setLoading(false);
    }
  }, [getSubCompanies]);

  /**
   * Delete a sub company
   */
  const deleteSubCompany = useCallback(async (companyId: number) => {
    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await ServiceRegistry.subCompanies.deleteSubCompany(companyId);
      
      if (response.success) {
        setSuccess(true);
        // Refresh the list
        await getSubCompanies();
        return true;
      } else {
        throw new Error(response.error || 'Failed to delete sub company');
      }
    } catch (e: any) {
      setError(e.message || "Failed to delete sub company");
      return false;
    } finally {
      setLoading(false);
    }
  }, [getSubCompanies]);

  /**
   * Get sub company by ID
   */
  const getSubCompany = useCallback(async (companyId: number) => {
    setError(null);
    try {
      const response = await ServiceRegistry.subCompanies.getSubCompany(companyId);
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get sub company');
      }
    } catch (e: any) {
      setError(e.message || "Failed to get sub company");
      throw e;
    }
  }, []);

  /**
   * Get sub companies by parent company ID
   */
  const getSubCompaniesByParent = useCallback(async (parentCompanyId: number) => {
    setError(null);
    try {
      const response = await ServiceRegistry.subCompanies.getSubCompaniesByParent(parentCompanyId);
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get sub companies by parent');
      }
    } catch (e: any) {
      setError(e.message || "Failed to get sub companies by parent");
      throw e;
    }
  }, []);

  /**
   * Get sub companies by database ID
   */
  const getSubCompaniesByDatabase = useCallback(async (databaseId: number) => {
    setError(null);
    try {
      const response = await ServiceRegistry.subCompanies.getSubCompaniesByDatabase(databaseId);
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get sub companies by database');
      }
    } catch (e: any) {
      setError(e.message || "Failed to get sub companies by database");
      throw e;
    }
  }, []);

  /**
   * Get sub company statistics
   */
  const getSubCompanyStats = useCallback(async () => {
    setError(null);
    try {
      const response = await ServiceRegistry.subCompanies.getSubCompanyStats();
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to get sub company stats');
      }
    } catch (e: any) {
      setError(e.message || "Failed to get sub company stats");
      throw e;
    }
  }, []);

  /**
   * Check if a sub company can be created under a parent
   */
  const canCreateSubCompany = useCallback(async (parentCompanyId: number) => {
    setError(null);
    try {
      const response = await ServiceRegistry.subCompanies.canCreateSubCompany(parentCompanyId);
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || 'Failed to check sub company creation eligibility');
      }
    } catch (e: any) {
      setError(e.message || "Failed to check sub company creation eligibility");
      throw e;
    }
  }, []);

  /**
   * Validate sub company data
   */
  const validateSubCompany = useCallback((companyData: SubCompanyCreateRequest) => {
    return ServiceRegistry.subCompanies.validateSubCompanyData(companyData);
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
    setSubCompanies([]);
  }, []);

  return {
    // State
    loading,
    error,
    success,
    subCompanies,

    // Actions
    getSubCompanies,
    createSubCompany,
    updateSubCompany,
    deleteSubCompany,
    getSubCompany,
    getSubCompaniesByParent,
    getSubCompaniesByDatabase,
    getSubCompanyStats,
    canCreateSubCompany,
    validateSubCompany,

    // Utilities
    clearError,
    clearSuccess,
    reset,
  };
}