import { useState, useCallback } from "react";
import { ServiceRegistry } from "../api";

/**
 * Hook for managing user access operations using standardized ServiceRegistry
 */
export function useUserAccess() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userAccessConfigs, setUserAccessConfigs] = useState<any[]>([]);
  const [userAccess, setUserAccess] = useState<any>(null);
  const [accessibleDatabases, setAccessibleDatabases] = useState<any[]>([]);
  const [accessSummary, setAccessSummary] = useState<any>(null);
  const [createLoading, setCreateLoading] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createSuccess, setCreateSuccess] = useState(false);

  /**
   * Create user access configuration
   */
  const createUserAccess = useCallback(async (accessConfig: any) => {
    setCreateLoading(true);
    setCreateError(null);
    setCreateSuccess(false);

    try {
      const response = await ServiceRegistry.userAccess.createUserAccess(accessConfig);
      
      if (response.success) {
        setCreateSuccess(true);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to create user access");
      }
    } catch (e: any) {
      setCreateError(e.message || "Failed to create user access");
      return null;
    } finally {
      setCreateLoading(false);
    }
  }, []);

  /**
   * Get all user access configurations
   */
  const getUserAccessConfigs = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.userAccess.getUserAccessConfigs();
      
      if (response.success) {
        setUserAccessConfigs(response.data?.access_configs || []);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to fetch user access configs");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch user access configs");
      setUserAccessConfigs([]);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Get access configurations for a specific user
   */
  const getUserAccess = useCallback(async (userId: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.userAccess.getUserAccess(userId);
      
      if (response.success) {
        setUserAccess(response.data);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to fetch user access");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch user access");
      setUserAccess(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Get accessible databases for a specific user
   */
  const getUserAccessibleDatabases = useCallback(async (userId: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.userAccess.getUserAccessibleDatabases(userId);
      
      if (response.success) {
        setAccessibleDatabases(response.data.databases || []);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to fetch accessible databases");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch accessible databases");
      setAccessibleDatabases([]);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Update user access configuration
   */
  const updateUserAccess = useCallback(async (userId: string, accessConfig: any) => {
    setCreateLoading(true);
    setCreateError(null);
    setCreateSuccess(false);

    try {
      const response = await ServiceRegistry.userAccess.updateUserAccess(userId, accessConfig);
      
      if (response.success) {
        setCreateSuccess(true);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to update user access");
      }
    } catch (e: any) {
      setCreateError(e.message || "Failed to update user access");
      return null;
    } finally {
      setCreateLoading(false);
    }
  }, []);

  /**
   * Delete user access configuration
   */
  const deleteUserAccess = useCallback(async (userId: string) => {
    setCreateLoading(true);
    setCreateError(null);

    try {
      const response = await ServiceRegistry.userAccess.deleteUserAccess(userId);
      
      if (response.success) {
        setCreateSuccess(true);
        return true;
      } else {
        throw new Error(response.error || "Failed to delete user access");
      }
    } catch (e: any) {
      setCreateError(e.message || "Failed to delete user access");
      return false;
    } finally {
      setCreateLoading(false);
    }
  }, []);

  /**
   * Check if user has access to a specific database
   */
  const checkUserDatabaseAccess = useCallback(async (userId: string, databaseId: number) => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.userAccess.checkUserDatabaseAccess(userId, databaseId);
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || "Failed to check database access");
      }
    } catch (e: any) {
      setError(e.message || "Failed to check database access");
      return { hasAccess: false };
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Get user access summary
   */
  const getUserAccessSummary = useCallback(async (userId: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.userAccess.getUserAccessSummary(userId);
      
      if (response.success) {
        setAccessSummary(response.data);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to fetch access summary");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch access summary");
      setAccessSummary(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Clear functions
  const clearError = useCallback(() => {
    setError(null);
    setCreateError(null);
  }, []);

  const clearSuccess = useCallback(() => {
    setCreateSuccess(false);
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setCreateError(null);
    setCreateSuccess(false);
    setUserAccessConfigs([]);
    setUserAccess(null);
    setAccessibleDatabases([]);
    setAccessSummary(null);
  }, []);

  return {
    // State
    loading,
    error,
    userAccessConfigs,
    userAccess,
    accessibleDatabases,
    accessSummary,
    createLoading,
    createError,
    createSuccess,

    // Actions
    createUserAccess,
    getUserAccessConfigs,
    getUserAccess,
    getUserAccessibleDatabases,
    updateUserAccess,
    deleteUserAccess,
    checkUserDatabaseAccess,
    getUserAccessSummary,

    // Utilities
    clearError,
    clearSuccess,
    reset,
  };
}
