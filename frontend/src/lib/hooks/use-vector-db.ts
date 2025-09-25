import { useState, useCallback } from "react";
import { ServiceRegistry } from "../api";

/**
 * Hook for Vector Database operations using standardized ServiceRegistry
 */
export function useVectorDB() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [vectorDBConfigs, setVectorDBConfigs] = useState<any[]>([]);
  const [userTableNames, setUserTableNames] = useState<string[]>([]);
  const [userConfig, setUserConfig] = useState<any>(null);
  const [createLoading, setCreateLoading] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createSuccess, setCreateSuccess] = useState(false);

  /**
   * Get all available vector database configurations
   */
  const getVectorDBConfigs = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.vectorDB.getVectorDBConfigs();
      
      if (response.success) {
        setVectorDBConfigs(response.data || []);
        return response.data;
        } else {
        throw new Error(response.error || "Failed to fetch vector DB configs");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch vector DB configs");
      setVectorDBConfigs([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Get user table names
   */
  const getUserTableNames = useCallback(async (userId?: string) => {
    setLoading(true);
    setError(null);

    try {
      // If no userId provided, try to get from auth context
      if (!userId) {
        throw new Error('User ID is required to fetch table names');
      }
      
      const response = await ServiceRegistry.vectorDB.getUserTableNames(userId);
      
      if (response.success) {
        setUserTableNames(response.data || []);
        return response.data;
        } else {
        throw new Error(response.error || "Failed to fetch user table names");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch user table names");
      setUserTableNames([]);
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Get user table names with metadata
   */
  const getUserTableNamesWithMetadata = useCallback(async (userId?: string) => {
    setLoading(true);
    setError(null);

    try {
      // If no userId provided, try to get from auth context
      if (!userId) {
        throw new Error('User ID is required to fetch table names with metadata');
      }
      
      const response = await ServiceRegistry.vectorDB.getUserTableNamesWithMetadata(userId);
      
      if (response.success) {
        setUserTableNames(response.data.tableNames || []);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to fetch table names with metadata");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch table names with metadata");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Create vector database access
   */
  const createVectorDBAccess = useCallback(async (request: {
    vector_db_id: number;
    accessible_tables: string[];
    access_level: string;
  }) => {
    setCreateLoading(true);
    setCreateError(null);
    setCreateSuccess(false);

    try {
      const response = await ServiceRegistry.vectorDB.createVectorDBAccess(request);
      
      if (response.success) {
        setCreateSuccess(true);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to create vector DB access");
      }
    } catch (e: any) {
      setCreateError(e.message || "Failed to create vector DB access");
      return null;
    } finally {
      setCreateLoading(false);
    }
  }, []);

  /**
   * Get user configuration by database ID
   */
  const getUserConfigByDB = useCallback(async (dbId: number) => {
    setLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.vectorDB.getUserConfigByDB(dbId);
      
      if (response.success) {
        setUserConfig(response.data);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to fetch user config");
      }
    } catch (e: any) {
      setError(e.message || "Failed to fetch user config");
      setUserConfig(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * Add user table name
   */
  const addUserTableName = useCallback(async (tableName: string, userId: string) => {
    setCreateLoading(true);
    setCreateError(null);
    setCreateSuccess(false);

    try {
      const response = await ServiceRegistry.vectorDB.addUserTableName(tableName, userId);
      
      if (response.success) {
        setCreateSuccess(true);
        // Refresh table names
        await getUserTableNames();
        return response.data;
      } else {
        throw new Error(response.error || "Failed to add table name");
      }
    } catch (e: any) {
      setCreateError(e.message || "Failed to add table name");
      return null;
    } finally {
      setCreateLoading(false);
    }
  }, [getUserTableNames]);

  /**
   * Delete user table name
   */
  const deleteUserTableName = useCallback(async (tableName: string, userId: string) => {
    setCreateLoading(true);
    setCreateError(null);

    try {
      const response = await ServiceRegistry.vectorDB.deleteUserTableName(tableName, userId);
      
      if (response.success) {
        setCreateSuccess(true);
        // Refresh table names
        await getUserTableNames(userId);
        return true;
      } else {
        throw new Error(response.error || "Failed to delete table name");
      }
    } catch (e: any) {
      setCreateError(e.message || "Failed to delete table name");
      return false;
    } finally {
      setCreateLoading(false);
    }
  }, [getUserTableNames]);

  /**
   * Add multiple table names
   */
  const addMultipleTableNames = useCallback(async (tableNames: string[], userId: string) => {
    setCreateLoading(true);
    setCreateError(null);
    setCreateSuccess(false);

    try {
      const response = await ServiceRegistry.vectorDB.addMultipleTableNames(tableNames, userId);
      
      if (response.success) {
        setCreateSuccess(true);
        // Refresh table names
        await getUserTableNames(userId);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to add table names");
      }
    } catch (e: any) {
      setCreateError(e.message || "Failed to add table names");
      return null;
    } finally {
      setCreateLoading(false);
    }
  }, [getUserTableNames]);

  /**
   * Validate table name format
   */
  const validateTableName = useCallback((tableName: string) => {
    return ServiceRegistry.vectorDB.validateTableName(tableName);
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
    setVectorDBConfigs([]);
    setUserTableNames([]);
    setUserConfig(null);
  }, []);

  return {
    // State
    loading,
    error,
    vectorDBConfigs,
    userTableNames,
    userConfig,
    createLoading,
    createError,
    createSuccess,

    // Actions
    getVectorDBConfigs,
    getUserTableNames,
    getUserTableNamesWithMetadata,
    createVectorDBAccess,
    getUserConfigByDB,
    addUserTableName,
    deleteUserTableName,
    addMultipleTableNames,
    validateTableName,

    // Utilities
    clearError,
    clearSuccess,
    reset,
  };
} 