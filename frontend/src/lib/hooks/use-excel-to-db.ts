import { useState, useCallback } from "react";
import { ServiceRegistry } from "../api";

/**
 * Hook for Excel to Database operations using standardized ServiceRegistry
 */
export function useExcelToDB() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<any>(null);
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [aiMapping, setAiMapping] = useState<any>(null);
  const [validationResult, setValidationResult] = useState<any>(null);

  /**
   * Check health status of Excel to Database service
   */
  const checkHealth = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await ServiceRegistry.excelToDB.checkHealth();
      
      if (result.success) {
        setHealthStatus(result.data);
        return result.data;
      } else {
        throw new Error(result.error || "Health check failed");
      }
    } catch (e: any) {
      setError(e.message || "Health check failed");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Push Excel data to database
   */
  const pushDataToDatabase = useCallback(async (request: {
    user_id: string;
    table_full_name: string;
    column_mapping: Record<string, string>;
    excel_file: File;
    skip_first_row?: boolean;
  }) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const result = await ServiceRegistry.excelToDB.pushDataToDatabase(request);
      
      if (result.success) {
        setResponse(result.data);
        return result.data;
      } else {
        throw new Error(result.error || "Failed to push data to database");
      }
    } catch (e: any) {
      setError(e.message || "Failed to push data to database");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Get AI-powered column mapping suggestions
   */
  const getAIMapping = useCallback(async (request: {
    user_id: string;
    table_full_name: string;
    excel_file: File;
  }) => {
    setIsLoading(true);
    setError(null);
    setAiMapping(null);

    try {
      const result = await ServiceRegistry.excelToDB.getAIMapping(request);
      
      if (result.success) {
        setAiMapping(result.data);
        return result.data;
      } else {
        throw new Error(result.error || "Failed to get AI mapping");
      }
    } catch (e: any) {
      setError(e.message || "Failed to get AI mapping");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Validate Excel file
   */
  const validateExcelFile = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    setValidationResult(null);

    try {
      const result = await ServiceRegistry.excelToDB.validateExcelFile(file);
      
      if (result.success) {
        setValidationResult(result.data);
        return result.data;
      } else {
        throw new Error(result.error || "File validation failed");
      }
    } catch (e: any) {
      setError(e.message || "File validation failed");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Get supported file formats
   */
  const getSupportedFormats = useCallback(() => {
    const result = ServiceRegistry.excelToDB.getSupportedFormats();
    return result.data;
  }, []);

  /**
   * Preview Excel file data
   */
  const previewExcelData = useCallback(async (file: File, maxRows: number = 10) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await ServiceRegistry.excelToDB.previewExcelData(file, maxRows);
      
      if (result.success) {
        return result.data;
      } else {
        throw new Error(result.error || "Failed to preview Excel data");
      }
    } catch (e: any) {
      setError(e.message || "Failed to preview Excel data");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Clear functions
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearResponse = useCallback(() => {
    setResponse(null);
  }, []);

  const clearAiMapping = useCallback(() => {
    setAiMapping(null);
  }, []);

  const clearValidation = useCallback(() => {
    setValidationResult(null);
  }, []);

  const reset = useCallback(() => {
    setError(null);
    setResponse(null);
    setHealthStatus(null);
    setAiMapping(null);
    setValidationResult(null);
  }, []);

  return {
    // State
    isLoading,
    error,
    response,
    healthStatus,
    aiMapping,
    validationResult,

    // Actions
    checkHealth,
    pushDataToDatabase,
    getAIMapping,
    validateExcelFile,
    getSupportedFormats,
    previewExcelData,

    // Utilities
    clearError,
    clearResponse,
    clearAiMapping,
    clearValidation,
    reset,
  };
}