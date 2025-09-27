import { useState, useCallback, useRef } from "react";
import { ServiceRegistry } from "../api";

interface UseFileOperationsReturn {
  // File upload operations
  uploadToSmartFileSystem: (request: {
    files: File[];
    file_descriptions: string[];
    table_names: string[];
  }) => Promise<any>;

  // Bundle status operations
  getBundleTaskStatus: (bundleId: string) => Promise<any>;
  getAllBundleTaskStatuses: () => Promise<any>;

  // File search operations
  searchFiles: (request: {
    query: string;
    user_id?: string;
    intent_top_k?: number;
    chunk_top_k?: number;
    max_chunks_for_answer?: number;
  }) => Promise<any>;

  // State
  uploadResponse: any;
  bundleStatus: any;
  allBundleStatuses: any;
  searchResults: any;
  isLoading: boolean;
  error: string | null;
  isPolling: boolean;
  uploadProgress: number;

  // Utility functions
  clearError: () => void;
  reset: () => void;
}

/**
 * Custom hook for managing file operations with progress tracking and polling
 * Uses standardized ServiceRegistry
 */
export function useFileOperations(): UseFileOperationsReturn {
  const [uploadResponse, setUploadResponse] = useState<any>(null);
  const [bundleStatus, setBundleStatus] = useState<any>(null);
  const [allBundleStatuses, setAllBundleStatuses] = useState<any>(null);
  const [searchResults, setSearchResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const reset = useCallback(() => {
    stopPolling();
    setUploadResponse(null);
    setBundleStatus(null);
    setAllBundleStatuses(null);
    setSearchResults(null);
    setError(null);
    setUploadProgress(0);
  }, []);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setIsPolling(false);
  }, []);

  const startPolling = useCallback((bundleId: string, interval: number = 2000) => {
    if (pollingRef.current) {
      stopPolling();
    }

    setIsPolling(true);
    pollingRef.current = setInterval(async () => {
      try {
        const response = await ServiceRegistry.file.getBundleTaskStatus(bundleId);
        
        if (response.success) {
          setBundleStatus(response.data);
          
          // Stop polling if completed or failed
          if (response.data.status === 'completed' || response.data.status === 'failed') {
            stopPolling();
          }
        } else {
          console.error('Failed to get bundle status:', response.error);
          stopPolling();
        }
      } catch (err: any) {
        console.error('Error polling bundle status:', err);
        stopPolling();
      }
    }, interval);
  }, [stopPolling]);

  /**
   * Upload files to smart file system
   */
  const uploadToSmartFileSystem = useCallback(async (request: {
    files: File[];
    file_descriptions: string[];
    table_names: string[];
  }) => {
    setIsLoading(true);
    setError(null);
    setUploadProgress(0);
    setUploadResponse(null);

    try {
      // Validate files first
      const validationPromises = request.files.map(file => 
        ServiceRegistry.file.validateExcelFile(file)
      );
      
      const validationResults = await Promise.all(validationPromises);
      const hasValidationErrors = validationResults.some(result => 
        result && result.data && !result.data.isValid
      );

      if (hasValidationErrors) {
        const errors = validationResults
          .filter(result => result && result.data && !result.data.isValid)
          .map(result => result!.data.errors.join(', '))
          .join('; ');
        throw new Error(`File validation failed: ${errors}`);
      }

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + 10;
        });
      }, 200);

      const response = await ServiceRegistry.file.uploadToSmartFileSystem(request);
      
      clearInterval(progressInterval);
      setUploadProgress(100);

      if (response.success) {
        setUploadResponse(response.data);
        
        // Start polling if bundle_id is provided
        if (response.data.bundle_id) {
          startPolling(response.data.bundle_id);
        }
        
        return response.data;
      } else {
        throw new Error(response.error || "Upload failed");
      }
    } catch (e: any) {
      setError(e.message || "Upload failed");
      setUploadProgress(0);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [startPolling]);

  /**
   * Get bundle task status
   */
  const getBundleTaskStatus = useCallback(async (bundleId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.file.getBundleTaskStatus(bundleId);
      
      if (response.success) {
        setBundleStatus(response.data);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to get bundle status");
      }
    } catch (e: any) {
      setError(e.message || "Failed to get bundle status");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Get all bundle task statuses
   */
  const getAllBundleTaskStatuses = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.file.getAllBundleTaskStatuses();
      
      if (response.success) {
        setAllBundleStatuses(response.data);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to get all bundle statuses");
      }
    } catch (e: any) {
      setError(e.message || "Failed to get all bundle statuses");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Search files
   */
  const searchFiles = useCallback(async (request: {
    query: string;
    user_id?: string;
    intent_top_k?: number;
    chunk_top_k?: number;
    max_chunks_for_answer?: number;
  }) => {
    setIsLoading(true);
    setError(null);
    setSearchResults(null);

    try {
      // Ensure user_id is included in the search request
      const searchRequest = {
        ...request,
        user_id: request.user_id, // Pass user_id to the API
      };
      
      const response = await ServiceRegistry.file.searchFiles(searchRequest);
      
      if (response.success) {
        setSearchResults(response.data);
        return response.data;
      } else {
        throw new Error(response.error || "Search failed");
      }
    } catch (e: any) {
      setError(e.message || "Search failed");
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Get upload progress for a bundle
   */
  const getUploadProgress = useCallback(async (bundleId: string) => {
    try {
      const response = await ServiceRegistry.file.getUploadProgress(bundleId);
      
      if (response.success) {
        setUploadProgress(response.data.overallProgress);
        return response.data;
      } else {
        throw new Error(response.error || "Failed to get upload progress");
      }
    } catch (e: any) {
      setError(e.message || "Failed to get upload progress");
      return null;
    }
  }, []);

  /**
   * Get supported file types
   */
  const getSupportedFileTypes = useCallback(() => {
    const response = ServiceRegistry.file.getSupportedFileTypes();
    return response.data;
  }, []);

  return {
    // File operations
    uploadToSmartFileSystem,
    getBundleTaskStatus,
    getAllBundleTaskStatuses,
    searchFiles,
    getUploadProgress,
    getSupportedFileTypes,

    // State
    uploadResponse,
    bundleStatus,
    allBundleStatuses,
    searchResults,
    isLoading,
    error,
    isPolling,
    uploadProgress,

    // Utilities
    clearError,
    reset,
    startPolling,
    stopPolling,
  };
}

// Export with alias for backward compatibility
export const useSmartFileUpload = useFileOperations;
