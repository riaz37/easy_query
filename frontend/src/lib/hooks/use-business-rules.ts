import { useState, useCallback } from "react";
import { ServiceRegistry } from "../api";

export function useBusinessRules() {
  const [businessRulesText, setBusinessRulesText] = useState<string>("");
  const [businessRulesLoading, setBusinessRulesLoading] = useState(false);
  const [businessRulesError, setBusinessRulesError] = useState<string | null>(null);
  const [uploadingRules, setUploadingRules] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  
  // Task-based updates state
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<'pending' | 'running' | 'completed' | 'failed' | null>(null);
  const [taskProgress, setTaskProgress] = useState<string>("");

  const fetchBusinessRules = useCallback(async (databaseId: number) => {
    if (!databaseId) {
      setBusinessRulesError("Database ID is required");
      return;
    }

    setBusinessRulesLoading(true);
    setBusinessRulesError(null);

    try {
      const response = await ServiceRegistry.businessRules.getBusinessRules(databaseId);
      if (response.success) {
        setBusinessRulesText(response.data || "");
      } else {
        throw new Error(response.error || "Failed to fetch business rules");
      }
    } catch (e: any) {
      setBusinessRulesError(e.message || "Failed to fetch business rules");
      setBusinessRulesText("");
    } finally {
      setBusinessRulesLoading(false);
    }
  }, []);

  const fetchBusinessRulesForCurrentDatabase = useCallback(async () => {
    setBusinessRulesLoading(true);
    setBusinessRulesError(null);

    try {
      const response = await ServiceRegistry.businessRules.getBusinessRulesForCurrentDatabase();
      if (response.success) {
        setBusinessRulesText(response.data || "");
      } else {
        throw new Error(response.error || "Failed to fetch business rules");
      }
    } catch (e: any) {
      setBusinessRulesError(e.message || "Failed to fetch business rules");
      setBusinessRulesText("");
    } finally {
      setBusinessRulesLoading(false);
    }
  }, []);

  const updateBusinessRules = useCallback(
    async (content: string, databaseId: number, options: any = {}) => {
    if (!databaseId) {
        setUploadError("Database ID is required");
      return false;
    }

    setUploadingRules(true);
    setUploadError(null);
    setUploadSuccess(false);
    setTaskStatus(null);

      try {
        const response = await ServiceRegistry.businessRules.updateBusinessRules(content, databaseId, options);
        
        if (response.success) {
          // If response contains task information, track it
          if (response.data?.task_id) {
            setCurrentTaskId(response.data.task_id);
            setTaskStatus('pending');
            setTaskProgress("Business rules update initiated...");
            
            // Start polling for task status
            await pollTaskStatus(response.data.task_id);
          } else {
            // Immediate success
            setUploadSuccess(true);
            setBusinessRulesText(content);
          }
          return true;
        } else {
          throw new Error(response.error || "Failed to update business rules");
        }
      } catch (e: any) {
        setUploadError(e.message || "Failed to update business rules");
        setTaskStatus('failed');
        return false;
      } finally {
        setUploadingRules(false);
      }
    },
    []
  );

  const updateBusinessRulesForCurrentDatabase = useCallback(async (content: string) => {
    setUploadingRules(true);
    setUploadError(null);
    setUploadSuccess(false);

    try {
      const response = await ServiceRegistry.businessRules.updateBusinessRulesForCurrentDatabase(content);
      
      if (response.success) {
        setUploadSuccess(true);
        setBusinessRulesText(content);
        return true;
      } else {
        throw new Error(response.error || "Failed to update business rules");
      }
    } catch (e: any) {
        setUploadError(e.message || "Failed to update business rules");
      return false;
    } finally {
      setUploadingRules(false);
    }
  }, []);

  const pollTaskStatus = useCallback(async (taskId: string) => {
    const maxAttempts = 30; // 30 attempts with 2-second intervals = 1 minute max
    let attempts = 0;

    const checkStatus = async () => {
      try {
        attempts++;
        const response = await ServiceRegistry.businessRules.getTaskStatus(taskId);
        
        if (response.success) {
          const status = response.data;
          setTaskStatus(status.status as any);
          setTaskProgress(status.progress || "Processing...");

          if (status.status === 'completed') {
            setUploadSuccess(true);
            setCurrentTaskId(null);
            return;
        } else if (status.status === 'failed') {
            setUploadError(status.error_message || "Task failed");
            setCurrentTaskId(null);
            return;
          } else if (attempts < maxAttempts) {
            // Continue polling
            setTimeout(checkStatus, 2000);
          } else {
            // Max attempts reached
            setUploadError("Task polling timeout");
            setCurrentTaskId(null);
          }
        } else {
          throw new Error(response.error || "Failed to get task status");
        }
      } catch (e: any) {
        setUploadError(e.message || "Failed to check task status");
        setCurrentTaskId(null);
      }
    };

    checkStatus();
  }, []);

  const getAllBusinessRules = useCallback(async () => {
    setBusinessRulesLoading(true);
    setBusinessRulesError(null);

    try {
      const response = await ServiceRegistry.businessRules.getAllBusinessRules();
      
      if (response.success) {
        return response.data;
      } else {
        throw new Error(response.error || "Failed to fetch all business rules");
      }
    } catch (e: any) {
      setBusinessRulesError(e.message || "Failed to fetch all business rules");
      return [];
    } finally {
      setBusinessRulesLoading(false);
    }
  }, []);

  const validateBusinessRulesContent = useCallback((content: string) => {
    return ServiceRegistry.businessRules.validateBusinessRulesContent(content);
  }, []);

  const checkTaskStatus = useCallback(async (taskId: string) => {
    if (!taskId) return null;

    try {
      const response = await ServiceRegistry.businessRules.getTaskStatus(taskId);
      
      if (response.success) {
        setTaskStatus(response.data.status as any);
        setTaskProgress(response.data.progress || "");
        return response.data;
      } else {
        throw new Error(response.error || "Failed to get task status");
      }
    } catch (e: any) {
      setUploadError(e.message || "Failed to check task status");
      return null;
    }
  }, []);

  // Clear functions
  const clearError = useCallback(() => {
    setBusinessRulesError(null);
    setUploadError(null);
  }, []);

  const clearSuccess = useCallback(() => {
    setUploadSuccess(false);
  }, []);

  const clearTask = useCallback(() => {
    setCurrentTaskId(null);
    setTaskStatus(null);
    setTaskProgress("");
  }, []);

  const reset = useCallback(() => {
    setBusinessRulesText("");
    setBusinessRulesError(null);
    setUploadError(null);
    setUploadSuccess(false);
    setCurrentTaskId(null);
    setTaskStatus(null);
    setTaskProgress("");
  }, []);

  return {
    // State
    businessRulesText,
    businessRulesLoading,
    businessRulesError,
    uploadingRules,
    uploadError,
    uploadSuccess,
    currentTaskId,
    taskStatus,
    taskProgress,

    // Actions
    fetchBusinessRules,
    fetchBusinessRulesForCurrentDatabase,
    updateBusinessRules,
    updateBusinessRulesForCurrentDatabase,
    getAllBusinessRules,
    validateBusinessRulesContent,
    checkTaskStatus,

    // Utilities
    clearError,
    clearSuccess,
    clearTask,
    reset,
  };
}
