import { useState, useCallback, useEffect, useRef } from 'react';
import { ServiceRegistry } from '@/lib/api/services/service-registry';
import { UserTasksResponse, UserTask, GetUserTasksRequest } from '@/types/reports';

interface UseUserTasksState {
  userTasks: UserTasksResponse | null;
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

interface UseUserTasksReturn extends UseUserTasksState {
  // Actions
  fetchUserTasks: (request: GetUserTasksRequest) => Promise<void>;
  fetchUserTasksWithLimit: (userId: string, limit?: number) => Promise<void>;
  fetchAllUserTasks: (userId: string) => Promise<void>;
  fetchUserTasksByStatus: (userId: string, status: UserTask['status']) => Promise<void>;
  
  // Utility methods
  refresh: () => Promise<void>;
  clearError: () => void;
  reset: () => void;
  
  // Computed values
  completedTasks: UserTask[];
  pendingTasks: UserTask[];
  processingTasks: UserTask[];
  failedTasks: UserTask[];
  totalTasksCount: number;
  hasTasks: boolean;
}

/**
 * Hook for managing user tasks
 */
export function useUserTasks(): UseUserTasksReturn {
  const [state, setState] = useState<UseUserTasksState>({
    userTasks: null,
    isLoading: false,
    error: null,
    lastUpdated: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  /**
   * Fetch user tasks with custom request parameters
   */
  const fetchUserTasks = useCallback(async (request: GetUserTasksRequest): Promise<void> => {
    try {
      // Cancel any ongoing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      // Create new abort controller
      abortControllerRef.current = new AbortController();

      setState(prev => ({
        ...prev,
        isLoading: true,
        error: null,
      }));

      const response = await ServiceRegistry.userTasks.getUserTasks(request);
      
      setState(prev => ({
        ...prev,
        userTasks: response,
        isLoading: false,
        lastUpdated: new Date().toISOString(),
      }));
    } catch (error) {
      // Don't update state if request was cancelled
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }

      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to fetch user tasks',
      }));
    }
  }, []);

  /**
   * Fetch user tasks with a specific limit
   */
  const fetchUserTasksWithLimit = useCallback(async (userId: string, limit: number = 10): Promise<void> => {
    await fetchUserTasks({ userId, limit });
  }, [fetchUserTasks]);

  /**
   * Fetch all user tasks (no limit)
   */
  const fetchAllUserTasks = useCallback(async (userId: string): Promise<void> => {
    await fetchUserTasks({ userId });
  }, [fetchUserTasks]);

  /**
   * Fetch user tasks by status
   */
  const fetchUserTasksByStatus = useCallback(async (userId: string, status: UserTask['status']): Promise<void> => {
    await fetchUserTasks({ userId, status });
  }, [fetchUserTasks]);

  /**
   * Refresh current user tasks
   */
  const refresh = useCallback(async (): Promise<void> => {
    if (state.userTasks) {
      await fetchUserTasks({ userId: state.userTasks.user_id });
    }
  }, [fetchUserTasks, state.userTasks]);

  /**
   * Clear error state
   */
  const clearError = useCallback((): void => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  /**
   * Reset hook state
   */
  const reset = useCallback((): void => {
    setState({
      userTasks: null,
      isLoading: false,
      error: null,
      lastUpdated: null,
    });
  }, []);

  // Computed values
  const completedTasks = state.userTasks?.tasks.filter(task => task.status === 'completed') || [];
  const pendingTasks = state.userTasks?.tasks.filter(task => task.status === 'pending') || [];
  const processingTasks = state.userTasks?.tasks.filter(task => task.status === 'processing') || [];
  const failedTasks = state.userTasks?.tasks.filter(task => task.status === 'failed') || [];
  const totalTasksCount = state.userTasks?.total_tasks || 0;
  const hasTasks = totalTasksCount > 0;

  return {
    ...state,
    fetchUserTasks,
    fetchUserTasksWithLimit,
    fetchAllUserTasks,
    fetchUserTasksByStatus,
    refresh,
    clearError,
    reset,
    completedTasks,
    pendingTasks,
    processingTasks,
    failedTasks,
    totalTasksCount,
    hasTasks,
  };
} 