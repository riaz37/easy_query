import { useState, useCallback, useEffect } from 'react';
import { ServiceRegistry } from '../api';
import { ReportHistoryItem, ReportFilterOptions } from '@/types/reports';

interface UseReportHistoryState {
  history: ReportHistoryItem[];
  isLoading: boolean;
  error: string | null;
  filters: ReportFilterOptions;
  totalCount: number;
}

interface UseReportHistoryReturn extends UseReportHistoryState {
  // History management
  loadHistory: (filters?: ReportFilterOptions) => Promise<void>;
  refreshHistory: () => Promise<void>;
  
  // Filtering
  updateFilters: (newFilters: Partial<ReportFilterOptions>) => void;
  clearFilters: () => void;
  
  // Utility methods
  getHistoryByStatus: (status: string) => ReportHistoryItem[];
  getHistoryByDateRange: (from: string, to: string) => ReportHistoryItem[];
  getHistoryByUser: (userId: string) => ReportHistoryItem[];
  deleteHistoryItem: (taskId: string) => Promise<void>;
  reset: () => void;
}

/**
 * Hook for managing report history
 */
export function useReportHistory(): UseReportHistoryReturn {
  const [state, setState] = useState<UseReportHistoryState>({
    history: [],
    isLoading: false,
    error: null,
    filters: {},
    totalCount: 0,
  });

  /**
   * Load report history with optional filters
   */
  const loadHistory = useCallback(async (filters?: ReportFilterOptions) => {
    try {
      setState(prev => ({
        ...prev,
        isLoading: true,
        error: null,
      }));

      const history = await ServiceRegistry.reports.getReportHistory(filters);
      
      setState(prev => ({
        ...prev,
        history,
        totalCount: history.length,
        isLoading: false,
        filters: filters || {},
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load report history';
      setState(prev => ({
        ...prev,
        error: errorMessage,
        isLoading: false,
      }));
      throw error;
    }
  }, []);

  /**
   * Refresh the current history with existing filters
   */
  const refreshHistory = useCallback(async () => {
    await loadHistory(state.filters);
  }, [loadHistory, state.filters]);

  /**
   * Update filters and reload history
   */
  const updateFilters = useCallback(async (newFilters: Partial<ReportFilterOptions>) => {
    const updatedFilters = { ...state.filters, ...newFilters };
    await loadHistory(updatedFilters);
  }, [loadHistory, state.filters]);

  /**
   * Clear all filters and reload history
   */
  const clearFilters = useCallback(async () => {
    await loadHistory({});
  }, [loadHistory]);

  /**
   * Get history items filtered by status
   */
  const getHistoryByStatus = useCallback((status: string): ReportHistoryItem[] => {
    return state.history.filter(item => item.status === status);
  }, [state.history]);

  /**
   * Get history items filtered by date range
   */
  const getHistoryByDateRange = useCallback((from: string, to: string): ReportHistoryItem[] => {
    const fromDate = new Date(from);
    const toDate = new Date(to);
    
    return state.history.filter(item => {
      const itemDate = new Date(item.created_at);
      return itemDate >= fromDate && itemDate <= toDate;
    });
  }, [state.history]);

  /**
   * Get history items filtered by user
   */
  const getHistoryByUser = useCallback((userId: string): ReportHistoryItem[] => {
    return state.history.filter(item => item.user_id === userId);
  }, [state.history]);

  /**
   * Delete a history item
   */
  const deleteHistoryItem = useCallback(async (taskId: string): Promise<void> => {
    try {
      await ServiceRegistry.reports.deleteReportTask(taskId);
      
      // Remove from local state
      setState(prev => ({
        ...prev,
        history: prev.history.filter(item => item.id !== taskId),
        totalCount: prev.totalCount - 1,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete history item';
      setState(prev => ({
        ...prev,
        error: errorMessage,
      }));
      throw error;
    }
  }, []);

  /**
   * Reset the hook state
   */
  const reset = useCallback(() => {
    setState({
      history: [],
      isLoading: false,
      error: null,
      filters: {},
      totalCount: 0,
    });
  }, []);

  // Load initial history on mount
  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  return {
    ...state,
    loadHistory,
    refreshHistory,
    updateFilters,
    clearFilters,
    getHistoryByStatus,
    getHistoryByDateRange,
    getHistoryByUser,
    deleteHistoryItem,
    reset,
  };
} 