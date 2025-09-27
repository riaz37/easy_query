import { useState } from "react";
import { ServiceRegistry } from "../api";
import type { DbQueryParams } from "../api";
import { useToast } from "./use-toast";

/**
 * Hook for managing database operations
 * All operations use JWT authentication - user ID is extracted from token on backend
 */
export function useDatabaseOperations() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dbResponse, setDbResponse] = useState<any>(null);
  const [reloadLoading, setReloadLoading] = useState(false);
  const [history, setHistory] = useState<any[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const { success, error: showError } = useToast();

  /**
   * Fetch query history for the authenticated user
   * User ID is extracted from JWT token on backend
   */
  const fetchQueryHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);

    try {
      const response = await ServiceRegistry.history.fetchQueryHistory();
      setHistory(response.data);
    } catch (e: any) {
      setHistoryError(e.message || "Unknown error");
    } finally {
      setHistoryLoading(false);
    }
  };

  /**
   * Clear query history for the authenticated user
   * User ID is extracted from JWT token on backend
   */
  const clearHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);

    try {
      await ServiceRegistry.history.clearHistory();
      setHistory([]);
    } catch (e: any) {
      setHistoryError(e.message || "Unknown error");
    } finally {
      setHistoryLoading(false);
    }
  };

  /**
   * Reload the database
   */
  const reloadDatabase = async () => {
    setReloadLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.database.reloadDatabase();
      success("Database reloaded successfully");
      return response;
    } catch (e: any) {
      const errorMessage = e.message || "Failed to reload database";
      setError(errorMessage);
      showError(errorMessage);
      throw e;
    } finally {
      setReloadLoading(false);
    }
  };

  /**
   * Send a database query
   */
  const sendQuery = async (params: DbQueryParams) => {
    setLoading(true);
    setError(null);

    try {
      // Get user ID from auth context or params
      const userId = params.userId || 'default';
      
      console.log("Database operations - sendQuery called with:", { params, userId }); // Debug log
      
      const response = await ServiceRegistry.query.sendDatabaseQuery(params.question, userId, params.model);
      setDbResponse(response);
      return response;
    } catch (e: any) {
      const errorMessage = e.message || "Query execution failed";
      setError(errorMessage);
      showError(errorMessage);
      throw e;
    } finally {
      setLoading(false);
    }
  };

  return {
    // State
    loading,
    error,
    dbResponse,
    reloadLoading,
    history,
    historyLoading,
    historyError,

    // Actions
    fetchQueryHistory,
    clearHistory,
    reloadDatabase,
    sendQuery,

    // Utilities
    clearError: () => setError(null),
    clearHistoryError: () => setHistoryError(null),
  };
}
