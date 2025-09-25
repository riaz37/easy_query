import { useState } from "react";
import { ServiceRegistry } from "../api";
import type { SearchQueryParams } from "../api";

/**
 * Hook for making file queries using standardized ServiceRegistry
 */
export function useFileQuery() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<any>(null);
  const [rawResponse, setRawResponse] = useState<any>(null);

  /**
   * Send a query to the API
   */
  const sendQuery = async (params: SearchQueryParams & { user_id?: string }) => {
    setLoading(true);
    setError(null);
    setResponse(null);
    setRawResponse(null);

    try {
      // Ensure user_id is included in the search params
      const searchParams = {
        ...params,
        user_id: params.user_id, // Pass user_id to the API
      };
      
      const result = await ServiceRegistry.query.search(searchParams);
      
      if (result.success) {
        setRawResponse(result);
        setResponse(result.data?.answer || result.data);
      } else {
        throw new Error(result.error || "Query failed");
      }
    } catch (err: any) {
      setError(err.message || "Failed to process query");
    } finally {
      setLoading(false);
    }
  };

  return {
    loading,
    error,
    response,
    rawResponse,
    sendQuery,
    setResponse,
    clearError: () => setError(null),
  };
}
