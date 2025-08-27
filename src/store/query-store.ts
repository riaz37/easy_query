import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { queryService } from '@/lib/api/services/query-service';
import { fileService } from '@/lib/api/services/file-service';
import { businessRulesService } from '@/lib/api/services/business-rules-service';
import { BusinessRulesValidator } from '@/lib/utils/business-rules-validator';
import { historyService } from '@/lib/api/services/history-service';

// Query Types
export interface QueryRequest {
  id: string;
  type: 'file' | 'database';
  query: string;
  userId: string;
  timestamp: Date;
  parameters?: Record<string, any>;
}

export interface QueryResult {
  id: string;
  queryId: string;
  data: any;
  metadata: {
    rowCount: number;
    executionTime: number;
    columns: string[];
  };
  timestamp: Date;
  status: 'success' | 'error' | 'pending';
  error?: string;
}

export interface QueryHistoryItem {
  id: string;
  query: string;
  type: 'file' | 'database';
  userId: string;
  timestamp: Date;
  status: 'success' | 'error' | 'pending';
  executionTime?: number;
  rowCount?: number;
}

export interface FileQueryRequest {
  fileId: string;
  query: string;
  userId: string;
  parameters?: {
    table_specific?: boolean;
    tables?: string[];
    use_intent_reranker?: boolean;
    use_chunk_reranker?: boolean;
    use_dual_embeddings?: boolean;
    intent_top_k?: number;
    chunk_top_k?: number;
    chunk_source?: string;
    max_chunks_for_answer?: number;
    answer_style?: string;
  };
}

export interface DatabaseQueryRequest {
  databaseId: number;
  query: string;
  userId: string;
  parameters?: Record<string, any>;
}

export interface SavedQuery {
  id: string;
  name: string;
  description: string;
  query: string;
  type: 'file' | 'database';
  userId: string;
  createdAt: Date;
  updatedAt: Date;
  tags: string[];
}

// Query Store Interface
interface QueryStore {
  // State
  currentQuery: QueryRequest | null;
  queryResults: QueryResult | null;
  queryLoading: boolean;
  queryError: string | null;
  
  // File Queries
  fileQueries: FileQueryRequest[];
  fileQueryHistory: QueryHistoryItem[];
  
  // Database Queries
  databaseQueries: DatabaseQueryRequest[];
  databaseQueryHistory: QueryHistoryItem[];
  
  // Saved Queries
  savedQueries: SavedQuery[];
  
  // Actions
  setCurrentQuery: (query: QueryRequest | null) => void;
  setQueryResults: (results: QueryResult | null) => void;
  setQueryLoading: (loading: boolean) => void;
  setQueryError: (error: string | null) => void;
  
  // Query Execution
  executeFileQuery: (query: FileQueryRequest) => Promise<void>;
  executeDatabaseQuery: (query: DatabaseQueryRequest) => Promise<void>;
  
  // Query Management
  saveQuery: (query: Omit<SavedQuery, 'id' | 'createdAt' | 'updatedAt'>) => Promise<void>;
  loadQueryHistory: (userId: string, type?: 'file' | 'database') => Promise<void>;
  clearQueryResults: () => void;
  
  // History Management
  addToHistory: (item: QueryHistoryItem) => void;
  clearHistory: (userId: string) => void;
  
  // Saved Queries Management
  loadSavedQueries: (userId: string) => Promise<void>;
  deleteSavedQuery: (queryId: string) => Promise<void>;
  updateSavedQuery: (queryId: string, updates: Partial<SavedQuery>) => Promise<void>;
}

export const useQueryStore = create<QueryStore>()(
  devtools(
    (set, get) => ({
      // Initial State
      currentQuery: null,
      queryResults: null,
      queryLoading: false,
      queryError: null,
      fileQueries: [],
      fileQueryHistory: [],
      databaseQueries: [],
      databaseQueryHistory: [],
      savedQueries: [],

      // Basic Setters
      setCurrentQuery: (query) => set({ currentQuery: query }),
      setQueryResults: (results) => set({ queryResults: results }),
      setQueryLoading: (loading) => set({ queryLoading: loading }),
      setQueryError: (error) => set({ queryError: error }),

      // Query Execution - Using REAL APIs
      executeFileQuery: async (queryRequest) => {
        const { setQueryLoading, setQueryError, setQueryResults, addToHistory } = get();
        
        try {
          setQueryLoading(true);
          setQueryError(null);
          
          // Create query request
          const query: QueryRequest = {
            id: Math.random().toString(36).substr(2, 9),
            type: 'file',
            query: queryRequest.query,
            userId: queryRequest.userId, // Keep for compatibility but not used in API
            timestamp: new Date(),
            parameters: queryRequest.parameters,
          };
          
          // Use authenticated FileService.searchFiles API with user_id
          const searchParams: any = {
            query: queryRequest.query,
            user_id: queryRequest.userId, // Pass user_id from the request
            intent_top_k: 20,
            chunk_top_k: 40,
            max_chunks_for_answer: 40,
            use_intent_reranker: false,
            use_chunk_reranker: false,
            use_dual_embeddings: true,
            chunk_source: "reranked",
            answer_style: "detailed",
            table_specific: false,
          };

          // Add table-specific parameters if provided
          if (queryRequest.parameters) {
            if (queryRequest.parameters.table_specific !== undefined) {
              searchParams.table_specific = queryRequest.parameters.table_specific;
            }
            if (queryRequest.parameters.tables && queryRequest.parameters.tables.length > 0) {
              searchParams.tables = queryRequest.parameters.tables;
            }
          }

          const result = await fileService.searchFiles(searchParams);
          
          // Transform API response to our QueryResult format
          const queryResult: QueryResult = {
            id: query.id,
            queryId: query.id,
            data: result.data || [],
            success: result.success,
            error: result.error,
            timestamp: new Date(),
            metadata: {
              executionTime: 0, // Would come from backend
              rowCount: result.data?.length || 0,
            },
          };
          
          setQueryResults(queryResult);
          addToHistory(query);
          
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'File query failed';
          console.error('File query error:', error);
          setQueryError(errorMessage);
          
          // Add failed query to history
          addToHistory({
            id: Math.random().toString(36).substr(2, 9),
            query: queryRequest.query,
            type: 'file',
            userId: queryRequest.userId,
            timestamp: new Date(),
            status: 'error',
          });
        } finally {
          setQueryLoading(false);
        }
      },

      executeDatabaseQuery: async (queryRequest) => {
        const { setQueryLoading, setQueryError, setQueryResults, addToHistory } = get();
        
        try {
          setQueryLoading(true);
          setQueryError(null);
          
          // Validate query against business rules first
          try {
            // Business rules validation is now handled by context providers
            // No need to manually fetch business rules here
          } catch (error) {
            console.warn('Failed to validate against business rules:', error);
            // Continue with query execution even if business rules validation fails
          }
          
          // Create query request
          const query: QueryRequest = {
            id: Math.random().toString(36).substr(2, 9),
            type: 'database',
            query: queryRequest.query,
            userId: queryRequest.userId, // Keep for compatibility but not used in API
            timestamp: new Date(),
            parameters: queryRequest.parameters,
          };
          
          // Use authenticated QueryService.sendDatabaseQuery API (userId required)
          const result = await queryService.sendDatabaseQuery(queryRequest.query, queryRequest.userId || 'default');
          
          // Transform API response to our QueryResult format
          const queryResult: QueryResult = {
            id: query.id,
            queryId: query.id,
            data: result.data || [],
            success: result.success,
            error: result.error,
            timestamp: new Date(),
            metadata: {
              executionTime: 0, // Would come from backend
              rowCount: result.data?.length || 0,
            },
          };
          
          setQueryResults(queryResult);
          addToHistory(query);
          
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Database query failed';
          console.error('Database query error:', error);
          setQueryError(errorMessage);
          
          // Add failed query to history
          addToHistory({
            id: Math.random().toString(36).substr(2, 9),
            query: queryRequest.query,
            type: 'database',
            userId: queryRequest.userId,
            timestamp: new Date(),
            status: 'error',
          });
        } finally {
          setQueryLoading(false);
        }
      },

      // Query Management - Currently no backend APIs for these, so we'll store locally
      saveQuery: async (queryData) => {
        const { savedQueries } = get();
        const newQuery: SavedQuery = {
          ...queryData,
          id: Math.random().toString(36).substr(2, 9),
          createdAt: new Date(),
          updatedAt: new Date(),
        };
        
        set({ savedQueries: [...savedQueries, newQuery] });
        
        // TODO: When backend API is available, implement:
        // await QueryService.saveQuery(newQuery);
      },

      // History loading now uses authenticated services
      loadQueryHistory: async (userId, type) => {
        try {
          // Use authenticated history service (userId required)
          const historyResponse = await historyService.fetchQueryHistory(userId);
          
          // Check if the response was successful
          if (!historyResponse.success || !historyResponse.data) {
            console.warn('History service returned unsuccessful response:', historyResponse);
            // Set empty arrays for the requested type
            if (type === 'file') {
              set({ fileQueryHistory: [] });
            } else if (type === 'database') {
              set({ databaseQueryHistory: [] });
            } else {
              set({ 
                fileQueryHistory: [],
                databaseQueryHistory: [],
              });
            }
            return;
          }
          
          // Extract the history data from the response
          let history = historyResponse.data;
          
          // Ensure history is an array
          if (!Array.isArray(history)) {
            console.warn('History service returned non-array data:', history);
            history = [];
          }
          
          // Filter by type and set appropriate history
          if (type === 'file') {
            const fileHistory = history.filter(item => item.type === 'file');
            set({ fileQueryHistory: fileHistory });
          } else if (type === 'database') {
            const dbHistory = history.filter(item => item.type === 'database');
            set({ databaseQueryHistory: dbHistory });
          } else {
            // Load all history
            const fileHistory = history.filter(item => item.type === 'file');
            const dbHistory = history.filter(item => item.type === 'database');
            set({ 
              fileQueryHistory: fileHistory,
              databaseQueryHistory: dbHistory,
            });
          }
        } catch (error) {
          console.error('Failed to load query history:', error);
          // Fallback to empty arrays
        if (type === 'file') {
          set({ fileQueryHistory: [] });
        } else if (type === 'database') {
          set({ databaseQueryHistory: [] });
        } else {
          set({ 
            fileQueryHistory: [],
            databaseQueryHistory: [],
          });
          }
        }
      },

      clearQueryResults: () => set({ queryResults: null, queryError: null }),

      // History Management - Now uses authenticated backend service
      addToHistory: (item) => {
        const { fileQueryHistory, databaseQueryHistory } = get();
        
        if (item.type === 'file') {
          set({ 
            fileQueryHistory: [item, ...fileQueryHistory.slice(0, 99)] // Keep last 100 items
          });
        } else {
          set({ 
            databaseQueryHistory: [item, ...databaseQueryHistory.slice(0, 99)] // Keep last 100 items
          });
        }
      },

      // Clear history now uses authenticated service
      clearHistory: async (userId) => {
        try {
          // Use authenticated history service (userId required)
          await historyService.clearHistory(userId);
          set({ 
            fileQueryHistory: [],
            databaseQueryHistory: [],
          });
        } catch (error) {
          console.error('Failed to clear history:', error);
          // Still clear local state even if backend fails
        set({ 
          fileQueryHistory: [],
          databaseQueryHistory: [],
        });
        }
      },

      // Saved Queries Management - Local state management
      loadSavedQueries: async (userId) => {
        // Currently no backend API for saved queries, so we'll use local state
        // TODO: When backend API is available, implement:
        // const savedQueries = await QueryService.getSavedQueries(userId);
        
        // For now, just ensure we have empty array
        set({ savedQueries: [] });
      },

      deleteSavedQuery: async (queryId) => {
        try {
          // Currently no backend API for deleting saved queries
          // TODO: When backend API is available, implement:
          // await QueryService.deleteSavedQuery(queryId);
          
          const { savedQueries } = get();
          set({ savedQueries: savedQueries.filter(q => q.id !== queryId) });
        } catch (error) {
          console.error('Failed to delete saved query:', error);
          throw error;
        }
      },

      updateSavedQuery: async (queryId, updates) => {
        try {
          // Currently no backend API for updating saved queries
          // TODO: When backend API is available, implement:
          // await QueryService.updateSavedQuery(queryId, updates);
          
          const { savedQueries } = get();
          set({
            savedQueries: savedQueries.map(q => 
              q.id === queryId 
                ? { ...q, ...updates, updatedAt: new Date() }
                : q
            )
          });
        } catch (error) {
          console.error('Failed to update saved query:', error);
          throw error;
        }
      },
    }),
    {
      name: 'query-store',
    }
  )
); 