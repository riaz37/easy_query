import { API_ENDPOINTS, buildEndpointWithQueryParams } from "../endpoints";
import { BaseService, ServiceResponse, ServiceError } from "./base";
import { transformResponse } from "../transformers";

/**
 * Search query parameters
 */
export interface SearchQueryParams {
  query: string;
  useIntentReranker?: boolean;
  useChunkReranker?: boolean;
  useDualEmbeddings?: boolean;
  intentTopK?: number;
  chunkTopK?: number;
  chunkSource?: string;
  maxChunksForAnswer?: number;
  answerStyle?: string;
}

/**
 * Database query parameters
 */
export interface DbQueryParams {
  question: string;
  userId?: string; // Added userId to DbQueryParams
  database_id?: number; // Added database_id for database selection
  model?: string; // Added model parameter for AI model selection
}

/**
 * Query result data structure
 */
export interface QueryResultData {
  results: any[];
  sql_query?: string;
  query_history?: any[];
  metadata?: Record<string, any>;
}

/**
 * Service for handling query-related API calls
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class QueryService extends BaseService {
  protected readonly serviceName = 'QueryService';

  /**
   * Send a search query to the API
   */
  async search(params: SearchQueryParams): Promise<ServiceResponse<QueryResultData>> {
    // Validate required parameters
    this.validateRequired(params, ['query']);
    this.validateTypes(params, {
      query: 'string',
      useIntentReranker: 'boolean',
      useChunkReranker: 'boolean',
      useDualEmbeddings: 'boolean',
      intentTopK: 'number',
      chunkTopK: 'number',
      chunkSource: 'string',
      maxChunksForAnswer: 'number',
      answerStyle: 'string',
    });

    const requestBody = {
      query: params.query,
      use_intent_reranker: params.useIntentReranker,
      use_chunk_reranker: params.useChunkReranker,
      use_dual_embeddings: params.useDualEmbeddings ?? true,
      intent_top_k: params.intentTopK ?? 20,
      chunk_top_k: params.chunkTopK ?? 40,
      chunk_source: params.chunkSource ?? "reranked",
      max_chunks_for_answer: params.maxChunksForAnswer ?? 40,
      answer_style: params.answerStyle ?? "detailed",
    };

    return this.post<QueryResultData>(API_ENDPOINTS.SEARCH, requestBody);
  }

  /**
   * Send a database query to the API
   * User ID is extracted from JWT token on backend
   */
  async query(params: DbQueryParams): Promise<ServiceResponse<QueryResultData>> {
    // Validate required parameters
    this.validateRequired(params, ['question']);
    this.validateTypes(params, { question: 'string' });

    if (!params.question.trim()) {
      throw this.createValidationError('Query question cannot be empty');
    }

    // Build the endpoint with query parameters as expected by the API
    const endpoint = buildEndpointWithQueryParams(API_ENDPOINTS.QUERY, {
      question: params.question,
      user_id: params.userId, // Add user_id parameter
      model: params.model, // Add model parameter
    });

    console.log("QueryService - Sending request to:", endpoint); // Debug log
    console.log("QueryService - Parameters:", { question: params.question, user_id: params.userId, model: params.model }); // Debug log

    // Send POST request with empty string body (matching curl format)
    const response = await this.post<QueryResultData>(endpoint, "");
    
    // Log the raw response to debug the issue
    console.log("QueryService - Raw API response:", response);
    
    return response;
  }

  /**
   * Send a database query directly with question string and user ID
   * User ID is required for database access
   */
  async sendDatabaseQuery(question: string, userId: string, model?: string): Promise<ServiceResponse<QueryResultData>> {
    if (!question || typeof question !== 'string') {
      throw this.createValidationError('Question must be a non-empty string');
    }

    if (!userId || typeof userId !== 'string') {
      throw this.createValidationError('User ID is required for database queries');
    }

    return this.query({ question, userId, model });
  }

  /**
   * Validate query content for potential security issues
   */
  validateQuerySecurity(query: string): { isValid: boolean; warnings: string[] } {
    const warnings: string[] = [];
    
    // Check for potential SQL injection patterns
    const suspiciousPatterns = [
      /;\s*drop\s+/i,
      /;\s*delete\s+from\s+/i,
      /;\s*truncate\s+/i,
      /;\s*alter\s+table\s+/i,
      /union\s+select/i,
      /exec\s*\(/i,
      /execute\s*\(/i,
    ];

    for (const pattern of suspiciousPatterns) {
      if (pattern.test(query)) {
        warnings.push('Query contains potentially dangerous SQL patterns');
        break;
      }
    }

    // Check for very long queries
    if (query.length > 10000) {
      warnings.push('Query is very long and may impact performance');
    }

    return {
      isValid: warnings.length === 0,
      warnings,
    };
  }

  /**
   * Get query execution statistics
   */
  async getQueryStats(): Promise<ServiceResponse<any>> {
    // This would typically call a dedicated stats endpoint
    // For now, return a placeholder
    return {
      data: {
        totalQueries: 0,
        averageExecutionTime: 0,
        successRate: 100,
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }
}

// Export singleton instance
export const queryService = new QueryService();

// Export for backward compatibility
export default queryService;
