import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";

/**
 * Conversation history item
 */
export interface ConversationHistoryItem {
  id: string;
  query: string;
  response: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

/**
 * Service for handling conversation history API calls
 * All methods use JWT authentication - user ID is extracted from token on backend
 */
export class HistoryService extends BaseService {
  protected readonly serviceName = 'HistoryService';

  /**
   * Fetch conversation history for the authenticated user
   * User ID is extracted from JWT token on backend
   */
  async getConversationHistory(userId: string): Promise<ServiceResponse<ConversationHistoryItem[]>> {
    if (!userId) {
      throw this.createValidationError('User ID is required');
    }
    
    const response = await this.get<any>(API_ENDPOINTS.CONVERSATION_HISTORY(userId));
    
    // Normalize the response data to ensure we return an array
    let historyData: ConversationHistoryItem[] = [];
    
    if (Array.isArray(response.data)) {
      historyData = response.data;
    } else if (response.data && typeof response.data === 'object') {
      // Handle different response structures
      if ('payload' in response.data && Array.isArray(response.data.payload)) {
        historyData = response.data.payload;
      } else if ('data' in response.data && Array.isArray(response.data.data)) {
        historyData = response.data.data;
      } else if ('history' in response.data && Array.isArray(response.data.history)) {
        historyData = response.data.history;
      }
    }

    return {
      data: historyData,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Clear conversation history for the authenticated user
   * User ID is extracted from JWT token on backend
   */
  async clearHistory(userId: string): Promise<ServiceResponse<void>> {
    if (!userId) {
      throw this.createValidationError('User ID is required');
    }
    
    return this.post<void>(API_ENDPOINTS.CLEAR_HISTORY(userId));
  }

  /**
   * Fetch query history for the authenticated user
   * This is an alias for getConversationHistory for backward compatibility
   */
  async fetchQueryHistory(userId: string): Promise<ServiceResponse<ConversationHistoryItem[]>> {
    return this.getConversationHistory(userId);
  }

  /**
   * Get conversation history with pagination
   */
  async getConversationHistoryPaginated(
    userId: string,
    page: number = 1,
    limit: number = 20
  ): Promise<ServiceResponse<{
    items: ConversationHistoryItem[];
    pagination: {
      page: number;
      limit: number;
      total: number;
      hasMore: boolean;
    };
  }>> {
    // For now, get all history and paginate client-side
    // In a real implementation, this would be handled server-side
    const allHistoryResponse = await this.getConversationHistory(userId);
    
    if (!allHistoryResponse.success) {
      return {
        data: {
          items: [],
          pagination: { page, limit, total: 0, hasMore: false },
        },
        success: false,
        error: allHistoryResponse.error,
        timestamp: allHistoryResponse.timestamp,
      };
    }

    const allItems = allHistoryResponse.data;
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const paginatedItems = allItems.slice(startIndex, endIndex);

    return {
      data: {
        items: paginatedItems,
        pagination: {
          page,
          limit,
          total: allItems.length,
          hasMore: endIndex < allItems.length,
        },
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Search conversation history by query text
   */
  async searchHistory(userId: string, searchTerm: string): Promise<ServiceResponse<ConversationHistoryItem[]>> {
    if (!searchTerm || typeof searchTerm !== 'string') {
      throw this.createValidationError('Search term must be a non-empty string');
    }

    const allHistoryResponse = await this.getConversationHistory(userId);
    
    if (!allHistoryResponse.success) {
      return allHistoryResponse;
    }

    const searchTermLower = searchTerm.toLowerCase();
    const filteredItems = allHistoryResponse.data.filter(item =>
      item.query.toLowerCase().includes(searchTermLower) ||
      item.response.toLowerCase().includes(searchTermLower)
    );

    return {
      data: filteredItems,
      success: true,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Get conversation history statistics
   */
  async getHistoryStats(): Promise<ServiceResponse<{
    totalConversations: number;
    averageResponseTime: number;
    mostCommonQueries: Array<{ query: string; count: number }>;
    recentActivity: Array<{ date: string; count: number }>;
  }>> {
    const allHistoryResponse = await this.getConversationHistory();
    
    if (!allHistoryResponse.success) {
      return {
        data: {
          totalConversations: 0,
          averageResponseTime: 0,
          mostCommonQueries: [],
          recentActivity: [],
        },
        success: false,
        error: allHistoryResponse.error,
        timestamp: allHistoryResponse.timestamp,
      };
    }

    const items = allHistoryResponse.data;
    
    // Calculate basic statistics
    const totalConversations = items.length;
    
    // Count query frequencies
    const queryCount = new Map<string, number>();
    items.forEach(item => {
      const query = item.query.toLowerCase();
      queryCount.set(query, (queryCount.get(query) || 0) + 1);
    });

    const mostCommonQueries = Array.from(queryCount.entries())
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([query, count]) => ({ query, count }));

    // Calculate recent activity (last 7 days)
    const now = new Date();
    const recentActivity = Array.from({ length: 7 }, (_, i) => {
      const date = new Date(now);
      date.setDate(date.getDate() - i);
      const dateStr = date.toISOString().split('T')[0];
      
      const count = items.filter(item => 
        item.timestamp.startsWith(dateStr)
      ).length;
      
      return { date: dateStr, count };
    }).reverse();

    return {
      data: {
        totalConversations,
        averageResponseTime: 0, // Would need response time data from API
        mostCommonQueries,
        recentActivity,
      },
      success: true,
      timestamp: new Date().toISOString(),
    };
  }
}

// Export singleton instance
export const historyService = new HistoryService();

// Export for backward compatibility
export default historyService;
