import { BaseService } from './base';
import { API_ENDPOINTS, buildEndpointWithQueryParams } from '../endpoints';
import { UserTasksResponse, GetUserTasksRequest } from '@/types/reports';

/**
 * Service for managing user tasks
 */
export class UserTasksService extends BaseService {
  protected readonly serviceName = 'UserTasksService';

  /**
   * Get user tasks with optional filtering and pagination
   */
  async getUserTasks(request: GetUserTasksRequest): Promise<UserTasksResponse> {
    const { userId, limit, offset, status } = request;
    
    // Build query parameters
    const queryParams: Record<string, any> = {};
    if (limit !== undefined) queryParams.limit = limit;
    if (offset !== undefined) queryParams.offset = offset;
    if (status !== undefined) queryParams.status = status;

    const endpoint = buildEndpointWithQueryParams(
      API_ENDPOINTS.GET_USER_TASKS(userId),
      queryParams
    );

    const response = await this.get<UserTasksResponse>(endpoint);
    return response.data;
  }

  /**
   * Get user tasks with a specific limit
   */
  async getUserTasksWithLimit(userId: string, limit: number = 10): Promise<UserTasksResponse> {
    return this.getUserTasks({ userId, limit });
  }

  /**
   * Get all user tasks (no limit)
   */
  async getAllUserTasks(userId: string): Promise<UserTasksResponse> {
    return this.getUserTasks({ userId });
  }

  /**
   * Get user tasks by status
   */
  async getUserTasksByStatus(userId: string, status: UserTasksResponse['tasks'][0]['status']): Promise<UserTasksResponse> {
    return this.getUserTasks({ userId, status });
  }
}

// Export singleton instance
export const userTasksService = new UserTasksService(); 