/**
 * Storage utility functions for Easy Query application
 */

// Storage keys for Easy Query application
export const STORAGE_KEYS = {
  // Database context
  CURRENT_DATABASE: 'easy_query_current_database',
  AVAILABLE_DATABASES: 'easy_query_available_databases',
  USER_DATABASES: 'easy_query_user_databases',
  MSSQL_DATABASES: 'easy_query_mssql_databases',
  
  // Business rules context
  BUSINESS_RULES: 'easy_query_business_rules',
  
  // Auth (if needed)
  AUTH_TOKENS: 'auth_tokens',
  AUTH_USER: 'auth_user',
} as const;

/**
 * Clear all Easy Query-related storage for a specific user
 */
export function clearEasyQueryStorage(userId?: string): void {
  try {
    if (userId) {
      // Clear user-specific storage
      Object.values(STORAGE_KEYS).forEach(key => {
        localStorage.removeItem(`${key}_${userId}`);
      });
    } else {
      // Clear all Easy Query storage (for logout)
      Object.keys(localStorage).forEach(key => {
        if (key.startsWith('easy_query_')) {
          localStorage.removeItem(key);
        }
      });
    }
    console.log('Easy Query storage cleared successfully');
  } catch (error) {
    console.error('Failed to clear Easy Query storage:', error);
  }
}

/**
 * Clear all Easy Query storage (used during logout)
 */
export function clearAllEasyQueryStorage(): void {
  clearEasyQueryStorage();
}

/**
 * Get storage key for a specific user
 */
export function getUserStorageKey(key: string, userId: string): string {
  return `${key}_${userId}`;
}

/**
 * Save data to user-specific storage
 */
export function saveToUserStorage<T>(key: string, userId: string, data: T): void {
  try {
    localStorage.setItem(getUserStorageKey(key, userId), JSON.stringify(data));
  } catch (error) {
    console.error(`Failed to save ${key} to storage:`, error);
  }
}

/**
 * Load data from user-specific storage
 */
export function loadFromUserStorage<T>(key: string, userId: string): T | null {
  try {
    const stored = localStorage.getItem(getUserStorageKey(key, userId));
    return stored ? JSON.parse(stored) : null;
  } catch (error) {
    console.error(`Failed to load ${key} from storage:`, error);
    return null;
  }
} 