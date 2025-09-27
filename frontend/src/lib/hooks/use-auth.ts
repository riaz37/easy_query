import { useState, useEffect, useCallback, useMemo } from 'react';
import { ServiceRegistry } from '@/lib/api';
import { clearAllEasyQueryStorage } from '@/lib/utils/storage';
import type { 
  LoginRequest, 
  SignupRequest,
  ChangePasswordRequest,
  AuthResponse,
  UserProfile,
} from '@/lib/api';

interface AuthTokens {
  accessToken: string;
  tokenType: string;
  expiresIn?: number;
}

interface AuthState {
  user: UserProfile | null;
  tokens: AuthTokens | null;
  isLoading: boolean;
  error: string | null;
  isAuthenticated: boolean;
}

/**
 * Custom hook for managing authentication state and operations
 * Uses standardized AuthService from ServiceRegistry
 */
export function useAuth() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Check if user is authenticated
  const isAuthenticated = useMemo(() => {
    return !!user && !!tokens?.accessToken && !ServiceRegistry.auth.isTokenExpired(tokens.accessToken);
  }, [user, tokens]);

  // Initialize auth state from localStorage on mount
  useEffect(() => {
    const initializeAuth = () => {
      try {
        const storedTokens = localStorage.getItem('auth_tokens');
        const storedUser = localStorage.getItem('auth_user');
        
        if (storedTokens && storedUser) {
          const parsedTokens: AuthTokens = JSON.parse(storedTokens);
          const parsedUser: UserProfile = JSON.parse(storedUser);
          
          // Check if token is still valid
          if (!ServiceRegistry.auth.isTokenExpired(parsedTokens.accessToken)) {
            setTokens(parsedTokens);
            setUser(parsedUser);
          } else {
            // Token expired, clear storage
            clearAuthStorage();
          }
        }
      } catch (error) {
        console.error('Failed to initialize auth from localStorage:', error);
        clearAuthStorage();
      }
    };

    initializeAuth();
  }, []);

  // Save auth data to localStorage
  const saveAuthToStorage = useCallback((authTokens: AuthTokens, userData: UserProfile) => {
    try {
      localStorage.setItem('auth_tokens', JSON.stringify(authTokens));
      localStorage.setItem('auth_user', JSON.stringify(userData));
    } catch (error) {
      console.error('Failed to save auth data to localStorage:', error);
    }
  }, []);

  // Clear auth data from localStorage
  const clearAuthStorage = useCallback(() => {
    try {
      localStorage.removeItem('auth_tokens');
      localStorage.removeItem('auth_user');
    } catch (error) {
      console.error('Failed to clear auth data from localStorage:', error);
    }
  }, []);

  // Login function
  const login = useCallback(async (credentials: LoginRequest): Promise<void> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Login and get tokens
      const authResponse = await ServiceRegistry.auth.login(credentials);
      
      if (!authResponse.success) {
        throw new Error(authResponse.error || 'Login failed');
      }
      
      const authTokens: AuthTokens = {
        accessToken: authResponse.data.access_token,
        tokenType: authResponse.data.token_type,
        expiresIn: authResponse.data.expires_in,
      };

      // Get user profile
      const profileResponse = await ServiceRegistry.auth.getProfile(authTokens.accessToken);
      
      if (!profileResponse.success) {
        throw new Error(profileResponse.error || 'Failed to get user profile');
      }

      // Update state
      setTokens(authTokens);
      setUser(profileResponse.data);

      // Save to localStorage
      saveAuthToStorage(authTokens, profileResponse.data);
      
    } catch (err: any) {
      const errorMessage = err.message || 'Login failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [saveAuthToStorage]);

  // Signup function
  const signup = useCallback(async (userData: SignupRequest): Promise<void> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await ServiceRegistry.auth.signup(userData);
      
      if (!response.success) {
        throw new Error(response.error || 'Signup failed');
      }

      // After successful signup, automatically log in
      await login({
        username: userData.username,
        password: userData.password,
      });

    } catch (err: any) {
      const errorMessage = err.message || 'Signup failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [login]);

  // Logout function
  const logout = useCallback(async (): Promise<void> => {
    setIsLoading(true);

    try {
      // Clear local state
    setUser(null);
    setTokens(null);
      setError(null);

      // Clear storage
    clearAuthStorage();
    clearAllEasyQueryStorage();

    } catch (err: any) {
      console.error('Logout error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [clearAuthStorage]);

  // Change password function
  const changePassword = useCallback(async (passwordData: ChangePasswordRequest): Promise<void> => {
    if (!tokens?.accessToken) {
      throw new Error('No access token available');
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await ServiceRegistry.auth.changePassword(passwordData, tokens.accessToken);
      
      if (!response.success) {
        throw new Error(response.error || 'Password change failed');
      }

      // Password changed successfully - no need to update tokens as they remain valid

    } catch (err: any) {
      const errorMessage = err.message || 'Password change failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [tokens]);

  // Refresh user profile
  const refreshProfile = useCallback(async (): Promise<void> => {
    if (!tokens?.accessToken) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await ServiceRegistry.auth.getProfile(tokens.accessToken);
      
      if (!response.success) {
        throw new Error(response.error || 'Failed to refresh profile');
      }

      setUser(response.data);
      saveAuthToStorage(tokens, response.data);

    } catch (err: any) {
      const errorMessage = err.message || 'Failed to refresh profile';
      setError(errorMessage);
      
      // If token is invalid, logout
      if (err.statusCode === 401) {
        await logout();
      }
    } finally {
      setIsLoading(false);
    }
  }, [tokens, saveAuthToStorage, logout]);

  // Check token validity
  const checkTokenValidity = useCallback((): boolean => {
    if (!tokens?.accessToken) {
      return false;
    }

    return !ServiceRegistry.auth.isTokenExpired(tokens.accessToken);
  }, [tokens]);

  // Get user ID from token
  const getUserId = useCallback((): string | null => {
    if (!tokens?.accessToken) {
      return null;
    }

    return ServiceRegistry.auth.getUserIdFromToken(tokens.accessToken);
  }, [tokens]);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Auth state object
  const authState: AuthState = {
    user,
    tokens,
    isLoading,
    error,
    isAuthenticated,
  };

  return {
    // State
    ...authState,

    // Actions
    login,
    signup,
    logout,
    changePassword,
    refreshProfile,

    // Utilities
    checkTokenValidity,
    getUserId,
    clearError,
    clearAuthStorage,
  };
} 