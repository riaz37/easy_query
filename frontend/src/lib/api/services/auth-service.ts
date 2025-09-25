import { API_ENDPOINTS } from "../endpoints";
import { BaseService, ServiceResponse } from "./base";

/**
 * Login request interface
 */
export interface LoginRequest {
  username: string;
  password: string;
}

/**
 * Signup request interface
 */
export interface SignupRequest {
  username: string;
  email: string;
  password: string;
  confirmPassword?: string;
}

/**
 * Change password request interface
 */
export interface ChangePasswordRequest {
  currentPassword: string;
  newPassword: string;
}

/**
 * Authentication response interface
 */
export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in?: number;
}

/**
 * User profile interface
 */
export interface UserProfile {
  user_id: string;
  username: string;
  email: string;
  created_at: string;
  updated_at: string;
}

/**
 * JWT payload interface
 */
export interface JWTPayload {
  sub: string;
  exp: number;
  iat: number;
  user_id: string;
  username: string;
}

/**
 * Service for handling authentication-related API calls
 */
export class AuthService extends BaseService {
  protected readonly serviceName = 'AuthService';

  /**
   * User signup
   */
  async signup(request: SignupRequest): Promise<ServiceResponse<AuthResponse>> {
    this.validateRequired(request, ['username', 'email', 'password']);
    this.validateTypes(request, {
      username: 'string',
      email: 'string',
      password: 'string',
    });

    // Validate signup data
    const validation = this.validateSignupData(request);
    if (!validation.isValid) {
      throw this.createValidationError(
        `Signup validation failed: ${validation.errors.join(', ')}`,
        { validationErrors: validation.errors }
      );
    }

    try {
      return await this.post<AuthResponse>(API_ENDPOINTS.AUTH_SIGNUP, {
        username: request.username.trim(),
        email: request.email.toLowerCase().trim(),
        password: request.password,
      });
    } catch (error: any) {
      throw this.handleAuthError(error, 'signup');
    }
  }

  /**
   * User login
   */
  async login(request: LoginRequest): Promise<ServiceResponse<AuthResponse>> {
    this.validateRequired(request, ['username', 'password']);
    this.validateTypes(request, {
      username: 'string',
      password: 'string',
    });

    if (request.username.trim().length === 0) {
      throw this.createValidationError('Username cannot be empty');
    }

    if (request.password.length === 0) {
      throw this.createValidationError('Password cannot be empty');
    }

    try {
      return await this.post<AuthResponse>(API_ENDPOINTS.AUTH_LOGIN, {
        username: request.username.trim(),
        password: request.password,
      });
    } catch (error: any) {
      throw this.handleAuthError(error, 'login');
    }
  }

  /**
   * Get user profile
   */
  async getProfile(accessToken?: string): Promise<ServiceResponse<UserProfile>> {
    const config = accessToken ? {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
      },
    } : undefined;

    try {
      const response = await this.get<UserProfile>(API_ENDPOINTS.AUTH_PROFILE, undefined, config);
      
      // Transform response to include userId alias for compatibility
      const transformedData = this.transformUserResponse(response.data);
      
      return {
        data: transformedData,
        success: true,
        timestamp: new Date().toISOString(),
      };
    } catch (error: any) {
      throw this.handleAuthError(error, 'profile');
    }
  }

  /**
   * Change user password
   */
  async changePassword(
    request: ChangePasswordRequest,
    accessToken: string
  ): Promise<ServiceResponse<void>> {
    this.validateRequired(request, ['currentPassword', 'newPassword']);
    this.validateTypes(request, {
      currentPassword: 'string',
      newPassword: 'string',
    });

    if (!accessToken) {
      throw this.createAuthError('Access token is required');
    }

    // Validate password strength
    const validation = this.validatePasswordStrength(request.newPassword);
    if (!validation.isValid) {
      throw this.createValidationError(
        `Password validation failed: ${validation.errors.join(', ')}`,
        { validationErrors: validation.errors }
      );
    }

    try {
      return await this.post<void>(
        API_ENDPOINTS.AUTH_CHANGE_PASSWORD,
        {
          current_password: request.currentPassword,
          new_password: request.newPassword,
        },
        {
          headers: {
            'Authorization': `Bearer ${accessToken}`,
          },
        }
      );
    } catch (error: any) {
      throw this.handleAuthError(error, 'change_password');
    }
  }

  /**
   * Parse JWT token
   */
  parseJWT(token: string): JWTPayload | null {
    try {
      if (!token) return null;
      
      const base64Url = token.split('.')[1];
      if (!base64Url) return null;
      
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const jsonPayload = decodeURIComponent(
        atob(base64)
          .split('')
          .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
          .join('')
      );
      
      return JSON.parse(jsonPayload) as JWTPayload;
    } catch (error) {
      console.error('Failed to parse JWT token:', error);
      return null;
    }
  }

  /**
   * Check if JWT token is expired
   */
  isTokenExpired(token: string): boolean {
    try {
      const payload = this.parseJWT(token);
      if (!payload || !payload.exp) return true;
      
      const currentTime = Math.floor(Date.now() / 1000);
      return payload.exp < currentTime;
    } catch (error) {
      return true;
    }
  }

  /**
   * Get user ID from JWT token
   */
  getUserIdFromToken(token: string): string | null {
    try {
      const payload = this.parseJWT(token);
      return payload?.user_id || payload?.sub || null;
    } catch (error) {
      console.error('Failed to extract user ID from token:', error);
      return null;
    }
  }

  /**
   * Transform API user response to include userId alias for compatibility
   */
  private transformUserResponse(userData: any): any {
    return {
      ...userData,
      userId: userData.user_id, // Add userId alias for compatibility with existing codebase
    };
  }

  /**
   * Handle authentication-specific errors
   */
  private handleAuthError(error: any, operation: string): Error {
    if (error.statusCode) {
      switch (error.statusCode) {
        case 400:
          if (operation === 'signup') {
            return this.createValidationError(error.message || 'Invalid signup data');
          }
          if (operation === 'login') {
            return this.createValidationError(error.message || 'Invalid credentials');
          }
          if (operation === 'change_password') {
            return this.createValidationError(error.message || 'Invalid password data');
          }
          return this.createValidationError(error.message || 'Bad request');
          
        case 401:
          if (operation === 'login') {
            return this.createAuthError('Invalid username or password');
          }
          return this.createAuthError('Authentication required');
          
        case 403:
          return this.createAuthorizationError('Access denied');
          
        case 409:
          if (operation === 'signup') {
            return this.createValidationError('Username or email already exists');
          }
          return this.createValidationError('Conflict with existing resource');
          
        case 422:
          return this.createValidationError(error.message || 'Validation error');
          
        case 500:
          return new Error('Internal server error');
          
        default:
          return new Error(error.message || 'Authentication failed');
      }
    }
    
    return error;
  }

  /**
   * Validate signup data
   */
  private validateSignupData(data: SignupRequest): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    // Username validation
    if (data.username.length < 3) {
      errors.push('Username must be at least 3 characters long');
    }
    if (data.username.length > 50) {
      errors.push('Username cannot be longer than 50 characters');
    }
    if (!/^[a-zA-Z0-9_]+$/.test(data.username)) {
      errors.push('Username can only contain letters, numbers, and underscores');
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(data.email)) {
      errors.push('Invalid email format');
    }

    // Password validation
    const passwordValidation = this.validatePasswordStrength(data.password);
    errors.push(...passwordValidation.errors);

    // Confirm password validation
    if (data.confirmPassword && data.password !== data.confirmPassword) {
      errors.push('Passwords do not match');
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate password strength
   */
  private validatePasswordStrength(password: string): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (password.length < 8) {
      errors.push('Password must be at least 8 characters long');
    }

    if (password.length > 128) {
      errors.push('Password cannot be longer than 128 characters');
    }

    if (!/(?=.*[a-z])/.test(password)) {
      errors.push('Password must contain at least one lowercase letter');
    }

    if (!/(?=.*[A-Z])/.test(password)) {
      errors.push('Password must contain at least one uppercase letter');
    }

    if (!/(?=.*\d)/.test(password)) {
      errors.push('Password must contain at least one number');
    }

    if (!/(?=.*[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?])/.test(password)) {
      errors.push('Password must contain at least one special character');
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }

  /**
   * Validate token format
   */
  validateTokenFormat(token: string): {
    isValid: boolean;
    errors: string[];
  } {
    const errors: string[] = [];

    if (!token) {
      errors.push('Token is required');
      return { isValid: false, errors };
    }

    if (typeof token !== 'string') {
      errors.push('Token must be a string');
      return { isValid: false, errors };
    }

    const parts = token.split('.');
    if (parts.length !== 3) {
      errors.push('Invalid JWT token format');
    }

    return {
      isValid: errors.length === 0,
      errors,
    };
  }
}

// Export singleton instance
export const authService = new AuthService();

// Export for backward compatibility
export default authService; 