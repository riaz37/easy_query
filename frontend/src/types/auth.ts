// Authentication-related types and interfaces

export interface User {
  user_id: string;
  username: string;
  email: string;
  is_active: boolean;
  created_at: string;
  updated_at: string | null;
}

export interface SignupRequest {
  username: string;
  email: string;
  password: string;
}

export interface SignupResponse {
  user_id: string;
  username: string;
  email: string;
  is_active: boolean;
  created_at: string;
  updated_at: string | null;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
}

export interface ChangePasswordResponse {
  message: string;
  success: boolean;
}

export interface AuthTokens {
  accessToken: string;
  tokenType: string;
  expiresAt?: number;
}

export interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

export interface AuthContextData extends AuthState {
  signup: (data: SignupRequest) => Promise<User>;
  login: (data: LoginRequest) => Promise<User>;
  logout: () => void;
  changePassword: (data: ChangePasswordRequest) => Promise<void>;
  refreshUser: () => Promise<void>;
  clearError: () => void;
}

// JWT Token payload structure (based on the token from the API)
export interface JWTPayload {
  sub: string; // username
  user_id: string;
  roles: string[];
  permissions: string[];
  exp: number; // expiration timestamp
  jti: string; // JWT ID
}

// Authentication error types
export type AuthErrorType = 
  | 'invalid_credentials'
  | 'user_not_found'
  | 'user_already_exists'
  | 'invalid_token'
  | 'token_expired'
  | 'insufficient_permissions'
  | 'account_disabled'
  | 'validation_error'
  | 'network_error'
  | 'unknown_error';

export interface AuthError {
  type: AuthErrorType;
  message: string;
  details?: Record<string, any>;
  statusCode?: number;
} 