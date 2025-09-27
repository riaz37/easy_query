// Error handling types and interfaces

export interface AppError {
  code: string;
  message: string;
  details?: Record<string, any>;
  statusCode?: number;
  timestamp: Date;
  stack?: string;
}

export interface ValidationError extends AppError {
  field: string;
  value: any;
  constraint: string;
}

export interface NetworkError extends AppError {
  url: string;
  method: string;
  timeout?: boolean;
  offline?: boolean;
}

export interface FileError extends AppError {
  fileName: string;
  fileSize?: number;
  fileType?: string;
  reason: 'size_exceeded' | 'invalid_type' | 'upload_failed' | 'processing_failed';
}

export interface AuthError extends AppError {
  reason: 'unauthorized' | 'forbidden' | 'token_expired' | 'invalid_credentials';
}

// Error boundary types
export interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: {
    componentStack: string;
  };
}

export interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<ErrorFallbackProps>;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  resetOnPropsChange?: boolean;
  resetKeys?: Array<string | number>;
}

export interface ErrorFallbackProps {
  error: Error;
  resetError: () => void;
  errorInfo?: {
    componentStack: string;
  };
}

// Error handler function types
export type ErrorHandler = (error: AppError) => void;

export type AsyncErrorHandler<T = any> = (error: AppError) => Promise<T>;

// Error severity levels
export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

// Error categories for logging and monitoring
export type ErrorCategory = 
  | 'network'
  | 'validation'
  | 'authentication'
  | 'authorization'
  | 'file_upload'
  | 'data_processing'
  | 'ui_component'
  | 'unknown';

// Enhanced error interface with additional metadata
export interface EnhancedError extends AppError {
  severity: ErrorSeverity;
  category: ErrorCategory;
  userId?: string;
  sessionId?: string;
  userAgent?: string;
  url?: string;
  retryable: boolean;
  retryCount?: number;
  context?: Record<string, any>;
}

// Error reporting configuration
export interface ErrorReportingConfig {
  enabled: boolean;
  endpoint?: string;
  apiKey?: string;
  includeStackTrace: boolean;
  includeUserInfo: boolean;
  sampleRate: number;
  ignoredErrors: string[];
}
  
