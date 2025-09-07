import { ReactNode } from 'react';

export type LoadingSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type LoadingVariant = 'primary' | 'primary-dark' | 'primary-light' | 'accent-blue' | 'accent-purple' | 'accent-orange' | 'success' | 'warning' | 'error' | 'info';
export type LoadingTheme = 'light' | 'dark' | 'auto';

export interface LoadingProps {
  size?: LoadingSize;
  variant?: LoadingVariant;
  theme?: LoadingTheme;
  className?: string;
  children?: ReactNode;
}

export interface ProgressLoadingProps extends LoadingProps {
  progress?: number;
  showPercentage?: boolean;
  animated?: boolean;
}

export interface SkeletonLoadingProps extends LoadingProps {
  lines?: number;
  width?: string | number;
  height?: string | number;
  rounded?: boolean;
}

export interface ButtonLoadingProps extends LoadingProps, React.ButtonHTMLAttributes<HTMLButtonElement> {
  loading?: boolean;
  disabled?: boolean;
  text?: string;
}

export interface OverlayLoadingProps extends LoadingProps {
  visible?: boolean;
  backdrop?: boolean;
  message?: string;
  zIndex?: number;
}

export interface PageLoadingProps extends LoadingProps {
  message?: string;
  description?: string;
  showProgress?: boolean;
  progress?: number;
}

export interface LoadingState {
  isLoading: boolean;
  message?: string;
  progress?: number;
  variant?: LoadingVariant;
}

export interface LoadingContextType {
  loadingStates: Record<string, LoadingState>;
  setLoading: (key: string, state: LoadingState) => void;
  clearLoading: (key: string) => void;
  isAnyLoading: boolean;
}
