"use client";

import { useThemeTransition } from '@/lib/hooks/useThemeTransition';

interface ThemeTransitionProviderProps {
  children: React.ReactNode;
}

/**
 * Provider component that handles theme transition setup
 */
export function ThemeTransitionProvider({ children }: ThemeTransitionProviderProps) {
  useThemeTransition();
  
  return <>{children}</>;
}
