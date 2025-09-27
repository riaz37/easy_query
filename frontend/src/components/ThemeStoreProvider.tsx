"use client";

import React, { useEffect } from 'react';
import { useThemeStore } from '@/store/theme-store';

interface ThemeStoreProviderProps {
  children: React.ReactNode;
}

export function ThemeStoreProvider({ children }: ThemeStoreProviderProps) {
  const initializeTheme = useThemeStore((state) => state.initializeTheme);
  const mounted = useThemeStore((state) => state.mounted);

  useEffect(() => {
    initializeTheme();
  }, [initializeTheme]);

  // Prevent hydration mismatch by not rendering until mounted
  if (!mounted) {
    return <div style={{ visibility: 'hidden' }}>{children}</div>;
  }

  return <>{children}</>;
}