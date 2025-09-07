"use client";

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { LoadingContextType, LoadingState } from './types';

const LoadingContext = createContext<LoadingContextType | undefined>(undefined);

interface LoadingProviderProps {
  children: ReactNode;
}

export function LoadingProvider({ children }: LoadingProviderProps) {
  const [loadingStates, setLoadingStates] = useState<Record<string, LoadingState>>({});

  const setLoading = useCallback((key: string, state: LoadingState) => {
    setLoadingStates(prev => ({
      ...prev,
      [key]: state
    }));
  }, []);

  const clearLoading = useCallback((key: string) => {
    setLoadingStates(prev => {
      const newStates = { ...prev };
      delete newStates[key];
      return newStates;
    });
  }, []);

  const isAnyLoading = Object.values(loadingStates).some(state => state.isLoading);

  const value: LoadingContextType = {
    loadingStates,
    setLoading,
    clearLoading,
    isAnyLoading
  };

  return (
    <LoadingContext.Provider value={value}>
      {children}
    </LoadingContext.Provider>
  );
}

export function useLoading() {
  const context = useContext(LoadingContext);
  if (context === undefined) {
    throw new Error('useLoading must be used within a LoadingProvider');
  }
  return context;
}
