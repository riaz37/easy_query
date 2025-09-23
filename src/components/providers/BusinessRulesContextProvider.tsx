"use client";

import React, { createContext, useContext, ReactNode, useState, useCallback, useMemo, useEffect } from 'react';
import { useAuthContext } from './AuthContextProvider';
import { BusinessRulesValidator } from '@/lib/utils/business-rules-validator';
import { STORAGE_KEYS, saveToUserStorage, loadFromUserStorage, clearEasyQueryStorage } from '@/lib/utils/storage';

// Types for business rules context
export interface BusinessRulesState {
  content: string;
  status: 'none' | 'loading' | 'loaded' | 'error';
  error?: string;
  lastUpdated?: string;
  contentLength: number;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  suggestions: string[];
}

export interface BusinessRulesContextData {
  // State
  businessRules: BusinessRulesState;
  isLoading: boolean;
  error: string | null;
  
  // Actions (called by user-configuration page only)
  loadBusinessRulesFromConfig: (rules: string) => void;
  updateBusinessRules: (rules: string) => void;
  validateQuery: (query: string) => ValidationResult;
  clearBusinessRules: () => void;
  refreshBusinessRules: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Computed values
  hasBusinessRules: boolean;
  businessRulesCount: number;
}

const BusinessRulesContext = createContext<BusinessRulesContextData | undefined>(undefined);

export { BusinessRulesContext };

export function useBusinessRulesContext() {
  const context = useContext(BusinessRulesContext);
  if (!context) {
    throw new Error('useBusinessRulesContext must be used within a BusinessRulesContextProvider');
  }
  return context;
}

interface BusinessRulesContextProviderProps {
  children: ReactNode;
}

export function BusinessRulesContextProvider({ children }: BusinessRulesContextProviderProps) {
  const { user } = useAuthContext();

  // State
  const [businessRules, setBusinessRules] = useState<BusinessRulesState>({
    content: '',
    status: 'none',
    contentLength: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Memoized values
  const hasBusinessRules = useMemo(() => 
    businessRules.status === 'loaded' && businessRules.content.trim().length > 0, 
    [businessRules.status, businessRules.content]
  );

  const businessRulesCount = useMemo(() => 
    businessRules.content.split('\n').filter(line => line.trim().length > 0).length, 
    [businessRules.content]
  );

  // Load configuration from localStorage on mount
  useEffect(() => {
    if (user?.user_id) {
      loadConfigurationFromStorage();
    }
  }, [user?.user_id]);

  // Load configuration from localStorage
  const loadConfigurationFromStorage = useCallback(() => {
    try {
      console.log('Loading business rules from storage for user:', user?.user_id);
      const storedBusinessRules = loadFromUserStorage(STORAGE_KEYS.BUSINESS_RULES, user?.user_id || '');
      console.log('Stored business rules:', storedBusinessRules);
      if (storedBusinessRules) {
        setBusinessRules(storedBusinessRules);
        console.log('Business rules loaded from storage:', storedBusinessRules);
      } else {
        console.log('No business rules found in storage');
      }
    } catch (error) {
      console.error('Error loading business rules from storage:', error);
      // Clear corrupted storage
      clearEasyQueryStorage(user?.user_id);
    }
  }, [user?.user_id]);

  // Load business rules from user-configuration page
  const loadBusinessRulesFromConfig = useCallback((rules: string) => {
    console.log('Loading business rules from config:', rules);
    const hasContent = rules && rules.trim().length > 0;
    
    const newBusinessRules: BusinessRulesState = {
      content: rules || '',
      status: hasContent ? 'loaded' : 'none',
      contentLength: rules?.length || 0,
      lastUpdated: new Date().toISOString(),
    };
    
    console.log('Setting new business rules:', newBusinessRules);
    setBusinessRules(newBusinessRules);
    
    // Save to localStorage
    saveToUserStorage(STORAGE_KEYS.BUSINESS_RULES, user?.user_id || '', newBusinessRules);
    console.log('Business rules saved to storage');
  }, [user?.user_id]);

  // Update business rules (called by user-configuration page only)
  const updateBusinessRules = useCallback((rules: string) => {
    const hasContent = rules && rules.trim().length > 0;
    
    const newBusinessRules: BusinessRulesState = {
      content: rules,
      status: hasContent ? 'loaded' : 'none',
      contentLength: rules.length,
      lastUpdated: new Date().toISOString(),
    };
    
    setBusinessRules(newBusinessRules);
    
    // Save to localStorage
    saveToUserStorage(STORAGE_KEYS.BUSINESS_RULES, user?.user_id || '', newBusinessRules);
  }, [user?.user_id]);

  // Validate query against business rules
  const validateQuery = useCallback((query: string, databaseId?: string): ValidationResult => {
    console.log('Validating query:', query, 'against business rules. Status:', businessRules.status, 'Content length:', businessRules.content.length);
    
    if (businessRules.status !== 'loaded' || !businessRules.content) {
      console.log('No business rules loaded, allowing query');
      return {
        isValid: true, // Allow queries if no business rules are configured
        errors: [],
        warnings: [],
        suggestions: [],
      };
    }

    console.log('Business rules found, validating against:', businessRules.content.substring(0, 100) + '...');
    // Pass databaseId to validator for database-specific rules
    const result = BusinessRulesValidator.validateQuery(query.trim(), businessRules.content, databaseId);
    console.log('Validation result:', result);
    return result;
  }, [businessRules.status, businessRules.content]);

  // Clear business rules
  const clearBusinessRules = useCallback(() => {
    const clearedRules: BusinessRulesState = {
      content: '',
      status: 'none',
      contentLength: 0,
    };
    
    setBusinessRules(clearedRules);
    setError(null);
    
    // Save to localStorage
    saveToUserStorage(STORAGE_KEYS.BUSINESS_RULES, user?.user_id || '', clearedRules);
  }, [user?.user_id]);

  // Refresh business rules
  const refreshBusinessRules = useCallback(() => {
    // Re-render with current state
    setBusinessRules(prev => ({ ...prev }));
  }, []);

  // Set loading state
  const setLoadingState = useCallback((loading: boolean) => {
    setIsLoading(loading);
  }, []);

  // Set error state
  const setErrorState = useCallback((error: string | null) => {
    setError(error);
  }, []);

  // Memoized context value
  const contextValue = useMemo<BusinessRulesContextData>(() => ({
    businessRules,
    isLoading,
    error,
    loadBusinessRulesFromConfig,
    updateBusinessRules,
    validateQuery,
    clearBusinessRules,
    refreshBusinessRules,
    setLoading: setLoadingState,
    setError: setErrorState,
    hasBusinessRules,
    businessRulesCount,
  }), [
    businessRules,
    isLoading,
    error,
    loadBusinessRulesFromConfig,
    updateBusinessRules,
    validateQuery,
    clearBusinessRules,
    refreshBusinessRules,
    setLoadingState,
    setErrorState,
    hasBusinessRules,
    businessRulesCount,
  ]);

  return (
    <BusinessRulesContext.Provider value={contextValue}>
      {children}
    </BusinessRulesContext.Provider>
  );
} 