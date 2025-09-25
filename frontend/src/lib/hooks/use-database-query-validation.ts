import { useState, useCallback } from 'react';
import { BusinessRulesValidator, BusinessRuleValidation } from '@/lib/utils/business-rules-validator';
import { useBusinessRulesContext } from '@/components/providers';

/**
 * Hook for validating database queries against business rules
 * Now uses the Business Rules context for better state management
 */
export function useDatabaseQueryValidation() {
  const [validationResult, setValidationResult] = useState<BusinessRuleValidation | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Use the business rules context
  const { businessRules, validateQuery } = useBusinessRulesContext();

  /**
   * Validate a database query against business rules
   */
  const validateQueryAgainstRules = useCallback(async (
    query: string
  ): Promise<BusinessRuleValidation> => {
    setIsValidating(true);
    setValidationError(null);
    setValidationResult(null);

    try {
      // Use the context's validateQuery method
      const validation = validateQuery(query);
      
      setValidationResult(validation);
      return validation;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to validate query';
      setValidationError(errorMessage);
      
      // Return a default validation result on error
      const defaultValidation: BusinessRuleValidation = {
        isValid: true, // Allow query execution if validation fails
        errors: [],
        warnings: [],
        suggestions: []
      };
      
      setValidationResult(defaultValidation);
      return defaultValidation;
    } finally {
      setIsValidating(false);
    }
  }, [validateQuery]);

  /**
   * Clear validation results
   */
  const clearValidation = useCallback(() => {
    setValidationResult(null);
    setValidationError(null);
  }, []);

  /**
   * Check if query can be executed
   */
  const canExecuteQuery = useCallback((query: string): boolean => {
    if (!businessRules.content || businessRules.status !== 'loaded') {
      return true; // Allow execution if no business rules are configured
    }

    const validation = validateQuery(query);
    return validation.isValid;
  }, [businessRules, validateQuery]);

  /**
   * Get validation status for a query
   */
  const getValidationStatus = useCallback((query: string): {
    isValid: boolean;
    errors: string[];
    warnings: string[];
    suggestions: string[];
  } => {
    if (!businessRules.content || businessRules.status !== 'loaded') {
      return {
        isValid: true,
        errors: [],
        warnings: [],
        suggestions: []
      };
    }

    return validateQuery(query);
  }, [businessRules, validateQuery]);

  return {
    // State
    validationResult,
    isValidating,
    validationError,
    
    // Actions
    validateQuery: validateQueryAgainstRules,
    clearValidation,
    
    // Computed
    canExecuteQuery,
    getValidationStatus,
    
    // Context state
    businessRulesStatus: businessRules.status,
    hasBusinessRules: businessRules.status === 'loaded' && !!businessRules.content
  };
} 