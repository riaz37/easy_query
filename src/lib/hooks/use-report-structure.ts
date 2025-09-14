import { useState, useCallback, useRef, useEffect } from "react";
import { ServiceRegistry } from "../api";
import { ReportStructure, UpdateReportStructureRequest } from "../../types/reports";

interface UseReportStructureState {
  structure: ReportStructure | null;
  isLoading: boolean;
  error: string | null;
  loadedUserId: string | null;
}

interface UseReportStructureReturn extends UseReportStructureState {
  loadStructure: (userId: string) => Promise<void>;
  updateStructure: (userId: string, structure: UpdateReportStructureRequest) => Promise<void>;
  parseStructure: (structureString: string) => ReportStructure;
  stringifyStructure: (structure: ReportStructure) => string;
  validateStructure: (structure: ReportStructure) => boolean;
  reset: () => void;
}

export function useReportStructure(): UseReportStructureReturn {
  const [state, setState] = useState<UseReportStructureState>({
    structure: null,
    isLoading: false,
    error: null,
    loadedUserId: null,
  });

  // Use ref to track current state without causing re-renders
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const loadStructure = useCallback(async (userId: string) => {
    // Check if we already have the structure for this userId
    const currentState = stateRef.current;
    if (currentState.structure && currentState.loadedUserId === userId && !currentState.isLoading) {
      return; // Already loaded for this userId, don't reload
    }

    // Prevent multiple simultaneous calls
    if (currentState.isLoading) {
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const structureString = await ServiceRegistry.reports.getReportStructure(userId);
      
      if (structureString) {
        const parsedStructure = JSON.parse(structureString);
        setState(prev => ({
          ...prev,
          structure: parsedStructure,
          isLoading: false,
          loadedUserId: userId,
        }));
      } else {
        setState(prev => ({
          ...prev,
          structure: null,
          isLoading: false,
          loadedUserId: userId,
        }));
      }
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to load report structure',
        isLoading: false,
      }));
    }
  }, []); // Empty dependency array since we're using ref

  const updateStructure = useCallback(async (userId: string, structure: UpdateReportStructureRequest) => {
    // Prevent multiple simultaneous calls
    const currentState = stateRef.current;
    if (currentState.isLoading) {
      return;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      // Use the correct API endpoint for user-based updates
      await ServiceRegistry.reports.updateUserReportStructure(userId, structure);
      
      // Reload the structure after update
      await loadStructure(userId);
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to update report structure',
        isLoading: false,
      }));
    }
  }, [loadStructure]);

  const parseStructure = useCallback((structureString: string): ReportStructure => {
    try {
      return JSON.parse(structureString);
    } catch (error) {
      throw new Error('Invalid report structure format');
    }
  }, []);

  const stringifyStructure = useCallback((structure: ReportStructure): string => {
    try {
      return JSON.stringify(structure, null, 2);
    } catch (error) {
      throw new Error('Failed to stringify report structure');
    }
  }, []);

  const validateStructure = useCallback((structure: ReportStructure): boolean => {
    return structure && typeof structure === 'object' && Object.keys(structure).length > 0;
  }, []);

  const reset = useCallback(() => {
    setState({
      structure: null,
      isLoading: false,
      error: null,
      loadedUserId: null,
    });
  }, []);

  return {
    ...state,
    loadStructure,
    updateStructure,
    parseStructure,
    stringifyStructure,
    validateStructure,
    reset,
  };
} 