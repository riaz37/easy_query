import { useState, useCallback } from "react";
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

  const loadStructure = useCallback(async (userId: string) => {
    // Check if we already have the structure for this userId
    if (state.structure && state.loadedUserId === userId && !state.isLoading) {
      return; // Already loaded for this userId, don't reload
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
  }, [state.structure, state.loadedUserId, state.isLoading]);

  const updateStructure = useCallback(async (userId: string, structure: UpdateReportStructureRequest) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      await ServiceRegistry.reports.updateReportStructure(userId, structure);
      
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