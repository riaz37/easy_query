import { useState, useCallback } from 'react';
import { AVAILABLE_MODELS, AvailableModel } from '../api/endpoints';
import { Brain, Sparkles, Bot } from 'lucide-react';

/**
 * Model display information interface
 */
export interface ModelDisplayInfo {
  name: string;
  icon: any;
  description: string;
  color: string;
}

/**
 * Hook for managing AI model selection
 */
export function useAIModels() {
  const [selectedModel, setSelectedModel] = useState<AvailableModel>(
    AVAILABLE_MODELS.LLAMA_3_3_70B_VERSATILE
  );

  /**
   * Get all available models
   */
  const getAvailableModels = useCallback(() => {
    return AVAILABLE_MODELS;
  }, []);

  /**
   * Get model display information
   */
  const getModelDisplayInfo = useCallback((modelKey: string): ModelDisplayInfo => {
    switch (modelKey) {
      case AVAILABLE_MODELS.LLAMA_3_3_70B_VERSATILE:
        return {
          name: "Llama 3.3 70B",
          icon: Brain,
          description: "Versatile & Fast",
          color: "text-emerald-400"
        };
      case AVAILABLE_MODELS.OPENAI_GPT_OSS_120B:
        return {
          name: "GPT OSS 120B",
          icon: Sparkles,
          description: "Advanced Reasoning",
          color: "text-blue-400"
        };
      case AVAILABLE_MODELS.GEMINI:
        return {
          name: "Gemini",
          icon: Bot,
          description: "Google AI",
          color: "text-purple-400"
        };
      default:
        return {
          name: "Unknown Model",
          icon: Brain,
          description: "",
          color: "text-gray-400"
        };
    }
  }, []);

  /**
   * Get all models with their display info
   */
  const getModelsWithDisplayInfo = useCallback(() => {
    return Object.entries(AVAILABLE_MODELS).map(([key, value]) => ({
      key,
      value,
      ...getModelDisplayInfo(value)
    }));
  }, [getModelDisplayInfo]);

  /**
   * Change selected model
   */
  const changeModel = useCallback((model: AvailableModel) => {
    setSelectedModel(model);
  }, []);

  /**
   * Reset to default model
   */
  const resetToDefault = useCallback(() => {
    setSelectedModel(AVAILABLE_MODELS.LLAMA_3_3_70B_VERSATILE);
  }, []);

  return {
    // State
    selectedModel,
    
    // Actions
    changeModel,
    resetToDefault,
    
    // Utilities
    getAvailableModels,
    getModelDisplayInfo,
    getModelsWithDisplayInfo,
    
    // Constants
    MODELS: AVAILABLE_MODELS,
  };
}
