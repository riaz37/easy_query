import { useState, useCallback, useMemo } from 'react';
import { toast } from 'sonner';
import { useBusinessRulesContext } from '@/components/providers/BusinessRulesContextProvider';
import { ServiceRegistry } from '@/lib/api';
import type { BusinessRulesEditorState } from '../types';

interface UseBusinessRulesEditorProps {
  currentDatabaseId?: number;
  businessRulesContent: string;
  onRefresh: () => void;
}

export const useBusinessRulesEditor = ({
  currentDatabaseId,
  businessRulesContent,
  onRefresh,
}: UseBusinessRulesEditorProps) => {
  const { updateBusinessRules, setLoading: setBusinessRulesLoading, setError: setBusinessRulesError } = useBusinessRulesContext();

  // Business rules editing state
  const [editorState, setEditorState] = useState<BusinessRulesEditorState>({
    isEditing: false,
    editedContent: '',
    hasUnsavedChanges: false,
    contentError: null,
  });

  // Business rules editing handlers
  const handleRulesEdit = useCallback(() => {
    setEditorState({
      isEditing: true,
      editedContent: businessRulesContent,
      hasUnsavedChanges: false,
      contentError: null,
    });
  }, [businessRulesContent]);

  const handleRulesSave = useCallback(async () => {
    if (!currentDatabaseId) {
      toast.error('No database selected');
      return;
    }

    // Validate content
    if (!editorState.editedContent.trim()) {
      setEditorState(prev => ({
        ...prev,
        contentError: 'Business rules content cannot be empty',
      }));
      return;
    }

    if (editorState.editedContent.length < 10) {
      setEditorState(prev => ({
        ...prev,
        contentError: 'Business rules content is too short (minimum 10 characters)',
      }));
      return;
    }

    if (editorState.editedContent.length > 50000) {
      setEditorState(prev => ({
        ...prev,
        contentError: 'Business rules content is too long (maximum 50,000 characters)',
      }));
      return;
    }

    setBusinessRulesLoading(true);
    setEditorState(prev => ({ ...prev, contentError: null }));

    try {
      const response = await ServiceRegistry.businessRules.updateBusinessRules(
        editorState.editedContent,
        currentDatabaseId,
      );

      if (response.success) {
        // Update business rules context
        updateBusinessRules(editorState.editedContent);

        // Reset editing state
        setEditorState({
          isEditing: false,
          editedContent: '',
          hasUnsavedChanges: false,
          contentError: null,
        });

        toast.success('Business rules updated successfully');

        // Refresh the configuration to get updated data
        await onRefresh();
      } else {
        throw new Error(response.error || 'Failed to update business rules');
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Failed to update business rules';
      setEditorState(prev => ({ ...prev, contentError: errorMessage }));
      setBusinessRulesError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setBusinessRulesLoading(false);
    }
  }, [
    currentDatabaseId,
    editorState.editedContent,
    updateBusinessRules,
    onRefresh,
    setBusinessRulesLoading,
    setBusinessRulesError,
  ]);

  const handleRulesCancel = useCallback(() => {
    setEditorState({
      isEditing: false,
      editedContent: '',
      hasUnsavedChanges: false,
      contentError: null,
    });
  }, []);

  const handleRulesContentChange = useCallback(
    (content: string) => {
      setEditorState(prev => ({
        ...prev,
        editedContent: content,
        hasUnsavedChanges: content !== businessRulesContent,
        contentError: null,
      }));
    },
    [businessRulesContent],
  );

  // Memoize the return object to prevent unnecessary re-renders
  return useMemo(() => ({
    editorState,
    handleRulesEdit,
    handleRulesSave,
    handleRulesCancel,
    handleRulesContentChange,
  }), [
    editorState,
    handleRulesEdit,
    handleRulesSave,
    handleRulesCancel,
    handleRulesContentChange,
  ]);
};
