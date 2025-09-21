import React from 'react';
import { BusinessRulesEditor } from './BusinessRulesEditor';
import type { BusinessRulesTabProps } from '../types';

export const BusinessRulesTab = React.memo<BusinessRulesTabProps>(({
  currentDatabaseId,
  currentDatabaseName,
  businessRules,
  businessRulesCount,
  hasBusinessRules,
  editorState,
  onRefresh,
  onEdit,
  onSave,
  onCancel,
  onContentChange,
}) => {
  return (
    <div>
      <BusinessRulesEditor
        currentDatabaseId={currentDatabaseId}
        businessRules={businessRules}
        editorState={editorState}
        onContentChange={onContentChange}
        onEdit={onEdit}
        onSave={onSave}
        onCancel={onCancel}
      />
    </div>
  );
});

BusinessRulesTab.displayName = 'BusinessRulesTab';
