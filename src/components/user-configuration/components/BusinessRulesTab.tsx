import React from 'react';
import { BusinessRulesStatusCard } from './BusinessRulesStatusCard';
import { BusinessRulesEditor } from './BusinessRulesEditor';
import { ContextInfoCard } from './ContextInfoCard';
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
  onReset,
  onContentChange,
}) => {
  return (
    <div className="space-y-6 mt-6">
      <BusinessRulesStatusCard
        currentDatabaseName={currentDatabaseName}
        businessRules={businessRules}
        businessRulesCount={businessRulesCount}
        hasBusinessRules={hasBusinessRules}
        editorState={editorState}
        onRefresh={onRefresh}
        onEdit={onEdit}
        onSave={onSave}
        onCancel={onCancel}
        onReset={onReset}
      />
      
      <BusinessRulesEditor
        currentDatabaseId={currentDatabaseId}
        businessRules={businessRules}
        editorState={editorState}
        onContentChange={onContentChange}
      />
      
      <ContextInfoCard />
    </div>
  );
});

BusinessRulesTab.displayName = 'BusinessRulesTab';
