import React from 'react';
import { DatabaseSelectionCard } from './DatabaseSelectionCard';
import type { DatabaseTabProps } from '../types';

export const DatabaseTab = React.memo<DatabaseTabProps>(({
  databases,
  loading,
  onDatabaseChange,
  onRefresh,
  businessRules,
}) => {
  return (
    <div className="space-y-6 mt-6">
      <DatabaseSelectionCard
        databases={databases}
        loading={loading}
        onDatabaseChange={onDatabaseChange}
        onRefresh={onRefresh}
        businessRules={businessRules}
      />
    </div>
  );
});

DatabaseTab.displayName = 'DatabaseTab';
