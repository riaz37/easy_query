import React from 'react';
import { UserInfoCard } from './UserInfoCard';
import { CurrentStatusCard } from './CurrentStatusCard';
import { QuickActionsCard } from './QuickActionsCard';
import type { OverviewTabProps } from '../types';

export const OverviewTab = React.memo<OverviewTabProps>(({
  user,
  currentDatabaseName,
  businessRules,
  businessRulesCount,
  hasBusinessRules,
  onNavigateToTab,
}) => {
  return (
    <div className="space-y-6 mt-6">
      <UserInfoCard user={user} />
      <CurrentStatusCard
        currentDatabaseName={currentDatabaseName}
        businessRules={businessRules}
        businessRulesCount={businessRulesCount}
        hasBusinessRules={hasBusinessRules}
      />
      <QuickActionsCard onNavigateToTab={onNavigateToTab} />
    </div>
  );
});

OverviewTab.displayName = 'OverviewTab';
