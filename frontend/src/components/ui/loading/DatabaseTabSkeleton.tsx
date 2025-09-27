import React from 'react';
import { cn } from '@/lib/utils';
import { DatabaseSelectionSkeleton } from './DatabaseSelectionSkeleton';

interface DatabaseTabSkeletonProps {
  className?: string;
  cardCount?: number;
  showHeader?: boolean;
  showFooter?: boolean;
  databaseCount?: number;
}

export const DatabaseTabSkeleton: React.FC<DatabaseTabSkeletonProps> = ({
  className,
  cardCount = 6,
  showHeader = true,
  showFooter = true,
  databaseCount
}) => {
  return (
    <div className={cn("space-y-6 mt-6", className)}>
      <DatabaseSelectionSkeleton
        cardCount={cardCount}
        showHeader={showHeader}
        showFooter={showFooter}
        databaseCount={databaseCount}
      />
    </div>
  );
};
