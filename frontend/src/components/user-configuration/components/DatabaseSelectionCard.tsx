import React, { useMemo } from 'react';
import { DatabaseCard } from './DatabaseCard';
import { DatabaseSelectionSkeleton } from '@/components/ui/loading';
import type { DatabaseSelectionCardProps } from '../types';

export const DatabaseSelectionCard = React.memo<DatabaseSelectionCardProps>(({
  databases,
  loading,
  onDatabaseChange,
  businessRules,
}) => {
  const databaseCards = useMemo(() => {
    return databases.map((db) => (
      <DatabaseCard
        key={db.db_id}
        database={db}
        businessRules={businessRules}
        onSelect={onDatabaseChange}
      />
    ));
  }, [databases, businessRules, onDatabaseChange]);

  return (
    <div className="query-content-gradient rounded-[32px] p-6">
      <div className="space-y-4">
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-white">Database Selection</h2>
          <p className="text-gray-400 text-sm">Choose your current working database</p>
        </div>
        <div className="space-y-4">
        {loading ? (
          <DatabaseSelectionSkeleton
            cardCount={databases.length > 0 ? databases.length : 6}
            showHeader={false}
            showFooter={true}
            databaseCount={databases.length}
          />
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {databaseCards}
            </div>

            <div className="flex justify-end items-center pt-4">
              <div className="text-sm text-gray-400">
                {databases.length} database
                {databases.length !== 1 ? 's' : ''} available
              </div>
            </div>
          </>
        )}
        </div>
      </div>
    </div>
  );
});

DatabaseSelectionCard.displayName = 'DatabaseSelectionCard';
