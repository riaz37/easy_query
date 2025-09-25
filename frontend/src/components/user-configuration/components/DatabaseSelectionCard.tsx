import React, { useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Loader2, RefreshCw } from 'lucide-react';
import { DatabaseCard } from './DatabaseCard';
import type { DatabaseSelectionCardProps } from '../types';

export const DatabaseSelectionCard = React.memo<DatabaseSelectionCardProps>(({
  databases,
  loading,
  onDatabaseChange,
  onRefresh,
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
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin text-emerald-400" />
            <span className="ml-2 text-gray-400">
              Loading databases...
            </span>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {databaseCards}
            </div>

            <div className="flex justify-between items-center pt-4">
              <Button
                variant="outline"
                onClick={onRefresh}
                disabled={loading}
              >
                <RefreshCw
                  className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`}
                />
                Refresh
              </Button>

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
