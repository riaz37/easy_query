import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
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
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white">Database Selection</CardTitle>
        <CardDescription className="text-gray-400">
          Choose your current working database
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
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
      </CardContent>
    </Card>
  );
});

DatabaseSelectionCard.displayName = 'DatabaseSelectionCard';
