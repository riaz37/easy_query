import React from 'react';
import {
  Card,
  CardContent,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { CheckCircle } from 'lucide-react';
import type { DatabaseCardProps } from '../types';

export const DatabaseCard = React.memo<DatabaseCardProps>(({
  database,
  businessRules,
  onSelect,
}) => {
  const handleClick = () => {
    onSelect(database.db_id);
  };

  const getRulesInfo = () => {
    if (database.is_current && businessRules.status === 'loaded') {
      return `${businessRules.content.length} chars`;
    }
    return database.business_rule ? `${database.business_rule.length} chars` : 'None';
  };

  return (
    <Card
      className={`cursor-pointer transition-all hover:scale-105 ${
        database.is_current
          ? 'bg-emerald-900/30 border-emerald-500'
          : 'bg-slate-700/50 border-slate-600 hover:border-slate-500'
      }`}
      onClick={handleClick}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-white">
            {database.db_name}
          </h3>
          {database.is_current && (
            <CheckCircle className="w-5 h-5 text-emerald-400" />
          )}
        </div>
        <div className="text-sm text-gray-400 space-y-1">
          <div>Type: {database.db_type}</div>
          <div className="truncate">URL: {database.db_url}</div>
          <div>Rules: {getRulesInfo()}</div>
        </div>
        <Badge
          variant={database.is_current ? 'default' : 'secondary'}
          className={`mt-2 ${
            database.is_current
              ? 'bg-emerald-600 text-white'
              : 'bg-slate-600 text-gray-300'
          }`}
        >
          {database.is_current ? 'Current' : 'Select'}
        </Badge>
      </CardContent>
    </Card>
  );
});

DatabaseCard.displayName = 'DatabaseCard';
