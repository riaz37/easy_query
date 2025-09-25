import React from 'react';
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
    <div
      className={`cursor-pointer transition-all hover:scale-105 rounded-[32px] p-4 query-content-gradient h-[184px] ${
        database.is_current
          ? 'border-2 border-emerald-500'
          : 'border-2 border-transparent'
      }`}
      onClick={handleClick}
    >
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-white text-lg">
            {database.db_name}
          </h3>
          {database.is_current && (
            <div className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-white"></div>
            </div>
          )}
        </div>
        <div className="text-sm space-y-1">
          <div><span className="text-gray-400">Type</span> <span className="text-white">{database.db_type}</span></div>
          <div className="truncate"><span className="text-gray-400">URL</span> <span className="text-white">{database.db_url}</span></div>
          <div><span className="text-gray-400">Rules</span> <span className="text-white">{getRulesInfo()}</span></div>
        </div>
      </div>
    </div>
  );
});

DatabaseCard.displayName = 'DatabaseCard';
