import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Database,
  Shield,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import type { CurrentStatusCardProps } from '../types';

export const CurrentStatusCard = React.memo<CurrentStatusCardProps>(({
  currentDatabaseName,
  businessRules,
  businessRulesCount,
  hasBusinessRules,
}) => {
  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced">Current Status</div>
          <p className="card-description-enhanced">
            Overview of your current configuration
          </p>
        </div>
        <div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Database Status */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-emerald-400" />
              <span className="text-white font-medium">
                Database Context
              </span>
            </div>
            <div className="ml-7">
              {currentDatabaseName ? (
                <div className="space-y-2">
                  <div className="text-white">
                    {currentDatabaseName}
                  </div>
                  <Badge
                    variant="outline"
                    className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white border-emerald-400/30 shadow-lg shadow-emerald-500/25"
                  >
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Active
                  </Badge>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-gray-400">
                    No database selected
                  </div>
                  <Badge
                    variant="outline"
                    className="bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700 text-white border-yellow-400/30 shadow-lg shadow-yellow-500/25"
                  >
                    <AlertCircle className="w-3 h-3 mr-1" />
                    Not Configured
                  </Badge>
                </div>
              )}
            </div>
          </div>

          {/* Business Rules Status */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-emerald-400" />
              <span className="text-white font-medium">
                Business Rules
              </span>
            </div>
            <div className="ml-7">
              {businessRules.status === 'loaded' && hasBusinessRules ? (
                <div className="space-y-2">
                  <div className="text-white">
                    {businessRulesCount} rules active
                  </div>
                  <Badge
                    variant="outline"
                    className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white border-emerald-400/30 shadow-lg shadow-emerald-500/25"
                  >
                    <CheckCircle className="w-3 h-3 mr-1" />
                    Active
                  </Badge>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-gray-400">
                    No rules configured
                  </div>
                  <Badge
                    variant="outline"
                    className="bg-gradient-to-r from-yellow-500 to-yellow-600 hover:from-yellow-600 hover:to-yellow-700 text-white border-yellow-400/30 shadow-lg shadow-yellow-500/25"
                  >
                    <AlertCircle className="w-3 h-3 mr-1" />
                    Not Configured
                  </Badge>
                </div>
              )}
            </div>
          </div>
        </div>
        </div>
      </div>
    </div>
  );
});

CurrentStatusCard.displayName = 'CurrentStatusCard';
