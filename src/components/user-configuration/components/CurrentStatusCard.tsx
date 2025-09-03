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
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white">Current Status</CardTitle>
        <CardDescription className="text-gray-400">
          Overview of your current configuration
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Database Status */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-400" />
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
                    className="text-green-400 border-green-400"
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
                    className="text-yellow-400 border-yellow-400"
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
                    className="text-green-400 border-green-400"
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
                    className="text-yellow-400 border-yellow-400"
                  >
                    <AlertCircle className="w-3 h-3 mr-1" />
                    Not Configured
                  </Badge>
                </div>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

CurrentStatusCard.displayName = 'CurrentStatusCard';
