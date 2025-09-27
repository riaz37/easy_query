import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Globe, Key, Database } from 'lucide-react';
import type { ContextInfoCardProps } from '../types';

export const ContextInfoCard = React.memo<ContextInfoCardProps>(({
  className,
}) => {
  return (
    <Card className={`bg-slate-800/50 border-slate-700 ${className || ''}`}>
      <CardHeader>
        <CardTitle className="text-white">
          Context Information
        </CardTitle>
        <CardDescription className="text-gray-400">
          How business rules are applied in your system
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-3 text-sm text-gray-300">
          <div className="flex items-start gap-2">
            <Globe className="w-4 h-4 text-blue-400 mt-0.5" />
            <span>
              Business rules are automatically applied to all database
              queries
            </span>
          </div>
          <div className="flex items-start gap-2">
            <Key className="w-4 h-4 text-emerald-400 mt-0.5" />
            <span>
              Rules ensure data integrity and compliance across your
              system
            </span>
          </div>
          <div className="flex items-start gap-2">
            <Database className="w-4 h-4 text-purple-400 mt-0.5" />
            <span>
              Rules are database-specific and automatically loaded when
              switching databases
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
});

ContextInfoCard.displayName = 'ContextInfoCard';
