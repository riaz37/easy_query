import React from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Database, Shield } from 'lucide-react';
import type { QuickActionsCardProps } from '../types';

export const QuickActionsCard = React.memo<QuickActionsCardProps>(({
  onNavigateToTab,
}) => {
  return (
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white">Quick Actions</CardTitle>
        <CardDescription className="text-gray-400">
          Common configuration tasks
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Button
            variant="outline"
            onClick={() => onNavigateToTab('database')}
            className="w-full bg-slate-700 border-slate-600 text-white hover:bg-slate-600"
          >
            <Database className="w-4 h-4 mr-2" />
            Configure Database
          </Button>

          <Button
            variant="outline"
            onClick={() => onNavigateToTab('business-rules')}
            className="w-full bg-slate-700 border-slate-600 text-white hover:bg-slate-600"
          >
            <Shield className="w-4 h-4 mr-2" />
            Manage Business Rules
          </Button>
        </div>
      </CardContent>
    </Card>
  );
});

QuickActionsCard.displayName = 'QuickActionsCard';
