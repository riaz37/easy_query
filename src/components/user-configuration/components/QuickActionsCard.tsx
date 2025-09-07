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
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced">Quick Actions</div>
          <p className="card-description-enhanced">
            Common configuration tasks
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Button
            variant="outline"
            onClick={() => onNavigateToTab('database')}
            className="card-button-enhanced w-full"
          >
            <Database className="w-4 h-4 mr-2" />
            Configure Database
          </Button>

          <Button
            variant="outline"
            onClick={() => onNavigateToTab('business-rules')}
            className="card-button-enhanced w-full"
          >
            <Shield className="w-4 h-4 mr-2" />
            Manage Business Rules
          </Button>
        </div>
      </div>
    </div>
  );
});

QuickActionsCard.displayName = 'QuickActionsCard';
