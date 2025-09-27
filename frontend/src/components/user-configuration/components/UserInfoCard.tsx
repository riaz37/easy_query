import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { User } from 'lucide-react';
import type { UserInfoCardProps } from '../types';

export const UserInfoCard = React.memo<UserInfoCardProps>(({ user }) => {
  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="card-title-enhanced flex items-center gap-2">
            <User className="h-5 w-5 text-emerald-400" />
            User Information
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label className="text-gray-400">User ID</Label>
            <div className="text-white font-medium">
              {user?.user_id}
            </div>
          </div>
          <div>
            <Label className="text-gray-400">Email</Label>
            <div className="text-white font-medium">
              {user?.email || 'Not provided'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

UserInfoCard.displayName = 'UserInfoCard';
