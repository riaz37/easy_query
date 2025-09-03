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
    <Card className="bg-slate-800/50 border-slate-700">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-white">
          <User className="h-5 w-5 text-blue-400" />
          User Information
        </CardTitle>
      </CardHeader>
      <CardContent>
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
      </CardContent>
    </Card>
  );
});

UserInfoCard.displayName = 'UserInfoCard';
