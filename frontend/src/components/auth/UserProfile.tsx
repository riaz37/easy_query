"use client";

import React, { useState } from 'react';
import { useAuthContext } from '@/components/providers';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ChangePasswordForm } from './ChangePasswordForm';
import { 
  User, 
  Mail, 
  Calendar, 
  Shield, 
  LogOut, 
  Edit3,
  CheckCircle,
  X
} from 'lucide-react';

interface UserProfileProps {
  onLogout?: () => void;
  showChangePassword?: boolean;
  onToggleChangePassword?: () => void;
}

export function UserProfile({ 
  onLogout, 
  showChangePassword = false,
  onToggleChangePassword 
}: UserProfileProps) {
  const { user, logout, isAuthenticated } = useAuthContext();
  const [isEditing, setIsEditing] = useState(false);

  if (!isAuthenticated || !user) {
    return null;
  }

  const handleLogout = () => {
    logout();
    onLogout?.();
  };

  const handleToggleChangePassword = () => {
    onToggleChangePassword?.();
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  if (showChangePassword) {
    return (
      <Card className="w-full max-w-md">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <div>
            <CardTitle className="text-lg font-semibold">Change Password</CardTitle>
            <CardDescription>Update your account password</CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleToggleChangePassword}
          >
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent>
          <ChangePasswordForm
            onSuccess={handleToggleChangePassword}
            onCancel={handleToggleChangePassword}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full max-w-md">
      <CardHeader className="text-center">
        <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <User className="w-10 h-10 text-blue-600" />
        </div>
        <CardTitle className="text-xl">{user.username}</CardTitle>
        <CardDescription>{user.email}</CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Mail className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600">Email</span>
            </div>
            <span className="text-sm font-medium text-gray-900">{user.email}</span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600">Member since</span>
            </div>
            <span className="text-sm font-medium text-gray-900">
              {formatDate(user.created_at)}
            </span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-gray-500" />
              <span className="text-sm text-gray-600">Status</span>
            </div>
            <Badge variant={user.is_active ? "default" : "secondary"}>
              {user.is_active ? 'Active' : 'Inactive'}
            </Badge>
          </div>
        </div>

        <Separator />

        <div className="space-y-2">
          <Button
            onClick={handleToggleChangePassword}
            variant="outline"
            className="w-full"
          >
            <Shield className="w-4 h-4 mr-2" />
            Change Password
          </Button>
          
          <Button
            onClick={handleLogout}
            variant="destructive"
            className="w-full"
          >
            <LogOut className="w-4 h-4 mr-2" />
            Sign Out
          </Button>
        </div>
      </CardContent>
    </Card>
  );
} 