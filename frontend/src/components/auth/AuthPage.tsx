"use client";

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { LoginForm } from './LoginForm';
import { SignupForm } from './SignupForm';
import { ChangePasswordForm } from './ChangePasswordForm';
import { useAuthContext } from '@/components/providers';
import { useTheme } from '@/store/theme-store';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

type AuthTab = 'login' | 'signup' | 'change-password';

interface AuthPageProps {
  onAuthSuccess?: () => void;
}

export function AuthPage({ onAuthSuccess }: AuthPageProps) {
  const { isAuthenticated, user, logout } = useAuthContext();
  const router = useRouter();
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState<AuthTab>('login');

  const handleAuthSuccess = () => {
    onAuthSuccess?.();
  };

  const handleLogout = () => {
    logout();
    setActiveTab('login');
  };

  // Redirect authenticated users to dashboard
  useEffect(() => {
    if (isAuthenticated && user) {
      router.push('/');
    }
  }, [isAuthenticated, user, router]);

  // Show loading state while redirecting
  if (isAuthenticated && user) {
    return (
      <div className="w-full max-w-md mx-auto">
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-400 mx-auto mb-4" />
              <p className={cn(
                "text-lg font-medium",
                theme === "dark" ? "text-white" : "text-gray-800"
              )}>
                Redirecting to dashboard...
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md mx-auto">
      {activeTab === 'login' && (
        <LoginForm
          onSuccess={handleAuthSuccess}
          onSwitchToSignup={() => setActiveTab('signup')}
        />
      )}
      
      {activeTab === 'signup' && (
        <SignupForm
          onSuccess={handleAuthSuccess}
          onSwitchToLogin={() => setActiveTab('login')}
        />
      )}

      {activeTab === 'change-password' && (
        <ChangePasswordForm
          onSuccess={() => setActiveTab('login')}
          onCancel={() => setActiveTab('login')}
        />
      )}
    </div>
  );
} 