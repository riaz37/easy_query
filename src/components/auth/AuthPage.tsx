"use client";

import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LoginForm } from './LoginForm';
import { SignupForm } from './SignupForm';
import { ChangePasswordForm } from './ChangePasswordForm';
import { useAuthContext } from '@/components/providers';
import { useTheme } from '@/store/theme-store';
import { Button } from '@/components/ui/button';
import { LogOut, User, Shield } from 'lucide-react';
import { cn } from '@/lib/utils';

type AuthTab = 'login' | 'signup' | 'change-password';

interface AuthPageProps {
  onAuthSuccess?: () => void;
}

export function AuthPage({ onAuthSuccess }: AuthPageProps) {
  const { isAuthenticated, user, logout } = useAuthContext();
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState<AuthTab>('login');

  const handleAuthSuccess = () => {
    onAuthSuccess?.();
  };

  const handleLogout = () => {
    logout();
    setActiveTab('login');
  };

  // If user is authenticated, show user info and logout option
  if (isAuthenticated && user) {
    return (
      <div className="w-full max-w-md mx-auto">
        <div className="card-enhanced">
          <div className="card-content-enhanced">
            <div className="card-header-enhanced">
              <div className="text-center mb-6">
                <div className={cn(
                  "w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 border",
                  theme === "dark" 
                    ? "bg-emerald-500/20 border-emerald-500/30" 
                    : "bg-emerald-500/10 border-emerald-500/20"
                )}>
                  <User className={cn(
                    "w-8 h-8",
                    theme === "dark" ? "text-emerald-400" : "text-emerald-500"
                  )} />
                </div>
                <h2 className={cn(
                  "text-xl font-semibold",
                  theme === "dark" ? "text-white" : "text-gray-800"
                )}>Welcome back!</h2>
                <p className={cn(
                  theme === "dark" ? "text-emerald-300" : "text-emerald-600"
                )}>{user.username}</p>
                <p className={cn(
                  "text-sm",
                  theme === "dark" ? "text-gray-300" : "text-gray-600"
                )}>{user.email}</p>
              </div>
            </div>
            
            <div className="space-y-3">
              <div className={cn(
                "flex items-center justify-between p-3 rounded-lg border",
                theme === "dark" 
                  ? "bg-white/5 border-white/10" 
                  : "bg-emerald-50/50 border-emerald-200/50"
              )}>
                <span className={cn(
                  "text-sm",
                  theme === "dark" ? "text-gray-300" : "text-gray-600"
                )}>Account Status</span>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  user.is_active 
                    ? theme === "dark"
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : 'bg-emerald-100 text-emerald-700 border border-emerald-200'
                    : theme === "dark"
                      ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                      : 'bg-red-100 text-red-700 border border-red-200'
                }`}>
                  {user.is_active ? 'Active' : 'Inactive'}
                </span>
              </div>
              
              <div className={cn(
                "flex items-center justify-between p-3 rounded-lg border",
                theme === "dark" 
                  ? "bg-white/5 border-white/10" 
                  : "bg-emerald-50/50 border-emerald-200/50"
              )}>
                <span className={cn(
                  "text-sm",
                  theme === "dark" ? "text-gray-300" : "text-gray-600"
                )}>Member since</span>
                <span className={cn(
                  "text-sm",
                  theme === "dark" ? "text-white" : "text-gray-800"
                )}>
                  {new Date(user.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>

            <div className="mt-6 space-y-3">
              <Button
                onClick={() => setActiveTab('change-password')}
                variant="outline"
                className={cn(
                  "w-full",
                  theme === "dark"
                    ? "border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20"
                    : "border-emerald-500/30 text-emerald-600 hover:bg-emerald-50"
                )}
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
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-md mx-auto">
      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as AuthTab)}>
        <TabsList className={cn(
          "grid w-full grid-cols-2 mb-6 backdrop-blur-xl shadow-xl",
          theme === "dark"
            ? "bg-white/20 border border-white/30"
            : "bg-white/95 border border-emerald-500/30 shadow-sm"
        )}>
          <TabsTrigger 
            value="login" 
            className={cn(
              "transition-all duration-200",
              theme === "dark"
                ? "data-[state=active]:bg-emerald-500/30 data-[state=active]:text-emerald-300 data-[state=active]:border-emerald-500/50 text-white"
                : "data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md text-gray-600 hover:text-emerald-600"
            )}
          >
            Sign In
          </TabsTrigger>
          <TabsTrigger 
            value="signup" 
            className={cn(
              "transition-all duration-200",
              theme === "dark"
                ? "data-[state=active]:bg-emerald-500/30 data-[state=active]:text-emerald-300 data-[state=active]:border-emerald-500/50 text-white"
                : "data-[state=active]:bg-emerald-500 data-[state=active]:text-white data-[state=active]:shadow-md text-gray-600 hover:text-emerald-600"
            )}
          >
            Sign Up
          </TabsTrigger>
        </TabsList>

        <TabsContent value="login">
          <LoginForm
            onSuccess={handleAuthSuccess}
            onSwitchToSignup={() => setActiveTab('signup')}
          />
        </TabsContent>

        <TabsContent value="signup">
          <SignupForm
            onSuccess={handleAuthSuccess}
            onSwitchToLogin={() => setActiveTab('login')}
          />
        </TabsContent>

        <TabsContent value="change-password">
          <ChangePasswordForm
            onSuccess={() => setActiveTab('login')}
            onCancel={() => setActiveTab('login')}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
} 