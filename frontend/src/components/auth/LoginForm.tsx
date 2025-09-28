"use client";

import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useAuthContext } from '@/components/providers';
import { useTheme } from '@/store/theme-store';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import Image from 'next/image';
import { ButtonLoader } from '@/components/ui/loading';
import { cn } from '@/lib/utils';
import { ExternalLink, Shield, AlertTriangle } from 'lucide-react';

// Validation schema for login form
const loginSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  password: z.string().min(1, 'Password is required'),
});

type LoginFormData = z.infer<typeof loginSchema>;

interface LoginFormProps {
  onSuccess?: () => void;
  onSwitchToSignup?: () => void;
}

export function LoginForm({ onSuccess, onSwitchToSignup }: LoginFormProps) {
  const { login, isLoading, error, clearError } = useAuthContext();
  const theme = useTheme();
  const [showPassword, setShowPassword] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
  });

  const onSubmit = async (data: LoginFormData) => {
    try {
      clearError();
      await login(data);
      onSuccess?.();
    } catch (error) {
      // Error is handled by the auth context
      console.error('Login failed:', error);
    }
  };

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword);
  };

  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="text-left mb-6">
            <h2 className={cn(
              "text-2xl font-bold mb-2",
              theme === "dark" ? "text-white" : "text-gray-800"
            )}>
              Sign in to your account
            </h2>
            <p className={cn(
              "text-sm",
              theme === "dark" ? "text-gray-300" : "text-gray-600"
            )}>
              Don't have an account?{' '}
              <button
                type="button"
                onClick={onSwitchToSignup}
                className={cn(
                  "font-medium hover:underline"
                )}
                style={{
                  color: 'var(--primary-main, #13F584)'
                }}
              >
                Sign up
              </button>
            </p>
          </div>
        </div>
        
        {/* SSL Instructions Banner */}
        <div className={cn(
          "mb-6 p-4 rounded-lg border-l-4",
          theme === "dark" 
            ? "bg-amber-900/20 border-amber-500 text-amber-100" 
            : "bg-amber-50 border-amber-400 text-amber-800"
        )}>
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-500 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <h3 className="font-semibold text-sm mb-2">
                Important: SSL Certificate Setup Required
              </h3>
              <p className="text-sm mb-3 leading-relaxed">
                Before signing in, please visit our API documentation to accept the SSL certificate. 
                When you see the security warning:
              </p>
              <ol className="text-sm mb-3 ml-4 space-y-1 list-decimal">
                <li>Click <strong>"Advanced..."</strong> button</li>
                <li>Click <strong>"Accept the Risk and Continue"</strong></li>
                <li>This ensures secure communication with our backend services</li>
              </ol>
              <a
                href="https://176.9.16.194:8200/docs#/"
                target="_blank"
                rel="noopener noreferrer"
                className={cn(
                  "inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 hover:scale-105",
                  theme === "dark"
                    ? "bg-amber-600 hover:bg-amber-700 text-white"
                    : "bg-amber-500 hover:bg-amber-600 text-white"
                )}
              >
                <Shield className="w-4 h-4" />
                Visit API Documentation
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>

        <div className="space-y-4">
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="space-y-2">
            <Label htmlFor="username" className={cn(
              "text-sm font-medium",
              theme === "dark" ? "text-white" : "text-gray-700"
            )}>Username</Label>
            <Input
              id="username"
              type="text"
              placeholder="Enter your username"
              {...register('username')}
              className={cn(
                "modal-input-enhanced",
                errors.username ? 'border-red-500' : ''
              )}
              style={{
                border: '1px solid var(--components-paper-outlined, #FFFFFF1F)',
                height: '54px'
              }}
            />
            {errors.username && (
              <p className="text-sm text-red-400">{errors.username.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="password" className={cn(
                "text-sm font-medium",
                theme === "dark" ? "text-white" : "text-gray-700"
              )}>Password</Label>
              <button
                type="button"
                className={cn(
                  "text-sm font-medium hover:underline"
                )}
                style={{
                  color: 'var(--text-primary, #FFFFFF)'
                }}
              >
                Forgot password?
              </button>
            </div>
            <div className="relative">
              <Input
                id="password"
                type={showPassword ? 'text' : 'password'}
                placeholder="Enter your password"
                {...register('password')}
                className={cn(
                  "modal-input-enhanced pr-10",
                  errors.password ? 'border-red-500' : ''
                )}
                style={{
                  border: '1px solid var(--components-paper-outlined, #FFFFFF1F)',
                  height: '54px'
                }}
              />
              <button
                type="button"
                onClick={togglePasswordVisibility}
                className={cn(
                  "absolute right-3 top-1/2 transform -translate-y-1/2",
                  theme === "dark" ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-gray-700"
                )}
              >
                <Image
                  src="/dashboard/eye.svg"
                  alt={showPassword ? "Hide password" : "Show password"}
                  width={16}
                  height={16}
                  className={cn(
                    "transition-opacity duration-200",
                    showPassword ? "opacity-50" : "opacity-100"
                  )}
                />
              </button>
            </div>
            {errors.password && (
              <p className="text-sm text-red-400">{errors.password.message}</p>
            )}
          </div>

          <ButtonLoader
            type="submit"
            loading={isLoading}
            text="Signing in..."
            size="md"
            variant="primary"
            className={cn(
              "w-full text-base font-semibold transition-colors duration-200",
              theme === "dark"
                ? "text-white bg-white/4 hover:bg-white/10"
                : "text-emerald-600 bg-emerald-50/20 hover:bg-emerald-50/40"
            )}
            style={{
              borderRadius: '99px',
              height: '54px'
            }}
          >
            Sign In
          </ButtonLoader>
        </form>
        </div>
      </div>
    </div>
  );
} 