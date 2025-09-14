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
import { Eye, EyeOff, CheckCircle } from 'lucide-react';
import { ButtonLoader } from '@/components/ui/loading';
import { cn } from '@/lib/utils';

// Validation schema for change password form
const changePasswordSchema = z.object({
  currentPassword: z.string().min(1, 'Current password is required'),
  newPassword: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, 'Password must contain at least one lowercase letter, one uppercase letter, and one number'),
  confirmNewPassword: z.string(),
}).refine((data) => data.newPassword === data.confirmNewPassword, {
  message: "Passwords don't match",
  path: ["confirmNewPassword"],
});

type ChangePasswordFormData = z.infer<typeof changePasswordSchema>;

interface ChangePasswordFormProps {
  onSuccess?: () => void;
  onCancel?: () => void;
}

export function ChangePasswordForm({ onSuccess, onCancel }: ChangePasswordFormProps) {
  const { changePassword, isLoading, error, clearError } = useAuthContext();
  const theme = useTheme();
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<ChangePasswordFormData>({
    resolver: zodResolver(changePasswordSchema),
  });

  const onSubmit = async (data: ChangePasswordFormData) => {
    try {
      clearError();
      await changePassword({
        current_password: data.currentPassword,
        new_password: data.newPassword,
      });
      setIsSuccess(true);
      reset();
      
      // Show success message for a few seconds before calling onSuccess
      setTimeout(() => {
        onSuccess?.();
      }, 2000);
    } catch (error) {
      // Error is handled by the auth context
      console.error('Password change failed:', error);
    }
  };

  const toggleCurrentPasswordVisibility = () => {
    setShowCurrentPassword(!showCurrentPassword);
  };

  const toggleNewPasswordVisibility = () => {
    setShowNewPassword(!showNewPassword);
  };

  const toggleConfirmPasswordVisibility = () => {
    setShowConfirmPassword(!showConfirmPassword);
  };

  if (isSuccess) {
    return (
      <div className="card-enhanced">
        <div className="card-content-enhanced">
          <div className="text-center py-6">
            <CheckCircle className={cn(
              "mx-auto h-12 w-12 mb-4",
              theme === "dark" ? "text-emerald-400" : "text-emerald-500"
            )} />
            <h3 className={cn(
              "text-lg font-semibold mb-2",
              theme === "dark" ? "text-emerald-300" : "text-emerald-600"
            )}>
              Password changed successfully!
            </h3>
            <p className={cn(
              "text-sm mb-4",
              theme === "dark" ? "text-gray-300" : "text-gray-600"
            )}>
              You will be redirected to login in a few seconds...
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card-enhanced">
      <div className="card-content-enhanced">
        <div className="card-header-enhanced">
          <div className="text-center mb-6">
            <h2 className={cn(
              "text-2xl font-bold mb-2",
              theme === "dark" ? "text-white" : "text-gray-800"
            )}>
              Change Password
            </h2>
            <p className={cn(
              "text-sm",
              theme === "dark" ? "text-gray-300" : "text-gray-600"
            )}>
              Enter your current password and choose a new one
            </p>
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
            <Label htmlFor="currentPassword" className={cn(
              "text-sm font-medium",
              theme === "dark" ? "text-white" : "text-gray-700"
            )}>Current Password</Label>
            <div className="relative">
              <Input
                id="currentPassword"
                type={showCurrentPassword ? 'text' : 'password'}
                placeholder="Enter your current password"
                {...register('currentPassword')}
                className={cn(
                  "modal-input-enhanced pr-10",
                  errors.currentPassword ? 'border-red-500' : ''
                )}
              />
              <button
                type="button"
                onClick={toggleCurrentPasswordVisibility}
                className={cn(
                  "absolute right-3 top-1/2 transform -translate-y-1/2",
                  theme === "dark" ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-gray-700"
                )}
              >
                {showCurrentPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {errors.currentPassword && (
              <p className="text-sm text-red-400">{errors.currentPassword.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="newPassword" className={cn(
              "text-sm font-medium",
              theme === "dark" ? "text-white" : "text-gray-700"
            )}>New Password</Label>
            <div className="relative">
              <Input
                id="newPassword"
                type={showNewPassword ? 'text' : 'password'}
                placeholder="Enter your new password"
                {...register('newPassword')}
                className={cn(
                  "modal-input-enhanced pr-10",
                  errors.newPassword ? 'border-red-500' : ''
                )}
              />
              <button
                type="button"
                onClick={toggleNewPasswordVisibility}
                className={cn(
                  "absolute right-3 top-1/2 transform -translate-y-1/2",
                  theme === "dark" ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-gray-700"
                )}
              >
                {showNewPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {errors.newPassword && (
              <p className="text-sm text-red-400">{errors.newPassword.message}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="confirmNewPassword" className={cn(
              "text-sm font-medium",
              theme === "dark" ? "text-white" : "text-gray-700"
            )}>Confirm New Password</Label>
            <div className="relative">
              <Input
                id="confirmNewPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                placeholder="Confirm your new password"
                {...register('confirmNewPassword')}
                className={cn(
                  "modal-input-enhanced pr-10",
                  errors.confirmNewPassword ? 'border-red-500' : ''
                )}
              />
              <button
                type="button"
                onClick={toggleConfirmPasswordVisibility}
                className={cn(
                  "absolute right-3 top-1/2 transform -translate-y-1/2",
                  theme === "dark" ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-gray-700"
                )}
              >
                {showConfirmPassword ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {errors.confirmNewPassword && (
              <p className="text-sm text-red-400">{errors.confirmNewPassword.message}</p>
            )}
          </div>

          <div className="flex space-x-3">
            <Button
              type="button"
              variant="outline"
              className={cn(
                "flex-1",
                theme === "dark"
                  ? "border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20"
                  : "border-emerald-500/30 text-emerald-600 hover:bg-emerald-50"
              )}
              onClick={onCancel}
              disabled={isLoading}
            >
              Cancel
            </Button>
            <ButtonLoader
              type="submit"
              loading={isLoading}
              text="Changing..."
              size="md"
              variant="primary"
              className={cn(
                "flex-1 h-12 text-base font-semibold",
                theme === "dark"
                  ? "bg-emerald-500 hover:bg-emerald-600 text-white shadow-lg hover:shadow-emerald-500/25"
                  : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-lg hover:shadow-emerald-500/25"
              )}
            >
              Change Password
            </ButtonLoader>
          </div>
        </form>
        </div>
      </div>
    </div>
  );
} 