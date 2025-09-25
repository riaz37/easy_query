"use client";

import { toast } from "sonner";

export const useToast = () => {
  return {
    toast: (message: string, options?: {
      description?: string;
      action?: {
        label: string;
        onClick: () => void;
      };
      duration?: number;
    }) => {
      return toast(message, {
        description: options?.description,
        action: options?.action,
        duration: options?.duration,
      });
    },
    
    success: (message: string, options?: {
      description?: string;
      duration?: number;
    }) => {
      return toast.success(message, {
        description: options?.description,
        duration: options?.duration,
      });
    },
    
    error: (message: string, options?: {
      description?: string;
      duration?: number;
    }) => {
      return toast.error(message, {
        description: options?.description,
        duration: options?.duration,
      });
    },
    
    warning: (message: string, options?: {
      description?: string;
      duration?: number;
    }) => {
      return toast.warning(message, {
        description: options?.description,
        duration: options?.duration,
      });
    },
    
    info: (message: string, options?: {
      description?: string;
      duration?: number;
    }) => {
      return toast.info(message, {
        description: options?.description,
        duration: options?.duration,
      });
    },
    
    loading: (message: string, options?: {
      description?: string;
    }) => {
      return toast.loading(message, {
        description: options?.description,
      });
    },
    
    promise: <T>(
      promise: Promise<T>,
      options: {
        loading: string;
        success: string | ((data: T) => string);
        error: string | ((error: any) => string);
      }
    ) => {
      return toast.promise(promise, options);
    },
    
    dismiss: (toastId?: string | number) => {
      return toast.dismiss(toastId);
    },
  };
};
