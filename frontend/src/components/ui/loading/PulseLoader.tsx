"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

const sizeClasses = {
  xs: "w-2 h-2",
  sm: "w-3 h-3",
  md: "w-4 h-4", 
  lg: "w-6 h-6",
  xl: "w-8 h-8"
};

const variantClasses = {
  primary: "bg-[#10b981]", // Emerald-500
  "primary-dark": "bg-[#047857]", // Emerald-700
  "primary-light": "bg-[#34d399]", // Emerald-400
  "accent-blue": "bg-[#3b82f6]", // Blue-500
  "accent-purple": "bg-[#8b5cf6]", // Violet-500
  "accent-orange": "bg-[#f59e0b]", // Amber-500
  secondary: "bg-secondary",
  success: "bg-green-500", 
  warning: "bg-yellow-500",
  error: "bg-red-500",
  info: "bg-blue-500"
};

export function PulseLoader({ 
  size = "md", 
  variant = "primary", 
  className 
}: LoadingProps) {
  return (
    <div className={cn("flex items-center justify-center space-x-1", className)}>
      <div
        className={cn(
          "rounded-full animate-pulse",
          sizeClasses[size],
          variantClasses[variant]
        )}
        style={{ animationDelay: "0ms" }}
      />
      <div
        className={cn(
          "rounded-full animate-pulse",
          sizeClasses[size],
          variantClasses[variant]
        )}
        style={{ animationDelay: "150ms" }}
      />
      <div
        className={cn(
          "rounded-full animate-pulse",
          sizeClasses[size],
          variantClasses[variant]
        )}
        style={{ animationDelay: "300ms" }}
      />
    </div>
  );
}
