"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

const sizeClasses = {
  xs: "w-3 h-3",
  sm: "w-4 h-4", 
  md: "w-6 h-6",
  lg: "w-8 h-8",
  xl: "w-12 h-12"
};

const variantClasses = {
  primary: "text-[#10b981]", // Emerald-500
  "primary-dark": "text-[#047857]", // Emerald-700
  "primary-light": "text-[#34d399]", // Emerald-400
  "accent-blue": "text-[#3b82f6]", // Blue-500
  "accent-purple": "text-[#8b5cf6]", // Violet-500
  "accent-orange": "text-[#f59e0b]", // Amber-500
  secondary: "text-secondary", 
  success: "text-green-500",
  warning: "text-yellow-500",
  error: "text-red-500",
  info: "text-blue-500"
};

export function Spinner({ 
  size = "md", 
  variant = "primary", 
  className 
}: LoadingProps) {
  return (
    <div
      className={cn(
        "animate-spin rounded-full border-2 border-current border-t-transparent",
        sizeClasses[size],
        variantClasses[variant],
        className
      )}
      role="status"
      aria-label="Loading"
    >
      <span className="sr-only">Loading...</span>
    </div>
  );
}
