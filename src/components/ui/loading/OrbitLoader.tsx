"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

const sizeClasses = {
  xs: "w-6 h-6",
  sm: "w-8 h-8",
  md: "w-12 h-12",
  lg: "w-16 h-16",
  xl: "w-20 h-20"
};

const variantClasses = {
  primary: "border-[#10b981]", // Emerald-500
  "primary-dark": "border-[#047857]", // Emerald-700
  "primary-light": "border-[#34d399]", // Emerald-400
  "accent-blue": "border-[#3b82f6]", // Blue-500
  "accent-purple": "border-[#8b5cf6]", // Violet-500
  "accent-orange": "border-[#f59e0b]", // Amber-500
  secondary: "border-secondary",
  success: "border-green-500",
  warning: "border-yellow-500",
  error: "border-red-500",
  info: "border-blue-500"
};

const dotSizeClasses = {
  xs: "w-1 h-1",
  sm: "w-1.5 h-1.5", 
  md: "w-2 h-2",
  lg: "w-3 h-3",
  xl: "w-4 h-4"
};

const dotVariantClasses = {
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

export function OrbitLoader({ 
  size = "md", 
  variant = "primary", 
  className 
}: LoadingProps) {
  return (
    <div className={cn("relative", sizeClasses[size], className)}>
      {/* Outer ring */}
      <div
        className={cn(
          "absolute inset-0 rounded-full border-2 border-t-transparent animate-spin",
          variantClasses[variant]
        )}
        style={{ animationDuration: "2s" }}
      />
      
      {/* Inner ring */}
      <div
        className={cn(
          "absolute inset-2 rounded-full border border-t-transparent animate-spin",
          variantClasses[variant]
        )}
        style={{ animationDuration: "1.5s", animationDirection: "reverse" }}
      />
      
      {/* Center dot */}
      <div
        className={cn(
          "absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 rounded-full animate-pulse",
          dotSizeClasses[size],
          dotVariantClasses[variant]
        )}
        style={{ animationDuration: "1s" }}
      />
    </div>
  );
}
