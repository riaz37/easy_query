"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

const sizeClasses = {
  xs: "w-1 h-1",
  sm: "w-1.5 h-1.5",
  md: "w-2 h-2",
  lg: "w-3 h-3", 
  xl: "w-4 h-4"
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

export function DotsLoader({ 
  size = "md", 
  variant = "primary", 
  className 
}: LoadingProps) {
  return (
    <div className={cn("flex items-center justify-center space-x-1", className)}>
      <div
        className={cn(
          "rounded-full animate-bounce",
          sizeClasses[size],
          variantClasses[variant]
        )}
        style={{ animationDelay: "0ms", animationDuration: "1.4s" }}
      />
      <div
        className={cn(
          "rounded-full animate-bounce",
          sizeClasses[size],
          variantClasses[variant]
        )}
        style={{ animationDelay: "160ms", animationDuration: "1.4s" }}
      />
      <div
        className={cn(
          "rounded-full animate-bounce",
          sizeClasses[size],
          variantClasses[variant]
        )}
        style={{ animationDelay: "320ms", animationDuration: "1.4s" }}
      />
    </div>
  );
}
