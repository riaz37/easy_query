"use client";

import { cn } from "@/lib/utils";
import { ProgressLoadingProps } from "./types";

const sizeClasses = {
  xs: "h-1",
  sm: "h-2",
  md: "h-3",
  lg: "h-4",
  xl: "h-6"
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

export function ProgressLoader({ 
  size = "md", 
  variant = "primary", 
  progress = 0,
  showPercentage = true,
  animated = true,
  className 
}: ProgressLoadingProps) {
  const progressValue = Math.min(Math.max(progress, 0), 100);
  
  return (
    <div className={cn("w-full", className)}>
      <div className={cn(
        "w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden",
        sizeClasses[size]
      )}>
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500 ease-out",
            variantClasses[variant],
            animated && "animate-pulse"
          )}
          style={{ 
            width: `${progressValue}%`,
            transition: animated ? "width 0.5s ease-out" : "none"
          }}
        />
      </div>
      
      {showPercentage && (
        <div className="flex justify-between items-center mt-2 text-sm text-gray-600 dark:text-gray-400">
          <span>Progress</span>
          <span className="font-medium">{Math.round(progressValue)}%</span>
        </div>
      )}
    </div>
  );
}
