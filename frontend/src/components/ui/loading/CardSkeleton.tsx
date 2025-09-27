"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

export function CardSkeleton({ 
  size = "md", 
  className 
}: LoadingProps) {
  const cardSizes = {
    xs: "p-3",
    sm: "p-4", 
    md: "p-6",
    lg: "p-8",
    xl: "p-10"
  };

  return (
    <div className={cn(
      "bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm",
      cardSizes[size],
      className
    )}>
      {/* Header skeleton */}
      <div className="flex items-center space-x-3 mb-4">
        <div 
          className="w-8 h-8 rounded-full animate-pulse"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
        <div className="flex-1 space-y-2">
          <div 
            className="h-4 rounded animate-pulse w-3/4"
            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
          />
          <div 
            className="h-3 rounded animate-pulse w-1/2"
            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
          />
        </div>
      </div>
      
      {/* Content skeleton */}
      <div className="space-y-3">
        <div 
          className="h-4 rounded animate-pulse"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
        <div 
          className="h-4 rounded animate-pulse w-5/6"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
        <div 
          className="h-4 rounded animate-pulse w-4/6"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </div>
      
      {/* Footer skeleton */}
      <div className="flex justify-between items-center mt-6">
        <div 
          className="h-3 rounded animate-pulse w-1/4"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
        <div 
          className="h-8 rounded animate-pulse w-20"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </div>
    </div>
  );
}
