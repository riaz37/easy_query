"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

interface PaginationSkeletonProps extends LoadingProps {
  showInfo?: boolean;
  showPageNumbers?: boolean;
}

export function PaginationSkeleton({ 
  size = "md", 
  className,
  showInfo = true,
  showPageNumbers = true
}: PaginationSkeletonProps) {
  const textSizes = {
    xs: "h-3",
    sm: "h-4", 
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  const textHeight = textSizes[size];

  return (
    <div className={cn("flex items-center justify-between", className)}>
      {/* Pagination Info Skeleton */}
      {showInfo && (
        <div className="space-y-1">
          <div 
            className={cn("rounded animate-pulse", textHeight, "w-48")}
            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
          />
        </div>
      )}
      
      {/* Pagination Controls Skeleton */}
      <div className="flex items-center gap-2">
        {/* Previous Button Skeleton */}
        <div 
          className="rounded-full animate-pulse h-8 w-20"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
        
        {/* Page Numbers Skeleton */}
        {showPageNumbers && (
          <div className="flex items-center gap-2">
            <div 
              className="rounded animate-pulse h-6 w-16"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className="rounded-full animate-pulse h-8 w-8"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className="rounded-full animate-pulse h-8 w-8"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className="rounded-full animate-pulse h-8 w-8"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
        )}
        
        {/* Next Button Skeleton */}
        <div 
          className="rounded-full animate-pulse h-8 w-16"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </div>
    </div>
  );
}
