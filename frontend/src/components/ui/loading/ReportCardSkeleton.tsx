"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

interface ReportCardSkeletonProps extends LoadingProps {
  showActions?: boolean;
  showMetadata?: boolean;
}

export function ReportCardSkeleton({ 
  size = "md", 
  className,
  showActions = true,
  showMetadata = true
}: ReportCardSkeletonProps) {
  const cardSizes = {
    xs: "p-3",
    sm: "p-4", 
    md: "p-6",
    lg: "p-8",
    xl: "p-10"
  };

  const textSizes = {
    xs: "h-3",
    sm: "h-4", 
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  const cardPadding = cardSizes[size];
  const textHeight = textSizes[size];

  return (
    <div className={cn(
      "bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm",
      cardPadding,
      className
    )}>
      {/* Header Section */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 space-y-2">
          {/* Report Title */}
          <div 
            className={cn("rounded animate-pulse", textHeight, "w-3/4")}
            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
          />
          <div 
            className={cn("rounded animate-pulse", textHeight, "w-1/2")}
            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
          />
        </div>
        
        {/* Status Badge Skeleton */}
        <div 
          className="rounded-full animate-pulse h-6 w-16"
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </div>

      {/* Content Section */}
      <div className="space-y-3 mb-4">
        {/* Description lines */}
        <div 
          className={cn("rounded animate-pulse", textHeight)}
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
        <div 
          className={cn("rounded animate-pulse", textHeight, "w-5/6")}
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
        <div 
          className={cn("rounded animate-pulse", textHeight, "w-4/6")}
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </div>

      {/* Metadata Section */}
      {showMetadata && (
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="space-y-2">
            <div 
              className={cn("rounded animate-pulse", textHeight, "w-16")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className={cn("rounded animate-pulse", textHeight, "w-20")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
          <div className="space-y-2">
            <div 
              className={cn("rounded animate-pulse", textHeight, "w-20")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className={cn("rounded animate-pulse", textHeight, "w-12")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
        </div>
      )}

      {/* Footer Section */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          {/* Date Skeleton */}
          <div className="flex items-center space-x-2">
            <div 
              className="rounded animate-pulse h-4 w-4"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className={cn("rounded animate-pulse", textHeight, "w-24")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
          
          {/* Queries Count Skeleton */}
          <div className="flex items-center space-x-2">
            <div 
              className="rounded animate-pulse h-4 w-4"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className={cn("rounded animate-pulse", textHeight, "w-8")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
        </div>

        {/* Actions Skeleton */}
        {showActions && (
          <div className="flex items-center gap-2">
            <div 
              className="rounded-full animate-pulse h-8 w-20"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className="rounded-full animate-pulse h-8 w-24"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
