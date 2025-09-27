"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

interface UsersTableHeaderSkeletonProps extends LoadingProps {
  isMSSQL?: boolean;
  showTabs?: boolean;
  showSearch?: boolean;
  showActions?: boolean;
}

export function UsersTableHeaderSkeleton({ 
  size = "md", 
  className,
  isMSSQL = true,
  showTabs = true,
  showSearch = true,
  showActions = true
}: UsersTableHeaderSkeletonProps) {
  const textSizes = {
    xs: "h-3",
    sm: "h-4", 
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  const cellHeight = textSizes[size];

  return (
    <div className={cn("p-6", className)}>
      {/* Tabs Skeleton */}
      {showTabs && (
        <div className="flex gap-8 mb-6">
          <div 
            className="text-sm font-medium pb-2 border-b-2 rounded animate-pulse"
            style={{ 
              background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
              width: "200px",
              height: "24px"
            }}
          />
          <div 
            className="text-sm font-medium pb-2 border-b-2 rounded animate-pulse"
            style={{ 
              background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
              width: "200px",
              height: "24px"
            }}
          />
        </div>
      )}

      {/* Action Bar Skeleton */}
      <div className="flex items-center justify-between">
        {/* Search Input Skeleton */}
        {showSearch && (
          <div className="flex items-center gap-4 flex-1">
            <div 
              className="relative flex-1 max-w-md rounded-full animate-pulse"
              style={{
                background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                height: "50px"
              }}
            />
          </div>
        )}

        {/* Action Buttons Skeleton */}
        {showActions && (
          <div className="flex items-center gap-3">
            <div 
              className="rounded-full animate-pulse"
              style={{
                background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                width: "48px",
                height: "48px"
              }}
            />
            <div 
              className="rounded-full animate-pulse"
              style={{
                background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                width: "48px",
                height: "48px"
              }}
            />
            <div 
              className="rounded-full animate-pulse"
              style={{
                background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                width: "48px",
                height: "48px"
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
