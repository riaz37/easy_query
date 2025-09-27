"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

interface UsersPaginationSkeletonProps extends LoadingProps {
  showSelectionStatus?: boolean;
  showRowsPerPage?: boolean;
  showPageControls?: boolean;
}

export function UsersPaginationSkeleton({ 
  size = "md", 
  className,
  showSelectionStatus = true,
  showRowsPerPage = true,
  showPageControls = true
}: UsersPaginationSkeletonProps) {
  const textSizes = {
    xs: "h-3",
    sm: "h-4", 
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  const cellHeight = textSizes[size];

  return (
    <div className={cn("px-6 py-4 flex items-center justify-between", className)}>
      {/* Left: Selection Status Skeleton */}
      {showSelectionStatus && (
        <div className="flex items-center">
          <div 
            className={cn("rounded animate-pulse", cellHeight, "w-48")}
            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
          />
        </div>
      )}
      
      {/* Right: Pagination Controls Skeleton */}
      <div className="flex items-center gap-6">
        {/* Rows per page selector skeleton */}
        {showRowsPerPage && (
          <div className="flex items-center gap-2">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-24")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div 
              className="rounded-full animate-pulse"
              style={{
                background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                width: "60px",
                height: "32px"
              }}
            />
          </div>
        )}
        
        {/* Page info and controls skeleton */}
        {showPageControls && (
          <div className="flex items-center gap-2">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-20")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div className="flex gap-1">
              <div 
                className="rounded-full animate-pulse"
                style={{
                  background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                  width: "32px",
                  height: "32px"
                }}
              />
              <div 
                className="rounded-full animate-pulse"
                style={{
                  background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                  width: "32px",
                  height: "32px"
                }}
              />
              <div 
                className="rounded-full animate-pulse"
                style={{
                  background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                  width: "32px",
                  height: "32px"
                }}
              />
              <div 
                className="rounded-full animate-pulse"
                style={{
                  background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
                  width: "32px",
                  height: "32px"
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
