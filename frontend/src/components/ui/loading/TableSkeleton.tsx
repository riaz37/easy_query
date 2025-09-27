"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

export function TableSkeleton({ 
  size = "md", 
  rows = 5,
  columns = 4,
  className 
}: LoadingProps & { rows?: number; columns?: number }) {
  const cellSizes = {
    xs: "h-3",
    sm: "h-4",
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  return (
    <div className={cn("w-full", className)}>
      {/* Table header skeleton */}
      <div className="grid gap-4 mb-4" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {Array.from({ length: columns }).map((_, index) => (
          <div
            key={`header-${index}`}
            className={cn(
              "rounded animate-pulse",
              cellSizes[size]
            )}
            style={{
              background: 'var(--primary-8, rgba(19, 245, 132, 0.08))'
            }}
          />
        ))}
      </div>
      
      {/* Table rows skeleton */}
      <div className="space-y-3">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div 
            key={`row-${rowIndex}`}
            className="grid gap-4" 
            style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
          >
            {Array.from({ length: columns }).map((_, colIndex) => (
              <div
                key={`cell-${rowIndex}-${colIndex}`}
                className={cn(
                  "rounded animate-pulse",
                  cellSizes[size]
                )}
                style={{ 
                  width: colIndex === 0 ? "80%" : colIndex === columns - 1 ? "60%" : "100%",
                  background: 'var(--primary-8, rgba(19, 245, 132, 0.08))'
                }}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
