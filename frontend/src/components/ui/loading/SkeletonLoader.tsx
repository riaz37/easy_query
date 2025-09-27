"use client";

import { cn } from "@/lib/utils";
import { SkeletonLoadingProps } from "./types";

const sizeClasses = {
  xs: "h-2",
  sm: "h-3",
  md: "h-4",
  lg: "h-6",
  xl: "h-8"
};

export function SkeletonLoader({ 
  size = "md", 
  lines = 1,
  width = "100%",
  height,
  rounded = true,
  className 
}: SkeletonLoadingProps) {
  const skeletonHeight = height || sizeClasses[size];
  
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: lines }).map((_, index) => (
        <div
          key={index}
          className={cn(
            "animate-pulse",
            rounded && "rounded-md",
            skeletonHeight
          )}
          style={{ 
            width: typeof width === 'number' ? `${width}px` : width,
            height: typeof height === 'number' ? `${height}px` : undefined,
            background: 'var(--primary-8, rgba(19, 245, 132, 0.08))'
          }}
        />
      ))}
    </div>
  );
}
