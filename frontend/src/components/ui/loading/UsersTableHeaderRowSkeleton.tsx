"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

interface UsersTableHeaderRowSkeletonProps extends LoadingProps {
  isMSSQL?: boolean;
}

export function UsersTableHeaderRowSkeleton({ 
  size = "md", 
  className,
  isMSSQL = true
}: UsersTableHeaderRowSkeletonProps) {
  const textSizes = {
    xs: "h-3",
    sm: "h-4", 
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  const cellHeight = textSizes[size];

  return (
    <tr 
      className={className}
      style={{
        background: "var(--components-Table-Head-filled, rgba(145, 158, 171, 0.08))",
        borderRadius: "12px 12px 0 0"
      }}
    >
      {/* Checkbox Header */}
      <th className="px-6 py-4 text-left rounded-tl-xl">
        <div 
          className="rounded animate-pulse"
          style={{
            background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
            width: "20px",
            height: "20px"
          }}
        />
      </th>
      
      {/* User ID Header */}
      <th className="px-6 py-4 text-left">
        <div 
          className={cn("rounded animate-pulse", cellHeight, "w-24")}
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </th>
      
      {/* Conditional Headers based on tab type */}
      {isMSSQL ? (
        <>
          {/* Parent Company Header */}
          <th className="px-6 py-4 text-left">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-32")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </th>
          
          {/* Sub Companies Header */}
          <th className="px-6 py-4 text-left">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-28")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </th>
          
          {/* Databases Header */}
          <th className="px-6 py-4 text-left">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-20")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </th>
        </>
      ) : (
        <>
          {/* Database Header */}
          <th className="px-6 py-4 text-left">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-24")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </th>
          
          {/* Access Level Header */}
          <th className="px-6 py-4 text-left">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-28")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </th>
          
          {/* Tables Header */}
          <th className="px-6 py-4 text-left">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-20")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </th>
        </>
      )}
      
      {/* Date Header */}
      <th className="px-6 py-4 text-left">
        <div 
          className={cn("rounded animate-pulse", cellHeight, "w-16")}
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </th>
      
      {/* Actions Header */}
      <th className="px-6 py-4 text-right text-white font-medium text-sm rounded-tr-xl">
        <div 
          className={cn("rounded animate-pulse", cellHeight, "w-16")}
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </th>
    </tr>
  );
}
