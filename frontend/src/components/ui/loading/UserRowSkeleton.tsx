"use client";

import { cn } from "@/lib/utils";
import { LoadingProps } from "./types";

interface UserRowSkeletonProps extends LoadingProps {
  isMSSQL?: boolean;
  showCheckbox?: boolean;
  showActions?: boolean;
}

export function UserRowSkeleton({ 
  size = "md", 
  className,
  isMSSQL = true,
  showCheckbox = true,
  showActions = true
}: UserRowSkeletonProps) {
  const textSizes = {
    xs: "h-3",
    sm: "h-4", 
    md: "h-5",
    lg: "h-6",
    xl: "h-7"
  };

  const cellHeight = textSizes[size];

  return (
    <tr className={cn("border-b border-white/10", className)}>
      {/* Checkbox Column */}
      {showCheckbox && (
        <td className="px-6 py-4">
          <div 
            className="rounded animate-pulse"
            style={{
              background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
              width: "20px",
              height: "20px"
            }}
          />
        </td>
      )}
      
      {/* User ID Column */}
      <td className="px-6 py-4">
        <div 
          className={cn("rounded animate-pulse", cellHeight, "w-48")}
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </td>
      
      {/* Conditional Columns based on tab type */}
      {isMSSQL ? (
        <>
          {/* Parent Company Column */}
          <td className="px-6 py-4">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-32")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </td>
          
          {/* Sub Companies Column */}
          <td className="px-6 py-4">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-24")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </td>
          
          {/* Databases Column */}
          <td className="px-6 py-4">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-8")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </td>
        </>
      ) : (
        <>
          {/* Database Column */}
          <td className="px-6 py-4">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-24")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </td>
          
          {/* Access Level Column */}
          <td className="px-6 py-4">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-16")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </td>
          
          {/* Tables Column */}
          <td className="px-6 py-4">
            <div 
              className={cn("rounded animate-pulse", cellHeight, "w-20")}
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </td>
        </>
      )}
      
      {/* Date Column */}
      <td className="px-6 py-4">
        <div 
          className={cn("rounded animate-pulse", cellHeight, "w-20")}
          style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
        />
      </td>
      
      {/* Actions Column */}
      {showActions && (
        <td className="px-6 py-4 text-right">
          <div 
            className="rounded-full animate-pulse"
            style={{
              background: 'var(--primary-8, rgba(19, 245, 132, 0.08))',
              width: "32px",
              height: "32px"
            }}
          />
        </td>
      )}
    </tr>
  );
}
