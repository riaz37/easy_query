import React from 'react';
import { cn } from '@/lib/utils';

interface DatabaseCardSkeletonProps {
  className?: string;
  isSelected?: boolean;
}

export const DatabaseCardSkeleton: React.FC<DatabaseCardSkeletonProps> = ({
  className,
  isSelected = false
}) => {
  return (
    <div
      className={cn(
        "cursor-pointer transition-all rounded-[32px] p-4 query-content-gradient h-[184px] border-2",
        isSelected ? "border-emerald-500" : "border-transparent",
        className
      )}
    >
      <div className="space-y-3">
        {/* Header with title and selection indicator */}
        <div className="flex items-center justify-between">
          {/* Database name skeleton */}
          <div
            className="h-6 rounded animate-pulse w-32"
            style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
          />
          {/* Selection indicator skeleton */}
          <div
            className={cn(
              "aspect-square h-4 w-4 rounded-full border flex items-center justify-center",
              isSelected ? "bg-green-500 border-green-500" : "border-gray-300 bg-transparent"
            )}
          >
            {isSelected && (
              <div
                className="h-2.5 w-2.5 rounded-full animate-pulse"
                style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
              />
            )}
          </div>
        </div>

        {/* Database details skeleton */}
        <div className="text-sm space-y-1">
          {/* Type field */}
          <div className="flex items-center gap-2">
            <div
              className="h-4 rounded animate-pulse w-8"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div
              className="h-4 rounded animate-pulse w-16"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
          
          {/* URL field */}
          <div className="flex items-center gap-2">
            <div
              className="h-4 rounded animate-pulse w-8"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div
              className="h-4 rounded animate-pulse w-24"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
          
          {/* Rules field */}
          <div className="flex items-center gap-2">
            <div
              className="h-4 rounded animate-pulse w-12"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            <div
              className="h-4 rounded animate-pulse w-16"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
