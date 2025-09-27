import React from 'react';
import { cn } from '@/lib/utils';
import { DatabaseCardSkeleton } from './DatabaseCardSkeleton';

interface DatabaseSelectionSkeletonProps {
  className?: string;
  cardCount?: number;
  showHeader?: boolean;
  showFooter?: boolean;
  databaseCount?: number;
}

export const DatabaseSelectionSkeleton: React.FC<DatabaseSelectionSkeletonProps> = ({
  className,
  cardCount = 6,
  showHeader = true,
  showFooter = true,
  databaseCount
}) => {
  return (
    <div className={cn("query-content-gradient rounded-[32px] p-6", className)}>
      <div className="space-y-4">
        {showHeader && (
          <div className="mb-6">
            {/* Title skeleton */}
            <div
              className="h-7 rounded animate-pulse w-48 mb-2"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
            {/* Description skeleton */}
            <div
              className="h-4 rounded animate-pulse w-64"
              style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
            />
          </div>
        )}

        <div className="space-y-4">
          {/* Database cards grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: cardCount }).map((_, index) => (
              <DatabaseCardSkeleton
                key={`database-card-skeleton-${index}`}
                isSelected={index === 0} // First card appears selected
              />
            ))}
          </div>

          {showFooter && (
            <div className="flex justify-end items-center pt-4">
              {/* Database count skeleton - show actual count if available */}
              {databaseCount !== undefined ? (
                <div className="text-sm text-gray-400">
                  {databaseCount} database{databaseCount !== 1 ? 's' : ''} available
                </div>
              ) : (
                <div
                  className="h-4 rounded animate-pulse w-32"
                  style={{ background: 'var(--primary-8, rgba(19, 245, 132, 0.08))' }}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
